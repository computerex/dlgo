//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/mmap"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

// GpuPipeline bundles a model on GPU with all state needed for inference.
type GpuPipeline struct {
	CPUModel        *llm.Model
	GpuModel        *GpuModel
	Tokenizer       *llm.Tokenizer
	KVCache         *GpuKVCache
	RunState        *GpuRunState
	MaxSeqLen       int
	LogitsBuf       []float32
	LayerConfs      []*LayerConf
	Q8_1Scratch     Buf
	BatchState      *GpuBatchState
	BatchLayerConfs []*LayerConf
	UseFusedForward bool

	HasSSM    bool
	HasGatedQ bool
	HasMoE    bool
	HasMLA    bool

	RoPECosTable Buf
	RoPESinTable Buf

	// Partial GPU offloading: layers [0, NumGPULayers) are on GPU,
	// layers [NumGPULayers, NumLayers) run on CPU (RAM or mmap).
	NumGPULayers int
	IsPartialGPU bool // true when some layers are on CPU

	// CPU-side state for hybrid MoE or partial GPU offloading
	CPURunState  *llm.RunState
	CPUKVCache   *memory.MultiLayerKVCache // KV cache for CPU layers
	CPUBatchState *llm.BatchState           // batch state for CPU prefill

	AllCPUAttn bool // true if ALL GPU layers use CPU attention fallback
}

// MaxVRAMBytes caps the VRAM the GPU pipeline may use. 0 = no cap (use all free).
// Set via DLGO_MAX_VRAM_MB environment variable or --max-vram flag.
var MaxVRAMBytes int64
var maxVRAMLoaded bool

// loadMaxVRAM reads the DLGO_MAX_VRAM_MB env var once.
func loadMaxVRAM() {
	if maxVRAMLoaded {
		return
	}
	maxVRAMLoaded = true
	if s := os.Getenv("DLGO_MAX_VRAM_MB"); s != "" {
		var mb int
		if _, err := fmt.Sscanf(s, "%d", &mb); err == nil && mb > 0 {
			MaxVRAMBytes = int64(mb) * 1024 * 1024
			fmt.Printf("[dlgo/gpu] VRAM cap set to %d MB via DLGO_MAX_VRAM_MB\n", mb)
		}
	}
}

// effectiveFreeVRAM returns the free VRAM, capped by MaxVRAMBytes if set.
func effectiveFreeVRAM() int64 {
	loadMaxVRAM()
	free := int64(VRAMFreeBytes())
	if MaxVRAMBytes > 0 && free > MaxVRAMBytes {
		free = MaxVRAMBytes
	}
	return free
}

// estimateFixedVRAM estimates GPU memory for non-per-layer allocations
// (run state, batch state, SSM scratch, RoPE tables, q8_1 scratch).
// Does NOT include KV cache or per-layer weights — those are computed
// per-layer in the budget solver.
func estimateFixedVRAM(cfg llm.ModelConfig, maxSeqLen int) int64 {
	dim := int64(cfg.EmbeddingDim)
	qDim := int64(cfg.NumHeads * cfg.HeadDim)
	kvDim := int64(cfg.NumKVHeads * cfg.HeadDim)
	ffnDim := int64(cfg.FFNDim)
	vocab := int64(cfg.VocabSize)

	var total int64

	// Run state buffers
	total += (dim + dim + qDim + kvDim + kvDim + qDim + dim + dim + dim + ffnDim + ffnDim + ffnDim + dim + vocab) * 4

	// SSM scratch (shared, not per-layer)
	if cfg.SSMInnerSize > 0 {
		numHeads := int64(cfg.SSMTimeStepRank)
		headVDim := int64(cfg.SSMInnerSize) / numHeads
		numKVGroups := int64(cfg.SSMGroupCount)
		if numKVGroups <= 0 {
			numKVGroups = numHeads
		}
		headKDim := int64(cfg.SSMStateSize)
		keyDim := numKVGroups * headKDim
		qkvDim := keyDim*2 + numHeads*headVDim
		total += (qkvDim + numHeads*headVDim + numHeads + numHeads + numHeads*headVDim) * 4
	}

	// MoE scratch buffers (shared, not per-layer)
	if cfg.ExpertCount > 0 {
		expDim := int64(cfg.ExpertFFNDim)
		shDim := int64(cfg.SharedExpertFFNDim)
		if shDim == 0 {
			shDim = expDim
		}
		total += (int64(cfg.ExpertCount) + 3*expDim + 2*dim + 3*shDim) * 4
	}

	// Batch state: estimate for prefillChunkSize tokens (512) to match
	// actual runtime allocation in PrefillAndDecode. Using 128 was a
	// severe underestimate that caused the budget solver to accept
	// contexts that didn't actually fit.
	batchTokens := int64(512)
	total += batchTokens * (dim + dim + qDim + kvDim + kvDim + qDim + dim + dim + dim + ffnDim + ffnDim + ffnDim + dim) * 4

	// GatedQ batch buffers (allocated at runtime in AllocGatedQBatch)
	if cfg.FullAttentionInterval > 0 {
		total += batchTokens * (2*qDim + qDim) * 4
	}

	// SSM batch buffers (allocated at runtime in AllocSSMBatch)
	if cfg.SSMInnerSize > 0 {
		numHeads := int64(cfg.SSMTimeStepRank)
		headVDim := int64(cfg.SSMInnerSize) / numHeads
		numKVGroups := int64(cfg.SSMGroupCount)
		if numKVGroups <= 0 {
			numKVGroups = numHeads
		}
		headKDim := int64(cfg.SSMStateSize)
		keyDim := numKVGroups * headKDim
		qkvDim := keyDim*2 + numHeads*headVDim
		valueDim := numHeads * headVDim
		total += batchTokens * (qkvDim + valueDim + numHeads + numHeads + valueDim) * 4
	}

	// RoPE cos/sin tables: 2 * maxSeqLen * (ropeDim/2) * 4 bytes each
	ropeDim := int64(cfg.RopeDim)
	if ropeDim <= 0 || ropeDim > int64(cfg.HeadDim) {
		ropeDim = int64(cfg.HeadDim)
	}
	total += 2 * int64(maxSeqLen) * (ropeDim / 2) * 4

	// q8_1 scratch buffer for dp4a path
	maxDim := dim
	if ffnDim > maxDim {
		maxDim = ffnDim
	}
	q8_1Blocks := (maxDim + 31) / 32
	total += q8_1Blocks * 36

	// Safety margin: reserve VRAM for Windows display compositor, video decode,
	// browser GPU acceleration, and Vulkan driver internals. The C-level budget
	// check in create_buffer() enforces a hard 512 MB floor before each
	// vkAllocateMemory call, so the Go-level margin is a softer estimate.
	// Use max(1 GB, 6% of total VRAM) as the budget-solver reserve.
	vram := int64(VRAMBytes())
	margin := int64(1024) * 1024 * 1024 // 1 GB minimum
	if pct := vram * 6 / 100; pct > margin {
		margin = pct
	}
	total += margin

	return total
}

// layerNeedsKV returns true if a layer uses GPU KV cache (attention or GatedQ).
// SSM and MLA layers have their own state and don't use the KV cache buffers.
func layerNeedsKV(layer *llm.Layer) bool {
	return layer.Spec.Core == llm.CoreAttention || layer.Spec.GatedQ
}

// computeMaxGPUContext finds the largest context length (up to maxSeqLen) that
// allows ALL model layers to fit in VRAM. Binary-searches between minCtx and
// maxSeqLen. Returns maxSeqLen if it already fits, or a reduced value.
//
// If no context >= minCtx fits all layers, returns minCtx and the caller
// must use partial GPU offloading. This is safer than reducing context to
// extremely small values (e.g. 2048) just to force all layers onto GPU —
// that leaves no VRAM safety margin and crashes Windows.
func computeMaxGPUContext(m *llm.Model, maxSeqLen int) int {
	// Start by checking if the full context fits all layers.
	if computeGPULayerBudget(m, maxSeqLen) >= len(m.Layers) {
		return maxSeqLen
	}

	// Binary search for the largest context where all layers fit.
	// Don't go below 8K — it's better to have some layers on CPU at a
	// useful context than all layers on GPU at a tiny context.
	const minCtx = 8192
	lo, hi := minCtx, maxSeqLen
	best := 0 // 0 = can't fit all layers even at minCtx
	for lo <= hi {
		mid := lo + (hi-lo)/2
		if computeGPULayerBudget(m, mid) >= len(m.Layers) {
			best = mid
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}

	if best == 0 {
		// Can't fit all layers even at minimum context.
		// Return minCtx — partial offloading will handle the rest.
		// This gives the most GPU layers at a usable context.
		return minCtx
	}

	// Round down to a nice power-of-two-ish boundary for cleaner allocation.
	for _, nice := range []int{131072, 65536, 32768, 16384, 8192} {
		if nice <= best {
			return nice
		}
	}
	return best
}

// computeGPULayerBudget determines how many layers fit in available VRAM,
// accounting for both weight data AND per-layer KV cache VRAM.
//
// IMPORTANT: The estimates here are intentionally 30% over what we think is
// needed. Vulkan buffer allocations have alignment overhead, descriptor sets,
// staging buffers, and batch state that are not explicitly tracked. On Windows
// the GPU driver can freeze the entire system if VRAM is exhausted — there is
// no graceful OOM. Being conservative here is critical for system stability.
func computeGPULayerBudget(m *llm.Model, maxSeqLen int) int {
	freeVRAM := effectiveFreeVRAM()
	if freeVRAM <= 0 {
		return 0
	}

	fixedOverhead := estimateFixedVRAM(m.Config, maxSeqLen)

	// Non-layer weight VRAM (embed + output)
	var nonLayerBytes int64
	if m.TokenEmbed != nil {
		nonLayerBytes += int64(len(m.TokenEmbed.Data))
	}
	if m.Output != nil && m.Output != m.TokenEmbed {
		nonLayerBytes += int64(len(m.Output.Data))
	}
	nonLayerBytes += int64(m.Config.EmbeddingDim * 4) // output norm

	// Apply 15% safety multiplier for Vulkan allocation alignment overhead,
	// descriptor sets, and other untracked small allocations.
	fixedOverhead = fixedOverhead * 115 / 100
	nonLayerBytes = nonLayerBytes * 115 / 100

	available := freeVRAM - fixedOverhead - nonLayerBytes
	if available <= 0 {
		return 0
	}

	// Per-layer KV cache cost
	kvDim := int64(m.Config.NumKVHeads * m.Config.HeadDim)
	kvPerLayer := 2 * int64(maxSeqLen) * kvDim * 4 // K + V buffers (FP32)

	// Per-layer SSM state cost (only for SSM layers)
	var ssmPerLayer int64
	if m.Config.SSMInnerSize > 0 {
		numHeads := int64(m.Config.SSMTimeStepRank)
		headKDim := int64(m.Config.SSMStateSize)
		headVDim := int64(m.Config.SSMInnerSize) / numHeads
		numKVGroups := int64(m.Config.SSMGroupCount)
		if numKVGroups <= 0 {
			numKVGroups = numHeads
		}
		keyDim := numKVGroups * headKDim
		qkvDim := keyDim*2 + numHeads*headVDim
		convK := int64(m.Config.SSMConvKernel)
		ssmPerLayer = numHeads*headKDim*headVDim*4 + convK*qkvDim*4
	}

	// Greedily add layers: each layer costs weights + KV cache + optional SSM.
	// 25% per-layer multiplier accounts for Vulkan buffer alignment, descriptor
	// overhead, and staging buffers. This is higher than fixed overhead (15%)
	// because per-layer costs include KV cache which dominates at large contexts
	// and whose actual VRAM exceeds the formula due to alignment.
	numLayers := 0
	for l := 0; l < len(m.Layers); l++ {
		layerCost := llm.EstimateLayerBytes(&m.Layers[l])
		if layerNeedsKV(&m.Layers[l]) {
			layerCost += kvPerLayer
		}
		if m.Layers[l].Spec.Core == llm.CoreSSM {
			layerCost += ssmPerLayer
		}
		layerCost = layerCost * 125 / 100 // 25% safety margin
		if layerCost > available {
			break
		}
		available -= layerCost
		numLayers++
	}
	return numLayers
}

// UploadModel copies all model weights to GPU memory. If numGPULayers is -1,
// all layers are uploaded. Otherwise, only the first numGPULayers layers are
// uploaded. Layers beyond that limit have OnGPU=false and empty GPU tensors.
func UploadModel(m *llm.Model, numGPULayers ...int) (*GpuModel, error) {
	maxLayers := len(m.Layers)
	if len(numGPULayers) > 0 && numGPULayers[0] >= 0 {
		maxLayers = numGPULayers[0]
		if maxLayers > len(m.Layers) {
			maxLayers = len(m.Layers)
		}
	}
	gm := &GpuModel{
		Layers: make([]GpuLayer, len(m.Layers)),
	}

	// VRAM floor: stop uploading layers if free VRAM drops below this.
	// On Windows the GPU driver freezes the entire system on exhaustion.
	const uploadFloor int64 = 4 * 1024 * 1024 * 1024 // 4 GB

	var err error
	gm.TokenEmbed, err = UploadTensor(m.TokenEmbed)
	if err != nil {
		return nil, fmt.Errorf("upload token_embed: %w", err)
	}

	if m.OutputNorm != nil {
		gm.OutputNorm, err = UploadF32Slice(m.OutputNorm)
		if err != nil {
			return nil, fmt.Errorf("upload output_norm: %w", err)
		}
	}
	if m.OutputNormBias != nil {
		gm.OutputNormBias, err = UploadF32Slice(m.OutputNormBias)
		if err != nil {
			return nil, fmt.Errorf("upload output_norm_bias: %w", err)
		}
	}

	gm.Output, err = UploadTensor(m.Output)
	if err != nil {
		return nil, fmt.Errorf("upload output: %w", err)
	}

	if m.OutputBias != nil {
		gm.OutputBias, err = UploadF32Slice(m.OutputBias)
		if err != nil {
			return nil, fmt.Errorf("upload output_bias: %w", err)
		}
	}

	for l := 0; l < len(m.Layers); l++ {
		cl := &m.Layers[l]
		gl := &gm.Layers[l]

		if l >= maxLayers {
			gl.OnGPU = false
			continue
		}
		gl.OnGPU = true

		if cl.AttnNorm != nil {
			gl.AttnNorm, err = UploadF32Slice(cl.AttnNorm)
			if err != nil {
				return nil, fmt.Errorf("layer %d attn_norm: %w", l, err)
			}
		}
		if cl.AttnNormBias != nil {
			gl.AttnNormBias, err = UploadF32Slice(cl.AttnNormBias)
			if err != nil {
				return nil, fmt.Errorf("layer %d attn_norm_bias: %w", l, err)
			}
		}

		if cl.Wq != nil {
			gl.Wq, err = UploadTensor(cl.Wq)
			if err != nil {
				return nil, fmt.Errorf("layer %d wq: %w", l, err)
			}
		}
		if cl.Wk != nil {
			gl.Wk, err = UploadTensor(cl.Wk)
			if err != nil {
				return nil, fmt.Errorf("layer %d wk: %w", l, err)
			}
		}
		if cl.Wv != nil {
			gl.Wv, err = UploadTensor(cl.Wv)
			if err != nil {
				return nil, fmt.Errorf("layer %d wv: %w", l, err)
			}
		}
		if cl.Wo != nil {
			gl.Wo, err = UploadTensor(cl.Wo)
			if err != nil {
				return nil, fmt.Errorf("layer %d wo: %w", l, err)
			}
		}

		if cl.AttnSinks != nil {
			gl.AttnSinks, _ = UploadF32Slice(cl.AttnSinks)
		}

		if cl.Bq != nil {
			gl.Bq, _ = UploadF32Slice(cl.Bq)
		}
		if cl.Bk != nil {
			gl.Bk, _ = UploadF32Slice(cl.Bk)
		}
		if cl.Bv != nil {
			gl.Bv, _ = UploadF32Slice(cl.Bv)
		}
		if cl.Bo != nil {
			gl.Bo, _ = UploadF32Slice(cl.Bo)
		}
		if cl.AttnQNorm != nil {
			gl.AttnQNorm, _ = UploadF32Slice(cl.AttnQNorm)
		}
		if cl.AttnKNorm != nil {
			gl.AttnKNorm, _ = UploadF32Slice(cl.AttnKNorm)
		}
		if cl.PostAttnNorm != nil {
			gl.PostAttnNorm, _ = UploadF32Slice(cl.PostAttnNorm)
		}
		if cl.FFNNorm != nil {
			gl.FFNNorm, _ = UploadF32Slice(cl.FFNNorm)
		}

		if cl.Spec.FFN == llm.FFNMoE || cl.Spec.FFN == llm.FFNMoESwiOAI {
			gl.IsMoE = true
			// Try to upload packed expert weights to GPU
			moeUploaded := true
			if cl.FFNRouter != nil {
				gl.FFNRouter, err = UploadTensor(cl.FFNRouter)
				if err != nil {
					moeUploaded = false
				}
			}
			if cl.FFNRouterBias != nil {
				gl.FFNRouterBias, _ = UploadF32Slice(cl.FFNRouterBias)
			}
			if moeUploaded && cl.FFNGateUpExps != nil && supportsGPUQType(cl.FFNGateUpExps.Type) {
				gl.FFNGateUpExps, err = UploadTensor(cl.FFNGateUpExps)
				if err != nil {
					moeUploaded = false
				}
			} else if moeUploaded && cl.FFNGateExps != nil && supportsGPUQType(cl.FFNGateExps.Type) {
				gl.FFNGateExps, err = UploadTensor(cl.FFNGateExps)
				if err != nil {
					moeUploaded = false
				}
				if moeUploaded && cl.FFNUpExps != nil {
					gl.FFNUpExps, err = UploadTensor(cl.FFNUpExps)
					if err != nil {
						moeUploaded = false
					}
				}
			} else {
				moeUploaded = false
			}
			if moeUploaded && cl.FFNDownExps != nil {
				gl.FFNDownExps, err = UploadTensor(cl.FFNDownExps)
				if err != nil {
					moeUploaded = false
				}
			}
			gl.MoEOnGPU = moeUploaded
			if moeUploaded && cl.FFNGateExpsBias != nil {
				gl.FFNGateExpsBias, _ = UploadF32Slice(cl.FFNGateExpsBias)
			}
			if moeUploaded && cl.FFNUpExpsBias != nil {
				gl.FFNUpExpsBias, _ = UploadF32Slice(cl.FFNUpExpsBias)
			}
			if moeUploaded && cl.FFNDownExpsBias != nil {
				gl.FFNDownExpsBias, _ = UploadF32Slice(cl.FFNDownExpsBias)
			}
			if cl.FFNGateShared != nil {
				gl.FFNGateShared, _ = UploadTensor(cl.FFNGateShared)
			}
			if cl.FFNUpShared != nil {
				gl.FFNUpShared, _ = UploadTensor(cl.FFNUpShared)
			}
			if cl.FFNDownShared != nil {
				gl.FFNDownShared, _ = UploadTensor(cl.FFNDownShared)
			}
			if cl.FFNRouterShared != nil {
				gl.FFNRouterShared, _ = UploadF32Slice(cl.FFNRouterShared)
			}
		} else {
			gl.FFNGate, _ = UploadTensor(cl.FFNGate)
			if cl.FFNUp != nil {
				gl.FFNUp, err = UploadTensor(cl.FFNUp)
				if err != nil {
					return nil, fmt.Errorf("layer %d ffn_up: %w", l, err)
				}
			}
			if cl.FFNDown != nil {
				gl.FFNDown, err = UploadTensor(cl.FFNDown)
				if err != nil {
					return nil, fmt.Errorf("layer %d ffn_down: %w", l, err)
				}
			}
		}

		if cl.FFNUpBias != nil {
			gl.FFNUpBias, _ = UploadF32Slice(cl.FFNUpBias)
		}
		if cl.FFNDownBias != nil {
			gl.FFNDownBias, _ = UploadF32Slice(cl.FFNDownBias)
		}
		if cl.PostFFNNorm != nil {
			gl.PostFFNNorm, _ = UploadF32Slice(cl.PostFFNNorm)
		}

		// SSM (Gated Delta Net) weights
		if cl.SSMInProj != nil {
			gl.SSMInProj, _ = UploadTensor(cl.SSMInProj)
		}
		if cl.AttnGate != nil {
			gl.SSMGate, _ = UploadTensor(cl.AttnGate)
		}
		if cl.SSMAlpha != nil {
			gl.SSMAlpha, _ = UploadTensor(cl.SSMAlpha)
		}
		if cl.SSMBeta != nil {
			gl.SSMBeta, _ = UploadTensor(cl.SSMBeta)
		}
		if cl.SSMConv1dW != nil {
			gl.SSMConv1dW, _ = UploadF32Slice(cl.SSMConv1dW)
		}
		if cl.SSMA != nil {
			gl.SSMA, _ = UploadF32Slice(cl.SSMA)
		}
		if cl.SSMDtBias != nil {
			gl.SSMDtBias, _ = UploadF32Slice(cl.SSMDtBias)
		}
		if cl.SSMNorm != nil {
			gl.SSMNorm, _ = UploadF32Slice(cl.SSMNorm)
		}
		if cl.SSMOut != nil {
			gl.SSMOut, _ = UploadTensor(cl.SSMOut)
		}

		// Per-layer VRAM floor check: sync the GPU and verify we haven't
		// consumed too much VRAM. If free memory drops below the floor,
		// free this layer's allocations and stop — remaining layers stay
		// on CPU. This prevents the Windows GPU driver from freezing the
		// entire system when VRAM is exhausted.
		Sync()
		if postFree := int64(VRAMFreeBytes()); postFree < uploadFloor {
			fmt.Printf("[dlgo/gpu] VRAM floor hit after layer %d (%.0f MB free < %.0f MB floor), stopping upload\n",
				l, float64(postFree)/(1024*1024), float64(uploadFloor)/(1024*1024))
			// Free this layer and mark it + all remaining as CPU-only.
			*gl = GpuLayer{} // zero out — freeTensor/freeBuf on zero is no-op
			gl.OnGPU = false
			for j := l + 1; j < len(m.Layers); j++ {
				gm.Layers[j].OnGPU = false
			}
			gm.NumGPULayers = l
			return gm, nil
		}
	}

	gm.NumGPULayers = maxLayers
	return gm, nil
}

// NewGpuPipeline creates a GPU-accelerated inference pipeline.
// Automatically determines how many layers fit in VRAM and places the rest on CPU.
// This enables running models of ANY size — throughput degrades gracefully but
// the system never fails due to insufficient VRAM.
func NewGpuPipeline(cpuPipeline *llm.Pipeline) (*GpuPipeline, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	if err := InitIQTables(); err != nil {
		return nil, fmt.Errorf("gpu: IQ table upload failed: %w", err)
	}

	m := cpuPipeline.Model
	cfg := m.Config

	totalVRAM := float64(VRAMBytes()) / (1024 * 1024)
	freeVRAM := float64(effectiveFreeVRAM()) / (1024 * 1024)
	fmt.Printf("[dlgo/gpu] Uploading model to %s (%.0f MB total, %.0f MB usable)...\n",
		DeviceName(), totalVRAM, freeVRAM)

	// GPU-aware context capping: if the native context is too large to fit
	// all layers in VRAM, reduce it to the largest value that does fit.
	// This prevents OOM crashes with models that have very large native
	// contexts (e.g., 256K for Qwen3.5) on GPUs with limited VRAM.
	maxSeqLen := cpuPipeline.MaxSeqLen
	gpuSafeCtx := computeMaxGPUContext(m, maxSeqLen)
	if gpuSafeCtx < maxSeqLen {
		fmt.Printf("[dlgo/gpu] Reducing context from %d to %d tokens to fit all layers in VRAM (%.0f MB)\n",
			maxSeqLen, gpuSafeCtx, freeVRAM)
		maxSeqLen = gpuSafeCtx
		cpuPipeline.MaxSeqLen = maxSeqLen
	}

	// Determine how many layers fit in VRAM.
	// DLGO_GPU_LAYERS overrides the automatic VRAM budget calculation.
	numGPULayers := computeGPULayerBudget(m, maxSeqLen)
	if numGPULayers > cfg.NumLayers {
		numGPULayers = cfg.NumLayers
	}
	if envLayers := os.Getenv("DLGO_GPU_LAYERS"); envLayers != "" {
		if n, err := fmt.Sscanf(envLayers, "%d", &numGPULayers); n == 1 && err == nil {
			if numGPULayers < 0 {
				numGPULayers = 0
			}
			if numGPULayers > cfg.NumLayers {
				numGPULayers = cfg.NumLayers
			}
			fmt.Printf("[dlgo/gpu] DLGO_GPU_LAYERS=%d override\n", numGPULayers)
		}
	}

	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim

	// Retry loop: if GPU allocation fails, reduce layers and retry.
	// Limited to 3 retries to avoid VRAM fragmentation from repeated
	// partial allocations. Each retry does a full Sync + buffer table
	// reset to guarantee VRAM is actually reclaimed.
	const maxRetries = 3
	var gm *GpuModel
	var rs *GpuRunState
	var kv *GpuKVCache
	var ropeCosTable, ropeSinTable Buf
	var layerConfs []*LayerConf
	var q8_1Scratch Buf
	var isPartial bool

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if numGPULayers <= 0 {
			return nil, fmt.Errorf("insufficient VRAM (%.0f MB) for even 1 layer — use CPU mode", totalVRAM)
		}
		isPartial = numGPULayers < cfg.NumLayers

		allocErr := func() error {
			var err error
			gm, err = UploadModel(m, numGPULayers)
			if err != nil {
				return fmt.Errorf("upload model: %w", err)
			}

			// UploadModel may have stopped early due to VRAM floor check.
			// Use the actual number of layers that made it to GPU.
			if gm.NumGPULayers < numGPULayers {
				fmt.Printf("[dlgo/gpu] Upload stopped early: %d/%d layers on GPU (VRAM floor)\n",
					gm.NumGPULayers, numGPULayers)
				numGPULayers = gm.NumGPULayers
				isPartial = numGPULayers < cfg.NumLayers
			}

			rs, err = NewGpuRunState(dim, qDim, kvDim, ffnDim, cfg.VocabSize)
			if err != nil {
				return err
			}
			needsKV := make([]bool, cfg.NumLayers)
			for l := 0; l < cfg.NumLayers; l++ {
				needsKV[l] = layerNeedsKV(&m.Layers[l])
			}
			kv, err = NewGpuKVCache(cfg.NumLayers, numGPULayers, maxSeqLen, kvDim, needsKV)
			if err != nil {
				return err
			}

			// Upload RoPE tables. Prefer pre-computed tables from CPU RunState
			// (if available), otherwise compute from config. RunState may be nil
			// when CPU buffers were freed before GPU pipeline creation.
			var cosTable, sinTable []float32
			if cpuPipeline.RunState != nil {
				cosTable, sinTable = cpuPipeline.RunState.RoPETables()
			}
			if cosTable != nil && sinTable != nil {
				ropeCosTable, err = UploadF32Slice(cosTable)
				if err != nil {
					return fmt.Errorf("upload RoPE cos table: %w", err)
				}
				ropeSinTable, err = UploadF32Slice(sinTable)
				if err != nil {
					return fmt.Errorf("upload RoPE sin table: %w", err)
				}
			} else {
				ropeDim := cfg.RopeDim
				if ropeDim <= 0 || ropeDim > cfg.HeadDim {
					ropeDim = cfg.HeadDim
				}
				cos, sin := ops.RoPEFrequencyTable(maxSeqLen, ropeDim, cfg.RopeFreqBase)
				ropeCosTable, err = UploadF32Slice(cos)
				if err != nil {
					return fmt.Errorf("upload RoPE cos table: %w", err)
				}
				ropeSinTable, err = UploadF32Slice(sin)
				if err != nil {
					return fmt.Errorf("upload RoPE sin table: %w", err)
				}
			}

			tempPipe := &GpuPipeline{
				RoPECosTable: ropeCosTable,
				RoPESinTable: ropeSinTable,
			}
			layerConfs = BuildLayerConfs(m, gm, tempPipe, rs, kv)

			maxDim := dim
			if ffnDim > maxDim {
				maxDim = ffnDim
			}
			q8_1NumBlocks := (maxDim + 31) / 32
			q8_1Scratch = Alloc(uint64(q8_1NumBlocks) * 36)
			if q8_1Scratch == 0 {
				return fmt.Errorf("alloc q8_1 scratch")
			}

			// Allocate SSM per-layer state (inside retry loop so floor check covers it).
			if cfg.FullAttentionInterval > 0 && cfg.SSMInnerSize > 0 {
				numSSMHeads := cfg.SSMTimeStepRank
				numSSMKVGroups := cfg.SSMGroupCount
				if numSSMKVGroups <= 0 {
					numSSMKVGroups = numSSMHeads
				}
				ssHeadVDim := cfg.SSMInnerSize / numSSMHeads
				ssHeadKDim := cfg.SSMStateSize
				sValueDim := numSSMHeads * ssHeadVDim
				sKeyDim := numSSMKVGroups * ssHeadKDim
				sQkvDim := sKeyDim*2 + sValueDim
				ssConvK := cfg.SSMConvKernel

				if err := rs.AllocSSMScratch(sQkvDim, sValueDim, numSSMHeads); err != nil {
					return err
				}

				ssmAlloc := allocChecker{}
				for sl := 0; sl < cfg.NumLayers; sl++ {
					if m.Layers[sl].Spec.Core == llm.CoreSSM && sl < numGPULayers {
						gl := &gm.Layers[sl]
						gl.SSMState = ssmAlloc.alloc(uint64(numSSMHeads * ssHeadKDim * ssHeadVDim * 4))
						gl.SSMConvBuf = ssmAlloc.alloc(uint64(ssConvK * sQkvDim * 4))
					}
				}
				if ssmAlloc.err != nil {
					return fmt.Errorf("gpu: SSM state alloc: %w", ssmAlloc.err)
				}
			}

			// Allocate GatedQ scratch (inside retry loop so floor check covers it).
			for gl := 0; gl < cfg.NumLayers; gl++ {
				if m.Layers[gl].Spec.GatedQ {
					if err := rs.AllocGatedQScratch(qDim); err != nil {
						return err
					}
					break
				}
			}

			// Post-allocation VRAM floor check: verify the GPU still has
			// enough free memory for the Windows display compositor, video
			// decode, and other system GPU users. Without this, the budget
			// solver can be slightly optimistic and leave the system in a
			// state where the display driver can't service frame requests,
			// causing a full system freeze.
			Sync() // flush so the driver sees all allocations
			postFree := int64(VRAMFreeBytes())
			// Soft floor: the C-level create_buffer() enforces a hard 512 MB
			// floor before each vkAllocateMemory, so this Go check is a
			// secondary guard with a 1 GB threshold.
			const vramFloor = 1024 * 1024 * 1024 // 1 GB
			if postFree < vramFloor {
				return fmt.Errorf("VRAM floor violated: only %.0f MB free (need %.0f MB)",
					float64(postFree)/(1024*1024), float64(vramFloor)/(1024*1024))
			}

			return nil
		}()

		if allocErr == nil {
			break
		}

		// Free everything, sync GPU, and reset buffer table to guarantee
		// VRAM is fully reclaimed before retrying with fewer layers.
		if gm != nil {
			gm.FreeAll()
		}
		if rs != nil {
			rs.FreeAll()
		}
		if kv != nil {
			kv.FreeAll()
		}
		freeBuf(ropeCosTable)
		freeBuf(ropeSinTable)
		freeBuf(q8_1Scratch)
		Sync()
		ResetBufferTable()
		ropeCosTable, ropeSinTable, q8_1Scratch = 0, 0, 0
		gm, rs, kv = nil, nil, nil
		layerConfs = nil

		if attempt >= maxRetries {
			return nil, fmt.Errorf("VRAM alloc failed after %d retries with %d layers: %v — use CPU mode",
				maxRetries, numGPULayers, allocErr)
		}

		// Halve layers on each retry instead of decrementing by 1.
		// This converges faster and avoids VRAM fragmentation from many small retries.
		prev := numGPULayers
		numGPULayers = numGPULayers / 2
		fmt.Printf("[dlgo/gpu] VRAM alloc failed with %d layers (%v), retrying with %d... (allocated %.0f MB, driver free %.0f MB)\n",
			prev, allocErr, numGPULayers, float64(AllocatedBytes())/(1024*1024), float64(VRAMFreeBytes())/(1024*1024))
	}

	// Release mmap pages from physical RAM. GPU upload reads the entire model
	// file through mmap, pulling ~N GB into the page cache. These pages are no
	// longer needed (data is now in VRAM) and would otherwise compete with the
	// CPU-side KV cache and run state allocations that follow.
	mmap.TrimWorkingSet()

	dp4aAvail := HasDp4a()
	dp4aDisabled := os.Getenv("DLGO_NO_DP4A") == "1"
	if dp4aAvail && !dp4aDisabled {
		for _, lc := range layerConfs {
			if lc == nil {
				continue
			}
			lc.SetDP4A(q8_1Scratch)
		}
		rs.MoEUseDp4a = true
		fmt.Println("[dlgo/gpu] dp4a enabled for attention + FFN + MoE (per-tensor safe types)")
	} else if dp4aDisabled {
		fmt.Println("[dlgo/gpu] dp4a disabled via DLGO_NO_DP4A=1")
	} else {
		fmt.Println("[dlgo/gpu] dp4a not available on this GPU")
	}

	if isPartial {
		fmt.Printf("[dlgo/gpu] Partial GPU: %d/%d layers on GPU, %d on CPU\n",
			numGPULayers, cfg.NumLayers, cfg.NumLayers-numGPULayers)
	} else {
		fmt.Printf("[dlgo/gpu] Model loaded to GPU (%d layers)\n", cfg.NumLayers)
	}

	pipe := &GpuPipeline{
		CPUModel:        m,
		GpuModel:        gm,
		Tokenizer:       cpuPipeline.Tokenizer,
		KVCache:         kv,
		RunState:        rs,
		MaxSeqLen:       maxSeqLen,
		LogitsBuf:       make([]float32, cfg.VocabSize),
		LayerConfs:      layerConfs,
		Q8_1Scratch:     q8_1Scratch,
		NumGPULayers:    numGPULayers,
		IsPartialGPU:    isPartial,
		RoPECosTable:    ropeCosTable,
		RoPESinTable:    ropeSinTable,
	}

	// Use fused forward when ALL layers are on GPU (including MoE models).
	// For MoE: C side handles attention + residual + norm, returns early (ffn_type=3);
	// Go side then handles MoE FFN dispatch.
	if !isPartial {
		pipe.UseFusedForward = supportsFusedForwardGPU(m)
	}

	// GatedQ and SSM state were allocated inside the retry loop (so the VRAM
	// floor check covers them). Just set the pipeline flags here.
	for l := 0; l < cfg.NumLayers; l++ {
		if m.Layers[l].Spec.GatedQ {
			pipe.HasGatedQ = true
			break
		}
	}

	if cfg.FullAttentionInterval > 0 && cfg.SSMInnerSize > 0 {
		ssmLayerCount := 0
		for l := 0; l < cfg.NumLayers; l++ {
			if m.Layers[l].Spec.Core == llm.CoreSSM && l < numGPULayers {
				ssmLayerCount++
			}
		}
		pipe.HasSSM = true
		numHeads := cfg.SSMTimeStepRank
		numKVGroups := cfg.SSMGroupCount
		if numKVGroups <= 0 {
			numKVGroups = numHeads
		}
		headVDim := cfg.SSMInnerSize / numHeads
		headKDim := cfg.SSMStateSize
		fmt.Printf("[dlgo/gpu] SSM state on GPU (%d SSM layers, %d heads, %d KV groups, state=%dx%d)\n",
			ssmLayerCount, numHeads, numKVGroups, headKDim, headVDim)
	}

	if cfg.ExpertCount > 0 {
		pipe.HasMoE = true
	}

	// Detect MLA (Multi-head Latent Attention) layers
	for l := 0; l < cfg.NumLayers; l++ {
		if m.Layers[l].Spec.Core == llm.CoreMLA {
			pipe.HasMLA = true
			break
		}
	}

	// RoPE tables already uploaded above (before BuildLayerConfs)

	// Check if any layer needs CPU attention fallback
	hasCPUAttn := false
	cpuAttnCount := 0
	for l := 0; l < numGPULayers; l++ {
		if gm.Layers[l].CPUAttn {
			hasCPUAttn = true
			cpuAttnCount++
		}
	}
	if hasCPUAttn {
		fmt.Printf("[dlgo/gpu] CPU attention fallback: %d/%d layers need it\n", cpuAttnCount, numGPULayers)
		printedCPU := false
		for l := 0; l < numGPULayers; l++ {
			wqType := uint32(0)
			if m.Layers[l].Wq != nil {
				wqType = m.Layers[l].Wq.Type
			}
			if !gm.Layers[l].CPUAttn {
				fmt.Printf("[dlgo/gpu]   Layer %d: GPU attention (Wq type=%d)\n", l, wqType)
			} else if !printedCPU {
				fmt.Printf("[dlgo/gpu]   Layer %d: CPU attention (Wq type=%d)\n", l, wqType)
				printedCPU = true
			}
		}
		if cpuAttnCount == numGPULayers {
			pipe.AllCPUAttn = true
		}
	}

	// Allocate CPU-side state if needed (partial GPU, MoE, MLA, CPU attn, or hybrid SSM).
	// Each allocation is guarded by a RAM check: if system RAM usage would exceed
	// 85% of total, we skip the allocation and let mmap handle it instead.
	needCPUState := isPartial || cfg.ExpertCount > 0 || pipe.HasMLA || hasCPUAttn
	if needCPUState {
		if canAllocRAM(int64(llm.EstimateRuntimeBytes(cfg, maxSeqLen))) {
			pipe.CPURunState = llm.NewRunState(cfg, maxSeqLen)
		} else {
			fmt.Printf("[dlgo/gpu] WARNING: skipping CPU RunState allocation (RAM pressure)\n")
		}

		if isPartial {
			cpuLayers := cfg.NumLayers - numGPULayers
			kvCacheBytes := int64(2 * cpuLayers * maxSeqLen * kvDim * 4)
			if canAllocRAM(kvCacheBytes) {
				pipe.CPUKVCache = memory.NewMultiLayerKVCache(cfg.NumLayers, maxSeqLen, kvDim)
			} else {
				fmt.Printf("[dlgo/gpu] WARNING: skipping CPU KV cache allocation (RAM pressure, need %.0f MB)\n",
					float64(kvCacheBytes)/(1024*1024))
			}
			if pipe.CPURunState != nil {
				pipe.CPUBatchState = llm.NewBatchState(cfg, maxSeqLen)
			}
		}

		if (pipe.HasMLA || hasCPUAttn) && pipe.CPUKVCache == nil {
			kvCacheBytes := int64(2 * cfg.NumLayers * maxSeqLen * kvDim * 4)
			if canAllocRAM(kvCacheBytes) {
				pipe.CPUKVCache = memory.NewMultiLayerKVCache(cfg.NumLayers, maxSeqLen, kvDim)
				if hasCPUAttn {
					fmt.Printf("[dlgo/gpu] CPU attention fallback: allocated CPU KV cache (%d layers)\n", cfg.NumLayers)
				}
			} else {
				fmt.Printf("[dlgo/gpu] WARNING: skipping CPU KV cache for MLA/attn (RAM pressure)\n")
			}
		}

		if cfg.ExpertCount > 0 {
			moeLayerCount := 0
			gpuMoECount := 0
			for l := 0; l < cfg.NumLayers; l++ {
				if m.Layers[l].Spec.FFN == llm.FFNMoE || m.Layers[l].Spec.FFN == llm.FFNMoESwiOAI {
					moeLayerCount++
					if l < numGPULayers && gm.Layers[l].MoEOnGPU {
						gpuMoECount++
					}
				}
			}
			if gpuMoECount > 0 {
				fmt.Printf("[dlgo/gpu] MoE: %d/%d MoE layers on GPU, %d on CPU\n",
					gpuMoECount, moeLayerCount, moeLayerCount-gpuMoECount)
			} else {
				fmt.Printf("[dlgo/gpu] Hybrid MoE: %d MoE layers (expert FFN on CPU)\n", moeLayerCount)
			}
		}
	}

	// Pin CPU layers to RAM for optimal inference speed (avoid page faults).
	// Budget: never push total system RAM usage past 85%.
	if isPartial {
		pinCPULayersToRAM(m, numGPULayers)
		mmap.TrimWorkingSet()
	}

	return pipe, nil
}

// ResetState zeros all caches and SSM state for a fresh inference.
func (p *GpuPipeline) ResetState() {
	p.KVCache.Reset()
	if p.CPUKVCache != nil {
		p.CPUKVCache.Reset()
	}
	if p.HasSSM {
		mcfg2 := p.CPUModel.Config
		numHeads := mcfg2.SSMTimeStepRank
		numKVGroups := mcfg2.SSMGroupCount
		if numKVGroups <= 0 {
			numKVGroups = numHeads
		}
		headVDim := mcfg2.SSMInnerSize / numHeads
		headKDim := mcfg2.SSMStateSize
		keyDim := numKVGroups * headKDim
		qkvDim := keyDim*2 + numHeads*headVDim
		convK := mcfg2.SSMConvKernel
		for l := 0; l < mcfg2.NumLayers; l++ {
			gl := &p.GpuModel.Layers[l]
			if gl.SSMState != 0 {
				ZeroFill(gl.SSMState, uint64(numHeads*headKDim*headVDim*4))
				ZeroFill(gl.SSMConvBuf, uint64(convK*qkvDim*4))
			}
		}
		if p.CPURunState != nil && p.CPURunState.SSMState != nil {
			p.CPURunState.SSMState.Reset()
		}
	}
}

// canAllocRAM checks whether allocating nbytes of heap memory would push
// total system RAM usage past 85% of physical RAM. Returns false if the
// allocation should be skipped to prevent system instability.
func canAllocRAM(nbytes int64) bool {
	memInfo, err := mmap.GetSystemMemInfo()
	if err != nil {
		return true // can't check, assume OK
	}
	totalRAM := int64(memInfo.TotalPhysical)
	availRAM := int64(memInfo.AvailablePhysical)
	usedRAM := totalRAM - availRAM
	ceiling := int64(float64(totalRAM) * 0.85)
	return usedRAM+nbytes < ceiling
}

// pinCPULayersToRAM copies non-GPU layer weights from mmap to heap memory,
// prioritizing earlier layers and respecting a system RAM budget.
//
// Budget is computed against total physical RAM to prevent the system from
// thrashing: we allow pinning only until total system RAM usage would reach
// 85% of total physical RAM. Remaining layers stay on mmap and are served
// via demand paging (the OS page cache handles them transparently).
func pinCPULayersToRAM(m *llm.Model, numGPULayers int) {
	memInfo, err := mmap.GetSystemMemInfo()
	if err != nil {
		fmt.Printf("[dlgo/gpu] Warning: couldn't query system RAM: %v\n", err)
		return
	}

	totalRAM := int64(memInfo.TotalPhysical)
	availRAM := int64(memInfo.AvailablePhysical)
	usedRAM := totalRAM - availRAM

	// Use a conservative 70% ceiling for pinning. The remaining 15%
	// (up to the 85% process limit) is headroom for mmap page cache,
	// Go runtime/GC overhead, and other system processes.
	maxUsage := int64(float64(totalRAM) * 0.70)
	budget := maxUsage - usedRAM
	if budget < 0 {
		budget = 0
	}

	fmt.Printf("[dlgo/gpu] RAM budget: %.0f MB free of %.0f MB total (%.0f%% used), pin budget %.0f MB\n",
		float64(availRAM)/(1024*1024), float64(totalRAM)/(1024*1024),
		float64(usedRAM)/float64(totalRAM)*100, float64(budget)/(1024*1024))

	pinnedBytes := int64(0)
	pinnedLayers := 0

	for l := numGPULayers; l < len(m.Layers); l++ {
		layerBytes := llm.EstimateLayerBytes(&m.Layers[l])
		if pinnedBytes+layerBytes > budget {
			break
		}
		if !canAllocRAM(layerBytes) {
			fmt.Printf("[dlgo/gpu] Stopping pin: canAllocRAM rejected %d MB layer\n",
				layerBytes/(1024*1024))
			break
		}
		llm.PinLayerToRAM(&m.Layers[l])
		pinnedBytes += layerBytes
		pinnedLayers++

		// Every 4 layers, trim working set to evict the mmap source pages
		// that were read during the copy but are no longer needed.
		if pinnedLayers%4 == 0 {
			mmap.TrimWorkingSet()
		}
	}

	mmap.TrimWorkingSet()
	remaining := len(m.Layers) - numGPULayers - pinnedLayers
	fmt.Printf("[dlgo/gpu] Pinned %d CPU layers to RAM (%.0f MB), %d layers on mmap\n",
		pinnedLayers, float64(pinnedBytes)/(1024*1024), remaining)
}

// FreeAll releases all GPU resources held by this pipeline.
// Sync is called first to ensure no in-flight commands reference these buffers,
// which allows the Vulkan driver to immediately reclaim the device memory.
func (p *GpuPipeline) FreeAll() {
	if p == nil {
		return
	}
	Sync()
	p.GpuModel.FreeAll()
	p.RunState.FreeAll()
	p.KVCache.FreeAll()
	p.BatchState.Free()
	freeBuf(p.Q8_1Scratch)
	freeBuf(p.RoPECosTable)
	freeBuf(p.RoPESinTable)
	ResetBufferTable()
}

// GenerateResult holds detailed output from a GPU generation run.
type GenerateResult struct {
	Text           string
	Tokens         []int32
	TokensPerSec   float64
	PrefillTimeMs  float64
	GenerateTimeMs float64
	TotalTokens    int
	PromptTokens   int
}

// GenerateDetailed runs generation on GPU with detailed timing.
func (p *GpuPipeline) GenerateDetailed(prompt string, cfg llm.GenerateConfig) (*GenerateResult, error) {
	tokens := p.Tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenizer produced no tokens")
	}
	if len(tokens) >= p.MaxSeqLen {
		return nil, fmt.Errorf("prompt too long: %d tokens (max %d)", len(tokens), p.MaxSeqLen)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	if cfg.Seed < 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	p.ResetState()

	mcfg := p.CPUModel.Config
	npos := len(tokens)

	// Chunked prefill: cap batch buffer size to avoid large VRAM allocation
	// Keep prefill chunks small to avoid large VRAM spikes from BatchState
	// allocation during inference. With a 9B model, 4096 tokens × ffnDim
	// can allocate 1+ GB of transient VRAM for batch buffers.
	const prefillChunkSize = 512
	batchSize := npos
	if batchSize > prefillChunkSize {
		batchSize = prefillChunkSize
	}

	if p.BatchState == nil || p.BatchState.Npos < batchSize {
		if p.BatchState != nil {
			p.BatchState.Free()
		}
		dim := mcfg.EmbeddingDim
		qDim := mcfg.NumHeads * mcfg.HeadDim
		kvDim := mcfg.NumKVHeads * mcfg.HeadDim
		ffnDim := mcfg.FFNDim
		var bsErr error
		p.BatchState, bsErr = NewGpuBatchState(batchSize, dim, qDim, kvDim, ffnDim)
		if bsErr != nil {
			return nil, fmt.Errorf("gpu: prefill batch alloc: %w", bsErr)
		}
		p.BatchLayerConfs = BuildBatchLayerConfs(p.CPUModel, p.GpuModel, p, p.BatchState, p.KVCache)
		if p.HasGatedQ {
			if err := p.BatchState.AllocGatedQBatch(batchSize, qDim); err != nil {
				return nil, fmt.Errorf("gpu: prefill GatedQ batch alloc: %w", err)
			}
		}
		if p.HasSSM {
			numHeads := mcfg.SSMTimeStepRank
			numKVGroups := mcfg.SSMGroupCount
			if numKVGroups <= 0 {
				numKVGroups = numHeads
			}
			headVDim := mcfg.SSMInnerSize / numHeads
			headKDim := mcfg.SSMStateSize
			keyDim := numKVGroups * headKDim
			qkvDim := keyDim*2 + numHeads*headVDim
			valueDim := numHeads * headVDim
			if err := p.BatchState.AllocSSMBatch(batchSize, qkvDim, valueDim, numHeads); err != nil {
				return nil, fmt.Errorf("gpu: prefill SSM batch alloc: %w", err)
			}
		}
	}

	prefillStart := time.Now()
	if p.IsPartialGPU {
		for i, tok := range tokens {
			GpuForwardPartial(p.CPUModel, p.GpuModel, tok, i, p.KVCache, p.RunState, p.LogitsBuf, p.LayerConfs, p)
		}
	} else if !p.UseFusedForward {
		for i, tok := range tokens {
			GpuForward(p.CPUModel, p.GpuModel, tok, i, p.KVCache, p.RunState, p.LogitsBuf, p)
		}
	} else if p.HasMoE {
		// MoE prefill: per-token fused forward (batch prefill not yet supported for MoE)
		for i, tok := range tokens {
			GpuForwardFusedSSM(p.CPUModel, p.GpuModel, tok, i, p.KVCache, p.RunState, p.LogitsBuf, p.LayerConfs, p)
		}
	} else {
		isHybrid := isHybridSSMModel(p.CPUModel)
		if isHybrid {
			// Chunked prefill: process prompt in chunks to bound VRAM usage
			for startPos := 0; startPos < npos; startPos += prefillChunkSize {
				end := startPos + prefillChunkSize
				if end > npos {
					end = npos
				}
				chunkTokens := tokens[startPos:end]
				isLast := end >= npos
				GpuForwardPrefillBatchHybrid(p.CPUModel, p.GpuModel, chunkTokens, p.KVCache, p.RunState,
					p.BatchState, p.LogitsBuf, p.BatchLayerConfs, p, startPos, isLast)
				Sync()
			}
		} else {
			GpuForwardPrefillBatch(p.CPUModel, p.GpuModel, tokens, p.KVCache, p.RunState,
				p.BatchState, p.LogitsBuf, p.BatchLayerConfs)
		}
	}
	Sync()
	prefillMs := float64(time.Since(prefillStart).Microseconds()) / 1000.0

	// Generate
	genStart := time.Now()
	var generated []int32
	var recentTokens []int32
	var genText strings.Builder
	stopStrings := gpuStopStrings()
	pos := len(tokens)

	// Build token pieces for grammar masking
	var tokenPieces []string
	var eosTokens map[int32]bool
	if cfg.Grammar != nil {
		vocabSize := p.Tokenizer.VocabSize()
		tokenPieces = make([]string, vocabSize)
		for i := 0; i < vocabSize; i++ {
			tokenPieces[i] = p.Tokenizer.DecodeToken(int32(i))
		}
		eosTokens = map[int32]bool{p.CPUModel.Config.EOS: true}
		for _, stop := range p.CPUModel.Config.StopTokens {
			eosTokens[stop] = true
		}
	}

	// Grammar-aware sampling helper
	gpuGrammarSample := func(logits []float32) int {
		if cfg.Grammar != nil {
			cfg.Grammar.ApplyToLogits(logits, tokenPieces, eosTokens)
		}
		return ops.SampleToken(logits, cfg.Sampler, recentTokens, rng)
	}

	nextToken := gpuGrammarSample(p.LogitsBuf)
	var tokenText string

	firstTok := int32(nextToken)
	if firstTok == p.CPUModel.Config.EOS {
		goto done
	}
	for _, stop := range p.CPUModel.Config.StopTokens {
		if firstTok == stop {
			goto done
		}
	}

	generated = append(generated, int32(nextToken))
	recentTokens = append(recentTokens, int32(nextToken))

	// Advance grammar state
	if cfg.Grammar != nil {
		cfg.Grammar.AcceptToken(p.Tokenizer.DecodeToken(int32(nextToken)))
	}

	tokenText = p.Tokenizer.DecodeToken(int32(nextToken))
	genText.WriteString(tokenText)
	if !gpuCheckTextStop(genText.String(), stopStrings) && cfg.Stream != nil {
		cfg.Stream(tokenText)
	}

	for step := 1; step < cfg.MaxTokens; step++ {
		if pos >= p.MaxSeqLen-1 {
			break
		}
		lastTok := int32(nextToken)
		if lastTok == p.CPUModel.Config.EOS {
			break
		}
		for _, stop := range p.CPUModel.Config.StopTokens {
			if lastTok == stop {
				goto done
			}
		}

		if p.IsPartialGPU {
			GpuForwardPartial(p.CPUModel, p.GpuModel, lastTok, pos, p.KVCache, p.RunState, p.LogitsBuf, p.LayerConfs, p)
			if step%32 == 0 {
				mmap.TrimWorkingSet()
			}
		} else if p.UseFusedForward {
			GpuForwardFusedSSM(p.CPUModel, p.GpuModel, lastTok, pos, p.KVCache, p.RunState, p.LogitsBuf, p.LayerConfs, p)
		} else if err := GpuForward(p.CPUModel, p.GpuModel, lastTok, pos, p.KVCache, p.RunState, p.LogitsBuf, p); err != nil {
			return nil, err
		}
		pos++

		nextToken = gpuGrammarSample(p.LogitsBuf)
		generated = append(generated, int32(nextToken))
		recentTokens = append(recentTokens, int32(nextToken))
		if len(recentTokens) > 256 {
			recentTokens = recentTokens[1:]
		}

		// Advance grammar state
		if cfg.Grammar != nil && !eosTokens[int32(nextToken)] {
			cfg.Grammar.AcceptToken(p.Tokenizer.DecodeToken(int32(nextToken)))
		}

		tokenText = p.Tokenizer.DecodeToken(int32(nextToken))
		genText.WriteString(tokenText)

		if gpuCheckTextStop(genText.String(), stopStrings) {
			break
		}

		isStop := int32(nextToken) == p.CPUModel.Config.EOS
		if !isStop {
			for _, st := range p.CPUModel.Config.StopTokens {
				if int32(nextToken) == st {
					isStop = true
					break
				}
			}
		}
		if cfg.Stream != nil && !isStop {
			cfg.Stream(tokenText)
		}
	}

done:
	Sync()
	genMs := float64(time.Since(genStart).Microseconds()) / 1000.0

	text := llm.TrimStopText(p.Tokenizer.Decode(generated), p.CPUModel.Config)
	var tokPerSec float64
	if genMs > 0 {
		tokPerSec = float64(len(generated)) / (genMs / 1000.0)
	}

	return &GenerateResult{
		Text:           text,
		Tokens:         generated,
		TokensPerSec:   tokPerSec,
		PrefillTimeMs:  prefillMs,
		GenerateTimeMs: genMs,
		TotalTokens:    len(generated),
		PromptTokens:   len(tokens),
	}, nil
}

// supportsBatchPrefillGPU gates the batched prefill path.
// All models now support batch prefill: pure attention models use the standard
// batch path, and hybrid SSM+attention models use the hybrid batch path.
func supportsBatchPrefillGPU(m *llm.Model) bool {
	return true
}

// isHybridSSMModel returns true if the model has both SSM and attention layers,
// or if it has MoE layers (which require CPU-side expert FFN).
func isHybridSSMModel(m *llm.Model) bool {
	if m.Config.ExpertCount > 0 {
		return true
	}
	return m.Config.FullAttentionInterval > 0 && m.Config.SSMInnerSize > 0
}

// supportsFusedForwardGPU reports whether the fused single-token path can
// execute the model without silently skipping any quantized matvecs. The
// fused C path does not have CPU fallback, so every tensor it touches must
// have a native GPU kernel.
func supportsFusedForwardGPU(m *llm.Model) bool {
	supported := func(t *core.QuantizedTensor) bool {
		if t == nil {
			return true
		}
		return supportsGPUQType(t.Type)
	}
	if m.Output != nil && !supported(m.Output) {
		return false
	}
	for i := range m.Layers {
		l := &m.Layers[i]
		// Fused C path only supports RMSNorm; force non-fused for LayerNorm models.
		if l.Spec.Norm == llm.NormLayer {
			return false
		}
		if l.Spec.FFN == llm.FFNMoE || l.Spec.FFN == llm.FFNMoESwiOAI {
			for _, t := range []*core.QuantizedTensor{
				l.SSMInProj, l.AttnGate, l.SSMAlpha, l.SSMBeta, l.SSMOut,
				l.Wq, l.Wk, l.Wv, l.Wo,
			} {
				if !supported(t) {
					return false
				}
			}
			continue
		}
		if l.Spec.Core == llm.CoreSSM {
			for _, t := range []*core.QuantizedTensor{
				l.SSMInProj, l.AttnGate, l.SSMAlpha, l.SSMBeta, l.SSMOut,
				l.FFNGate, l.FFNUp, l.FFNDown,
			} {
				if !supported(t) {
					return false
				}
			}
		} else if l.Spec.GatedQ {
			for _, t := range []*core.QuantizedTensor{
				l.Wq, l.Wk, l.Wv, l.Wo,
				l.FFNGate, l.FFNUp, l.FFNDown,
			} {
				if !supported(t) {
					return false
				}
			}
		} else {
			for _, t := range []*core.QuantizedTensor{
				l.Wq, l.Wk, l.Wv, l.Wo,
				l.FFNGate, l.FFNUp, l.FFNDown,
			} {
				if !supported(t) {
					return false
				}
			}
		}
	}
	return true
}

func gpuStopStrings() []string {
	return []string{
		"<end_of_turn><eos>",
		"<eos>",
		"<|im_end|>",
		"<|endoftext|>",
		"<|end|>",
		"<|return|>",
		"</s>",
		"<|assistant|>",
		"<|user|>",
		"<|observation|>",
		"<end_of_turn>",
		"<|eot_id|>",
		"<|channel|>",
		"<|start|>",
		"<|message|>",
		"<|constrain|>",
		"<|call|>",
	}
}

func gpuCheckTextStop(text string, stops []string) bool {
	for _, ss := range stops {
		if strings.HasSuffix(text, ss) {
			return true
		}
	}
	return false
}
