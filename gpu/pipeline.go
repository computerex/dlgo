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

	// Batch state (estimate for 128 tokens)
	batchTokens := int64(128)
	total += batchTokens * (dim + dim + qDim + kvDim + kvDim + qDim + dim + dim + dim + ffnDim + ffnDim + ffnDim + dim) * 4

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

	// Safety margin to account for IQ tables, Vulkan driver overhead,
	// descriptor sets, command buffers, and runtime fragmentation.
	// Use max(256 MB, 3% of total VRAM) to prevent near-OOM corruption.
	vram := int64(VRAMBytes())
	margin := int64(256 * 1024 * 1024)
	if pct := vram * 3 / 100; pct > margin {
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

// computeGPULayerBudget determines how many layers fit in available VRAM,
// accounting for both weight data AND per-layer KV cache VRAM.
func computeGPULayerBudget(m *llm.Model, maxSeqLen int) int {
	freeVRAM := int64(VRAMFreeBytes())
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

	available := freeVRAM - fixedOverhead - nonLayerBytes
	if available <= 0 {
		return 0
	}

	// Per-layer KV cache cost
	kvDim := int64(m.Config.NumKVHeads * m.Config.HeadDim)
	kvPerLayer := 2 * int64(maxSeqLen) * kvDim * 2 // K + V buffers (FP16)

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

	// Greedily add layers: each layer costs weights + KV cache (attention only) + optional SSM state
	numLayers := 0
	for l := 0; l < len(m.Layers); l++ {
		layerCost := llm.EstimateLayerBytes(&m.Layers[l])
		if layerNeedsKV(&m.Layers[l]) {
			layerCost += kvPerLayer
		}
		if m.Layers[l].Spec.Core == llm.CoreSSM {
			layerCost += ssmPerLayer
		}
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
	}

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
	freeVRAM := float64(VRAMFreeBytes()) / (1024 * 1024)
	fmt.Printf("[dlgo/gpu] Uploading model to %s (%.0f MB total, %.0f MB free)...\n",
		DeviceName(), totalVRAM, freeVRAM)

	// Determine how many layers fit in VRAM.
	// DLGO_GPU_LAYERS overrides the automatic VRAM budget calculation.
	numGPULayers := computeGPULayerBudget(m, cpuPipeline.MaxSeqLen)
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

			rs = NewGpuRunState(dim, qDim, kvDim, ffnDim, cfg.VocabSize)
			needsKV := make([]bool, cfg.NumLayers)
			for l := 0; l < cfg.NumLayers; l++ {
				needsKV[l] = layerNeedsKV(&m.Layers[l])
			}
			kv = NewGpuKVCache(cfg.NumLayers, numGPULayers, cpuPipeline.MaxSeqLen, kvDim, needsKV)

			cosTable, sinTable := cpuPipeline.RunState.RoPETables()
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
				cos, sin := ops.RoPEFrequencyTable(cpuPipeline.MaxSeqLen, ropeDim, cfg.RopeFreqBase)
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
		fmt.Printf("[dlgo/gpu] VRAM alloc failed with %d layers (%v), retrying with %d...\n",
			prev, allocErr, numGPULayers)
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
		MaxSeqLen:       cpuPipeline.MaxSeqLen,
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


	hasGatedQ := false
	for l := 0; l < cfg.NumLayers; l++ {
		if m.Layers[l].Spec.GatedQ {
			hasGatedQ = true
			break
		}
	}
	if hasGatedQ {
		rs.AllocGatedQScratch(qDim)
		pipe.HasGatedQ = true
	}

	if cfg.FullAttentionInterval > 0 && cfg.SSMInnerSize > 0 {
		numHeads := cfg.SSMTimeStepRank
		numKVGroups := cfg.SSMGroupCount
		if numKVGroups <= 0 {
			numKVGroups = numHeads
		}
		headVDim := cfg.SSMInnerSize / numHeads
		headKDim := cfg.SSMStateSize
		valueDim := numHeads * headVDim
		keyDim := numKVGroups * headKDim
		qkvDim := keyDim*2 + valueDim
		convK := cfg.SSMConvKernel

		rs.AllocSSMScratch(qkvDim, valueDim, numHeads)

		ssmLayerCount := 0
		for l := 0; l < cfg.NumLayers; l++ {
			if m.Layers[l].Spec.Core == llm.CoreSSM && l < numGPULayers {
				gl := &gm.Layers[l]
				gl.SSMState = Alloc(uint64(numHeads * headKDim * headVDim * 4))
				gl.SSMConvBuf = Alloc(uint64(convK * qkvDim * 4))
				ssmLayerCount++
			}
		}

		pipe.HasSSM = true
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
		if canAllocRAM(int64(llm.EstimateRuntimeBytes(cfg, cpuPipeline.MaxSeqLen))) {
			pipe.CPURunState = llm.NewRunState(cfg, cpuPipeline.MaxSeqLen)
		} else {
			fmt.Printf("[dlgo/gpu] WARNING: skipping CPU RunState allocation (RAM pressure)\n")
		}

		if isPartial {
			cpuLayers := cfg.NumLayers - numGPULayers
			kvCacheBytes := int64(2 * cpuLayers * cpuPipeline.MaxSeqLen * kvDim * 4)
			if canAllocRAM(kvCacheBytes) {
				pipe.CPUKVCache = memory.NewMultiLayerKVCache(cfg.NumLayers, cpuPipeline.MaxSeqLen, kvDim)
			} else {
				fmt.Printf("[dlgo/gpu] WARNING: skipping CPU KV cache allocation (RAM pressure, need %.0f MB)\n",
					float64(kvCacheBytes)/(1024*1024))
			}
			if pipe.CPURunState != nil {
				pipe.CPUBatchState = llm.NewBatchState(cfg, cpuPipeline.MaxSeqLen)
			}
		}

		if (pipe.HasMLA || hasCPUAttn) && pipe.CPUKVCache == nil {
			kvCacheBytes := int64(2 * cfg.NumLayers * cpuPipeline.MaxSeqLen * kvDim * 4)
			if canAllocRAM(kvCacheBytes) {
				pipe.CPUKVCache = memory.NewMultiLayerKVCache(cfg.NumLayers, cpuPipeline.MaxSeqLen, kvDim)
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
	const prefillChunkSize = 4096
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
		p.BatchState = NewGpuBatchState(batchSize, dim, qDim, kvDim, ffnDim)
		p.BatchLayerConfs = BuildBatchLayerConfs(p.CPUModel, p.GpuModel, p, p.BatchState, p.KVCache)
		if p.HasGatedQ {
			p.BatchState.AllocGatedQBatch(batchSize, qDim)
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
			p.BatchState.AllocSSMBatch(batchSize, qkvDim, valueDim, numHeads)
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
	nextToken := ops.SampleToken(p.LogitsBuf, cfg.Sampler, recentTokens, rng)
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

		nextToken = ops.SampleToken(p.LogitsBuf, cfg.Sampler, recentTokens, rng)
		generated = append(generated, int32(nextToken))
		recentTokens = append(recentTokens, int32(nextToken))
		if len(recentTokens) > 256 {
			recentTokens = recentTokens[1:]
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
