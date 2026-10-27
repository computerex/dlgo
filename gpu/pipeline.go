//go:build cgo && vulkan

package gpu

import (
	"crypto/sha256"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
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

	RoPECosTable    Buf
	RoPESinTable    Buf
	RoPECosTableSWA Buf // SWA-specific RoPE tables (0 if not needed)
	RoPESinTableSWA Buf

	// Qwen M-RoPE: when mropeSections is set, image-token cos/sin rows are
	// computed on CPU and patched into the RoPE tables at the token's slot
	// before each embedding forward.
	mropeSections [4]int
	mropeFreq     float32
	mropeDim      int
	mropeEnabled  bool
	// cpuRoPECos/cpuRoPESin hold a CPU copy of the standard RoPE tables, used
	// to patch slot rows for text tokens whose logical position != slot.
	cpuRoPECos []float32
	cpuRoPESin []float32

	// Partial GPU offloading: layers [0, NumGPULayers) are on GPU,
	// layers [NumGPULayers, NumLayers) run on CPU (RAM or mmap).
	NumGPULayers int
	IsPartialGPU bool // true when some layers are on CPU

	// CPU-side state for hybrid MoE or partial GPU offloading
	CPURunState   *llm.RunState
	CPUKVCache    *memory.MultiLayerKVCache // KV cache for CPU layers
	CPUBatchState *llm.BatchState           // batch state for CPU prefill

	AllCPUAttn bool // true if ALL GPU layers use CPU attention fallback

	// UseBatchPrefill routes prompt processing for hybrid short-convolution +
	// MoE models through GpuForwardPrefillBatchShortConv instead of one forward
	// per token. Set lazily by ensurePrefillBatchState, which is also where the
	// extra scratch gets allocated. DLGO_PREFILL_BATCH=0 disables it.
	UseBatchPrefill bool

	// UseBatchPrefillHybridMoE routes prompt processing for hybrid SSM+GatedQ
	// attention models that carry a MoE FFN (Qwen3.5-class) through
	// GpuForwardPrefillBatchHybrid with the position-batched GPU MoE instead of
	// one full forward per token. Without it such a model re-reads every active
	// expert once per prompt token; with it the expert tensor is swept once per
	// chunk. Set by ensurePrefillBatchState, which is also where the MoE scratch
	// is allocated; DLGO_PREFILL_BATCH=0 disables it.
	UseBatchPrefillHybridMoE bool

	// hasCPUAttn is true when ANY GPU layer needs the CPU attention fallback;
	// batched prefill cannot serve those layers, so System One hidden
	// extraction refuses to run on such models.
	hasCPUAttn bool

	// CPUPipeline is the wrapped CPU pipeline, kept so System One multi can
	// degrade wide questions to CPU continuation scoring.
	CPUPipeline *llm.Pipeline

	// prefixCache holds device-side state snapshots keyed by the shared
	// System One prefix text, so repeated decisions over the same state
	// (the common document-briefing pattern) skip the long prefill entirely.
	// Entries are owned by the pipeline and freed on eviction or FreeAll.
	// Inference and snapshot lifetime require external serialization (the
	// server's inference gate). The mutex protects metadata only; it does not
	// make concurrent pipeline inference or eviction safe.
	prefixCacheMu sync.Mutex
	prefixCache   []prefixCacheEntry

	// Cross-request chat prefix reuse (agentic use). The previous generation's
	// prompt is remembered; when the next request's tokens extend it, only the
	// new suffix is prefilled. We keep the KV for [0, chatPrefixTokens) resident
	// (generation only appends beyond it) and snapshot just the GPU SSM/conv
	// recurrent state at the prefix boundary (cheap, ~tens of MB) so it can be
	// rewound. Only valid when SSM runs on the GPU (not cpuSSMDelta/cpuSSMFull)
	// and the hybrid batched prefill path is used. See chatPrefixEligible.
	chatPrefixTokens []int32
	chatPrefixSnap   *GpuStateSnapshot
	chatPrefixValid  bool
}

// maxThinkBudget bounds how many generated tokens may have EOS suppressed
// while the model is inside its thinking block. It intentionally does not
// scale with max_tokens: a client requesting a 128k budget must not also grant
// ~100k tokens of forced non-termination, which turns a rambling thinking
// block into an unstoppable one.
const maxThinkBudget = 4096

// prefixReuseEnabled toggles cross-request KV prefix reuse for chat. Default on;
// set DLGO_PREFIX_REUSE=0 to disable.
var prefixReuseEnabled = os.Getenv("DLGO_PREFIX_REUSE") != "0"

// prefixCacheMaxEntries caps cached prefixes. Each entry holds a KV+SSM
// snapshot for the cached prefix length (order ~100-200 MB for a 10K-token
// document), so this stays well inside a consumer GPU's spare VRAM.
const prefixCacheMaxEntries = 4

type prefixCacheEntry struct {
	key       [32]byte // sha256 of the prefix text
	prefixLen int      // prefix token count the snapshot was taken at
	snap      *GpuStateSnapshot
	lastUsed  time.Time
}

// lookupPrefixCache returns the cached snapshot for this exact prefix, or nil.
func (p *GpuPipeline) lookupPrefixCache(key [32]byte, prefixLen int) *GpuStateSnapshot {
	p.prefixCacheMu.Lock()
	defer p.prefixCacheMu.Unlock()
	for i := range p.prefixCache {
		e := &p.prefixCache[i]
		if e.key == key && e.prefixLen == prefixLen {
			e.lastUsed = time.Now()
			return e.snap
		}
	}
	return nil
}

// storePrefixCache inserts (or refreshes) a snapshot, evicting the
// least-recently-used entry past the cap. Ownership of snap passes to the
// cache; it must not be freed by the caller.
func (p *GpuPipeline) storePrefixCache(key [32]byte, prefixLen int, snap *GpuStateSnapshot) {
	p.prefixCacheMu.Lock()
	defer p.prefixCacheMu.Unlock()
	for i := range p.prefixCache {
		if p.prefixCache[i].key == key && p.prefixCache[i].prefixLen == prefixLen {
			p.prefixCache[i].snap.Free()
			p.prefixCache[i].snap = snap
			p.prefixCache[i].lastUsed = time.Now()
			return
		}
	}
	if len(p.prefixCache) >= prefixCacheMaxEntries {
		oldest := 0
		for i := range p.prefixCache {
			if p.prefixCache[i].lastUsed.Before(p.prefixCache[oldest].lastUsed) {
				oldest = i
			}
		}
		p.prefixCache[oldest].snap.Free()
		p.prefixCache = append(p.prefixCache[:oldest], p.prefixCache[oldest+1:]...)
	}
	p.prefixCache = append(p.prefixCache, prefixCacheEntry{key: key, prefixLen: prefixLen, snap: snap, lastUsed: time.Now()})
}

// freePrefixCache releases every cached snapshot.
func (p *GpuPipeline) freePrefixCache() {
	p.prefixCacheMu.Lock()
	defer p.prefixCacheMu.Unlock()
	for i := range p.prefixCache {
		p.prefixCache[i].snap.Free()
	}
	p.prefixCache = nil
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
	ropeTableBytes := 2 * int64(maxSeqLen) * (ropeDim / 2) * 4
	total += ropeTableBytes
	if cfg.RopeFreqBaseSWA > 0 && cfg.RopeFreqBaseSWA != cfg.RopeFreqBase {
		total += ropeTableBytes // second set for SWA layers
	}

	// q8_1 scratch buffer for dp4a path
	maxDim := dim
	if ffnDim > maxDim {
		maxDim = ffnDim
	}
	q8_1Blocks := (maxDim + 31) / 32
	total += q8_1Blocks * 36

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
	for _, nice := range []int{1048576, 524288, 262144, 131072, 65536, 32768, 16384, 8192} {
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
	kvPerLayer := 2 * int64(maxSeqLen) * kvDim * 2 // K + V buffers (FP16, matches runtime allocation)

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

	// No VRAM ceiling — the C-level gpu_alloc uses a llama.cpp-style fallback
	// chain: try device-local first, overflow to host-visible (system RAM
	// accessible by GPU shaders via PCIe). All layers stay on the GPU compute
	// path regardless of which memory type they land in.

	var err error
	// TokenEmbed/Output are projection-class weights: route through the
	// native-or-Affine8 uploader so an exotic scheme never falls back to CPU.
	gm.TokenEmbed, err = uploadProjWeight(m.TokenEmbed)
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

	gm.Output, err = uploadProjWeight(m.Output)
	if err != nil {
		return nil, fmt.Errorf("upload output: %w", err)
	}

	if m.OutputBias != nil {
		gm.OutputBias, err = UploadF32Slice(m.OutputBias)
		if err != nil {
			return nil, fmt.Errorf("upload output_bias: %w", err)
		}
	}

	// Which MoE layers still need their packed expert matrices uploaded. Those
	// are deferred to a second pass, after every hot tensor has been placed.
	moeExpertsPending := make([]bool, len(m.Layers))

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
			gl.Wq, gl.WqA8, err = uploadWeightPair(cl.Wq)
			if err != nil {
				return nil, fmt.Errorf("layer %d wq: %w", l, err)
			}
		}
		if cl.Wk != nil {
			gl.Wk, gl.WkA8, err = uploadWeightPair(cl.Wk)
			if err != nil {
				return nil, fmt.Errorf("layer %d wk: %w", l, err)
			}
		}
		if cl.Wv != nil {
			gl.Wv, gl.WvA8, err = uploadWeightPair(cl.Wv)
			if err != nil {
				return nil, fmt.Errorf("layer %d wv: %w", l, err)
			}
		}
		if cl.Wo != nil {
			gl.Wo, gl.WoA8, err = uploadWeightPair(cl.Wo)
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
			if cl.FFNExpertSelBias != nil {
				gl.FFNExpertSelBias, _ = UploadF32Slice(cl.FFNExpertSelBias)
			}
			// The packed expert matrices are deferred to the second pass below so
			// that every tensor read in full on each token — the attention and SSM
			// projections above, the norms, the router and the shared expert —
			// claims device-local memory first. moeUploaded so far only reflects
			// the router upload; the expert pass sets the final MoEOnGPU.
			// The shared expert is hot: it runs for every token, so it stays here.
			if cl.FFNGateShared != nil {
				gl.FFNGateShared, _ = uploadProjWeight(cl.FFNGateShared)
			}
			if cl.FFNUpShared != nil {
				gl.FFNUpShared, _ = uploadProjWeight(cl.FFNUpShared)
			}
			if cl.FFNDownShared != nil {
				gl.FFNDownShared, _ = uploadProjWeight(cl.FFNDownShared)
			}
			if cl.FFNRouterShared != nil {
				gl.FFNRouterShared, _ = UploadF32Slice(cl.FFNRouterShared)
			}
			moeExpertsPending[l] = moeUploaded
		} else {
			gl.FFNGate, gl.GateA8, _ = uploadWeightPair(cl.FFNGate)
			if cl.FFNUp != nil {
				gl.FFNUp, gl.UpA8, err = uploadWeightPair(cl.FFNUp)
				if err != nil {
					return nil, fmt.Errorf("layer %d ffn_up: %w", l, err)
				}
			}
			if cl.FFNDown != nil {
				gl.FFNDown, gl.DownA8, err = uploadWeightPair(cl.FFNDown)
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
			gl.SSMInProj, gl.SSMInProjA8, _ = uploadWeightPair(cl.SSMInProj)
		}
		if cl.AttnGate != nil {
			gl.SSMGate, gl.SSMGateA8, _ = uploadWeightPair(cl.AttnGate)
		}
		if cl.SSMAlpha != nil {
			gl.SSMAlpha, gl.SSMAlphaA8, _ = uploadWeightPair(cl.SSMAlpha)
		}
		if cl.SSMBeta != nil {
			gl.SSMBeta, gl.SSMBetaA8, _ = uploadWeightPair(cl.SSMBeta)
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
			gl.SSMOut, gl.SSMOutA8, _ = uploadWeightPair(cl.SSMOut)
		}

		// LFM2/LFM2MoE causal short convolution: in_proj (dim -> 3*dim),
		// out_proj (dim -> dim), the depthwise taps, and the rolling b*x
		// history that the conv shader advances in place.
		if cl.ShortConvInProj != nil {
			gl.ShortConvInProj, _, err = uploadWeightPair(cl.ShortConvInProj)
			if err != nil {
				return nil, fmt.Errorf("layer %d shortconv in_proj: %w", l, err)
			}
		}
		if cl.ShortConvOutProj != nil {
			gl.ShortConvOutProj, _, err = uploadWeightPair(cl.ShortConvOutProj)
			if err != nil {
				return nil, fmt.Errorf("layer %d shortconv out_proj: %w", l, err)
			}
		}
		if len(cl.ShortConvConv) > 0 {
			gl.ShortConvConv, err = UploadF32Slice(cl.ShortConvConv)
			if err != nil {
				return nil, fmt.Errorf("layer %d shortconv conv: %w", l, err)
			}
		}
		if cl.Spec.Core == llm.CoreShortConv {
			lcache := m.Config.ShortConvLCache
			if lcache < 1 {
				lcache = 3
			}
			stateLen := (lcache - 1) * m.Config.EmbeddingDim
			if stateLen > 0 {
				gl.ShortConvState = Alloc(uint64(stateLen * 4))
				if gl.ShortConvState == 0 {
					return nil, fmt.Errorf("layer %d shortconv state alloc (%d floats)", l, stateLen)
				}
				ZeroFill(gl.ShortConvState, uint64(stateLen*4))
			}
		}

	}

	// Second pass: the packed expert matrices, uploaded only after every hot
	// tensor above has claimed device-local memory.
	//
	// gpu_alloc is first-come-first-served — it prefers device-local and falls
	// back to host-visible once the ceiling is reached — so uploading strictly
	// layer-by-layer meant that once the weights overflowed VRAM the LAST layers
	// went to system RAM whole: their attention and SSM projections, norms and
	// shared expert along with their experts. That is exactly backwards for an
	// MoE. Those hot tensors are read in full on every token, while only
	// ExpertUsedCount of ExpertCount experts are; on Qwen3.6-35B-A3B it put ~18
	// of 40 layers entirely in host memory, making the always-read part of the
	// spilled set cost more per token than the sparse part it was supposed to
	// trade against.
	//
	// Uploading the sparse part last puts the overflow where an MoE can afford
	// it: on expert matrices, of which a token touches 8 of 256.
	for l := 0; l < maxLayers; l++ {
		cl := &m.Layers[l]
		gl := &gm.Layers[l]
		if !gl.IsMoE || !moeExpertsPending[l] {
			continue
		}
		moeUploaded := true
		if cl.FFNGateUpExps != nil && (supportsGPUQType(cl.FFNGateUpExps.Type) || llmUnifiedQuant()) {
			gl.FFNGateUpExps, err = uploadProjWeight(cl.FFNGateUpExps)
			if err != nil {
				moeUploaded = false
			}
		} else if cl.FFNGateExps != nil && (supportsGPUQType(cl.FFNGateExps.Type) || llmUnifiedQuant()) {
			gl.FFNGateExps, err = uploadProjWeight(cl.FFNGateExps)
			if err != nil {
				moeUploaded = false
			}
			if moeUploaded && cl.FFNUpExps != nil {
				gl.FFNUpExps, err = uploadProjWeight(cl.FFNUpExps)
				if err != nil {
					moeUploaded = false
				}
			}
		} else {
			moeUploaded = false
		}
		if moeUploaded && cl.FFNDownExps != nil {
			gl.FFNDownExps, err = uploadProjWeight(cl.FFNDownExps)
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

	// LFM2/LFM2MoE hybrid short-convolution layers run on the non-fused forward
	// path below: the fused per-layer C dispatcher has no short-conv core, and
	// the short-conv step is a sequential scan over the token axis that does not
	// compose with the fused norm/attention/residual pipeline. Everything else
	// (projections, MoE experts, attention) still runs on the GPU.
	if cfg.HasShortConv() {
		fmt.Printf("[dlgo/gpu] %s: %d short-convolution layers run on the non-fused GPU path\n",
			cfg.Architecture, cfg.NumLayers-cfg.AttentionLayerCount())
	}

	totalVRAM := float64(VRAMBytes()) / (1024 * 1024)
	freeVRAM := float64(effectiveFreeVRAM()) / (1024 * 1024)
	fmt.Printf("[dlgo/gpu] Uploading model to %s (%.0f MB total, %.0f MB usable)...\n",
		DeviceName(), totalVRAM, freeVRAM)

	// GPU-aware context capping.
	//
	// Two strategies:
	//
	//   spill (default): weights are allowed to overflow VRAM into system RAM
	//     via the allocator's host-visible fallback, so the context is bounded
	//     by (VRAM + host budget) rather than by VRAM alone. This is what lets
	//     a model whose weights exceed VRAM still run at a useful context.
	//
	//   no-spill (DLGO_GPU_SPILL=0): the historical rule — every layer's weights
	//     plus its KV must fit in device-local VRAM, otherwise shrink until
	//     they do. Kept as a fallback for debugging and for machines where
	//     pinning multi-GB host-visible buffers is undesirable.
	//
	// Release the CPU-side KV cache / run state / batch state BEFORE planning,
	// not just before the upload. Both planners below read the remaining Windows
	// commit charge, and the CPU scratch is a first-class consumer of it: its KV
	// mirror is context-sized, so at a large context it can consume more commit
	// than the host weight tier needs. Planning first then measured a host tier
	// of zero and clamped the context to the 8K floor while 20 GB of physical RAM
	// sat free — the model was told it could not spill for lack of the very
	// memory it was about to release. The CPU buffers are rebuilt below if GPU
	// creation ultimately fails and the caller falls back to CPU.
	//
	// The RoPE tables are Go slices computed by the CPU run state, and they carry
	// the model's rope scaling (YaRN tables are not what RoPEFrequencyTable
	// produces from the config alone), so hold them across the release instead of
	// recomputing them.
	var cpuCos, cpuSin, cpuCosSWA, cpuSinSWA []float32
	if cpuPipeline.RunState != nil {
		cpuCos, cpuSin = cpuPipeline.RunState.RoPETables()
		if cfg.RopeFreqBaseSWA > 0 && cfg.RopeFreqBaseSWA != cfg.RopeFreqBase {
			cpuCosSWA, cpuSinSWA = cpuPipeline.RunState.RoPETablesSWA()
		}
	}
	cpuPipeline.FreeForGPU()
	runtime.GC()
	debug.FreeOSMemory()
	mmap.TrimWorkingSet()

	layerBytes, nonLayerBytes := modelWeightBytes(m)
	modelBytes := layerBytes + nonLayerBytes

	maxSeqLen := cpuPipeline.MaxSeqLen
	if SpillEnabled() {
		// Spilling is strictly additive: the streaming budget may allow a
		// larger context than the VRAM-only rule, but must never shrink one
		// below what that rule would have allowed — the VRAM-only rule is what
		// protects WDDM from freezing, and it is the conservative bound.
		streamed := computeStreamingContext(m, maxSeqLen)
		vramOnly := computeMaxGPUContext(m, maxSeqLen)
		planned := vramOnly
		if streamed > planned {
			planned = streamed
		}
		if planned <= 0 {
			planned = 4096
		}
		if planned < maxSeqLen {
			fmt.Printf("[dlgo/gpu] Context %d -> %d tokens (weights %.0f MB; VRAM %.0f MB + RAM %.0f MB)\n",
				maxSeqLen, planned, float64(modelBytes)/mib,
				float64(vramSpillBudget())/mib, float64(hostSpillBudget())/mib)
			maxSeqLen = planned
			cpuPipeline.MaxSeqLen = maxSeqLen
		} else {
			fmt.Printf("[dlgo/gpu] Context %d tokens fits (weights %.0f MB; VRAM %.0f MB + RAM %.0f MB)\n",
				maxSeqLen, float64(modelBytes)/mib,
				float64(vramSpillBudget())/mib, float64(hostSpillBudget())/mib)
		}
	} else {
		gpuSafeCtx := computeMaxGPUContext(m, maxSeqLen)
		if gpuSafeCtx < maxSeqLen {
			fmt.Printf("[dlgo/gpu] Reducing context from %d to %d tokens to fit all layers in VRAM (%.0f MB)\n",
				maxSeqLen, gpuSafeCtx, freeVRAM)
			maxSeqLen = gpuSafeCtx
			cpuPipeline.MaxSeqLen = maxSeqLen
		}
	}

	// Default: upload ALL layers. The C-level gpu_alloc fallback chain handles
	// VRAM limits (device-local → host-visible). All layers stay on GPU compute.
	// DLGO_GPU_LAYERS can override for testing.
	numGPULayers := cfg.NumLayers
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

	// The CPU-side KV cache / run state / batch state were released above, before
	// context planning, because their commit charge feeds the host spill budget.

	// Retry loop: if GPU allocation fails, reduce layers and retry.
	// Limited to 3 retries to avoid VRAM fragmentation from repeated
	// partial allocations. Each retry does a full Sync + buffer table
	// reset to guarantee VRAM is actually reclaimed.
	const maxRetries = 3
	var gm *GpuModel
	var rs *GpuRunState
	var kv *GpuKVCache
	var ropeCosTable, ropeSinTable Buf
	var ropeCosTableSWA, ropeSinTableSWA Buf
	var cpuRoPECos, cpuRoPESin []float32
	var layerConfs []*LayerConf
	var q8_1Scratch Buf
	var isPartial bool

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if numGPULayers <= 0 {
			cpuPipeline.RebuildBuffers()
			return nil, fmt.Errorf("insufficient VRAM (%.0f MB) for even 1 layer — use CPU mode", totalVRAM)
		}
		isPartial = numGPULayers < cfg.NumLayers

		allocErr := func() error {
			var err error

			// Allocate the KV cache BEFORE the weights.
			//
			// The allocator fills device-local memory first-come-first-served and
			// falls back to host-visible once it is exhausted, so uploading the
			// weights first lets the cold MoE expert matrices claim every byte of
			// VRAM and leaves the KV cache in system RAM. That is backwards: the
			// KV cache is read in full on every decode step and grows with the
			// context, while only ExpertUsedCount of ExpertCount experts are
			// touched per token. Measured on Qwen3.6-35B-A3B at ctx 100864 the
			// 2 GB KV cache had been placed in host memory, i.e. a 2 GB PCIe read
			// per decode step; spilling the same 2 GB of experts instead costs
			// about 2 GB * (8/256) of expert bytes = 60 MB per token.
			//
			// needsKV and the KV size depend only on the model config, so this
			// does not need the uploaded model.
			needsKV := make([]bool, cfg.NumLayers)
			for l := 0; l < cfg.NumLayers; l++ {
				needsKV[l] = layerNeedsKV(&m.Layers[l])
			}
			kv, err = NewGpuKVCache(cfg.NumLayers, numGPULayers, maxSeqLen, kvDim, needsKV)
			if err != nil {
				return err
			}

			gm, err = UploadModel(m, numGPULayers)
			if err != nil {
				return fmt.Errorf("upload model: %w", err)
			}

			// With the fallback chain, all requested layers should be uploaded.
			// If UploadModel reports fewer, update our count.
			if gm.NumGPULayers < numGPULayers {
				// The KV cache was sized for the larger count; drop the tail so
				// the VRAM it holds can still be used by the weights.
				for l := gm.NumGPULayers; l < numGPULayers; l++ {
					freeBuf(kv.KeyBufs[l])
					freeBuf(kv.ValBufs[l])
					kv.KeyBufs[l], kv.ValBufs[l] = 0, 0
				}
				numGPULayers = gm.NumGPULayers
				isPartial = numGPULayers < cfg.NumLayers
			}

			// Report the VRAM/RAM residency split for the weights just placed.
			if attempt == 0 {
				residencyReport(modelBytes)
			}

			rs, err = NewGpuRunState(dim, qDim, kvDim, ffnDim, cfg.VocabSize)
			if err != nil {
				return err
			}

			// Upload RoPE tables. The CPU tables captured before the release above
			// carry the model's rope scaling; only fall back to config-derived
			// tables when the CPU pipeline never had a run state.
			var cosTable, sinTable []float32
			if cpuCos != nil && cpuSin != nil {
				cosTable, sinTable = cpuCos, cpuSin
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
				cosTable, sinTable = cos, sin
			}
			cpuRoPECos, cpuRoPESin = cosTable, sinTable

			// Upload SWA RoPE tables if per-layer RoPE is needed
			if cfg.RopeFreqBaseSWA > 0 && cfg.RopeFreqBaseSWA != cfg.RopeFreqBase {
				var swaCos, swaSin []float32
				if cpuCosSWA != nil && cpuSinSWA != nil {
					swaCos, swaSin = cpuCosSWA, cpuSinSWA
				}
				if swaCos != nil && swaSin != nil {
					ropeCosTableSWA, err = UploadF32Slice(swaCos)
					if err != nil {
						return fmt.Errorf("upload SWA RoPE cos table: %w", err)
					}
					ropeSinTableSWA, err = UploadF32Slice(swaSin)
					if err != nil {
						return fmt.Errorf("upload SWA RoPE sin table: %w", err)
					}
				} else {
					ropeDimSWA := cfg.RopeDim
					if ropeDimSWA <= 0 || ropeDimSWA > cfg.HeadDim {
						ropeDimSWA = cfg.HeadDim
					}
					swaCos, swaSin := ops.RoPEFrequencyTable(maxSeqLen, ropeDimSWA, cfg.RopeFreqBaseSWA)
					ropeCosTableSWA, err = UploadF32Slice(swaCos)
					if err != nil {
						return fmt.Errorf("upload SWA RoPE cos table: %w", err)
					}
					ropeSinTableSWA, err = UploadF32Slice(swaSin)
					if err != nil {
						return fmt.Errorf("upload SWA RoPE sin table: %w", err)
					}
				}
			}

			tempPipe := &GpuPipeline{
				RoPECosTable:    ropeCosTable,
				RoPESinTable:    ropeSinTable,
				RoPECosTableSWA: ropeCosTableSWA,
				RoPESinTableSWA: ropeSinTableSWA,
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

			// Short-convolution scratch (LFM2/LFM2MoE). Per-layer conv state is
			// allocated with the weights in UploadModel.
			if cfg.HasShortConv() && cfg.AttentionLayerCount() < cfg.NumLayers {
				if err := rs.AllocShortConvScratch(dim); err != nil {
					return err
				}
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
		freeBuf(ropeCosTableSWA)
		freeBuf(ropeSinTableSWA)
		freeBuf(q8_1Scratch)
		Sync()
		ResetBufferTable()
		ropeCosTable, ropeSinTable, q8_1Scratch = 0, 0, 0
		ropeCosTableSWA, ropeSinTableSWA = 0, 0
		gm, rs, kv = nil, nil, nil
		layerConfs = nil

		if attempt >= maxRetries {
			cpuPipeline.RebuildBuffers()
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

	// Memory safety: cap working set and set Go heap limit to prevent
	// the process from consuming all system RAM and freezing Windows.
	EnforceWorkingSetLimit(uint64(AllocatedBytes()))
	InitMemorySafety()

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

	// Report memory split like llama.cpp does
	allocMB := float64(AllocatedBytes()) / (1024 * 1024)
	hvMB := float64(HostVisibleBytes()) / (1024 * 1024)
	vramMB := allocMB - hvMB
	if isPartial {
		fmt.Printf("[dlgo/gpu] Partial GPU: %d/%d layers on GPU, %d on CPU\n",
			numGPULayers, cfg.NumLayers, cfg.NumLayers-numGPULayers)
	} else {
		fmt.Printf("[dlgo/gpu] All %d layers on GPU\n", cfg.NumLayers)
	}
	fmt.Printf("[dlgo/gpu] Memory: %.0f MB VRAM + %.0f MB host-visible = %.0f MB total (FP16 KV cache)\n",
		vramMB, hvMB, allocMB)

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
		RoPECosTableSWA: ropeCosTableSWA,
		RoPESinTableSWA: ropeSinTableSWA,
		CPUPipeline:     cpuPipeline,
	}

	// Keep a CPU copy of the standard RoPE tables for M-RoPE slot patching.
	pipe.cpuRoPECos, pipe.cpuRoPESin = cpuRoPECos, cpuRoPESin

	// Qwen M-RoPE: enable CPU-side row computation + table patching when the
	// model declares rope dimension sections (Qwen3-VL / Qwen3.5 IMROPE).
	if len(cfg.RopeDimensionSections) >= 3 {
		for i := 0; i < 3 && i < len(cfg.RopeDimensionSections); i++ {
			pipe.mropeSections[i] = cfg.RopeDimensionSections[i]
		}
		if len(cfg.RopeDimensionSections) > 3 {
			pipe.mropeSections[3] = cfg.RopeDimensionSections[3]
		}
		pipe.mropeFreq = cfg.RopeFreqBase
		pipe.mropeDim = cfg.RopeDim
		if pipe.mropeDim <= 0 || pipe.mropeDim > cfg.HeadDim {
			pipe.mropeDim = cfg.HeadDim
		}
		pipe.mropeEnabled = true
	}

	// Use fused forward when ALL layers are on GPU (including MoE models).
	// For MoE: C side handles attention + residual + norm, returns early (ffn_type=3);
	// Go side then handles MoE FFN dispatch.
	// Short-convolution models must use the non-fused path: the fused dispatcher
	// has no short-conv core.
	if !isPartial && !cfg.HasShortConv() {
		pipe.UseFusedForward = supportsFusedForwardGPU(m)
	}
	if os.Getenv("DLGO_NO_FUSED") == "1" {
		pipe.UseFusedForward = false
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
	pipe.hasCPUAttn = hasCPUAttn

	// Allocate CPU-side state if needed (partial GPU, MoE, MLA, CPU attn, or hybrid SSM).
	needCPUState := isPartial || cfg.ExpertCount > 0 || pipe.HasMLA || hasCPUAttn || (pipe.HasSSM && (cpuSSMDelta || cpuSSMFull))
	if needCPUState {
		pipe.CPURunState = llm.NewRunState(cfg, maxSeqLen)

		if isPartial {
			pipe.CPUKVCache = memory.NewMultiLayerKVCache(cfg.NumLayers, maxSeqLen, kvDim)
			pipe.CPUBatchState = llm.NewBatchState(cfg, maxSeqLen)
		}

		if (pipe.HasMLA || hasCPUAttn) && pipe.CPUKVCache == nil {
			pipe.CPUKVCache = memory.NewMultiLayerKVCache(cfg.NumLayers, maxSeqLen, kvDim)
			if hasCPUAttn {
				fmt.Printf("[dlgo/gpu] CPU attention fallback: allocated CPU KV cache (%d layers)\n", cfg.NumLayers)
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

	// If any layers are on CPU (rare with fallback chain), prefetch their mmap
	// pages so the OS page cache is warm. Unlike pinning (removed), this does
	// NOT copy data to heap — zero commit charge overhead.
	if isPartial && m.MmapFile != nil {
		fmt.Printf("[dlgo/gpu] Prefetching %d CPU layer mmap regions...\n", cfg.NumLayers-numGPULayers)
		mmap.PrefetchRegion(m.MmapFile.Data)
	}

	return pipe, nil
}

// NewRunner is the one-call, memory-safe way to get an inference backend.
//
// It loads the model, initializes Vulkan, and builds the GPU pipeline. Weight
// upload releases the CPU-side scratch first (so a large context can never
// double peak host memory / exhaust the Windows commit limit), and if GPU
// creation fails the CPU buffers are rebuilt so the returned Runner is always
// usable. A missing/unusable GPU degrades to CPU rather than erroring.
//
//	runner, backend, err := gpu.NewRunner("model.gguf", 32768)
//	if err != nil { log.Fatal(err) }
//	res, err := runner.GenerateDetailed(prompt, cfg)
//
// backend is "GPU (Vulkan)" or "CPU". err is non-nil only for hard failures
// (e.g. the model file could not be loaded).
func NewRunner(modelPath string, ctx int) (llm.Runner, string, error) {
	pipe, err := llm.NewPipeline(modelPath, ctx, llm.WithoutCPUState())
	if err != nil {
		return nil, "", err
	}
	if err := Init(); err != nil {
		return pipe, "CPU", nil
	}
	gp, err := NewGpuPipeline(pipe)
	if err != nil {
		// NewGpuPipeline rebuilt the CPU buffers on failure.
		return pipe, "CPU", nil
	}
	return gp, "GPU (" + DeviceName() + ")", nil
}

// invalidateChatPrefix drops any cached cross-request chat prefix snapshot and
// forgets the remembered prompt tokens.
func (p *GpuPipeline) invalidateChatPrefix() {
	if p.chatPrefixSnap != nil {
		p.chatPrefixSnap.Free()
		p.chatPrefixSnap = nil
	}
	p.chatPrefixTokens = nil
	p.chatPrefixValid = false
}

// chatPrefixEligible reports whether cross-request prefix reuse can be used
// safely for this pipeline: SSM must run on the GPU (not the CPU delta/full
// fallbacks), no layer may offload attention to the CPU, and prefill must take
// the hybrid batched path (which accepts an arbitrary start position).
func (p *GpuPipeline) chatPrefixEligible() bool {
	if !prefixReuseEnabled || !p.HasSSM {
		return false
	}
	if cpuSSMDelta || cpuSSMFull || p.IsPartialGPU || p.HasMoE || p.hasCPUAttn {
		return false
	}
	if !p.UseFusedForward || !isHybridSSMModel(p.CPUModel) {
		return false
	}
	return true
}

// tryChatPrefixReuse restores the GPU SSM state to the boundary of the
// previously cached prompt when the new prompt tokens extend it. On success it
// returns the length of the reusable prefix and leaves the resident KV for
// [0, reuseLen) intact so the caller can prefill only tokens[reuseLen:].
func (p *GpuPipeline) tryChatPrefixReuse(tokenSegs [][]int32, images [][][]float32) (int, bool) {
	if !p.chatPrefixEligible() || len(images) != 0 || len(tokenSegs) != 1 {
		return 0, false
	}
	if !p.chatPrefixValid || p.chatPrefixSnap == nil {
		return 0, false
	}
	tokens := tokenSegs[0]
	reuseLen := len(p.chatPrefixTokens)
	// Require the new prompt to strictly extend the cached prefix so there is
	// always fresh suffix to prefill (and so a repeat of the exact same prompt
	// does a clean full prefill instead of a zero-token continuation).
	if reuseLen == 0 || reuseLen >= len(tokens) || reuseLen >= p.MaxSeqLen {
		return 0, false
	}
	for i := 0; i < reuseLen; i++ {
		if tokens[i] != p.chatPrefixTokens[i] {
			return 0, false
		}
	}
	// Restore only the GPU SSM/conv recurrent state and rewind KVCache.Len to the
	// boundary. The KV rows for [0, reuseLen) are still resident from the prior
	// prefill (generation only appended beyond reuseLen), so they need no copy.
	if err := p.restoreState(p.chatPrefixSnap, false); err != nil {
		return 0, false
	}
	return reuseLen, true
}

// storeChatPrefix captures the GPU SSM/conv recurrent state at the prompt
// boundary (npos) so the next request can resume from it. The KV rows remain in
// the live cache; only the recurrent state is copied.
func (p *GpuPipeline) storeChatPrefix(tokens []int32, npos int) {
	if !p.chatPrefixEligible() {
		return
	}
	snap, err := p.snapshotRecurrentState(npos)
	if err != nil || snap == nil {
		p.invalidateChatPrefix()
		return
	}
	if p.chatPrefixSnap != nil {
		p.chatPrefixSnap.Free()
	}
	p.chatPrefixSnap = snap
	p.chatPrefixTokens = append([]int32(nil), tokens...)
	p.chatPrefixValid = true
}

// generationPrimerTokenLen returns the number of trailing tokens that make up
// the chat template's generation primer appended after the assistant marker
// (e.g. the empty thinking block). On the next turn the assistant's real
// content occupies this position, so the reusable prefix must stop just before
// these tokens. Returns 0 if no known primer is present.
func (p *GpuPipeline) generationPrimerTokenLen(prompt string, tokens []int32) int {
	for _, suf := range []string{"<think></think>\n", "<think>\n"} {
		if !strings.HasSuffix(prompt, suf) {
			continue
		}
		st := p.Tokenizer.EncodeNoBOS(suf)
		if len(st) == 0 || len(st) >= len(tokens) {
			return 0
		}
		off := len(tokens) - len(st)
		match := true
		for i := range st {
			if tokens[off+i] != st[i] {
				match = false
				break
			}
		}
		if match {
			return len(st)
		}
	}
	return 0
}

// ResetState zeros all caches and SSM state for a fresh inference.
func (p *GpuPipeline) ResetState() {
	p.invalidateChatPrefix()
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

	// Zero the LFM2/LFM2MoE short-convolution rolling history so a new request
	// never inherits the previous one's conv state (the CPU path's equivalent
	// bug produced cross-request contamination that looked like a modelling
	// error).
	mcfg := p.CPUModel.Config
	if mcfg.HasShortConv() {
		for l := 0; l < mcfg.NumLayers; l++ {
			gl := &p.GpuModel.Layers[l]
			if gl.ShortConvState == 0 {
				continue
			}
			lcache := mcfg.ShortConvLCache
			if lcache < 1 {
				lcache = 3
			}
			n := (lcache - 1) * mcfg.EmbeddingDim
			if n > 0 {
				ZeroFill(gl.ShortConvState, uint64(n*4))
			}
		}
	}
}

// FreeAll releases all GPU resources held by this pipeline.
// Sync is called first to ensure no in-flight commands reference these buffers,
// which allows the Vulkan driver to immediately reclaim the device memory.
func (p *GpuPipeline) FreeAll() {
	if p == nil {
		return
	}
	StopMemoryMonitor()
	Sync()
	p.invalidateChatPrefix()
	p.freePrefixCache()
	p.GpuModel.FreeAll()
	p.RunState.FreeAll()
	p.KVCache.FreeAll()
	p.BatchState.Free()
	freeBuf(p.Q8_1Scratch)
	freeBuf(p.RoPECosTable)
	freeBuf(p.RoPESinTable)
	freeBuf(p.RoPECosTableSWA)
	freeBuf(p.RoPESinTableSWA)
	ResetBufferTable()
}

// GenerateResult is the shared generation result. It is an alias (not a new
// type) of llm.GenerateResult so the GPU and CPU pipelines are interchangeable
// behind llm.Runner and no caller needs to copy fields between two structs.
type GenerateResult = llm.GenerateResult

// GenerateDetailed runs generation on GPU with detailed timing.
func (p *GpuPipeline) GenerateDetailed(prompt string, cfg llm.GenerateConfig) (*GenerateResult, error) {
	tokens := p.Tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenizer produced no tokens")
	}
	if len(tokens) >= p.MaxSeqLen {
		return nil, fmt.Errorf("prompt too long: %d tokens (max %d)", len(tokens), p.MaxSeqLen)
	}
	return p.generateSegments(prompt, [][]int32{tokens}, nil, cfg)
}

// GenerateDetailedWithEmbeddings is like GenerateDetailed but replaces each
// occurrence of the image marker (cfg.ImageMarker, default "<image>") in the
// prompt with a block of precomputed embedding rows (e.g. from a vision
// encoder). Text segments are prefilled with the batched GPU path; image rows
// go through the single-token embedding forward (GpuForwardEmbedding), so KV
// and SSM state advance exactly as if the rows were tokens.
func (p *GpuPipeline) GenerateDetailedWithEmbeddings(prompt string, images [][][]float32, cfg llm.GenerateConfig) (*GenerateResult, error) {
	if len(images) == 0 {
		return p.GenerateDetailed(prompt, cfg)
	}
	marker := cfg.ImageMarker
	if marker == "" {
		marker = "<image>"
	}
	segments := strings.Split(prompt, marker)
	if len(segments) != len(images)+1 {
		return nil, fmt.Errorf("image placeholder count mismatch: prompt has %d markers, %d images provided",
			len(segments)-1, len(images))
	}
	dim := p.CPUModel.Config.EmbeddingDim
	for i, blk := range images {
		if len(blk) == 0 {
			return nil, fmt.Errorf("image %d has no embedding rows", i)
		}
		for r, row := range blk {
			if len(row) != dim {
				return nil, fmt.Errorf("image %d row %d has %d dims, want %d", i, r, len(row), dim)
			}
		}
	}
	tokenSegs := make([][]int32, len(segments))
	for i, s := range segments {
		if i == 0 {
			tokenSegs[i] = p.Tokenizer.Encode(s)
		} else {
			tokenSegs[i] = p.Tokenizer.EncodeNoBOS(s)
		}
	}
	return p.generateSegments(prompt, tokenSegs, images, cfg)
}

// imropeRow computes the cos/sin row for one head under Qwen interleaved
// M-RoPE, mirroring ggml compute_imrope_cache. Returns rows of half = dim/2.
func (p *GpuPipeline) imropeRow(pt, ph, pw, pe int) (cos, sin []float32) {
	half := p.mropeDim / 2
	cos = make([]float32, half)
	sin = make([]float32, half)
	thetaScale := math.Pow(float64(p.mropeFreq), -2.0/float64(p.mropeDim))
	thetaT := float64(pt)
	thetaH := float64(ph)
	thetaW := float64(pw)
	thetaE := float64(pe)
	sec := p.mropeSections
	sectDims := sec[0] + sec[1] + sec[2] + sec[3]
	for i := 0; i < half; i++ {
		sector := i % sectDims
		var theta float64
		switch {
		case sector%3 == 1 && sector < 3*sec[1]:
			theta = thetaH
		case sector%3 == 2 && sector < 3*sec[2]:
			theta = thetaW
		case sector%3 == 0 && sector < 3*sec[0]:
			theta = thetaT
		default:
			theta = thetaE
		}
		cos[i] = float32(math.Cos(theta))
		sin[i] = float32(math.Sin(theta))
		thetaT *= thetaScale
		thetaH *= thetaScale
		thetaW *= thetaScale
		thetaE *= thetaScale
	}
	return cos, sin
}

// patchMRoPERow overwrites the RoPE cos/sin table row at slot with the M-RoPE
// row for positions (pt,ph,pw,pe). The kernel indexes cos[slot*half + i].
func (p *GpuPipeline) patchMRoPERow(slot, pt, ph, pw, pe int) error {
	if !p.mropeEnabled {
		return nil
	}
	cos, sin := p.imropeRow(pt, ph, pw, pe)
	half := p.mropeDim / 2
	if err := UploadF32At(p.RoPECosTable, slot*half, cos); err != nil {
		return err
	}
	return UploadF32At(p.RoPESinTable, slot*half, sin)
}

// patchScalarRow copies the standard RoPE table row at logicalPos into the slot
// row. Used for text tokens after an image, whose logical position differs from
// their KV slot.
func (p *GpuPipeline) patchScalarRow(slot, logicalPos int) error {
	if !p.mropeEnabled || slot == logicalPos {
		return nil
	}
	half := p.mropeDim / 2
	cosSrc, sinSrc := p.cpuRoPECos, p.cpuRoPESin
	if cosSrc == nil || logicalPos < 0 || (logicalPos+1)*half > len(cosSrc) {
		return nil
	}
	if err := UploadF32At(p.RoPECosTable, slot*half, cosSrc[logicalPos*half:(logicalPos+1)*half]); err != nil {
		return err
	}
	return UploadF32At(p.RoPESinTable, slot*half, sinSrc[logicalPos*half:(logicalPos+1)*half])
}

// generateSegments runs the shared prefill + decode loop. tokenSegs[i] is the
// text tokenized for segment i; images[i] (if non-nil) is spliced after it.
func (p *GpuPipeline) generateSegments(prompt string, tokenSegs [][]int32, images [][][]float32, cfg llm.GenerateConfig) (*GenerateResult, error) {
	rng := rand.New(rand.NewSource(cfg.Seed))
	if cfg.Seed < 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	npos := 0
	for i, seg := range tokenSegs {
		npos += len(seg)
		if i < len(images) {
			npos += len(images[i])
		}
	}
	if npos >= p.MaxSeqLen {
		return nil, fmt.Errorf("prompt too long: %d tokens (max %d)", npos, p.MaxSeqLen)
	}

	// Cross-request prefix reuse (agentic chat): when this prompt extends the
	// previously cached one, rewind the GPU SSM state to the cached boundary and
	// prefill only the new suffix. Falls back to a clean full prefill otherwise.
	prefillFrom := 0
	if reuseLen, ok := p.tryChatPrefixReuse(tokenSegs, images); ok {
		prefillFrom = reuseLen
	} else {
		p.ResetState()
	}

	// Chunked prefill: cap batch buffer size to avoid large VRAM allocation
	// Chunked prefill size. The batch buffers scale with it, so it is capped to
	// avoid a VRAM spike during inference (a 9B model at 4096 tokens × ffnDim can
	// allocate 1+ GB of transient buffers).
	//
	// Within that cap, bigger is faster here: each chunk re-reads the whole expert
	// tensor once (past a handful of tokens every expert is active), so a larger
	// chunk amortises that read over more tokens. Measured on LFM2.5-24B-A2B at
	// 1923 prompt tokens: 128 -> 80 tok/s, 512 -> 136, 1024 -> 155, one chunk ->
	// 178, against 86 for the per-token path. DLGO_PREFILL_CHUNK overrides it.
	prefillChunkSize := 1024
	if v := os.Getenv("DLGO_PREFILL_CHUNK"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			prefillChunkSize = n
		}
	}
	batchSize := npos
	if batchSize > prefillChunkSize {
		batchSize = prefillChunkSize
	}

	if err := p.ensurePrefillBatchState(batchSize); err != nil {
		return nil, err
	}

	prefillStart := time.Now()
	pos := 0
	slot := 0
	if len(images) == 0 {
		// No image splice: use the original fast batched prefill path verbatim.
		tokens := tokenSegs[0]
		if p.IsPartialGPU {
			for i, tok := range tokens {
				GpuForwardPartial(p.CPUModel, p.GpuModel, tok, i, p.KVCache, p.RunState, p.LogitsBuf, p.LayerConfs, p)
			}
		} else if p.UseBatchPrefill {
			// Hybrid short-conv + MoE: run the prompt in chunks with the whole
			// token axis batched. This is the only path that can batch these
			// models — the fused C dispatcher has no short-conv core, and the
			// hybrid batch path runs MoE per position on the CPU.
			for cs := prefillFrom; cs < npos; cs += prefillChunkSize {
				ce := cs + prefillChunkSize
				if ce > npos {
					ce = npos
				}
				if err := GpuForwardPrefillBatchShortConv(p.CPUModel, p.GpuModel, tokens[cs:ce],
					p.KVCache, p.RunState, p.BatchState, p.LogitsBuf, p, cs, ce >= npos); err != nil {
					return nil, err
				}
			}
		} else if !p.UseFusedForward {
			for i, tok := range tokens {
				GpuForward(p.CPUModel, p.GpuModel, tok, i, p.KVCache, p.RunState, p.LogitsBuf, p)
			}
		} else if p.HasMoE && !p.UseBatchPrefillHybridMoE {
			for i, tok := range tokens {
				GpuForwardFusedSSM(p.CPUModel, p.GpuModel, tok, i, p.KVCache, p.RunState, p.LogitsBuf, p.LayerConfs, p)
			}
		} else if isHybridSSMModel(p.CPUModel) {
			// Split the prompt at the pre-primer boundary so we can snapshot the
			// recurrent state there: the chat template's generation primer (the
			// closing think block) differs on the next turn -- the assistant's
			// real content follows instead -- so it must not be part of the
			// reusable prefix.
			cacheMode := p.chatPrefixEligible() && len(tokenSegs) == 1
			cacheableLen := npos
			if cacheMode {
				if pl := p.generationPrimerTokenLen(prompt, tokens); pl > 0 {
					cacheableLen = npos - pl
				}
			}
			// The cached prefix must always stop at the same pre-primer boundary.
			// Do not derive it from prefillFrom: on a reusing turn prefillFrom is
			// already npos-pl, and any guard that compares the two collapses
			// cacheableLen back to npos, which stores the generation primer as
			// part of the reusable prefix. The next turn then has the assistant's
			// real content at that position instead of the primer, the prefix
			// comparison fails, and the whole prompt is re-prefilled -- i.e. every
			// other agentic turn loses all reuse.
			if cacheableLen < prefillFrom {
				cacheableLen = prefillFrom
			}
			runChunks := func(from, to int) {
				for cs := from; cs < to; cs += prefillChunkSize {
					ce := cs + prefillChunkSize
					if ce > to {
						ce = to
					}
					isLast := ce >= npos
					GpuForwardPrefillBatchHybrid(p.CPUModel, p.GpuModel, tokens[cs:ce], p.KVCache, p.RunState,
						p.BatchState, p.LogitsBuf, p.BatchLayerConfs, p, cs, isLast, nil)
					Sync()
				}
			}
			runChunks(prefillFrom, cacheableLen)
			if cacheMode {
				p.storeChatPrefix(tokens[:cacheableLen], cacheableLen)
			}
			runChunks(cacheableLen, npos)
		} else {
			GpuForwardPrefillBatch(p.CPUModel, p.GpuModel, tokens, p.KVCache, p.RunState,
				p.BatchState, p.LogitsBuf, p.BatchLayerConfs, nil)
		}
		pos = npos
		slot = npos
	} else {
		// Image splice: per-token text forward interleaved with embedding rows.
		// (Batched prefill assumes a contiguous span from position 0, so we use
		// the single-token path across segment boundaries.)
		//
		// slot indexes KV storage (advances by 1 per token); pos is the logical
		// RoPE position. Image tokens occupy len(rows) slots but advance pos by
		// max(ny,nx) (llama M-RoPE). For image tokens the cos/sin row at their
		// slot is patched with the M-RoPE row; post-image text rows are patched
		// with the standard row for their logical position.
		gpuFwd := func(tok int32, slot int) {
			if p.IsPartialGPU {
				GpuForwardPartial(p.CPUModel, p.GpuModel, tok, slot, p.KVCache, p.RunState, p.LogitsBuf, p.LayerConfs, p)
			} else if p.UseFusedForward && p.HasMoE {
				GpuForwardFusedSSM(p.CPUModel, p.GpuModel, tok, slot, p.KVCache, p.RunState, p.LogitsBuf, p.LayerConfs, p)
			} else if p.UseFusedForward && isHybridSSMModel(p.CPUModel) {
				GpuForwardPrefillBatchHybrid(p.CPUModel, p.GpuModel, []int32{tok}, p.KVCache, p.RunState,
					p.BatchState, p.LogitsBuf, p.BatchLayerConfs, p, slot, true, nil)
				Sync()
			} else {
				GpuForward(p.CPUModel, p.GpuModel, tok, slot, p.KVCache, p.RunState, p.LogitsBuf, p)
			}
		}
		feedText := func(seg []int32) error {
			for _, tok := range seg {
				if err := p.patchScalarRow(slot, pos); err != nil {
					return err
				}
				gpuFwd(tok, slot)
				slot++
				pos++
			}
			return nil
		}
		for i, seg := range tokenSegs {
			if err := feedText(seg); err != nil {
				return nil, err
			}
			if i < len(images) {
				var nx, ny int
				useGrid := false
				if i < len(cfg.ImageGrids) {
					ny, nx = cfg.ImageGrids[i][0], cfg.ImageGrids[i][1]
					useGrid = nx > 0 && ny > 0 && nx*ny == len(images[i])
				}
				pos0 := pos
				for r, row := range images[i] {
					if useGrid {
						if err := p.patchMRoPERow(slot, pos0, pos0+r/nx, pos0+r%nx, pos0); err != nil {
							return nil, err
						}
					} else if err := p.patchScalarRow(slot, pos); err != nil {
						return nil, err
					}
					if err := GpuForwardEmbedding(p.CPUModel, p.GpuModel, row, slot, p.KVCache, p.RunState, p.LogitsBuf, p); err != nil {
						return nil, err
					}
					slot++
					pos++
				}
				if useGrid {
					if nx > ny {
						pos = pos0 + nx
					} else {
						pos = pos0 + ny
					}
				}
			}
		}
	}
	// After prefill, slot == npos only when slot==pos; recompute the decode
	// start counters so an image block's slot/pos divergence is preserved.
	Sync()
	prefillMs := float64(time.Since(prefillStart).Microseconds()) / 1000.0

	// Sync GPU SSM state to CPU after prefill so CPU delta rule / CPU SSM
	// paths start from the correct recurrent state.
	// When cpuSSMDelta ran during prefill, CPU state64 is already authoritative;
	// skip GPU→CPU state download. When only cpuSSMFull (no cpuSSMDelta),
	// GPU handled the delta rule during prefill, so download GPU state.
	if p.HasSSM && p.CPURunState != nil && p.CPURunState.SSMState != nil {
		mcfg2 := p.CPUModel.Config
		numHeads := mcfg2.SSMTimeStepRank
		headVDim := mcfg2.SSMInnerSize / numHeads
		headKDim := mcfg2.SSMStateSize
		numKVGroups := mcfg2.SSMGroupCount
		if numKVGroups <= 0 {
			numKVGroups = numHeads
		}
		keyDim := numKVGroups * headKDim
		qkvDim := keyDim*2 + numHeads*headVDim
		convK := mcfg2.SSMConvKernel
		stateSize := numHeads * headKDim * headVDim
		convBufSize := convK * qkvDim
		gpuOwnsState := !cpuSSMDelta
		needSync := false
		for l := 0; l < mcfg2.NumLayers; l++ {
			gl := &p.GpuModel.Layers[l]
			cpuState := p.CPURunState.SSMState.Layers[l]
			if gl.SSMState == 0 || cpuState == nil {
				continue
			}
			if gpuOwnsState && (cpuSSMDelta || cpuSSMFull) {
				if len(cpuState.State) < stateSize {
					cpuState.State = make([]float32, stateSize)
				}
				DownloadF32(gl.SSMState, cpuState.State[:stateSize])
				if cpuSSMDelta {
					s64 := cpuState.EnsureState64()
					for i := 0; i < stateSize; i++ {
						s64[i] = float64(cpuState.State[i])
					}
				}
				needSync = true
			}
			if cpuSSMFull {
				if len(cpuState.ConvBuf) < convBufSize {
					cpuState.ConvBuf = make([]float32, convBufSize)
				}
				DownloadF32(gl.SSMConvBuf, cpuState.ConvBuf[:convBufSize])
				needSync = true
			}
		}
		if needSync {
			Sync()
			fmt.Fprintf(os.Stderr, "[dlgo/gpu] synced SSM state from GPU to CPU after prefill (gpuOwns=%v, %d layers)\n", gpuOwnsState, mcfg2.NumLayers)
		}
	}

	// DEBUG: zero SSM state/conv after prefill to isolate corruption
	if zeroMode := os.Getenv("DLGO_ZERO_SSM_AFTER_PREFILL"); zeroMode != "" && p.HasSSM {
		mcfg2 := p.CPUModel.Config
		numHeads := mcfg2.SSMTimeStepRank
		headVDim := mcfg2.SSMInnerSize / numHeads
		headKDim := mcfg2.SSMStateSize
		numKVGroups := mcfg2.SSMGroupCount
		if numKVGroups <= 0 {
			numKVGroups = numHeads
		}
		keyDim := numKVGroups * headKDim
		qkvDim := keyDim*2 + numHeads*headVDim
		convK := mcfg2.SSMConvKernel
		for l := 0; l < mcfg2.NumLayers; l++ {
			gl := &p.GpuModel.Layers[l]
			if gl.SSMState != 0 {
				if zeroMode == "state" || zeroMode == "1" {
					ZeroFill(gl.SSMState, uint64(numHeads*headKDim*headVDim*4))
				}
				if zeroMode == "conv" || zeroMode == "1" {
					ZeroFill(gl.SSMConvBuf, uint64(convK*qkvDim*4))
				}
			}
		}
		Sync()
		fmt.Fprintf(os.Stderr, "[dlgo/gpu] DEBUG: zeroed SSM %s after prefill\n", zeroMode)
	}

	// Generate
	genStart := time.Now()
	var generated []int32
	var recentTokens []int32
	var genText strings.Builder
	template := llm.GetArchDescriptor(p.CPUModel.Config.Architecture).ChatTemplate
	if p.CPUModel.Config.ChatTemplate != "" {
		template = p.CPUModel.Config.ChatTemplate
	}
	stopStrings := gpuStopStrings(template)

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

	// Detect thinking mode: prompt ends with "<think>\n" when thinking is enabled.
	// During the thinking phase, suppress EOS/stop tokens to prevent premature
	// end-of-generation from quantization noise (especially IQ3_XXS/IQ2_XXS).
	gpuInThinkBlock := strings.HasSuffix(prompt, "<think>\n")
	// EOS suppression during thinking is only a nudge against quantization
	// noise, so it must be bounded independently of the requested max_tokens.
	// Scaling it off MaxTokens is harmful: a client asking for a large budget
	// (e.g. 128k) also suppresses EOS for ~100k tokens, so a model that rambles
	// inside its thinking block is structurally forbidden from stopping.
	thinkBudget := cfg.MaxTokens * 4 / 5
	if thinkBudget > maxThinkBudget {
		thinkBudget = maxThinkBudget
	}
	// Native-protocol models emit <tool_call>...</tool_call>, and frequently do
	// so inside the thinking block and then repeat the block many times. Once
	// one complete call exists there is nothing left to generate, so stop
	// immediately: this kills the repetition loop and the wasted tokens.
	nativeToolCall := llm.GetArchDescriptor(p.CPUModel.Config.Architecture).ToolCallFormat != llm.ToolCallFormatNone
	gpuStep := 0
	// Guard against an empty assistant turn: aggressive quantizations sometimes
	// close </think> and immediately emit EOS, producing a response with no
	// content and no tool call. Clients cannot act on that and can sit waiting
	// forever. Track when thinking closed and whether any real output followed.
	thinkClosed := false
	thinkCloseStep := 0
	answerStarted := false
	const postThinkGuardSteps = 64

	// Grammar-aware sampling helper (applies EOG suppression + grammar)
	gpuGrammarSample := func(logits []float32) int {
		for _, id := range p.CPUModel.Config.SuppressTokens {
			if int(id) < len(logits) {
				logits[id] = float32(math.Inf(-1))
			}
		}
		suppressStop := gpuInThinkBlock && gpuStep < thinkBudget
		// After thinking closes, do not allow a stop token until at least one
		// non-whitespace output token exists. Without this, a degenerate sample
		// ends the turn with empty content and no tool call.
		if thinkClosed && !answerStarted && gpuStep-thinkCloseStep < postThinkGuardSteps {
			suppressStop = true
		}
		if suppressStop {
			eos := p.CPUModel.Config.EOS
			if int(eos) < len(logits) {
				logits[eos] = float32(math.Inf(-1))
			}
			for _, st := range p.CPUModel.Config.StopTokens {
				if int(st) < len(logits) {
					logits[st] = float32(math.Inf(-1))
				}
			}
		}
		gpuStep++
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
		if slot >= p.MaxSeqLen-1 {
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

		if err := p.patchScalarRow(slot, pos); err != nil {
			return nil, err
		}
		if p.IsPartialGPU {
			GpuForwardPartial(p.CPUModel, p.GpuModel, lastTok, slot, p.KVCache, p.RunState, p.LogitsBuf, p.LayerConfs, p)
			if step%32 == 0 {
				mmap.TrimWorkingSet()
			}
		} else if p.UseFusedForward {
			GpuForwardFusedSSM(p.CPUModel, p.GpuModel, lastTok, slot, p.KVCache, p.RunState, p.LogitsBuf, p.LayerConfs, p)
		} else if err := GpuForward(p.CPUModel, p.GpuModel, lastTok, slot, p.KVCache, p.RunState, p.LogitsBuf, p); err != nil {
			return nil, err
		}
		slot++
		pos++

		nextToken = gpuGrammarSample(p.LogitsBuf)
		generated = append(generated, int32(nextToken))
		recentTokens = append(recentTokens, int32(nextToken))

		// Advance grammar state
		if cfg.Grammar != nil && !eosTokens[int32(nextToken)] {
			cfg.Grammar.AcceptToken(p.Tokenizer.DecodeToken(int32(nextToken)))
		}

		tokenText = p.Tokenizer.DecodeToken(int32(nextToken))
		genText.WriteString(tokenText)
		if thinkClosed && strings.TrimSpace(tokenText) != "" {
			answerStarted = true
		}
		// One complete native tool call is a finished answer: stop generating
		// instead of letting the model re-emit the same block repeatedly.
		if nativeToolCall && strings.Contains(genText.String(), "</tool_call>") {
			break
		}

		if gpuInThinkBlock && strings.Contains(genText.String(), "</"+"think>") {
			gpuInThinkBlock = false
			thinkClosed = true
			thinkCloseStep = gpuStep
		}

		if gpuCheckTextStop(genText.String(), stopStrings) {
			// While the thinking block is still open, hitting a stop string means
			// the model tried to bail mid-thought (aggressive quantizations
			// sometimes spell out <|im_end|>). Honouring it there yields a
			// response with no content and no tool call, which a client cannot
			// act on. Logit-level EOS is already suppressed in this phase, so
			// keep the text-level check consistent - bounded by the same budget.
			if !(gpuInThinkBlock && gpuStep < thinkBudget) {
				break
			}
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
		PromptTokens:   npos,
	}, nil
}

// ensurePrefillBatchState allocates or grows the batched-prefill buffers and
// layer confs for batchSize tokens. GenerateDetailed and System One hidden
// extraction share it so both paths pay the allocation only once.
func (p *GpuPipeline) ensurePrefillBatchState(batchSize int) error {
	if p.BatchState != nil && p.BatchState.Npos >= batchSize {
		return nil
	}
	if p.BatchState != nil {
		p.BatchState.Free()
	}
	mcfg := p.CPUModel.Config
	dim := mcfg.EmbeddingDim
	qDim := mcfg.NumHeads * mcfg.HeadDim
	kvDim := mcfg.NumKVHeads * mcfg.HeadDim
	ffnDim := batchFFNScratchDim(p.CPUModel)
	var bsErr error
	p.BatchState, bsErr = NewGpuBatchState(batchSize, dim, qDim, kvDim, ffnDim)
	if bsErr != nil {
		return fmt.Errorf("gpu: prefill batch alloc: %w", bsErr)
	}
	p.BatchLayerConfs = BuildBatchLayerConfs(p.CPUModel, p.GpuModel, p, p.BatchState, p.KVCache)
	if p.HasGatedQ {
		if err := p.BatchState.AllocGatedQBatch(batchSize, qDim); err != nil {
			return fmt.Errorf("gpu: prefill GatedQ batch alloc: %w", err)
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
			return fmt.Errorf("gpu: prefill SSM batch alloc: %w", err)
		}
	}
	// Hybrid short-conv + MoE models get their own chunked prefill path, which
	// needs per-expert MoE scratch and a [npos*3*dim] short-conv buffer. Decide
	// once, here, because this is where the batch state is (re)built.
	//
	// On by default: the batched chunk is bit-identical to driving the same code
	// one token at a time at every chunk size and chunking tested
	// (gpu/prefill_batch_test.go), so it removes the per-token dispatch overhead
	// without changing the answer. DLGO_PREFILL_BATCH=0 falls back to the
	// per-token path.
	p.UseBatchPrefill = false
	if !p.IsPartialGPU && os.Getenv("DLGO_PREFILL_BATCH") != "0" &&
		shortConvBatchSupported(p.CPUModel, p.GpuModel, mcfg) {
		if err := p.BatchState.AllocMoEBatch(batchSize, dim, mcfg.ExpertCount); err != nil {
			return fmt.Errorf("gpu: prefill MoE batch alloc: %w", err)
		}
		// Pair-major scratch for the grouped MoE, which runs one dispatch per
		// projection instead of one per expert. Not fatal on failure: the
		// per-expert loop is the fallback and needs only the scratch above.
		if err := p.BatchState.AllocMoEGrouped(batchSize, dim, mcfg.ExpertFFNDim,
			mcfg.ExpertUsedCount, mcfg.ExpertCount); err != nil {
			fmt.Fprintf(os.Stderr, "[dlgo/gpu] grouped MoE scratch unavailable, using the per-expert loop: %v\n", err)
		}
		if err := p.BatchState.AllocShortConvBatch(batchSize, dim); err != nil {
			return fmt.Errorf("gpu: prefill shortconv batch alloc: %w", err)
		}
		p.UseBatchPrefill = true
	}

	// Hybrid SSM + GatedQ attention models with a MoE FFN (Qwen3.5-class) go
	// through GpuForwardPrefillBatchHybrid with the position-batched GPU MoE.
	// The alternative branch — one full forward per token — re-reads every
	// active expert for every prompt token, which on a model large enough to
	// spill its experts to host RAM is almost entirely PCIe traffic.
	//
	// A failed MoE scratch allocation is not fatal: UseBatchPrefillHybridMoE
	// stays false and the model keeps the per-token path it has today.
	//
	// !p.UseBatchPrefill: the short-conv path above already allocated this
	// scratch, and AllocMoEBatch would overwrite the handles and leak the first
	// set. A model that satisfies both descriptions takes the short-conv path,
	// which is checked first in generateSegments.
	p.UseBatchPrefillHybridMoE = false
	if !p.IsPartialGPU && !p.UseBatchPrefill && os.Getenv("DLGO_PREFILL_BATCH") != "0" && p.HasMoE {
		if ok, why := hybridBatchMoESupported(p.CPUModel, p.GpuModel, mcfg,
			p.HasSSM, p.HasGatedQ); !ok {
			fmt.Fprintf(os.Stderr, "[dlgo/gpu] batched prefill MoE not usable (%s); "+
				"prompt tokens will each re-read their experts\n", why)
		} else if err := p.BatchState.AllocMoEBatch(batchSize, dim, mcfg.ExpertCount); err != nil {
			fmt.Fprintf(os.Stderr, "[dlgo/gpu] batched MoE scratch unavailable, keeping the per-token prefill: %v\n", err)
		} else {
			if err := p.BatchState.AllocMoEGrouped(batchSize, dim, mcfg.ExpertFFNDim,
				mcfg.ExpertUsedCount, mcfg.ExpertCount); err != nil {
				fmt.Fprintf(os.Stderr, "[dlgo/gpu] grouped MoE scratch unavailable, using the per-expert loop: %v\n", err)
			}
			p.UseBatchPrefillHybridMoE = true
			fmt.Fprintf(os.Stderr, "[dlgo/gpu] batched prefill MoE active: %d-token chunks, "+
				"router %s, grouped=%v\n", batchSize, p.CPUModel.Config.Architecture,
				p.BatchState.MoEPairOut != 0)
		}
	}
	return nil
}

// batchFFNScratchDim returns the row count the batched-FFN scratch (bs.Gate /
// bs.Up / bs.Hidden) has to hold. It is not just cfg.FFNDim: models whose layers
// are all MoE report ffnDim = 0, and both the batched MoE and the shared expert
// reuse that scratch for their own gate/up/hidden rows.
func batchFFNScratchDim(m *llm.Model) int {
	d := m.Config.FFNDim
	if m.Config.ExpertFFNDim > d {
		d = m.Config.ExpertFFNDim
	}
	for i := range m.Layers {
		if g := m.Layers[i].FFNGate; g != nil && g.Rows > d {
			d = g.Rows
		}
		if g := m.Layers[i].FFNGateUpExps; g != nil {
			// Fused [expert][gate | up]: one expert's gate (or up) half.
			if n := g.Rows; n > 0 {
				if e := m.Config.ExpertCount; e > 1 {
					n /= e
				}
				if n > d {
					d = n
				}
			}
		}
		if g := m.Layers[i].FFNGateShared; g != nil && g.Rows > d {
			d = g.Rows
		}
	}
	return d
}

// hybridBatchMoESupported reports whether GpuForwardPrefillBatchHybrid can run
// this whole model with the position-batched GPU MoE.
//
// It mirrors the routing inside that function: every layer's core must be one
// the batched loop handles itself (SSM, GatedQ attention) or the fused C graph
// handles (plain attention with q/k/v/o), and every MoE layer must satisfy
// moeBatchSupported. Residual and norm style are restricted to the ones the
// batched kernels implement.
//
// On rejection it returns a human-readable reason naming the first offending
// layer, which the pipeline prints so a model that silently keeps the slow path
// can be diagnosed without a rebuild.
func hybridBatchMoESupported(m *llm.Model, gm *GpuModel, cfg llm.ModelConfig,
	hasSSM, hasGatedQ bool) (bool, string) {
	if !hasSSM {
		return false, "no SSM state on the GPU"
	}
	if !isHybridSSMModel(m) {
		return false, "not a hybrid SSM model"
	}
	for l := range m.Layers {
		layer := &m.Layers[l]
		gl := &gm.Layers[l]
		if !gl.OnGPU {
			return false, fmt.Sprintf("layer %d is not on the GPU", l)
		}
		if layer.Spec.Norm != llm.NormRMS {
			return false, fmt.Sprintf("layer %d norm is not RMSNorm", l)
		}
		// ResStandard and ResPostAttnFFN both compile to residual_type == 0 and
		// only differ in which weight normalises the FFN input, which
		// BuildBatchLayerConfs already resolves. ResParallel is a different
		// graph (it feeds x_norm to the FFN and never materialises ffn_in) and
		// cannot support the post-MoE residual below.
		if layer.Spec.Residual != llm.ResStandard && layer.Spec.Residual != llm.ResPostAttnFFN {
			return false, fmt.Sprintf("layer %d residual kind %d is not batched", l, layer.Spec.Residual)
		}
		switch {
		case layer.Spec.Core == llm.CoreSSM:
			// Handled by the batched SSM loop (batched input projections, then
			// the per-position recurrence).
		case layer.Spec.GatedQ:
			if !hasGatedQ || gl.Wq == nil || gl.Wk == nil || gl.Wv == nil || gl.Wo == nil {
				return false, fmt.Sprintf("layer %d gated attention is incomplete", l)
			}
		case layer.Spec.Core == llm.CoreAttention:
			if gl.CPUAttn || gl.Wq == nil || gl.Wk == nil || gl.Wv == nil || gl.Wo == nil {
				return false, fmt.Sprintf("layer %d attention is not fully on the GPU", l)
			}
		default:
			return false, fmt.Sprintf("layer %d core kind %d is not batched", l, layer.Spec.Core)
		}
		switch layer.Spec.FFN {
		case llm.FFNMoE:
			if !moeBatchSupported(layer, gl, cfg) {
				return false, fmt.Sprintf("layer %d MoE is not batchable", l)
			}
		case llm.FFNSwiGLU, llm.FFNGeGLU:
			if gl.FFNGate == nil || gl.FFNUp == nil || gl.FFNDown == nil {
				return false, fmt.Sprintf("layer %d dense FFN tensors are incomplete", l)
			}
		default:
			return false, fmt.Sprintf("layer %d FFN kind %d is not batched", l, layer.Spec.FFN)
		}
	}
	return true, ""
}

// SystemOneHidden extracts the final normalized hidden state for a compiled
// System One prompt entirely on GPU: one batched prefill with the vocabulary
// projection skipped, mirroring llm.Pipeline.SystemOneHidden. It requires a
// fully-GPU, non-MoE model (the batched prefill path has no MoE or CPU-attention
// fallback for hidden extraction).
func (p *GpuPipeline) SystemOneHidden(state string, q llm.SystemOneQuestion, cfg llm.SystemOneConfig) ([]float32, []llm.Choice, int, error) {
	if err := cfg.Err(); err != nil {
		return nil, nil, 0, err
	}
	if p.HasMoE || p.IsPartialGPU || p.hasCPUAttn {
		return nil, nil, 0, fmt.Errorf("gpu: System One hidden extraction requires a fully-GPU non-MoE model without CPU-attention layers")
	}
	prompt, choices, err := llm.CompileSystemOnePrompt(p.CPUModel.Config, state, q, cfg)
	if err != nil {
		return nil, nil, 0, err
	}
	tokens := p.Tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return nil, nil, 0, fmt.Errorf("tokenizer produced no tokens")
	}
	if len(tokens) >= p.MaxSeqLen {
		return nil, nil, 0, fmt.Errorf("prompt too long: %d tokens (max %d)", len(tokens), p.MaxSeqLen)
	}

	p.ResetState()

	mcfg := p.CPUModel.Config
	dim := mcfg.EmbeddingDim
	npos := len(tokens)
	hidden := make([]float32, dim)

	if isHybridSSMModel(p.CPUModel) {
		// Chunked prefill with the hidden download on the final chunk.
		const prefillChunkSize = 512
		batchSize := npos
		if batchSize > prefillChunkSize {
			batchSize = prefillChunkSize
		}
		if err := p.ensurePrefillBatchState(batchSize); err != nil {
			return nil, nil, 0, fmt.Errorf("gpu: systemone batch alloc: %w", err)
		}
		for startPos := 0; startPos < npos; startPos += prefillChunkSize {
			if err := cfg.Err(); err != nil {
				return nil, nil, 0, err
			}
			end := startPos + prefillChunkSize
			if end > npos {
				end = npos
			}
			chunkTokens := tokens[startPos:end]
			isLast := end >= npos
			GpuForwardPrefillBatchHybrid(p.CPUModel, p.GpuModel, chunkTokens, p.KVCache, p.RunState,
				p.BatchState, nil, p.BatchLayerConfs, p, startPos, isLast, hidden)
			Sync()
		}
	} else {
		// Pure-attention model: single batched prefill covering every token.
		if err := p.ensurePrefillBatchState(npos); err != nil {
			return nil, nil, 0, fmt.Errorf("gpu: systemone batch alloc: %w", err)
		}
		GpuForwardPrefillBatch(p.CPUModel, p.GpuModel, tokens, p.KVCache, p.RunState,
			p.BatchState, nil, p.BatchLayerConfs, hidden)
		Sync()
	}

	// A failed download would leave the buffer all zeros; RMSNorm output can
	// never legitimately be all-zero, so treat that as an extraction failure.
	nonZero := false
	for _, v := range hidden {
		if v != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		return nil, nil, 0, fmt.Errorf("gpu: hidden extraction produced an all-zero state")
	}
	return hidden, choices, npos, nil
}

// SystemOneWithReadout evaluates a closed-set question on GPU: batched prefill
// extracts the final hidden state without the vocabulary projection, then the
// trained linear readout is applied on CPU (a tiny matmul over the downloaded
// features). Results use the same normalization and abstention rules as the
// CPU engine.
func (p *GpuPipeline) SystemOneWithReadout(state string, q llm.SystemOneQuestion, cfg llm.SystemOneConfig, readout *llm.SystemOneReadout) (*llm.SystemOneResult, error) {
	if readout == nil {
		return nil, fmt.Errorf("System One readout is nil")
	}
	if d := readout.NormalizeDevice(); d != "gpu" && d != "any" {
		return nil, fmt.Errorf("System One readout was trained on %s features; the GPU engine requires a gpu or any-device readout", readout.NormalizeDevice())
	}
	start := time.Now()
	hidden, choices, promptTokens, err := p.SystemOneHidden(state, q, cfg)
	if err != nil {
		return nil, err
	}
	prefillMs := float64(time.Since(start).Microseconds()) / 1000.0
	result, err := llm.ApplySystemOneReadout(q, choices, hidden, readout, cfg, promptTokens, prefillMs)
	if err != nil {
		return nil, err
	}
	result.Engine = "gpu"
	return result, nil
}

// SystemOneMulti evaluates several questions against the same state on GPU.
// When the model supports state branching (fully-GPU, non-MoE, no CPU-attention
// fallback, no CPU-side SSM state, hybrid SSM batch path), the shared prompt
// prefix (system + state) is prefilled exactly once, the KV/SSM state is
// snapshotted on device, and each question runs only its short suffix from a
// restored copy of that state — mirroring the CPU SystemOneMulti contract.
// Questions wider than the readout's option count degrade to CPU continuation
// scoring; questions whose suffix does not concatenate at the token boundary
// (a BPE merge spanning the prefix/suffix seam) fall back to a standalone
// per-question prefill, so results are always identical to evaluating the
// question alone. Results are returned in caller question order.
func (p *GpuPipeline) SystemOneMulti(state string, questions []llm.SystemOneQuestion, cfg llm.SystemOneConfig, readout *llm.SystemOneReadout) ([]*llm.SystemOneResult, error) {
	if err := cfg.Err(); err != nil {
		return nil, err
	}
	if len(questions) == 0 {
		return nil, fmt.Errorf("SystemOneMulti requires at least one question")
	}
	if readout != nil {
		if d := readout.NormalizeDevice(); d != "gpu" && d != "any" {
			return nil, fmt.Errorf("System One readout was trained on %s features; the GPU engine requires a gpu or any-device readout", readout.NormalizeDevice())
		}
	}
	if p.HasMoE || p.IsPartialGPU || p.hasCPUAttn || p.CPURunState != nil || !isHybridSSMModel(p.CPUModel) {
		return p.systemOneMultiPerQuestion(state, questions, cfg, readout)
	}

	// Compile every question up front so validation errors surface before any
	// expensive work, and resolve per-question suffix tokenizations.
	type compiledQuestion struct {
		suffixTokens []int32
		choices      []llm.Choice
		question     llm.SystemOneQuestion
	}
	compiled := make([]compiledQuestion, len(questions))
	prefix := ""
	var prefixTokens []int32
	for i, q := range questions {
		if len(q.Choices) < 2 {
			return nil, fmt.Errorf("question %d (%q): System One requires at least two choices", i, q.ID)
		}
		if q.Prompt == "" {
			return nil, fmt.Errorf("question %d (%q): prompt is empty", i, q.ID)
		}
		qPrefix, qSuffix, choices, err := llm.CompileSystemOneParts(state, q, cfg)
		if err != nil {
			return nil, fmt.Errorf("question %d (%q): %w", i, q.ID, err)
		}
		if i == 0 {
			prefix = qPrefix
			prefixTokens = p.Tokenizer.Encode(prefix)
			if len(prefixTokens) == 0 {
				return nil, fmt.Errorf("System One tokenizer produced no prefix tokens")
			}
			if len(prefixTokens) >= p.MaxSeqLen {
				return nil, fmt.Errorf("System One prefix too long: %d tokens (max %d)", len(prefixTokens), p.MaxSeqLen)
			}
		} else if qPrefix != prefix {
			// All questions must share the same system+state prefix.
			return nil, fmt.Errorf("question %d (%q): inconsistent prompt prefix", i, q.ID)
		}
		suffixTokens := p.Tokenizer.Encode(qSuffix)
		if len(suffixTokens) == 0 {
			return nil, fmt.Errorf("question %d (%q): tokenizer produced no suffix tokens", i, q.ID)
		}
		if len(prefixTokens)+len(suffixTokens) >= p.MaxSeqLen {
			return nil, fmt.Errorf("question %d (%q): prompt too long: %d tokens (max %d)", i, q.ID, len(prefixTokens)+len(suffixTokens), p.MaxSeqLen)
		}
		compiled[i] = compiledQuestion{suffixTokens: suffixTokens, choices: choices, question: q}
	}

	// Token-boundary safety: tokenizing prefix and suffix separately must
	// produce exactly the token stream of the concatenated prompt. A BPE merge
	// spanning the seam would make the shared KV state invalid for the suffix.
	// Seam-local check: only a bounded window of bytes around the seam can be
	// affected by merges that cross it (BPE merges have bounded length, bounded
	// by the longest merge in the vocab); encoding the full prefix again per
	// question is O(N) tokenization work per question and dominated long-state
	// multi requests. We encode the concatenation of the seam window and
	// compare against separately-encoded prefix-window and suffix tokens; if
	// anything mismatches we conservatively mark the question token-unsafe
	// (standalone fallback), preserving correctness.
	for i := range compiled {
		_, qSuffix, _, err := llm.CompileSystemOneParts(state, questions[i], cfg)
		if err != nil {
			return nil, fmt.Errorf("question %d (%q): %w", i, questions[i].ID, err)
		}
		compiled[i].suffixTokens = p.checkSeamBoundary(prefix, prefixTokens, qSuffix)
	}

	// Prefill the shared prefix once.
	maxSuffix := 0
	for i := range compiled {
		if compiled[i].suffixTokens != nil && len(compiled[i].suffixTokens) > maxSuffix {
			maxSuffix = len(compiled[i].suffixTokens)
		}
	}
	// Prefix prefill is chunked: allocating batch scratch for the entire
	// document wastes VRAM and first-request time without doing larger GEMMs.
	batchNeeded := len(prefixTokens)
	if batchNeeded > 512 {
		batchNeeded = 512
	}
	if maxSuffix > batchNeeded {
		batchNeeded = maxSuffix
	}
	if err := p.ensurePrefillBatchState(batchNeeded); err != nil {
		return nil, fmt.Errorf("gpu: systemone multi batch alloc: %w", err)
	}

	// Shared-prefix state: serve from the prefix cache when this exact state
	// was prefilled recently (repeated decisions over the same document skip
	// the long prefill entirely), otherwise prefill once and cache the
	// snapshot. The snapshot is cache-owned on both paths — do not free it.
	// Numerical safety: a cache hit restores byte-identical KV/SSM/conv state
	// and sets KVCache.Len; positions beyond the prefix are overwritten by
	// each suffix pass before they are ever read, so restored-from-cache
	// results are bit-identical to a fresh prefill (verified by the parity
	// tests and the cached-vs-fresh regression test).
	var snap *GpuStateSnapshot
	// A cache lookup does not imply its KV prefix is live: generation or
	// another decision request may have replaced it. Fresh prefill does.
	prefixResident := false
	var sharedPrefillMs float64
	prefixKey := sha256.Sum256([]byte(prefix))
	if cached := p.lookupPrefixCache(prefixKey, len(prefixTokens)); cached != nil {
		snap = cached
	} else {
		p.ResetState()
		prefillStart := time.Now()
		const prefillChunkSize = 512
		for startPos := 0; startPos < len(prefixTokens); startPos += prefillChunkSize {
			if err := cfg.Err(); err != nil {
				return nil, err
			}
			end := startPos + prefillChunkSize
			if end > len(prefixTokens) {
				end = len(prefixTokens)
			}
			GpuForwardPrefillBatchHybrid(p.CPUModel, p.GpuModel, prefixTokens[startPos:end], p.KVCache, p.RunState,
				p.BatchState, nil, p.BatchLayerConfs, p, startPos, false, nil)
			Sync()
		}
		sharedPrefillMs = float64(time.Since(prefillStart).Microseconds()) / 1000.0
		if err := cfg.Err(); err != nil {
			return nil, err
		}
		s, err := p.SnapshotState(len(prefixTokens))
		if err != nil {
			return nil, fmt.Errorf("gpu: systemone multi state snapshot: %w", err)
		}
		p.storePrefixCache(prefixKey, len(prefixTokens), s)
		snap = s
		prefixResident = true
	}

	results := make([]*llm.SystemOneResult, len(questions))
	for i, cq := range compiled {
		if err := cfg.Err(); err != nil {
			return nil, err
		}
		// Per-question graceful degradation: the trained readout covers a
		// fixed number of answer slots; questions wider than that fall back
		// to generic continuation scoring instead of failing the request.
		effectiveReadout := readout
		degraded := false
		if readout != nil && len(cq.choices) > readout.MaxOptions {
			effectiveReadout = nil
			degraded = true
		}
		if effectiveReadout == nil {
			// Degraded (too wide) or readout-less: CPU continuation scoring.
			if p.CPUPipeline == nil {
				return nil, fmt.Errorf("question %d (%q): no CPU pipeline available for continuation scoring", i, cq.question.ID)
			}
			r, err := p.CPUPipeline.SystemOne(state, cq.question, cfg)
			if err != nil {
				return nil, fmt.Errorf("question %d (%q): %w", i, cq.question.ID, err)
			}
			r.Degraded = readout != nil
			r.Engine = "cpu"
			r.SharedPrefillTimeMs = sharedPrefillMs
			results[i] = r
			continue
		}
		if cq.suffixTokens == nil {
			// Standalone prefill replaces prefix KV; the next branch must
			// restore the full snapshot, not just recurrent state.
			prefixResident = false
			// Token-boundary mismatch: evaluate this question standalone.
			r, err := p.SystemOneWithReadout(state, cq.question, cfg, effectiveReadout)
			if err != nil {
				return nil, fmt.Errorf("question %d (%q): %w", i, cq.question.ID, err)
			}
			r.SharedPrefillTimeMs = sharedPrefillMs
			r.Degraded = degraded
			results[i] = r
			continue
		}

		// Suffix passes append KV beyond the immutable prefix. Once that
		// prefix is resident, branch only the recurrent state instead of
		// copying the entire document KV again for every question.
		var restoreErr error
		if prefixResident {
			restoreErr = p.RestoreRecurrentState(snap)
		} else {
			restoreErr = p.RestoreState(snap)
		}
		if restoreErr != nil {
			return nil, fmt.Errorf("question %d (%q): %w", i, cq.question.ID, restoreErr)
		}
		prefixResident = true
		suffixStart := time.Now()
		hidden := make([]float32, p.CPUModel.Config.EmbeddingDim)
		GpuForwardPrefillBatchHybrid(p.CPUModel, p.GpuModel, cq.suffixTokens, p.KVCache, p.RunState,
			p.BatchState, nil, p.BatchLayerConfs, p, len(prefixTokens), true, hidden)
		Sync()
		suffixMs := float64(time.Since(suffixStart).Microseconds()) / 1000.0

		// A failed download would leave the buffer all zeros; RMSNorm output
		// can never legitimately be all-zero.
		nonZero := false
		for _, v := range hidden {
			if v != 0 {
				nonZero = true
				break
			}
		}
		if !nonZero {
			return nil, fmt.Errorf("gpu: question %d (%q): hidden extraction produced an all-zero state", i, cq.question.ID)
		}
		r, err := llm.ApplySystemOneReadout(cq.question, cq.choices, hidden, effectiveReadout, cfg, len(prefixTokens)+len(cq.suffixTokens), suffixMs)
		if err != nil {
			return nil, fmt.Errorf("question %d (%q): %w", i, cq.question.ID, err)
		}
		r.SharedPrefillTimeMs = sharedPrefillMs
		r.Engine = "gpu"
		results[i] = r
	}
	return results, nil
}

// seamCheckTokens is how many trailing prefix tokens the seam-local boundary
// check re-encodes. BPE merges are pairwise and only ever span the immediate
// seam; a chain of such merges consumes one more token per link, so a window
// of this many tokens vastly exceeds any real cross-seam chain.
const seamCheckTokens = 16

// checkSeamBoundary verifies that encode(prefix)+encode(suffix) is exactly
// encode(prefix+suffix) by re-encoding only a token-aligned window at the end
// of the prefix plus the suffix. It returns the suffix tokens, or nil when the
// seam cannot be proven safe (the caller then evaluates the question
// standalone).
//
// Correctness argument: take the last K prefix tokens and decode them to their
// exact bytes W. Because W starts at a token boundary of the prefix's final
// segmentation, no BPE merge pairs across that boundary (a spanning merge
// would have produced a token spanning it), so BPE's merges strictly inside W
// happen identically whether or not the rest of the prefix is present — hence
// encode(W) == tail tokens. Now encode(W+suffix): any cross-seam merge or
// merge chain alters the joined token stream within the window, making it
// differ from tail+suffixTokens. If it does not differ, the full-text encoding
// equals encode(prefix)+encode(suffix) everywhere: bytes left of the window
// tokenize identically (their merges are all strictly inside that region and
// unaffected), and the window+suffix region is checked exactly.
func (p *GpuPipeline) checkSeamBoundary(prefix string, prefixTokens []int32, suffix string) []int32 {
	suffixTokens := p.Tokenizer.Encode(suffix)
	if len(suffixTokens) == 0 || len(prefixTokens) == 0 {
		return nil
	}
	k := seamCheckTokens
	if k > len(prefixTokens) {
		k = len(prefixTokens)
	}
	tail := prefixTokens[len(prefixTokens)-k:]
	windowText := p.Tokenizer.Decode(append([]int32(nil), tail...))
	if windowText == "" {
		return nil
	}
	// Insurance: the window must re-encode to exactly the tail tokens (holds
	// by boundary independence for well-formed vocabularies; fall back to the
	// standalone path if this tokenizer behaves unexpectedly).
	wt := p.Tokenizer.Encode(windowText)
	if len(wt) != len(tail) {
		return nil
	}
	for i := range wt {
		if wt[i] != tail[i] {
			return nil
		}
	}
	joined := p.Tokenizer.Encode(windowText + suffix)
	want := append(append([]int32(nil), tail...), suffixTokens...)
	if len(joined) != len(want) {
		return nil
	}
	for i := range joined {
		if joined[i] != want[i] {
			return nil
		}
	}
	return suffixTokens
}

// systemOneMultiPerQuestion evaluates each question with an independent full
// prefill. This is the fallback for models the shared-prefill path cannot
// serve (MoE, partial GPU, CPU-attention layers, CPU-side SSM state, or
// non-hybrid batch paths that prefill from position 0).
func (p *GpuPipeline) systemOneMultiPerQuestion(state string, questions []llm.SystemOneQuestion, cfg llm.SystemOneConfig, readout *llm.SystemOneReadout) ([]*llm.SystemOneResult, error) {
	results := make([]*llm.SystemOneResult, len(questions))
	for i, q := range questions {
		if err := cfg.Err(); err != nil {
			return nil, err
		}
		useGPU := readout != nil && len(q.Choices) <= readout.MaxOptions
		if !useGPU {
			// Degraded (too wide) or readout-less: CPU continuation scoring.
			if p.CPUPipeline == nil {
				return nil, fmt.Errorf("question %d (%q): no CPU pipeline available for continuation scoring", i, q.ID)
			}
			r, err := p.CPUPipeline.SystemOne(state, q, cfg)
			if err != nil {
				return nil, fmt.Errorf("question %d (%q): %w", i, q.ID, err)
			}
			r.Degraded = readout != nil
			r.Engine = "cpu"
			results[i] = r
			continue
		}
		r, err := p.SystemOneWithReadout(state, q, cfg, readout)
		if err != nil {
			return nil, fmt.Errorf("question %d (%q): %w", i, q.ID, err)
		}
		results[i] = r
	}
	return results, nil
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

func gpuStopStrings(chatTemplate string) []string {
	if chatTemplate == "harmony" {
		// Harmony models (GPT-OSS) use <|channel|>, <|start|>, <|message|>,
		// <|end|>, <|constrain|> as structural protocol tokens during normal
		// generation. Only <|return|> and <|endoftext|> are true stops.
		return []string{
			"<|return|>",
			"<|endoftext|>",
		}
	}
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
