//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math/rand"
	"os"
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
// (run state, batch state, SSM scratch). Does NOT include KV cache or
// per-layer weights — those are computed per-layer in the budget solver.
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

	// Non-layer weight buffers (embed, output, norms)
	// Accounted separately from per-layer weights.

	// Safety margin (64 MB)
	total += 64 * 1024 * 1024

	return total
}

// computeGPULayerBudget determines how many layers fit in available VRAM,
// accounting for both weight data AND per-layer KV cache VRAM.
func computeGPULayerBudget(m *llm.Model, maxSeqLen int) int {
	totalVRAM := int64(VRAMBytes())
	if totalVRAM <= 0 {
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

	available := totalVRAM - fixedOverhead - nonLayerBytes
	if available <= 0 {
		return 0
	}

	// Per-layer KV cache cost
	kvDim := int64(m.Config.NumKVHeads * m.Config.HeadDim)
	kvPerLayer := 2 * int64(maxSeqLen) * kvDim * 4 // K + V buffers

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

	// Greedily add layers: each layer costs weights + KV cache + optional SSM state
	numLayers := 0
	for l := 0; l < len(m.Layers); l++ {
		layerCost := llm.EstimateLayerBytes(&m.Layers[l]) + kvPerLayer
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

		if cl.Wq != nil && !supportsGPUQType(cl.Wq.Type) {
			gl.CPUAttn = true
		}
		if cl.AttnSinks != nil {
			gl.CPUAttn = true
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
	fmt.Printf("[dlgo/gpu] Uploading model to %s (%.0f MB VRAM)...\n",
		DeviceName(), totalVRAM)

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

	isPartial := numGPULayers < cfg.NumLayers

	if numGPULayers == 0 {
		return nil, fmt.Errorf("insufficient VRAM (%.0f MB) for even 1 layer — use CPU mode", totalVRAM)
	}

	gm, err := UploadModel(m, numGPULayers)
	if err != nil {
		return nil, fmt.Errorf("gpu upload: %w", err)
	}

	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim

	rs := NewGpuRunState(dim, qDim, kvDim, ffnDim, cfg.VocabSize)
	kv := NewGpuKVCache(cfg.NumLayers, numGPULayers, cpuPipeline.MaxSeqLen, kvDim)

	layerConfs := BuildLayerConfs(m, gm, rs, kv)

	maxDim := dim
	if ffnDim > maxDim {
		maxDim = ffnDim
	}
	q8_1NumBlocks := (maxDim + 31) / 32
	q8_1Scratch := Alloc(uint64(q8_1NumBlocks) * 36)

	if os.Getenv("DLGO_DP4A") == "1" {
		for _, lc := range layerConfs {
			lc.SetDP4A(q8_1Scratch)
		}
		fmt.Println("[dlgo/gpu] dp4a enabled via DLGO_DP4A=1")
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
	}

	// Only use fused forward when ALL layers are on GPU and model has no MoE
	if !isPartial && cfg.ExpertCount == 0 {
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

	// Allocate CPU-side state if needed (partial GPU, MoE, MLA, CPU attn, or hybrid SSM)
	needCPUState := isPartial || cfg.ExpertCount > 0 || pipe.HasMLA || hasCPUAttn
	if needCPUState {
		pipe.CPURunState = llm.NewRunState(cfg, cpuPipeline.MaxSeqLen)

		if isPartial {
			pipe.CPUKVCache = memory.NewMultiLayerKVCache(cfg.NumLayers, cpuPipeline.MaxSeqLen, kvDim)
			pipe.CPUBatchState = llm.NewBatchState(cfg, cpuPipeline.MaxSeqLen)
		}

		// MLA or CPU attention needs a CPU KV cache
		if (pipe.HasMLA || hasCPUAttn) && pipe.CPUKVCache == nil {
			pipe.CPUKVCache = memory.NewMultiLayerKVCache(cfg.NumLayers, cpuPipeline.MaxSeqLen, kvDim)
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

	// Pin CPU layers to RAM for optimal inference speed (avoid page faults).
	// Budget: use up to 80% of available physical RAM.
	if isPartial {
		pinCPULayersToRAM(m, numGPULayers)
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

// pinCPULayersToRAM copies non-GPU layer weights from mmap to heap memory,
// prioritizing earlier layers and respecting a system RAM budget.
func pinCPULayersToRAM(m *llm.Model, numGPULayers int) {
	memInfo, err := mmap.GetSystemMemInfo()
	if err != nil {
		fmt.Printf("[dlgo/gpu] Warning: couldn't query system RAM: %v\n", err)
		return
	}

	// Use at most 80% of available physical RAM for pinning
	budget := int64(float64(memInfo.AvailablePhysical) * 0.80)
	pinnedBytes := int64(0)
	pinnedLayers := 0

	for l := numGPULayers; l < len(m.Layers); l++ {
		layerBytes := llm.EstimateLayerBytes(&m.Layers[l])
		if pinnedBytes+layerBytes > budget {
			break
		}
		llm.PinLayerToRAM(&m.Layers[l])
		pinnedBytes += layerBytes
		pinnedLayers++
	}

	remaining := len(m.Layers) - numGPULayers - pinnedLayers
	fmt.Printf("[dlgo/gpu] Pinned %d CPU layers to RAM (%.0f MB), %d layers on mmap\n",
		pinnedLayers, float64(pinnedBytes)/(1024*1024), remaining)
}

// FreeAll releases all GPU resources held by this pipeline.
func (p *GpuPipeline) FreeAll() {
	if p == nil {
		return
	}
	p.GpuModel.FreeAll()
	p.RunState.FreeAll()
	p.KVCache.FreeAll()
	p.BatchState.Free()
	freeBuf(p.Q8_1Scratch)
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

	if p.BatchState == nil || p.BatchState.Npos < npos {
		if p.BatchState != nil {
			p.BatchState.Free()
		}
		dim := mcfg.EmbeddingDim
		qDim := mcfg.NumHeads * mcfg.HeadDim
		kvDim := mcfg.NumKVHeads * mcfg.HeadDim
		ffnDim := mcfg.FFNDim
		p.BatchState = NewGpuBatchState(npos, dim, qDim, kvDim, ffnDim)
		p.BatchLayerConfs = BuildBatchLayerConfs(p.CPUModel, p.GpuModel, p.BatchState, p.KVCache)
		if p.HasGatedQ {
			p.BatchState.AllocGatedQBatch(npos, qDim)
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
			p.BatchState.AllocSSMBatch(npos, qkvDim, valueDim, numHeads)
		}
	}

	prefillStart := time.Now()
	if p.IsPartialGPU {
		for i, tok := range tokens {
			GpuForwardPartial(p.CPUModel, p.GpuModel, tok, i, p.KVCache, p.RunState, p.LogitsBuf, p.LayerConfs, p)
		}
	} else if !p.UseFusedForward {
		// Per-token prefill using GpuForward (has CPU fallback for unsupported quant types)
		for i, tok := range tokens {
			GpuForward(p.CPUModel, p.GpuModel, tok, i, p.KVCache, p.RunState, p.LogitsBuf, p)
		}
	} else {
		isHybrid := isHybridSSMModel(p.CPUModel)
		if isHybrid {
			GpuForwardPrefillBatchHybrid(p.CPUModel, p.GpuModel, tokens, p.KVCache, p.RunState,
				p.BatchState, p.LogitsBuf, p.BatchLayerConfs, p)
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

	pos := len(tokens)
	nextToken := ops.SampleToken(p.LogitsBuf, cfg.Sampler, recentTokens, rng)
	generated = append(generated, int32(nextToken))
	recentTokens = append(recentTokens, int32(nextToken))

	if cfg.Stream != nil {
		cfg.Stream(p.Tokenizer.DecodeToken(int32(nextToken)))
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
		} else if p.UseFusedForward {
			GpuForwardFusedSSM(p.CPUModel, p.GpuModel, lastTok, pos, p.KVCache, p.RunState, p.LogitsBuf, p.LayerConfs, p)
		} else if err := GpuForward(p.CPUModel, p.GpuModel, lastTok, pos, p.KVCache, p.RunState, p.LogitsBuf, p); err != nil {
			return nil, err
		}
		pos++

		nextToken = ops.SampleToken(p.LogitsBuf, cfg.Sampler, recentTokens, rng)
		generated = append(generated, int32(nextToken))
		recentTokens = append(recentTokens, int32(nextToken))
		if len(recentTokens) > 64 {
			recentTokens = recentTokens[1:]
		}

		if cfg.Stream != nil {
			cfg.Stream(p.Tokenizer.DecodeToken(int32(nextToken)))
		}
	}

done:
	Sync()
	genMs := float64(time.Since(genStart).Microseconds()) / 1000.0

	text := p.Tokenizer.Decode(generated)
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
