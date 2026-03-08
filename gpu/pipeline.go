//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/computerex/dlgo/core"
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
}

// UploadModel copies all model weights to GPU memory.
func UploadModel(m *llm.Model) (*GpuModel, error) {
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
func NewGpuPipeline(cpuPipeline *llm.Pipeline) (*GpuPipeline, error) {
	if err := Init(); err != nil {
		return nil, err
	}

	m := cpuPipeline.Model
	cfg := m.Config

	fmt.Printf("[dlgo/gpu] Uploading model to %s (%.0f MB VRAM)...\n",
		DeviceName(), float64(VRAMBytes())/(1024*1024))

	gm, err := UploadModel(m)
	if err != nil {
		return nil, fmt.Errorf("gpu upload: %w", err)
	}

	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim

	rs := NewGpuRunState(dim, qDim, kvDim, ffnDim, cfg.VocabSize)
	kv := NewGpuKVCache(cfg.NumLayers, cpuPipeline.MaxSeqLen, kvDim)

	layerConfs := BuildLayerConfs(m, gm, rs, kv)

	// dp4a disabled: the quantize+barrier overhead per layer (~22µs × layers)
	// negates ALU savings. TODO: fuse quantization into MatVec shader.
	var q8_1Scratch Buf

	fmt.Printf("[dlgo/gpu] Model loaded to GPU (%d layers)\n", cfg.NumLayers)

	pipe := &GpuPipeline{
		CPUModel:    m,
		GpuModel:    gm,
		Tokenizer:   cpuPipeline.Tokenizer,
		KVCache:     kv,
		RunState:    rs,
		MaxSeqLen:   cpuPipeline.MaxSeqLen,
		LogitsBuf:   make([]float32, cfg.VocabSize),
		LayerConfs:  layerConfs,
		Q8_1Scratch: q8_1Scratch,
		UseFusedForward: supportsFusedForwardGPU(m),
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
		headVDim := cfg.SSMInnerSize / numHeads
		headKDim := cfg.SSMStateSize
		valueDim := numHeads * headVDim
		keyDim := numHeads * headKDim
		qkvDim := keyDim*2 + valueDim
		convK := cfg.SSMConvKernel

		rs.AllocSSMScratch(qkvDim, valueDim, numHeads)

		ssmLayerCount := 0
		for l := 0; l < cfg.NumLayers; l++ {
			if m.Layers[l].Spec.Core == llm.CoreSSM {
				gl := &gm.Layers[l]
				gl.SSMState = Alloc(uint64(numHeads * headKDim * headVDim * 4))
				gl.SSMConvBuf = Alloc(uint64(convK * qkvDim * 4))
				ssmLayerCount++
			}
		}

		pipe.HasSSM = true
		fmt.Printf("[dlgo/gpu] SSM state on GPU (%d SSM layers, %d heads, state=%dx%d)\n",
			ssmLayerCount, numHeads, headKDim, headVDim)
	}

	return pipe, nil
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

	p.KVCache.Reset()
	if p.HasSSM {
		mcfg2 := p.CPUModel.Config
		numHeads := mcfg2.SSMTimeStepRank
		headVDim := mcfg2.SSMInnerSize / numHeads
		headKDim := mcfg2.SSMStateSize
		qkvDim := numHeads*headKDim*2 + numHeads*headVDim
		convK := mcfg2.SSMConvKernel
		for l := 0; l < mcfg2.NumLayers; l++ {
			gl := &p.GpuModel.Layers[l]
			if gl.SSMState != 0 {
				ZeroFill(gl.SSMState, uint64(numHeads*headKDim*headVDim*4))
				ZeroFill(gl.SSMConvBuf, uint64(convK*qkvDim*4))
			}
		}
	}

	mcfg := p.CPUModel.Config
	npos := len(tokens)

	useBatchPrefill := supportsBatchPrefillGPU(p.CPUModel)
	if useBatchPrefill {
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
		}
	}

	prefillStart := time.Now()
	if useBatchPrefill {
		GpuForwardPrefillBatch(p.CPUModel, p.GpuModel, tokens, p.KVCache, p.RunState, p.BatchState, p.LogitsBuf, p.BatchLayerConfs)
	} else {
		for i, tok := range tokens {
			if p.UseFusedForward {
				GpuForwardFusedSSM(p.CPUModel, p.GpuModel, tok, i, p.KVCache, p.RunState, p.LogitsBuf, p.LayerConfs, p)
			} else if err := GpuForward(p.CPUModel, p.GpuModel, tok, i, p.KVCache, p.RunState, p.LogitsBuf, p); err != nil {
				return nil, err
			}
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

		if p.UseFusedForward {
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

// supportsBatchPrefillGPU gates the newer batched prefill path behind
// per-architecture validation. SSM layers are recurrent and cannot batch
// across positions.
func supportsBatchPrefillGPU(m *llm.Model) bool {
	if m.Config.FullAttentionInterval > 0 && m.Config.SSMInnerSize > 0 {
		return false
	}
	for i := range m.Layers {
		if m.Layers[i].Spec.GatedQ {
			return false
		}
	}
	return true
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
