package llm

import (
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/ops"
)

// RunState holds pre-allocated buffers for inference, avoiding per-token allocations.
type RunState struct {
	X       []float32 // [dim] current activation
	XNorm   []float32 // [dim] normalized activation
	Q       []float32 // [qDim] query projection
	K       []float32 // [kvDim] key projection
	V       []float32 // [kvDim] value projection
	AttnOut []float32 // [qDim] attention output
	AttnProj []float32 // [dim] output projection
	FFNIn   []float32 // [dim] FFN input (after residual)
	FFNNorm []float32 // [dim] FFN normalized
	Gate    []float32 // [ffnDim] gate projection
	Up      []float32 // [ffnDim] up projection
	Hidden  []float32 // [ffnDim] gated hidden
	FFNOut  []float32 // [dim] FFN output
	Logits  []float32 // [vocabSize] output logits
	Scores  []float32 // [maxSeqLen] attention scores scratch

	// Qwen3.5 gated attention: Wq outputs interleaved [Q,gate] per head
	QFull  []float32 // [2*qDim] fused Q+gate output (nil for non-gated models)
	QGate  []float32 // [qDim] attention gate values (nil for non-gated models)

	// Precomputed RoPE tables (populated by PrecomputeRoPE)
	ropeCos     []float32
	ropeSin     []float32
	ropeHeadDim int
	ropeDim     int // partial RoPE dimension (may be < ropeHeadDim)
	ropeNeox    bool

	// SSM (Gated Delta Net) scratch buffers — nil for pure transformer models
	SSMRun   *SSMRunState
	SSMState *memory.SSMStateCache

	// Worker pool for parallel matmul
	Pool *blas.Pool
}

// NewRunState allocates all buffers for a model.
func NewRunState(cfg ModelConfig, maxSeqLen int) *RunState {
	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim

	rs := &RunState{
		X:       make([]float32, dim),
		XNorm:   make([]float32, dim),
		Q:       make([]float32, qDim),
		K:       make([]float32, kvDim),
		V:       make([]float32, kvDim),
		AttnOut: make([]float32, qDim),
		AttnProj: make([]float32, dim),
		FFNIn:   make([]float32, dim),
		FFNNorm: make([]float32, dim),
		Gate:    make([]float32, ffnDim),
		Up:      make([]float32, ffnDim),
		Hidden:  make([]float32, ffnDim),
		FFNOut:  make([]float32, dim),
		Logits:  make([]float32, cfg.VocabSize),
		Scores:  make([]float32, maxSeqLen),
		Pool:    blas.DefaultPool(),
	}
	rs.PrecomputeRoPE(maxSeqLen, cfg.RopeDim, cfg.HeadDim, cfg.RopeFreqBase)
	rs.SetRopeNeox(cfg.RopeNeox)

	// Allocate gated attention buffers for Qwen3.5 (fused Q+gate in Wq)
	if cfg.FullAttentionInterval > 0 {
		rs.QFull = make([]float32, 2*qDim)
		rs.QGate = make([]float32, qDim)
	}

	// Allocate SSM buffers for hybrid Mamba/Attention models
	if cfg.FullAttentionInterval > 0 && cfg.SSMInnerSize > 0 {
		numHeads := cfg.SSMTimeStepRank
		headVDim := cfg.SSMInnerSize / numHeads
		headKDim := cfg.SSMStateSize
		valueDim := numHeads * headVDim
		keyDim := numHeads * headKDim
		qkvDim := keyDim*2 + valueDim

		rs.SSMRun = &SSMRunState{
			QKV:   make([]float32, qkvDim),
			Z:     make([]float32, valueDim),
			Alpha: make([]float32, numHeads),
			Beta:  make([]float32, numHeads),
			Y:     make([]float32, valueDim),
		}
		rs.SSMState = memory.NewSSMStateCache(
			cfg.NumLayers, numHeads, headKDim, headVDim,
			qkvDim, cfg.SSMConvKernel,
			func(l int) bool { return isSSMLayer(l, cfg) },
		)
	}

	return rs
}

// isSSMLayer returns true if layer l uses the SSM/delta-net path instead of attention.
func isSSMLayer(l int, cfg ModelConfig) bool {
	if cfg.FullAttentionInterval <= 0 {
		return false
	}
	return ((l + 1) % cfg.FullAttentionInterval) != 0
}

// Forward performs a single-token forward pass through the model.
// Uses parallel matmul via the RunState's worker pool for acceleration.
// Returns logits for the next token.
func Forward(m *Model, token int32, pos int, kv *memory.MultiLayerKVCache, rs *RunState) []float32 {
	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvMul := numHeads / numKVHeads
	pool := rs.Pool

	// 1. Token embedding
	_ = m.TokenEmbed.DequantizeRow(int(token), rs.X)
	if cfg.EmbedScale != 0 {
		ops.Scale(rs.X, cfg.EmbedScale)
	}

	// 2. Layer loop
	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]

		// Pre-norm: LayerNorm (Phi-2) or RMSNorm
		if layer.AttnNormBias != nil {
			ops.LayerNorm(rs.XNorm, rs.X, layer.AttnNorm, layer.AttnNormBias, cfg.RMSNormEps)
		} else {
			ops.RMSNorm(rs.XNorm, rs.X, layer.AttnNorm, cfg.RMSNormEps)
		}

		// Branch: SSM (delta net) or attention
		if isSSMLayer(l, cfg) && layer.SSMInProj != nil {
			forwardSSMLayer(layer, rs, rs.SSMRun, rs.SSMState.Layers[l], rs.XNorm, cfg, pool)
		} else {
			forwardAttention(layer, rs, kv, l, pos, numHeads, numKVHeads, headDim, kvMul, cfg, pool)
		}

		if layer.FFNNorm != nil {
			// Standard with separate FFN norm (LLaMA, Gemma, Qwen2/3)
			if layer.PostAttnNorm != nil {
				ops.RMSNormInPlace(rs.AttnProj, layer.PostAttnNorm, cfg.RMSNormEps)
			}
			ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
			ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.FFNNorm, cfg.RMSNormEps)
			forwardFFN(layer, rs, rs.FFNNorm, cfg, pool)
			if layer.PostFFNNorm != nil {
				ops.RMSNormInPlace(rs.FFNOut, layer.PostFFNNorm, cfg.RMSNormEps)
			}
			ops.Add(rs.X, rs.FFNIn, rs.FFNOut)
		} else if layer.PostAttnNorm != nil {
			// Qwen3.5: post_attn_norm applied to residual, serves as FFN input norm
			ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
			ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.PostAttnNorm, cfg.RMSNormEps)
			forwardFFN(layer, rs, rs.FFNNorm, cfg, pool)
			ops.Add(rs.X, rs.FFNIn, rs.FFNOut)
		} else {
			// Parallel attention+FFN (Phi-2): both use same pre-norm, residual sums both
			forwardFFN(layer, rs, rs.XNorm, cfg, pool)
			for i := 0; i < dim; i++ {
				rs.X[i] = rs.X[i] + rs.AttnProj[i] + rs.FFNOut[i]
			}
		}
	}

	// 3. Final norm: LayerNorm or RMSNorm
	if m.OutputNormBias != nil {
		ops.LayerNorm(rs.X[:dim], rs.X[:dim], m.OutputNorm, m.OutputNormBias, cfg.RMSNormEps)
	} else {
		ops.RMSNormInPlace(rs.X[:dim], m.OutputNorm, cfg.RMSNormEps)
	}

	// 4. Logits — parallelize (largest single matmul: vocabSize rows)
	output := m.Output
	if output == nil {
		output = m.TokenEmbed
	}
	blas.QMatVecMulParallel(rs.Logits, output, rs.X, pool)

	if m.OutputBias != nil {
		ops.AddBias(rs.Logits, m.OutputBias)
	}

	return rs.Logits
}

// forwardAttention runs one full-attention layer. Writes result to rs.AttnProj.
func forwardAttention(
	layer *Layer, rs *RunState, kv *memory.MultiLayerKVCache,
	l, pos, numHeads, numKVHeads, headDim, kvMul int,
	cfg ModelConfig, pool *blas.Pool,
) {
	qDim := numHeads * headDim
	gatedQ := rs.QFull != nil && layer.Wq.Rows > qDim

	if gatedQ {
		// Fused Q+gate projection (Qwen3.5): Wq outputs [Q0,gate0,Q1,gate1,...] interleaved
		blas.QMatVecMulParallel(rs.QFull, layer.Wq, rs.XNorm, pool)
		for h := 0; h < numHeads; h++ {
			copy(rs.Q[h*headDim:(h+1)*headDim], rs.QFull[h*2*headDim:h*2*headDim+headDim])
			copy(rs.QGate[h*headDim:(h+1)*headDim], rs.QFull[h*2*headDim+headDim:(h+1)*2*headDim])
		}
	} else {
		blas.QMatVecMulParallel(rs.Q, layer.Wq, rs.XNorm, pool)
	}
	blas.QMatVecMul(rs.K, layer.Wk, rs.XNorm)
	blas.QMatVecMul(rs.V, layer.Wv, rs.XNorm)

	if layer.Bq != nil {
		ops.AddBias(rs.Q, layer.Bq)
	}
	if layer.Bk != nil {
		ops.AddBias(rs.K, layer.Bk)
	}
	if layer.Bv != nil {
		ops.AddBias(rs.V, layer.Bv)
	}

	if layer.AttnQNorm != nil {
		for h := 0; h < numHeads; h++ {
			ops.RMSNormInPlace(rs.Q[h*headDim:(h+1)*headDim], layer.AttnQNorm, cfg.RMSNormEps)
		}
	}
	if layer.AttnKNorm != nil {
		for h := 0; h < numKVHeads; h++ {
			ops.RMSNormInPlace(rs.K[h*headDim:(h+1)*headDim], layer.AttnKNorm, cfg.RMSNormEps)
		}
	}

	if rs.ropeCos != nil {
		for h := 0; h < numHeads; h++ {
			rs.ApplyRoPEFast(rs.Q[h*headDim:(h+1)*headDim], pos)
		}
		for h := 0; h < numKVHeads; h++ {
			rs.ApplyRoPEFast(rs.K[h*headDim:(h+1)*headDim], pos)
		}
	} else {
		ops.ApplyRoPEBatch(rs.Q, numHeads, rs.K, numKVHeads, pos, headDim, cfg.RopeFreqBase, cfg.RopeNeox)
	}

	kv.Layers[l].Store(pos, rs.K, rs.V)
	seqLen := pos + 1

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	ops.Clear(rs.AttnOut)

	for h := 0; h < numHeads; h++ {
		kvH := h / kvMul
		qHead := rs.Q[h*headDim : (h+1)*headDim]
		headOut := rs.AttnOut[h*headDim : (h+1)*headDim]

		for t := 0; t < seqLen; t++ {
			kHead := kv.Layers[l].Keys[t][kvH*headDim : (kvH+1)*headDim]
			rs.Scores[t] = ops.DotProduct(qHead, kHead, headDim) * scale
		}
		ops.Softmax(rs.Scores[:seqLen])
		for t := 0; t < seqLen; t++ {
			vHead := kv.Layers[l].Vals[t][kvH*headDim : (kvH+1)*headDim]
			ops.AddScaled(headOut, rs.Scores[t], vHead, headDim)
		}
	}

	// Gated attention (Qwen3.5): attn_out *= sigmoid(gate)
	if gatedQ {
		for i := 0; i < qDim; i++ {
			rs.AttnOut[i] *= ops.Sigmoid(rs.QGate[i])
		}
	}

	blas.QMatVecMulParallel(rs.AttnProj, layer.Wo, rs.AttnOut, pool)
	if layer.Bo != nil {
		ops.AddBias(rs.AttnProj, layer.Bo)
	}
}

// forwardFFN runs the feed-forward network for one layer.
// input is the normalized activation to feed into the FFN.
// Result is written to rs.FFNOut.
func forwardFFN(layer *Layer, rs *RunState, input []float32, cfg ModelConfig, pool *blas.Pool) {
	if layer.FFNGate != nil {
		// Gated FFN: SwiGLU or GeGLU
		blas.QDualMatVecMulParallel(rs.Gate, layer.FFNGate, rs.Up, layer.FFNUp, input, pool)
		if cfg.FFNGelu {
			ops.GeGLU(rs.Hidden, rs.Gate, rs.Up, cfg.FFNDim)
		} else {
			ops.SwiGLU(rs.Hidden, rs.Gate, rs.Up, cfg.FFNDim)
		}
		blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Hidden, pool)
	} else {
		// Plain MLP: up → GELU → down (Phi-2)
		blas.QMatVecMulParallel(rs.Up, layer.FFNUp, input, pool)
		if layer.FFNUpBias != nil {
			ops.AddBias(rs.Up, layer.FFNUpBias)
		}
		ops.GELU(rs.Up)
		blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Up, pool)
	}
	if layer.FFNDownBias != nil {
		ops.AddBias(rs.FFNOut, layer.FFNDownBias)
	}
}
