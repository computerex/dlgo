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

	// Precomputed RoPE tables (populated by PrecomputeRoPE)
	ropeCos     []float32
	ropeSin     []float32
	ropeHeadDim int
	ropeNeox    bool

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
	rs.PrecomputeRoPE(maxSeqLen, cfg.HeadDim, cfg.HeadDim, cfg.RopeFreqBase)
	rs.SetRopeNeox(cfg.RopeNeox)
	return rs
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

		// Pre-attention RMSNorm
		ops.RMSNorm(rs.XNorm, rs.X, layer.AttnNorm, cfg.RMSNormEps)

		// Q/K/V projections — Q is large, parallelize it; K/V are small
		blas.QMatVecMulParallel(rs.Q, layer.Wq, rs.XNorm, pool)
		blas.QMatVecMul(rs.K, layer.Wk, rs.XNorm)
		blas.QMatVecMul(rs.V, layer.Wv, rs.XNorm)

		// Optional biases (Qwen)
		if layer.Bq != nil {
			ops.AddBias(rs.Q, layer.Bq)
		}
		if layer.Bk != nil {
			ops.AddBias(rs.K, layer.Bk)
		}
		if layer.Bv != nil {
			ops.AddBias(rs.V, layer.Bv)
		}

		// QK norm (Gemma 3) — applied per-head before RoPE
		if layer.AttnQNorm != nil {
			for h := 0; h < numHeads; h++ {
				qHead := rs.Q[h*headDim : (h+1)*headDim]
				ops.RMSNormInPlace(qHead, layer.AttnQNorm, cfg.RMSNormEps)
			}
		}
		if layer.AttnKNorm != nil {
			for h := 0; h < numKVHeads; h++ {
				kHead := rs.K[h*headDim : (h+1)*headDim]
				ops.RMSNormInPlace(kHead, layer.AttnKNorm, cfg.RMSNormEps)
			}
		}

		// RoPE — use precomputed tables when available
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

		// Store K/V in cache
		kv.Layers[l].Store(pos, rs.K, rs.V)
		seqLen := pos + 1

		// Multi-head GQA attention
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

		// Output projection
		blas.QMatVecMulParallel(rs.AttnProj, layer.Wo, rs.AttnOut, pool)

		// Post-attention norm (Gemma 3)
		if layer.PostAttnNorm != nil {
			ops.RMSNormInPlace(rs.AttnProj, layer.PostAttnNorm, cfg.RMSNormEps)
		}

		// Residual connection
		ops.Add(rs.FFNIn, rs.X, rs.AttnProj)

		// Pre-FFN RMSNorm
		ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.FFNNorm, cfg.RMSNormEps)

		// Gate + Up projections — run in parallel (largest matmuls in FFN)
		blas.QDualMatVecMulParallel(rs.Gate, layer.FFNGate, rs.Up, layer.FFNUp, rs.FFNNorm, pool)

		// Gated activation
		if cfg.FFNGelu {
			ops.GeGLU(rs.Hidden, rs.Gate, rs.Up, cfg.FFNDim)
		} else {
			ops.SwiGLU(rs.Hidden, rs.Gate, rs.Up, cfg.FFNDim)
		}

		// Down projection
		blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Hidden, pool)

		// Post-FFN norm (Gemma 3)
		if layer.PostFFNNorm != nil {
			ops.RMSNormInPlace(rs.FFNOut, layer.PostFFNNorm, cfg.RMSNormEps)
		}

		// Residual connection
		ops.Add(rs.X, rs.FFNIn, rs.FFNOut)
	}

	// 3. Final RMSNorm
	ops.RMSNormInPlace(rs.X[:dim], m.OutputNorm, cfg.RMSNormEps)

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
