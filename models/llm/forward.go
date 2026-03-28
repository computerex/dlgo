package llm

import (
	"fmt"
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/ops"
	"github.com/computerex/dlgo/quant"
)

func vecNorm(v []float32) float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return float32(math.Sqrt(s))
}

// RunState holds pre-allocated buffers for inference, avoiding per-token allocations.
type RunState struct {
	X        []float32 // [dim] current activation
	XNorm    []float32 // [dim] normalized activation
	Q        []float32 // [qDim] query projection
	K        []float32 // [kvDim] key projection
	V        []float32 // [kvDim] value projection
	AttnOut  []float32 // [qDim] attention output
	AttnProj []float32 // [dim] output projection
	FFNIn    []float32 // [dim] FFN input (after residual)
	FFNNorm  []float32 // [dim] FFN normalized
	Gate     []float32 // [ffnDim] gate projection
	Up       []float32 // [ffnDim] up projection
	Hidden   []float32 // [ffnDim] gated hidden
	FFNOut   []float32 // [dim] FFN output
	Logits   []float32 // [vocabSize] output logits
	Scores     []float32   // [maxSeqLen] attention scores scratch (legacy)
	HeadScores [][]float32 // [numHeads][maxSeqLen] per-head score buffers for parallel attention

	// Qwen3.5 gated attention: Wq outputs interleaved [Q,gate] per head
	QFull []float32 // [2*qDim] fused Q+gate output (nil for non-gated models)
	QGate []float32 // [qDim] attention gate values (nil for non-gated models)

	// Precomputed RoPE tables (populated by PrecomputeRoPE)
	ropeCos     []float32
	ropeSin     []float32
	ropeHeadDim int
	ropeDim     int // partial RoPE dimension (may be < ropeHeadDim)
	ropeNeox    bool

	// SSM (Gated Delta Net) scratch buffers — nil for pure transformer models
	SSMRun   *SSMRunState
	SSMState *memory.SSMStateCache

	// MoE (Mixture of Experts) scratch buffers — nil for dense models
	MoELogits     []float32   // [expertCount] router logits
	MoEGates      [][]float32 // [nUsed][expertFFNDim] per-expert gate (parallel)
	MoEUps        [][]float32 // [nUsed][expertFFNDim] per-expert up (parallel)
	MoEHiddens    [][]float32 // [nUsed][expertFFNDim] per-expert hidden (parallel)
	MoEExpertOuts [][]float32 // [nUsed][dim] per-expert output (parallel)
	MoEShGate     []float32   // [sharedFFNDim] shared expert gate
	MoEShUp       []float32   // [sharedFFNDim] shared expert up
	MoEShHidden   []float32   // [sharedFFNDim] shared expert hidden
	MoEShOut      []float32   // [dim] shared expert output

	// MLA (Multi-head Latent Attention) scratch buffers
	MLAQComp    []float32 // [qLORARank] compressed Q intermediate
	MLAQAbsorbed []float32 // [numHeads * kvLORARank] absorbed key vectors per head
	MLAAttnKV   []float32 // [numHeads * kvLORARank] weighted KV sum per head

	// Worker pool for parallel matmul
	Pool *blas.Pool
}

// NewRunState allocates all buffers for a model.
func NewRunState(cfg ModelConfig, maxSeqLen int) *RunState {
	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim

	headScores := make([][]float32, cfg.NumHeads)
	for h := 0; h < cfg.NumHeads; h++ {
		headScores[h] = make([]float32, maxSeqLen+1)
	}

	rs := &RunState{
		X:          make([]float32, dim),
		XNorm:      make([]float32, dim),
		Q:          make([]float32, qDim),
		K:          make([]float32, kvDim),
		V:          make([]float32, kvDim),
		AttnOut:    make([]float32, qDim),
		AttnProj:   make([]float32, dim),
		FFNIn:      make([]float32, dim),
		FFNNorm:    make([]float32, dim),
		Gate:       make([]float32, ffnDim),
		Up:         make([]float32, ffnDim),
		Hidden:     make([]float32, ffnDim),
		FFNOut:     make([]float32, dim),
		Logits:     make([]float32, cfg.VocabSize),
		Scores:     make([]float32, maxSeqLen),
		HeadScores: headScores,
		Pool:       blas.DefaultPool(),
	}
	if cfg.RopeScaleType == 2 && cfg.RopeScaleFactor > 0 {
		rs.PrecomputeYaRNRoPE(maxSeqLen, cfg.RopeDim, cfg.HeadDim, cfg.RopeFreqBase,
			cfg.RopeScaleFactor, cfg.RopeOrigMaxPos, cfg.RopeYaRNBetaFast, cfg.RopeYaRNBetaSlow,
			cfg.RopeYaRNExtFactor, cfg.RopeYaRNAttnFactor)
	} else {
		rs.PrecomputeRoPE(maxSeqLen, cfg.RopeDim, cfg.HeadDim, cfg.RopeFreqBase)
	}
	rs.SetRopeNeox(cfg.RopeNeox)

	if cfg.FullAttentionInterval > 0 {
		rs.QFull = make([]float32, 2*qDim)
		rs.QGate = make([]float32, qDim)
	}

	if cfg.ExpertCount > 0 {
		expDim := cfg.ExpertFFNDim
		shDim := cfg.SharedExpertFFNDim
		if shDim == 0 {
			shDim = expDim
		}
		nUsed := cfg.ExpertUsedCount
		rs.MoELogits = make([]float32, cfg.ExpertCount)
		rs.MoEGates = make([][]float32, nUsed)
		rs.MoEUps = make([][]float32, nUsed)
		rs.MoEHiddens = make([][]float32, nUsed)
		rs.MoEExpertOuts = make([][]float32, nUsed)
		for i := 0; i < nUsed; i++ {
			rs.MoEGates[i] = make([]float32, expDim)
			rs.MoEUps[i] = make([]float32, expDim)
			rs.MoEHiddens[i] = make([]float32, expDim)
			rs.MoEExpertOuts[i] = make([]float32, dim)
		}
		rs.MoEShGate = make([]float32, shDim)
		rs.MoEShUp = make([]float32, shDim)
		rs.MoEShHidden = make([]float32, shDim)
		rs.MoEShOut = make([]float32, dim)
	}

	if cfg.QLORARank > 0 {
		rs.MLAQComp = make([]float32, cfg.QLORARank)
		rs.MLAQAbsorbed = make([]float32, cfg.NumHeads*cfg.KVLORARank)
		rs.MLAAttnKV = make([]float32, cfg.NumHeads*cfg.KVLORARank)
		rs.PrecomputeRoPE(maxSeqLen, cfg.QKRopeDim, cfg.QKRopeDim, cfg.RopeFreqBase)
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

		rs.SSMRun = &SSMRunState{
			QKV:     make([]float32, qkvDim),
			Z:       make([]float32, valueDim),
			Alpha:   make([]float32, numHeads),
			Beta:    make([]float32, numHeads),
			FusedBA: make([]float32, 2*numHeads),
			Y:       make([]float32, valueDim),
		}
		rs.SSMState = memory.NewSSMStateCache(
			cfg.NumLayers, numHeads, numKVGroups, headKDim, headVDim,
			qkvDim, cfg.SSMConvKernel,
			func(l int) bool { return isSSMLayer(l, cfg) },
		)
	}

	return rs
}

// DebugForward enables per-layer norm prints for the current forward pass.
var DebugForward bool

// isSSMLayer returns true if layer l uses the SSM/delta-net path instead of attention.
func isSSMLayer(l int, cfg ModelConfig) bool {
	if cfg.FullAttentionInterval <= 0 {
		return false
	}
	return ((l + 1) % cfg.FullAttentionInterval) != 0
}

// Forward performs a single-token forward pass through the model.
func Forward(m *Model, token int32, pos int, kv *memory.MultiLayerKVCache, rs *RunState) []float32 {
	return ForwardRange(m, token, pos, 0, m.Config.NumLayers, kv, rs)
}

// ForwardFromLayer resumes forward pass from a given layer.
// rs.X must already contain the hidden state from the previous layer.
func ForwardFromLayer(m *Model, startLayer, pos int, kv *memory.MultiLayerKVCache, rs *RunState) []float32 {
	return ForwardRange(m, -1, pos, startLayer, m.Config.NumLayers, kv, rs)
}

// ForwardRange performs a forward pass through layers [startLayer, endLayer).
// If startLayer == 0, token embedding is done. If endLayer == numLayers,
// the final norm and output projection are included.
func ForwardRange(m *Model, token int32, pos, startLayer, endLayer int, kv *memory.MultiLayerKVCache, rs *RunState) []float32 {
	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvMul := numHeads / numKVHeads
	pool := rs.Pool

	CPUDiag.Init()
	diagOn := CPUDiag.Enabled

	if startLayer == 0 {
		_ = m.TokenEmbed.DequantizeRow(int(token), rs.X)
		if cfg.EmbedScale != 0 {
			ops.Scale(rs.X, cfg.EmbedScale)
		}
		if diagOn && CPUDiag.Active(-1, pos) {
			CPUDiag.LogSlice(-1, pos, "Embed", rs.X[:dim])
		}
	}

	for l := startLayer; l < endLayer; l++ {
		layer := &m.Layers[l]
		spec := &layer.Spec
		diagL := diagOn && CPUDiag.Active(l, pos)
		if diagOn {
			DiagLayer = l
			DiagPos = pos
		}

		// Pre-norm
		switch spec.Norm {
		case NormRMS:
			ops.RMSNorm(rs.XNorm, rs.X, layer.AttnNorm, cfg.RMSNormEps)
		case NormLayer:
			ops.LayerNorm(rs.XNorm, rs.X, layer.AttnNorm, layer.AttnNormBias, cfg.RMSNormEps)
		}

		if diagL {
			CPUDiag.LogSlice(l, pos, "XNorm", rs.XNorm[:dim])
		}

		// Layer core
		switch spec.Core {
		case CoreSSM:
			ForwardSSMLayer(layer, rs, rs.SSMRun, rs.SSMState.Layers[l], rs.XNorm, cfg, pool)
		case CoreMLA:
			ForwardMLA(layer, rs, kv, l, pos, cfg, pool)
		case CoreAttention:
			ForwardAttention(layer, rs, kv, l, pos, numHeads, numKVHeads, headDim, kvMul, cfg, pool)
		}

		if diagL {
			CPUDiag.LogSlice(l, pos, "AttnProj", rs.AttnProj[:dim])
		}

		// Residual wiring + FFN
		switch spec.Residual {
		case ResStandard:
			if layer.PostAttnNorm != nil {
				ops.RMSNormInPlace(rs.AttnProj, layer.PostAttnNorm, cfg.RMSNormEps)
			}
			ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
			ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.FFNNorm, cfg.RMSNormEps)
			if diagL {
				CPUDiag.LogSlice(l, pos, "FFNNorm", rs.FFNNorm[:dim])
			}
			forwardFFN(layer, rs, rs.FFNNorm, cfg, pool)
			if layer.PostFFNNorm != nil {
				ops.RMSNormInPlace(rs.FFNOut, layer.PostFFNNorm, cfg.RMSNormEps)
			}
			if diagL {
				CPUDiag.LogSlice(l, pos, "FFNOut", rs.FFNOut[:dim])
			}
			ops.Add(rs.X, rs.FFNIn, rs.FFNOut)

		case ResPostAttnFFN:
			ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
			ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.PostAttnNorm, cfg.RMSNormEps)
			if diagL {
				CPUDiag.LogSlice(l, pos, "FFNNorm", rs.FFNNorm[:dim])
			}
			forwardFFN(layer, rs, rs.FFNNorm, cfg, pool)
			if diagL {
				CPUDiag.LogSlice(l, pos, "FFNOut", rs.FFNOut[:dim])
			}
			ops.Add(rs.X, rs.FFNIn, rs.FFNOut)

		case ResParallel:
			forwardFFN(layer, rs, rs.XNorm, cfg, pool)
			if diagL {
				CPUDiag.LogSlice(l, pos, "FFNOut", rs.FFNOut[:dim])
			}
			for i := 0; i < dim; i++ {
				rs.X[i] = rs.X[i] + rs.AttnProj[i] + rs.FFNOut[i]
			}
		}

		if diagL {
			CPUDiag.LogSlice(l, pos, "X(end-of-layer)", rs.X[:dim])
		}

		if DebugForward {
			coreStr := "ATN"
			if spec.Core == CoreSSM { coreStr = "SSM" }
			if spec.Core == CoreMLA { coreStr = "MLA" }
			fmt.Printf("  L%02d [%s] xnorm=%.4f attn=%.4f ffn=%.4f x=%.4f\n",
				l, coreStr, vecNorm(rs.XNorm), vecNorm(rs.AttnProj), vecNorm(rs.FFNOut), vecNorm(rs.X))
		}
	}

	if endLayer < cfg.NumLayers {
		return rs.X
	}

	// Final norm
	if m.OutputNormBias != nil {
		ops.LayerNorm(rs.X[:dim], rs.X[:dim], m.OutputNorm, m.OutputNormBias, cfg.RMSNormEps)
	} else {
		ops.RMSNormInPlace(rs.X[:dim], m.OutputNorm, cfg.RMSNormEps)
	}

	// Logits
	output := m.Output
	if output == nil {
		output = m.TokenEmbed
	}
	blas.QMatVecMulParallel(rs.Logits, output, rs.X, pool)

	if m.OutputBias != nil {
		ops.AddBias(rs.Logits, m.OutputBias)
	}

	if cfg.FinalLogitSoftcap > 0 {
		cap := cfg.FinalLogitSoftcap
		for i := range rs.Logits {
			rs.Logits[i] = cap * float32(math.Tanh(float64(rs.Logits[i]/cap)))
		}
	}

	return rs.Logits
}

// forwardAttention runs one full-attention layer. Writes result to rs.AttnProj.
func ForwardAttention(
	layer *Layer, rs *RunState, kv *memory.MultiLayerKVCache,
	l, pos, numHeads, numKVHeads, headDim, kvMul int,
	cfg ModelConfig, pool *blas.Pool,
) {
	qDim := numHeads * headDim

	// Q/K/V projections — fused into single dispatch when possible
	if layer.Spec.GatedQ {
		blas.QMatVecMulParallel(rs.QFull, layer.Wq, rs.XNorm, pool)
		for h := 0; h < numHeads; h++ {
			copy(rs.Q[h*headDim:(h+1)*headDim], rs.QFull[h*2*headDim:h*2*headDim+headDim])
			copy(rs.QGate[h*headDim:(h+1)*headDim], rs.QFull[h*2*headDim+headDim:(h+1)*2*headDim])
		}
		blas.QMatVecMulParallel(rs.K, layer.Wk, rs.XNorm, pool)
		blas.QMatVecMulParallel(rs.V, layer.Wv, rs.XNorm, pool)
	} else {
		blas.QTripleMatVecMulParallel(rs.Q, layer.Wq, rs.K, layer.Wk, rs.V, layer.Wv, rs.XNorm, pool)
	}

	if layer.Bq != nil {
		ops.AddBias(rs.Q, layer.Bq)
	}
	if layer.Bk != nil {
		ops.AddBias(rs.K, layer.Bk)
	}
	if layer.Bv != nil {
		ops.AddBias(rs.V, layer.Bv)
	}

	kvDim := numKVHeads * headDim
	diagL := CPUDiag.Active(l, pos)
	if diagL {
		CPUDiag.LogSlice(l, pos, "Q+bias", rs.Q[:numHeads*headDim])
		CPUDiag.LogSlice(l, pos, "K+bias", rs.K[:kvDim])
		CPUDiag.LogSlice(l, pos, "V+bias", rs.V[:kvDim])
	}

	if layer.Spec.QKNorm {
		for h := 0; h < numHeads; h++ {
			ops.RMSNormInPlace(rs.Q[h*headDim:(h+1)*headDim], layer.AttnQNorm, cfg.RMSNormEps)
		}
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

	if diagL {
		CPUDiag.LogSlice(l, pos, "Q post-RoPE", rs.Q[:numHeads*headDim])
		CPUDiag.LogSlice(l, pos, "K post-RoPE", rs.K[:kvDim])
	}

	kv.Layers[l].Store(pos, rs.K, rs.V)
	seqLen := pos + 1

	// Sliding window: limit attention span
	startPos := 0
	winSize := layer.Spec.SlidingWindow
	hasSinks := layer.AttnSinks != nil
	if winSize > 0 && seqLen > winSize {
		startPos = seqLen - winSize
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	ops.Clear(rs.AttnOut)

	pool.ParallelFor(numHeads, func(h int) {
		kvH := h / kvMul
		qHead := rs.Q[h*headDim : (h+1)*headDim]
		headOut := rs.AttnOut[h*headDim : (h+1)*headDim]

		windowLen := seqLen - startPos
		// Attention sinks: append an extra learnable logit to the softmax
		// (see "Attention Is Off By One" / OpenAI gpt-oss reference).
		// After softmax the extra sink weight is discarded, so real
		// attention weights sum to < 1, acting as a "trash bin" for
		// unneeded attention mass.
		extraSink := 0
		if hasSinks {
			extraSink = 1
		}
		totalLen := windowLen + extraSink
		scores := rs.HeadScores[h][:totalLen]

		// Score window positions
		for i, t := 0, startPos; t < seqLen; t++ {
			kHead := kv.Layers[l].Keys[t][kvH*headDim : (kvH+1)*headDim]
			scores[i] = ops.DotProduct(qHead, kHead, headDim) * scale
			i++
		}
		// Append learned sink logit as extra softmax column
		if hasSinks {
			scores[windowLen] = layer.AttnSinks[h]
		}
		// Attention logit soft-capping (Gemma 2)
		if cfg.AttnLogitSoftcap > 0 {
			cap := cfg.AttnLogitSoftcap
			for i := range scores {
				scores[i] = cap * float32(math.Tanh(float64(scores[i]/cap)))
			}
		}
		quant.SIMDSoftmax(scores)

		// Accumulate weighted values (skip the sink column at the end)
		for i, t := 0, startPos; t < seqLen; t++ {
			vHead := kv.Layers[l].Vals[t][kvH*headDim : (kvH+1)*headDim]
			ops.AddScaled(headOut, scores[i], vHead, headDim)
			i++
		}
	})

	if diagL {
		CPUDiag.LogSlice(l, pos, "AttnOut", rs.AttnOut[:numHeads*headDim])
	}

	if layer.Spec.GatedQ {
		for i := 0; i < qDim; i++ {
			rs.AttnOut[i] *= ops.Sigmoid(rs.QGate[i])
		}
	}

	blas.QMatVecMulParallel(rs.AttnProj, layer.Wo, rs.AttnOut, pool)
	if layer.Bo != nil {
		ops.AddBias(rs.AttnProj, layer.Bo)
	}
}

// ForwardMLA runs one Multi-head Latent Attention layer (DeepSeek-V2/GLM-4).
// Uses absorbed key/value approach to avoid expanding K for every cached position.
func ForwardMLA(
	layer *Layer, rs *RunState, kv *memory.MultiLayerKVCache,
	l, pos int, cfg ModelConfig, pool *blas.Pool,
) {
	numHeads := cfg.NumHeads
	qkNope := cfg.QKNopeDim
	qkRope := cfg.QKRopeDim
	kvLORARank := cfg.KVLORARank
	vHeadDim := cfg.VHeadDim
	qPerHead := qkNope + qkRope
	kvCompDim := kvLORARank + qkRope

	// === Q path: x → WqA → norm → WqB → Q ===
	blas.QMatVecMulParallel(rs.MLAQComp, layer.WqA, rs.XNorm, pool)
	ops.RMSNormInPlace(rs.MLAQComp, layer.WqANorm, cfg.RMSNormEps)

	mlaQDim := numHeads * qPerHead
	qFull := rs.Q[:mlaQDim]
	blas.QMatVecMulParallel(qFull, layer.WqB, rs.MLAQComp, pool)

	// Apply RoPE to rope portion of each head's Q
	for h := 0; h < numHeads; h++ {
		qRope := qFull[h*qPerHead+qkNope : (h+1)*qPerHead]
		rs.ApplyRoPEFastDim(qRope, pos, qkRope)
	}

	// === KV path: x → WkvA → split → norm → store ===
	kvFull := rs.K[:kvCompDim]
	blas.QMatVecMulParallel(kvFull, layer.WkvA, rs.XNorm, pool)

	kvComp := kvFull[:kvLORARank]
	kRope := kvFull[kvLORARank:]

	ops.RMSNormInPlace(kvComp, layer.WkvANorm, cfg.RMSNormEps)
	rs.ApplyRoPEFastDim(kRope, pos, qkRope)

	// Store compressed KV in cache K slot
	kv.Layers[l].Store(pos, kvFull[:kvCompDim], kvFull[:kvCompDim])

	seqLen := pos + 1
	scale := float32(1.0 / math.Sqrt(float64(qPerHead)))

	// === Absorbed attention (per head) ===
	pool.ParallelFor(numHeads, func(h int) {
		qNope := qFull[h*qPerHead : h*qPerHead+qkNope]
		qRopeH := qFull[h*qPerHead+qkNope : (h+1)*qPerHead]

		// Absorbed key: q_absorbed = WkB_h @ q_nope → [kvLORARank]
		wkbH := mlaHeadView(layer.WkB, h, kvLORARank, qkNope)
		qAbsorbed := rs.MLAQAbsorbed[h*kvLORARank : (h+1)*kvLORARank]
		blas.QMatVecMul(qAbsorbed, wkbH, qNope)

		scores := rs.HeadScores[h][:seqLen]

		// Score computation
		for t := 0; t < seqLen; t++ {
			cached := kv.Layers[l].Keys[t]
			kvCompT := cached[:kvLORARank]
			kRopeT := cached[kvLORARank:kvCompDim]

			scoreNope := ops.DotProduct(qAbsorbed, kvCompT, kvLORARank)
			scoreRope := ops.DotProduct(qRopeH, kRopeT, qkRope)
			scores[t] = (scoreNope + scoreRope) * scale
		}
		quant.SIMDSoftmax(scores)

		// Weighted sum of compressed KV
		attnKV := rs.MLAAttnKV[h*kvLORARank : (h+1)*kvLORARank]
		ops.Clear(attnKV)
		for t := 0; t < seqLen; t++ {
			cached := kv.Layers[l].Keys[t]
			kvCompT := cached[:kvLORARank]
			ops.AddScaled(attnKV, scores[t], kvCompT, kvLORARank)
		}

		// Expand V: v_h = WvB_h @ attn_kv → [vHeadDim]
		wvbH := mlaHeadView(layer.WvB, h, vHeadDim, kvLORARank)
		headOut := rs.AttnOut[h*vHeadDim : (h+1)*vHeadDim]
		blas.QMatVecMul(headOut, wvbH, attnKV)
	})

	// Output projection
	blas.QMatVecMulParallel(rs.AttnProj, layer.Wo, rs.AttnOut[:numHeads*vHeadDim], pool)
}

// mlaHeadView returns a zero-copy view into a 3D packed per-head tensor.
// The 3D tensor is stored as [numHeads][rowsPerHead][colsPerHead] in memory.
func mlaHeadView(packed *core.QuantizedTensor, headIdx, rowsPerHead, colsPerHead int) *core.QuantizedTensor {
	bytesPerRow := quant.BytesForN(packed.Type, colsPerHead)
	offset := headIdx * rowsPerHead * bytesPerRow
	size := rowsPerHead * bytesPerRow
	return &core.QuantizedTensor{
		Data: packed.Data[offset : offset+size],
		Type: packed.Type,
		Rows: rowsPerHead,
		Cols: colsPerHead,
	}
}

// expertView returns a zero-copy view into a packed expert tensor for one expert.
func expertView(packed *core.QuantizedTensor, expertIdx, expertRows int) *core.QuantizedTensor {
	bytesPerRow := quant.BytesForN(packed.Type, packed.Cols)
	offset := expertIdx * expertRows * bytesPerRow
	size := expertRows * bytesPerRow
	return &core.QuantizedTensor{
		Data: packed.Data[offset : offset+size],
		Type: packed.Type,
		Rows: expertRows,
		Cols: packed.Cols,
	}
}

// topKIndices returns the indices and values of the K largest elements.
func topKIndices(logits []float32, k int) ([]int, []float32) {
	n := len(logits)
	if k > n {
		k = n
	}
	indices := make([]int, k)
	values := make([]float32, k)
	for i := 0; i < k; i++ {
		indices[i] = -1
		values[i] = -math.MaxFloat32
	}
	for i, v := range logits {
		minIdx := 0
		for j := 1; j < k; j++ {
			if values[j] < values[minIdx] {
				minIdx = j
			}
		}
		if v > values[minIdx] {
			values[minIdx] = v
			indices[minIdx] = i
		}
	}
	return indices, values
}

// ForwardMoEFFNDispatch routes to the correct MoE implementation based on FFN type.
func ForwardMoEFFNDispatch(layer *Layer, rs *RunState, input []float32, cfg ModelConfig, pool *blas.Pool) {
	if layer.Spec.FFN == FFNMoESwiOAI {
		ForwardMoEFFN_OAI(layer, rs, input, cfg, pool)
	} else {
		ForwardMoEFFN(layer, rs, input, cfg, pool)
	}
}

// ForwardMoEFFN runs the Mixture-of-Experts FFN. Result written to rs.FFNOut.
func ForwardMoEFFN(layer *Layer, rs *RunState, input []float32, cfg ModelConfig, pool *blas.Pool) {
	dim := cfg.EmbeddingDim
	expDim := cfg.ExpertFFNDim
	nUsed := cfg.ExpertUsedCount

	// Router: compute gated probabilities over all experts
	nExperts := cfg.ExpertCount
	blas.QMatVecMulParallel(rs.MoELogits[:nExperts], layer.FFNRouter, input, pool)
	if layer.FFNRouterBias != nil {
		ops.AddBias(rs.MoELogits[:nExperts], layer.FFNRouterBias)
	}

	var indices []int
	var weights []float32
	switch cfg.ExpertGatingFunc {
	case 2:
		for i := 0; i < nExperts; i++ {
			rs.MoELogits[i] = ops.Sigmoid(rs.MoELogits[i])
		}
		indices, weights = topKIndices(rs.MoELogits[:nExperts], nUsed)
	case 3:
		indices, weights = topKIndices(rs.MoELogits[:nExperts], nUsed)
		quant.SIMDSoftmax(weights)
	default:
		quant.SIMDSoftmax(rs.MoELogits[:nExperts])
		indices, weights = topKIndices(rs.MoELogits[:nExperts], nUsed)
	}

	if CPUDiag.Active(DiagLayer, DiagPos) {
		CPUDiag.LogMoE(DiagLayer, DiagPos, rs.MoELogits[:nExperts], indices, weights, cfg.ExpertGatingFunc)
	}

	// Normalize selected weights
	if cfg.ExpertWeightsNorm {
		var wSum float32
		for _, w := range weights {
			wSum += w
		}
		if wSum < 1e-12 {
			wSum = 1e-12
		}
		invSum := 1.0 / wSum
		for i := range weights {
			weights[i] *= invSum
		}
	}
	if cfg.ExpertWeightsScale > 0 {
		for i := range weights {
			weights[i] *= cfg.ExpertWeightsScale
		}
	}

	// Build expert slices for batched dispatch
	downBpr := quant.BytesForN(layer.FFNDownExps.Type, layer.FFNDownExps.Cols)

	gateSlices := make([]blas.ExpertSlice, 0, nUsed)
	upSlices := make([]blas.ExpertSlice, 0, nUsed)
	activeExperts := make([]int, 0, nUsed)

	fused := layer.FFNGateUpExps != nil
	var gateTensor, upTensor *core.QuantizedTensor
	if fused {
		gateTensor = layer.FFNGateUpExps
		upTensor = layer.FFNGateUpExps
	} else {
		gateTensor = layer.FFNGateExps
		upTensor = layer.FFNUpExps
	}
	bpr := quant.BytesForN(gateTensor.Type, gateTensor.Cols)

	for e := 0; e < nUsed; e++ {
		idx := indices[e]
		if idx < 0 {
			continue
		}
		activeExperts = append(activeExperts, e)
		if fused {
			// Fused layout: per expert [gate_rows, up_rows] interleaved
			gateSlices = append(gateSlices, blas.ExpertSlice{
				Out: rs.MoEGates[e], Rows: expDim, Off: idx * 2 * expDim * bpr,
			})
			upSlices = append(upSlices, blas.ExpertSlice{
				Out: rs.MoEUps[e], Rows: expDim, Off: (idx*2 + 1) * expDim * bpr,
			})
		} else {
			gateSlices = append(gateSlices, blas.ExpertSlice{
				Out: rs.MoEGates[e], Rows: expDim, Off: idx * expDim * bpr,
			})
			upSlices = append(upSlices, blas.ExpertSlice{
				Out: rs.MoEUps[e], Rows: expDim, Off: idx * expDim * bpr,
			})
		}
	}

	// Single dispatch: all experts' gate+up projections
	blas.QDualMultiExpertMatVec(gateTensor, upTensor,
		gateSlices, upSlices, input, pool)

	// SwiGLU for all active experts
	for _, e := range activeExperts {
		quant.SIMDSwiGLU(rs.MoEHiddens[e], rs.MoEGates[e], rs.MoEUps[e], expDim)
	}

	// Down projections: dispatch all expert rows (nActive * dim) across all workers
	// for full core utilization. Each row uses its expert's hidden vector as input.
	ops.Clear(rs.FFNOut)
	nActive := len(activeExperts)
	totalDownRows := nActive * dim
	useFusedDown := quant.HasSIMDDot(layer.FFNDownExps.Type)
	downCols := layer.FFNDownExps.Cols
	pool.DispatchChunked(totalDownRows, pool.NumWorkers(), func(_, start, end int) {
		for row := start; row < end; {
			ei := row / dim
			rowInExpert := row % dim
			e := activeExperts[ei]
			idx := indices[e]
			endInExpert := end - ei*dim
			if endInExpert > dim {
				endInExpert = dim
			}
			nrows := endInExpert - rowInExpert
			downOff := idx*dim*downBpr + rowInExpert*downBpr
			out := rs.MoEExpertOuts[e]
			if useFusedDown {
				quant.SIMDDotBatch(
					layer.FFNDownExps.Data[downOff:downOff+nrows*downBpr],
					layer.FFNDownExps.Type, rs.MoEHiddens[e],
					downCols, out[rowInExpert:endInExpert], nrows, downBpr)
			} else {
				buf := make([]float32, downCols)
				for r := rowInExpert; r < endInExpert; r++ {
					rOff := idx*dim*downBpr + r*downBpr
					quant.DequantizeInto(buf, layer.FFNDownExps.Data[rOff:rOff+downBpr], layer.FFNDownExps.Type, downCols)
					out[r] = quant.SIMDDotF32(buf, rs.MoEHiddens[e], downCols)
				}
			}
			row = (ei + 1) * dim
		}
	})

	// Accumulate weighted expert outputs
	for _, e := range activeExperts {
		w := weights[e]
		out := rs.MoEExpertOuts[e]
		for i := 0; i < dim; i++ {
			rs.FFNOut[i] += w * out[i]
		}
	}

	// Shared expert (uses pool since it's larger)
	if layer.FFNGateShared != nil {
		shDim := layer.FFNGateShared.Rows
		blas.QDualMatVecMulParallel(rs.MoEShGate, layer.FFNGateShared, rs.MoEShUp, layer.FFNUpShared, input, pool)
		quant.SIMDSwiGLU(rs.MoEShHidden, rs.MoEShGate, rs.MoEShUp, shDim)
		blas.QMatVecMulParallel(rs.MoEShOut, layer.FFNDownShared, rs.MoEShHidden, pool)

		if layer.FFNRouterShared != nil {
			gate := ops.Sigmoid(ops.DotProduct(layer.FFNRouterShared, input, dim))
			for i := 0; i < dim; i++ {
				rs.FFNOut[i] += gate * rs.MoEShOut[i]
			}
		} else {
			for i := 0; i < dim; i++ {
				rs.FFNOut[i] += rs.MoEShOut[i]
			}
		}
	}
}

// ForwardMoEFFN_OAI runs the gpt-oss MoE with SOFTMAX_WEIGHT gating and SwiGLU_OAI activation.
func ForwardMoEFFN_OAI(layer *Layer, rs *RunState, input []float32, cfg ModelConfig, pool *blas.Pool) {
	dim := cfg.EmbeddingDim
	expDim := cfg.ExpertFFNDim
	nUsed := cfg.ExpertUsedCount
	nExperts := cfg.ExpertCount

	// Router logits
	blas.QMatVecMulParallel(rs.MoELogits[:nExperts], layer.FFNRouter, input, pool)
	if layer.FFNRouterBias != nil {
		ops.AddBias(rs.MoELogits[:nExperts], layer.FFNRouterBias)
	}

	// SOFTMAX_WEIGHT: top-k on RAW logits, then softmax only on selected
	indices, rawWeights := topKIndices(rs.MoELogits[:nExperts], nUsed)
	quant.SIMDSoftmax(rawWeights)

	if CPUDiag.Active(DiagLayer, DiagPos) {
		CPUDiag.LogMoE(DiagLayer, DiagPos, rs.MoELogits[:nExperts], indices, rawWeights, 3)
	}

	if cfg.ExpertWeightsScale > 0 && cfg.ExpertWeightsScale != 1.0 {
		for i := range rawWeights {
			rawWeights[i] *= cfg.ExpertWeightsScale
		}
	}

	gateBpr := quant.BytesForN(layer.FFNGateExps.Type, layer.FFNGateExps.Cols)
	upBpr := quant.BytesForN(layer.FFNUpExps.Type, layer.FFNUpExps.Cols)
	downBpr := quant.BytesForN(layer.FFNDownExps.Type, layer.FFNDownExps.Cols)

	gateSlices := make([]blas.ExpertSlice, 0, nUsed)
	upSlices := make([]blas.ExpertSlice, 0, nUsed)
	activeExperts := make([]int, 0, nUsed)

	for e := 0; e < nUsed; e++ {
		idx := indices[e]
		if idx < 0 {
			continue
		}
		activeExperts = append(activeExperts, e)
		gateSlices = append(gateSlices, blas.ExpertSlice{
			Out: rs.MoEGates[e], Rows: expDim, Off: idx * expDim * gateBpr,
		})
		upSlices = append(upSlices, blas.ExpertSlice{
			Out: rs.MoEUps[e], Rows: expDim, Off: idx * expDim * upBpr,
		})
	}

	blas.QDualMultiExpertMatVec(layer.FFNGateExps, layer.FFNUpExps,
		gateSlices, upSlices, input, pool)

	// Add expert biases and apply SwiGLU_OAI activation
	for _, e := range activeExperts {
		idx := indices[e]
		if layer.FFNGateExpsBias != nil {
			biasOff := idx * expDim
			for i := 0; i < expDim; i++ {
				rs.MoEGates[e][i] += layer.FFNGateExpsBias[biasOff+i]
			}
		}
		if layer.FFNUpExpsBias != nil {
			biasOff := idx * expDim
			for i := 0; i < expDim; i++ {
				rs.MoEUps[e][i] += layer.FFNUpExpsBias[biasOff+i]
			}
		}
		ops.SwiGLU_OAI(rs.MoEHiddens[e], rs.MoEGates[e], rs.MoEUps[e], expDim, 1.702, 7.0)
	}

	// Down projections
	ops.Clear(rs.FFNOut)
	nActive := len(activeExperts)
	totalDownRows := nActive * dim
	useFusedDown := quant.HasSIMDDot(layer.FFNDownExps.Type)
	downCols := layer.FFNDownExps.Cols
	pool.DispatchChunked(totalDownRows, pool.NumWorkers(), func(_, start, end int) {
		for row := start; row < end; {
			ei := row / dim
			rowInExpert := row % dim
			e := activeExperts[ei]
			idx := indices[e]
			endInExpert := end - ei*dim
			if endInExpert > dim {
				endInExpert = dim
			}
			nrows := endInExpert - rowInExpert
			downOff := idx*dim*downBpr + rowInExpert*downBpr
			out := rs.MoEExpertOuts[e]
			if useFusedDown {
				quant.SIMDDotBatch(
					layer.FFNDownExps.Data[downOff:downOff+nrows*downBpr],
					layer.FFNDownExps.Type, rs.MoEHiddens[e],
					downCols, out[rowInExpert:endInExpert], nrows, downBpr)
			} else {
				buf := make([]float32, downCols)
				for r := rowInExpert; r < endInExpert; r++ {
					rOff := idx*dim*downBpr + r*downBpr
					quant.DequantizeInto(buf, layer.FFNDownExps.Data[rOff:rOff+downBpr], layer.FFNDownExps.Type, downCols)
					out[r] = quant.SIMDDotF32(buf, rs.MoEHiddens[e], downCols)
				}
			}
			row = (ei + 1) * dim
		}
	})

	// Add down biases and accumulate weighted expert outputs
	for _, e := range activeExperts {
		w := rawWeights[e]
		out := rs.MoEExpertOuts[e]
		if layer.FFNDownExpsBias != nil {
			idx := indices[e]
			biasOff := idx * dim
			for i := 0; i < dim; i++ {
				rs.FFNOut[i] += w * (out[i] + layer.FFNDownExpsBias[biasOff+i])
			}
		} else {
			for i := 0; i < dim; i++ {
				rs.FFNOut[i] += w * out[i]
			}
		}
	}
}

// forwardFFN runs the feed-forward network. Result written to rs.FFNOut.
func forwardFFN(layer *Layer, rs *RunState, input []float32, cfg ModelConfig, pool *blas.Pool) {
	switch layer.Spec.FFN {
	case FFNSwiGLU:
		blas.QDualMatVecMulParallel(rs.Gate, layer.FFNGate, rs.Up, layer.FFNUp, input, pool)
		quant.SIMDSwiGLU(rs.Hidden, rs.Gate, rs.Up, len(rs.Gate))
		blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Hidden, pool)

	case FFNGeGLU:
		blas.QDualMatVecMulParallel(rs.Gate, layer.FFNGate, rs.Up, layer.FFNUp, input, pool)
		ops.GeGLU(rs.Hidden, rs.Gate, rs.Up, len(rs.Gate))
		blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Hidden, pool)

	case FFNPlain:
		blas.QMatVecMulParallel(rs.Up, layer.FFNUp, input, pool)
		if layer.FFNUpBias != nil {
			ops.AddBias(rs.Up, layer.FFNUpBias)
		}
		ops.GELU(rs.Up)
		blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Up, pool)

	case FFNMoE:
		ForwardMoEFFN(layer, rs, input, cfg, pool)
		return
	case FFNMoESwiOAI:
		ForwardMoEFFN_OAI(layer, rs, input, cfg, pool)
		return
	}

	if layer.FFNDownBias != nil {
		ops.AddBias(rs.FFNOut, layer.FFNDownBias)
	}
}
