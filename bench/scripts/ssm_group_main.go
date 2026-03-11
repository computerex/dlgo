//go:build ignore

package main

import (
	"fmt"
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	path := `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`
	pipe, err := llm.NewPipeline(path, 512)
	if err != nil {
		fmt.Printf("Load fail: %v\n", err)
		return
	}

	cfg := pipe.Model.Config
	m := pipe.Model
	pool := blas.DefaultPool()

	prompt := "The capital of France is"
	allTokens := pipe.Tokenizer.Encode(prompt)
	tokens := allTokens
	if len(tokens) > 0 && tokens[0] == int32(cfg.BOS) {
		tokens = tokens[1:]
	}
	fmt.Printf("Prompt: %q\nTokens: %v\n\n", prompt, tokens)

	type groupMode struct {
		name string
		fn   func(h, numKVGroups, headsPerGroup int) int
	}
	modes := []groupMode{
		{"contiguous (h/headsPerGroup)", func(h, _, hpg int) int { return h / hpg }},
		{"tiled (h%%numKVGroups)", func(h, nkg, _ int) int { return h % nkg }},
	}

	for _, mode := range modes {
		fmt.Printf("=== %s ===\n", mode.name)

		rs := llm.NewRunState(cfg, 512)
		kv := memory.NewMultiLayerKVCache(cfg.NumLayers, cfg.NumKVHeads*cfg.HeadDim, 512)

		// Process prompt
		for tIdx, tok := range tokens {
			_ = m.TokenEmbed.DequantizeRow(int(tok), rs.X)
			doForward(m, rs, kv, cfg, pool, tIdx, mode.fn)
		}

		top := argmax(rs.Logits)
		fmt.Printf("After prompt → tok=%d %q logit=%.4f\n", top, pipe.Tokenizer.DecodeToken(int32(top)), rs.Logits[top])

		// Generate 15 tokens
		fmt.Printf("Gen: ")
		pos := len(tokens)
		for i := 0; i < 15; i++ {
			best := argmax(rs.Logits)
			tok := int32(best)
			fmt.Printf("%s", pipe.Tokenizer.DecodeToken(tok))
			_ = m.TokenEmbed.DequantizeRow(int(tok), rs.X)
			doForward(m, rs, kv, cfg, pool, pos, mode.fn)
			pos++
		}
		fmt.Printf("\n\n")
	}
}

func doForward(m *llm.Model, rs *llm.RunState, kv *memory.MultiLayerKVCache,
	cfg llm.ModelConfig, pool *blas.Pool, pos int,
	groupFn func(h, numKVGroups, headsPerGroup int) int) {

	dim := cfg.EmbeddingDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	headDim := cfg.HeadDim
	kvMul := numHeads / numKVHeads

	if cfg.EmbedScale != 0 {
		ops.Scale(rs.X, cfg.EmbedScale)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		spec := &layer.Spec

		ops.RMSNorm(rs.XNorm, rs.X, layer.AttnNorm, cfg.RMSNormEps)

		switch spec.Core {
		case llm.CoreSSM:
			ssmGrouped(layer, rs, rs.SSMRun, rs.SSMState.Layers[l], rs.XNorm, cfg, pool, groupFn)
		case llm.CoreAttention:
			llm.ForwardAttention(layer, rs, kv, l, pos, numHeads, numKVHeads, headDim, kvMul, cfg, pool)
		}

		switch spec.Residual {
		case llm.ResStandard:
			if layer.PostAttnNorm != nil {
				ops.RMSNormInPlace(rs.AttnProj, layer.PostAttnNorm, cfg.RMSNormEps)
			}
			ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
			ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.FFNNorm, cfg.RMSNormEps)
			doFFN(layer, rs, rs.FFNNorm, pool)
			if layer.PostFFNNorm != nil {
				ops.RMSNormInPlace(rs.FFNOut, layer.PostFFNNorm, cfg.RMSNormEps)
			}
			ops.Add(rs.X, rs.FFNIn, rs.FFNOut)

		case llm.ResPostAttnFFN:
			ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
			ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.PostAttnNorm, cfg.RMSNormEps)
			doFFN(layer, rs, rs.FFNNorm, pool)
			ops.Add(rs.X, rs.FFNIn, rs.FFNOut)
		}
	}

	ops.RMSNormInPlace(rs.X[:dim], m.OutputNorm, cfg.RMSNormEps)
	out := m.Output
	if out == nil {
		out = m.TokenEmbed
	}
	blas.QMatVecMulParallel(rs.Logits, out, rs.X, pool)
}

func ssmGrouped(
	layer *llm.Layer, rs *llm.RunState, ssm *llm.SSMRunState,
	ssmState *memory.SSMLayerState, xnorm []float32,
	cfg llm.ModelConfig, pool *blas.Pool,
	groupFn func(h, numKVGroups, headsPerGroup int) int,
) {
	qkvDim := ssmState.Channels
	numHeads := ssmState.NumHeads
	numKVGroups := ssmState.NumKVGroups
	if numKVGroups <= 0 {
		numKVGroups = numHeads
	}
	headKDim := ssmState.HeadKDim
	headVDim := ssmState.HeadVDim
	convK := ssmState.ConvK
	valueDim := numHeads * headVDim
	keyDim := numKVGroups * headKDim

	blas.QMatVecMulParallel(ssm.QKV, layer.SSMInProj, xnorm, pool)
	blas.QMatVecMulParallel(ssm.Z, layer.AttnGate, xnorm, pool)
	blas.QMatVecMul(ssm.Alpha, layer.SSMAlpha, xnorm)
	blas.QMatVecMul(ssm.Beta, layer.SSMBeta, xnorm)

	buf := ssmState.ConvBuf
	copy(buf[0:(convK-1)*qkvDim], buf[qkvDim:convK*qkvDim])
	copy(buf[(convK-1)*qkvDim:convK*qkvDim], ssm.QKV[:qkvDim])
	w := layer.SSMConv1dW
	for c := 0; c < qkvDim; c++ {
		var acc float32
		wOff := c * convK
		for k := 0; k < convK; k++ {
			acc += buf[k*qkvDim+c] * w[wOff+k]
		}
		ssm.QKV[c] = acc
	}
	ops.SiLU(ssm.QKV[:qkvDim])

	q := ssm.QKV[:keyDim]
	k := ssm.QKV[keyDim : 2*keyDim]
	v := ssm.QKV[2*keyDim : 2*keyDim+valueDim]

	for h := 0; h < numHeads; h++ {
		a := ssm.Alpha[h]
		if layer.SSMDtBias != nil {
			a += layer.SSMDtBias[h]
		}
		ssm.Alpha[h] = layer.SSMA[h] * float32(math.Log(1.0+math.Exp(float64(a))))
		ssm.Beta[h] = ops.Sigmoid(ssm.Beta[h])
	}

	for g := 0; g < numKVGroups; g++ {
		l2norm(q[g*headKDim:(g+1)*headKDim], cfg.RMSNormEps)
		l2norm(k[g*headKDim:(g+1)*headKDim], cfg.RMSNormEps)
	}
	qScale := float32(1.0 / math.Sqrt(float64(headKDim)))
	for i := 0; i < keyDim; i++ {
		q[i] *= qScale
	}

	headsPerGroup := numHeads / numKVGroups
	state := ssmState.State
	for h := 0; h < numHeads; h++ {
		decay := float32(math.Exp(float64(ssm.Alpha[h])))
		lr := ssm.Beta[h]
		kvGroup := groupFn(h, numKVGroups, headsPerGroup)
		qH := q[kvGroup*headKDim : (kvGroup+1)*headKDim]
		kH := k[kvGroup*headKDim : (kvGroup+1)*headKDim]
		vH := v[h*headVDim : (h+1)*headVDim]
		sOff := h * headKDim * headVDim
		for idx := sOff; idx < sOff+headKDim*headVDim; idx++ {
			state[idx] *= decay
		}
		for j := 0; j < headVDim; j++ {
			var vPred float32
			for i := 0; i < headKDim; i++ {
				vPred += state[sOff+i*headVDim+j] * kH[i]
			}
			delta := vH[j] - vPred
			for i := 0; i < headKDim; i++ {
				state[sOff+i*headVDim+j] += lr * kH[i] * delta
			}
		}
		for j := 0; j < headVDim; j++ {
			var dot float32
			for i := 0; i < headKDim; i++ {
				dot += state[sOff+i*headVDim+j] * qH[i]
			}
			ssm.Y[h*headVDim+j] = dot
		}
	}

	for h := 0; h < numHeads; h++ {
		yH := ssm.Y[h*headVDim : (h+1)*headVDim]
		zH := ssm.Z[h*headVDim : (h+1)*headVDim]
		ops.RMSNormInPlace(yH, layer.SSMNorm, cfg.RMSNormEps)
		for j := 0; j < headVDim; j++ {
			yH[j] *= zH[j] * ops.Sigmoid(zH[j])
		}
	}
	blas.QMatVecMulParallel(rs.AttnProj, layer.SSMOut, ssm.Y, pool)
}

func doFFN(layer *llm.Layer, rs *llm.RunState, input []float32, pool *blas.Pool) {
	switch layer.Spec.FFN {
	case llm.FFNSwiGLU:
		blas.QDualMatVecMulParallel(rs.Gate, layer.FFNGate, rs.Up, layer.FFNUp, input, pool)
		for i := range rs.Gate {
			g := float64(rs.Gate[i])
			rs.Hidden[i] = rs.Up[i] * float32(g/(1.0+math.Exp(-g)))
		}
		blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Hidden, pool)
	}
}

func l2norm(v []float32, eps float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	n := float32(math.Sqrt(sum))
	if n < eps {
		n = eps
	}
	s := 1.0 / n
	for i := range v {
		v[i] *= s
	}
}

func argmax(v []float32) int {
	b := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[b] {
			b = i
		}
	}
	return b
}
