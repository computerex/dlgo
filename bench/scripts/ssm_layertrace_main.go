//go:build ignore

package main

import (
	"fmt"
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
	"github.com/computerex/dlgo/quant"
)

func main() {
	path := `C:\projects\gollm\Qwen3.5-2B.Q4_K_M.gguf`
	pipe, err := llm.NewPipeline(path, 512)
	if err != nil {
		fmt.Printf("Load fail: %v\n", err)
		return
	}
	cfg := pipe.Model.Config
	m := pipe.Model

	tokens := []int32{760, 6511, 314, 9338, 369} // "The capital of France is" without BOS

	kvDim := cfg.NumKVHeads * cfg.HeadDim
	rs := llm.NewRunState(cfg, 512)
	kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)
	pool := rs.Pool

	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvMul := numHeads / numKVHeads

	for tIdx, tok := range tokens {
		pos := tIdx
		_ = m.TokenEmbed.DequantizeRow(int(tok), rs.X)
		if cfg.EmbedScale != 0 {
			ops.Scale(rs.X, cfg.EmbedScale)
		}

		if tIdx < len(tokens)-1 {
			// Just process normally for all but the last token
			for l := 0; l < cfg.NumLayers; l++ {
				layer := &m.Layers[l]
				spec := &layer.Spec
				switch spec.Norm {
				case llm.NormRMS:
					ops.RMSNorm(rs.XNorm, rs.X, layer.AttnNorm, cfg.RMSNormEps)
				case llm.NormLayer:
					ops.LayerNorm(rs.XNorm, rs.X, layer.AttnNorm, layer.AttnNormBias, cfg.RMSNormEps)
				}
				switch spec.Core {
				case llm.CoreSSM:
					llm.ForwardSSMLayer(layer, rs, rs.SSMRun, rs.SSMState.Layers[l], rs.XNorm, cfg, pool)
				case llm.CoreAttention:
					llm.ForwardAttention(layer, rs, kv, l, pos, numHeads, numKVHeads, headDim, kvMul, cfg, pool)
				}
				switch spec.Residual {
				case llm.ResPostAttnFFN:
					ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
					ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.PostAttnNorm, cfg.RMSNormEps)
					forwardFFN(layer, rs, rs.FFNNorm, pool, cfg)
					ops.Add(rs.X, rs.FFNIn, rs.FFNOut)
				case llm.ResStandard:
					ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
					ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.FFNNorm, cfg.RMSNormEps)
					forwardFFN(layer, rs, rs.FFNNorm, pool, cfg)
					ops.Add(rs.X, rs.FFNIn, rs.FFNOut)
				}
			}
			continue
		}

		// For the LAST token, print per-layer diagnostics
		fmt.Printf("=== Token %d (%q) pos=%d ===\n", tok, pipe.Tokenizer.DecodeToken(tok), pos)
		fmt.Printf("Embed L2: %.4f\n", l2(rs.X))

		for l := 0; l < cfg.NumLayers; l++ {
			layer := &m.Layers[l]
			spec := &layer.Spec

			switch spec.Norm {
			case llm.NormRMS:
				ops.RMSNorm(rs.XNorm, rs.X, layer.AttnNorm, cfg.RMSNormEps)
			case llm.NormLayer:
				ops.LayerNorm(rs.XNorm, rs.X, layer.AttnNorm, layer.AttnNormBias, cfg.RMSNormEps)
			}

			layerType := "SSM"
			switch spec.Core {
			case llm.CoreSSM:
				llm.ForwardSSMLayer(layer, rs, rs.SSMRun, rs.SSMState.Layers[l], rs.XNorm, cfg, pool)
				layerType = "SSM"
			case llm.CoreAttention:
				llm.ForwardAttention(layer, rs, kv, l, pos, numHeads, numKVHeads, headDim, kvMul, cfg, pool)
				layerType = "ATN"
			}

			attnL2 := l2(rs.AttnProj)

			switch spec.Residual {
			case llm.ResPostAttnFFN:
				ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
				ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.PostAttnNorm, cfg.RMSNormEps)
				forwardFFN(layer, rs, rs.FFNNorm, pool, cfg)
				ops.Add(rs.X, rs.FFNIn, rs.FFNOut)
			case llm.ResStandard:
				ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
				ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.FFNNorm, cfg.RMSNormEps)
				forwardFFN(layer, rs, rs.FFNNorm, pool, cfg)
				ops.Add(rs.X, rs.FFNIn, rs.FFNOut)
			}

			xL2 := l2(rs.X)
			ffnL2 := l2(rs.FFNOut)
			fmt.Printf("  L%2d [%s] attn=%.3f ffn=%.3f X=%.3f\n", l, layerType, attnL2, ffnL2, xL2)
		}

		// Final norm + logits
		ops.RMSNormInPlace(rs.X[:dim], m.OutputNorm, cfg.RMSNormEps)
		output := m.Output
		if output == nil {
			output = m.TokenEmbed
		}
		blas.QMatVecMulParallel(rs.Logits, output, rs.X, pool)

		top10 := topK(rs.Logits, 10)
		fmt.Printf("Top-10:\n")
		for _, idx := range top10 {
			fmt.Printf("  tok=%d %q logit=%.4f\n", idx, pipe.Tokenizer.DecodeToken(int32(idx)), rs.Logits[idx])
		}
	}
}

func forwardFFN(layer *llm.Layer, rs *llm.RunState, input []float32, pool *blas.Pool, cfg llm.ModelConfig) {
	if layer.FFNGate != nil {
		if cfg.FFNGelu {
			blas.QDualMatVecMulParallel(rs.Gate, layer.FFNGate, rs.Up, layer.FFNUp, input, pool)
			ops.GeGLU(rs.Hidden, rs.Gate, rs.Up, len(rs.Gate))
		} else {
			blas.QDualMatVecMulParallel(rs.Gate, layer.FFNGate, rs.Up, layer.FFNUp, input, pool)
			quant.SIMDSwiGLU(rs.Hidden, rs.Gate, rs.Up, len(rs.Gate))
		}
		blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Hidden, pool)
	} else {
		blas.QMatVecMulParallel(rs.Up, layer.FFNUp, input, pool)
		ops.GELU(rs.Up)
		blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Up, pool)
	}
}

func l2(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func topK(x []float32, k int) []int {
	indices := make([]int, k)
	for j := 0; j < k; j++ {
		best := -1
		for i := 0; i < len(x); i++ {
			skip := false
			for _, prev := range indices[:j] {
				if i == prev {
					skip = true
					break
				}
			}
			if skip {
				continue
			}
			if best < 0 || x[i] > x[best] {
				best = i
			}
		}
		indices[j] = best
	}
	return indices
}
