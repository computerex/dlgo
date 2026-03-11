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
	paths := []struct {
		name string
		path string
	}{
		{"9B", `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`},
		{"2B", `C:\projects\gollm\Qwen3.5-2B.Q4_K_M.gguf`},
	}

	for _, p := range paths {
		fmt.Printf("\n========== %s ==========\n", p.name)
		pipe, err := llm.NewPipeline(p.path, 512)
		if err != nil {
			fmt.Printf("Load fail: %v\n", err)
			continue
		}

		cfg := pipe.Model.Config
		m := pipe.Model

		tokens := []int32{760, 6511, 314, 9338, 369}
		kvDim := cfg.NumKVHeads * cfg.HeadDim
		rs := llm.NewRunState(cfg, 512)
		kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)

		dim := cfg.EmbeddingDim
		numHeads := cfg.NumHeads
		numKVHeads := cfg.NumKVHeads
		headDim := cfg.HeadDim
		kvMul := numHeads / numKVHeads
		qDim := numHeads * headDim

		// Process all tokens but instrument layer 3 on the last token
		for tIdx, tok := range tokens {
			pos := tIdx
			_ = m.TokenEmbed.DequantizeRow(int(tok), rs.X)
			if cfg.EmbedScale != 0 {
				ops.Scale(rs.X, cfg.EmbedScale)
			}

			for l := 0; l < cfg.NumLayers; l++ {
				layer := &m.Layers[l]
				spec := &layer.Spec

				ops.RMSNorm(rs.XNorm, rs.X, layer.AttnNorm, cfg.RMSNormEps)

				if spec.Core == 1 { // SSM
					llm.ForwardSSMLayer(layer, rs, rs.SSMRun, rs.SSMState.Layers[l], rs.XNorm, cfg, rs.Pool)
				} else if l == 3 && tIdx == len(tokens)-1 {
					// Instrument attention at layer 3, last token
					fmt.Printf("Layer 3 attention diagnostic (last token, pos=%d):\n", pos)
					fmt.Printf("  XNorm L2: %.4f\n", l2(rs.XNorm))

					// Q projection (GatedQ)
					blas.QMatVecMulParallel(rs.QFull, layer.Wq, rs.XNorm, rs.Pool)
					fmt.Printf("  QFull L2 (after Wq): %.4f\n", l2(rs.QFull))

					for h := 0; h < numHeads; h++ {
						copy(rs.Q[h*headDim:(h+1)*headDim], rs.QFull[h*2*headDim:h*2*headDim+headDim])
						copy(rs.QGate[h*headDim:(h+1)*headDim], rs.QFull[h*2*headDim+headDim:(h+1)*2*headDim])
					}
					fmt.Printf("  Q L2: %.4f  QGate L2: %.4f\n", l2(rs.Q), l2(rs.QGate))

					blas.QMatVecMulParallel(rs.K, layer.Wk, rs.XNorm, rs.Pool)
					blas.QMatVecMulParallel(rs.V, layer.Wv, rs.XNorm, rs.Pool)
					fmt.Printf("  K L2 (after Wk): %.4f  V L2: %.4f\n", l2(rs.K), l2(rs.V))

					// QK Norm
					for h := 0; h < numHeads; h++ {
						ops.RMSNormInPlace(rs.Q[h*headDim:(h+1)*headDim], layer.AttnQNorm, cfg.RMSNormEps)
					}
					for h := 0; h < numKVHeads; h++ {
						ops.RMSNormInPlace(rs.K[h*headDim:(h+1)*headDim], layer.AttnKNorm, cfg.RMSNormEps)
					}
					fmt.Printf("  Q L2 (after QKNorm): %.4f  K L2: %.4f\n", l2(rs.Q), l2(rs.K))

					// RoPE
					for h := 0; h < numHeads; h++ {
						rs.ApplyRoPEFast(rs.Q[h*headDim:(h+1)*headDim], pos)
					}
					for h := 0; h < numKVHeads; h++ {
						rs.ApplyRoPEFast(rs.K[h*headDim:(h+1)*headDim], pos)
					}
					fmt.Printf("  Q L2 (after RoPE): %.4f  K L2: %.4f\n", l2(rs.Q), l2(rs.K))

					kv.Layers[l].Store(pos, rs.K, rs.V)
					seqLen := pos + 1
					scale := float32(1.0 / math.Sqrt(float64(headDim)))
					ops.Clear(rs.AttnOut)

					rs.Pool.ParallelFor(numHeads, func(h int) {
						kvH := h / kvMul
						qHead := rs.Q[h*headDim : (h+1)*headDim]
						headOut := rs.AttnOut[h*headDim : (h+1)*headDim]
						scores := rs.HeadScores[h][:seqLen]
						for t := 0; t < seqLen; t++ {
							kHead := kv.Layers[l].Keys[t][kvH*headDim : (kvH+1)*headDim]
							scores[t] = ops.DotProduct(qHead, kHead, headDim) * scale
						}
						quant.SIMDSoftmax(scores)
						for t := 0; t < seqLen; t++ {
							vHead := kv.Layers[l].Vals[t][kvH*headDim : (kvH+1)*headDim]
							ops.AddScaled(headOut, scores[t], vHead, headDim)
						}
					})
					fmt.Printf("  AttnOut L2 (before gate): %.4f\n", l2(rs.AttnOut))

					// Gate
					var gateSigAvg float64
					for i := 0; i < qDim; i++ {
						g := ops.Sigmoid(rs.QGate[i])
						gateSigAvg += float64(g)
						rs.AttnOut[i] *= g
					}
					gateSigAvg /= float64(qDim)
					fmt.Printf("  Gate sigmoid avg: %.4f\n", gateSigAvg)
					fmt.Printf("  AttnOut L2 (after gate): %.4f\n", l2(rs.AttnOut))

					blas.QMatVecMulParallel(rs.AttnProj, layer.Wo, rs.AttnOut, rs.Pool)
					fmt.Printf("  AttnProj L2 (after Wo): %.4f\n", l2(rs.AttnProj))
				} else {
					llm.ForwardAttention(layer, rs, kv, l, pos, numHeads, numKVHeads, headDim, kvMul, cfg, rs.Pool)
				}

				switch spec.Residual {
				case 1:
					ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
					ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.PostAttnNorm, cfg.RMSNormEps)
					blas.QDualMatVecMulParallel(rs.Gate, layer.FFNGate, rs.Up, layer.FFNUp, rs.FFNNorm, rs.Pool)
					for i := range rs.Gate {
						g := float64(rs.Gate[i])
						rs.Hidden[i] = rs.Up[i] * float32(g/(1.0+math.Exp(-g)))
					}
					blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Hidden, rs.Pool)
					ops.Add(rs.X, rs.FFNIn, rs.FFNOut)
				case 0:
					if layer.PostAttnNorm != nil {
						ops.RMSNormInPlace(rs.AttnProj, layer.PostAttnNorm, cfg.RMSNormEps)
					}
					ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
					ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.FFNNorm, cfg.RMSNormEps)
					blas.QDualMatVecMulParallel(rs.Gate, layer.FFNGate, rs.Up, layer.FFNUp, rs.FFNNorm, rs.Pool)
					quant.SIMDSwiGLU(rs.Hidden, rs.Gate, rs.Up, len(rs.Gate))
					blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Hidden, rs.Pool)
					if layer.PostFFNNorm != nil {
						ops.RMSNormInPlace(rs.FFNOut, layer.PostFFNNorm, cfg.RMSNormEps)
					}
					ops.Add(rs.X, rs.FFNIn, rs.FFNOut)
				}
			}
		}

		// Final norm + logits
		ops.RMSNormInPlace(rs.X[:dim], m.OutputNorm, cfg.RMSNormEps)
		output := m.Output
		if output == nil {
			output = m.TokenEmbed
		}
		blas.QMatVecMulParallel(rs.Logits, output, rs.X, rs.Pool)
		top := argmax(rs.Logits)
		fmt.Printf("Top: tok=%d %q logit=%.4f\n", top, pipe.Tokenizer.DecodeToken(int32(top)), rs.Logits[top])
	}
}

func l2(x []float32) float64 {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum)
}

func argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}
