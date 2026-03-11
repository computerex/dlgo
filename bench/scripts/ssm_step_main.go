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

		// Process all 5 prompt tokens, then check final prediction
		tokens := []int32{760, 6511, 314, 9338, 369} // "The capital of France is" without BOS
		kvDim := cfg.NumKVHeads * cfg.HeadDim
		rs := llm.NewRunState(cfg, 512)
		kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)

		fmt.Printf("Config: dim=%d layers=%d heads=%d kvHeads=%d headDim=%d\n",
			cfg.EmbeddingDim, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
		fmt.Printf("SSM: ssmHeads=%d groups=%d headK=%d headV=%d\n",
			cfg.SSMTimeStepRank, cfg.SSMGroupCount, cfg.SSMStateSize, cfg.SSMInnerSize/cfg.SSMTimeStepRank)

		for tIdx, tok := range tokens {
			pos := tIdx

			_ = m.TokenEmbed.DequantizeRow(int(tok), rs.X)
			if cfg.EmbedScale != 0 {
				ops.Scale(rs.X, cfg.EmbedScale)
			}

			if tIdx == len(tokens)-1 {
				fmt.Printf("\n--- Last token %q (pos=%d) ---\n", pipe.Tokenizer.DecodeToken(tok), pos)
				fmt.Printf("Embed L2: %.4f first3: [%.6f, %.6f, %.6f]\n", l2(rs.X), rs.X[0], rs.X[1], rs.X[2])
			}

			for l := 0; l < cfg.NumLayers; l++ {
				layer := &m.Layers[l]
				spec := &layer.Spec

				switch spec.Norm {
				case 0: // NormRMS
					ops.RMSNorm(rs.XNorm, rs.X, layer.AttnNorm, cfg.RMSNormEps)
				}

				switch spec.Core {
				case 1: // CoreSSM
					llm.ForwardSSMLayer(layer, rs, rs.SSMRun, rs.SSMState.Layers[l], rs.XNorm, cfg, rs.Pool)
				case 0: // CoreAttention
					llm.ForwardAttention(layer, rs, kv, l, pos,
						cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim,
						cfg.NumHeads/cfg.NumKVHeads, cfg, rs.Pool)
				}

				switch spec.Residual {
				case 1: // ResPostAttnFFN
					ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
					ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.PostAttnNorm, cfg.RMSNormEps)
					forwardFFN(layer, rs, rs.FFNNorm, rs.Pool, cfg)
					ops.Add(rs.X, rs.FFNIn, rs.FFNOut)
				case 0: // ResStandard
					if layer.PostAttnNorm != nil {
						ops.RMSNormInPlace(rs.AttnProj, layer.PostAttnNorm, cfg.RMSNormEps)
					}
					ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
					ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.FFNNorm, cfg.RMSNormEps)
					forwardFFN(layer, rs, rs.FFNNorm, rs.Pool, cfg)
					if layer.PostFFNNorm != nil {
						ops.RMSNormInPlace(rs.FFNOut, layer.PostFFNNorm, cfg.RMSNormEps)
					}
					ops.Add(rs.X, rs.FFNIn, rs.FFNOut)
				}

				if tIdx == len(tokens)-1 && (l < 4 || l >= cfg.NumLayers-2) {
					layerType := "ssm"
					if spec.Core == 0 {
						layerType = "attn"
					}
					fmt.Printf("  L%2d [%4s] attn=%.3f ffn=%.3f X=%.3f\n",
						l, layerType, l2(rs.AttnProj), l2(rs.FFNOut), l2(rs.X))
				}
			}

			if tIdx == len(tokens)-1 {
				// Final norm + logits
				if m.OutputNormBias != nil {
					ops.LayerNorm(rs.X[:cfg.EmbeddingDim], rs.X[:cfg.EmbeddingDim], m.OutputNorm, m.OutputNormBias, cfg.RMSNormEps)
				} else {
					ops.RMSNormInPlace(rs.X[:cfg.EmbeddingDim], m.OutputNorm, cfg.RMSNormEps)
				}
				output := m.Output
				if output == nil {
					output = m.TokenEmbed
				}
				blas.QMatVecMulParallel(rs.Logits, output, rs.X, rs.Pool)
				if m.OutputBias != nil {
					ops.AddBias(rs.Logits, m.OutputBias)
				}

				top := argmax(rs.Logits)
				fmt.Printf("\nTop prediction: tok=%d %q logit=%.4f\n", top, pipe.Tokenizer.DecodeToken(int32(top)), rs.Logits[top])

				// Paris logit
				parisIdx := 11751
				fmt.Printf("Paris logit (tok=%d): %.4f\n", parisIdx, rs.Logits[parisIdx])
			}
		}
	}
}

func forwardFFN(layer *llm.Layer, rs *llm.RunState, input []float32, pool *blas.Pool, cfg llm.ModelConfig) {
	switch layer.Spec.FFN {
	case 0: // FFNSwiGLU
		blas.QDualMatVecMulParallel(rs.Gate, layer.FFNGate, rs.Up, layer.FFNUp, input, pool)
		for i := range rs.Gate {
			g := float64(rs.Gate[i])
			rs.Hidden[i] = rs.Up[i] * float32(g/(1.0+math.Exp(-g)))
		}
		blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Hidden, pool)
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
