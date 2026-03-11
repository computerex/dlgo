//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"

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
		os.Exit(1)
	}
	cfg := pipe.Model.Config
	m := pipe.Model

	numHeads := cfg.SSMTimeStepRank
	numKVGroups := cfg.SSMGroupCount
	if numKVGroups <= 0 {
		numKVGroups = numHeads
	}
	headKDim := cfg.SSMStateSize
	headVDim := cfg.SSMInnerSize / numHeads
	valueDim := numHeads * headVDim
	keyDim := numKVGroups * headKDim
	qkvDim := keyDim*2 + valueDim

	fmt.Printf("numH=%d numKVG=%d headK=%d headV=%d keyDim=%d valDim=%d qkvDim=%d\n",
		numHeads, numKVGroups, headKDim, headVDim, keyDim, valueDim, qkvDim)

	rs := llm.NewRunState(cfg, 512)
	pool := blas.DefaultPool()
	layer0 := &m.Layers[0]
	ssm := rs.SSMRun
	ssmState := rs.SSMState.Layers[0]

	tokens := pipe.Tokenizer.Encode("The capital of France is")

	for pos, tok := range tokens {
		_ = m.TokenEmbed.DequantizeRow(int(tok), rs.X)
		if cfg.EmbedScale != 0 {
			ops.Scale(rs.X, cfg.EmbedScale)
		}
		ops.RMSNorm(rs.XNorm, rs.X, layer0.AttnNorm, cfg.RMSNormEps)

		blas.QMatVecMulParallel(ssm.QKV, layer0.SSMInProj, rs.XNorm, pool)
		blas.QMatVecMulParallel(ssm.Z, layer0.AttnGate, rs.XNorm, pool)
		blas.QMatVecMul(ssm.Alpha, layer0.SSMAlpha, rs.XNorm)
		blas.QMatVecMul(ssm.Beta, layer0.SSMBeta, rs.XNorm)

		convK := ssmState.ConvK
		buf := ssmState.ConvBuf
		copy(buf[0:(convK-1)*qkvDim], buf[qkvDim:convK*qkvDim])
		copy(buf[(convK-1)*qkvDim:convK*qkvDim], ssm.QKV[:qkvDim])
		w := layer0.SSMConv1dW
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
			if layer0.SSMDtBias != nil {
				a += layer0.SSMDtBias[h]
			}
			softplusA := float32(math.Log(1.0 + math.Exp(float64(a))))
			ssm.Alpha[h] = layer0.SSMA[h] * softplusA
			ssm.Beta[h] = ops.Sigmoid(ssm.Beta[h])
		}

		for g := 0; g < numKVGroups; g++ {
			l2Normalize(q[g*headKDim:(g+1)*headKDim], cfg.RMSNormEps)
			l2Normalize(k[g*headKDim:(g+1)*headKDim], cfg.RMSNormEps)
		}
		qScale := float32(1.0 / math.Sqrt(float64(headKDim)))
		for i := 0; i < keyDim; i++ {
			q[i] *= qScale
		}

		// Replicate Q and K to numHeads (like llama.cpp)
		headsPerGroup := numHeads / numKVGroups
		qRep := make([]float32, numHeads*headKDim)
		kRep := make([]float32, numHeads*headKDim)
		for h := 0; h < numHeads; h++ {
			g := h / headsPerGroup
			copy(qRep[h*headKDim:(h+1)*headKDim], q[g*headKDim:(g+1)*headKDim])
			copy(kRep[h*headKDim:(h+1)*headKDim], k[g*headKDim:(g+1)*headKDim])
		}

		state := ssmState.State
		for h := 0; h < numHeads; h++ {
			decay := float32(math.Exp(float64(ssm.Alpha[h])))
			lr := ssm.Beta[h]
			qH := qRep[h*headKDim : (h+1)*headKDim]
			kH := kRep[h*headKDim : (h+1)*headKDim]
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
			ops.RMSNormInPlace(yH, layer0.SSMNorm, cfg.RMSNormEps)
			for j := 0; j < headVDim; j++ {
				yH[j] *= zH[j] * ops.Sigmoid(zH[j])
			}
		}

		blas.QMatVecMulParallel(rs.AttnProj, layer0.SSMOut, ssm.Y, pool)
		_ = memory.NewMultiLayerKVCache(1, 1, 1)

		if pos == len(tokens)-1 {
			fmt.Printf("\nPos %d (%q) layer 0 SSM output fingerprint:\n", pos, pipe.Tokenizer.DecodeToken(tok))
			fmt.Printf("  Y[:4] = [%.6f, %.6f, %.6f, %.6f]\n", ssm.Y[0], ssm.Y[1], ssm.Y[2], ssm.Y[3])
			fmt.Printf("  AttnProj[:4] = [%.6f, %.6f, %.6f, %.6f]\n", rs.AttnProj[0], rs.AttnProj[1], rs.AttnProj[2], rs.AttnProj[3])

			var stateNorm float32
			for i := 0; i < headKDim*headVDim; i++ {
				stateNorm += state[i] * state[i]
			}
			fmt.Printf("  Head 0 state L2 norm: %.6f\n", math.Sqrt(float64(stateNorm)))
		}
	}

	fmt.Println("\nNow running full model with replicated Q/K delta rule...")

	rs2 := llm.NewRunState(cfg, 512)
	kv2 := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, cfg.NumKVHeads*cfg.HeadDim)
	for i, tok := range tokens {
		llm.Forward(m, tok, i, kv2, rs2)
	}
	top := 0
	for i := 1; i < len(rs2.Logits); i++ {
		if rs2.Logits[i] > rs2.Logits[top] {
			top = i
		}
	}
	fmt.Printf("Standard Forward top: tok=%d %q logit=%.4f\n", top, pipe.Tokenizer.DecodeToken(int32(top)), rs2.Logits[top])
}

func l2Normalize(v []float32, eps float32) {
	var norm float32
	for _, x := range v {
		norm += x * x
	}
	invNorm := float32(1.0 / math.Sqrt(float64(norm) + float64(eps)))
	for i := range v {
		v[i] *= invNorm
	}
}
