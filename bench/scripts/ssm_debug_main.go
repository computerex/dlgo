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
	fmt.Printf("Config: dim=%d layers=%d heads=%d kvHeads=%d\n",
		cfg.EmbeddingDim, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads)
	fmt.Printf("SSM: inner=%d state=%d timestep=%d conv=%d groupCount=%d\n",
		cfg.SSMInnerSize, cfg.SSMStateSize, cfg.SSMTimeStepRank, cfg.SSMConvKernel, cfg.SSMGroupCount)

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

	fmt.Printf("Computed: numH=%d numKVG=%d headK=%d headV=%d keyDim=%d valDim=%d qkvDim=%d\n",
		numHeads, numKVGroups, headKDim, headVDim, keyDim, valueDim, qkvDim)

	layer0 := &m.Layers[0]
	fmt.Printf("Layer 0 SSMA[:8]: ")
	for i := 0; i < 8 && i < len(layer0.SSMA); i++ {
		fmt.Printf("%.4f ", layer0.SSMA[i])
	}
	fmt.Println()

	if layer0.SSMDtBias != nil {
		fmt.Printf("Layer 0 SSMDtBias[:8]: ")
		for i := 0; i < 8 && i < len(layer0.SSMDtBias); i++ {
			fmt.Printf("%.4f ", layer0.SSMDtBias[i])
		}
		fmt.Println()
	} else {
		fmt.Println("Layer 0 SSMDtBias: nil")
	}

	fmt.Printf("Layer 0 SSMNorm len=%d first4: ", len(layer0.SSMNorm))
	for i := 0; i < 4 && i < len(layer0.SSMNorm); i++ {
		fmt.Printf("%.6f ", layer0.SSMNorm[i])
	}
	fmt.Println()

	tokens := pipe.Tokenizer.Encode("Hello")
	fmt.Printf("Tokens for 'Hello': %v\n", tokens)

	rs := llm.NewRunState(cfg, 512)
	kvCache := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, cfg.NumKVHeads*cfg.HeadDim)
	pool := blas.DefaultPool()

	_ = m.TokenEmbed.DequantizeRow(int(tokens[0]), rs.X)
	if cfg.EmbedScale != 0 {
		ops.Scale(rs.X, cfg.EmbedScale)
	}

	ops.RMSNorm(rs.XNorm, rs.X, layer0.AttnNorm, cfg.RMSNormEps)

	ssm := rs.SSMRun
	ssmState := rs.SSMState.Layers[0]

	blas.QMatVecMulParallel(ssm.QKV, layer0.SSMInProj, rs.XNorm, pool)
	fmt.Printf("After InProj QKV[:8]: ")
	for i := 0; i < 8; i++ {
		fmt.Printf("%.4f ", ssm.QKV[i])
	}
	fmt.Println()

	blas.QMatVecMulParallel(ssm.Z, layer0.AttnGate, rs.XNorm, pool)
	blas.QMatVecMul(ssm.Alpha, layer0.SSMAlpha, rs.XNorm)
	blas.QMatVecMul(ssm.Beta, layer0.SSMBeta, rs.XNorm)

	fmt.Printf("Raw Alpha[:8]: ")
	for i := 0; i < 8 && i < len(ssm.Alpha); i++ {
		fmt.Printf("%.4f ", ssm.Alpha[i])
	}
	fmt.Println()
	fmt.Printf("Raw Beta[:8]: ")
	for i := 0; i < 8 && i < len(ssm.Beta); i++ {
		fmt.Printf("%.4f ", ssm.Beta[i])
	}
	fmt.Println()

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

	fmt.Printf("After Conv+SiLU QKV[:8]: ")
	for i := 0; i < 8; i++ {
		fmt.Printf("%.4f ", ssm.QKV[i])
	}
	fmt.Println()

	q := ssm.QKV[:keyDim]
	k := ssm.QKV[keyDim : 2*keyDim]
	v := ssm.QKV[2*keyDim : 2*keyDim+valueDim]

	fmt.Printf("Q[:4]: ")
	for i := 0; i < 4; i++ {
		fmt.Printf("%.6f ", q[i])
	}
	fmt.Printf("\nK[:4]: ")
	for i := 0; i < 4; i++ {
		fmt.Printf("%.6f ", k[i])
	}
	fmt.Printf("\nV[:4]: ")
	for i := 0; i < 4; i++ {
		fmt.Printf("%.6f ", v[i])
	}
	fmt.Println()

	for h := 0; h < numHeads; h++ {
		a := ssm.Alpha[h]
		if layer0.SSMDtBias != nil {
			a += layer0.SSMDtBias[h]
		}
		softplusA := float32(math.Log(1.0 + math.Exp(float64(a))))
		ssm.Alpha[h] = layer0.SSMA[h] * softplusA
		ssm.Beta[h] = ops.Sigmoid(ssm.Beta[h])
	}

	fmt.Printf("Computed Alpha[:8] (decay gate): ")
	for i := 0; i < 8; i++ {
		fmt.Printf("%.4f ", ssm.Alpha[i])
	}
	fmt.Printf("\nComputed Beta[:8] (lr): ")
	for i := 0; i < 8; i++ {
		fmt.Printf("%.4f ", ssm.Beta[i])
	}
	fmt.Printf("\nDecay[:8] (exp(alpha)): ")
	for i := 0; i < 8; i++ {
		fmt.Printf("%.6f ", math.Exp(float64(ssm.Alpha[i])))
	}
	fmt.Println()

	for g := 0; g < numKVGroups; g++ {
		l2Normalize(q[g*headKDim:(g+1)*headKDim], cfg.RMSNormEps)
		l2Normalize(k[g*headKDim:(g+1)*headKDim], cfg.RMSNormEps)
	}
	qScale := float32(1.0 / math.Sqrt(float64(headKDim)))
	for i := 0; i < keyDim; i++ {
		q[i] *= qScale
	}

	fmt.Printf("After L2Norm+scale Q[:4]: ")
	for i := 0; i < 4; i++ {
		fmt.Printf("%.6f ", q[i])
	}
	fmt.Println()

	headsPerGroup := numHeads / numKVGroups
	state := ssmState.State
	for h := 0; h < numHeads; h++ {
		decay := float32(math.Exp(float64(ssm.Alpha[h])))
		lr := ssm.Beta[h]
		kvGroup := h / headsPerGroup
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

		if h < 2 {
			fmt.Printf("Head %d: Y[:4]=[%.6f %.6f %.6f %.6f]\n", h,
				ssm.Y[h*headVDim], ssm.Y[h*headVDim+1], ssm.Y[h*headVDim+2], ssm.Y[h*headVDim+3])
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

	fmt.Printf("After NormGate Y[:4]: %.6f %.6f %.6f %.6f\n",
		ssm.Y[0], ssm.Y[1], ssm.Y[2], ssm.Y[3])

	blas.QMatVecMulParallel(rs.AttnProj, layer0.SSMOut, ssm.Y, pool)
	fmt.Printf("AttnProj[:4]: %.6f %.6f %.6f %.6f\n",
		rs.AttnProj[0], rs.AttnProj[1], rs.AttnProj[2], rs.AttnProj[3])

	_ = kvCache
	fmt.Println("\nDone. Use these values to compare with llama.cpp for correctness.")
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
