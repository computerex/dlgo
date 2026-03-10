package llm

import (
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/ops"
)

// SSMRunState holds pre-allocated scratch buffers for SSM (Gated Delta Net) layers.
type SSMRunState struct {
	QKV     []float32 // [qkvDim] in-projection output (goes through conv)
	Z       []float32 // [valueDim] gate projection output
	Alpha   []float32 // [numHeads] raw alpha (decay param)
	Beta    []float32 // [numHeads] raw beta (learning rate)
	FusedBA []float32 // [2*numHeads] fused beta+alpha output (interleaved per KV group)
	Y       []float32 // [valueDim] attention/SSM output
}

// ForwardSSMLayer runs one Gated Delta Net layer for single-token autoregressive inference.
//
// Implements the recurrent delta rule with error correction:
//
//	S[h] = exp(g[h]) * S[h]                         // decay
//	v_pred = S^T @ k                                 // predict value from key
//	delta  = v - v_pred                              // error signal
//	S[h]  += sigmoid(beta[h]) * outer(k, delta)      // error-corrected update
//	out[h] = S^T @ (q / sqrt(headKDim))              // scaled output
func ForwardSSMLayer(
	layer *Layer,
	rs *RunState,
	ssm *SSMRunState,
	ssmState *memory.SSMLayerState,
	xnorm []float32,
	cfg ModelConfig,
	pool *blas.Pool,
) []float32 {
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

	// 1. In-projection: dim -> qkvDim
	blas.QMatVecMulParallel(ssm.QKV, layer.SSMInProj, xnorm, pool)

	// 2. Gate projection: dim -> valueDim
	blas.QMatVecMulParallel(ssm.Z, layer.AttnGate, xnorm, pool)

	// 3. Alpha/Beta projections
	if layer.SSMFusedBA != nil {
		// Fused BA projection: output is interleaved per KV group.
		// Layout: [beta_g0_v0, beta_g0_v1, alpha_g0_v0, alpha_g0_v1, beta_g1_v0, ...]
		blas.QMatVecMulParallel(ssm.FusedBA, layer.SSMFusedBA, xnorm, pool)
		vPerGroup := numHeads / numKVGroups
		for g := 0; g < numKVGroups; g++ {
			base := g * vPerGroup * 2
			for vi := 0; vi < vPerGroup; vi++ {
				ssm.Beta[g*vPerGroup+vi] = ssm.FusedBA[base+vi]
				ssm.Alpha[g*vPerGroup+vi] = ssm.FusedBA[base+vPerGroup+vi]
			}
		}
	} else {
		blas.QMatVecMul(ssm.Alpha, layer.SSMAlpha, xnorm)
		blas.QMatVecMul(ssm.Beta, layer.SSMBeta, xnorm)
	}

	// 4. Causal conv1d: shift buffer, store current input, depthwise conv
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

	// 5. SiLU activation
	ops.SiLU(ssm.QKV[:qkvDim])

	// 6. Split into Q, K, V
	q := ssm.QKV[:keyDim]
	k := ssm.QKV[keyDim : 2*keyDim]
	v := ssm.QKV[2*keyDim : 2*keyDim+valueDim]

	// 7. Compute decay (g) and learning rate (beta)
	for h := 0; h < numHeads; h++ {
		a := ssm.Alpha[h]
		if layer.SSMDtBias != nil {
			a += layer.SSMDtBias[h]
		}
		softplusA := float32(math.Log(1.0 + math.Exp(float64(a))))
		ssm.Alpha[h] = layer.SSMA[h] * softplusA

		ssm.Beta[h] = ops.Sigmoid(ssm.Beta[h])
	}

	// 8. L2-normalize Q and K per KV group
	for g := 0; g < numKVGroups; g++ {
		l2Normalize(q[g*headKDim:(g+1)*headKDim], cfg.RMSNormEps)
		l2Normalize(k[g*headKDim:(g+1)*headKDim], cfg.RMSNormEps)
	}

	// 9. Scale Q by 1/sqrt(headKDim) (matches llama.cpp)
	qScale := float32(1.0 / math.Sqrt(float64(headKDim)))
	for i := 0; i < keyDim; i++ {
		q[i] *= qScale
	}

	// 10. Delta rule recurrent step + output (GQA-style: K/Q grouped across V heads)
	// Loop order: outer=i(key), inner=j(value) for sequential cache-friendly access
	state := ssmState.State
	for h := 0; h < numHeads; h++ {
		decay := float32(math.Exp(float64(ssm.Alpha[h])))
		lr := ssm.Beta[h]
		// Tiled V order (Qwen3.5): V head h → K group h%numKVGroups
		// Grouped V order (Qwen3Next): V head h → K group h/(numHeads/numKVGroups)
		kvGroup := h % numKVGroups
		if !cfg.SSMTiledVOrder {
			kvGroup = h / (numHeads / numKVGroups)
		}
		qH := q[kvGroup*headKDim : (kvGroup+1)*headKDim]
		kH := k[kvGroup*headKDim : (kvGroup+1)*headKDim]
		vH := v[h*headVDim : (h+1)*headVDim]
		sOff := h * headKDim * headVDim
		yH := ssm.Y[h*headVDim : (h+1)*headVDim]

		// Decay state
		for idx := sOff; idx < sOff+headKDim*headVDim; idx++ {
			state[idx] *= decay
		}

		// Predict: vPred = S^T @ k (row-major traversal)
		var vPred [256]float32
		for i := 0; i < headKDim; i++ {
			row := state[sOff+i*headVDim : sOff+(i+1)*headVDim]
			ki := kH[i]
			for j := 0; j < headVDim; j++ {
				vPred[j] += row[j] * ki
			}
		}

		// Update: S += lr * outer(k, v - vPred)
		for i := 0; i < headKDim; i++ {
			row := state[sOff+i*headVDim : sOff+(i+1)*headVDim]
			lrk := lr * kH[i]
			for j := 0; j < headVDim; j++ {
				row[j] += lrk * (vH[j] - vPred[j])
			}
		}

		// Output: y = S^T @ q (row-major traversal)
		for j := 0; j < headVDim; j++ {
			yH[j] = 0
		}
		for i := 0; i < headKDim; i++ {
			row := state[sOff+i*headVDim : sOff+(i+1)*headVDim]
			qi := qH[i]
			for j := 0; j < headVDim; j++ {
				yH[j] += row[j] * qi
			}
		}
	}

	// 11. Per-head RMSNorm + SiLU gate
	for h := 0; h < numHeads; h++ {
		yH := ssm.Y[h*headVDim : (h+1)*headVDim]
		zH := ssm.Z[h*headVDim : (h+1)*headVDim]

		ops.RMSNormInPlace(yH, layer.SSMNorm, cfg.RMSNormEps)
		for j := 0; j < headVDim; j++ {
			yH[j] *= zH[j] * ops.Sigmoid(zH[j])
		}
	}

	// 12. Out projection: valueDim -> dim
	blas.QMatVecMulParallel(rs.AttnProj, layer.SSMOut, ssm.Y, pool)

	return rs.AttnProj
}

func l2Normalize(v []float32, eps float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	n := float32(math.Sqrt(sum))
	if n < eps {
		n = eps
	}
	scale := 1.0 / n
	for i := range v {
		v[i] *= scale
	}
}
