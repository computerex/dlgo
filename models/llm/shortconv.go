package llm

import (
	"github.com/computerex/dlgo/blas"
)

// Short convolution core for LFM2 / LFM2MoE.
//
// Reference: llama.cpp src/models/lfm-base.cpp,
// llm_build_lfm_base::build_lfm_shortconv (causal path).
//
// Per layer the block computes, from the pre-attention-normed hidden state h:
//
//	bcx = in_proj @ h                            // [3*dim]
//	b, c, x = bcx[0:dim], bcx[dim:2dim], bcx[2dim:3dim]
//	bx  = b * x
//	y   = C0*bx[t-2] + C1*bx[t-1] + C2*bx[t]     // causal depthwise conv, kernel 3
//	out = out_proj @ (c * y)
//
// The taps C0..C2 are the rows of blk.N.shortconv.conv.weight, which GGUF stores
// as [kernel=3, dim] (kernel is the innermost/contiguous axis). We keep it as a
// flat []float32 indexed channel*kernel + tap.
//
// Recurrent state: the (kernel-1) most recent bx rows. Nothing here is
// position-dependent, so decode and chunked prefill share the same code.

// ShortConvStateLen returns the number of float32 values a layer's conv state
// holds: (kernel-1) previous rows of bx.
func ShortConvStateLen(lcache, dim int) int {
	if lcache <= 1 {
		return 0
	}
	return (lcache - 1) * dim
}

// shortConvStep performs one causal convolution step across a whole [dim] row.
//
//	bx    : b*x for the current token, length dim
//	state : (lcache-1)*dim rolling history, oldest row first; updated in place
//	y     : output, length dim
//
// Taps are applied oldest-first, matching ggml_ssm_conv
// (y[t,ch] = sum_k C[k,ch] * seq[t+k,ch]) and HF's left-padded depthwise Conv1d.
func shortConvStep(y, state, bx, conv []float32, lcache, dim int) {
	dConv := lcache - 1
	switch {
	case lcache <= 1:
		copy(y, bx)
		return
	case lcache == 2:
		for i := 0; i < dim; i++ {
			y[i] = state[i]*conv[i*2] + bx[i]*conv[i*2+1]
		}
	case lcache == 3:
		for i := 0; i < dim; i++ {
			t := conv[i*3 : i*3+3]
			y[i] = state[i]*t[0] + state[dim+i]*t[1] + bx[i]*t[2]
		}
	default:
		for i := 0; i < dim; i++ {
			t := conv[i*lcache : i*lcache+lcache]
			var acc float32
			for k := 0; k < dConv; k++ {
				acc += state[k*dim+i] * t[k]
			}
			y[i] = acc + bx[i]*t[dConv]
		}
	}
	// Shift the history: drop the oldest row, append the current bx.
	if dConv > 1 {
		copy(state[:dConv*dim-dim], state[dim:dConv*dim])
	}
	copy(state[(dConv-1)*dim:dConv*dim], bx)
}

// forwardShortConvLayer runs one short-convolution block on a single token.
// `in` is the pre-normed hidden state; rs.AttnProj receives the result.
func forwardShortConvLayer(l int, layer *Layer, rs *RunState, in []float32, cfg ModelConfig, pool *blas.Pool) {
	dim := cfg.EmbeddingDim
	lcache := cfg.ShortConvLCache
	if lcache <= 1 {
		lcache = 3
	}

	bcx := rs.ShortConvBCX[:3*dim]
	blas.QMatVecMulParallel(bcx, layer.ShortConvInProj, in, pool)

	c := bcx[dim : 2*dim]
	bcxSlice := bcx[2*dim : 3*dim]

	bx := rs.ShortConvBx[:dim]
	for i := 0; i < dim; i++ {
		bx[i] = bcx[i] * bcxSlice[i]
	}

	y := rs.ShortConvY[:dim]
	shortConvStep(y, rs.ShortConvState[l], bx, layer.ShortConvConv, lcache, dim)

	for i := 0; i < dim; i++ {
		y[i] *= c[i]
	}

	blas.QMatVecMulParallel(rs.AttnProj, layer.ShortConvOutProj, y, pool)
}

// shortConvPrefillRow applies conv → gate for one position of a batch.
// bcxRow is the [3*dim] in-projection result (destroyed), yRow receives c*conv_out.
func shortConvPrefillRow(state, conv, bcxRow, yRow []float32, lcache, dim int) {
	b := bcxRow[0:dim]
	c := bcxRow[dim : 2*dim]
	x := bcxRow[2*dim : 3*dim]

	for i := 0; i < dim; i++ {
		b[i] *= x[i]
	}
	shortConvStep(yRow, state, b, conv, lcache, dim)
	for i := 0; i < dim; i++ {
		yRow[i] *= c[i]
	}
}

// resetShortConvState zeroes every shortconv layer's recurrent state.
func resetShortConvState(rs *RunState) {
	for _, s := range rs.ShortConvState {
		for i := range s {
			s[i] = 0
		}
	}
}
