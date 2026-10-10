package llm

import (
	"math"
	"testing"
)

// refComputeIMRoPE mirrors ggml compute_imrope_cache (rope_f32.c) for a single
// head of NEOX half-split pairs, producing the cos/sin row for a given
// (t,h,w,e) position quadruple and {T,H,W,E} section counts.
func refComputeIMRoPE(ropeDim int, freqBase float32, pt, ph, pw, pe int, sections [4]int) (cos, sin []float32) {
	half := ropeDim / 2
	cos = make([]float32, half)
	sin = make([]float32, half)
	thetaScale := math.Pow(float64(freqBase), -2.0/float64(ropeDim))
	thetaT := float64(pt)
	thetaH := float64(ph)
	thetaW := float64(pw)
	thetaE := float64(pe)
	sectDims := sections[0] + sections[1] + sections[2] + sections[3]
	for i := 0; i < half; i++ {
		sector := i % sectDims
		var theta float64
		switch {
		case sector%3 == 1 && sector < 3*sections[1]:
			theta = thetaH
		case sector%3 == 2 && sector < 3*sections[2]:
			theta = thetaW
		case sector%3 == 0 && sector < 3*sections[0]:
			theta = thetaT
		default:
			theta = thetaE
		}
		cos[i] = float32(math.Cos(theta))
		sin[i] = float32(math.Sin(theta))
		thetaT *= thetaScale
		thetaH *= thetaScale
		thetaW *= thetaScale
		thetaE *= thetaScale
	}
	return cos, sin
}

func TestIMRoPEMatchesGGML(t *testing.T) {
	const ropeDim = 64
	const headDim = 256
	const freqBase = 1e7
	sections := [4]int{11, 11, 10, 0}

	cases := []struct{ pt, ph, pw, pe int }{
		{0, 0, 0, 0},
		{24, 24, 24, 24}, // text token at pos 24 (all channels equal)
		{0, 0, 0, 0},     // image token t=pos0
		{5, 12, 7, 5},    // image token with distinct axes
		{100, 50, 40, 0},
	}

	for _, tc := range cases {
		rs := &RunState{
			ropeHeadDim: headDim,
			ropeDim:     ropeDim,
		}
		rs.EnableIMRoPE(sections, freqBase)
		rs.SetMRoPEPos(tc.pt, tc.ph, tc.pw, tc.pe)

		// Build an input head with a known value at each element.
		vec := make([]float32, headDim)
		for i := range vec {
			vec[i] = float32(i+1) * 0.01
		}
		orig := append([]float32(nil), vec...)

		rs.applyIMRoPE(vec)

		wantCos, wantSin := refComputeIMRoPE(ropeDim, freqBase, tc.pt, tc.ph, tc.pw, tc.pe, sections)
		half := ropeDim / 2
		var maxErr float64
		for i := 0; i < half; i++ {
			w0 := orig[i]*wantCos[i] - orig[i+half]*wantSin[i]
			w1 := orig[i]*wantSin[i] + orig[i+half]*wantCos[i]
			e0 := math.Abs(float64(vec[i] - w0))
			e1 := math.Abs(float64(vec[i+half] - w1))
			if e0 > maxErr {
				maxErr = e0
			}
			if e1 > maxErr {
				maxErr = e1
			}
		}
		// Beyond ropeDim, values must be untouched.
		for i := ropeDim; i < headDim; i++ {
			if vec[i] != orig[i] {
				t.Fatalf("case %+v: dim %d beyond ropeDim modified", tc, i)
			}
		}
		if maxErr > 1e-5 {
			t.Errorf("case %+v: maxErr=%.3e (want < 1e-5)", tc, maxErr)
		}
	}
}

// TestIMRoPEClearFallsBackToScalar verifies that after ClearMRoPEPos the IMROPE
// path is skipped (mropePos[0] < 0), so scalar RoPE applies.
func TestIMRoPEClearFallsBackToScalar(t *testing.T) {
	rs := &RunState{ropeHeadDim: 64, ropeDim: 64}
	rs.EnableIMRoPE([4]int{11, 11, 10, 0}, 1e7)
	rs.ClearMRoPEPos()
	vec := make([]float32, 64)
	for i := range vec {
		vec[i] = float32(i + 1)
	}
	orig := append([]float32(nil), vec...)
	rs.applyIMRoPE(vec)
	for i := range vec {
		if vec[i] != orig[i] {
			t.Fatalf("applyIMRoPE modified vec when mropePos cleared")
		}
	}
}
