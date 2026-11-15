//go:build cgo && vulkan

package cuda

import (
	"math"
	"testing"
)

// TestQuantizeXQ8Parity checks the int8 activation quantiser that the int8 GEMM
// path needs: per-32-element scale, int8 codes, and the group sum of the
// reconstructed values.
//
// The group sum is the load-bearing part. An Affine4 weight block reconstructs
// w = d*q + m with integer q in [0,15], so a group contributes
// d*sum(q*xq) + m*sum(x); the m*sum(x) term is only exact if the quantiser
// reports the same sum the GEMM will reconstruct, which is what this asserts.
func TestQuantizeXQ8Parity(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	const npos, cols = 7, 96 // cols must be a multiple of 32
	ng := cols / 32

	xb := make([]byte, npos*cols*4)
	x := make([]float32, npos*cols)
	for i := range x {
		// Deterministic, both signs, and a couple of exact zeros.
		v := float32(math.Sin(float64(i)*0.37) * 3.5)
		if i%17 == 0 {
			v = 0
		}
		x[i] = v
		b := math.Float32bits(v)
		xb[i*4+0] = byte(b)
		xb[i*4+1] = byte(b >> 8)
		xb[i*4+2] = byte(b >> 16)
		xb[i*4+3] = byte(b >> 24)
	}

	alloc := func(n uint64) Buffer {
		b, err := Alloc(n)
		if err != nil {
			t.Fatalf("alloc %d: %v", n, err)
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	dx := alloc(uint64(len(xb)))
	if err := Upload(dx, xb); err != nil {
		t.Fatal(err)
	}
	// int8 codes, padded to a 4-byte multiple so the alloc is tidy.
	dxq := alloc(uint64((npos*cols + 3) &^ 3))
	dsx := alloc(uint64(npos * ng * 4))
	dxs := alloc(uint64(npos * ng * 4))

	if err := QuantizeXQ8(dx, dxq, dsx, dxs, npos, cols); err != nil {
		t.Fatalf("quantize: %v", err)
	}
	if err := Sync(); err != nil {
		t.Fatal(err)
	}

	codeBuf := make([]byte, (npos*cols+3)&^3)
	if err := Download(codeBuf, dxq); err != nil {
		t.Fatal(err)
	}
	sxBuf := make([]byte, npos*ng*4)
	if err := Download(sxBuf, dsx); err != nil {
		t.Fatal(err)
	}
	xsBuf := make([]byte, npos*ng*4)
	if err := Download(xsBuf, dxs); err != nil {
		t.Fatal(err)
	}
	f32 := func(b []byte, i int) float32 {
		return math.Float32frombits(uint32(b[i*4]) | uint32(b[i*4+1])<<8 |
			uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24)
	}

	var maxStepErr, maxSumAbsErr float64
	var maxCode int
	var wTok, wG, wJ int
	var wOrig, wGot, wS float32
	for tok := 0; tok < npos; tok++ {
		for g := 0; g < ng; g++ {
			s := f32(sxBuf, tok*ng+g)
			want := f32(xsBuf, tok*ng+g) // reconstructed group sum
			var gotSum float64
			for j := 0; j < 32; j++ {
				q := int(int8(codeBuf[tok*cols+g*32+j]))
				if q > maxCode {
					maxCode = q
				}
				if -q > maxCode {
					maxCode = -q
				}
				gotSum += float64(float32(q) * s)
			}
			// The reported sum must equal the sum of the codes it emitted.
			if d := math.Abs(gotSum - float64(want)); d > maxSumAbsErr {
				maxSumAbsErr = d
			}
			// Reconstruction error is bounded by half a quantisation step, in
			// absolute terms. Measuring it relatively is wrong: an element far
			// below its group's max rounds to zero by design.
			for j := 0; j < 32; j++ {
				q := int8(codeBuf[tok*cols+g*32+j])
				got := float32(q) * s
				orig := x[tok*cols+g*32+j]
				if s == 0 {
					if orig != 0 {
						t.Fatalf("group max 0 but element %g at tok=%d g=%d", orig, tok, g)
					}
					continue
				}
				if err := math.Abs(float64(got-orig)) / float64(s); err > maxStepErr {
					maxStepErr = err
					wTok, wG, wJ = tok, g, j
					wOrig, wGot, wS = orig, got, s
				}
			}
		}
	}

	t.Logf("maxStepErr=%.4f (steps)  maxSumAbsErr=%.6g  maxAbsCode=%d", maxStepErr, maxSumAbsErr, maxCode)
	if maxStepErr > 0.5 {
		t.Logf("worst: tok=%d g=%d j=%d orig=%g got=%g s=%g q=%d",
			wTok, wG, wJ, wOrig, wGot, wS, int(int8(codeBuf[wTok*cols+wG*32+wJ])))
	}
	if maxStepErr > 0.5 {
		t.Fatalf("code reconstruction %.4f steps off; should be <= 0.5", maxStepErr)
	}
	if maxSumAbsErr > 1e-2 {
		t.Fatalf("reported group sum off by %.6g", maxSumAbsErr)
	}
	if maxCode > 127 {
		t.Fatalf("code %d exceeds int8 range", maxCode)
	}
}
