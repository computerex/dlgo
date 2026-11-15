//go:build vulkan

package cuda

import (
	"testing"
)

// TestGemmAffine4I8Parity checks the int8 tensor-core prefill GEMM against the
// exact f64 reference. The reference uses the original f32 activations, so the
// only error source is the per-32-group int8 quantisation of X (the Affine4
// weight decode is exact: w = d*q + m with integer q in [0,15]).
func TestGemmAffine4I8Parity(t *testing.T) {
	affineInit(t)
	cases := []struct {
		name             string
		rows, cols, npos int
	}{
		{"i8-128", 128, 256, 128},
		{"i8-odd-rows", 200, 256, 128},
		{"i8-odd-tokens", 256, 320, 70},
		{"i8-one-token-tile", 256, 512, 37},
		{"i8-qwen-gate", 512, 512, 128},
		{"i8-deep-k", 256, 4096, 129},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := newAffineCase(t, tc.name, tc.rows, tc.cols, true)
			x := affRandF32(tc.npos*tc.cols, 0.5, 7)

			wb, err := Alloc(uint64(len(c.raw)))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(wb)
			if err := Upload(wb, c.raw); err != nil {
				t.Fatal(err)
			}

			ng := tc.cols / 32
			xb, err := Alloc(uint64(tc.npos * tc.cols * 4))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(xb)
			if err := Upload(xb, affF32Bytes(x)); err != nil {
				t.Fatal(err)
			}
			qb, err := Alloc(uint64(tc.npos * tc.cols))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(qb)
			sb, err := Alloc(uint64(tc.npos * ng * 4))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(sb)
			ub, err := Alloc(uint64(tc.npos * ng * 4))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(ub)
			if err := QuantizeXQ8(xb, qb, sb, ub, tc.npos, tc.cols); err != nil {
				t.Fatalf("quantize: %v", err)
			}

			ob, err := Alloc(uint64(tc.npos * tc.rows * 4))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(ob)
			if err := Memset8(ob, 0, uint64(tc.npos*tc.rows*4)); err != nil {
				t.Fatal(err)
			}
			if err := GemmAffine4I8(wb, qb, sb, ub, ob, tc.rows, tc.cols, tc.npos); err != nil {
				t.Fatalf("gemm i8: %v", err)
			}
			if err := Sync(); err != nil {
				t.Fatal(err)
			}
			raw := make([]byte, tc.npos*tc.rows*4)
			if err := Download(raw, ob); err != nil {
				t.Fatal(err)
			}
			got := affBytesF32(raw)
			want := cpuGemm(x, c.deq, tc.rows, tc.cols, tc.npos)
			rel, nf := affRelL2(got, want)
			// int8 activations at one scale per 32 elements: ~5e-3 typical. A
			// layout or stride bug shows up as ~1e0, so 3e-2 still catches those.
			if nf > 0 || rel > 3e-2 {
				t.Fatalf("affRelL2=%.3e nonfinite=%d (want <3e-2)", rel, nf)
			}
			t.Logf("%s affRelL2=%.3e", tc.name, rel)
		})
	}
}
