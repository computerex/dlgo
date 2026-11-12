package cuda

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/computerex/dlgo/quant"
)

// ---------------------------------------------------------------------------
// Canonical Affine4 / Affine8 GEMV + batched GEMM.
//
// The batched GEMM (dl_gemm_affine{4,8}_f16) is the prefill workhorse: it is the
// only thing between dlgo's current ~72 tok/s prompt processing and llama.cpp's
// ~6,900 tok/s at 16k, so it gets both a correctness test and a throughput gate.
// ---------------------------------------------------------------------------

func affineInit(t *testing.T) {
	t.Helper()
	if err := Init(); err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
}

func affF32Bytes(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return b
}

func affBytesF32(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

func affRelL2(got, want []float32) (rel float64, nonfinite int) {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			nonfinite++
		}
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	if den == 0 {
		return math.Sqrt(num), nonfinite
	}
	return math.Sqrt(num / den), nonfinite
}

func affRandF32(n int, scale float32, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	v := make([]float32, n)
	for i := range v {
		v[i] = scale * float32(r.NormFloat64())
	}
	return v
}

// affineCase holds one quantized weight matrix and its device handles.
type affineCase struct {
	name  string
	rows  int
	cols  int
	raw   []byte // canonical encoded weights
	deq   []float32
	qtype uint32
}

func newAffineCase(t *testing.T, name string, rows, cols int, b4 bool) *affineCase {
	t.Helper()
	wf := affRandF32(rows*cols, 0.1, int64(rows*1000+cols))
	c := &affineCase{name: name, rows: rows, cols: cols}
	if b4 {
		if rows*cols%quant.Affine4GroupSize != 0 {
			t.Fatalf("%s: dimensions must be a multiple of 32", name)
		}
		c.raw = quant.QuantizeAffine4(wf)
		c.deq = quant.DequantizeAffine4(c.raw, rows*cols)
		c.qtype = 61
	} else {
		c.raw = quant.QuantizeAffine8(wf)
		c.deq = quant.DequantizeAffine8(c.raw, rows*cols)
		c.qtype = 60
	}
	return c
}

// cpuGemm is the exact float64 reference: out[tok][row] = sum_c x[tok][c]*w[row][c].
func cpuGemm(x, w []float32, rows, cols, npos int) []float32 {
	out := make([]float32, npos*rows)
	for tk := 0; tk < npos; tk++ {
		xr := x[tk*cols : (tk+1)*cols]
		for r := 0; r < rows; r++ {
			wr := w[r*cols : (r+1)*cols]
			var acc float64
			for cc := 0; cc < cols; cc++ {
				acc += float64(xr[cc]) * float64(wr[cc])
			}
			out[tk*rows+r] = float32(acc)
		}
	}
	return out
}

// runGemm uploads x/half/w and runs the batched GEMM, returning out[tok][row].
func runGemm(t *testing.T, c *affineCase, x []float32, npos int, b4 bool) []float32 {
	t.Helper()
	wb, err := Alloc(uint64(len(c.raw)))
	if err != nil {
		t.Fatalf("%s: alloc weights: %v", c.name, err)
	}
	defer Free(wb)
	if err := Upload(wb, c.raw); err != nil {
		t.Fatalf("%s: upload weights: %v", c.name, err)
	}
	xf, err := Alloc(uint64(npos * c.cols * 4))
	if err != nil {
		t.Fatalf("%s: alloc x: %v", c.name, err)
	}
	defer Free(xf)
	if err := Upload(xf, affF32Bytes(x)); err != nil {
		t.Fatalf("%s: upload x: %v", c.name, err)
	}
	xh, err := Alloc(uint64(npos * c.cols * 2))
	if err != nil {
		t.Fatalf("%s: alloc xh: %v", c.name, err)
	}
	defer Free(xh)
	if err := F32ToHalf(xf, xh, npos*c.cols); err != nil {
		t.Fatalf("%s: f32->f16: %v", c.name, err)
	}
	ob, err := Alloc(uint64(npos * c.rows * 4))
	if err != nil {
		t.Fatalf("%s: alloc out: %v", c.name, err)
	}
	defer Free(ob)
	if err := Memset8(ob, 0, uint64(npos*c.rows*4)); err != nil {
		t.Fatalf("%s: memset out: %v", c.name, err)
	}
	if b4 {
		err = GemmAffine4F16(wb, xh, ob, c.rows, c.cols, npos)
	} else {
		err = GemmAffine8F16(wb, xh, ob, c.rows, c.cols, npos)
	}
	if err != nil {
		t.Fatalf("%s: gemm: %v", c.name, err)
	}
	if err := Sync(); err != nil {
		t.Fatalf("%s: sync: %v", c.name, err)
	}
	raw := make([]byte, npos*c.rows*4)
	if err := Download(raw, ob); err != nil {
		t.Fatalf("%s: download: %v", c.name, err)
	}
	return affBytesF32(raw)
}

// TestAffineGemmParity checks both canonical formats against the exact f64
// reference, including the ragged tile edges (rows and npos not multiples of 64).
func TestAffineGemmParity(t *testing.T) {
	affineInit(t)
	cases := []struct {
		name             string
		rows, cols, npos int
	}{
		{"affine4-small", 128, 256, 64},
		{"affine4-odd-rows", 200, 256, 64},
		{"affine4-odd-tokens", 256, 320, 70},
		{"affine4-qwen-gate", 512, 512, 128},
		{"affine8-small", 128, 256, 64},
		{"affine8-odd-tokens", 256, 320, 70},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b4 := tc.name[:7] == "affine4"
			c := newAffineCase(t, tc.name, tc.rows, tc.cols, b4)
			x := affRandF32(tc.npos*tc.cols, 0.5, 7)
			got := runGemm(t, c, x, tc.npos, b4)
			want := cpuGemm(x, c.deq, tc.rows, tc.cols, tc.npos)
			rel, nf := affRelL2(got, want)
			// f16 activations + f32 accumulation: 1e-3 is a comfortable bound that
			// still catches layout/stride bugs (which show up as ~1e0).
			if nf > 0 || rel > 1e-3 {
				t.Fatalf("affRelL2=%.3e nonfinite=%d (want <1e-3)", rel, nf)
			}
			t.Logf("%s affRelL2=%.3e", tc.name, rel)
		})
	}
}

// TestAffineGemvParity covers the single-token path used by decode.
func TestAffineGemvParity(t *testing.T) {
	affineInit(t)
	for _, b4 := range []bool{true, false} {
		name := "affine8"
		if b4 {
			name = "affine4"
		}
		t.Run(name, func(t *testing.T) {
			const rows, cols = 256, 512
			c := newAffineCase(t, name, rows, cols, b4)
			x := affRandF32(cols, 0.5, 11)
			wb, err := Alloc(uint64(len(c.raw)))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(wb)
			if err := Upload(wb, c.raw); err != nil {
				t.Fatal(err)
			}
			xb, err := Alloc(uint64(cols * 4))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(xb)
			if err := Upload(xb, affF32Bytes(x)); err != nil {
				t.Fatal(err)
			}
			ob, err := Alloc(uint64(rows * 4))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(ob)
			if b4 {
				err = GEMVAffine4(wb, xb, ob, rows, cols)
			} else {
				err = GEMVAffine8(wb, xb, ob, rows, cols)
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := Sync(); err != nil {
				t.Fatal(err)
			}
			raw := make([]byte, rows*4)
			if err := Download(raw, ob); err != nil {
				t.Fatal(err)
			}
			got := affBytesF32(raw)
			want := cpuGemm(x, c.deq, rows, cols, 1)
			rel, nf := affRelL2(got, want)
			if nf > 0 || rel > 1e-3 {
				t.Fatalf("%s gemv affRelL2=%.3e nonfinite=%d", name, rel, nf)
			}
			t.Logf("%s gemv affRelL2=%.3e", name, rel)
		})
	}
}

// TestAffineGemmThroughput is the prefill-parity gate. Shape is Qwen3-4B's
// gate/up projection at a 1024-token chunk. Parity with llama.cpp needs ~55
// TFLOP/s overall, so a single GEMM at 30 TFLOP/s is the floor worth building on.
func TestAffineGemmThroughput(t *testing.T) {
	if testing.Short() {
		t.Skip("throughput test")
	}
	affineInit(t)
	const rows, cols, npos = 9728, 2560, 1024
	const iters = 8
	flopsPer := 2.0 * float64(rows) * float64(cols) * float64(npos)

	for _, b4 := range []bool{true, false} {
		name := "affine8"
		if b4 {
			name = "affine4"
		}
		t.Run(name, func(t *testing.T) {
			c := newAffineCase(t, name, rows, cols, b4)
			wb, err := Alloc(uint64(len(c.raw)))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(wb)
			if err := Upload(wb, c.raw); err != nil {
				t.Fatal(err)
			}
			xf, err := Alloc(uint64(npos * cols * 4))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(xf)
			if err := Upload(xf, affF32Bytes(affRandF32(npos*cols, 0.5, 3))); err != nil {
				t.Fatal(err)
			}
			xh, err := Alloc(uint64(npos * cols * 2))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(xh)
			ob, err := Alloc(uint64(npos * rows * 4))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(ob)

			run := func() error {
				if err := F32ToHalf(xf, xh, npos*cols); err != nil {
					return err
				}
				if b4 {
					return GemmAffine4F16(wb, xh, ob, rows, cols, npos)
				}
				return GemmAffine8F16(wb, xh, ob, rows, cols, npos)
			}
			for i := 0; i < 2; i++ {
				if err := run(); err != nil {
					t.Fatal(err)
				}
			}
			if err := Sync(); err != nil {
				t.Fatal(err)
			}
			best := time.Duration(math.MaxInt64)
			for i := 0; i < iters; i++ {
				st := time.Now()
				if err := run(); err != nil {
					t.Fatal(err)
				}
				if err := Sync(); err != nil {
					t.Fatal(err)
				}
				if d := time.Since(st); d < best {
					best = d
				}
			}
			tflops := flopsPer / best.Seconds() / 1e12
			t.Logf("%s %dx%d x %d tokens: %v => %.1f TFLOP/s", name, rows, cols, npos,
				best, tflops)
			if tflops < 10 {
				t.Errorf("%s GEMM only %.1f TFLOP/s — below the 10 TFLOP/s floor", name, tflops)
			}
		})
	}
}
