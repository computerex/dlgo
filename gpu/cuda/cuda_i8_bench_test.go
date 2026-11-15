//go:build vulkan

package cuda

import (
	"fmt"
	"testing"
	"time"
)

// TestI8GemmThroughput compares the int8 tensor-core prefill GEMM against the f16
// one on the Qwen3-4B shapes, at prefill-realistic token counts.
func TestI8GemmThroughput(t *testing.T) {
	affineInit(t)
	shapes := []struct {
		name       string
		rows, cols int
	}{
		{"q-proj", 4096, 2560},
		{"kv-proj", 2048, 2560},
		{"o-proj", 2560, 4096},
		{"gate", 9728, 2560},
		{"down", 2560, 9728},
	}
	for _, npos := range []int{512, 4096} {
		for _, sh := range shapes {
			c := newAffineCase(t, sh.name, sh.rows, sh.cols, true)
			wb, err := Alloc(uint64(len(c.raw)))
			if err != nil {
				t.Fatal(err)
			}
			if err := Upload(wb, c.raw); err != nil {
				t.Fatal(err)
			}
			x := affRandF32(npos*sh.cols, 0.5, 5)
			xb, _ := Alloc(uint64(npos * sh.cols * 4))
			_ = Upload(xb, affF32Bytes(x))
			xh, _ := Alloc(uint64(npos * sh.cols * 2))
			_ = F32ToHalf(xb, xh, npos*sh.cols)
			ng := sh.cols / 32
			qb, _ := Alloc(uint64(npos * sh.cols))
			sb, _ := Alloc(uint64(npos * ng * 4))
			ub, _ := Alloc(uint64(npos * ng * 4))
			_ = QuantizeXQ8(xb, qb, sb, ub, npos, sh.cols)
			ob, _ := Alloc(uint64(npos * sh.rows * 4))
			_ = Sync()

			flops := 2.0 * float64(sh.rows) * float64(sh.cols) * float64(npos)
			reps := 20

			run := func(name string, f func() error) {
				if err := f(); err != nil {
					t.Fatal(err)
				}
				_ = Sync()
				st := time.Now()
				for i := 0; i < reps; i++ {
					if err := f(); err != nil {
						t.Fatal(err)
					}
				}
				_ = Sync()
				sec := time.Since(st).Seconds()
				t.Logf("npos=%-5d %-8s %-4s %8.1f us  %6.1f TFLOP/s",
					npos, sh.name, name, sec/float64(reps)*1e6, flops*float64(reps)/sec/1e12)
			}
			run("f16", func() error { return GemmAffine4F16(wb, xh, ob, sh.rows, sh.cols, npos) })
			run("i8", func() error { return GemmAffine4I8(wb, qb, sb, ub, ob, sh.rows, sh.cols, npos) })

			Free(wb)
			Free(xb)
			Free(xh)
			Free(qb)
			Free(sb)
			Free(ub)
			Free(ob)
		}
	}
	fmt.Println("done")
}
