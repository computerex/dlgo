//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"github.com/computerex/dlgo/quant"
	"math"
	"math/rand"
	"testing"
)

func TestGemmQ3KRecoveredGPU(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)
	for _, shape := range []struct{ rows, cols, npos int }{{65, 256, 17}, {128, 4096, 64}, {65, 12288, 3}} {
		t.Run(fmt.Sprintf("r%d_k%d_n%d", shape.rows, shape.cols, shape.npos), func(t *testing.T) {
			rows, cols, n := shape.rows, shape.cols, shape.npos
			rng := rand.New(rand.NewSource(99))
			weights := make([]float32, rows*cols)
			for i := range weights {
				weights[i] = (rng.Float32()*2 - 1) * .3
			}
			packed := make([]byte, 0, rows*(cols/256)*110)
			for r := 0; r < rows; r++ {
				packed = append(packed, packQ3KRow(weights[r*cols:(r+1)*cols])...)
			}
			// Vary packed scales independently to cover nonuniform six-bit scale groups.
			for block := 0; block < len(packed); block += 110 {
				for j := 96; j < 108; j++ {
					packed[block+j] = byte(rng.Intn(256))
				}
			}
			deq := quant.DequantizeQ3_K(packed, rows*cols)
			x := make([]float32, n*cols)
			for i := range x {
				x[i] = (rng.Float32()*2 - 1) * .5
			}
			ref := make([]float32, n*rows)
			for p := 0; p < n; p++ {
				for r := 0; r < rows; r++ {
					var sum float64
					for k := 0; k < cols; k++ {
						sum += float64(deq[r*cols+k]) * float64(x[p*cols+k])
					}
					ref[p*rows+r] = float32(sum)
				}
			}
			alloc := func(bytes int) Buf {
				b := Alloc(uint64(bytes))
				if b == 0 {
					t.Fatal("alloc")
				}
				t.Cleanup(func() { Free(b) })
				return b
			}
			check := func(err error) {
				t.Helper()
				if err != nil {
					t.Fatal(err)
				}
			}
			wb, xb, ob := alloc(len(packed)), alloc(len(x)*4), alloc(len(ref)*4)
			check(Upload(wb, packed))
			check(UploadF32(xb, x))
			sentinel := make([]float32, len(ref))
			for i := range sentinel {
				sentinel[i] = 12345
			}
			check(UploadF32(ob, sentinel))
			BeginBatch()
			check(gemmQ3KRecovered(ob, wb, xb, rows, cols, n))
			EndBatch()
			Sync()
			got := make([]float32, len(ref))
			check(DownloadF32(ob, got))
			var err2, ref2, maxAbs float64
			for i := range ref {
				if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || got[i] == 12345 {
					t.Fatalf("invalid/unwritten output at %d: %g", i, got[i])
				}
				d := float64(got[i] - ref[i])
				err2 += d * d
				ref2 += float64(ref[i]) * float64(ref[i])
				maxAbs = math.Max(maxAbs, math.Abs(d))
			}
			rel := math.Sqrt(err2 / ref2)
			t.Logf("relL2=%g maxAbs=%g", rel, maxAbs)
			if rel > 1e-3 {
				t.Fatalf("Q3_K coopmat recovery incorrect: relL2=%g", rel)
			}
		})
	}
}
