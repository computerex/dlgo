//go:build cgo && vulkan

package gpu

import (
	"testing"
	"time"
)

func TestGemmQ3KRecoveredTimingGPU(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)
	const rows, cols = 12288, 4096
	rng := seededRand()
	weights := make([]float32, rows*cols)
	for i := range weights {
		weights[i] = (rng.Float32()*2 - 1) * .2
	}
	packed := make([]byte, 0, rows*(cols/256)*110)
	for r := 0; r < rows; r++ {
		packed = append(packed, packQ3KRow(weights[r*cols:(r+1)*cols])...)
	}
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
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
	wb := alloc(len(packed))
	check(Upload(wb, packed))
	ob := alloc(rows * 512 * 4)
	xb := alloc(cols * 512 * 4)
	x := make([]float32, cols*512)
	for i := range x {
		x[i] = rng.Float32() - .5
	}
	bench := func(npos, reps int, run func() error) float64 {
		check(UploadF32(xb, x[:npos*cols]))
		BeginBatch()
		check(run())
		EndBatch()
		Sync()
		start := time.Now()
		for i := 0; i < reps; i++ {
			BeginBatch()
			check(run())
			EndBatch()
			Sync()
		}
		return float64(time.Since(start).Microseconds()) / 1000 / float64(reps)
	}
 for _, npos := range []int{128, 512} {
  reps := 5
  if npos == 128 { reps = 10 }
  SetGemmQ3K(true)
  tiled := bench(npos, reps, func() error { return BatchMatVec(ob, wb, xb, rows, cols, npos, 11) })
  SetGemmQ3K(false)
  recovered := bench(npos, reps, func() error { return gemmQ3KRecovered(ob, wb, xb, rows, cols, npos) })
  t.Logf("npos=%d tiled=%.3fms recovered=%.3fms speedup=%.3fx", npos, tiled, recovered, tiled/recovered)
 }
}
