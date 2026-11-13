//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/computerex/dlgo/quant"
)

// TestProbePrefillGemm measures the batched prefill GEMM at the exact Qwen3-4B
// shapes, with and without L2-resident weights. It is the iteration loop for the
// GEMM rewrite: almost all of prefill is these five GEMMs, so this number is the
// prefill number.
//
//	go test -tags vulkan -run TestProbePrefillGemm -v -timeout 900s ./gpu/
//
// PROBE_NPOS sets the token count, PROBE_REPS the inner repetitions per timed
// sample (Windows time.Now() granularity is coarse, so a single sub-ms call
// cannot be timed directly), PROBE_NBUF the number of weight copies to rotate
// through for the cold case.
func TestProbePrefillGemm(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	type shape struct {
		name       string
		rows, cols int
	}
	// Qwen3-4B: dim 2560, ffn 9728, q 32x128=4096, kv 8x128=1024.
	shapes := []shape{
		{"q-proj", 4096, 2560},
		{"kv-proj", 2048, 2560},
		{"o-proj", 2560, 4096},
		{"gate", 9728, 2560},
		{"down", 2560, 9728},
	}

	rng := rand.New(rand.NewSource(7))
	npos := benchEnvInt("PROBE_NPOS", 512)
	reps := benchEnvInt("PROBE_REPS", 16)
	nbuf := benchEnvInt("PROBE_NBUF", 12)

	alloc := func(n uint64) Buf {
		b := Alloc(n)
		if b == 0 {
			t.Fatalf("alloc %d failed", n)
		}
		t.Cleanup(func() { Free(b) })
		return b
	}

	// timeLoop returns the best mean per-call time over reps back-to-back calls.
	timeLoop := func(fn func(i int)) time.Duration {
		fn(-1) // warm
		Sync()
		const samples = 3
		best := time.Duration(1 << 62)
		for s := 0; s < samples; s++ {
			start := time.Now()
			for i := 0; i < reps; i++ {
				fn(i)
			}
			Sync()
			if d := time.Since(start) / time.Duration(reps); d < best {
				best = d
			}
		}
		return best
	}

	var sumFlops, sumTimeHot, sumTimeCold float64
	for _, s := range shapes {
		wf := make([]float32, s.rows*s.cols)
		for i := range wf {
			wf[i] = (rng.Float32()*2 - 1) * 0.05
		}
		enc := quant.QuantizeAffine4(wf)
		wf = nil
		x := make([]float32, npos*s.cols)
		for i := range x {
			x[i] = (rng.Float32()*2 - 1) * 0.05
		}
		xb := alloc(uint64(len(x)) * 4)
		if err := UploadF32(xb, x); err != nil {
			t.Fatal(err)
		}
		x = nil
		ob := alloc(uint64(npos) * uint64(s.rows) * 4)

		nb := nbuf
		if cap := capFor(int64(len(enc)), 220<<20); nb > cap {
			nb = cap
		}
		if nb < 1 {
			nb = 1
		}
		wbufs := make([]Buf, nb)
		for i := range wbufs {
			wbufs[i] = alloc(uint64(len(enc)))
			if err := Upload(wbufs[i], enc); err != nil {
				t.Fatal(err)
			}
		}

		flops := 2.0 * float64(s.rows) * float64(s.cols) * float64(npos)
		hot := timeLoop(func(int) {
			if err := cudaBatchMatVec(ob, wbufs[0], xb, s.rows, s.cols, npos, 61); err != nil {
				t.Fatal(err)
			}
		})
		cold := timeLoop(func(i int) {
			if err := cudaBatchMatVec(ob, wbufs[(i+1)%nb], xb, s.rows, s.cols, npos, 61); err != nil {
				t.Fatal(err)
			}
		})

		sumFlops += flops
		sumTimeHot += hot.Seconds()
		sumTimeCold += cold.Seconds()
		t.Logf("%-8s rows=%5d cols=%5d  %6.1f GFLOP  hot %6.1f TF (%7.1f us)  cold %6.1f TF (%7.1f us)  %s",
			s.name, s.rows, s.cols, flops/1e9,
			flops/1e12/hot.Seconds(), float64(hot.Microseconds()),
			flops/1e12/cold.Seconds(), float64(cold.Microseconds()), "")
	}
	// Sequential per-layer wall time, then the equivalent prefill throughput.
	layerHot := sumTimeHot / 36
	layerCold := sumTimeCold / 36
	tps := float64(npos) / (layerCold * 36)
	t.Logf("per-layer GEMM: hot %.3f ms  cold %.3f ms  |  model GEMM %.1f ms  => %.0f tok/s at npos=%d",
		layerHot*1e3, layerCold*1e3, layerCold*36*1e3, tps, npos)
	t.Logf("mean rate: hot %.1f TF  cold %.1f TF", sumFlops/sumTimeHot/1e12, sumFlops/sumTimeCold/1e12)
	fmt.Printf("PROBE npos=%d hot=%.1fTF cold=%.1fTF gemm_ms=%.1f tps=%.0f\n",
		npos, sumFlops/sumTimeHot/1e12, sumFlops/sumTimeCold/1e12, layerCold*36*1e3, tps)
}

// capFor returns how many copies of a buffer of size bytes fit in budget.
func capFor(size, budget int64) int {
	if size <= 0 {
		return 1
	}
	n := int(budget / size)
	if n < 1 {
		n = 1
	}
	return n
}
