//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// These tests deliberately fail rather than skip without a GPU: a passing run
// must have executed the actual compiled shader.
func TestSSMRecurrenceGPU(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer Shutdown()
	t.Logf("GPU: %s", DeviceName())
	for _, shape := range [][4]int{{4, 8, 7, 2}, {16, 128, 128, 4}, {32, 128, 128, 16}, {4, 129, 31, 2}} {
		h, k, v, g := shape[0], shape[1], shape[2], shape[3]
		t.Run(fmt.Sprintf("h%d_k%d_v%d_g%d", h, k, v, g), func(t *testing.T) {
			rng := rand.New(rand.NewSource(123))
			alloc := func(n int) Buf {
				b := Alloc(uint64(n * 4))
				if b == 0 {
					t.Fatal("alloc")
				}
				t.Cleanup(func() { Free(b) })
				return b
			}
			s1, s2 := alloc(h*k*v), alloc(h*k*v)
			q, a, b, p, y1, y2 := alloc(2*g*k+h*v), alloc(h), alloc(h), alloc(h*v), alloc(h*v), alloc(h*v)
			check := func(err error) {
				t.Helper()
				if err != nil {
					t.Fatal(err)
				}
			}
			initial := make([]float32, h*k*v)
			for i := range initial {
				initial[i] = (rng.Float32()*2 - 1) * 0.1
			}
			check(UploadF32(s1, initial))
			check(UploadF32(s2, initial))
			input := make([]float32, 2*g*k+h*v)
			alpha, beta := make([]float32, h), make([]float32, h)
			compare := func(label string, left, right Buf, n int) {
				x, z := make([]float32, n), make([]float32, n)
				check(DownloadF32(left, x))
				check(DownloadF32(right, z))
				mismatches := 0
				max := float64(0)
				for i := range x {
					if math.IsNaN(float64(x[i])) || math.IsInf(float64(x[i]), 0) || math.IsNaN(float64(z[i])) || math.IsInf(float64(z[i]), 0) {
						t.Fatalf("%s nonfinite at %d", label, i)
					}
					d := math.Abs(float64(x[i]) - float64(z[i]))
					if d > max {
						max = d
					}
					if math.Float32bits(x[i]) != math.Float32bits(z[i]) {
						mismatches++
					}
				}
				if mismatches != 0 {
					t.Fatalf("%s: %d bit mismatches, max abs %g", label, mismatches, max)
				}
			}
			old := ssmReference
			defer func() { ssmReference = old }()
			for token := 0; token < 256; token++ {
				for i := range input {
					input[i] = (rng.Float32()*2 - 1) / float32(math.Sqrt(float64(k)))
				}
				for i := range alpha {
					alpha[i] = -rng.Float32() * 0.1
					beta[i] = rng.Float32()
				}
				check(UploadF32(q, input))
				check(UploadF32(a, alpha))
				check(UploadF32(b, beta))
				// Both environment-selected and automatic size fallback paths.
				ssmReference = false
				BeginBatch()
				check(ssmRecurrence(s1, q, a, b, p, y1, h, k, v, g*k))
				EndBatch()
				Sync()
				ssmReference = true
				BeginBatch()
				check(ssmRecurrence(s2, q, a, b, p, y2, h, k, v, g*k))
				EndBatch()
				Sync()
				compare("output", y1, y2, h*v)
				if token%31 == 0 || token == 255 {
					compare("state", s1, s2, h*k*v)
				}
			}
			if k > 128 {
				if err := SSMDeltaRule(s1, q, a, b, y1, h, k, v, g*k); err == nil {
					t.Fatal("unguarded fused dimension")
				}
			}
		})
	}
}

func TestSSMRecurrenceLatencyGPU(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(Shutdown)
	const h, k, v, g, tokens = 32, 128, 128, 16, 512
	alloc := func(n int) Buf {
		b := Alloc(uint64(n * 4))
		if b == 0 {
			t.Fatal("alloc")
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	s, q, a, b, p, y := alloc(h*k*v), alloc(2*g*k+h*v), alloc(h), alloc(h), alloc(h*v), alloc(h*v)
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	check(UploadF32(s, make([]float32, h*k*v)))
	rng := rand.New(rand.NewSource(9))
	input := make([]float32, 2*g*k+h*v)
	for i := range input {
		input[i] = (rng.Float32()*2 - 1) / float32(math.Sqrt(k))
	}
	check(UploadF32(q, input))
	alpha, beta := make([]float32, h), make([]float32, h)
	for i := range alpha {
		alpha[i] = -0.01
		beta[i] = 0.3
	}
	check(UploadF32(a, alpha))
	check(UploadF32(b, beta))
	run := func(reference bool) time.Duration {
		start := time.Now()
		BeginBatch()
		for i := 0; i < tokens; i++ {
			Barrier()
			var err error
			if reference {
				err = ssmRecurrenceReference(s, q, a, b, p, y, h, k, v, g*k)
			} else {
				err = SSMDeltaRule(s, q, a, b, y, h, k, v, g*k)
			}
			check(err)
		}
		EndBatch()
		Sync()
		return time.Since(start)
	}
	run(true)
	run(false) // pipeline warmup excluded
	var ref, fused time.Duration
	for i := 0; i < 5; i++ {
		if i%2 == 0 {
			ref += run(true)
			fused += run(false)
		} else {
			fused += run(false)
			ref += run(true)
		}
	}
	t.Logf("GPU=%s tokens=%d reference=%.3fms fused=%.3fms speedup=%.2fx dispatches=%d->%d (record+submit+wait, 5-run mean; recurrence only)", DeviceName(), tokens, float64(ref)/5e6, float64(fused)/5e6, float64(ref)/float64(fused), 4*tokens, tokens)
}
