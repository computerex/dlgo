//go:build cgo && vulkan

package gpu

import (
	"math"
	"math/rand"
	"testing"
)

// TestShortConvGPU checks the causal depthwise short-convolution shader against
// a direct transcription of the CPU reference (models/llm shortConvStep +
// forwardShortConvLayer), across both the single-token decode case and a
// multi-token scan, and verifies that the rolling state carries over between
// dispatches exactly as the CPU path expects.
func TestShortConvGPU(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(Shutdown)

	const dim = 64 // small but > one workgroup's tail
	const kernel = 3
	const hist = kernel - 1

	rng := rand.New(rand.NewSource(4242))
	fill := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = (rng.Float32()*2 - 1) * 0.5
		}
		return x
	}
	alloc := func(n int) Buf {
		b := Alloc(uint64(n * 4))
		if b == 0 {
			t.Fatal("alloc failed")
		}
		t.Cleanup(func() { Free(b) })
		return b
	}

	convW := fill(dim * kernel)

	// cpuStep mirrors models/llm.shortConvStep + the b*x and c*gate wiring.
	cpuStep := func(y, state, bx, bcxRow []float32) {
		for i := 0; i < dim; i++ {
			bx[i] = bcxRow[i] * bcxRow[2*dim+i]
		}
		for i := 0; i < dim; i++ {
			acc := float32(0)
			for t := 0; t < hist; t++ {
				acc += state[t*dim+i] * convW[i*kernel+t]
			}
			acc += bx[i] * convW[i*kernel+hist]
			y[i] = bcxRow[dim+i] * acc
		}
		if hist > 1 {
			copy(state[:hist*dim-dim], state[dim:hist*dim])
		}
		copy(state[(hist-1)*dim:hist*dim], bx)
	}

	const npos = 5
	bcxHost := fill(npos * 3 * dim)

	// --- multi-token dispatch from a zeroed state ---
	gpuBCX := alloc(npos * 3 * dim)
	gpuY := alloc(npos * dim)
	gpuState := alloc(hist * dim)
	gpuConv := alloc(dim * kernel)
	if err := UploadF32(gpuBCX, bcxHost); err != nil {
		t.Fatal(err)
	}
	if err := UploadF32(gpuState, make([]float32, hist*dim)); err != nil {
		t.Fatal(err)
	}
	if err := UploadF32(gpuConv, convW); err != nil {
		t.Fatal(err)
	}
	BeginBatch()
	if err := ShortConv(gpuBCX, gpuY, gpuState, gpuConv, dim, kernel, npos); err != nil {
		t.Fatal(err)
	}
	EndBatch()
	Sync()

	got := make([]float32, npos*dim)
	if err := DownloadF32(gpuY, got); err != nil {
		t.Fatal(err)
	}
	gotState := make([]float32, hist*dim)
	if err := DownloadF32(gpuState, gotState); err != nil {
		t.Fatal(err)
	}

	wantState := make([]float32, hist*dim)
	want := make([]float32, npos*dim)
	bx := make([]float32, dim)
	for p := 0; p < npos; p++ {
		cpuStep(want[p*dim:(p+1)*dim], wantState, bx, bcxHost[p*3*dim:(p+1)*3*dim])
	}

	maxDiff := func(a, b []float32) float64 {
		m := 0.0
		for i := range a {
			d := math.Abs(float64(a[i] - b[i]))
			if d > m {
				m = d
			}
		}
		return m
	}
	if d := maxDiff(got, want); d > 2e-5 {
		// Print the raw ingredients for channel 0 so the mismatch can be
		// attributed to a specific convention (tap order, chunk order, gate).
		t.Logf("convW[0:3]=%v", convW[0:3])
		t.Logf("row0 b=%v c=%v x=%v", bcxHost[0], bcxHost[dim], bcxHost[2*dim])
		bx0 := bcxHost[0] * bcxHost[2*dim]
		t.Logf("row0 bx=%v -> tap0=%.8f tap1=%.8f tap2=%.8f",
			bx0, bx0*convW[0], bx0*convW[1], bx0*convW[2])
		t.Logf("row1 bx=%.8f", bcxHost[3*dim]*bcxHost[3*dim+2*dim])
		t.Fatalf("batched conv mismatch: maxAbsDiff=%g\n got[0:8]=%v\nwant[0:8]=%v",
			d, got[:8], want[:8])
	}
	if d := maxDiff(gotState, wantState); d > 2e-5 {
		t.Fatalf("batched conv state mismatch: maxAbsDiff=%g", d)
	}

	// --- single-token dispatches must continue the same scan ---
	// Reset and replay token-by-token; the result must match the batched run.
	if err := UploadF32(gpuState, make([]float32, hist*dim)); err != nil {
		t.Fatal(err)
	}
	oneY := alloc(dim)
	oneBCX := alloc(3 * dim)
	stepGot := make([]float32, 0, npos*dim)
	for p := 0; p < npos; p++ {
		if err := UploadF32(oneBCX, bcxHost[p*3*dim:(p+1)*3*dim]); err != nil {
			t.Fatal(err)
		}
		BeginBatch()
		if err := ShortConv(oneBCX, oneY, gpuState, gpuConv, dim, kernel, 1); err != nil {
			t.Fatal(err)
		}
		EndBatch()
		Sync()
		row := make([]float32, dim)
		if err := DownloadF32(oneY, row); err != nil {
			t.Fatal(err)
		}
		stepGot = append(stepGot, row...)
	}
	if d := maxDiff(stepGot, want); d > 2e-5 {
		t.Fatalf("decode-step conv mismatch vs batched: maxAbsDiff=%g", d)
	}
}
