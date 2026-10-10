//go:build cgo && vulkan

package gpu

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestSSMBatchT32ParityGPU verifies the opt-in 32-lane SSM recurrence
// (DLGO_SSM_REC_T32=1) produces bit-identical outputs, recurrent state, and
// convolution state versus the default 16-lane batched kernel across two
// sequential chunks, and reports the kernel-level speedup.
func TestSSMBatchT32ParityGPU(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	t.Cleanup(Shutdown)
	t.Cleanup(func() { SetSSMRecT32(false) })
	// Match Qwen3.5-9B: 32 value heads, 16 key groups, 512-token chunks.
	const h, k, v, g, ck, n = 32, 128, 128, 16, 4, 512
	const channels = 2*g*k + h*v
	rng := rand.New(rand.NewSource(777))
	alloc := func(n int) Buf {
		b := Alloc(uint64(n * 4))
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
	fill := func(n int, scale float32) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = (rng.Float32()*2 - 1) * scale
		}
		return x
	}
	raw, processed := alloc(n*channels), alloc(n*channels)
	ab, bb, zb := alloc(n*h), alloc(n*h), alloc(n*h*v)
	sb16, sb32 := alloc(h*k*v), alloc(h*k*v)
	cb16, cb32 := alloc(ck*channels), alloc(ck*channels)
	w, a, bias, norm := alloc(ck*channels), alloc(h), alloc(h), alloc(v)
	initial := fill(h*k*v, .1)
	conv16Snapshot := fill(ck*channels, .1)
	check(UploadF32(cb16, conv16Snapshot))
	check(UploadF32(cb32, conv16Snapshot))
	check(UploadF32(w, fill(ck*channels, .2)))
	av := make([]float32, h)
	for i := range av {
		av[i] = -.1
	}
	check(UploadF32(a, av))
	check(UploadF32(bias, fill(h, .2)))
	nv := make([]float32, v)
	for i := range nv {
		nv[i] = 1
	}
	check(UploadF32(norm, nv))
	y16 := alloc(n * h * v)
	y32 := alloc(n * h * v)
	equal := func(name string, x, y Buf, count int) {
		t.Helper()
		xx, yy := make([]float32, count), make([]float32, count)
		check(DownloadF32(x, xx))
		check(DownloadF32(y, yy))
		for i := range xx {
			if math.IsNaN(float64(xx[i])) || math.IsInf(float64(xx[i]), 0) || math.Float32bits(xx[i]) != math.Float32bits(yy[i]) {
				t.Fatalf("%s mismatch [%d]: %g / %g", name, i, xx[i], yy[i])
			}
		}
	}
	for chunk := 0; chunk < 2; chunk++ {
		rawIn := fill(n*channels, .4)
		alphaIn := fill(n*h, .3)
		betaIn := fill(n*h, .3)
		zIn := fill(n*h*v, .3)
		check(UploadF32(raw, rawIn))
		check(UploadF32(ab, alphaIn))
		check(UploadF32(bb, betaIn))
		check(UploadF32(zb, zIn))
		check(UploadF32(sb16, initial))
		check(UploadF32(sb32, initial))

		SetSSMRecT32(false)
		start := time.Now()
		BeginBatch()
		check(ssmBatchPrefill(raw, processed, ab, bb, zb, y16, sb16, cb16, w, a, bias, norm, h, k, v, g*k, ck, n, 1e-6))
		EndBatch()
		Sync()
		t16 := time.Since(start)

		// Stage 1 preprocesses alpha/beta in place: re-upload pristine inputs
		// so the second run sees identical data. The conv sliding-window state
		// also starts pristine for both runs.
		check(UploadF32(raw, rawIn))
		check(UploadF32(ab, alphaIn))
		check(UploadF32(bb, betaIn))
		check(UploadF32(zb, zIn))
		check(UploadF32(sb32, initial))
		check(UploadF32(cb32, conv16Snapshot))

		SetSSMRecT32(true)
		start = time.Now()
		BeginBatch()
		check(ssmBatchPrefill(raw, processed, ab, bb, zb, y32, sb32, cb32, w, a, bias, norm, h, k, v, g*k, ck, n, 1e-6))
		EndBatch()
		Sync()
		t32 := time.Since(start)
		SetSSMRecT32(false)

		// Bit-identical requirements.
		equal("output", y16, y32, n*h*v)
		equal("state", sb16, sb32, h*k*v)
		equal("conv state", cb16, cb32, ck*channels)
		t.Logf("chunk=%d t16=%s t32=%s speedup=%.2fx bit-identical", chunk, t16, t32, float64(t16)/float64(t32))

		// Carry state forward: next chunk starts from this chunk's state.
		initial = make([]float32, h*k*v)
		check(DownloadF32(sb16, initial))
		conv16Snapshot = make([]float32, ck*channels)
		check(DownloadF32(cb16, conv16Snapshot))
		check(UploadF32(cb16, conv16Snapshot))
	}
}
