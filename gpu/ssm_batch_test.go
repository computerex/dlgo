//go:build cgo && vulkan

package gpu

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestSSMBatchGPU(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(Shutdown)
	const h, k, v, g, ck, n = 16, 128, 128, 16, 4, 128
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
	ab, bb, zb, yb := alloc(n*h), alloc(n*h), alloc(n*h*v), alloc(n*h*v)
	ar, br, zr, yr, qr, pred := alloc(h), alloc(h), alloc(h*v), alloc(h*v), alloc(channels), alloc(h*v)
	sr, sb, cr, cb := alloc(h*k*v), alloc(h*k*v), alloc(ck*channels), alloc(ck*channels)
	w, a, bias, norm := alloc(ck*channels), alloc(h), alloc(h), alloc(v)
	initial := fill(h*k*v, .1)
	check(UploadF32(sr, initial))
	check(UploadF32(sb, initial))
	conv := fill(ck*channels, .1)
	check(UploadF32(cr, conv))
	check(UploadF32(cb, conv))
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
	outRef := alloc(n * h * v)
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
		check(UploadF32(raw, fill(n*channels, .4)))
		check(UploadF32(ab, fill(n*h, .3)))
		check(UploadF32(bb, fill(n*h, .3)))
		check(UploadF32(zb, fill(n*h*v, .3)))
		start := time.Now()
		BeginBatch()
		// Reference executes first; batch alpha/beta inputs are still raw.
		for token := 0; token < n; token++ {
			CopyRegion(qr, 0, raw, uint64(token*channels*4), uint64(channels*4))
			CopyRegion(ar, 0, ab, uint64(token*h*4), uint64(h*4))
			CopyRegion(br, 0, bb, uint64(token*h*4), uint64(h*4))
			CopyRegion(zr, 0, zb, uint64(token*h*v*4), uint64(h*v*4))
			Barrier()
			check(SSMConv1dSiLU(qr, cr, w, channels, ck))
			Barrier()
			check(SSMPreprocess(ar, br, a, bias, qr, h, k, g*k, 1e-6, true))
			Barrier()
			check(ssmRecurrenceReference(sr, qr, ar, br, pred, yr, h, k, v, g*k))
			Barrier()
			check(SSMNormGate(yr, zr, norm, h, v, 1e-6))
			Barrier()
			CopyRegion(outRef, uint64(token*h*v*4), yr, 0, uint64(h*v*4))
			Barrier()
		}
		EndBatch()
		Sync()
		rt := time.Since(start)
		start = time.Now()
		BeginBatch()
		check(ssmBatchPrefill(raw, processed, ab, bb, zb, yb, sb, cb, w, a, bias, norm, h, k, v, g*k, ck, n, 1e-6))
		EndBatch()
		Sync()
		bt := time.Since(start)
		equal("gated output", outRef, yb, n*h*v)
		equal("state", sr, sb, h*k*v)
		equal("conv state", cr, cb, ck*channels)
		t.Logf("chunk=%d tokens=%d ref=%s batch=%s speedup=%.2fx outputs/state/conv bit-identical", chunk, n, rt, bt, float64(rt)/float64(bt))
	}
}
