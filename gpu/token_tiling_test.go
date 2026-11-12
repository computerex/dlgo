//go:build cgo && vulkan

package gpu

// Guards for the Windows-TDR token tiling in the position-batched GEMM.
//
// The GEMM now splits the token axis across several dispatches when a single
// dispatch would run long enough to trip the display watchdog (TdrDelay, 2 s by
// default). The split is a buffer offset on x and out, so it must be
// numerically invisible: this test checks the numbers against an exact CPU
// reference AND checks that the split actually happened when the budget is
// tight.
//
// The chunk size is read from DLGO_VK_GEMM_UNITS on every call, but it cannot
// be varied in-process on Windows (the C runtime snapshots the environment at
// startup, so os.Setenv never reaches getenv). Run this test twice instead:
//
//	go test -tags vulkan -count=1 -run TestBatchMatVecTokenTiling ./gpu/
//	$env:DLGO_VK_GEMM_UNITS="1"; go test -tags vulkan -count=1 -run TestBatchMatVecTokenTiling ./gpu/
//
// Both must pass; the second must also issue more than one dispatch.

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/computerex/dlgo/quant"
)

func TestBatchMatVecTokenTiling(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	// 4 token tiles at BN=128, so a tight budget must split this into 4.
	const rows, cols, npos = 96, 128, 512

	rng := rand.New(rand.NewSource(7))
	wf := make([]float32, rows*cols)
	for i := range wf {
		wf[i] = (rng.Float32()*2 - 1) * 0.1
	}
	w := quant.QuantizeAffine8(wf)
	wd := quant.DequantizeAffine8(w, rows*cols)

	x := make([]float32, npos*cols)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 0.1
	}

	alloc := func(n uint64) Buf {
		b := Alloc(n)
		if b == 0 {
			t.Fatalf("alloc %d bytes failed", n)
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	wb := alloc(uint64(len(w)))
	if err := Upload(wb, w); err != nil {
		t.Fatal(err)
	}
	xb := alloc(uint64(len(x)) * 4)
	if err := UploadF32(xb, x); err != nil {
		t.Fatal(err)
	}
	ob := alloc(uint64(npos) * uint64(rows) * 4)

	PerfReset()
	BeginBatch()
	if err := BatchMatVec(ob, wb, xb, rows, cols, npos, QuantTypeAffine8); err != nil {
		t.Fatal(err)
	}
	EndBatch()
	Sync()
	dispatches := PerfGetDispatches()

	got := make([]float32, npos*rows)
	if err := DownloadF32(ob, got); err != nil {
		t.Fatal(err)
	}

	// CPU reference: out[n*rows+m] = sum_k W[m,k] * x[n,k]
	var refL2, diffL2, maxErr float64
	for n := 0; n < npos; n++ {
		for m := 0; m < rows; m++ {
			var s float32
			for k := 0; k < cols; k++ {
				s += wd[m*cols+k] * x[n*cols+k]
			}
			gotV := float64(got[n*rows+m])
			refV := float64(s)
			refL2 += refV * refV
			d := gotV - refV
			diffL2 += d * d
			if a := math.Abs(d); a > maxErr {
				maxErr = a
			}
		}
	}
	relL2 := math.Sqrt(diffL2 / refL2)
	t.Logf("dispatches=%d relL2=%.3e maxErr=%.3e", dispatches, relL2, maxErr)

	// The Affine8 GEMM dequantizes into f16 tiles, so ~1e-4 relative is expected.
	if relL2 > 2e-3 {
		t.Fatalf("relL2 %.3e exceeds 2e-3", relL2)
	}

	// With a tight budget the dispatch must be split; without one it must not.
	if os.Getenv("DLGO_VK_GEMM_UNITS") != "" {
		if dispatches < 2 {
			t.Fatalf("token tiling did not engage: %d dispatch(es) for %d tokens", dispatches, npos)
		}
	} else if dispatches != 1 {
		t.Fatalf("default budget unexpectedly split the GEMM into %d dispatches", dispatches)
	}
}

// TestBatchAttentionQueryTiling is the attention counterpart: the batched
// attention kernel puts one workgroup on each (head, position) and walks the
// whole key range, so a single dispatch is quadratic in the context. It is now
// split along the QUERY axis, which must be numerically invisible and must
// actually happen when the per-dispatch budget is tight.
//
// Self-contained: no model file, synthetic Q/K/V. Run twice, as above, with
// DLGO_VK_ATTN_UNITS=1 for the split path.
func TestBatchAttentionQueryTiling(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	const npos, nh, nkv, hd = 8, 4, 2, 8
	const kvCap = 32
	kvDim := nkv * hd
	qDim := nh * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	rng := rand.New(rand.NewSource(21))
	q := make([]float32, npos*qDim)
	k := make([]float32, npos*kvDim)
	v := make([]float32, npos*kvDim)
	for i := range q {
		q[i] = rng.Float32()*2 - 1
	}
	for i := range k {
		k[i] = rng.Float32()*2 - 1
	}
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}

	alloc := func(n uint64) Buf {
		b := Alloc(n)
		if b == 0 {
			t.Fatalf("alloc %d bytes failed", n)
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	qb := alloc(uint64(len(q)) * 4)
	kb := alloc(uint64(len(k)) * 4)
	vb := alloc(uint64(len(v)) * 4)
	// Packed-f16 KV cache: kvDim halves = kvDim*2 bytes per position.
	kc := alloc(uint64(kvCap) * uint64(kvDim) * 2)
	vc := alloc(uint64(kvCap) * uint64(kvDim) * 2)
	ob := alloc(uint64(npos) * uint64(qDim) * 4)

	BeginBatch()
	if err := UploadF32(qb, q); err != nil {
		t.Fatal(err)
	}
	if err := UploadF32(kb, k); err != nil {
		t.Fatal(err)
	}
	if err := UploadF32(vb, v); err != nil {
		t.Fatal(err)
	}
	Barrier()
	if err := BatchKVStoreF16(kc, vc, kb, vb, 0, kvDim, npos); err != nil {
		t.Fatal(err)
	}
	EndBatch()
	Sync()

	// startSeqLen=1 makes position p attend to keys 0..p (causal).
	PerfReset()
	BeginBatch()
	Barrier()
	if err := BatchAttentionF16(ob, qb, kc, vc, nh, nkv, hd, kvDim, 1, scale, 0, npos, 0); err != nil {
		t.Fatal(err)
	}
	EndBatch()
	Sync()
	dispatches := PerfGetDispatches()

	got := make([]float32, npos*qDim)
	if err := DownloadF32(ob, got); err != nil {
		t.Fatal(err)
	}

	// CPU reference: causal softmax attention, GQA-mapped kv heads.
	want := make([]float32, npos*qDim)
	kvMul := nh / nkv
	for pos := 0; pos < npos; pos++ {
		for h := 0; h < nh; h++ {
			kh := h / kvMul
			keys := pos + 1
			scores := make([]float64, keys)
			best := math.Inf(-1)
			for tk := 0; tk < keys; tk++ {
				var s float64
				for d := 0; d < hd; d++ {
					s += float64(q[pos*qDim+h*hd+d]) * float64(k[tk*kvDim+kh*hd+d])
				}
				s *= float64(scale)
				scores[tk] = s
				if s > best {
					best = s
				}
			}
			var sum float64
			for tk := range scores {
				scores[tk] = math.Exp(scores[tk] - best)
				sum += scores[tk]
			}
			for d := 0; d < hd; d++ {
				var acc float64
				for tk := 0; tk < keys; tk++ {
					acc += scores[tk] / sum * float64(v[tk*kvDim+kh*hd+d])
				}
				want[pos*qDim+h*hd+d] = float32(acc)
			}
		}
	}

	var refL2, diffL2, maxErr float64
	for i := range got {
		refV := float64(want[i])
		refL2 += refV * refV
		d := float64(got[i]) - refV
		diffL2 += d * d
		if a := math.Abs(d); a > maxErr {
			maxErr = a
		}
	}
	relL2 := math.Sqrt(diffL2 / refL2)
	t.Logf("dispatches=%d relL2=%.3e maxErr=%.3e", dispatches, relL2, maxErr)

	// The KV cache is f16, so ~1e-3 relative error against the f32 reference.
	if relL2 > 5e-3 {
		t.Fatalf("relL2 %.3e exceeds 5e-3", relL2)
	}
	if os.Getenv("DLGO_VK_ATTN_UNITS") != "" {
		if dispatches != npos {
			t.Fatalf("query tiling did not engage: %d dispatch(es), want %d (one per query)",
				dispatches, npos)
		}
	} else if dispatches != 1 {
		t.Fatalf("default budget unexpectedly split attention into %d dispatches", dispatches)
	}
}
