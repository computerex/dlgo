//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math"
	"testing"
	"time"
)

// TestProbePrefillAttn isolates why the model's prefill attention costs ~4.8 s at
// 16k while a single 16k call of the same kernel costs 130 ms for the same total
// work. It compares one whole-prompt call against the chunked sequence the
// pipeline actually issues.
func TestProbePrefillAttn(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	numHeads, numKVHeads, headDim := 32, 8, 128
	kvDim := numKVHeads * headDim
	total := benchEnvInt("PROBE_SEQ", 16384)
	chunk := benchEnvInt("PROBE_CHUNK", 1024)

	alloc := func(n uint64) Buf {
		b := Alloc(n)
		if b == 0 {
			t.Fatalf("alloc %d", n)
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	qf := make([]float32, total*numHeads*headDim)
	for i := range qf {
		qf[i] = float32(math.Sin(float64(i) * 0.001))
	}
	Q := alloc(uint64(len(qf)) * 4)
	if err := UploadF32(Q, qf); err != nil {
		t.Fatal(err)
	}
	O := alloc(uint64(len(qf)) * 4)
	kv := make([]byte, total*kvDim*2)
	for i := range kv {
		kv[i] = byte(i*7 + 1)
	}
	K := alloc(uint64(len(kv)))
	V := alloc(uint64(len(kv)))
	if err := Upload(K, kv); err != nil {
		t.Fatal(err)
	}
	if err := Upload(V, kv); err != nil {
		t.Fatal(err)
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	timeIt := func(name string, calls int, fn func()) {
		for i := 0; i < 2; i++ {
			fn()
		}
		Sync()
		best := 1e18
		for s := 0; s < 3; s++ {
			st := time.Now()
			fn()
			Sync()
			if d := time.Since(st).Seconds(); d < best {
				best = d
			}
		}
		t.Logf("%-28s %8.3f ms  (%d calls)", name, best*1e3, calls)
	}

	// (a) one call covering the whole prompt.
	timeIt("whole-prompt single call", 1, func() {
		if err := cudaBatchAttentionF16(O, Q, K, V, numHeads, numKVHeads, headDim, kvDim,
			0, total, scale, 0); err != nil {
			t.Fatal(err)
		}
	})

	// (b) the chunked sequence the pipeline issues.
	timeIt("chunked", (total+chunk-1)/chunk, func() {
		for cs := 0; cs < total; cs += chunk {
			ce := cs + chunk
			if ce > total {
				ce = total
			}
			if err := cudaBatchAttentionF16(O, Q, K, V, numHeads, numKVHeads, headDim, kvDim,
				cs, ce-cs, scale, 0); err != nil {
				t.Fatal(err)
			}
		}
	})

	// (c) chunk-sized queries but every chunk scanning the FULL key range, to
	// separate "more calls" from "a longer key range per call".
	timeIt("chunk queries, full keys", 1, func() {
		if err := cudaBatchAttentionF16(O, Q, K, V, numHeads, numKVHeads, headDim, kvDim,
			total-chunk, chunk, scale, 0); err != nil {
			t.Fatal(err)
		}
	})
	fmt.Println("done")
}
