//go:build cgo && vulkan

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// TestKVStoreBatchParity checks BatchKVStoreF16 (npos rows in one dispatch)
// against the per-token KVStoreF16 the decode path uses.
//
// This is the one step of the batched prefill that the attention parity test
// cannot see: that test feeds both paths the *same* cache, so a wrong batched
// store would still agree with itself. A bad store corrupts exactly the second
// and later positions of an attention layer, which is the signature the prefill
// bisection showed (n=1 exact, n=2 wrong, first divergence at the first
// attention layer).
func TestKVStoreBatchParity(t *testing.T) {
	const modelPath = `C:\Users\mohd\Downloads\LFM2.5-24B-A2B-QAD-Q4_0.gguf`
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not present: %v", err)
	}
	if err := Init(); err != nil {
		t.Skipf("no GPU: %v", err)
	}
	t.Cleanup(Shutdown)

	p, err := llm.NewPipeline(modelPath, 256)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	cfg := p.Model.Config
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim
	const npos = 4
	const ctxLen = 64
	const startPos = 3 // non-zero, so a stride bug cannot hide behind offset 0

	batchKV, err := NewGpuKVCache(1, 1, ctxLen, kvDim, nil)
	if err != nil {
		t.Fatalf("batch kv: %v", err)
	}
	perKV, err := NewGpuKVCache(1, 1, ctxLen, kvDim, nil)
	if err != nil {
		t.Fatalf("per kv: %v", err)
	}
	bs, err := NewGpuBatchState(npos, dim, qDim, kvDim, ffnDim)
	if err != nil {
		t.Fatalf("batch state: %v", err)
	}
	t.Cleanup(bs.Free)
	rs, err := NewGpuRunState(dim, qDim, kvDim, ffnDim, cfg.VocabSize)
	if err != nil {
		t.Fatalf("run state: %v", err)
	}
	t.Cleanup(rs.FreeAll)

	k := make([]float32, npos*kvDim)
	v := make([]float32, npos*kvDim)
	seed := uint32(31337)
	next := func() float32 {
		seed = seed*1664525 + 1013904223
		return (float32(seed>>16)/65536.0 - 0.5) * 2.0
	}
	for i := range k {
		k[i] = next()
	}
	for i := range v {
		v[i] = next()
	}

	// Batched store.
	BeginBatch()
	if err := UploadF32(bs.K, k); err != nil {
		t.Fatalf("upload k: %v", err)
	}
	if err := UploadF32(bs.V, v); err != nil {
		t.Fatalf("upload v: %v", err)
	}
	Barrier()
	if err := BatchKVStoreF16(batchKV.KeyBufs[0], batchKV.ValBufs[0], bs.K, bs.V,
		startPos, kvDim, npos); err != nil {
		t.Fatalf("batch kv store: %v", err)
	}
	Sync()

	// Per-token store into a separate cache.
	for pos := 0; pos < npos; pos++ {
		BeginBatch()
		if err := UploadF32(rs.K, k[pos*kvDim:(pos+1)*kvDim]); err != nil {
			t.Fatalf("upload k row: %v", err)
		}
		if err := UploadF32(rs.V, v[pos*kvDim:(pos+1)*kvDim]); err != nil {
			t.Fatalf("upload v row: %v", err)
		}
		Barrier()
		if err := KVStoreF16(perKV.KeyBufs[0], perKV.ValBufs[0], rs.K, rs.V, startPos+pos, kvDim); err != nil {
			t.Fatalf("kv store: %v", err)
		}
		Sync()
	}

	span := uint64(ctxLen * kvDim * 2) // FP16 cache, whole layer
	gotK := make([]byte, span)
	gotV := make([]byte, span)
	wantK := make([]byte, span)
	wantV := make([]byte, span)
	Sync()
	if err := Download(batchKV.KeyBufs[0], gotK); err != nil {
		t.Fatalf("dl batch k: %v", err)
	}
	if err := Download(batchKV.ValBufs[0], gotV); err != nil {
		t.Fatalf("dl batch v: %v", err)
	}
	if err := Download(perKV.KeyBufs[0], wantK); err != nil {
		t.Fatalf("dl per k: %v", err)
	}
	if err := Download(perKV.ValBufs[0], wantV); err != nil {
		t.Fatalf("dl per v: %v", err)
	}

	firstK, firstV := -1, -1
	for i := 0; i < len(gotK); i++ {
		if gotK[i] != wantK[i] {
			firstK = i
			break
		}
	}
	for i := 0; i < len(gotV); i++ {
		if gotV[i] != wantV[i] {
			firstV = i
			break
		}
	}
	if firstK >= 0 || firstV >= 0 {
		t.Fatalf("batched KV store differs from per-token: firstK=%d firstV=%d (kvDim=%d, byte stride per pos=%d)",
			firstK, firstV, kvDim, kvDim*2)
	}
	// Also confirm the data actually landed where expected.
	want := float16Bytes(k[0])
	if gotK[startPos*kvDim*2] != want[0] || gotK[startPos*kvDim*2+1] != want[1] {
		t.Fatalf("batch store did not write row %d at the expected offset", startPos)
	}
	_ = math.Abs
	t.Logf("batched KV store matches per-token exactly (span=%d bytes)", span)
}

// float16Bytes converts one float32 to its IEEE fp16 bit pattern as 2 bytes.
func float16Bytes(f float32) [2]byte {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int32((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff
	var h uint16
	if exp <= 0 {
		h = sign
	} else if exp >= 31 {
		h = sign | 0x7c00
	} else {
		h = sign | uint16(exp)<<10 | uint16(mant>>13)
	}
	return [2]byte{byte(h), byte(h >> 8)}
}
