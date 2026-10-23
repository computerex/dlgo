//go:build cgo && vulkan

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// TestAttentionBatchParity checks the batched attention kernel (npos queries at
// once) against the per-token attention the decode path uses, for the same KV.
// The short-conv scan and the batched MoE are both already exact, so this is the
// remaining cross-position kernel of the batched prefill.
func TestAttentionBatchParity(t *testing.T) {
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
	m := p.Model
	cfg := m.Config

	li := -1
	for l := 0; l < cfg.NumLayers; l++ {
		if m.Layers[l].Spec.Core == llm.CoreAttention {
			li = l
			break
		}
	}
	if li < 0 {
		t.Skip("no attention layer")
	}
	gm, err := UploadModel(m, li+1)
	if err != nil {
		t.Fatalf("upload: %v", err)
	}
	t.Cleanup(gm.FreeAll)

	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim
	qDim := numHeads * headDim
	ffnDim := cfg.FFNDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	const npos = 4

	kv, err := NewGpuKVCache(cfg.NumLayers, cfg.NumLayers, 256, kvDim, nil)
	if err != nil {
		t.Fatalf("kv: %v", err)
	}
	rs, err := NewGpuRunState(dim, qDim, kvDim, ffnDim, cfg.VocabSize)
	if err != nil {
		t.Fatalf("run state: %v", err)
	}
	t.Cleanup(rs.FreeAll)
	bs, err := NewGpuBatchState(npos, dim, qDim, kvDim, ffnDim)
	if err != nil {
		t.Fatalf("batch state: %v", err)
	}
	t.Cleanup(bs.Free)

	q := make([]float32, npos*qDim)
	k := make([]float32, npos*kvDim)
	v := make([]float32, npos*kvDim)
	seed := uint32(77)
	next := func() float32 {
		seed = seed*1664525 + 1013904223
		return (float32(seed>>16)/65536.0 - 0.5) * 2.0
	}
	for i := range q {
		q[i] = next()
	}
	for i := range k {
		k[i] = next()
	}
	for i := range v {
		v[i] = next()
	}

	// Fill the KV cache once for both paths.
	BeginBatch()
	if err := UploadF32(bs.Q, q); err != nil {
		t.Fatalf("upload q: %v", err)
	}
	if err := UploadF32(bs.K, k); err != nil {
		t.Fatalf("upload k: %v", err)
	}
	if err := UploadF32(bs.V, v); err != nil {
		t.Fatalf("upload v: %v", err)
	}
	Barrier()
	if err := BatchKVStoreF16(kv.KeyBufs[li], kv.ValBufs[li], bs.K, bs.V, 0, kvDim, npos); err != nil {
		t.Fatalf("kvstore: %v", err)
	}
	Barrier()
	if err := BatchAttentionF16(bs.AttnOut, bs.Q, kv.KeyBufs[li], kv.ValBufs[li],
		numHeads, numKVHeads, headDim, kvDim, 1, scale, 0, npos); err != nil {
		t.Fatalf("batch attention: %v", err)
	}
	Sync()
	got := make([]float32, npos*qDim)
	if err := DownloadF32(bs.AttnOut, got); err != nil {
		t.Fatalf("dl batched: %v", err)
	}

	// Per-token reference against the same KV cache.
	want := make([]float32, npos*qDim)
	row := make([]float32, qDim)
	for pos := 0; pos < npos; pos++ {
		BeginBatch()
		if err := UploadF32(rs.Q, q[pos*qDim:(pos+1)*qDim]); err != nil {
			t.Fatalf("upload q row: %v", err)
		}
		Barrier()
		if err := AttentionF16(rs.AttnProj, rs.Q, kv.KeyBufs[li], kv.ValBufs[li],
			numHeads, numKVHeads, headDim, kvDim, pos+1, 0, scale, 0); err != nil {
			t.Fatalf("attention pos %d: %v", pos, err)
		}
		Sync()
		if err := DownloadF32(rs.AttnProj, row); err != nil {
			t.Fatalf("dl row: %v", err)
		}
		copy(want[pos*qDim:(pos+1)*qDim], row)
	}

	var maxAbs float64
	for pos := 0; pos < npos; pos++ {
		var pa float64
		for c := 0; c < qDim; c++ {
			i := pos*qDim + c
			pa = math.Max(pa, math.Abs(float64(got[i]-want[i])))
		}
		t.Logf("pos %d: maxAbs=%.3e", pos, pa)
		maxAbs = math.Max(maxAbs, pa)
	}
	t.Logf("attention batch vs per-token: maxAbs=%.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("batched attention differs: maxAbs=%.3e", maxAbs)
	}
}
