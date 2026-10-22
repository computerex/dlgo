//go:build cgo && vulkan

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// TestMoEBatchParity checks gpuForwardMoEBatch (one GEMM per expert over a chunk)
// against the per-token GpuForwardMoEFFN path on real LFM2.5-24B-A2B weights.
//
// The two paths cannot be bit-identical: the per-token kernel sums the selected
// experts in slot order, the batched one accumulates them in expert order, so the
// fp32 addition order differs. Everything else (router, gating, top-k, expert
// arithmetic) must agree, so the tolerance is on the relative error, not equality.
func TestMoEBatchParity(t *testing.T) {
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

	// Find the first MoE layer; uploading through it is enough to exercise it.
	moeLayer := -1
	for l := 0; l < cfg.NumLayers; l++ {
		if m.Layers[l].Spec.FFN == llm.FFNMoE {
			moeLayer = l
			break
		}
	}
	if moeLayer < 0 {
		t.Skip("model has no MoE layer")
	}

	gm, err := UploadModel(m, moeLayer+1)
	if err != nil {
		t.Fatalf("upload model: %v", err)
	}
	t.Cleanup(gm.FreeAll)

	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim

	gl := &gm.Layers[moeLayer]
	layer := &m.Layers[moeLayer]
	if !moeBatchSupported(layer, gl, cfg) {
		t.Skipf("layer %d does not support the batched MoE path", moeLayer)
	}

	const npos = 16
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
	if err := bs.AllocMoEBatch(npos, dim, cfg.ExpertCount); err != nil {
		t.Fatalf("moe batch scratch: %v", err)
	}

	// Deterministic pseudo-random input with a realistic magnitude.
	x := make([]float32, npos*dim)
	seed := uint32(12345)
	for i := range x {
		seed = seed*1664525 + 1013904223
		x[i] = (float32(seed>>16)/65536.0 - 0.5) * 4.0
	}

	// Batched path.
	BeginBatch()
	if err := UploadF32(bs.FFNNorm, x); err != nil {
		t.Fatalf("upload: %v", err)
	}
	Barrier()
	if err := gpuForwardMoEBatch(layer, gl, bs, cfg, npos); err != nil {
		t.Fatalf("batched moe: %v", err)
	}
	Sync()
	got := make([]float32, npos*dim)
	if err := DownloadF32(bs.FFNOut, got); err != nil {
		t.Fatalf("download batch out: %v", err)
	}

	// Reference: the per-token path, one position at a time.
	want := make([]float32, npos*dim)
	row := make([]float32, dim)
	for pos := 0; pos < npos; pos++ {
		BeginBatch()
		if err := UploadF32(rs.FFNNorm, x[pos*dim:(pos+1)*dim]); err != nil {
			t.Fatalf("upload row: %v", err)
		}
		Barrier()
		if err := GpuForwardMoEFFN(gl, layer, rs, cfg, false, moeLayer, pos); err != nil {
			t.Fatalf("per-token moe: %v", err)
		}
		Sync()
		if err := DownloadF32(rs.FFNOut, row); err != nil {
			t.Fatalf("download row: %v", err)
		}
		copy(want[pos*dim:(pos+1)*dim], row)
	}

	var maxAbs, maxRel, sumAbs, sumMag float64
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		if d > maxAbs {
			maxAbs = d
		}
		mag := math.Abs(float64(want[i]))
		if mag > 1e-6 {
			if r := d / mag; r > maxRel {
				maxRel = r
			}
		}
		sumAbs += d
		sumMag += mag
	}
	meanRel := sumAbs / math.Max(sumMag, 1e-9)
	t.Logf("moebatch parity: maxAbs=%.3e maxRel=%.3e meanRel=%.3e", maxAbs, maxRel, meanRel)

	if maxAbs > 2e-2 {
		t.Fatalf("max abs diff %.3e too large", maxAbs)
	}
	if meanRel > 5e-3 {
		t.Fatalf("mean relative diff %.3e too large", meanRel)
	}
}
