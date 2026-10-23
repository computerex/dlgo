//go:build cgo && vulkan

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// TestShortConvLayerBatchParity checks the short-convolution core alone at
// npos > 1: batched in_proj -> ShortConv(npos) -> out_proj against the per-token
// form (in_proj -> ShortConv(npos=1) -> out_proj, state carried between tokens),
// which is what the decode path does.
//
// This is the first cross-position kernel a prompt hits (layer 0 of LFM2 is a
// short-conv layer), so it is the prime suspect when a 1-token chunk matches the
// per-token path exactly and a 2-token chunk does not.
func TestShortConvLayerBatchParity(t *testing.T) {
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

	// First short-conv layer.
	li := -1
	for l := 0; l < cfg.NumLayers; l++ {
		if m.Layers[l].Spec.Core == llm.CoreShortConv {
			li = l
			break
		}
	}
	if li < 0 {
		t.Skip("no shortconv layer")
	}
	gm, err := UploadModel(m, li+1)
	if err != nil {
		t.Fatalf("upload: %v", err)
	}
	t.Cleanup(gm.FreeAll)

	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim
	gl := &gm.Layers[li]
	scK := cfg.ShortConvLCache
	if scK < 1 {
		scK = 3
	}
	const npos = 4

	rs, err := NewGpuRunState(dim, qDim, kvDim, ffnDim, cfg.VocabSize)
	if err != nil {
		t.Fatalf("run state: %v", err)
	}
	t.Cleanup(rs.FreeAll)
	if err := rs.AllocShortConvScratch(dim); err != nil {
		t.Fatalf("shortconv scratch: %v", err)
	}
	bs, err := NewGpuBatchState(npos, dim, qDim, kvDim, ffnDim)
	if err != nil {
		t.Fatalf("batch state: %v", err)
	}
	t.Cleanup(bs.Free)
	if err := bs.AllocShortConvBatch(npos, dim); err != nil {
		t.Fatalf("shortconv batch scratch: %v", err)
	}
	// The conv output borrows MoEOut, which AllocMoEBatch provides.
	if err := bs.AllocMoEBatch(npos, dim, cfg.ExpertCount); err != nil {
		t.Fatalf("moe batch scratch: %v", err)
	}

	// Deterministic input rows.
	x := make([]float32, npos*dim)
	seed := uint32(4242)
	for i := range x {
		seed = seed*1664525 + 1013904223
		x[i] = (float32(seed>>16)/65536.0 - 0.5) * 2.0
	}

	resetConv := func() {
		BeginBatch()
		ZeroFill(gl.ShortConvState, uint64((scK-1)*dim*4))
		EndBatch()
		Sync()
	}

	// ---- batched ----
	resetConv()
	got := make([]float32, npos*dim)
	BeginBatch()
	if err := UploadF32(bs.XNorm, x); err != nil {
		t.Fatalf("upload: %v", err)
	}
	Barrier()
	if err := BatchMatVec(bs.ShortConvBCX, gl.ShortConvInProj.Buf, bs.XNorm,
		3*dim, dim, npos, gl.ShortConvInProj.Type); err != nil {
		t.Fatalf("batch in_proj: %v", err)
	}
	Barrier()
	if err := ShortConv(bs.ShortConvBCX, bs.MoEOut, gl.ShortConvState,
		gl.ShortConvConv, dim, scK, npos); err != nil {
		t.Fatalf("batch conv: %v", err)
	}
	Barrier()
	if err := BatchMatVec(bs.AttnProj, gl.ShortConvOutProj.Buf, bs.MoEOut,
		dim, dim, npos, gl.ShortConvOutProj.Type); err != nil {
		t.Fatalf("batch out_proj: %v", err)
	}
	Sync()
	if err := DownloadF32(bs.AttnProj, got); err != nil {
		t.Fatalf("dl batched: %v", err)
	}

	// ---- per-token reference ----
	resetConv()
	want := make([]float32, npos*dim)
	row := make([]float32, dim)
	for pos := 0; pos < npos; pos++ {
		BeginBatch()
		if err := UploadF32(rs.XNorm, x[pos*dim:(pos+1)*dim]); err != nil {
			t.Fatalf("upload row: %v", err)
		}
		Barrier()
		if err := MatVec(rs.ShortConvBCX, gl.ShortConvInProj.Buf, rs.XNorm,
			3*dim, dim, gl.ShortConvInProj.Type); err != nil {
			t.Fatalf("in_proj: %v", err)
		}
		Barrier()
		if err := ShortConv(rs.ShortConvBCX, rs.ShortConvY, gl.ShortConvState,
			gl.ShortConvConv, dim, scK, 1); err != nil {
			t.Fatalf("conv: %v", err)
		}
		Barrier()
		if err := MatVec(rs.AttnProj, gl.ShortConvOutProj.Buf, rs.ShortConvY,
			dim, dim, gl.ShortConvOutProj.Type); err != nil {
			t.Fatalf("out_proj: %v", err)
		}
		Sync()
		if err := DownloadF32(rs.AttnProj, row); err != nil {
			t.Fatalf("dl row: %v", err)
		}
		copy(want[pos*dim:(pos+1)*dim], row)
	}

	var maxAbs, sumAbs, sumMag float64
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		maxAbs = math.Max(maxAbs, d)
		sumAbs += d
		sumMag += math.Abs(float64(want[i]))
	}
	// Report per position so a scan-order bug shows up as a growing error.
	for pos := 0; pos < npos; pos++ {
		var pa, pm float64
		for c := 0; c < dim; c++ {
			i := pos*dim + c
			pa = math.Max(pa, math.Abs(float64(got[i]-want[i])))
			pm += math.Abs(float64(want[i]))
		}
		var ps float64
		for c := 0; c < dim; c++ {
			ps += math.Abs(float64(got[pos*dim+c] - want[pos*dim+c]))
		}
		t.Logf("pos %d: maxAbs=%.3e meanRel=%.3e", pos, pa, ps/math.Max(pm, 1e-9))
	}
	t.Logf("shortconv layer batch vs per-token: maxAbs=%.3e meanRel=%.3e", maxAbs, sumAbs/math.Max(sumMag, 1e-9))
	if maxAbs > 2e-4 {
		t.Fatalf("shortconv batch core differs: maxAbs=%.3e", maxAbs)
	}
}
