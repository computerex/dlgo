//go:build cgo && vulkan

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// TestBatchRMSNormHeadsParity checks the position-batched per-head RMSNorm used
// for the q/k norm in the batched prefill against the per-token form.
//
// rmsnorm_heads.comp derives its row stride from gl_NumWorkGroups.x, so a wrong
// groups_x silently normalises the wrong head range for every position past 0
// while position 0 still looks correct — the exact signature the prefill
// bisection showed (n=1 exact, n=2 wrong at the first attention layer).
func TestBatchRMSNormHeadsParity(t *testing.T) {
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
		if m.Layers[l].Spec.Core == llm.CoreAttention && m.Layers[l].Spec.QKNorm {
			li = l
			break
		}
	}
	if li < 0 {
		t.Skip("no qk-norm attention layer")
	}
	gm, err := UploadModel(m, li+1)
	if err != nil {
		t.Fatalf("upload: %v", err)
	}
	t.Cleanup(gm.FreeAll)

	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	qDim := numHeads * headDim
	kvDim := numKVHeads * headDim
	dim := cfg.EmbeddingDim
	ffnDim := cfg.FFNDim
	gl := &gm.Layers[li]
	const npos = 4

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

	// One buffer for npos rows of q, one for npos rows of k.
	q := make([]float32, npos*qDim)
	kk := make([]float32, npos*kvDim)
	seed := uint32(5150)
	next := func() float32 {
		seed = seed*1664525 + 1013904223
		return (float32(seed>>16)/65536.0 - 0.5) * 4.0
	}
	for i := range q {
		q[i] = next()
	}
	for i := range kk {
		kk[i] = next()
	}

	runBatch := func(in []float32, dst Buf, nh, rowsPerPos int) []float32 {
		if err := UploadF32(dst, in); err != nil {
			t.Fatalf("upload: %v", err)
		}
		Barrier()
		var w Buf
		if nh == numHeads {
			w = gl.AttnQNorm
		} else {
			w = gl.AttnKNorm
		}
		if err := BatchRMSNormHeads(dst, w, nh, headDim, cfg.RMSNormEps, npos); err != nil {
			t.Fatalf("batch rmsnorm heads: %v", err)
		}
		Sync()
		out := make([]float32, len(in))
		if err := DownloadF32(dst, out); err != nil {
			t.Fatalf("dl: %v", err)
		}
		return out
	}

	BeginBatch()
	gotQ := runBatch(q, bs.Q, numHeads, qDim)
	gotK := runBatch(kk, bs.K, numKVHeads, kvDim)

	// Per-token reference.
	wantQ := make([]float32, len(q))
	wantK := make([]float32, len(kk))
	row := make([]float32, qDim)
	for pos := 0; pos < npos; pos++ {
		if err := UploadF32(rs.Q, q[pos*qDim:(pos+1)*qDim]); err != nil {
			t.Fatalf("upload q row: %v", err)
		}
		Barrier()
		if err := RMSNormHeads(rs.Q, gl.AttnQNorm, numHeads, headDim, cfg.RMSNormEps); err != nil {
			t.Fatalf("qnorm: %v", err)
		}
		Sync()
		if err := DownloadF32(rs.Q, row); err != nil {
			t.Fatalf("dl q: %v", err)
		}
		copy(wantQ[pos*qDim:(pos+1)*qDim], row)
	}
	krow := make([]float32, kvDim)
	for pos := 0; pos < npos; pos++ {
		if err := UploadF32(rs.K, kk[pos*kvDim:(pos+1)*kvDim]); err != nil {
			t.Fatalf("upload k row: %v", err)
		}
		Barrier()
		if err := RMSNormHeads(rs.K, gl.AttnKNorm, numKVHeads, headDim, cfg.RMSNormEps); err != nil {
			t.Fatalf("knorm: %v", err)
		}
		Sync()
		if err := DownloadF32(rs.K, krow); err != nil {
			t.Fatalf("dl k: %v", err)
		}
		copy(wantK[pos*kvDim:(pos+1)*kvDim], krow)
	}

	stat := func(name string, got, want []float32, perPos int) {
		var maxAbs float64
		for i := range got {
			maxAbs = math.Max(maxAbs, math.Abs(float64(got[i]-want[i])))
		}
		worstPos := -1
		for pos := 0; pos < npos; pos++ {
			var pa float64
			for c := 0; c < perPos; c++ {
				i := pos*perPos + c
				pa = math.Max(pa, math.Abs(float64(got[i]-want[i])))
			}
			if pa > 1e-6 {
				worstPos = pos
			}
			t.Logf("%s pos %d: maxAbs=%.3e", name, pos, pa)
		}
		t.Logf("%s overall maxAbs=%.3e firstWrongPos=%d", name, maxAbs, worstPos)
		if maxAbs > 1e-5 {
			t.Errorf("%s batched per-head norm differs: maxAbs=%.3e", name, maxAbs)
		}
	}
	stat("qnorm", gotQ, wantQ, qDim)
	stat("knorm", gotK, wantK, kvDim)
}
