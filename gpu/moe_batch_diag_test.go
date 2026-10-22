//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// TestMoEBatchDiag isolates the pieces of the position-batched MoE: the batched
// router matvec, the per-expert offset matvec, and the gather/scatter kernels.
// Kept next to TestMoEBatchParity as the bisection tool for it.
func TestMoEBatchDiag(t *testing.T) {
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

	moeLayer := -1
	for l := 0; l < cfg.NumLayers; l++ {
		if m.Layers[l].Spec.FFN == llm.FFNMoE {
			moeLayer = l
			break
		}
	}
	if moeLayer < 0 {
		t.Skip("no MoE layer")
	}

	gm, err := UploadModel(m, moeLayer+1)
	if err != nil {
		t.Fatalf("upload: %v", err)
	}
	t.Cleanup(gm.FreeAll)

	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim
	expDim := cfg.ExpertFFNDim
	nExperts := cfg.ExpertCount
	gl := &gm.Layers[moeLayer]
	layer := &m.Layers[moeLayer]

	// Report the layout the executor will assume.
	t.Logf("fused=%v gateExp=(%d,%d) upExp=(%d,%d) gateUp=(%v) down=(%d,%d) align=%d strides=%v",
		gl.FFNGateUpExps != nil,
		gl.FFNGateExps.Rows, gl.FFNGateExps.Cols,
		gl.FFNUpExps.Rows, gl.FFNUpExps.Cols,
		gl.FFNGateUpExps != nil,
		gl.FFNDownExps.Rows, gl.FFNDownExps.Cols,
		MinSSBOOffsetAlign(), moeBatchExpertStrides(layer, gl, cfg))

	// ---- 1. batched router matvec vs per-token matvec ----
	const npos = 4
	x := make([]float32, npos*dim)
	seed := uint32(999)
	for i := range x {
		seed = seed*1664525 + 1013904223
		x[i] = (float32(seed>>16)/65536.0 - 0.5) * 2.0
	}
	bs, err := NewGpuBatchState(npos, dim, qDim, kvDim, ffnDim)
	if err != nil {
		t.Fatalf("batch state: %v", err)
	}
	t.Cleanup(bs.Free)
	if err := bs.AllocMoEBatch(npos, dim, nExperts); err != nil {
		t.Fatalf("scratch: %v", err)
	}
	rs, err := NewGpuRunState(dim, qDim, kvDim, ffnDim, cfg.VocabSize)
	if err != nil {
		t.Fatalf("run state: %v", err)
	}
	t.Cleanup(rs.FreeAll)
	if err := initMoEScratchBuffers(rs, cfg); err != nil {
		t.Fatalf("moe scratch: %v", err)
	}

	gotLogits := make([]float32, npos*nExperts)
	BeginBatch()
	if err := UploadF32(bs.FFNNorm, x); err != nil {
		t.Fatalf("upload: %v", err)
	}
	if err := BatchMatVec(bs.MoELogits, gl.FFNRouter.Buf, bs.FFNNorm,
		nExperts, dim, npos, gl.FFNRouter.Type); err != nil {
		t.Fatalf("router batch: %v", err)
	}
	Sync()
	if err := DownloadF32(bs.MoELogits, gotLogits); err != nil {
		t.Fatalf("dl: %v", err)
	}

	wantLogits := make([]float32, npos*nExperts)
	rrow := make([]float32, nExperts)
	for pos := 0; pos < npos; pos++ {
		BeginBatch()
		if err := UploadF32(rs.FFNNorm, x[pos*dim:(pos+1)*dim]); err != nil {
			t.Fatalf("upload row: %v", err)
		}
		if err := MatVec(rs.MoELogits, gl.FFNRouter.Buf, rs.FFNNorm,
			nExperts, dim, gl.FFNRouter.Type); err != nil {
			t.Fatalf("router matvec: %v", err)
		}
		Sync()
		if err := DownloadF32(rs.MoELogits, rrow); err != nil {
			t.Fatalf("dl row: %v", err)
		}
		copy(wantLogits[pos*nExperts:(pos+1)*nExperts], rrow)
	}
	t.Logf("router batched vs per-token: %s", diffStat(gotLogits, wantLogits))

	// ---- 2. per-expert offset matvec: npos=1 offset call vs explicit offset ----
	e := 3
	strides := moeBatchExpertStrides(layer, gl, cfg)
	xe := make([]float32, dim)
	for i := range xe {
		xe[i] = x[i]
	}
	gateA := make([]float32, expDim)
	gateB := make([]float32, expDim)
	gateC := make([]float32, expDim)
	blocksPerExpert := moeBlockStride(expDim, gl.FFNGateExps.Cols, gl.FFNGateExps.Type)
	BeginBatch()
	if err := UploadF32(bs.MoEXE, xe); err != nil {
		t.Fatalf("upload xe: %v", err)
	}
	if err := UploadF32(bs.MoEW, []float32{float32(e)}); err != nil {
		t.Fatalf("upload eidx: %v", err)
	}
	if err := BatchMatVecOffset(bs.Gate, gl.FFNGateExps.Buf, bs.MoEXE, expDim, dim, 1,
		gl.FFNGateExps.Type, uint64(e)*uint64(strides[0])); err != nil {
		t.Fatalf("batch offset: %v", err)
	}
	if err := MatVecOffset(bs.Up, 0, gl.FFNGateExps.Buf, e*strides[0], bs.MoEXE,
		expDim, dim, gl.FFNGateExps.Type); err != nil {
		t.Fatalf("matvec offset: %v", err)
	}
	// Same slice addressed the way the reference MoE does it: block units via a
	// GPU-side expert-index buffer.
	if err := MoEMatVecNative(bs.Hidden, gl.FFNGateExps.Buf, bs.MoEXE, bs.MoEW,
		expDim, dim, gl.FFNGateExps.Type, blocksPerExpert, 0, 0, 1); err != nil {
		t.Fatalf("moe matvec native: %v", err)
	}
	Sync()
	if err := DownloadF32(bs.Gate, gateA); err != nil {
		t.Fatalf("dl gate: %v", err)
	}
	if err := DownloadF32(bs.Up, gateB); err != nil {
		t.Fatalf("dl gate2: %v", err)
	}
	if err := DownloadF32(bs.Hidden, gateC); err != nil {
		t.Fatalf("dl gate3: %v", err)
	}
	t.Logf("expert matvec (batched-offset vs matvec-offset): %s", diffStat(gateA, gateB))
	t.Logf("expert matvec (byte-offset vs reference block-offset): %s", diffStat(gateA, gateC))

	// Anchor both conventions at expert 0, where the offset is zero: any
	// difference here is a kernel/layout difference rather than an offset one.
	gateP := make([]float32, expDim)
	gateQ := make([]float32, expDim)
	gateR := make([]float32, expDim)
	BeginBatch()
	if err := MatVec(bs.Gate, gl.FFNGateExps.Buf, bs.MoEXE, expDim, dim, gl.FFNGateExps.Type); err != nil {
		t.Fatalf("plain matvec: %v", err)
	}
	if err := MatVecOffset(bs.Up, 0, gl.FFNGateExps.Buf, 0, bs.MoEXE, expDim, dim, gl.FFNGateExps.Type); err != nil {
		t.Fatalf("offset0 matvec: %v", err)
	}
	if err := UploadF32(bs.MoEW, []float32{0}); err != nil {
		t.Fatalf("upload eidx0: %v", err)
	}
	if err := MoEMatVecNative(bs.Hidden, gl.FFNGateExps.Buf, bs.MoEXE, bs.MoEW,
		expDim, dim, gl.FFNGateExps.Type, blocksPerExpert, 0, 0, 1); err != nil {
		t.Fatalf("moe matvec e0: %v", err)
	}
	Sync()
	DownloadF32(bs.Gate, gateP)
	DownloadF32(bs.Up, gateQ)
	DownloadF32(bs.Hidden, gateR)
	t.Logf("expert 0: plain vs offset0 = %s", diffStat(gateP, gateQ))
	t.Logf("expert 0: plain vs moe-kernel = %s", diffStat(gateP, gateR))

	// ---- 3. gather / scatter round trip ----
	idx := []int32{2, 0, 3, 1}
	w := []float32{0.5, 0.25, 0.125, 0.0625}
	BeginBatch()
	if err := UploadI32(bs.MoEIdx, idx); err != nil {
		t.Fatalf("upload idx: %v", err)
	}
	if err := UploadF32(bs.MoEW, w); err != nil {
		t.Fatalf("upload w: %v", err)
	}
	if err := GatherRows(bs.MoEXE, bs.FFNNorm, bs.MoEIdx, len(idx), dim); err != nil {
		t.Fatalf("gather: %v", err)
	}
	Barrier()
	if err := Scale(bs.MoEOut, 0, 4*dim); err != nil {
		t.Fatalf("zero: %v", err)
	}
	Barrier()
	if err := ScatterRows(bs.MoEOut, bs.MoEXE, bs.MoEIdx, bs.MoEW, len(idx), dim, true); err != nil {
		t.Fatalf("scatter: %v", err)
	}
	Sync()
	gathered := make([]float32, len(idx)*dim)
	if err := DownloadF32(bs.MoEXE, gathered); err != nil {
		t.Fatalf("dl gathered: %v", err)
	}
	scattered := make([]float32, 4*dim)
	if err := DownloadF32(bs.MoEOut, scattered); err != nil {
		t.Fatalf("dl scattered: %v", err)
	}
	var gErr float64
	for i, src := range idx {
		for c := 0; c < dim; c++ {
			gErr = math.Max(gErr, math.Abs(float64(gathered[i*dim+c]-x[int(src)*dim+c])))
		}
	}
	var sErr float64
	for i, dst := range idx {
		for c := 0; c < dim; c++ {
			sErr = math.Max(sErr, math.Abs(float64(scattered[int(dst)*dim+c]-w[i]*x[int(dst)*dim+c])))
		}
	}
	t.Logf("gather maxErr=%.3e scatter maxErr=%.3e", gErr, sErr)
}

func diffStat(got, want []float32) string {
	var maxAbs, sumAbs, sumMag float64
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		maxAbs = math.Max(maxAbs, d)
		sumAbs += d
		sumMag += math.Abs(float64(want[i]))
	}
	return fmt.Sprintf("maxAbs=%.3e meanRel=%.3e", maxAbs, sumAbs/math.Max(sumMag, 1e-9))
}
