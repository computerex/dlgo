//go:build cgo && vulkan

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// TestMoEGroupedVsPerExpert checks the grouped MoE (one GEMM dispatch covering
// every expert) against the per-expert loop on real LFM2.5-24B-A2B weights, for
// every MoE layer in the model.
//
// Feeding both implementations the same activations is what makes this a tight
// check: the router logits are computed by the same kernel from the same buffer,
// so the routing decision — the top-4 experts and their weights — is identical on
// both sides, and the only remaining difference is the expert arithmetic itself
// (a cooperative-matrix FP16 GEMM versus a batched FP32 matvec). Anything larger
// than ~1e-3 would mean the grouped path selected or read something different.
//
// Do not compare these two implementations end to end through the whole model and
// expect a tight bound. A 1e-4 per-layer difference is enough to flip a hair-thin
// top-4 tie a layer later, and one flipped expert moves the final logits by O(1)
// (measured: mean rel ~0.1 over 40 layers). The end-to-end check that *is* exact
// is chunk-size invariance, in TestShortConvBatchPrefillParity.
func TestMoEGroupedVsPerExpert(t *testing.T) {
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

	// Upload every layer: this walks all 38 MoE layers, not just the first.
	gm, err := UploadModel(m, cfg.NumLayers)
	if err != nil {
		t.Fatalf("upload model: %v", err)
	}
	t.Cleanup(gm.FreeAll)

	if !moeBatchSupported(&m.Layers[0], &gm.Layers[0], cfg) {
		// Layer 0 may be dense; the check below skips dense layers anyway.
		t.Logf("layer 0 is not a batched-MoE layer (expected on this arch)")
	}

	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim

	rs, err := NewGpuRunState(dim, qDim, kvDim, ffnDim, cfg.VocabSize)
	if err != nil {
		t.Fatalf("run state: %v", err)
	}
	t.Cleanup(rs.FreeAll)

	moeLayers := 0
	for _, npos := range []int{1, 4, 16} {
		bs, err := NewGpuBatchState(npos, dim, qDim, kvDim, ffnDim)
		if err != nil {
			t.Fatalf("npos=%d batch state: %v", npos, err)
		}
		if err := bs.AllocMoEBatch(npos, dim, cfg.ExpertCount); err != nil {
			t.Fatalf("npos=%d moe batch scratch: %v", npos, err)
		}
		if err := bs.AllocMoEGrouped(npos, dim, cfg.ExpertFFNDim,
			cfg.ExpertUsedCount, cfg.ExpertCount); err != nil {
			t.Fatalf("npos=%d moe grouped scratch: %v", npos, err)
		}

		// Deterministic pseudo-random input with a realistic magnitude, identical
		// for every layer.
		x := make([]float32, npos*dim)
		seed := uint32(12345)
		for i := range x {
			seed = seed*1664525 + 1013904223
			x[i] = (float32(seed>>16)/65536.0 - 0.5) * 4.0
		}

		for l := 0; l < cfg.NumLayers; l++ {
			if m.Layers[l].Spec.FFN != llm.FFNMoE {
				continue
			}
			gl := &gm.Layers[l]
			layer := &m.Layers[l]
			if !moeBatchSupported(layer, gl, cfg) {
				t.Fatalf("layer %d: batched MoE unsupported", l)
			}
			if !moeGroupedSupported(layer, gl, cfg, bs) {
				t.Skip("grouped MoE not supported for this model (weights are not Affine?)")
			}
			moeLayers++

			run := func(fn func(*llm.Layer, *GpuLayer, *GpuBatchState, llm.ModelConfig, int) error) []float32 {
				BeginBatch()
				if err := UploadF32(bs.FFNNorm, x); err != nil {
					t.Fatalf("npos=%d layer %d upload: %v", npos, l, err)
				}
				Barrier()
				if err := fn(layer, gl, bs, cfg, npos); err != nil {
					t.Fatalf("npos=%d layer %d forward: %v", npos, l, err)
				}
				Sync()
				out := make([]float32, npos*dim)
				if err := DownloadF32(bs.FFNOut, out); err != nil {
					t.Fatalf("npos=%d layer %d download: %v", npos, l, err)
				}
				return out
			}

			want := run(gpuForwardMoEBatchPerExpert)
			got := run(gpuForwardMoEGrouped)
			again := run(gpuForwardMoEGrouped)

			var maxAbs, maxRel, sumAbs, sumMag, repeatAbs float64
			for i := range got {
				d := math.Abs(float64(got[i] - want[i]))
				maxAbs = math.Max(maxAbs, d)
				mag := math.Abs(float64(want[i]))
				if mag > 1e-6 {
					maxRel = math.Max(maxRel, d/mag)
				}
				sumAbs += d
				sumMag += mag
				repeatAbs = math.Max(repeatAbs, math.Abs(float64(got[i]-again[i])))
			}
			meanRel := sumAbs / math.Max(sumMag, 1e-9)
			t.Logf("npos=%2d layer %2d: maxAbs=%.3e maxRel=%.3e meanRel=%.3e repeat=%.3e",
				npos, l, maxAbs, maxRel, meanRel, repeatAbs)

			// The grouped path must reproduce itself exactly: nothing in it depends
			// on scheduling order (that is why the top-k sum is a separate kernel).
			if repeatAbs != 0 {
				t.Errorf("npos=%d layer %d: grouped path is not reproducible: maxAbs %.3e",
					npos, l, repeatAbs)
			}
			if maxAbs > 1e-3 {
				t.Errorf("npos=%d layer %d: grouped vs per-expert maxAbs %.3e too large",
					npos, l, maxAbs)
			}
			if meanRel > 1e-3 {
				t.Errorf("npos=%d layer %d: grouped vs per-expert meanRel %.3e too large",
					npos, l, meanRel)
			}
		}
		bs.Free()
	}
	if moeLayers == 0 {
		t.Fatal("no MoE layer was exercised")
	}
	t.Logf("checked %d MoE layers", moeLayers)
}
