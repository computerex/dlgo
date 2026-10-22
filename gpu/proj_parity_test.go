//go:build cgo && vulkan

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/quant"
)

// TestShortConvProjectionParity measures the GPU-vs-CPU error of the two
// LFM2 short-convolution projections on the real weights. The conv kernel
// itself is verified by TestShortConvGPU; this isolates whether the
// projections feeding it are accurate enough, since the layer output is a
// product (b*x, then c*conv) and therefore amplifies their relative error.
func TestShortConvProjectionParity(t *testing.T) {
	const modelPath = `C:\Users\mohd\Downloads\LFM2.5-24B-A2B-QAD-Q4_0.gguf`
	p, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		t.Skipf("model unavailable: %v", err)
	}

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(Shutdown)

	m := p.Model
	dim := m.Config.EmbeddingDim

	var layer *llm.Layer
	for i := range m.Layers {
		if m.Layers[i].Spec.Core == llm.CoreShortConv {
			layer = &m.Layers[i]
			break
		}
	}
	if layer == nil {
		t.Fatal("no shortconv layer found")
	}

	rng := rand.New(rand.NewSource(99))
	x := make([]float32, dim)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 0.3
	}

	compare := func(name string, qt *core.QuantizedTensor, xIn []float32) {
		t.Helper()
		if qt == nil {
			t.Fatalf("%s: nil tensor", name)
		}
		cpuOut := make([]float32, qt.Rows)
		blas.QMatVecMulParallel(cpuOut, qt, xIn, blas.DefaultPool())

		// Reference: round-trip the same weights through the canonical Affine4
		// encoder and run the same CPU matvec on the reconstructed values. This
		// separates encoder loss from GPU-kernel error.
		var a4Out []float32
		if n := qt.Rows * qt.Cols; n%quant.Affine4GroupSize == 0 {
			if raw, err := quant.RepackAffine4(qt.Data, qt.Type, n); err == nil {
				w := quant.DequantizeAffine4(raw, n)
				a4t := &core.QuantizedTensor{Rows: qt.Rows, Cols: qt.Cols, FP32Data: w}
				a4Out = make([]float32, qt.Rows)
				blas.QMatVecMulParallel(a4Out, a4t, xIn, blas.DefaultPool())
			}
		}

		xb := Alloc(uint64(len(xIn) * 4))
		defer Free(xb)
		if err := UploadF32(xb, xIn); err != nil {
			t.Fatal(err)
		}
		gt, err := uploadProjWeight(qt)
		if err != nil || gt == nil {
			t.Fatalf("%s: upload failed: %v", name, err)
		}
		defer Free(gt.Buf)

		ob := Alloc(uint64(qt.Rows * 4))
		defer Free(ob)
		BeginBatch()
		if err := MatVec(ob, gt.Buf, xb, gt.Rows, gt.Cols, gt.Type); err != nil {
			t.Fatalf("%s: matvec (qtype %d): %v", name, gt.Type, err)
		}
		EndBatch()
		Sync()

		gpuOut := make([]float32, qt.Rows)
		if err := DownloadF32(ob, gpuOut); err != nil {
			t.Fatal(err)
		}

		stats := func(label string, got, want []float32) {
			var sumAbs, sumRef float64
			for i := range want {
				sumAbs += math.Abs(float64(got[i] - want[i]))
				sumRef += math.Abs(float64(want[i]))
			}
			t.Logf("%s [%s]: meanAbsErr=%.4g meanRelErr=%.3f%%", name, label,
				sumAbs/float64(len(want)), 100*sumAbs/sumRef)
		}
		t.Logf("%s: gpu qtype=%d %dx%d", name, gt.Type, gt.Rows, gt.Cols)
		stats("gpu vs cpu-native", gpuOut, cpuOut)
		if a4Out != nil {
			stats("cpu-affine4 vs cpu-native", a4Out, cpuOut)
			stats("gpu vs cpu-affine4", gpuOut, a4Out)
		} else {
			t.Logf("%s: affine4 reference skipped (n not a multiple of the group size)", name)
		}
	}

	compare("in_proj", layer.ShortConvInProj, x)
	compare("out_proj", layer.ShortConvOutProj, x)
}
