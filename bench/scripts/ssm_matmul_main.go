//go:build ignore

package main

import (
	"fmt"
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	path := `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`
	pipe, err := llm.NewPipeline(path, 512)
	if err != nil {
		fmt.Printf("Load fail: %v\n", err)
		return
	}

	cfg := pipe.Model.Config
	m := pipe.Model

	x := make([]float32, cfg.EmbeddingDim)
	_ = m.TokenEmbed.DequantizeRow(760, x)
	xnorm := make([]float32, cfg.EmbeddingDim)
	ops.RMSNorm(xnorm, x, m.Layers[0].AttnNorm, cfg.RMSNormEps)
	fmt.Printf("Input L2: %.4f\n", l2(xnorm))

	// Test SSMInProj (layer 0)
	testMatVec("SSMInProj", m.Layers[0].SSMInProj, xnorm)

	// Test Wo (layer 3 - attention layer)
	// Need to create proper input for Wo
	qDim := cfg.NumHeads * cfg.HeadDim
	woIn := make([]float32, qDim)
	for i := range woIn {
		woIn[i] = float32(i%7-3) * 0.01
	}
	testMatVec("Wo_layer3", m.Layers[3].Wo, woIn)

	// Also test Wq for layer 3
	testMatVec("Wq_layer3", m.Layers[3].Wq, xnorm)

	// Test FFNDown for layer 0
	hidden := cfg.FFNDim
	ffnIn := make([]float32, hidden)
	for i := range ffnIn {
		ffnIn[i] = float32(i%5-2) * 0.001
	}
	testMatVec("FFNDown_layer0", m.Layers[0].FFNDown, ffnIn)
}

func testMatVec(name string, qt *core.QuantizedTensor, input []float32) {
	fmt.Printf("\n=== %s: rows=%d cols=%d type=%d ===\n", name, qt.Rows, qt.Cols, qt.Type)

	out1 := make([]float32, qt.Rows)
	blas.QMatVecMulParallel(out1, qt, input, blas.DefaultPool())
	fmt.Printf("QMatVecMul L2: %.4f first5: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
		l2(out1), out1[0], out1[1], out1[2], out1[3], out1[4])

	rowBuf := make([]float32, qt.Cols)
	nCheck := 10
	if nCheck > qt.Rows {
		nCheck = qt.Rows
	}
	var maxDiff float64
	for r := 0; r < nCheck; r++ {
		qt.DequantizeRow(r, rowBuf)
		var dot float64
		for j := 0; j < qt.Cols; j++ {
			dot += float64(rowBuf[j]) * float64(input[j])
		}
		diff := math.Abs(dot - float64(out1[r]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if r < 5 {
			fmt.Printf("  row %d: manual=%.6f QMatVec=%.6f diff=%.8f\n", r, dot, float64(out1[r]), diff)
		}
	}
	fmt.Printf("Max diff over %d rows: %.8f\n", nCheck, maxDiff)
}

func l2(x []float32) float64 {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum)
}
