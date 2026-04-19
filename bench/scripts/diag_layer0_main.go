//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
	"github.com/computerex/dlgo/quant"
)

func fp16ToF32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// denormal
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	} else if exp == 0x1F {
		return math.Float32frombits((sign << 31) | 0x7F800000 | (mant << 13))
	}
	exp = exp + 127 - 15
	return math.Float32frombits((sign << 31) | (exp << 23) | (mant << 13))
}

func maxAbsDiff(a, b []float32, n int) (float64, int) {
	var maxD float64
	maxI := 0
	for i := 0; i < n; i++ {
		d := math.Abs(float64(a[i] - b[i]))
		if d > maxD {
			maxD = d
			maxI = i
		}
	}
	return maxD, maxI
}

func printPreview(label string, v []float32, n int) {
	if n > len(v) {
		n = len(v)
	}
	fmt.Printf("  %s: %v\n", label, v[:n])
}

func gpuDownload(buf gpu.Buf, n int) []float32 {
	gpu.EndBatch()
	gpu.Sync()
	data := make([]float32, n)
	gpu.DownloadF32(buf, data)
	gpu.BeginBatch()
	return data
}

func report(name string, cpuData, gpuData []float32, n int) {
	md, mi := maxAbsDiff(cpuData, gpuData, n)
	status := "OK"
	if md > 0.01 {
		status = "WARN"
	}
	if md > 0.1 {
		status = "ERROR"
	}
	fmt.Printf("[%s] %-25s maxDiff=%.6f at [%d] (cpu=%.6f gpu=%.6f)\n",
		status, name, md, mi, cpuData[mi], gpuData[mi])
}

func main() {
	modelPath := `C:\models\gemma-3-270m-it-Q8_0.gguf`
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	fmt.Printf("=== Per-Operation Layer 0 Diagnostic ===\n")
	fmt.Printf("Model: %s\n\n", modelPath)

	// Load model for CPU
	cpuPipe, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		fmt.Printf("CPU pipeline error: %v\n", err)
		os.Exit(1)
	}
	cfg := cpuPipe.Model.Config
	m := cpuPipe.Model
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim
	kvMul := numHeads / numKVHeads
	pool := blas.DefaultPool()

	fmt.Printf("dim=%d headDim=%d numHeads=%d numKVHeads=%d kvDim=%d\n",
		dim, headDim, numHeads, numKVHeads, kvDim)
	fmt.Printf("FFN type: %d  Residual type: %d  QKNorm: %v\n",
		m.Layers[0].Spec.FFN, m.Layers[0].Spec.Residual, m.Layers[0].Spec.QKNorm)
	fmt.Printf("Wq type=%d  Wk type=%d  Wv type=%d  Wo type=%d\n",
		m.Layers[0].Wq.Type, m.Layers[0].Wk.Type, m.Layers[0].Wv.Type, m.Layers[0].Wo.Type)

	// Get token
	tokens := cpuPipe.Tokenizer.Encode(llm.FormatChat(cfg, "You are a helpful assistant.", "Explain what a computer is in exactly two sentences."))
	tok := tokens[len(tokens)-1]
	fmt.Printf("Token: %d (%q)\n\n", tok, cpuPipe.Tokenizer.DecodeToken(tok))

	// --- CPU Layer 0 step by step ---
	rs := cpuPipe.RunState
	layer := &m.Layers[0]

	// Embedding
	_ = m.TokenEmbed.DequantizeRow(int(tok), rs.X)
	if cfg.EmbedScale != 0 {
		ops.Scale(rs.X, cfg.EmbedScale)
	}
	cpuEmbed := make([]float32, dim)
	copy(cpuEmbed, rs.X)

	// RMSNorm
	ops.RMSNorm(rs.XNorm, rs.X, layer.AttnNorm, cfg.RMSNormEps)
	cpuXNorm := make([]float32, dim)
	copy(cpuXNorm, rs.XNorm)

	// Q/K/V projections
	blas.QTripleMatVecMulParallel(rs.Q, layer.Wq, rs.K, layer.Wk, rs.V, layer.Wv, rs.XNorm, pool)
	if layer.Bq != nil {
		ops.AddBias(rs.Q, layer.Bq)
	}
	if layer.Bk != nil {
		ops.AddBias(rs.K, layer.Bk)
	}
	if layer.Bv != nil {
		ops.AddBias(rs.V, layer.Bv)
	}
	cpuQ := make([]float32, numHeads*headDim)
	cpuK := make([]float32, kvDim)
	cpuV := make([]float32, kvDim)
	copy(cpuQ, rs.Q[:numHeads*headDim])
	copy(cpuK, rs.K[:kvDim])
	copy(cpuV, rs.V[:kvDim])

	// QKNorm
	if layer.Spec.QKNorm {
		for h := 0; h < numHeads; h++ {
			ops.RMSNormInPlace(rs.Q[h*headDim:(h+1)*headDim], layer.AttnQNorm, cfg.RMSNormEps)
		}
		for h := 0; h < numKVHeads; h++ {
			ops.RMSNormInPlace(rs.K[h*headDim:(h+1)*headDim], layer.AttnKNorm, cfg.RMSNormEps)
		}
	}
	cpuQNormed := make([]float32, numHeads*headDim)
	cpuKNormed := make([]float32, kvDim)
	copy(cpuQNormed, rs.Q[:numHeads*headDim])
	copy(cpuKNormed, rs.K[:kvDim])

	// RoPE
	cosT, _ := rs.RoPETables()
	if cosT != nil {
		for h := 0; h < numHeads; h++ {
			rs.ApplyRoPEFast(rs.Q[h*headDim:(h+1)*headDim], 0)
		}
		for h := 0; h < numKVHeads; h++ {
			rs.ApplyRoPEFast(rs.K[h*headDim:(h+1)*headDim], 0)
		}
	} else {
		ops.ApplyRoPEBatch(rs.Q, numHeads, rs.K, numKVHeads, 0, headDim, cfg.RopeFreqBase, cfg.RopeNeox)
	}
	cpuQRoPE := make([]float32, numHeads*headDim)
	cpuKRoPE := make([]float32, kvDim)
	copy(cpuQRoPE, rs.Q[:numHeads*headDim])
	copy(cpuKRoPE, rs.K[:kvDim])

	// KV Store + Attention (pos=0, seq_len=1)
	cpuPipe.KVCache.Layers[0].Store(0, rs.K, rs.V)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	ops.Clear(rs.AttnOut)
	for h := 0; h < numHeads; h++ {
		kvH := h / kvMul
		qHead := rs.Q[h*headDim : (h+1)*headDim]
		kHead := cpuPipe.KVCache.Layers[0].Keys[0][kvH*headDim : (kvH+1)*headDim]
		scoreVal := ops.DotProduct(qHead, kHead, headDim) * scale
		if cfg.AttnLogitSoftcap > 0 {
			cap := cfg.AttnLogitSoftcap
			scoreVal = cap * float32(math.Tanh(float64(scoreVal/cap)))
		}
		scores := []float32{scoreVal}
		quant.SIMDSoftmax(scores)
		vHead := cpuPipe.KVCache.Layers[0].Vals[0][kvH*headDim : (kvH+1)*headDim]
		headOut := rs.AttnOut[h*headDim : (h+1)*headDim]
		for d := 0; d < headDim; d++ {
			headOut[d] = scores[0] * vHead[d]
		}
	}
	cpuAttnOut := make([]float32, numHeads*headDim)
	copy(cpuAttnOut, rs.AttnOut[:numHeads*headDim])

	// Wo MatVec
	blas.QMatVecMulParallel(rs.AttnProj, layer.Wo, rs.AttnOut, pool)
	if layer.Bo != nil {
		ops.AddBias(rs.AttnProj, layer.Bo)
	}
	cpuAttnProj := make([]float32, dim)
	copy(cpuAttnProj, rs.AttnProj)

	// Residual + FFN Norm
	ops.Add(rs.FFNIn, rs.X, rs.AttnProj)
	ops.RMSNorm(rs.FFNNorm, rs.FFNIn, layer.FFNNorm, cfg.RMSNormEps)
	cpuFFNIn := make([]float32, dim)
	cpuFFNNorm := make([]float32, dim)
	copy(cpuFFNIn, rs.FFNIn)
	copy(cpuFFNNorm, rs.FFNNorm)

	// FFN Gate+Up
	blas.QDualMatVecMulParallel(rs.Gate, layer.FFNGate, rs.Up, layer.FFNUp, rs.FFNNorm, pool)
	cpuGate := make([]float32, len(rs.Gate))
	cpuUp := make([]float32, len(rs.Up))
	copy(cpuGate, rs.Gate)
	copy(cpuUp, rs.Up)

	// GeGLU activation
	ffnDim := len(rs.Gate)
	if layer.Spec.FFN == llm.FFNGeGLU {
		ops.GeGLU(rs.Hidden, rs.Gate, rs.Up, ffnDim)
	} else {
		quant.SIMDSwiGLU(rs.Hidden, rs.Gate, rs.Up, ffnDim)
	}
	cpuHidden := make([]float32, ffnDim)
	copy(cpuHidden, rs.Hidden)

	// FFN Down
	blas.QMatVecMulParallel(rs.FFNOut, layer.FFNDown, rs.Hidden, pool)
	cpuFFNOut := make([]float32, dim)
	copy(cpuFFNOut, rs.FFNOut)

	// Final residual
	ops.Add(rs.X, rs.FFNIn, rs.FFNOut)
	cpuXFinal := make([]float32, dim)
	copy(cpuXFinal, rs.X)

	fmt.Println("=== CPU Layer 0 complete ===\n")

	// --- GPU Layer 0 step by step ---
	if err := gpu.Init(); err != nil {
		fmt.Printf("GPU init error: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	gpuPipe, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		fmt.Printf("GPU pipe load error: %v\n", err)
		os.Exit(1)
	}
	gpuP, err := gpu.NewGpuPipeline(gpuPipe)
	if err != nil {
		fmt.Printf("GPU pipeline error: %v\n", err)
		os.Exit(1)
	}
	defer gpuP.FreeAll()

	grs := gpuP.RunState
	gm := gpuP.GpuModel
	gl := &gm.Layers[0]
	kv := gpuP.KVCache

	fmt.Printf("GPU Wq type=%d Wk type=%d Wv type=%d Wo type=%d\n", gl.Wq.Type, gl.Wk.Type, gl.Wv.Type, gl.Wo.Type)
	if gl.FFNGate != nil {
		fmt.Printf("GPU FFNGate type=%d FFNUp type=%d FFNDown type=%d\n", gl.FFNGate.Type, gl.FFNUp.Type, gl.FFNDown.Type)
	}
	fmt.Println()

	// 1. Upload embedding
	gpu.BeginBatch()
	gpu.UploadF32(grs.X, cpuEmbed)
	gpuEmbed := gpuDownload(grs.X, dim)
	report("1. Embed upload", cpuEmbed, gpuEmbed, dim)

	// 2. RMSNorm
	gpu.RMSNorm(grs.XNorm, grs.X, gl.AttnNorm, dim, cfg.RMSNormEps)
	gpuXNorm := gpuDownload(grs.XNorm, dim)
	report("2. RMSNorm", cpuXNorm, gpuXNorm, dim)

	// 3. Q MatVec
	gpu.MatVec(grs.Q, gl.Wq.Buf, grs.XNorm, gl.Wq.Rows, gl.Wq.Cols, gl.Wq.Type)
	gpuQRaw := gpuDownload(grs.Q, numHeads*headDim)
	if gl.Bq != 0 {
		gpu.Add(grs.Q, grs.Q, gl.Bq, numHeads*headDim)
		gpuQRaw = gpuDownload(grs.Q, numHeads*headDim)
	}
	report("3. Q (after bias)", cpuQ, gpuQRaw, numHeads*headDim)

	// 4. K MatVec
	gpu.MatVec(grs.K, gl.Wk.Buf, grs.XNorm, gl.Wk.Rows, gl.Wk.Cols, gl.Wk.Type)
	gpuKRaw := gpuDownload(grs.K, kvDim)
	if gl.Bk != 0 {
		gpu.Add(grs.K, grs.K, gl.Bk, kvDim)
		gpuKRaw = gpuDownload(grs.K, kvDim)
	}
	report("4. K (after bias)", cpuK, gpuKRaw, kvDim)

	// 5. V MatVec
	gpu.MatVec(grs.V, gl.Wv.Buf, grs.XNorm, gl.Wv.Rows, gl.Wv.Cols, gl.Wv.Type)
	gpuVRaw := gpuDownload(grs.V, kvDim)
	if gl.Bv != 0 {
		gpu.Add(grs.V, grs.V, gl.Bv, kvDim)
		gpuVRaw = gpuDownload(grs.V, kvDim)
	}
	report("5. V (after bias)", cpuV, gpuVRaw, kvDim)

	// 6. QKNorm
	if layer.Spec.QKNorm {
		gpu.Barrier()
		gpu.RMSNormHeads(grs.Q, gl.AttnQNorm, numHeads, headDim, cfg.RMSNormEps)
		gpu.RMSNormHeads(grs.K, gl.AttnKNorm, numKVHeads, headDim, cfg.RMSNormEps)
	}

	// 7. RoPE
	gpu.Barrier()
	gpu.RoPE(grs.Q, grs.K, gpuP.RoPECosTable, gpuP.RoPESinTable,
		numHeads, numKVHeads, headDim, cfg.RopeDim, 0, cfg.RopeNeox)
	gpuQNormedAndRoPE := gpuDownload(grs.Q, numHeads*headDim)
	gpuKNormedAndRoPE := gpuDownload(grs.K, kvDim)
	report("6-7. Q (QKNorm+RoPE)", cpuQRoPE, gpuQNormedAndRoPE, numHeads*headDim)
	report("6-7. K (QKNorm+RoPE)", cpuKRoPE, gpuKNormedAndRoPE, kvDim)

	// Verify V is still intact before KV store
	gpuVBefore := gpuDownload(grs.V, kvDim)
	fmt.Printf("  V[20] cpu=%.6f  gpu=%.6f (before KV store)\n", cpuV[20], gpuVBefore[20])
	fmt.Printf("  V[21] cpu=%.6f  gpu=%.6f\n", cpuV[21], gpuVBefore[21])

	// 8. KV Store
	gpu.KVStoreF16(kv.KeyBufs[0], kv.ValBufs[0], grs.K, grs.V, 0, kvDim)

	// Read back KV cache as raw bytes to verify FP16 storage
	gpu.EndBatch()
	gpu.Sync()
	kvRaw := make([]byte, kvDim*2) // kvDim/2 uint32s = kvDim*2 bytes
	gpu.Download(kv.ValBufs[0], kvRaw)
	// Unpack uint32 at index 10 (stores V[20] and V[21])
	u32 := uint32(kvRaw[40]) | uint32(kvRaw[41])<<8 | uint32(kvRaw[42])<<16 | uint32(kvRaw[43])<<24
	v20fp16 := math.Float32frombits(uint32(u32&0xFFFF) << 16) // crude: sign+exp only
	fmt.Printf("  KV cache raw uint32 at idx 10: 0x%08X\n", u32)
	// Use math.Float32frombits with proper FP16->FP32 conversion
	lo16 := uint16(u32 & 0xFFFF)
	hi16 := uint16((u32 >> 16) & 0xFFFF)
	fmt.Printf("  KV cache FP16 pair: lo=0x%04X hi=0x%04X\n", lo16, hi16)
	fmt.Printf("  Decoded: V[20]=%.6f  V[21]=%.6f\n", fp16ToF32(lo16), fp16ToF32(hi16))
	_ = v20fp16
	gpu.BeginBatch()

	// 9. Attention
	gpu.Barrier()
	gpu.AttentionF16(grs.AttnOut, grs.Q, kv.KeyBufs[0], kv.ValBufs[0],
		numHeads, numKVHeads, headDim, kvDim, 1, 0, scale, 0)
	gpuAttnOut := gpuDownload(grs.AttnOut, numHeads*headDim)
	report("9. AttnOut", cpuAttnOut, gpuAttnOut, numHeads*headDim)
	fmt.Printf("  AttnOut[276] cpu=%.6f  gpu=%.6f (should ≈ V[20] for head 1, dim 20)\n", cpuAttnOut[276], gpuAttnOut[276])

	// 10. Wo MatVec
	gpu.MatVec(grs.AttnProj, gl.Wo.Buf, grs.AttnOut, gl.Wo.Rows, gl.Wo.Cols, gl.Wo.Type)
	if gl.Bo != 0 {
		gpu.Barrier()
		gpu.Add(grs.AttnProj, grs.AttnProj, gl.Bo, dim)
	}
	gpuAttnProj := gpuDownload(grs.AttnProj, dim)
	report("10. AttnProj (after Wo)", cpuAttnProj, gpuAttnProj, dim)

	// 11. Residual + FFN Norm
	gpu.Add(grs.FFNIn, grs.X, grs.AttnProj, dim)
	gpu.Barrier()
	gpu.RMSNorm(grs.FFNNorm, grs.FFNIn, gl.FFNNorm, dim, cfg.RMSNormEps)
	gpuFFNIn := gpuDownload(grs.FFNIn, dim)
	gpuFFNNorm := gpuDownload(grs.FFNNorm, dim)
	report("11a. FFNIn (residual)", cpuFFNIn, gpuFFNIn, dim)
	report("11b. FFNNorm", cpuFFNNorm, gpuFFNNorm, dim)

	// 12. FFN Gate + Up MatVecs
	if gl.FFNGate != nil {
		gpu.MatVec(grs.Gate, gl.FFNGate.Buf, grs.FFNNorm, gl.FFNGate.Rows, gl.FFNGate.Cols, gl.FFNGate.Type)
		gpu.MatVec(grs.Up, gl.FFNUp.Buf, grs.FFNNorm, gl.FFNUp.Rows, gl.FFNUp.Cols, gl.FFNUp.Type)
		gpuGate := gpuDownload(grs.Gate, ffnDim)
		gpuUpV := gpuDownload(grs.Up, ffnDim)
		report("12a. FFN Gate", cpuGate, gpuGate, ffnDim)
		report("12b. FFN Up", cpuUp, gpuUpV, ffnDim)
	}

	// 13. Activation (GeGLU or SwiGLU)
	if layer.Spec.FFN == llm.FFNGeGLU {
		gpu.GeGLU(grs.Hidden, grs.Gate, grs.Up, ffnDim)
	} else {
		gpu.SwiGLU(grs.Hidden, grs.Gate, grs.Up, ffnDim)
	}
	gpuHidden := gpuDownload(grs.Hidden, ffnDim)
	report("13. Hidden (after activation)", cpuHidden, gpuHidden, ffnDim)

	// 14. FFN Down MatVec
	gpu.MatVec(grs.FFNOut, gl.FFNDown.Buf, grs.Hidden, gl.FFNDown.Rows, gl.FFNDown.Cols, gl.FFNDown.Type)
	gpuFFNOut := gpuDownload(grs.FFNOut, dim)
	report("14. FFNOut (after Down)", cpuFFNOut, gpuFFNOut, dim)

	// 15. Final residual
	gpu.Add(grs.X, grs.FFNIn, grs.FFNOut, dim)
	gpuXFinal := gpuDownload(grs.X, dim)
	report("15. X (end of layer 0)", cpuXFinal, gpuXFinal, dim)

	// Print summaries
	fmt.Println("\n=== Summary ===")
	printPreview("CPU X final", cpuXFinal, 5)
	printPreview("GPU X final", gpuXFinal, 5)
	d, _ := maxAbsDiff(cpuXFinal, gpuXFinal, dim)
	fmt.Printf("Layer 0 final X maxDiff: %.6f\n", d)
}

