//go:build ignore

package main

import (
	"fmt"
	"os"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	if err := gpu.Init(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	path := `C:\projects\evoke\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`
	pipe, err := llm.NewPipeline(path, 512)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	cfg := pipe.Model.Config
	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim

	gpuModel, _ := gpu.UploadModel(pipe.Model)
	gpuRS := gpu.NewGpuRunState(dim, qDim, kvDim, cfg.FFNDim, cfg.VocabSize)
	gpuKV := gpu.NewGpuKVCache(cfg.NumLayers, 512, kvDim)
	logitsBuf := make([]float32, cfg.VocabSize)

	// Warm up
	gpu.GpuForward(pipe.Model, gpuModel, 2, 0, gpuKV, gpuRS, logitsBuf)

	// Benchmark 10 tokens
	nTokens := 10
	start := time.Now()
	for i := 0; i < nTokens; i++ {
		gpu.GpuForward(pipe.Model, gpuModel, 2, i, gpuKV, gpuRS, logitsBuf)
	}
	elapsed := time.Since(start)
	perToken := elapsed / time.Duration(nTokens)
	tps := float64(nTokens) / elapsed.Seconds()
	fmt.Printf("TinyLlama: %d tokens in %v (%.2fms/tok, %.1f tok/s)\n",
		nTokens, elapsed, float64(perToken.Microseconds())/1000.0, tps)

	// Time individual components
	xCPU := make([]float32, dim)
	_ = pipe.Model.TokenEmbed.DequantizeRow(2, xCPU)

	// Time just the upload
	start = time.Now()
	for i := 0; i < 100; i++ {
		gpu.UploadF32(gpuRS.X, xCPU)
	}
	fmt.Printf("Upload (100x): %v (%.3fms each)\n", time.Since(start), float64(time.Since(start).Microseconds())/100000.0)

	// Time just the download
	start = time.Now()
	for i := 0; i < 100; i++ {
		gpu.DownloadF32(gpuRS.Logits, logitsBuf)
	}
	fmt.Printf("Download (100x): %v (%.3fms each)\n", time.Since(start), float64(time.Since(start).Microseconds())/100000.0)

	// Time a single matmec (large - FFN gate: 5632 x 2048)
	gl := &gpuModel.Layers[0]
	start = time.Now()
	for i := 0; i < 100; i++ {
		gpu.BeginBatch()
		gpu.MatVec(gpuRS.Gate, gl.FFNGate.Buf, gpuRS.XNorm, gl.FFNGate.Rows, gl.FFNGate.Cols, gl.FFNGate.Type)
		gpu.EndBatch()
	}
	fmt.Printf("MatVec FFN gate 5632x2048 (100x): %v (%.3fms each)\n", time.Since(start), float64(time.Since(start).Microseconds())/100000.0)

	// Time a batch of 3 independent matmuls (Q/K/V)
	start = time.Now()
	for i := 0; i < 100; i++ {
		gpu.BeginBatch()
		gpu.MatVec(gpuRS.Q, gl.Wq.Buf, gpuRS.XNorm, gl.Wq.Rows, gl.Wq.Cols, gl.Wq.Type)
		gpu.MatVec(gpuRS.K, gl.Wk.Buf, gpuRS.XNorm, gl.Wk.Rows, gl.Wk.Cols, gl.Wk.Type)
		gpu.MatVec(gpuRS.V, gl.Wv.Buf, gpuRS.XNorm, gl.Wv.Rows, gl.Wv.Cols, gl.Wv.Type)
		gpu.EndBatch()
	}
	fmt.Printf("MatVec Q+K+V batch (100x): %v (%.3fms each)\n", time.Since(start), float64(time.Since(start).Microseconds())/100000.0)

	// Time empty batch (overhead only)
	start = time.Now()
	for i := 0; i < 1000; i++ {
		gpu.BeginBatch()
		gpu.Barrier()
		gpu.RMSNorm(gpuRS.XNorm, gpuRS.X, gl.AttnNorm, dim, cfg.RMSNormEps)
		gpu.EndBatch()
	}
	fmt.Printf("RMSNorm batch overhead (1000x): %v (%.4fms each)\n", time.Since(start), float64(time.Since(start).Microseconds())/1000000.0)

	// Time many matmuls in a single batch to measure GPU compute vs overhead
	nOps := 100
	start = time.Now()
	gpu.BeginBatch()
	for i := 0; i < nOps; i++ {
		gpu.Barrier()
		gpu.MatVec(gpuRS.Gate, gl.FFNGate.Buf, gpuRS.XNorm, gl.FFNGate.Rows, gl.FFNGate.Cols, gl.FFNGate.Type)
	}
	gpu.EndBatch()
	fmt.Printf("100x MatVec FFN gate in SINGLE batch: %v (%.3fms each)\n", time.Since(start), float64(time.Since(start).Microseconds())/float64(nOps)/1000.0)

	// Time submit_and_wait overhead with minimal work
	start = time.Now()
	for i := 0; i < 1000; i++ {
		gpu.BeginBatch()
		gpu.EndBatch()
	}
	fmt.Printf("Empty batch submit/wait (1000x): %v (%.4fms each)\n", time.Since(start), float64(time.Since(start).Microseconds())/1000000.0)
}
