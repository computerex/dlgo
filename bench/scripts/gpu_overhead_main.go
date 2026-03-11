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
	gl := &gpuModel.Layers[0]

	N := 200

	// Test 1: N MatVecs WITHOUT barriers in one batch
	start := time.Now()
	gpu.BeginBatch()
	for i := 0; i < N; i++ {
		gpu.MatVec(gpuRS.Gate, gl.FFNGate.Buf, gpuRS.XNorm, gl.FFNGate.Rows, gl.FFNGate.Cols, gl.FFNGate.Type)
	}
	gpu.EndBatch()
	noBarrier := time.Since(start)
	fmt.Printf("Test 1: %d MatVecs NO barriers:   %v (%.3fms each)\n", N, noBarrier, float64(noBarrier.Microseconds())/float64(N)/1000.0)

	// Test 2: N MatVecs WITH barriers in one batch
	start = time.Now()
	gpu.BeginBatch()
	for i := 0; i < N; i++ {
		gpu.Barrier()
		gpu.MatVec(gpuRS.Gate, gl.FFNGate.Buf, gpuRS.XNorm, gl.FFNGate.Rows, gl.FFNGate.Cols, gl.FFNGate.Type)
	}
	gpu.EndBatch()
	withBarrier := time.Since(start)
	fmt.Printf("Test 2: %d MatVecs WITH barriers: %v (%.3fms each)\n", N, withBarrier, float64(withBarrier.Microseconds())/float64(N)/1000.0)

	barrierCost := float64(withBarrier.Microseconds()-noBarrier.Microseconds()) / float64(N)
	fmt.Printf("Barrier cost per dispatch: %.1f µs\n\n", barrierCost)

	// Test 3: Measure a full forward pass breakdown
	gpuKV := gpu.NewGpuKVCache(cfg.NumLayers, 512, kvDim)
	logitsBuf := make([]float32, cfg.VocabSize)
	gpu.GpuForward(pipe.Model, gpuModel, 2, 0, gpuKV, gpuRS, logitsBuf) // warmup

	nTokens := 20
	start = time.Now()
	for i := 0; i < nTokens; i++ {
		gpu.GpuForward(pipe.Model, gpuModel, 2, i, gpuKV, gpuRS, logitsBuf)
	}
	elapsed := time.Since(start)
	fmt.Printf("Full forward: %d tokens in %v (%.2fms/tok, %.1f tok/s)\n",
		nTokens, elapsed, float64(elapsed.Microseconds())/float64(nTokens)/1000.0,
		float64(nTokens)/elapsed.Seconds())

	// Test 4: Small MatVec (attention projection, smaller matrix)
	start = time.Now()
	gpu.BeginBatch()
	for i := 0; i < N; i++ {
		gpu.MatVec(gpuRS.Q, gl.Wq.Buf, gpuRS.XNorm, gl.Wq.Rows, gl.Wq.Cols, gl.Wq.Type)
	}
	gpu.EndBatch()
	fmt.Printf("Wq MatVec %dx%d no barriers (%dx): %v (%.3fms each)\n",
		gl.Wq.Rows, gl.Wq.Cols, N, time.Since(start),
		float64(time.Since(start).Microseconds())/float64(N)/1000.0)

	// Test 5: RMSNorm throughput
	start = time.Now()
	gpu.BeginBatch()
	for i := 0; i < N; i++ {
		gpu.Barrier()
		gpu.RMSNorm(gpuRS.XNorm, gpuRS.X, gl.AttnNorm, dim, cfg.RMSNormEps)
	}
	gpu.EndBatch()
	fmt.Printf("RMSNorm %d with barriers (%dx): %v (%.3fms each)\n",
		dim, N, time.Since(start),
		float64(time.Since(start).Microseconds())/float64(N)/1000.0)

	// Test 6: Measure per-layer cost
	fmt.Printf("\nModel: TinyLlama 1.1B Q4_0 (%d layers, dim=%d)\n", cfg.NumLayers, dim)
	fmt.Printf("FFN gate: %dx%d, Wq: %dx%d\n", gl.FFNGate.Rows, gl.FFNGate.Cols, gl.Wq.Rows, gl.Wq.Cols)
	perTokenMs := float64(elapsed.Microseconds()) / float64(nTokens) / 1000.0
	perLayerMs := perTokenMs / float64(cfg.NumLayers)
	fmt.Printf("Per-token: %.2fms, per-layer: %.3fms\n", perTokenMs, perLayerMs)
}
