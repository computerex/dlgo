//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	modelPath := `C:\models\gemma-3-270m-it-Q8_0.gguf`
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	fmt.Printf("Loading model: %s\n", modelPath)

	pipe, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		fmt.Printf("Load error: %v\n", err)
		os.Exit(1)
	}
	cfg := pipe.Model.Config
	dim := cfg.EmbeddingDim
	fmt.Printf("dim=%d, eps=%e, EmbedScale=%.4f\n", dim, cfg.RMSNormEps, cfg.EmbedScale)

	// Get token embedding
	tokens := pipe.Tokenizer.Encode(llm.FormatChat(cfg, "You are a helpful assistant.", "Explain what a computer is in exactly two sentences."))
	lastTok := tokens[len(tokens)-1]
	fmt.Printf("Last prompt token: %d (%q)\n", lastTok, pipe.Tokenizer.DecodeToken(lastTok))

	xCPU := make([]float32, dim)
	_ = pipe.Model.TokenEmbed.DequantizeRow(int(lastTok), xCPU)
	if cfg.EmbedScale != 0 {
		for i := range xCPU {
			xCPU[i] *= cfg.EmbedScale
		}
	}

	// CPU RMSNorm
	cpuNorm := make([]float32, dim)
	copy(cpuNorm, xCPU)
	ops.RMSNorm(cpuNorm, xCPU, pipe.Model.Layers[0].AttnNorm, cfg.RMSNormEps)

	fmt.Printf("\nCPU embed[0:5]:    %v\n", xCPU[:5])
	fmt.Printf("CPU norm[0:5]:     %v\n", cpuNorm[:5])

	// GPU
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

	// Upload embedding
	gpu.BeginBatch()
	gpu.UploadF32(gpuP.RunState.X, xCPU)
	gpu.Barrier()

	// Download X back to verify upload
	gpuX := make([]float32, dim)
	gpu.DownloadF32(gpuP.RunState.X, gpuX)

	fmt.Printf("\nGPU embed[0:5]:    %v\n", gpuX[:5])
	embedMatch := true
	var embedMaxDiff float64
	for i := 0; i < dim; i++ {
		d := math.Abs(float64(xCPU[i] - gpuX[i]))
		if d > embedMaxDiff {
			embedMaxDiff = d
		}
		if d > 0.001 {
			embedMatch = false
		}
	}
	fmt.Printf("Embed upload match: %v (maxDiff=%.6f)\n", embedMatch, embedMaxDiff)

	// GPU RMSNorm
	gpu.BeginBatch()
	gpu.UploadF32(gpuP.RunState.X, xCPU)
	gpu.Barrier()
	gpu.RMSNorm(gpuP.RunState.XNorm, gpuP.RunState.X,
		gpuP.GpuModel.Layers[0].AttnNorm, dim, cfg.RMSNormEps)
	gpu.Barrier()

	gpuNorm := make([]float32, dim)
	gpu.DownloadF32(gpuP.RunState.XNorm, gpuNorm)
	fmt.Printf("\nGPU norm[0:5]:     %v\n", gpuNorm[:5])

	var normMaxDiff float64
	for i := 0; i < dim; i++ {
		d := math.Abs(float64(cpuNorm[i] - gpuNorm[i]))
		if d > normMaxDiff {
			normMaxDiff = d
		}
	}
	fmt.Printf("RMSNorm match: maxDiff=%.6f\n", normMaxDiff)

	// Skip individual Q matvec test - test full forward instead

	// Now test the FULL first layer: CPU forward vs GPU ForwardLayer
	fmt.Println("\n=== Full Layer 0 Test ===")

	// CPU: do full forward for just position 0 with the last prompt token
	llm.Forward(pipe.Model, lastTok, 0, pipe.KVCache, pipe.RunState)
	cpuXLayer0 := make([]float32, dim)
	copy(cpuXLayer0, pipe.RunState.X)
	fmt.Printf("CPU X after layer 0..17 [0:5]: %v\n", cpuXLayer0[:5])

	// GPU: do full forward for position 0 with the same token
	gpuP.ResetState()
	gpu.GpuForwardFusedSSM(gpuPipe.Model, gpuP.GpuModel, lastTok, 0,
		gpuP.KVCache, gpuP.RunState, gpuP.LogitsBuf,
		gpu.BuildLayerConfs(gpuPipe.Model, gpuP.GpuModel, gpuP, gpuP.RunState, gpuP.KVCache), gpuP)
	gpu.Sync()

	gpuXFull := make([]float32, dim)
	gpu.BeginBatch()
	gpu.DownloadF32(gpuP.RunState.X, gpuXFull)
	fmt.Printf("GPU X after full forward [0:5]: %v\n", gpuXFull[:5])

	var fullMaxDiff float64
	for i := 0; i < dim; i++ {
		d := math.Abs(float64(cpuXLayer0[i] - gpuXFull[i]))
		if d > fullMaxDiff {
			fullMaxDiff = d
		}
	}
	fmt.Printf("Full forward X match: maxDiff=%.6f\n", fullMaxDiff)

	fmt.Printf("\nCPU logits[0:5]: %v\n", pipe.RunState.Logits[:5])
	fmt.Printf("GPU logits[0:5]: %v\n", gpuP.LogitsBuf[:5])

	var logitMaxDiff float64
	for i := 0; i < cfg.VocabSize; i++ {
		d := math.Abs(float64(pipe.RunState.Logits[i] - gpuP.LogitsBuf[i]))
		if d > logitMaxDiff {
			logitMaxDiff = d
		}
	}
	fmt.Printf("Single-token logits match: maxDiff=%.6f\n", logitMaxDiff)
}
