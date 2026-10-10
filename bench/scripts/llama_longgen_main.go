//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := `C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf`
	maxTok := 600

	if err := gpu.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "gpu init: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	pipe, err := llm.NewPipeline(modelPath, 4096)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gpu pipeline: %v\n", err)
		os.Exit(1)
	}
	defer gpuPipe.FreeAll()

	cfg := pipe.Model.Config
	prompt := llm.FormatChat(cfg, "You are a helpful assistant.",
		"Explain the concept of recursion in programming. Give multiple examples in Python.")

	genCfg := llm.DefaultGenerateConfig()
	genCfg.MaxTokens = maxTok
	genCfg.Seed = 42
	genCfg.Sampler.Temperature = 0

	genCfg.Stream = func(tok string) {
		fmt.Print(tok)
	}

	result, err := gpuPipe.GenerateDetailed(prompt, genCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\ngen error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n\n--- STATS ---\n")
	fmt.Printf("Tokens: %d\n", result.TotalTokens)
	fmt.Printf("Speed: %.1f tok/s\n", result.TokensPerSec)
	fmt.Printf("Prefill: %.1f ms\n", result.PrefillTimeMs)
	fmt.Printf("Generate: %.1f ms\n", result.GenerateTimeMs)
}
