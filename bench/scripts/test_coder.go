//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	path := `C:\models\Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf`
	ctxSize := 8192
	nTokens := 8000
	prompt := "Write a comprehensive tutorial on building a REST API with Python Flask. Cover project setup, routing, request handling, database integration with SQLAlchemy, authentication with JWT tokens, error handling, testing, deployment, and best practices. Include complete code examples for each section."
	useGPU := os.Getenv("USE_GPU") == "1"

	pipe, err := llm.NewPipeline(path, ctxSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load error: %v\n", err)
		os.Exit(1)
	}

	tokenCount := 0
	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = nTokens
	cfg.Sampler.Temperature = 0.0
	cfg.Stream = func(token string) {
		tokenCount++
		if tokenCount%500 == 0 {
			fmt.Fprintf(os.Stderr, "  progress: %d tokens\n", tokenCount)
		}
	}

	if useGPU {
		gpuPipe, err := gpu.NewGpuPipeline(pipe)
		if err != nil {
			fmt.Fprintf(os.Stderr, "gpu init error: %v\n", err)
			os.Exit(1)
		}
		defer gpuPipe.FreeAll()
		fmt.Fprintf(os.Stderr, "GPU mode, generating up to %d tokens...\n", nTokens)
		result, err := gpuPipe.GenerateDetailed(prompt, cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "generate error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("=== OUTPUT (%d tokens, %.1f tok/s) ===\n", result.TotalTokens, result.TokensPerSec)
		fmt.Println(result.Text)
		fmt.Println("=== END ===")
		return
	}

	cfg.MaxTokens = 200
	fmt.Fprintf(os.Stderr, "CPU mode, generating %d tokens (temp=0)...\n", cfg.MaxTokens)
	result, err := pipe.GenerateDetailed(prompt, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("=== OUTPUT (%d tokens, %.1f tok/s) ===\n", result.TotalTokens, result.TokensPerSec)
	fmt.Println(result.Text)
	fmt.Println("=== END ===")
}
