//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := `C:\models\Qwen3.5-2B.Q4_K_M.gguf`
	maxTok := 600

	pipe, err := llm.NewPipeline(modelPath, 4096)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

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

	result, err := pipe.GenerateDetailed(prompt, genCfg)
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
