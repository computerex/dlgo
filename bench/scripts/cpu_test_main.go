//go:build ignore

package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/computerex/dlgo/models/llm"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: cpu_test <gguf_path> <max_tokens>")
		os.Exit(1)
	}
	ggufPath := os.Args[1]
	maxTokens := 20
	if v, err := strconv.Atoi(os.Args[2]); err == nil {
		maxTokens = v
	}

	pipe, err := llm.NewPipeline(ggufPath, 512)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	prompt := "The history of artificial intelligence begins"
	fmt.Fprintf(os.Stderr, "Prompt: %q\n", prompt)

	genCfg := llm.DefaultGenerateConfig()
	genCfg.MaxTokens = maxTokens
	genCfg.Seed = 42
	genCfg.Sampler.Temperature = 0

	result, err := pipe.GenerateDetailed(prompt, genCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gen: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "%.1f tok/s, %d tokens\n", result.TokensPerSec, result.TotalTokens)
	fmt.Println(result.Text)
}
