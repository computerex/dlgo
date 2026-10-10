//go:build ignore

package main

import (
	"fmt"
	"os"
	"time"

	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := os.Args[1]

	pipe, err := llm.NewPipeline(modelPath, 2048)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = 200
	cfg.Sampler.Temperature = 0.7
	cfg.Sampler.TopK = 40
	cfg.Sampler.TopP = 0.9
	cfg.Seed = 42

	prompt := "Explain Python decorators in 3 sentences."

	// Test 1: thinking enabled (default for qwen3moe)
	fmt.Fprintf(os.Stderr, "=== TEST 1: Thinking ENABLED ===\n")
	start := time.Now()
	text1, tps1, err := pipe.Chat("", prompt, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Time: %.1fs, %.1f tok/s\n", time.Since(start).Seconds(), tps1)
	fmt.Fprintf(os.Stderr, "Output (%d chars):\n%s\n\n", len(text1), text1)

	// Test 2: thinking disabled
	fmt.Fprintf(os.Stderr, "=== TEST 2: Thinking DISABLED ===\n")
	disabled := false
	fmtOpts := llm.FormatOptions{EnableThinking: &disabled}
	formatted := llm.FormatMessages(pipe.Model.Config, []llm.Message{
		{Role: "user", Content: prompt},
	}, fmtOpts)
	start = time.Now()
	text2, tps2, err := pipe.GenerateText(formatted, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Time: %.1fs, %.1f tok/s\n", time.Since(start).Seconds(), tps2)
	fmt.Fprintf(os.Stderr, "Output (%d chars):\n%s\n\n", len(text2), text2)

	if len(text1) > 20 && len(text2) > 20 {
		fmt.Fprintf(os.Stderr, "PASS: Both modes generated coherent output\n")
	} else {
		fmt.Fprintf(os.Stderr, "FAIL: Output too short (think=%d, no-think=%d)\n", len(text1), len(text2))
		os.Exit(1)
	}
}
