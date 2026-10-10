//go:build ignore

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	modelPath := `C:\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`

	// Run with dp4a disabled to isolate
	os.Setenv("DLGO_NO_DP4A", "1")

	if err := gpu.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "gpu init: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	pipe, err := llm.NewPipeline(modelPath, 1024)
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
	msgs := []llm.Message{
		{Role: "system", Content: "You are a knowledgeable physics professor."},
		{Role: "user", Content: "Write a comprehensive explanation of quantum mechanics. Cover the historical development, key principles (wave-particle duality, uncertainty principle, superposition, entanglement), the mathematical framework, major interpretations, and modern applications. Be thorough and detailed."},
	}
	disableThinking := false
	prompt := llm.FormatMessages(cfg, msgs, llm.FormatOptions{EnableThinking: &disableThinking})

	genCfg := llm.DefaultGenerateConfig()
	genCfg.MaxTokens = 500
	genCfg.Seed = 42
	genCfg.Sampler = ops.SamplerConfig{
		Temperature:       0,
		RepetitionPenalty: 1.1,
	}

	fmt.Fprintf(os.Stderr, "Generating 500 tokens (NO DP4A)...\n")
	result, err := gpuPipe.GenerateDetailed(prompt, genCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	text := result.Text
	fmt.Println(text)

	// Check for repetition
	lines := strings.Split(text, "\n")
	repeatCount := 0
	lastLine := ""
	for _, line := range lines {
		if line == lastLine && len(line) > 10 {
			repeatCount++
		}
		lastLine = line
	}
	genTokens := result.TotalTokens - result.PromptTokens
	fmt.Fprintf(os.Stderr, "\n--- Diagnostics ---\n")
	fmt.Fprintf(os.Stderr, "Generated: %d tokens\n", genTokens)
	fmt.Fprintf(os.Stderr, "Repeated lines: %d\n", repeatCount)
	if repeatCount > 3 {
		fmt.Fprintf(os.Stderr, "VERDICT: STILL REPETITIVE (issue is NOT dp4a-specific)\n")
	} else {
		fmt.Fprintf(os.Stderr, "VERDICT: No repetition (issue IS dp4a-specific)\n")
	}
}
