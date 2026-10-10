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
	modelPath := `C:\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`

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
	fmt.Fprintf(os.Stderr, "Architecture: %s, SSMTiledVOrder: %v, SSMTimeStepRank: %d, SSMGroupCount: %d\n",
		cfg.Architecture, cfg.SSMTiledVOrder, cfg.SSMTimeStepRank, cfg.SSMGroupCount)
	msgs := []llm.Message{
		{Role: "system", Content: "You are a knowledgeable physics professor."},
		{Role: "user", Content: "Write a comprehensive explanation of quantum mechanics. Cover the historical development, key principles (wave-particle duality, uncertainty principle, superposition, entanglement), the mathematical framework, major interpretations, and modern applications. Be thorough and detailed."},
	}
	disableThinking := false
	prompt := llm.FormatMessages(cfg, msgs, llm.FormatOptions{EnableThinking: &disableThinking})

	genCfg := llm.DefaultGenerateConfig()
	genCfg.MaxTokens = 200
	genCfg.Seed = 42
	genCfg.Sampler.Temperature = 0
	genCfg.Sampler.RepetitionPenalty = 1.0
	genCfg.Sampler.NoRepeatNgramSize = 0

	fmt.Fprintf(os.Stderr, "Generating 4000 tokens with Qwen3.6-35B-A3B IQ3_XXS...\n")
	start := time.Now()

	result, err := gpuPipe.GenerateDetailed(prompt, genCfg)
	elapsed := time.Since(start)

	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(result.Text)

	genTokens := result.TotalTokens - result.PromptTokens
	tokPerSec := float64(genTokens) / elapsed.Seconds()
	fmt.Fprintf(os.Stderr, "\n--- Stats ---\n")
	fmt.Fprintf(os.Stderr, "Prompt tokens: %d\n", result.PromptTokens)
	fmt.Fprintf(os.Stderr, "Generated tokens: %d\n", genTokens)
	fmt.Fprintf(os.Stderr, "Total time: %.1fs\n", elapsed.Seconds())
	fmt.Fprintf(os.Stderr, "Speed: %.1f tok/s\n", tokPerSec)
}
