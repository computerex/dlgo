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
	path := `C:\Users\mohd\Downloads\gpt-oss-20b-MXFP4.gguf`
	fmt.Fprintf(os.Stderr, "Loading %s...\n", path)
	t0 := time.Now()
	pipe, err := llm.NewPipeline(path, 512)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Load error: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Loaded in %v\n", time.Since(t0))

	cfg0 := pipe.Model.Config
	fmt.Fprintf(os.Stderr, "Config: dim=%d headDim=%d numHeads=%d numKVHeads=%d\n",
		cfg0.EmbeddingDim, cfg0.HeadDim, cfg0.NumHeads, cfg0.NumKVHeads)
	fmt.Fprintf(os.Stderr, "  experts=%d used=%d expertFFNDim=%d layers=%d\n",
		cfg0.ExpertCount, cfg0.ExpertUsedCount, cfg0.ExpertFFNDim, cfg0.NumLayers)
	fmt.Fprintf(os.Stderr, "  GatedQ=%v AttnSinks=%v\n",
		pipe.Model.Layers[0].Spec.GatedQ, pipe.Model.Layers[0].AttnSinks != nil)

	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Fprintf(os.Stderr, "GPU pipeline error: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "GPU pipeline ready\n")

	prompt := "Explain the theory of relativity in simple terms."
	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = 64
	cfg.Seed = 42
	cfg.Sampler.Temperature = 0
	cfg.Stream = func(s string) { fmt.Print(s) }

	// Per-token profiling: generate 10 tokens and measure per-token wall time
	fmt.Fprintf(os.Stderr, "\nProfiling per-token time...\n")
	for i := 0; i < 3; i++ {
		result2, _ := gpuPipe.GenerateDetailed(prompt, cfg)
		ms := result2.GenerateTimeMs
		ntok := float64(result2.TotalTokens - result2.PromptTokens)
		if ntok < 1 { ntok = 1 }
		fmt.Fprintf(os.Stderr, "  Run %d: %.1f tok/s (%.2fms/tok, gen=%.1fms, %d toks)\n",
			i, result2.TokensPerSec, ms/ntok, ms, int(ntok))
	}

	fmt.Fprintf(os.Stderr, "\nFinal run...\n")
	result, err := gpuPipe.GenerateDetailed(prompt, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Generate error: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "\n\nResult: %d tokens, %.1f tok/s, prefill=%.1fms, gen=%.1fms\n",
		result.TotalTokens, result.TokensPerSec, result.PrefillTimeMs, result.GenerateTimeMs)
}
