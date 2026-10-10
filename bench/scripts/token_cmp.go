//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	modelPath := os.Args[1]

	pipe, err := llm.NewPipeline(modelPath, 8192)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	prompt := "Write an extremely detailed and comprehensive textbook chapter on building production-grade REST APIs with Python Flask."
	promptTokens := pipe.Tokenizer.Encode(prompt)
	fmt.Fprintf(os.Stderr, "prompt_tokens=%d\n", len(promptTokens))

	cfg := pipe.Model.Config
	vocabSize := cfg.VocabSize
	sampler := ops.DefaultSamplerConfig()
	sampler.Temperature = 0.0

	// === CPU forward ===
	cpuRS := llm.NewRunState(cfg, 8192)
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	cpuKV := memory.NewMultiLayerKVCache(cfg.NumLayers, 8192, kvDim)
	for i, tok := range promptTokens {
		llm.Forward(pipe.Model, int32(tok), i, cpuKV, cpuRS)
	}
	fmt.Fprintf(os.Stderr, "CPU prefill done\n")

	// === GPU forward ===
	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR GPU: %v\n", err)
		os.Exit(1)
	}
	defer gpuPipe.FreeAll()

	var gpuTokens []int32
	gpuGenCfg := llm.GenerateConfig{
		MaxTokens: 50,
		Sampler:   sampler,
		Seed:      42,
		Stream: func(token string) {
			// just count
		},
	}
	gpuResult, err := gpuPipe.GenerateDetailed(prompt, gpuGenCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR GPU gen: %v\n", err)
		os.Exit(1)
	}
	_ = gpuResult
	_ = gpuTokens

	// Generate 50 tokens from CPU
	cpuLogits := make([]float32, vocabSize)
	pos := len(promptTokens)
	fmt.Fprintf(os.Stderr, "\n%4s  %8s  %30s  %30s  %s\n", "TOK#", "MATCH", "CPU", "GPU", "")
	fmt.Fprintf(os.Stderr, "%s\n", "----  --------  ------------------------------  ------------------------------")

	// For CPU generation, we need to sample tokens one by one
	// Unfortunately we can't easily get the GPU's per-token logits
	// Let's just compare the sampled text
	copy(cpuLogits, cpuRS.Logits)
	for i := 0; i < 50; i++ {
		cpuTok := int32(ops.SampleToken(cpuLogits, sampler, nil, nil))
		cpuText := pipe.Tokenizer.DecodeToken(cpuTok)
		fmt.Fprintf(os.Stderr, "CPU[%2d]: id=%6d %q\n", i, cpuTok, cpuText)

		if cpuTok == cfg.EOS {
			break
		}
		llm.Forward(pipe.Model, cpuTok, pos, cpuKV, cpuRS)
		copy(cpuLogits, cpuRS.Logits)
		pos++
	}

	fmt.Fprintf(os.Stderr, "\nGPU output (%d tokens): %q\n", gpuResult.TotalTokens, gpuResult.Text)
}
