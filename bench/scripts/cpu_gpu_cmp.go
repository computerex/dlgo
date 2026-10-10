//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	modelPath := os.Args[1]
	maxTokens := 2000

	pipe, err := llm.NewPipeline(modelPath, 8192)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	prompt := "Write a comprehensive tutorial on building a REST API with Python Flask."
	promptTokens := pipe.Tokenizer.Encode(prompt)
	fmt.Fprintf(os.Stderr, "prompt_tokens=%d\n", len(promptTokens))

	// GPU generation
	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR GPU: %v\n", err)
		os.Exit(1)
	}

	sampler := ops.DefaultSamplerConfig()
	sampler.Temperature = 0.7
	sampler.TopP = 0.9
	tokenCount := 0
	var allGpuText string
	genCfg := llm.GenerateConfig{
		MaxTokens: maxTokens,
		Sampler:   sampler,
		Seed:      42,
		Stream: func(token string) {
			tokenCount++
			allGpuText += token
			if tokenCount%200 == 0 {
				tail := allGpuText
				if len(tail) > 200 {
					tail = tail[len(tail)-200:]
				}
				fmt.Fprintf(os.Stderr, "  [%d tok] ...%s\n", tokenCount, tail)
			}
		},
	}

	t0 := time.Now()
	gpuResult, err := gpuPipe.GenerateDetailed(prompt, genCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR GPU gen: %v\n", err)
		os.Exit(1)
	}
	gpuElapsed := time.Since(t0)
	gpuPipe.FreeAll()

	fmt.Fprintf(os.Stderr, "GPU: %d tokens in %.1fs (%.1f tok/s)\n",
		gpuResult.TotalTokens, gpuElapsed.Seconds(), gpuResult.TokensPerSec)
	gpuText := gpuResult.Text
	if len(gpuText) > 600 {
		fmt.Fprintf(os.Stderr, "GPU_HEAD: %s\n", gpuText[:300])
		fmt.Fprintf(os.Stderr, "GPU_TAIL: %s\n", gpuText[len(gpuText)-300:])
	} else {
		fmt.Fprintf(os.Stderr, "GPU_FULL: %s\n", gpuText)
	}

	_ = math.Abs(0)
	_ = time.Now()
}
