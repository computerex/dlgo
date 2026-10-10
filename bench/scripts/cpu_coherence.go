//go:build ignore

package main

import (
	"fmt"
	"os"
	"time"

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

	rawPrompt := "Explain how Python decorators work. Give a simple example."
	messages := []llm.Message{
		{Role: "user", Content: rawPrompt},
	}
	prompt := llm.FormatMessages(pipe.Model.Config, messages)

	fmt.Fprintf(os.Stderr, "PROMPT: %q\n", prompt)
	fmt.Fprintf(os.Stderr, "AddBOS=%v BOS=%d EOS=%d\n", pipe.Model.Config.AddBOS, pipe.Model.Config.BOS, pipe.Model.Config.EOS)
	promptTokens := pipe.Tokenizer.Encode(prompt)
	fmt.Fprintf(os.Stderr, "PROMPT_TOKENS=%d\n", len(promptTokens))
	fmt.Fprintf(os.Stderr, "ALL_TOKENS: ")
	for i, tid := range promptTokens {
		fmt.Fprintf(os.Stderr, "[%d]%d=%q ", i, tid, pipe.Tokenizer.DecodeToken(tid))
	}
	fmt.Fprintf(os.Stderr, "\n")

	sampler := ops.DefaultSamplerConfig()
	sampler.Temperature = 0.7

	tokenCount := 0
	var allText string
	genCfg := llm.GenerateConfig{
		MaxTokens: maxTokens,
		Sampler:   sampler,
		Seed:      42,
		Stream: func(token string) {
			tokenCount++
			allText += token
			if tokenCount%500 == 0 {
				tail := allText
				if len(tail) > 200 {
					tail = tail[len(tail)-200:]
				}
				fmt.Fprintf(os.Stderr, "  [%d tokens] ...%s\n", tokenCount, tail)
			}
		},
	}

	start := time.Now()
	result, err := pipe.GenerateDetailed(prompt, genCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	elapsed := time.Since(start)
	fmt.Fprintf(os.Stderr, "GENERATED_TOKENS: ")
	for i, tid := range result.Tokens {
		if i >= 20 { break }
		fmt.Fprintf(os.Stderr, "%d=%q ", tid, pipe.Tokenizer.DecodeToken(tid))
	}
	fmt.Fprintf(os.Stderr, "\n")
	fmt.Fprintf(os.Stderr, "RESULT gen_tokens=%d chars=%d tok/s=%.1f elapsed=%.1fs\n",
		result.TotalTokens, len(result.Text), result.TokensPerSec, elapsed.Seconds())

	tail := result.Text
	if len(tail) > 500 {
		tail = tail[len(tail)-500:]
	}
	fmt.Fprintf(os.Stderr, "TAIL=%s\n", tail)
}
