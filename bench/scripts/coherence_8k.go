//go:build ignore

package main

import (
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: coherence_8k <model.gguf> [max_tokens]\n")
		os.Exit(1)
	}
	modelPath := os.Args[1]
	maxTokens := 8192
	if len(os.Args) > 2 {
		fmt.Sscanf(os.Args[2], "%d", &maxTokens)
	}

	runtime.GC()

	pipe, err := llm.NewPipeline(modelPath, maxTokens)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: NewPipeline: %v\n", err)
		os.Exit(1)
	}
	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: NewGpuPipeline: %v\n", err)
		os.Exit(1)
	}
	defer gpuPipe.FreeAll()

	rawPrompt := "Explain how Python decorators work. Give a simple example."

	messages := []llm.Message{
		{Role: "user", Content: rawPrompt},
	}
	prompt := llm.FormatMessages(pipe.Model.Config, messages)

	promptTokens := pipe.Tokenizer.Encode(prompt)
	prefix := prompt
	if len(prefix) > 120 {
		prefix = prefix[:60] + " ... " + prompt[len(prompt)-60:]
	}
	fmt.Fprintf(os.Stderr, "PROMPT_TOKENS=%d PROMPT_CHARS=%d\n", len(promptTokens), len(prompt))
	fmt.Fprintf(os.Stderr, "PROMPT_PREVIEW: %q\n", prefix)
	fmt.Fprintf(os.Stderr, "LAST_10_TOKEN_IDS: %v\n", promptTokens[len(promptTokens)-min(10, len(promptTokens)):])
	fmt.Fprintf(os.Stderr, "LAST_10_TOKENS_DECODED: ")
	for _, tid := range promptTokens[len(promptTokens)-min(10, len(promptTokens)):] {
		fmt.Fprintf(os.Stderr, "%q ", pipe.Tokenizer.DecodeToken(tid))
	}
	fmt.Fprintf(os.Stderr, "\n")
	fmt.Fprintf(os.Stderr, "EOS=%d BOS=%d StopTokens=%v\n", pipe.Model.Config.EOS, pipe.Model.Config.BOS, pipe.Model.Config.StopTokens)

	tokenCount := 0
	var allText string
	sampler := ops.DefaultSamplerConfig()
	sampler.Temperature = 0.7
	genCfg := llm.GenerateConfig{
		MaxTokens: maxTokens,
		Sampler:   sampler,
		Seed:      42,
		Stream: func(token string) {
			tokenCount++
			allText += token
			if tokenCount <= 20 {
				fmt.Fprintf(os.Stderr, "  TOKEN[%d]: %q\n", tokenCount, token)
			}
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
	result, err := gpuPipe.GenerateDetailed(prompt, genCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: GenerateDetailed: %v\n", err)
		os.Exit(1)
	}

	elapsed := time.Since(start)
	text := result.Text
	genTokens := result.TotalTokens
	tokPerSec := result.TokensPerSec

	fmt.Fprintf(os.Stderr, "GENERATED_TOKENS: %v\n", result.Tokens)
	for i, tid := range result.Tokens {
		fmt.Fprintf(os.Stderr, "  GEN[%d] id=%d text=%q\n", i, tid, pipe.Tokenizer.DecodeToken(tid))
	}
	coherent := assessCoherence(text)

	fmt.Fprintf(os.Stderr, "RESULT gen_tokens=%d prompt_tokens=%d chars=%d coherent=%s tok/s=%.1f elapsed=%.1fs\n",
		genTokens, result.PromptTokens, len(text), coherent, tokPerSec, elapsed.Seconds())

	tail := text
	if len(tail) > 500 {
		tail = tail[len(tail)-500:]
	}
	fmt.Fprintf(os.Stderr, "TAIL=%s\n", tail)
}

func assessCoherence(text string) string {
	if len(text) < 100 {
		return "TOO_SHORT"
	}

	tail := text
	if len(tail) > 400 {
		tail = tail[len(tail)-400:]
	}

	nonAlpha := 0
	for _, c := range tail {
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
			c == ' ' || c == '\n' || c == '\t' || c == '.' || c == ',' ||
			c == ':' || c == ';' || c == '-' || c == '_' || c == '(' || c == ')' ||
			c == '"' || c == '\'' || c == '!' || c == '?' || c == '`' ||
			c == '#' || c == '*' || c == '/' || c == '{' || c == '}' ||
			c == '[' || c == ']' || c == '=' || c == '+' || c == '<' || c == '>' ||
			c == '@' || c == '&' || c == '%' || c == '$' || c == '\\' || c == '|') {
			nonAlpha++
		}
	}
	if float64(nonAlpha)/float64(len(tail)) > 0.35 {
		return "DEGENERATE"
	}

	if len(text) > 400 {
		last200 := text[len(text)-200:]
		prev200 := text[len(text)-400 : len(text)-200]
		if last200 == prev200 {
			return "PHRASE_REPEAT"
		}
	}

	return "COHERENT"
}
