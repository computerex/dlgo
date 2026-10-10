//go:build ignore

package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: raw_test <gguf_path> <max_tokens> [raw|chat]")
		os.Exit(1)
	}
	ggufPath := os.Args[1]
	maxTokens := 50
	if v, err := strconv.Atoi(os.Args[2]); err == nil {
		maxTokens = v
	}
	mode := "raw"
	if len(os.Args) > 3 {
		mode = os.Args[3]
	}

	useGPU := os.Getenv("DLGO_GPU") != "0"
	if useGPU {
		if err := gpu.Init(); err != nil {
			fmt.Fprintf(os.Stderr, "gpu init: %v\n", err)
			os.Exit(1)
		}
		defer gpu.Shutdown()
	}

	pipe, err := llm.NewPipeline(ggufPath, 4096)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	var gpuPipe *gpu.GpuPipeline
	if useGPU {
		gpuPipe, err = gpu.NewGpuPipeline(pipe)
		if err != nil {
			fmt.Fprintf(os.Stderr, "gpu: %v\n", err)
			os.Exit(1)
		}
		defer gpuPipe.FreeAll()
	}

	cfg := pipe.Model.Config
	var prompt string
	switch mode {
	case "chat":
		prompt = llm.FormatChat(cfg, "You are a helpful assistant.", "Write about the history of AI.")
		fmt.Fprintf(os.Stderr, "Using chat template: %s (thinking=auto)\n", cfg.ChatTemplate)
	case "chat_nothink":
		noThink := false
		msgs := []llm.Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "Write about the history of AI."},
		}
		prompt = llm.FormatMessages(cfg, msgs, llm.FormatOptions{EnableThinking: &noThink})
		fmt.Fprintf(os.Stderr, "Using chat template: %s (thinking=disabled)\n", cfg.ChatTemplate)
	case "chat_plain":
		prompt = llm.FormatChat(cfg, "You are a helpful assistant.", "Write a detailed essay about the history of artificial intelligence")
		// Strip any <think> injection - let the model decide
		if idx := len(prompt); idx > 0 {
			for strings.HasSuffix(prompt, "<think>\n") || strings.HasSuffix(prompt, "<think></think>\n") {
				if strings.HasSuffix(prompt, "<think>\n") {
					prompt = prompt[:len(prompt)-len("<think>\n")]
				} else if strings.HasSuffix(prompt, "<think></think>\n") {
					prompt = prompt[:len(prompt)-len("<think></think>\n")]
				}
			}
		}
		fmt.Fprintf(os.Stderr, "Using chat template: %s (no thinking injection)\n", cfg.ChatTemplate)
	default:
		prompt = "The history of artificial intelligence begins"
		if envPrompt := os.Getenv("DLGO_PROMPT"); envPrompt != "" {
			prompt = envPrompt
		}
		fmt.Fprintf(os.Stderr, "Using raw prompt (no template)\n")
	}
	fmt.Fprintf(os.Stderr, "Prompt: %q\n", prompt[:min(len(prompt), 200)])

	genCfg := llm.DefaultGenerateConfig()
	genCfg.MaxTokens = maxTokens
	genCfg.Seed = 42
	genCfg.Sampler.Temperature = 0

	if gpuPipe != nil {
		result, err := gpuPipe.GenerateDetailed(prompt, genCfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "gen: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "%.1f tok/s, %d tokens\n", result.TokensPerSec, result.TotalTokens)
		fmt.Println(result.Text)
	} else {
		result, err := pipe.GenerateDetailed(prompt, genCfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "gen: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "%.1f tok/s, %d tokens\n", result.TokensPerSec, result.TotalTokens)
		fmt.Println(result.Text)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
