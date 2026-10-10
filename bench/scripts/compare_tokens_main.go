//go:build ignore

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := `C:\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`

	if err := gpu.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "gpu init: %v\n", err)
		os.Exit(1)
	}

	pipe, err := llm.NewPipeline(modelPath, 4096)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	cfg := pipe.Model.Config
	fmt.Fprintf(os.Stderr, "Model: %s, layers=%d, dim=%d\n", cfg.Architecture, cfg.NumLayers, cfg.EmbeddingDim)

	msgs := []llm.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Write a detailed timeline of the major events of World War 2, from 1939 to 1945."},
	}
	thinkOff := false
	prompt := llm.FormatMessages(cfg, msgs, llm.FormatOptions{EnableThinking: &thinkOff})

	fmt.Fprintf(os.Stderr, "DLGO_CPU_MOE=%s\n", os.Getenv("DLGO_CPU_MOE"))

	// GPU greedy
	fmt.Fprintf(os.Stderr, "\n=== GPU greedy (500 tokens) ===\n")
	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gpu pipeline: %v\n", err)
		os.Exit(1)
	}

	gpuCfg := llm.DefaultGenerateConfig()
	gpuCfg.MaxTokens = 2000
	gpuCfg.Sampler.Temperature = 0
	var gpuTokens []string
	gpuCfg.Stream = func(token string) {
		gpuTokens = append(gpuTokens, token)
		fmt.Print(token)
	}
	gpuPipe.GenerateDetailed(prompt, gpuCfg)
	gpuText := strings.Join(gpuTokens, "")
	fmt.Fprintf(os.Stderr, "\n\nGPU total tokens: %d\n", len(gpuTokens))

	// Robust repetition detection: check for repeated n-grams and substrings
	words := strings.Fields(gpuText)
	detected := false
	if len(words) > 40 {
		// 1. Check consecutive identical words (3+ in a row)
		consec := 1
		maxConsec := 1
		for i := 1; i < len(words); i++ {
			if words[i] == words[i-1] {
				consec++
				if consec > maxConsec {
					maxConsec = consec
				}
			} else {
				consec = 1
			}
		}
		if maxConsec >= 5 {
			fmt.Fprintf(os.Stderr, "WARNING: %d consecutive identical words detected!\n", maxConsec)
			detected = true
		}

		// 2. Check repeated 4-grams in a sliding window
		type ngram [4]string
		ngramCount := make(map[ngram]int)
		for i := 0; i+4 <= len(words); i++ {
			ng := ngram{words[i], words[i+1], words[i+2], words[i+3]}
			ngramCount[ng]++
		}
		maxNG := 0
		for _, c := range ngramCount {
			if c > maxNG {
				maxNG = c
			}
		}
		if maxNG >= 10 {
			fmt.Fprintf(os.Stderr, "WARNING: 4-gram repeated %d times!\n", maxNG)
			detected = true
		}

		// 3. Check for short repeated substring loops (e.g. "X/X/X/")
		if len(gpuText) > 200 {
			tail := gpuText[len(gpuText)-200:]
			for patLen := 3; patLen <= 40; patLen++ {
				pat := tail[len(tail)-patLen:]
				reps := 0
				for off := len(tail) - patLen; off >= 0; off -= patLen {
					if tail[off:off+patLen] == pat {
						reps++
					} else {
						break
					}
				}
				if reps >= 4 {
					fmt.Fprintf(os.Stderr, "WARNING: tail substring %q repeated %d times!\n", pat, reps)
					detected = true
					break
				}
			}
		}
	}
	if !detected {
		fmt.Fprintf(os.Stderr, "OK: No significant repetition detected.\n")
	}

	gpuPipe.FreeAll()
	gpu.Shutdown()
}
