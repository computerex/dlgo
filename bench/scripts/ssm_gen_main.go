//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	path := `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`
	pipe, err := llm.NewPipeline(path, 512)
	if err != nil {
		fmt.Printf("Load fail: %v\n", err)
		os.Exit(1)
	}

	cfg := pipe.Model.Config
	m := pipe.Model

	prompt := "The capital of France is"
	allTokens := pipe.Tokenizer.Encode(prompt)

	// Remove BOS to match Ollama raw behavior
	var tokens []int32
	if len(allTokens) > 0 && allTokens[0] == cfg.BOS {
		tokens = allTokens[1:]
	} else {
		tokens = allTokens
	}
	fmt.Printf("Prompt: %q\nTokens (%d): %v\n", prompt, len(tokens), tokens)

	kvDim := cfg.NumKVHeads * cfg.HeadDim
	rs := llm.NewRunState(cfg, 512)
	kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)

	// Process prompt tokens
	for i, tok := range tokens {
		llm.Forward(m, tok, i, kv, rs)
	}

	// Generate 20 tokens
	fmt.Printf("\nGeneration (greedy, no BOS): ")
	pos := len(tokens)
	for g := 0; g < 20; g++ {
		top := argmax(rs.Logits)
		tok := int32(top)
		text := pipe.Tokenizer.DecodeToken(tok)
		fmt.Printf("%s", text)
		llm.Forward(m, tok, pos, kv, rs)
		pos++
	}
	fmt.Printf("\n")

	// Also show top-10 after prompt
	fmt.Printf("\nTop-10 after prompt:\n")

	// Reset and re-process to get logits after prompt
	rs2 := llm.NewRunState(cfg, 512)
	kv2 := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)
	for i, tok := range tokens {
		llm.Forward(m, tok, i, kv2, rs2)
	}
	indices := topK(rs2.Logits, 10)
	for _, idx := range indices {
		fmt.Printf("  tok=%d %q logit=%.4f\n", idx, pipe.Tokenizer.DecodeToken(int32(idx)), rs2.Logits[idx])
	}
}

func argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

func topK(x []float32, k int) []int {
	indices := make([]int, k)
	for j := 0; j < k; j++ {
		best := -1
		for i := 0; i < len(x); i++ {
			skip := false
			for _, prev := range indices[:j] {
				if i == prev {
					skip = true
					break
				}
			}
			if skip {
				continue
			}
			if best < 0 || x[i] > x[best] {
				best = i
			}
		}
		indices[j] = best
	}
	return indices
}
