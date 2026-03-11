//go:build ignore

package main

import (
	"fmt"

	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	paths := []struct {
		name string
		path string
	}{
		{"9B", `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`},
		{"2B", `C:\projects\gollm\Qwen3.5-2B.Q4_K_M.gguf`},
	}

	for _, p := range paths {
		fmt.Printf("\n========== %s ==========\n", p.name)
		pipe, err := llm.NewPipeline(p.path, 512)
		if err != nil {
			fmt.Printf("Load fail: %v\n", err)
			continue
		}
		cfg := pipe.Model.Config
		m := pipe.Model

		kvDim := cfg.NumKVHeads * cfg.HeadDim
		rs := llm.NewRunState(cfg, 512)
		kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)

		prompt := "The capital of France is"

		// Get all tokens from Encode, then remove BOS if present
		allTokens := pipe.Tokenizer.Encode(prompt)
		fmt.Printf("With BOS: %v\n", allTokens)

		// Tokens WITHOUT BOS (matching Ollama behavior)
		var tokens []int32
		if len(allTokens) > 0 && allTokens[0] == cfg.BOS {
			tokens = allTokens[1:]
		} else {
			tokens = allTokens
		}
		fmt.Printf("Without BOS: %v\n", tokens)

		// Run WITHOUT BOS
		for i, tok := range tokens {
			llm.Forward(m, tok, i, kv, rs)
		}

		top := argmax(rs.Logits)
		fmt.Printf("WITHOUT BOS: top=%d %q logit=%.4f\n", top, pipe.Tokenizer.DecodeToken(int32(top)), rs.Logits[top])

		top10 := topK(rs.Logits, 10)
		for _, idx := range top10 {
			fmt.Printf("  tok=%d %q logit=%.4f\n", idx, pipe.Tokenizer.DecodeToken(int32(idx)), rs.Logits[idx])
		}

		// Now run WITH BOS
		rs2 := llm.NewRunState(cfg, 512)
		kv2 := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)
		for i, tok := range allTokens {
			llm.Forward(m, tok, i, kv2, rs2)
		}
		top2 := argmax(rs2.Logits)
		fmt.Printf("WITH BOS: top=%d %q logit=%.4f\n", top2, pipe.Tokenizer.DecodeToken(int32(top2)), rs2.Logits[top2])
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
