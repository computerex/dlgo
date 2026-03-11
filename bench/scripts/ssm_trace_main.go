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
	tokens := pipe.Tokenizer.Encode(prompt)
	fmt.Printf("Prompt: %q\nTokens (%d): %v\n\n", prompt, len(tokens), tokens)

	kvDim := cfg.NumKVHeads * cfg.HeadDim
	rs := llm.NewRunState(cfg, 512)
	kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)

	for i, tok := range tokens {
		llm.Forward(m, tok, i, kv, rs)
		top := argmax(rs.Logits)
		topStr := pipe.Tokenizer.DecodeToken(int32(top))
		top2, _ := argmax2(rs.Logits, top)
		top2Str := pipe.Tokenizer.DecodeToken(int32(top2))

		layerType := "?"
		if i < cfg.NumLayers {
			if (i+1)%cfg.FullAttentionInterval == 0 {
				layerType = "attn"
			} else {
				layerType = "ssm"
			}
		}

		fmt.Printf("pos=%d tok=%d (%q) → top=%d %q (%.2f) 2nd=%d %q (%.2f) [first layer=%s]\n",
			i, tok, pipe.Tokenizer.DecodeToken(tok), top, topStr, rs.Logits[top],
			top2, top2Str, rs.Logits[top2], layerType)
	}

	fmt.Printf("\nExpected next token: 'Paris' or similar\n")

	fmt.Printf("\nTop-10 logits after all prompt tokens:\n")
	indices := topK(rs.Logits, 10)
	for _, idx := range indices {
		fmt.Printf("  tok=%d %q logit=%.4f\n", idx, pipe.Tokenizer.DecodeToken(int32(idx)), rs.Logits[idx])
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

func argmax2(x []float32, exclude int) (int, float32) {
	best := -1
	for i := 0; i < len(x); i++ {
		if i == exclude {
			continue
		}
		if best < 0 || x[i] > x[best] {
			best = i
		}
	}
	return best, x[best]
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
