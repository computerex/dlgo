//go:build ignore

package main

import (
	"fmt"

	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	path := `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`
	pipe, err := llm.NewPipeline(path, 512)
	if err != nil {
		fmt.Printf("Load fail: %v\n", err)
		return
	}

	prompts := []string{
		"The capital of France is",
		"Water boils at",
		"The largest planet in our solar system is",
	}

	cfg := pipe.Model.Config
	m := pipe.Model

	for _, prompt := range prompts {
		rs := llm.NewRunState(cfg, 512)
		kv := memory.NewMultiLayerKVCache(cfg.NumLayers, cfg.NumKVHeads*cfg.HeadDim, 512)

		allTokens := pipe.Tokenizer.Encode(prompt)
		tokens := allTokens
		if len(tokens) > 0 && tokens[0] == int32(cfg.BOS) {
			tokens = tokens[1:]
		}

		for i, tok := range tokens {
			llm.Forward(m, tok, i, kv, rs)
		}

		fmt.Printf("Prompt: %q\nGen: ", prompt)
		pos := len(tokens)
		for g := 0; g < 25; g++ {
			best := argmax(rs.Logits)
			tok := int32(best)
			fmt.Printf("%s", pipe.Tokenizer.DecodeToken(tok))
			llm.Forward(m, tok, pos, kv, rs)
			pos++
		}
		fmt.Printf("\n\n")
	}
}

func argmax(v []float32) int {
	b := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[b] {
			b = i
		}
	}
	return b
}
