//go:build ignore

package main

import (
	"fmt"
	"runtime"

	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	path := `C:\projects\gollm\Qwen3.5-27B-Q3_K_M.gguf`
	fmt.Println("Loading 27B model...")
	pipe, err := llm.NewPipeline(path, 256)
	if err != nil {
		fmt.Printf("Load fail: %v\n", err)
		return
	}

	cfg := pipe.Model.Config
	m := pipe.Model
	fmt.Printf("Config: layers=%d dim=%d heads=%d kvHeads=%d headDim=%d\n",
		cfg.NumLayers, cfg.EmbeddingDim, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
	fmt.Printf("SSM: inner=%d state=%d conv=%d rank=%d groups=%d interval=%d\n",
		cfg.SSMInnerSize, cfg.SSMStateSize, cfg.SSMConvKernel, cfg.SSMTimeStepRank, cfg.SSMGroupCount, cfg.FullAttentionInterval)

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	fmt.Printf("Memory: alloc=%.1f MB, sys=%.1f MB\n", float64(ms.Alloc)/1e6, float64(ms.Sys)/1e6)

	rs := llm.NewRunState(cfg, 256)
	kv := memory.NewMultiLayerKVCache(cfg.NumLayers, cfg.NumKVHeads*cfg.HeadDim, 256)

	prompt := "The capital of France is"
	allTokens := pipe.Tokenizer.Encode(prompt)
	tokens := allTokens
	if len(tokens) > 0 && tokens[0] == int32(cfg.BOS) {
		tokens = tokens[1:]
	}
	fmt.Printf("\nPrompt: %q  Tokens: %v\n", prompt, tokens)

	for i, tok := range tokens {
		llm.Forward(m, tok, i, kv, rs)
	}

	fmt.Printf("Gen: ")
	pos := len(tokens)
	for g := 0; g < 20; g++ {
		best := argmax(rs.Logits)
		tok := int32(best)
		fmt.Printf("%s", pipe.Tokenizer.DecodeToken(tok))
		llm.Forward(m, tok, pos, kv, rs)
		pos++
	}
	fmt.Printf("\n")
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
