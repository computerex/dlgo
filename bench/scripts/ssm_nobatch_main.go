//go:build ignore

package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	models := []struct{ name, path string }{
		{"Qwen3.5-0.8B", `C:\projects\gollm\Qwen3.5-0.8B-Q8_0.gguf`},
		{"Qwen3.5-9B", `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`},
	}
	for _, mod := range models {
		fmt.Printf("\n=== %s ===\n", mod.name)
		testModelNoBatch(mod.path)
	}
}

func testModelNoBatch(path string) {
	pipe, err := llm.NewPipeline(path, 512)
	if err != nil {
		fmt.Printf("Load fail: %v\n", err)
		return
	}

	cfg := pipe.Model.Config
	m := pipe.Model
	prompt := llm.FormatChat(cfg, "You are a helpful assistant.", "Explain what a computer is in one sentence.")
	tokens := pipe.Tokenizer.Encode(prompt)
	fmt.Printf("Prompt tokens: %d\n", len(tokens))

	kvDim := cfg.NumKVHeads * cfg.HeadDim
	rs := llm.NewRunState(cfg, 512)
	kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)

	fmt.Println("Prefilling token-by-token (no batch)...")
	start := time.Now()
	for i, tok := range tokens {
		llm.Forward(m, tok, i, kv, rs)
	}
	prefillMs := time.Since(start).Milliseconds()
	fmt.Printf("Prefill: %d tokens in %dms\n", len(tokens), prefillMs)

	rng := rand.New(rand.NewSource(42))
	sampler := ops.SamplerConfig{Temperature: 0}

	genStart := time.Now()
	var generated []int32
	var recentTokens []int32
	pos := len(tokens)

	nextToken := ops.SampleToken(rs.Logits, sampler, recentTokens, rng)
	generated = append(generated, int32(nextToken))

	for step := 1; step < 64; step++ {
		if pos >= 511 {
			break
		}
		lastTok := int32(nextToken)
		if lastTok == cfg.EOS {
			break
		}
		for _, stop := range cfg.StopTokens {
			if lastTok == stop {
				goto done
			}
		}
		llm.Forward(m, lastTok, pos, kv, rs)
		pos++
		nextToken = ops.SampleToken(rs.Logits, sampler, recentTokens, rng)
		generated = append(generated, int32(nextToken))
		recentTokens = append(recentTokens, int32(nextToken))
		if len(recentTokens) > 64 {
			recentTokens = recentTokens[1:]
		}
	}

done:
	genMs := time.Since(genStart).Milliseconds()
	text := pipe.Tokenizer.Decode(generated)
	fmt.Printf("Generated %d tokens in %dms\n", len(generated), genMs)
	fmt.Printf("Output: %s\n", text)
}

func unused() { _ = os.Exit }
