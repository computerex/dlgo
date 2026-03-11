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
	path := `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`
	pipe, err := llm.NewPipeline(path, 512)
	if err != nil {
		fmt.Printf("Load fail: %v\n", err)
		os.Exit(1)
	}

	cfg := pipe.Model.Config
	m := pipe.Model

	prompts := []string{
		"The capital of France is",
		"1 + 1 =",
		"Hello, my name is",
	}

	for _, prompt := range prompts {
		fmt.Printf("\n=== Prompt: %q ===\n", prompt)
		tokens := pipe.Tokenizer.Encode(prompt)
		fmt.Printf("Tokens (%d): %v\n", len(tokens), tokens)

		kvDim := cfg.NumKVHeads * cfg.HeadDim
		rs := llm.NewRunState(cfg, 512)
		kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)

		for i, tok := range tokens {
			llm.Forward(m, tok, i, kv, rs)
		}

		rng := rand.New(rand.NewSource(42))
		sampler := ops.SamplerConfig{Temperature: 0}
		pos := len(tokens)

		var generated []int32
		var recentTokens []int32

		start := time.Now()
		for step := 0; step < 32; step++ {
			nextToken := ops.SampleToken(rs.Logits, sampler, recentTokens, rng)
			generated = append(generated, int32(nextToken))
			recentTokens = append(recentTokens, int32(nextToken))

			lastTok := int32(nextToken)
			if lastTok == cfg.EOS {
				break
			}
			stop := false
			for _, st := range cfg.StopTokens {
				if lastTok == st {
					stop = true
					break
				}
			}
			if stop {
				break
			}

			llm.Forward(m, lastTok, pos, kv, rs)
			pos++
		}
		elapsed := time.Since(start)
		text := pipe.Tokenizer.Decode(generated)
		fmt.Printf("Output (%dms): %s\n", elapsed.Milliseconds(), text)
	}
}
