//go:build ignore

package main

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	modelPath := os.Args[1]

	pipe, err := llm.NewPipeline(modelPath, 8192)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	cfg := pipe.Model.Config
	kvDim := cfg.NumKVHeads * cfg.HeadDim

	// Chat prompt WITHOUT <think>
	prompt := "<|im_start|>user\nExplain how Python decorators work. Give a simple example.<|im_end|>\n<|im_start|>assistant\n"
	tokens := pipe.Tokenizer.Encode(prompt)

	fmt.Fprintf(os.Stderr, "PROMPT: %q\n", prompt)
	fmt.Fprintf(os.Stderr, "TOKENS: %d\n\n", len(tokens))

	rs := llm.NewRunState(cfg, 8192)
	kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 8192, kvDim)
	bs := llm.NewBatchState(cfg, 8192)

	llm.ForwardBatch(pipe.Model, tokens, 0, kv, rs, bs)

	rng := rand.New(rand.NewSource(42))
	sampler := ops.SamplerConfig{Temperature: 0.7, TopK: 40, TopP: 0.9}

	eosTokens := map[int32]bool{
		151643: true, // <|endoftext|>
		151645: true, // <|im_end|>
	}

	pos := len(tokens)
	var generated []int32
	var recentTokens []int32

	nextToken := ops.SampleToken(rs.Logits, sampler, recentTokens, rng)
	generated = append(generated, int32(nextToken))
	recentTokens = append(recentTokens, int32(nextToken))

	maxGen := 200
	for i := 1; i < maxGen; i++ {
		if eosTokens[int32(nextToken)] {
			break
		}
		llm.Forward(pipe.Model, int32(nextToken), pos, kv, rs)
		pos++
		nextToken = ops.SampleToken(rs.Logits, sampler, recentTokens, rng)
		generated = append(generated, int32(nextToken))
		recentTokens = append(recentTokens, int32(nextToken))
	}

	text := ""
	for _, t := range generated {
		text += pipe.Tokenizer.DecodeToken(t)
	}

	fmt.Fprintf(os.Stderr, "GENERATED (%d tokens):\n%s\n", len(generated), text)
}
