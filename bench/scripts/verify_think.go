//go:build ignore

package main

import (
	"fmt"
	"math"
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

	// Chat prompt WITH <think> — previously caused immediate <|endoftext|>
	prompt := "<|im_start|>user\nExplain how Python decorators work. Give a simple example.<|im_end|>\n<|im_start|>assistant\n<think>\n"
	tokens := pipe.Tokenizer.Encode(prompt)

	fmt.Fprintf(os.Stderr, "PROMPT: %q\n", prompt)
	fmt.Fprintf(os.Stderr, "TOKENS: %d\n", len(tokens))
	fmt.Fprintf(os.Stderr, "SUPPRESS TOKENS: %v\n\n", cfg.SuppressTokens)

	rs := llm.NewRunState(cfg, 8192)
	kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 8192, kvDim)
	bs := llm.NewBatchState(cfg, 8192)

	llm.ForwardBatch(pipe.Model, tokens, 0, kv, rs, bs)

	rng := rand.New(rand.NewSource(42))
	sampler := ops.SamplerConfig{Temperature: 0.7, TopK: 40, TopP: 0.9}

	// Apply logit suppression before first sample
	for _, id := range cfg.SuppressTokens {
		if int(id) < len(rs.Logits) {
			rs.Logits[id] = float32(math.Inf(-1))
		}
	}

	// Show top-5 logits after suppression
	type lp struct {
		id    int
		logit float32
	}
	top5 := make([]lp, 5)
	for i := range top5 {
		top5[i].logit = float32(math.Inf(-1))
	}
	for i, v := range rs.Logits {
		for j := range top5 {
			if v > top5[j].logit {
				copy(top5[j+1:], top5[j:4])
				top5[j] = lp{i, v}
				break
			}
		}
	}
	fmt.Fprintf(os.Stderr, "TOP-5 LOGITS (after suppress):\n")
	for _, t := range top5 {
		piece := pipe.Tokenizer.DecodeToken(int32(t.id))
		fmt.Fprintf(os.Stderr, "  [%d] %q  logit=%.4f\n", t.id, piece, t.logit)
	}
	fmt.Fprintln(os.Stderr)

	eosTokens := map[int32]bool{
		151643: true, // <|endoftext|>
		151645: true, // <|im_end|>
	}

	var recentTokens []int32
	nextToken := ops.SampleToken(rs.Logits, sampler, recentTokens, rng)
	var generated []int32
	generated = append(generated, int32(nextToken))
	recentTokens = append(recentTokens, int32(nextToken))

	pos := len(tokens)
	maxGen := 200
	for i := 1; i < maxGen; i++ {
		if eosTokens[int32(nextToken)] {
			break
		}
		llm.Forward(pipe.Model, int32(nextToken), pos, kv, rs)
		pos++
		// Apply suppression
		for _, id := range cfg.SuppressTokens {
			if int(id) < len(rs.Logits) {
				rs.Logits[id] = float32(math.Inf(-1))
			}
		}
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
