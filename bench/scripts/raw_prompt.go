//go:build ignore

package main

import (
	"fmt"
	"os"
	"sort"

	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := os.Args[1]

	pipe, err := llm.NewPipeline(modelPath, 8192)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	cfg := pipe.Model.Config
	vocabSize := cfg.VocabSize
	kvDim := cfg.NumKVHeads * cfg.HeadDim

	type logitEntry struct {
		idx int
		val float32
	}

	testPrompts := []struct {
		name   string
		prompt string
	}{
		{"chatml_no_think", "<|im_start|>user\nExplain how Python decorators work. Give a simple example.<|im_end|>\n<|im_start|>assistant\n"},
		{"chatml_with_think", "<|im_start|>user\nExplain how Python decorators work. Give a simple example.<|im_end|>\n<|im_start|>assistant\n<think>\n"},
		{"raw_text", "The quick brown fox jumps over the lazy dog. The meaning of life is"},
		{"raw_code", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"},
	}

	for _, test := range testPrompts {
		tokens := pipe.Tokenizer.Encode(test.prompt)
		fmt.Fprintf(os.Stderr, "\n=== %s (%d tokens) ===\n", test.name, len(tokens))
		fmt.Fprintf(os.Stderr, "PROMPT: %q\n", test.prompt)

		rs := llm.NewRunState(cfg, 8192)
		kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 8192, kvDim)
		for pos, tok := range tokens {
			llm.Forward(pipe.Model, int32(tok), pos, kv, rs)
		}

		sorted := make([]logitEntry, vocabSize)
		for i := 0; i < vocabSize; i++ {
			sorted[i] = logitEntry{i, rs.Logits[i]}
		}
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].val > sorted[j].val })

		fmt.Fprintf(os.Stderr, "Top-10:\n")
		for i := 0; i < 10 && i < len(sorted); i++ {
			e := sorted[i]
			fmt.Fprintf(os.Stderr, "  %2d: [%6d] %10.4f  %q\n", i, e.idx, e.val, pipe.Tokenizer.DecodeToken(int32(e.idx)))
		}

		eotID := 151643
		eotLogit := rs.Logits[eotID]
		eotRank := -1
		for r, e := range sorted {
			if e.idx == eotID {
				eotRank = r
				break
			}
		}
		fmt.Fprintf(os.Stderr, "  eot: rank=%d logit=%.4f\n", eotRank, eotLogit)
		fmt.Fprintf(os.Stderr, "  gap(top1-top2): %.4f\n", sorted[0].val-sorted[1].val)
	}
}
