//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/llm"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: dump_tokens.go <model.gguf>\n")
		os.Exit(1)
	}
	modelPath := os.Args[1]

	pipe, err := llm.NewPipeline(modelPath, 4096)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	m := pipe.Model
	cfg := m.Config
	rs := pipe.RunState

	promptText := "Write a comprehensive tutorial on building a REST API with Python Flask. Cover project setup, routing, request handling, database integration with SQLAlchemy, authentication with JWT tokens, error handling, testing, deployment, and best practices. Include complete code examples for each section."
	noThink := false
	msgs := []llm.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: promptText},
	}
	prompt := llm.FormatMessages(cfg, msgs, llm.FormatOptions{EnableThinking: &noThink})
	tokens := pipe.Tokenizer.Encode(prompt)
	fmt.Printf("Architecture: %s, prompt tokens: %d\n", cfg.Architecture, len(tokens))
	fmt.Printf("EmbedScale: %.4f, HeadDim: %d, Heads: %d, KVHeads: %d\n",
		cfg.EmbedScale, cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads)
	fmt.Printf("SlidingWindow: %d, Pattern: %d\n", cfg.SlidingWindow, cfg.SlidingWindowPattern)
	fmt.Printf("AttnLogitSoftcap: %.4f, FinalLogitSoftcap: %.4f\n", cfg.AttnLogitSoftcap, cfg.FinalLogitSoftcap)
	fmt.Printf("FFNGelu: %v, RopeNeox: %v\n", cfg.FFNGelu, cfg.RopeNeox)
	fmt.Printf("Prompt tokens: %v\n", tokens[:10])

	var logits []float32
	for i, tok := range tokens {
		logits = llm.Forward(m, tok, i, pipe.KVCache, rs)
	}

	pos := len(tokens)
	for step := 0; step < 50; step++ {
		tok := argmax(logits[:cfg.VocabSize])
		tokStr := pipe.Tokenizer.DecodeToken(tok)

		top3 := topK(logits[:cfg.VocabSize], 3)
		fmt.Printf("step=%3d pos=%3d tok=%6d \"%s\" top3=[", step, pos, tok, escape(tokStr))
		for i, t := range top3 {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%d:%.3f", t.id, t.val)
		}
		fmt.Println("]")

		logits = llm.Forward(m, tok, pos, pipe.KVCache, rs)
		pos++
	}
}

func argmax(v []float32) int32 {
	best := int32(0)
	for i := int32(1); i < int32(len(v)); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

type tokVal struct {
	id  int32
	val float32
}

func topK(v []float32, k int) []tokVal {
	result := make([]tokVal, k)
	for i := range result {
		result[i].val = -1e30
	}
	for i, val := range v {
		minIdx := 0
		for j := 1; j < k; j++ {
			if result[j].val < result[minIdx].val {
				minIdx = j
			}
		}
		if val > result[minIdx].val {
			result[minIdx] = tokVal{int32(i), val}
		}
	}
	for i := 0; i < k-1; i++ {
		for j := i + 1; j < k; j++ {
			if result[j].val > result[i].val {
				result[i], result[j] = result[j], result[i]
			}
		}
	}
	return result
}

func escape(s string) string {
	out := ""
	for _, c := range s {
		if c == '\n' {
			out += "\\n"
		} else if c == '\t' {
			out += "\\t"
		} else {
			out += string(c)
		}
	}
	return out
}
