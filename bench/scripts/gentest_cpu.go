//go:build ignore

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/computerex/dlgo/models/llm"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: gentest_cpu.go <model.gguf>\n")
		os.Exit(1)
	}
	modelPath := os.Args[1]
	maxTokens := 1000

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
	fmt.Printf("Prompt: %d tokens, generating up to %d tokens (CPU only)\n", len(tokens), maxTokens)
	fmt.Printf("Architecture: %s, dim=%d, heads=%d, kvHeads=%d, headDim=%d, layers=%d\n",
		cfg.Architecture, cfg.EmbeddingDim, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.NumLayers)

	var logits []float32

	logits = llm.Forward(m, tokens[0], 0, pipe.KVCache, rs)
	for i := 1; i < len(tokens); i++ {
		logits = llm.Forward(m, tokens[i], i, pipe.KVCache, rs)
	}

	var sb strings.Builder
	pos := len(tokens)
	tok := argmax(logits[:cfg.VocabSize])
	repeatCount := 0
	lastTok := int32(-1)

	for step := 0; step < maxTokens; step++ {
		if tok == cfg.EOS {
			break
		}
		isStop := false
		for _, stop := range cfg.StopTokens {
			if tok == stop {
				isStop = true
				break
			}
		}
		if isStop {
			break
		}

		tokStr := pipe.Tokenizer.DecodeToken(tok)
		sb.WriteString(tokStr)
		fmt.Print(tokStr)

		if tok == lastTok {
			repeatCount++
			if repeatCount > 20 {
				fmt.Printf("\n\n[DEGENERATION: token repeated 20+ times at step %d]\n", step)
				os.Exit(2)
			}
		} else {
			repeatCount = 0
		}
		lastTok = tok

		text := sb.String()
		if len(text) > 200 {
			last100 := text[len(text)-100:]
			prev100 := text[len(text)-200 : len(text)-100]
			if last100 == prev100 {
				fmt.Printf("\n\n[PHRASE REPETITION at step %d]\n", step)
				os.Exit(2)
			}
		}

		logits = llm.Forward(m, tok, pos, pipe.KVCache, rs)
		tok = argmax(logits[:cfg.VocabSize])
		pos++
	}

	generated := pos - len(tokens)
	fmt.Printf("\n\n=== Generated %d tokens (CPU only) ===\n", generated)
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
