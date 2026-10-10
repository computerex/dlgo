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
		fmt.Fprintf(os.Stderr, "usage: cpu_gentest.go <model.gguf> [prompt]\n")
		os.Exit(1)
	}
	modelPath := os.Args[1]
	promptText := "Write a comprehensive tutorial on building a REST API with Python Flask. Cover project setup, routing, request handling, database integration with SQLAlchemy, authentication with JWT tokens, error handling, testing, deployment, and best practices. Include complete code examples for each section."
	if len(os.Args) > 2 {
		promptText = os.Args[2]
	}

	pipe, err := llm.NewPipeline(modelPath, 4096)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	m := pipe.Model
	cfg := m.Config
	rs := llm.NewRunState(cfg, 4096)

	noThink := false
	msgs := []llm.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: promptText},
	}
	prompt := llm.FormatMessages(cfg, msgs, llm.FormatOptions{EnableThinking: &noThink})
	tokens := pipe.Tokenizer.Encode(prompt)
	fmt.Fprintf(os.Stderr, "Prompt: %d tokens, generating up to 4096 tokens\n", len(tokens))

	var logits []float32
	for i, tok := range tokens {
		logits = llm.Forward(m, tok, i, pipe.KVCache, rs)
	}

	var sb strings.Builder
	pos := len(tokens)
	for step := 0; step < 4096; step++ {
		tok := argmax(logits[:cfg.VocabSize])
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
		text := sb.String()
		if len(text) > 200 {
			last100 := text[len(text)-100:]
			prev100 := text[len(text)-200 : len(text)-100]
			if last100 == prev100 {
				fmt.Fprintf(os.Stderr, "\n[PHRASE REPETITION at step %d]\n", step)
				os.Exit(1)
			}
		}
		logits = llm.Forward(m, tok, pos, pipe.KVCache, rs)
		pos++
	}
	generated := pos - len(tokens)
	fmt.Fprintf(os.Stderr, "\n=== Generated %d tokens (coherent) ===\n", generated)
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
