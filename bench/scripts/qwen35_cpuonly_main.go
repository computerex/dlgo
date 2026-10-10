//go:build ignore

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := `C:\models\Qwen3.5-2B.Q4_K_M.gguf`
	maxTokens := 600

	pipe, err := llm.NewPipeline(modelPath, 4096)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	m := pipe.Model
	cfg := m.Config
	rs := llm.NewRunState(cfg, 4096)

	kvDim := cfg.NumKVHeads * cfg.HeadDim
	kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 4096, kvDim)

	promptText := os.Getenv("DLGO_PROMPT")
	if promptText == "" {
		promptText = "Write a Python function for binary search and explain how it works step by step."
	}
	noThink := false
	msgs := []llm.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: promptText},
	}
	prompt := llm.FormatMessages(cfg, msgs, llm.FormatOptions{EnableThinking: &noThink})
	tokens := pipe.Tokenizer.Encode(prompt)
	fmt.Printf("[CPU-ONLY] Prompt: %d tokens, generating up to %d tokens\n", len(tokens), maxTokens)
	fmt.Printf("[CPU-ONLY] Architecture: %s, dim=%d, heads=%d, kvHeads=%d, headDim=%d, layers=%d\n",
		cfg.Architecture, cfg.EmbeddingDim, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.NumLayers)
	fmt.Printf("[CPU-ONLY] FullAttentionInterval=%d, SSM inner=%d, state=%d\n",
		cfg.FullAttentionInterval, cfg.SSMInnerSize, cfg.SSMStateSize)

	var logits []float32
	for i, tok := range tokens {
		logits = llm.Forward(m, tok, i, kv, rs)
	}

	var sb strings.Builder
	pos := len(tokens)
	tok := argmax(logits[:cfg.VocabSize])

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

		text := sb.String()
		if len(text) > 200 {
			last100 := text[len(text)-100:]
			prev100 := text[len(text)-200 : len(text)-100]
			if last100 == prev100 {
				fmt.Printf("\n\n[PHRASE REPETITION DETECTED at step %d]\n", step)
				break
			}
		}

		logits = llm.Forward(m, tok, pos, kv, rs)
		tok = argmax(logits[:cfg.VocabSize])
		pos++
	}

	fmt.Printf("\n\n=== Generated %d tokens (CPU-only, no GPU) ===\n", pos-len(tokens))
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
