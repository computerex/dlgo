//go:build ignore

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := `C:\models\Qwen3.5-2B.Q4_K_M.gguf`
	maxTokens := 4096

	if err := gpu.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "gpu init: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	pipe, err := llm.NewPipeline(modelPath, 4096)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gpu pipeline: %v\n", err)
		os.Exit(1)
	}
	defer gpuPipe.FreeAll()

	m := pipe.Model
	cfg := m.Config
	gm := gpuPipe.GpuModel
	rs := gpuPipe.RunState
	kv := gpuPipe.KVCache

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
	fmt.Printf("Prompt: %d tokens, generating up to %d tokens\n", len(tokens), maxTokens)

	logits := make([]float32, cfg.VocabSize)
	gpuPipe.ResetState()

	// Use the fused forward path (what the real pipeline uses)
	useFused := os.Getenv("DLGO_NO_FUSED") != "1"
	fmt.Printf("Using fused forward: %v, dp4a disabled: %v\n", useFused, os.Getenv("DLGO_NO_DP4A") == "1")

	layerConfs := gpuPipe.LayerConfs

	// Prefill
	for i, tok := range tokens {
		if useFused {
			gpu.GpuForwardFused(m, gm, gpuPipe, tok, i, kv, rs, logits, layerConfs)
		} else {
			gpu.GpuForward(m, gm, tok, i, kv, rs, logits, gpuPipe)
			gpu.Sync()
		}
	}

	// Generate
	var sb strings.Builder
	pos := len(tokens)
	tok := argmax(logits[:cfg.VocabSize])
	repeatCount := 0
	lastTok := int32(-1)
	lastPhrase := ""

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
				fmt.Printf("\n\n[DEGENERATION DETECTED at step %d: token %d repeated 20+ times]\n", step, tok)
				break
			}
		} else {
			repeatCount = 0
		}
		lastTok = tok

		// Check for phrase-level repetition
		text := sb.String()
		if len(text) > 200 {
			last100 := text[len(text)-100:]
			prev100 := text[len(text)-200 : len(text)-100]
			if last100 == prev100 && last100 != lastPhrase {
				lastPhrase = last100
				fmt.Printf("\n\n[PHRASE REPETITION DETECTED at step %d]\n", step)
				break
			}
		}

		if useFused {
			gpu.GpuForwardFused(m, gm, gpuPipe, tok, pos, kv, rs, logits, layerConfs)
		} else {
			gpu.GpuForward(m, gm, tok, pos, kv, rs, logits, gpuPipe)
			gpu.Sync()
		}
		tok = argmax(logits[:cfg.VocabSize])
		pos++
	}

	fmt.Printf("\n\n=== Generated %d tokens ===\n", pos-len(tokens))
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
