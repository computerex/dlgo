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
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: gentest_main.go <model.gguf>\n")
		os.Exit(1)
	}
	modelPath := os.Args[1]
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

	if os.Getenv("DLGO_NO_SWA") == "1" {
		pipe.Model.Config.SlidingWindow = 0
		pipe.Model.Config.SlidingWindowPattern = 0
		for i := range pipe.Model.Layers {
			pipe.Model.Layers[i].Spec.SlidingWindow = 0
		}
		fmt.Println("[gentest] SWA disabled via DLGO_NO_SWA=1")
	}
	if os.Getenv("DLGO_NO_SINKS") == "1" {
		for i := range pipe.Model.Layers {
			pipe.Model.Layers[i].AttnSinks = nil
		}
		fmt.Println("[gentest] Attention sinks disabled via DLGO_NO_SINKS=1")
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

	promptText := "Write a comprehensive tutorial on building a REST API with Python Flask. Cover project setup, routing, request handling, database integration with SQLAlchemy, authentication with JWT tokens, error handling, testing, deployment, and best practices. Include complete code examples for each section."
	noThink := false
	msgs := []llm.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: promptText},
	}
	prompt := llm.FormatMessages(cfg, msgs, llm.FormatOptions{EnableThinking: &noThink})
	tokens := pipe.Tokenizer.Encode(prompt)
	fmt.Printf("Prompt: %d tokens, generating up to %d tokens\n", len(tokens), maxTokens)
	fmt.Printf("Architecture: %s, dim=%d, heads=%d, kvHeads=%d, headDim=%d, layers=%d\n",
		cfg.Architecture, cfg.EmbeddingDim, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.NumLayers)

	logits := make([]float32, cfg.VocabSize)
	gpuPipe.ResetState()

	layerConfs := gpuPipe.LayerConfs

	for i, tok := range tokens {
		gpu.GpuForwardFused(m, gm, gpuPipe, tok, i, kv, rs, logits, layerConfs)
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

		gpu.GpuForwardFused(m, gm, gpuPipe, tok, pos, kv, rs, logits, layerConfs)
		tok = argmax(logits[:cfg.VocabSize])
		pos++
	}

	generated := pos - len(tokens)
	fmt.Printf("\n\n=== Generated %d tokens (coherent) ===\n", generated)
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
