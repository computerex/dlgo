//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"
	"sort"

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

	rawPrompt := "Explain how Python decorators work. Give a simple example."
	messages := []llm.Message{
		{Role: "user", Content: rawPrompt},
	}
	prompt := llm.FormatMessages(pipe.Model.Config, messages)
	fmt.Fprintf(os.Stderr, "PROMPT: %q\n", prompt)
	promptTokens := pipe.Tokenizer.Encode(prompt)
	cfg := pipe.Model.Config
	vocabSize := cfg.VocabSize
	fmt.Fprintf(os.Stderr, "prompt_tokens=%d\n", len(promptTokens))
	fmt.Fprintf(os.Stderr, "VocabSize=%d EmbedRows=%d EmbedCols=%d\n", cfg.VocabSize, pipe.Model.TokenEmbed.Rows, pipe.Model.TokenEmbed.Cols)
	fmt.Fprintf(os.Stderr, "ALL_TOKENS: ")
	for i, tid := range promptTokens {
		fmt.Fprintf(os.Stderr, "[%d]%d=%q ", i, tid, pipe.Tokenizer.DecodeToken(tid))
	}
	fmt.Fprintf(os.Stderr, "\n")
	
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	
	// Also test with a minimal 3-token prompt
	miniTokens := promptTokens[:3]
	fmt.Fprintf(os.Stderr, "\n=== MINI PROMPT (first 3 tokens) ===\n")
	cpuRS2 := llm.NewRunState(cfg, 8192)
	cpuKV2 := memory.NewMultiLayerKVCache(cfg.NumLayers, 8192, kvDim)
	for i, tok := range miniTokens {
		llm.Forward(pipe.Model, int32(tok), i, cpuKV2, cpuRS2)
	}
	miniLogits := make([]float32, vocabSize)
	copy(miniLogits, cpuRS2.Logits)

	// === CPU forward (per-token, matching GPU MoE path) ===
	cpuRS := llm.NewRunState(cfg, 8192)
	cpuKV := memory.NewMultiLayerKVCache(cfg.NumLayers, 8192, kvDim)
	for i, tok := range promptTokens {
		llm.Forward(pipe.Model, int32(tok), i, cpuKV, cpuRS)
	}
	cpuLogits := make([]float32, vocabSize)
	copy(cpuLogits, cpuRS.Logits)
	fmt.Fprintf(os.Stderr, "CPU prefill done\n")

	// CPU sample with temp=0
	sampler := ops.DefaultSamplerConfig()
	sampler.Temperature = 0.0
	cpuSampled := ops.SampleToken(cpuLogits, sampler, nil, nil)
	cpuText := pipe.Tokenizer.DecodeToken(int32(cpuSampled))
	fmt.Fprintf(os.Stderr, "CPU sampled: %q (id=%d)\n", cpuText, cpuSampled)

	// Since we can't directly access GPU logits, compare at the token level
	// But we CAN look at the CPU logits to understand the distribution
	type logitEntry struct {
		idx int
		val float32
	}
	cpuTop := make([]logitEntry, vocabSize)
	for i := 0; i < vocabSize; i++ {
		cpuTop[i] = logitEntry{i, cpuLogits[i]}
	}
	sort.Slice(cpuTop, func(i, j int) bool { return cpuTop[i].val > cpuTop[j].val })

	fmt.Fprintf(os.Stderr, "\nCPU top-20 logits:\n")
	for i := 0; i < 20 && i < len(cpuTop); i++ {
		tok := pipe.Tokenizer.DecodeToken(int32(cpuTop[i].idx))
		fmt.Fprintf(os.Stderr, "  %2d: [%6d] %8.4f  %q\n", i, cpuTop[i].idx, cpuTop[i].val, tok)
	}

	// Check the gap between #1 and #2
	if len(cpuTop) >= 2 {
		gap := cpuTop[0].val - cpuTop[1].val
		fmt.Fprintf(os.Stderr, "\nCPU top1-top2 gap: %.4f (%.2f%%)\n", gap, 100*float64(gap)/float64(math.Abs(float64(cpuTop[0].val))))
	}

	// Mini prompt analysis
	fmt.Fprintf(os.Stderr, "\n=== MINI PROMPT top-10 logits ===\n")
	miniTop := make([]logitEntry, vocabSize)
	for i := 0; i < vocabSize; i++ {
		miniTop[i] = logitEntry{i, miniLogits[i]}
	}
	sort.Slice(miniTop, func(i, j int) bool { return miniTop[i].val > miniTop[j].val })
	for i := 0; i < 10 && i < len(miniTop); i++ {
		tok := pipe.Tokenizer.DecodeToken(int32(miniTop[i].idx))
		fmt.Fprintf(os.Stderr, "  %2d: [%6d] %8.4f  %q\n", i, miniTop[i].idx, miniTop[i].val, tok)
	}
	// Check endoftext position in mini
	for i, e := range miniTop {
		if e.idx == 151643 {
			fmt.Fprintf(os.Stderr, "  <|endoftext|> rank=%d logit=%.4f\n", i, e.val)
			break
		}
	}
}
