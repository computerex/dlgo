//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

type logitEntry struct {
	idx int
	val float32
}

func topK(logits []float32, k int) []logitEntry {
	entries := make([]logitEntry, len(logits))
	for i, v := range logits {
		entries[i] = logitEntry{i, v}
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].val > entries[j].val
	})
	if k > len(entries) {
		k = len(entries)
	}
	return entries[:k]
}

func l2(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

func diffL2(a, b []float32) float64 {
	var s float64
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		d := float64(a[i] - b[i])
		s += d * d
	}
	return math.Sqrt(s)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: logit_cmp <gguf_path>")
		os.Exit(1)
	}
	ggufPath := os.Args[1]

	if err := gpu.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "gpu init: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	pipe, err := llm.NewPipeline(ggufPath, 4096)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gpu: %v\n", err)
		os.Exit(1)
	}
	defer gpuPipe.FreeAll()

	cfg := pipe.Model.Config
	prompt := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
	tokens := pipe.Tokenizer.Encode(prompt)
	fmt.Printf("Prompt: %d tokens, dim: %d, layers: %d\n", len(tokens), cfg.EmbeddingDim, cfg.NumLayers)
	fmt.Printf("Fused forward: %v, HasSSM: %v, HasGatedQ: %v\n", gpuPipe.UseFusedForward, gpuPipe.HasSSM, gpuPipe.HasGatedQ)

	// CPU prefill: batch forward
	pipe.KVCache.Reset()
	if pipe.RunState.SSMState != nil {
		pipe.RunState.SSMState.Reset()
	}
	llm.ForwardBatch(pipe.Model, tokens, 0, pipe.KVCache, pipe.RunState, pipe.BatchState)
	cpuLogits := make([]float32, cfg.VocabSize)
	copy(cpuLogits, pipe.RunState.Logits)

	// GPU prefill: use the pipeline's normal prefill path (same as GenerateDetailed)
	gpuPipe.ResetState()
	genCfg := llm.DefaultGenerateConfig()
	genCfg.MaxTokens = 1
	genCfg.Seed = 42
	genCfg.Sampler.Temperature = 0
	gpuResult, err := gpuPipe.GenerateDetailed(prompt, genCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gpu gen: %v\n", err)
		os.Exit(1)
	}
	gpuLogits := make([]float32, cfg.VocabSize)
	copy(gpuLogits, gpuPipe.LogitsBuf)

	// CPU: first generated token
	cpuTop := topK(cpuLogits, 10)
	fmt.Printf("\n=== CPU Top-10 ===\n")
	for _, e := range cpuTop {
		tok := pipe.Tokenizer.DecodeToken(int32(e.idx))
		fmt.Printf("  [%6d] %8.4f  %q\n", e.idx, e.val, tok)
	}

	// GPU: first generated token  
	gpuTop := topK(gpuLogits, 10)
	fmt.Printf("\n=== GPU Top-10 ===\n")
	for _, e := range gpuTop {
		tok := pipe.Tokenizer.DecodeToken(int32(e.idx))
		fmt.Printf("  [%6d] %8.4f  %q\n", e.idx, e.val, tok)
	}

	fmt.Printf("\n=== GPU first token: %q ===\n", gpuResult.Text)

	// Norms
	cN := l2(cpuLogits)
	gN := l2(gpuLogits)
	dN := diffL2(cpuLogits, gpuLogits)
	fmt.Printf("\nCPU logit norm: %.4f\n", cN)
	fmt.Printf("GPU logit norm: %.4f\n", gN)
	fmt.Printf("Diff L2:        %.4f (%.2f%%)\n", dN, 100*dN/cN)

	// Max abs diff
	maxDiff := float32(0)
	maxIdx := 0
	for i := 0; i < len(cpuLogits) && i < len(gpuLogits); i++ {
		d := cpuLogits[i] - gpuLogits[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
			maxIdx = i
		}
	}
	fmt.Printf("Max abs diff:   %.6f at [%d]\n", maxDiff, maxIdx)
}
