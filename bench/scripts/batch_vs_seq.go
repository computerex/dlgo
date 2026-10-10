//go:build ignore

package main

import (
	"fmt"
	"math"
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

	rawPrompt := "Explain how Python decorators work. Give a simple example."
	messages := []llm.Message{
		{Role: "user", Content: rawPrompt},
	}
	prompt := llm.FormatMessages(pipe.Model.Config, messages)
	promptTokens := pipe.Tokenizer.Encode(prompt)

	cfg := pipe.Model.Config
	vocabSize := cfg.VocabSize
	kvDim := cfg.NumKVHeads * cfg.HeadDim

	fmt.Fprintf(os.Stderr, "PROMPT: %q\n", prompt)
	fmt.Fprintf(os.Stderr, "TOKENS (%d):", len(promptTokens))
	for i, t := range promptTokens {
		fmt.Fprintf(os.Stderr, " [%d]%d=%q", i, t, pipe.Tokenizer.DecodeToken(int32(t)))
	}
	fmt.Fprintf(os.Stderr, "\n\n")

	// === Path 1: Sequential (token-by-token Forward) ===
	seqRS := llm.NewRunState(cfg, 8192)
	seqKV := memory.NewMultiLayerKVCache(cfg.NumLayers, 8192, kvDim)
	for pos, tok := range promptTokens {
		llm.Forward(pipe.Model, int32(tok), pos, seqKV, seqRS)
	}
	seqLogits := make([]float32, vocabSize)
	copy(seqLogits, seqRS.Logits)

	// === Path 2: Batch (ForwardBatch) ===
	batchRS := llm.NewRunState(cfg, 8192)
	batchKV := memory.NewMultiLayerKVCache(cfg.NumLayers, 8192, kvDim)
	batchBS := llm.NewBatchState(cfg, 8192)
	llm.ForwardBatch(pipe.Model, promptTokens, 0, batchKV, batchRS, batchBS)
	batchLogits := make([]float32, vocabSize)
	copy(batchLogits, batchRS.Logits)

	// === Compare ===
	type logitEntry struct {
		idx int
		val float32
	}

	fmt.Fprintf(os.Stderr, "=== SEQUENTIAL top-20 logits ===\n")
	seqSorted := make([]logitEntry, vocabSize)
	for i := 0; i < vocabSize; i++ {
		seqSorted[i] = logitEntry{i, seqLogits[i]}
	}
	sort.Slice(seqSorted, func(i, j int) bool { return seqSorted[i].val > seqSorted[j].val })
	for i := 0; i < 20 && i < len(seqSorted); i++ {
		e := seqSorted[i]
		fmt.Fprintf(os.Stderr, "  %2d: [%6d] %10.4f  %q\n", i, e.idx, e.val, pipe.Tokenizer.DecodeToken(int32(e.idx)))
	}

	fmt.Fprintf(os.Stderr, "\n=== BATCH top-20 logits ===\n")
	batchSorted := make([]logitEntry, vocabSize)
	for i := 0; i < vocabSize; i++ {
		batchSorted[i] = logitEntry{i, batchLogits[i]}
	}
	sort.Slice(batchSorted, func(i, j int) bool { return batchSorted[i].val > batchSorted[j].val })
	for i := 0; i < 20 && i < len(batchSorted); i++ {
		e := batchSorted[i]
		fmt.Fprintf(os.Stderr, "  %2d: [%6d] %10.4f  %q\n", i, e.idx, e.val, pipe.Tokenizer.DecodeToken(int32(e.idx)))
	}

	// Max absolute difference
	maxDiff := float32(0)
	maxDiffIdx := 0
	sumDiff := float64(0)
	for i := 0; i < vocabSize; i++ {
		d := float32(math.Abs(float64(seqLogits[i] - batchLogits[i])))
		sumDiff += float64(d)
		if d > maxDiff {
			maxDiff = d
			maxDiffIdx = i
		}
	}
	fmt.Fprintf(os.Stderr, "\n=== COMPARISON ===\n")
	fmt.Fprintf(os.Stderr, "max_abs_diff=%.6f at token %d (%q)\n", maxDiff, maxDiffIdx, pipe.Tokenizer.DecodeToken(int32(maxDiffIdx)))
	fmt.Fprintf(os.Stderr, "mean_abs_diff=%.6f\n", sumDiff/float64(vocabSize))
	fmt.Fprintf(os.Stderr, "seq_top1=[%d] %.4f %q\n", seqSorted[0].idx, seqSorted[0].val, pipe.Tokenizer.DecodeToken(int32(seqSorted[0].idx)))
	fmt.Fprintf(os.Stderr, "batch_top1=[%d] %.4f %q\n", batchSorted[0].idx, batchSorted[0].val, pipe.Tokenizer.DecodeToken(int32(batchSorted[0].idx)))

	eotID := 151643
	fmt.Fprintf(os.Stderr, "seq_eot_logit=%.4f  batch_eot_logit=%.4f\n", seqLogits[eotID], batchLogits[eotID])
}
