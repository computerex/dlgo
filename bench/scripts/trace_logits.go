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

	rawPrompt := "Explain how Python decorators work. Give a simple example."
	messages := []llm.Message{
		{Role: "user", Content: rawPrompt},
	}
	prompt := llm.FormatMessages(pipe.Model.Config, messages)
	promptTokens := pipe.Tokenizer.Encode(prompt)

	cfg := pipe.Model.Config
	vocabSize := cfg.VocabSize
	kvDim := cfg.NumKVHeads * cfg.HeadDim

	cpuRS := llm.NewRunState(cfg, 8192)
	cpuKV := memory.NewMultiLayerKVCache(cfg.NumLayers, 8192, kvDim)

	eotID := 151643

	type logitEntry struct {
		idx int
		val float32
	}

	for pos, tok := range promptTokens {
		llm.Forward(pipe.Model, int32(tok), pos, cpuKV, cpuRS)

		logits := make([]logitEntry, vocabSize)
		for i := 0; i < vocabSize; i++ {
			logits[i] = logitEntry{i, cpuRS.Logits[i]}
		}
		sort.Slice(logits, func(i, j int) bool { return logits[i].val > logits[j].val })

		eotRank := -1
		eotLogit := float32(0)
		for r, e := range logits {
			if e.idx == eotID {
				eotRank = r
				eotLogit = e.val
				break
			}
		}

		tokText := pipe.Tokenizer.DecodeToken(int32(tok))
		top1Text := pipe.Tokenizer.DecodeToken(int32(logits[0].idx))
		fmt.Fprintf(os.Stderr, "pos=%2d tok=%6d %12q  top1=[%6d]%8.2f %12q  eot_rank=%5d eot_logit=%8.2f  gap=%.2f\n",
			pos, tok, tokText,
			logits[0].idx, logits[0].val, top1Text,
			eotRank, eotLogit,
			logits[0].val-eotLogit)
	}
}
