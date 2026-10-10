//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/llm"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: dump_arch <gguf_path>")
		os.Exit(1)
	}
	pipe, err := llm.NewPipeline(os.Args[1], 512)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load error: %v\n", err)
		os.Exit(1)
	}
	cfg := pipe.Model.Config
	fmt.Printf("Architecture: %s\n", cfg.Architecture)
	fmt.Printf("NumLayers: %d\n", cfg.NumLayers)
	fmt.Printf("EmbeddingDim: %d\n", cfg.EmbeddingDim)
	fmt.Printf("NumHeads: %d\n", cfg.NumHeads)
	fmt.Printf("NumKVHeads: %d\n", cfg.NumKVHeads)
	fmt.Printf("VocabSize: %d\n", cfg.VocabSize)
	fmt.Printf("FFNDim: %d\n", cfg.FFNDim)
	fmt.Printf("HeadDim: %d\n", cfg.HeadDim)
	fmt.Printf("ExpertCount: %d\n", cfg.ExpertCount)
	fmt.Printf("ExpertUsedCount: %d\n", cfg.ExpertUsedCount)
	fmt.Printf("ExpertFFNDim: %d\n", cfg.ExpertFFNDim)
	fmt.Printf("ExpertGatingFunc: %d\n", cfg.ExpertGatingFunc)
	fmt.Printf("ExpertWeightsNorm: %v\n", cfg.ExpertWeightsNorm)
	fmt.Printf("ExpertWeightsScale: %f\n", cfg.ExpertWeightsScale)
	fmt.Printf("RopeFreqBase: %f\n", cfg.RopeFreqBase)
	fmt.Printf("RopeDim: %d\n", cfg.RopeDim)
	fmt.Printf("ChatTemplate: %s\n", cfg.ChatTemplate)
	fmt.Printf("BOS: %d\n", cfg.BOS)
	fmt.Printf("EOS: %d\n", cfg.EOS)
	fmt.Printf("StopTokens: %v\n", cfg.StopTokens)
	fmt.Printf("AddBOS: %v\n", cfg.AddBOS)
	fmt.Printf("ContextLength: %d\n", cfg.ContextLength)
	fmt.Printf("SlidingWindow: %d\n", cfg.SlidingWindow)
	fmt.Printf("SSMInnerSize: %d\n", cfg.SSMInnerSize)
	fmt.Printf("SSMStateSize: %d\n", cfg.SSMStateSize)
	fmt.Printf("SSMTimeStepRank: %d\n", cfg.SSMTimeStepRank)
	fmt.Printf("SSMGroupCount: %d\n", cfg.SSMGroupCount)
	fmt.Printf("SSMConvKernel: %d\n", cfg.SSMConvKernel)
	fmt.Printf("SSMTiledVOrder: %v\n", cfg.SSMTiledVOrder)
	fmt.Printf("FullAttentionInterval: %d\n", cfg.FullAttentionInterval)

	for _, idx := range []int{0, 1, cfg.NumLayers - 1} {
		if idx >= len(pipe.Model.Layers) {
			continue
		}
		l := pipe.Model.Layers[idx]
		fmt.Printf("\nLayer %d spec:\n", idx)
		fmt.Printf("  Core: %d\n", l.Spec.Core)
		fmt.Printf("  FFN: %d\n", l.Spec.FFN)
		fmt.Printf("  GatedQ: %v\n", l.Spec.GatedQ)
		fmt.Printf("  QKNorm: %v\n", l.Spec.QKNorm)
		fmt.Printf("  SlidingWindow: %d\n", l.Spec.SlidingWindow)
		if l.Wq != nil {
			fmt.Printf("  Wq: rows=%d cols=%d type=%d\n", l.Wq.Rows, l.Wq.Cols, l.Wq.Type)
		}
		if l.FFNGateExps != nil {
			fmt.Printf("  FFNGateExps: rows=%d cols=%d type=%d\n", l.FFNGateExps.Rows, l.FFNGateExps.Cols, l.FFNGateExps.Type)
		}
		if l.FFNRouter != nil {
			fmt.Printf("  FFNRouter: rows=%d cols=%d type=%d\n", l.FFNRouter.Rows, l.FFNRouter.Cols, l.FFNRouter.Type)
		}
	}
}
