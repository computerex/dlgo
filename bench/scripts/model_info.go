//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := os.Args[1]

	pipe, err := llm.NewPipeline(modelPath, 8192)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	cfg := pipe.Model.Config
	fmt.Fprintf(os.Stderr, "Architecture: %s\n", cfg.Architecture)
	fmt.Fprintf(os.Stderr, "EmbeddingDim=%d NumHeads=%d NumKVHeads=%d HeadDim=%d\n", cfg.EmbeddingDim, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
	fmt.Fprintf(os.Stderr, "NumLayers=%d VocabSize=%d\n", cfg.NumLayers, cfg.VocabSize)
	fmt.Fprintf(os.Stderr, "RopeFreqBase=%.1f RopeDim=%d RopeNeox=%v\n", cfg.RopeFreqBase, cfg.RopeDim, cfg.RopeNeox)
	fmt.Fprintf(os.Stderr, "RMSNormEps=%e\n", cfg.RMSNormEps)
	fmt.Fprintf(os.Stderr, "ExpertCount=%d ExpertUsedCount=%d ExpertFFNDim=%d SharedExpertFFNDim=%d\n",
		cfg.ExpertCount, cfg.ExpertUsedCount, cfg.ExpertFFNDim, cfg.SharedExpertFFNDim)
	fmt.Fprintf(os.Stderr, "ExpertWeightsNorm: %v\n", cfg.ExpertWeightsNorm)
	fmt.Fprintf(os.Stderr, "ExpertGatingFunc: %d\n", cfg.ExpertGatingFunc)
	fmt.Fprintf(os.Stderr, "ExpertWeightsScale: %f\n", cfg.ExpertWeightsScale)
	fmt.Fprintf(os.Stderr, "AddBOS=%v BOS=%d EOS=%d\n", cfg.AddBOS, cfg.BOS, cfg.EOS)
	fmt.Fprintf(os.Stderr, "WeightTied=%v\n", pipe.Model.Output == nil)

	for i := 0; i < 3 && i < len(pipe.Model.Layers); i++ {
		l := &pipe.Model.Layers[i]
		fmt.Fprintf(os.Stderr, "\nLayer %d:\n", i)
		fmt.Fprintf(os.Stderr, "  Bo (attn_output.bias): %v\n", l.Bo != nil)
		fmt.Fprintf(os.Stderr, "  Bq: %v, Bk: %v, Bv: %v\n", l.Bq != nil, l.Bk != nil, l.Bv != nil)
		fmt.Fprintf(os.Stderr, "  AttnQNorm: %v, AttnKNorm: %v\n", l.AttnQNorm != nil, l.AttnKNorm != nil)
		fmt.Fprintf(os.Stderr, "  FFNRouter: %v (type %d)\n", l.FFNRouter != nil, func() int { if l.FFNRouter != nil { return int(l.FFNRouter.Type) }; return -1 }())
		fmt.Fprintf(os.Stderr, "  FFNRouterBias: %v\n", l.FFNRouterBias != nil)
		fmt.Fprintf(os.Stderr, "  Spec: Core=%d FFN=%d Residual=%d QKNorm=%v\n",
			l.Spec.Core, l.Spec.FFN, l.Spec.Residual, l.Spec.QKNorm)
	}

	fmt.Fprintf(os.Stderr, "\nOutputBias: %v\n", pipe.Model.OutputBias != nil)
	fmt.Fprintf(os.Stderr, "OutputNorm: %v\n", pipe.Model.OutputNorm != nil)
}
