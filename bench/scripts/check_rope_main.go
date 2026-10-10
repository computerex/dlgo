package main

import (
	"fmt"
	"math"
	"os"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := `C:\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`

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

	cfg := pipe.Model.Config
	fmt.Printf("SSM Heads: %d, StateSize: %d, InnerSize: %d, GroupCount: %d\n",
		cfg.SSMTimeStepRank, cfg.SSMStateSize, cfg.SSMInnerSize, cfg.SSMGroupCount)

	for l := 0; l < 5; l++ {
		layer := &pipe.Model.Layers[l]
		if layer.SSMA == nil {
			fmt.Printf("Layer %d: no SSMA (attention layer)\n", l)
			continue
		}
		fmt.Printf("Layer %d SSMA values: [", l)
		minA, maxA := float32(math.MaxFloat32), float32(-math.MaxFloat32)
		for i, v := range layer.SSMA {
			if i < 5 {
				fmt.Printf("%.4f ", v)
			}
			if v < minA {
				minA = v
			}
			if v > maxA {
				maxA = v
			}
		}
		fmt.Printf("...] min=%.4f max=%.4f len=%d\n", minA, maxA, len(layer.SSMA))

		if layer.SSMDtBias != nil {
			fmt.Printf("Layer %d DtBias: [", l)
			for i, v := range layer.SSMDtBias {
				if i < 5 {
					fmt.Printf("%.4f ", v)
				}
			}
			fmt.Printf("...] len=%d\n", len(layer.SSMDtBias))
		}
	}
}
