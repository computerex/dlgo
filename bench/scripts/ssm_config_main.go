//go:build ignore

package main

import (
	"fmt"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	models := []struct{ name, path string }{
		{"Qwen3.5-2B", `C:\projects\gollm\Qwen3.5-2B.Q4_K_M.gguf`},
		{"Qwen3.5-9B", `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`},
	}
	for _, m := range models {
		fmt.Printf("=== %s ===\n", m.name)
		pipe, err := llm.NewPipeline(m.path, 64)
		if err != nil {
			fmt.Printf("  Load fail: %v\n", err)
			continue
		}
		cfg := pipe.Model.Config
		fmt.Printf("  dim=%d layers=%d attnHeads=%d attnKVHeads=%d headDim=%d\n",
			cfg.EmbeddingDim, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
		fmt.Printf("  SSM: inner=%d state=%d dtRank(numVHeads)=%d convK=%d groupCount(numKHeads)=%d\n",
			cfg.SSMInnerSize, cfg.SSMStateSize, cfg.SSMTimeStepRank, cfg.SSMConvKernel, cfg.SSMGroupCount)
		numH := cfg.SSMTimeStepRank
		numKVG := cfg.SSMGroupCount
		if numKVG <= 0 {
			numKVG = numH
		}
		headK := cfg.SSMStateSize
		headV := cfg.SSMInnerSize / numH
		fmt.Printf("  Computed: numVHeads=%d numKHeads=%d headK=%d headV=%d headsPerGroup=%d\n",
			numH, numKVG, headK, headV, numH/numKVG)
		fmt.Printf("  Grouped SSM: %v\n", numKVG != numH)
		fmt.Println()
	}
}
