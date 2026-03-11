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
		numH := cfg.SSMTimeStepRank
		numKVG := cfg.SSMGroupCount
		fmt.Printf("  numVHeads=%d numKHeads=%d\n", numH, numKVG)

		for l := 0; l < 4; l++ {
			layer := &pipe.Model.Layers[l]
			isSSM := (l+1)%cfg.FullAttentionInterval != 0
			layerType := "attn"
			if isSSM {
				layerType = "ssm"
			}
			fmt.Printf("  Layer %d (%s):\n", l, layerType)
			if layer.SSMInProj != nil {
				fmt.Printf("    SSMInProj: rows=%d cols=%d\n", layer.SSMInProj.Rows, layer.SSMInProj.Cols)
			}
			if layer.AttnGate != nil {
				fmt.Printf("    AttnGate(Z): rows=%d cols=%d\n", layer.AttnGate.Rows, layer.AttnGate.Cols)
			}
			if layer.SSMAlpha != nil {
				fmt.Printf("    SSMAlpha: rows=%d cols=%d\n", layer.SSMAlpha.Rows, layer.SSMAlpha.Cols)
			}
			if layer.SSMBeta != nil {
				fmt.Printf("    SSMBeta: rows=%d cols=%d\n", layer.SSMBeta.Rows, layer.SSMBeta.Cols)
			}
			if layer.SSMOut != nil {
				fmt.Printf("    SSMOut: rows=%d cols=%d\n", layer.SSMOut.Rows, layer.SSMOut.Cols)
			}
			if layer.SSMConv1dW != nil {
				fmt.Printf("    SSMConv1dW: len=%d\n", len(layer.SSMConv1dW))
			}
			if layer.SSMA != nil {
				fmt.Printf("    SSMA: len=%d\n", len(layer.SSMA))
			}
			if layer.SSMDtBias != nil {
				fmt.Printf("    SSMDtBias: len=%d\n", len(layer.SSMDtBias))
			}
			if layer.SSMNorm != nil {
				fmt.Printf("    SSMNorm: len=%d\n", len(layer.SSMNorm))
			}
			if layer.Wq != nil {
				fmt.Printf("    Wq: rows=%d cols=%d\n", layer.Wq.Rows, layer.Wq.Cols)
			}
			if layer.Wk != nil {
				fmt.Printf("    Wk: rows=%d cols=%d\n", layer.Wk.Rows, layer.Wk.Cols)
			}
			if layer.Wv != nil {
				fmt.Printf("    Wv: rows=%d cols=%d\n", layer.Wv.Rows, layer.Wv.Cols)
			}
			if layer.Wo != nil {
				fmt.Printf("    Wo: rows=%d cols=%d\n", layer.Wo.Rows, layer.Wo.Cols)
			}
		}
		fmt.Println()
	}
}
