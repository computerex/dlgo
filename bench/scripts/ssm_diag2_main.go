//go:build ignore

package main

import (
	"fmt"
	"math"

	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	models := []struct {
		name string
		path string
	}{
		{"2B", `C:\projects\gollm\Qwen3.5-2B.Q4_K_M.gguf`},
		{"9B", `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`},
	}

	for _, model := range models {
		fmt.Printf("\n========== %s ==========\n", model.name)
		pipe, err := llm.NewPipeline(model.path, 512)
		if err != nil {
			fmt.Printf("Load fail: %v\n", err)
			continue
		}
		cfg := pipe.Model.Config
		m := pipe.Model

		numHeads := cfg.SSMTimeStepRank
		numKVGroups := cfg.SSMGroupCount
		if numKVGroups <= 0 {
			numKVGroups = numHeads
		}
		headKDim := cfg.SSMStateSize
		headVDim := cfg.SSMInnerSize / numHeads

		fmt.Printf("Config: numHeads=%d numKVGroups=%d headK=%d headV=%d\n",
			numHeads, numKVGroups, headKDim, headVDim)
		fmt.Printf("SSMA[0..3]: %.6f %.6f %.6f %.6f\n",
			m.Layers[0].SSMA[0], m.Layers[0].SSMA[1], m.Layers[0].SSMA[2], m.Layers[0].SSMA[3])
		if m.Layers[0].SSMDtBias != nil {
			fmt.Printf("DtBias[0..3]: %.6f %.6f %.6f %.6f\n",
				m.Layers[0].SSMDtBias[0], m.Layers[0].SSMDtBias[1], m.Layers[0].SSMDtBias[2], m.Layers[0].SSMDtBias[3])
		}
		fmt.Printf("SSMNorm[0..3]: %.6f %.6f %.6f %.6f (len=%d)\n",
			m.Layers[0].SSMNorm[0], m.Layers[0].SSMNorm[1], m.Layers[0].SSMNorm[2], m.Layers[0].SSMNorm[3], len(m.Layers[0].SSMNorm))
		fmt.Printf("AddBOS=%v BOS=%d\n", cfg.AddBOS, cfg.BOS)

		kvDim := cfg.NumKVHeads * cfg.HeadDim
		rs := llm.NewRunState(cfg, 512)
		kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)

		prompt := "The capital of France is"
		tokens := pipe.Tokenizer.Encode(prompt)
		fmt.Printf("Prompt: %q -> tokens: %v\n", prompt, tokens)

		for i, tok := range tokens {
			llm.Forward(m, tok, i, kv, rs)
		}

		top := 0
		for i := 1; i < len(rs.Logits); i++ {
			if rs.Logits[i] > rs.Logits[top] {
				top = i
			}
		}
		fmt.Printf("Top prediction: tok=%d %q logit=%.4f\n", top, pipe.Tokenizer.DecodeToken(int32(top)), rs.Logits[top])

		// Check SSM state health for layer 0
		ssmState := rs.SSMState.Layers[0]
		if ssmState != nil {
			fmt.Printf("\nSSM Layer 0 state health (after %d tokens):\n", len(tokens))
			for h := 0; h < numHeads; h++ {
				sOff := h * headKDim * headVDim
				var stateL2 float64
				var stateMax float32
				for i := 0; i < headKDim*headVDim; i++ {
					v := ssmState.State[sOff+i]
					stateL2 += float64(v) * float64(v)
					if v > stateMax || -v > stateMax {
						if v > 0 {
							stateMax = v
						} else {
							stateMax = -v
						}
					}
				}
				if h < 4 || h >= numHeads-2 {
					fmt.Printf("  Head %2d: L2=%.6f max=%.6f\n", h, math.Sqrt(stateL2), stateMax)
				} else if h == 4 {
					fmt.Println("  ...")
				}
			}
		}

		// Also check a deeper SSM layer
		for _, layerIdx := range []int{6, 12, 24} {
			if layerIdx >= cfg.NumLayers {
				continue
			}
			ssmState2 := rs.SSMState.Layers[layerIdx]
			if ssmState2 != nil {
				var totalL2 float64
				for i := range ssmState2.State {
					v := float64(ssmState2.State[i])
					totalL2 += v * v
				}
				fmt.Printf("  Layer %2d total state L2: %.6f\n", layerIdx, math.Sqrt(totalL2))
			}
		}
	}
}
