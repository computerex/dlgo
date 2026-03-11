//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	path := `C:\projects\gollm\Qwen3.5-35B-A3B-Q3_K_M.gguf`
	fmt.Printf("Loading %s...\n", path)
	t0 := time.Now()
	pipe, err := llm.NewPipeline(path, 512)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Load error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded in %v\n", time.Since(t0))

	cfg := pipe.Model.Config
	m := pipe.Model
	fmt.Printf("Config: arch=%s layers=%d dim=%d heads=%d headDim=%d kvHeads=%d\n",
		cfg.Architecture, cfg.NumLayers, cfg.EmbeddingDim, cfg.NumHeads, cfg.HeadDim, cfg.NumKVHeads)
	fmt.Printf("MoE: experts=%d used=%d expFFNDim=%d sharedFFNDim=%d\n",
		cfg.ExpertCount, cfg.ExpertUsedCount, cfg.ExpertFFNDim, cfg.SharedExpertFFNDim)
	fmt.Printf("SSM: interval=%d innerSize=%d stateSize=%d\n",
		cfg.FullAttentionInterval, cfg.SSMInnerSize, cfg.SSMStateSize)

	// Check layer specs
	for i := 0; i < cfg.NumLayers && i < 5; i++ {
		s := m.Layers[i].Spec
		fmt.Printf("Layer %d: core=%d ffn=%d res=%d gatedQ=%v qkNorm=%v\n",
			i, s.Core, s.FFN, s.Residual, s.GatedQ, s.QKNorm)
		l := &m.Layers[i]
		fmt.Printf("  AttnNorm=%v SSMInProj=%v AttnGate=%v Wq=%v Wo=%v\n",
			l.AttnNorm != nil, l.SSMInProj != nil, l.AttnGate != nil,
			l.Wq != nil, l.Wo != nil)
		fmt.Printf("  FFNRouter=%v FFNGateExps=%v FFNGateShared=%v FFNRouterShared=%v\n",
			l.FFNRouter != nil, l.FFNGateExps != nil,
			l.FFNGateShared != nil, l.FFNRouterShared != nil)
	}
	// Also show an attention layer
	for i := 0; i < cfg.NumLayers; i++ {
		if m.Layers[i].Spec.Core == llm.CoreAttention {
			s := m.Layers[i].Spec
			l := &m.Layers[i]
			fmt.Printf("Layer %d (attn): core=%d ffn=%d res=%d gatedQ=%v qkNorm=%v\n",
				i, s.Core, s.FFN, s.Residual, s.GatedQ, s.QKNorm)
			fmt.Printf("  Wq=%v (rows=%d) Wk=%v Wv=%v Wo=%v\n",
				l.Wq != nil, func() int { if l.Wq != nil { return l.Wq.Rows }; return 0 }(),
				l.Wk != nil, l.Wv != nil, l.Wo != nil)
			fmt.Printf("  AttnQNorm=%v AttnKNorm=%v PostAttnNorm=%v\n",
				l.AttnQNorm != nil, l.AttnKNorm != nil, l.PostAttnNorm != nil)
			break
		}
	}

	prompt := "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
	tokens := pipe.Tokenizer.Encode(prompt)
	fmt.Printf("\nPrompt: %q\n", prompt)
	fmt.Printf("Tokens: %d\n", len(tokens))

	rs := llm.NewRunState(cfg, 512)
	bs := llm.NewBatchState(cfg, 512)
	kv := memory.NewMultiLayerKVCache(cfg.NumLayers, cfg.NumKVHeads*cfg.HeadDim, 512)

	fmt.Println("\nPrefilling...")
	t1 := time.Now()
	llm.ForwardBatch(m, tokens, 0, kv, rs, bs)
	prefillDur := time.Since(t1)
	fmt.Printf("Prefill: %v (%.1f ms/tok)\n", prefillDur, float64(prefillDur.Milliseconds())/float64(len(tokens)))

	fmt.Println("\nGenerating 50 tokens...")
	pos := len(tokens)
	t2 := time.Now()
	for g := 0; g < 50; g++ {
		best := argmax(rs.Logits)
		word := pipe.Tokenizer.Decode([]int32{int32(best)})
		fmt.Print(word)
		if best == int(cfg.EOS) {
			break
		}
		for _, st := range cfg.StopTokens {
			if int32(best) == st {
				fmt.Println()
				goto done
			}
		}
		llm.Forward(m, int32(best), pos, kv, rs)
		pos++
	}
done:
	genDur := time.Since(t2)
	genTokens := pos - len(tokens)
	fmt.Printf("\nGeneration: %v (%.1f tok/s)\n", genDur, float64(genTokens)/genDur.Seconds())
}

func argmax(logits []float32) int {
	best := 0
	bestVal := float32(-math.MaxFloat32)
	for i, v := range logits {
		if v > bestVal {
			bestVal = v
			best = i
		}
	}
	return best
}
