//go:build ignore

package main

import (
	"fmt"
	"os"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"

)

func main() {
	models := []struct {
		name, path string
	}{
		{"Qwen3.5-9B-Q3_K_M", `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`},
	}

	if err := gpu.Init(); err != nil {
		fmt.Printf("GPU init failed: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()
	fmt.Printf("GPU: %s (%.0f MB VRAM)\n\n", gpu.DeviceName(), float64(gpu.VRAMBytes())/(1024*1024))

	for _, m := range models {
		fmt.Printf("═══ %s ═══\n", m.name)
		testModel(m.name, m.path)
		fmt.Println()
	}
}

func testModel(name, path string) {
	pipe, err := llm.NewPipeline(path, 512)
	if err != nil {
		fmt.Printf("  SKIP: load fail: %v\n", err)
		return
	}

	cfg := pipe.Model.Config
	fmt.Printf("  Config: dim=%d, layers=%d, heads=%d, kvHeads=%d, vocab=%d\n",
		cfg.EmbeddingDim, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.VocabSize)
	fmt.Printf("  SSM: innerSize=%d, stateSize=%d, fullAttnInterval=%d, timeStepRank=%d, convK=%d\n",
		cfg.SSMInnerSize, cfg.SSMStateSize, cfg.FullAttentionInterval, cfg.SSMTimeStepRank, cfg.SSMConvKernel)
	if cfg.SSMTimeStepRank > 0 {
		numH := cfg.SSMTimeStepRank
		headV := cfg.SSMInnerSize / numH
		headK := cfg.SSMStateSize
		keyD := numH * headK
		valD := numH * headV
		qkvD := keyD*2 + valD
		fmt.Printf("  SSM dims: numHeads=%d, headV=%d, headK=%d, keyDim=%d, valDim=%d, qkvDim=%d\n",
			numH, headV, headK, keyD, valD, qkvD)
		if len(pipe.Model.Layers) > 0 {
			l0 := &pipe.Model.Layers[0]
			if l0.SSMConv1dW != nil {
				fmt.Printf("  SSM conv1d weight len=%d (expected qkvDim*convK=%d)\n",
					len(l0.SSMConv1dW), qkvD*cfg.SSMConvKernel)
			}
			if l0.SSMInProj != nil {
				fmt.Printf("  SSM InProj: rows=%d, cols=%d (actual qkvDim=%d)\n",
					l0.SSMInProj.Rows, l0.SSMInProj.Cols, l0.SSMInProj.Rows)
			}
			if l0.AttnGate != nil {
				fmt.Printf("  SSM Gate: rows=%d, cols=%d\n", l0.AttnGate.Rows, l0.AttnGate.Cols)
			}
			if l0.SSMAlpha != nil {
				fmt.Printf("  SSM Alpha: rows=%d, cols=%d\n", l0.SSMAlpha.Rows, l0.SSMAlpha.Cols)
			}
			if l0.SSMOut != nil {
				fmt.Printf("  SSM Out: rows=%d, cols=%d\n", l0.SSMOut.Rows, l0.SSMOut.Cols)
			}
			if l0.SSMA != nil {
				fmt.Printf("  SSM A (decay): len=%d\n", len(l0.SSMA))
			}
		}
		fmt.Printf("  SSMGroupCount=%d\n", cfg.SSMGroupCount)
	}

	vocabSize := cfg.VocabSize
	kvDim := cfg.NumKVHeads * cfg.HeadDim

	tokens := pipe.Tokenizer.Encode("Hello")
	if len(tokens) == 0 {
		tokens = []int32{1}
	}

	fmt.Printf("  [Correctness] CPU logits...\n")
	cpuLogits := make([]float32, vocabSize)
	cpuRS := llm.NewRunState(cfg, 512)
	cpuKV := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)
	for i, tok := range tokens {
		llm.Forward(pipe.Model, tok, i, cpuKV, cpuRS)
	}
	copy(cpuLogits, cpuRS.Logits)

	cpuTop := argmax(cpuLogits)
	fmt.Printf("  CPU top token: %d (logit=%.4f)\n", cpuTop, cpuLogits[cpuTop])

	fmt.Printf("  [CPU Generation]...\n")

	prompt := llm.FormatChat(cfg, "You are a helpful assistant.", "Explain what a computer is in one sentence.")
	genCfg := llm.DefaultGenerateConfig()
	genCfg.MaxTokens = 64
	genCfg.Seed = 42
	genCfg.Sampler.Temperature = 0

	start := time.Now()
	result, err := pipe.GenerateDetailed(prompt, genCfg)
	elapsed := time.Since(start)
	if err != nil {
		fmt.Printf("  CPU Gen fail: %v\n", err)
		return
	}
	fmt.Printf("  CPU: %d tok in %.1fms (%.1f tok/s)\n",
		result.TotalTokens, float64(elapsed.Milliseconds()), result.TokensPerSec)
	fmt.Printf("  CPU → %s\n", truncate(result.Text, 200))

	fmt.Printf("  [GPU Generation]...\n")
	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Printf("  GPU pipeline fail: %v\n", err)
		return
	}
	defer gpuPipe.FreeAll()

	start = time.Now()
	gpuResult, gpuErr := gpuPipe.GenerateDetailed(prompt, genCfg)
	elapsed = time.Since(start)
	if gpuErr != nil {
		fmt.Printf("  GPU Gen fail: %v\n", gpuErr)
		return
	}
	fmt.Printf("  GPU: %d tok in %.1fms (%.1f tok/s)\n",
		gpuResult.TotalTokens, float64(elapsed.Milliseconds()), gpuResult.TokensPerSec)
	fmt.Printf("  GPU → %s\n", truncate(gpuResult.Text, 200))
}

func argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}
