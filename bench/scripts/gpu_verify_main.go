//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	models := []struct {
		name, path string
	}{
		{"SmolLM2 360M Q8_0", `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`},
		{"TinyLlama 1.1B Q4_0", `C:\projects\evoke\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`},
		{"Qwen 2.5 0.5B Q4_K_M", `C:\projects\evoke\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`},
		{"Gemma 3 1B Q4_K_M", `C:\projects\evoke\models\gemma-3-1b-it-Q4_K_M.gguf`},
	}

	if err := gpu.Init(); err != nil {
		fmt.Printf("GPU init failed: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	for _, m := range models {
		fmt.Printf("\n═══ %s ═══\n", m.name)
		pipe, err := llm.NewPipeline(m.path, 512)
		if err != nil {
			fmt.Printf("  SKIP: %v\n", err)
			continue
		}

		cfg := pipe.Model.Config
		dim := cfg.EmbeddingDim
		vocabSize := cfg.VocabSize

		gpuModel, err := gpu.UploadModel(pipe.Model)
		if err != nil {
			fmt.Printf("  GPU upload fail: %v\n", err)
			continue
		}

		qDim := cfg.NumHeads * cfg.HeadDim
		kvDim := cfg.NumKVHeads * cfg.HeadDim
		ffnDim := cfg.FFNDim

		rs := gpu.NewGpuRunState(dim, qDim, kvDim, ffnDim, vocabSize)
		kv := gpu.NewGpuKVCache(cfg.NumLayers, 512, kvDim)

		prompt := "Hello"
		tokens := pipe.Tokenizer.Encode(prompt)
		if len(tokens) == 0 {
			tokens = []int32{1}
		}

		cpuLogits := make([]float32, vocabSize)
		gpuLogits := make([]float32, vocabSize)

		// CPU forward
		cpuRS := llm.NewRunState(cfg, 512)
		cpuKV := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)
		for i, tok := range tokens {
			llm.Forward(pipe.Model, tok, i, cpuKV, cpuRS)
		}
		copy(cpuLogits, cpuRS.Logits)

		// GPU forward
		for i, tok := range tokens {
			gpu.GpuForward(pipe.Model, gpuModel, tok, i, kv, rs, gpuLogits)
		}

		// Compare
		maxErr := float32(0)
		maxIdx := 0
		sumErr := float64(0)
		for i := 0; i < vocabSize; i++ {
			diff := float32(math.Abs(float64(cpuLogits[i] - gpuLogits[i])))
			sumErr += float64(diff)
			if diff > maxErr {
				maxErr = diff
				maxIdx = i
			}
		}
		avgErr := sumErr / float64(vocabSize)

		cpuTop := argmax(cpuLogits)
		gpuTop := argmax(gpuLogits)

		fmt.Printf("  Max logit error: %.4f at idx %d (cpu=%.4f gpu=%.4f)\n",
			maxErr, maxIdx, cpuLogits[maxIdx], gpuLogits[maxIdx])
		fmt.Printf("  Avg logit error: %.6f\n", avgErr)
		fmt.Printf("  CPU top token: %d (%.4f)  GPU top token: %d (%.4f)  Match: %v\n",
			cpuTop, cpuLogits[cpuTop], gpuTop, gpuLogits[gpuTop], cpuTop == gpuTop)

		// Show top-5 for both
		fmt.Printf("  CPU top-5: ")
		for _, idx := range topK(cpuLogits, 5) {
			tok := pipe.Tokenizer.DecodeToken(int32(idx))
			fmt.Printf("%d(%q=%.2f) ", idx, tok, cpuLogits[idx])
		}
		fmt.Printf("\n  GPU top-5: ")
		for _, idx := range topK(gpuLogits, 5) {
			tok := pipe.Tokenizer.DecodeToken(int32(idx))
			fmt.Printf("%d(%q=%.2f) ", idx, tok, gpuLogits[idx])
		}
		fmt.Println()
	}
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

func topK(x []float32, k int) []int {
	indices := make([]int, k)
	vals := make([]float32, k)
	for i := range vals {
		vals[i] = -1e30
	}
	for i, v := range x {
		if v > vals[k-1] {
			vals[k-1] = v
			indices[k-1] = i
			for j := k - 2; j >= 0; j-- {
				if vals[j+1] > vals[j] {
					vals[j], vals[j+1] = vals[j+1], vals[j]
					indices[j], indices[j+1] = indices[j+1], indices[j]
				}
			}
		}
	}
	return indices
}
