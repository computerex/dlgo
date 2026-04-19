//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

type modelSpec struct {
	name, path string
}

var models = []modelSpec{
	{"SmolLM2 360M Q8_0", `C:\models\smollm2-360m-instruct-q8_0.gguf`},
	{"Llama 3.2 1B Q4_K_M", `C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf`},
	{"Qwen3.5 2B Q4_K_M", `C:\models\Qwen3.5-2B.Q4_K_M.gguf`},
	{"Qwen3.5 9B Q3_K_M", `C:\models\Qwen3.5-9B-Q3_K_M.gguf`},
	{"Qwen3.6 35B-A3B IQ3_XXS", `C:\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`},
}

const testPrompt = "Explain what a computer is in exactly two sentences."
const numTokensGen = 30

func main() {
	fmt.Println("═══ dlgo Profiled Benchmark ═══")

	for _, m := range models {
		if _, err := os.Stat(m.path); os.IsNotExist(err) {
			fmt.Printf("%-30s SKIP (not found)\n", m.name)
			continue
		}

		benchModel(m.name, m.path)
		fmt.Println()
	}
}

func benchModel(name, path string) {
	if err := gpu.Init(); err != nil {
		fmt.Printf("%-30s  gpu init: %v\n", name, err)
		return
	}
	defer gpu.Shutdown()

	pipe, err := llm.NewPipeline(path, 2048)
	if err != nil {
		fmt.Printf("%-30s  load: %v\n", name, err)
		return
	}

	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Printf("%-30s  gpu: %v\n", name, err)
		return
	}
	defer gpuPipe.FreeAll()

	cfg := pipe.Model.Config
	chatPrompt := llm.FormatChat(cfg, "You are a helpful assistant.", testPrompt)

	genCfg := llm.DefaultGenerateConfig()
	genCfg.MaxTokens = numTokensGen
	genCfg.Seed = 42
	genCfg.Sampler.Temperature = 0

	// Warmup
	gpu.PerfReset()
	warmCfg := genCfg
	warmCfg.MaxTokens = 3
	gpuPipe.GenerateDetailed(chatPrompt, warmCfg)

	// Profile run
	gpu.PerfReset()
	start := time.Now()
	result, err := gpuPipe.GenerateDetailed(chatPrompt, genCfg)
	elapsed := time.Since(start)

	if err != nil {
		fmt.Printf("%-30s  gen: %v\n", name, err)
		return
	}

	gpuUs := gpu.PerfGetGpuUs()
	dispatches := gpu.PerfGetDispatches()
	barriers := gpu.PerfGetBarriers()

	genTokens := result.TotalTokens - result.PromptTokens
	if genTokens <= 0 {
		genTokens = 1
	}

	totalMs := elapsed.Seconds() * 1000
	gpuMs := gpuUs / 1000.0
	cpuMs := totalMs - gpuMs
	perTokenGpuMs := gpuMs / float64(genTokens)
	perTokenWallMs := 1000.0 / result.TokensPerSec

	layers := cfg.NumLayers
	barriersPerToken := float64(barriers) / float64(genTokens)
	dispatchesPerToken := float64(dispatches) / float64(genTokens)
	barriersPerLayer := barriersPerToken / float64(layers)
	dispatchesPerLayer := dispatchesPerToken / float64(layers)

	dimStr := fmt.Sprintf("dim=%d, layers=%d, heads=%d", cfg.EmbeddingDim, layers, cfg.NumHeads)

	// Weight data per layer (approximate)
	bytesPerElement := 0.5 // Q4_K ~ 4.5 bits, approximate
	switch {
	case strings.Contains(name, "Q8_0"):
		bytesPerElement = 1.0
	case strings.Contains(name, "Q3_K"):
		bytesPerElement = 0.44
	case strings.Contains(name, "IQ3"):
		bytesPerElement = 0.44
	}
	dim := float64(cfg.EmbeddingDim)
	ffnDim := float64(cfg.FFNDim)
	if ffnDim == 0 {
		ffnDim = dim * 4
	}
	weightBytesPerLayer := (dim*dim*4 + dim*ffnDim*3) * bytesPerElement
	totalWeightBytes := weightBytesPerLayer * float64(layers)
	bwGBs := totalWeightBytes / (perTokenGpuMs / 1000.0) / 1e9
	maxBW := 672.0 // RTX 4070 Ti Super

	fmt.Printf("─── %s (%s) ───\n", name, dimStr)
	fmt.Printf("  Speed:       %.1f tok/s (%.2f ms/tok wall, %.2f ms/tok GPU)\n",
		result.TokensPerSec, perTokenWallMs, perTokenGpuMs)
	fmt.Printf("  Prefill:     %.0f tok/s (%.0f prompt tokens in %.1f ms)\n",
		float64(result.PromptTokens)/(result.PrefillTimeMs/1000.0), float64(result.PromptTokens), result.PrefillTimeMs)
	fmt.Printf("  Overhead:    %.2f ms CPU overhead per token (%.0f%% of wall time)\n",
		cpuMs/float64(genTokens), cpuMs/totalMs*100)
	fmt.Printf("  Per token:   %.0f dispatches (%.1f/layer), %.0f barriers (%.1f/layer)\n",
		dispatchesPerToken, dispatchesPerLayer, barriersPerToken, barriersPerLayer)
	fmt.Printf("  Bandwidth:   %.0f GB/s effective (%.0f%% of %.0f GB/s peak)\n",
		bwGBs, bwGBs/maxBW*100, maxBW)
	fmt.Printf("  Barrier cost est: %.2f ms/tok (assuming %.0f us/barrier)\n",
		barriersPerToken*3.0/1000.0, 3.0)
	fmt.Printf("  Text: %.50s...\n", result.Text)

	// Theoretical minimum
	theoMs := totalWeightBytes / 672e6 // GB/s → bytes/ms
	theoTokS := 1000.0 / theoMs
	fmt.Printf("  Theoretical: %.1f tok/s (%.2f ms/tok at peak BW), actual = %.1fx\n",
		theoTokS, theoMs, result.TokensPerSec/theoTokS)

	// Estimate barrier cost from gap
	gap := perTokenWallMs - theoMs
	estBarrierUs := gap * 1000.0 / barriersPerToken
	_ = math.Abs(estBarrierUs)
	fmt.Printf("  Est barrier: %.1f us/barrier (from wall-theoretical gap)\n", estBarrierUs)
}
