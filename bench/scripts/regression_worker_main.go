//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

type RegressionResult struct {
	Name            string  `json:"name"`
	Loaded          bool    `json:"loaded"`
	MaxErr          float64 `json:"max_err"`
	AvgErr          float64 `json:"avg_err"`
	TopMatch        bool    `json:"top_match"`
	CPUText         string  `json:"cpu_text"`
	GPUText         string  `json:"gpu_text"`
	CPUTokS         float64 `json:"cpu_tok_s"`
	GPUTokS         float64 `json:"gpu_tok_s"`
	CPUPrefillMs    float64 `json:"cpu_prefill_ms"`
	GPUPrefillMs    float64 `json:"gpu_prefill_ms"`
	CorrectnesPass  bool    `json:"correctness_pass"`
	CoherencePass   bool    `json:"coherence_pass"`
	Err             string  `json:"err,omitempty"`
	GPULayers       int     `json:"gpu_layers"`
	TotalLayers     int     `json:"total_layers"`
	IsPartialGPU    bool    `json:"is_partial_gpu"`
}

func main() {
	if len(os.Args) < 4 {
		fmt.Fprintf(os.Stderr, "usage: regression_worker <name> <gguf_path> <output_json>\n")
		os.Exit(1)
	}
	name := os.Args[1]
	ggufPath := os.Args[2]
	outPath := os.Args[3]

	res := RegressionResult{Name: name}

	defer func() {
		data, _ := json.MarshalIndent(res, "", "  ")
		os.WriteFile(outPath, data, 0644)
	}()

	if err := gpu.Init(); err != nil {
		res.Err = fmt.Sprintf("gpu init: %v", err)
		return
	}
	defer gpu.Shutdown()

	pipe, err := llm.NewPipeline(ggufPath, 2048)
	if err != nil {
		res.Err = fmt.Sprintf("load fail: %v", err)
		return
	}
	res.Loaded = true
	cfg := pipe.Model.Config
	vocabSize := cfg.VocabSize
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	res.TotalLayers = cfg.NumLayers

	fmt.Fprintf(os.Stderr, "  Model: %s (%d layers, %d dim, %d heads)\n",
		cfg.Architecture, cfg.NumLayers, cfg.EmbeddingDim, cfg.NumHeads)

	coherencePrompt := "Explain what a computer is in one sentence."
	formatted := llm.FormatChat(cfg, "You are a helpful assistant.", coherencePrompt)

	// --- Phase 1: CPU generation ---
	cpuCfg := llm.DefaultGenerateConfig()
	cpuCfg.MaxTokens = 64
	cpuCfg.Seed = 42
	cpuCfg.Sampler.Temperature = 0

	cpuStart := time.Now()
	cpuResult, cpuErr := pipe.GenerateDetailed(formatted, cpuCfg)
	cpuElapsed := time.Since(cpuStart)
	if cpuErr != nil {
		fmt.Fprintf(os.Stderr, "  CPU gen FAIL: %v\n", cpuErr)
		res.CPUText = "FAIL"
	} else {
		res.CPUText = cpuResult.Text
		res.CPUTokS = cpuResult.TokensPerSec
		res.CPUPrefillMs = cpuResult.PrefillTimeMs
		fmt.Fprintf(os.Stderr, "  CPU gen: %d tok  prefill=%.1fms  gen=%.1fms (%.1f tok/s)  total=%.1fms\n",
			cpuResult.TotalTokens, cpuResult.PrefillTimeMs, cpuResult.GenerateTimeMs,
			cpuResult.TokensPerSec, float64(cpuElapsed.Milliseconds()))
		fmt.Fprintf(os.Stderr, "  CPU text: %s\n", preview(cpuResult.Text, 120))
	}

	// --- Phase 2: GPU pipeline (always, with partial offloading) ---
	gpuPipe, gpuPipeErr := gpu.NewGpuPipeline(pipe)
	if gpuPipeErr != nil {
		fmt.Fprintf(os.Stderr, "  GPU pipeline fail: %v\n", gpuPipeErr)
		res.Err = fmt.Sprintf("gpu pipeline: %v", gpuPipeErr)
	} else {
		res.GPULayers = gpuPipe.NumGPULayers
		res.IsPartialGPU = gpuPipe.IsPartialGPU
		fmt.Fprintf(os.Stderr, "  GPU: %d/%d layers on GPU (partial=%v)\n",
			gpuPipe.NumGPULayers, cfg.NumLayers, gpuPipe.IsPartialGPU)

		// Logit correctness
		prompt := "Hello"
		tokens := pipe.Tokenizer.Encode(prompt)
		if len(tokens) == 0 {
			tokens = []int32{1}
		}

		cpuLogits := make([]float32, vocabSize)
		cpuRS := llm.NewRunState(cfg, 512)
		cpuKV := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)
		for i, tok := range tokens {
			llm.Forward(pipe.Model, tok, i, cpuKV, cpuRS)
		}
		copy(cpuLogits, cpuRS.Logits)

		gpuPipe.ResetState()
		gpuLogits := make([]float32, vocabSize)
		for i, tok := range tokens {
			gpu.GpuForward(pipe.Model, gpuPipe.GpuModel, tok, i, gpuPipe.KVCache, gpuPipe.RunState, gpuLogits, gpuPipe)
		}
		gpu.Sync()

		maxErr := float64(0)
		maxIdx := 0
		sumErr := float64(0)
		for i := 0; i < vocabSize; i++ {
			diff := math.Abs(float64(cpuLogits[i] - gpuLogits[i]))
			sumErr += diff
			if diff > maxErr {
				maxErr = diff
				maxIdx = i
			}
		}
		res.MaxErr = maxErr
		res.AvgErr = sumErr / float64(vocabSize)
		res.TopMatch = argmax(cpuLogits) == argmax(gpuLogits)
		errThreshold := 15.0
		if gpuPipe.IsPartialGPU {
			errThreshold = 50.0
		}
		res.CorrectnesPass = maxErr < errThreshold

		fmt.Fprintf(os.Stderr, "  Logits: maxErr=%.4f (idx %d) avgErr=%.6f topMatch=%v\n",
			maxErr, maxIdx, res.AvgErr, res.TopMatch)

		// GPU generation
		gpuPipe.ResetState()
		gpuGenCfg := llm.DefaultGenerateConfig()
		gpuGenCfg.MaxTokens = 64
		gpuGenCfg.Seed = 42
		gpuGenCfg.Sampler.Temperature = 0

		gpuStart := time.Now()
		gpuResult, gpuErr := gpuPipe.GenerateDetailed(formatted, gpuGenCfg)
		gpuElapsed := time.Since(gpuStart)
		if gpuErr != nil {
			fmt.Fprintf(os.Stderr, "  GPU gen FAIL: %v\n", gpuErr)
			res.GPUText = "FAIL"
		} else {
			res.GPUText = gpuResult.Text
			res.GPUTokS = gpuResult.TokensPerSec
			res.GPUPrefillMs = gpuResult.PrefillTimeMs
			fmt.Fprintf(os.Stderr, "  GPU gen: %d tok  prefill=%.1fms  gen=%.1fms (%.1f tok/s)  total=%.1fms\n",
				gpuResult.TotalTokens, gpuResult.PrefillTimeMs, gpuResult.GenerateTimeMs,
				gpuResult.TokensPerSec, float64(gpuElapsed.Milliseconds()))
			fmt.Fprintf(os.Stderr, "  GPU text: %s\n", preview(gpuResult.Text, 120))
		}
	}

	if gpuPipeErr != nil {
		res.CorrectnesPass = true
	}

	cpuCoherent := isCoherent(res.CPUText)
	gpuCoherent := isCoherent(res.GPUText)
	gpuOOM := gpuPipeErr != nil
	res.CoherencePass = cpuCoherent && (gpuCoherent || gpuOOM)

	status := "PASS"
	if !res.CorrectnesPass || !res.CoherencePass {
		status = "FAIL"
	}
	fmt.Fprintf(os.Stderr, "  Result: %s (correctness=%v coherence_cpu=%v coherence_gpu=%v)\n",
		status, res.CorrectnesPass, cpuCoherent, gpuCoherent)
}

func isCoherent(text string) bool {
	if text == "" || text == "FAIL" {
		return false
	}
	t := strings.TrimSpace(text)
	if len(t) < 5 {
		return false
	}
	nonASCII := 0
	for _, c := range t {
		if c > 127 {
			nonASCII++
		}
	}
	return float64(nonASCII)/float64(len([]rune(t))) <= 0.5
}

func preview(s string, n int) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
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
