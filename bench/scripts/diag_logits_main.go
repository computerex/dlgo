//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

type logitEntry struct {
	TokenID int     `json:"id"`
	Value   float32 `json:"val"`
	Text    string  `json:"text"`
}

type stepResult struct {
	Step       int          `json:"step"`
	CPUTop10   []logitEntry `json:"cpu_top10"`
	GPUTop10   []logitEntry `json:"gpu_top10"`
	CPUArgmax  int          `json:"cpu_argmax"`
	GPUArgmax  int          `json:"gpu_argmax"`
	CPUSampled int          `json:"cpu_sampled"`
	GPUSampled int          `json:"gpu_sampled"`
	MaxDiff    float64      `json:"max_diff"`
	MeanDiff   float64      `json:"mean_diff"`
}

func main() {
	modelPath := `C:\models\gemma-3-270m-it-Q8_0.gguf`
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	fmt.Printf("Loading model: %s\n", modelPath)

	// Load for CPU
	cpuPipe, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		fmt.Printf("CPU load error: %v\n", err)
		os.Exit(1)
	}

	cfg := cpuPipe.Model.Config
	prompt := llm.FormatChat(cfg, "You are a helpful assistant.", "Explain what a computer is in exactly two sentences.")
	tokens := cpuPipe.Tokenizer.Encode(prompt)
	fmt.Printf("Arch: %s, Template: %q\n", cfg.Architecture, cfg.ChatTemplate)
	fmt.Printf("Prompt tokens: %d\n", len(tokens))

	// Initialize GPU
	if err := gpu.Init(); err != nil {
		fmt.Printf("GPU init error: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	// Load for GPU (separate pipeline from the same model file)
	gpuCPUPipe, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		fmt.Printf("GPU CPU pipe load error: %v\n", err)
		os.Exit(1)
	}

	gpuPipe, err := gpu.NewGpuPipeline(gpuCPUPipe)
	if err != nil {
		fmt.Printf("GPU pipeline error: %v\n", err)
		os.Exit(1)
	}
	defer gpuPipe.FreeAll()

	// CPU: do prefill
	fmt.Println("\n--- CPU Prefill ---")
	cpuPipe.KVCache.Reset()
	llm.ForwardBatch(cpuPipe.Model, tokens, 0, cpuPipe.KVCache, cpuPipe.RunState, cpuPipe.BatchState)
	cpuLogits := make([]float32, len(cpuPipe.RunState.Logits))
	copy(cpuLogits, cpuPipe.RunState.Logits)

	// GPU: do prefill
	fmt.Println("--- GPU Prefill ---")
	gpuPipe.ResetState()
	for i, tok := range tokens {
		gpu.GpuForward(gpuCPUPipe.Model, gpuPipe.GpuModel, tok, i, gpuPipe.KVCache, gpuPipe.RunState, gpuPipe.LogitsBuf, gpuPipe)
	}
	gpu.Sync()
	gpuLogits := make([]float32, len(gpuPipe.LogitsBuf))
	copy(gpuLogits, gpuPipe.LogitsBuf)

	// Compare prefill logits
	fmt.Println("\n=== PREFILL LOGITS COMPARISON ===")
	compareLogits(cpuLogits, gpuLogits, cpuPipe.Tokenizer, "After Prefill")

	// Check for NaN/Inf in GPU logits
	nanCount, infCount := 0, 0
	for _, v := range gpuLogits {
		if math.IsNaN(float64(v)) {
			nanCount++
		}
		if math.IsInf(float64(v), 0) {
			infCount++
		}
	}
	fmt.Printf("GPU logits: NaN=%d, Inf=%d, total=%d\n", nanCount, infCount, len(gpuLogits))

	// Sample first token from each
	sampler := ops.SamplerConfig{Temperature: 0, RepetitionPenalty: 1.1}
	cpuToken := ops.SampleToken(cpuLogits, sampler, nil, nil)
	gpuToken := ops.SampleToken(gpuLogits, sampler, nil, nil)
	fmt.Printf("\nFirst token: CPU=%d (%q), GPU=%d (%q)\n",
		cpuToken, cpuPipe.Tokenizer.DecodeToken(int32(cpuToken)),
		gpuToken, cpuPipe.Tokenizer.DecodeToken(int32(gpuToken)))

	// Continue for a few steps
	var steps []stepResult
	cpuPos := len(tokens)
	gpuPos := len(tokens)
	cpuRecentTokens := []int32{int32(cpuToken)}
	gpuRecentTokens := []int32{int32(gpuToken)}

	for step := 0; step < 10; step++ {
		cpuTok := int32(cpuToken)
		gpuTok := int32(gpuToken)

		// CPU forward
		llm.Forward(cpuPipe.Model, cpuTok, cpuPos, cpuPipe.KVCache, cpuPipe.RunState)
		cpuPos++
		copy(cpuLogits, cpuPipe.RunState.Logits)

		// GPU forward
		gpu.GpuForward(gpuCPUPipe.Model, gpuPipe.GpuModel, gpuTok, gpuPos-1, gpuPipe.KVCache, gpuPipe.RunState, gpuPipe.LogitsBuf, gpuPipe)
		gpuPos++
		gpu.Sync()
		copy(gpuLogits, gpuPipe.LogitsBuf)

		// Compare
		fmt.Printf("\n=== Step %d (CPU input tok=%d %q, GPU input tok=%d %q) ===\n",
			step+1, cpuTok, cpuPipe.Tokenizer.DecodeToken(cpuTok),
			gpuTok, cpuPipe.Tokenizer.DecodeToken(gpuTok))
		sr := compareLogits(cpuLogits, gpuLogits, cpuPipe.Tokenizer, fmt.Sprintf("Step %d", step+1))

		cpuToken = ops.SampleToken(cpuLogits, sampler, cpuRecentTokens, nil)
		gpuToken = ops.SampleToken(gpuLogits, sampler, gpuRecentTokens, nil)
		cpuRecentTokens = append(cpuRecentTokens, int32(cpuToken))
		gpuRecentTokens = append(gpuRecentTokens, int32(gpuToken))

		sr.CPUSampled = cpuToken
		sr.GPUSampled = gpuToken
		sr.Step = step + 1
		steps = append(steps, sr)

		fmt.Printf("Sampled: CPU=%d (%q), GPU=%d (%q)\n",
			cpuToken, cpuPipe.Tokenizer.DecodeToken(int32(cpuToken)),
			gpuToken, cpuPipe.Tokenizer.DecodeToken(int32(gpuToken)))
	}

	// Write JSON
	data, _ := json.MarshalIndent(steps, "", "  ")
	os.WriteFile("diag_logits.json", data, 0644)
	fmt.Println("\nWrote diag_logits.json")
}

func compareLogits(cpu, gpu []float32, tok *llm.Tokenizer, label string) stepResult {
	n := len(cpu)
	if len(gpu) < n {
		n = len(gpu)
	}

	var maxDiff float64
	var sumDiff float64
	maxDiffIdx := 0
	for i := 0; i < n; i++ {
		d := math.Abs(float64(cpu[i] - gpu[i]))
		sumDiff += d
		if d > maxDiff {
			maxDiff = d
			maxDiffIdx = i
		}
	}
	meanDiff := sumDiff / float64(n)

	fmt.Printf("[%s] vocab=%d, maxDiff=%.6f (at token %d), meanDiff=%.6f\n",
		label, n, maxDiff, maxDiffIdx, meanDiff)

	// Top 10 CPU
	type idxVal struct {
		idx int
		val float32
	}
	cpuSorted := make([]idxVal, n)
	gpuSorted := make([]idxVal, n)
	for i := 0; i < n; i++ {
		cpuSorted[i] = idxVal{i, cpu[i]}
		gpuSorted[i] = idxVal{i, gpu[i]}
	}
	sort.Slice(cpuSorted, func(i, j int) bool { return cpuSorted[i].val > cpuSorted[j].val })
	sort.Slice(gpuSorted, func(i, j int) bool { return gpuSorted[i].val > gpuSorted[j].val })

	sr := stepResult{
		MaxDiff:  maxDiff,
		MeanDiff: meanDiff,
	}

	fmt.Printf("  CPU Top-10: ")
	for i := 0; i < 10 && i < len(cpuSorted); i++ {
		e := cpuSorted[i]
		text := tok.DecodeToken(int32(e.idx))
		fmt.Printf("%d(%q)=%.2f ", e.idx, text, e.val)
		sr.CPUTop10 = append(sr.CPUTop10, logitEntry{e.idx, e.val, text})
	}
	fmt.Println()

	fmt.Printf("  GPU Top-10: ")
	for i := 0; i < 10 && i < len(gpuSorted); i++ {
		e := gpuSorted[i]
		text := tok.DecodeToken(int32(e.idx))
		fmt.Printf("%d(%q)=%.2f ", e.idx, text, e.val)
		sr.GPUTop10 = append(sr.GPUTop10, logitEntry{e.idx, e.val, text})
	}
	fmt.Println()

	sr.CPUArgmax = cpuSorted[0].idx
	sr.GPUArgmax = gpuSorted[0].idx

	return sr
}
