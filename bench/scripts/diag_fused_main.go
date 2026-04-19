//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	modelPath := `C:\models\gemma-3-270m-it-Q8_0.gguf`
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	fmt.Printf("Loading model: %s\n", modelPath)

	// Load CPU pipeline
	cpuPipe, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		fmt.Printf("CPU load error: %v\n", err)
		os.Exit(1)
	}

	cfg := cpuPipe.Model.Config
	prompt := llm.FormatChat(cfg, "You are a helpful assistant.", "Explain what a computer is in exactly two sentences.")
	tokens := cpuPipe.Tokenizer.Encode(prompt)
	fmt.Printf("Arch: %s, Template: %q, EmbedScale: %.2f\n", cfg.Architecture, cfg.ChatTemplate, cfg.EmbedScale)
	fmt.Printf("FinalLogitSoftcap: %.2f, AttnLogitSoftcap: %.2f\n", cfg.FinalLogitSoftcap, cfg.AttnLogitSoftcap)
	fmt.Printf("Prompt tokens: %d\n", len(tokens))
	fmt.Printf("FFNGelu: %v\n", cfg.FFNGelu)

	// Init GPU
	if err := gpu.Init(); err != nil {
		fmt.Printf("GPU init error: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	// Load GPU pipeline (separate model instance)
	gpuCPUPipe, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		fmt.Printf("GPU model load error: %v\n", err)
		os.Exit(1)
	}

	gpuPipe, err := gpu.NewGpuPipeline(gpuCPUPipe)
	if err != nil {
		fmt.Printf("GPU pipeline error: %v\n", err)
		os.Exit(1)
	}
	defer gpuPipe.FreeAll()

	fmt.Printf("UseFusedForward: %v, HasMoE: %v, HasSSM: %v\n",
		gpuPipe.UseFusedForward, gpuPipe.HasMoE, gpuPipe.HasSSM)

	// CPU prefill
	fmt.Println("\n--- CPU Prefill (batch) ---")
	cpuPipe.KVCache.Reset()
	llm.ForwardBatch(cpuPipe.Model, tokens, 0, cpuPipe.KVCache, cpuPipe.RunState, cpuPipe.BatchState)
	cpuLogits := make([]float32, len(cpuPipe.RunState.Logits))
	copy(cpuLogits, cpuPipe.RunState.Logits)

	// GPU prefill using the ACTUAL path: GpuForwardFusedSSM per-token
	fmt.Println("--- GPU Prefill (fused per-token, as used by GenerateDetailed for Gemma) ---")
	gpuPipe.ResetState()

	layerConfs := gpuPipe.LayerConfs
	if layerConfs == nil {
		layerConfs = gpu.BuildLayerConfs(gpuCPUPipe.Model, gpuPipe.GpuModel, gpuPipe, gpuPipe.RunState, gpuPipe.KVCache)
		gpuPipe.LayerConfs = layerConfs
	}

	gpu.BeginBatch()
	for i, tok := range tokens {
		gpu.GpuForwardFusedSSM(gpuCPUPipe.Model, gpuPipe.GpuModel, tok, i, gpuPipe.KVCache, gpuPipe.RunState, gpuPipe.LogitsBuf, layerConfs, gpuPipe)
	}
	gpu.Sync()
	gpuLogits := make([]float32, len(gpuPipe.LogitsBuf))
	copy(gpuLogits, gpuPipe.LogitsBuf)

	fmt.Println("\n=== PREFILL LOGITS COMPARISON (fused path) ===")
	compareLogits(cpuLogits, gpuLogits, cpuPipe.Tokenizer, "After Prefill")

	// Sample first token
	sampler := ops.SamplerConfig{Temperature: 0, RepetitionPenalty: 1.1}
	cpuToken := ops.SampleToken(cpuLogits, sampler, nil, nil)
	gpuToken := ops.SampleToken(gpuLogits, sampler, nil, nil)
	fmt.Printf("\nFirst token: CPU=%d (%q), GPU=%d (%q)\n",
		cpuToken, cpuPipe.Tokenizer.DecodeToken(int32(cpuToken)),
		gpuToken, cpuPipe.Tokenizer.DecodeToken(int32(gpuToken)))

	// Decode steps using the actual fused path
	cpuPos := len(tokens)
	gpuPos := len(tokens)
	cpuRecentTokens := []int32{int32(cpuToken)}
	gpuRecentTokens := []int32{int32(gpuToken)}

	for step := 0; step < 15; step++ {
		cpuTok := int32(cpuToken)
		gpuTok := int32(gpuToken)

		// CPU forward
		llm.Forward(cpuPipe.Model, cpuTok, cpuPos, cpuPipe.KVCache, cpuPipe.RunState)
		cpuPos++
		copy(cpuLogits, cpuPipe.RunState.Logits)

		// GPU forward using fused path
		gpu.GpuForwardFusedSSM(gpuCPUPipe.Model, gpuPipe.GpuModel, gpuTok, gpuPos-1, gpuPipe.KVCache, gpuPipe.RunState, gpuPipe.LogitsBuf, layerConfs, gpuPipe)
		gpuPos++
		gpu.Sync()
		copy(gpuLogits, gpuPipe.LogitsBuf)

		fmt.Printf("\n=== Step %d (CPU tok=%d %q, GPU tok=%d %q) ===\n",
			step+1, cpuTok, cpuPipe.Tokenizer.DecodeToken(cpuTok),
			gpuTok, cpuPipe.Tokenizer.DecodeToken(gpuTok))
		compareLogits(cpuLogits, gpuLogits, cpuPipe.Tokenizer, fmt.Sprintf("Step %d", step+1))

		cpuToken = ops.SampleToken(cpuLogits, sampler, cpuRecentTokens, nil)
		gpuToken = ops.SampleToken(gpuLogits, sampler, gpuRecentTokens, nil)
		cpuRecentTokens = append(cpuRecentTokens, int32(cpuToken))
		gpuRecentTokens = append(gpuRecentTokens, int32(gpuToken))

		fmt.Printf("Sampled: CPU=%d (%q), GPU=%d (%q)\n",
			cpuToken, cpuPipe.Tokenizer.DecodeToken(int32(cpuToken)),
			gpuToken, cpuPipe.Tokenizer.DecodeToken(int32(gpuToken)))

		if int32(cpuToken) == cfg.EOS && int32(gpuToken) == cfg.EOS {
			fmt.Println("Both hit EOS")
			break
		}
	}
}

func compareLogits(cpu, gpu []float32, tok *llm.Tokenizer, label string) {
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

	fmt.Printf("[%s] maxDiff=%.4f (tok %d), meanDiff=%.4f\n",
		label, maxDiff, maxDiffIdx, meanDiff)

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

	fmt.Printf("  CPU Top-5: ")
	for i := 0; i < 5 && i < len(cpuSorted); i++ {
		e := cpuSorted[i]
		fmt.Printf("%d(%q)=%.2f ", e.idx, tok.DecodeToken(int32(e.idx)), e.val)
	}
	fmt.Println()
	fmt.Printf("  GPU Top-5: ")
	for i := 0; i < 5 && i < len(gpuSorted); i++ {
		e := gpuSorted[i]
		fmt.Printf("%d(%q)=%.2f ", e.idx, tok.DecodeToken(int32(e.idx)), e.val)
	}
	fmt.Println()
}

// Make exported functions accessible
var _ = sort.Strings
