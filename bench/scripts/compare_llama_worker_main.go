//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

type CompareResult struct {
	Name           string  `json:"name"`
	Text           string  `json:"text"`
	TokS           float64 `json:"tok_s"`
	PrefillMs      float64 `json:"prefill_ms"`
	GenerateMs     float64 `json:"generate_ms"`
	TotalTokens    int     `json:"total_tokens"`
	PromptTokens   int     `json:"prompt_tokens"`
	GPULayers      int     `json:"gpu_layers"`
	Dp4a           bool    `json:"dp4a"`
	Err            string  `json:"err,omitempty"`
}

func main() {
	if len(os.Args) < 5 {
		fmt.Fprintf(os.Stderr, "usage: compare_worker <name> <gguf_path> <prompt> <output_json> [max_tokens] [context]\n")
		os.Exit(1)
	}
	name := os.Args[1]
	ggufPath := os.Args[2]
	prompt := os.Args[3]
	outPath := os.Args[4]

	maxTokens := 150
	ctxLen := 4096
	if len(os.Args) > 5 {
		if v, err := strconv.Atoi(os.Args[5]); err == nil { maxTokens = v }
	}
	if len(os.Args) > 6 {
		if v, err := strconv.Atoi(os.Args[6]); err == nil { ctxLen = v }
	}

	res := CompareResult{Name: name}
	defer func() {
		data, _ := json.MarshalIndent(res, "", "  ")
		os.WriteFile(outPath, data, 0644)
	}()

	if err := gpu.Init(); err != nil {
		res.Err = fmt.Sprintf("gpu init: %v", err)
		return
	}
	defer gpu.Shutdown()

	pipe, err := llm.NewPipeline(ggufPath, ctxLen)
	if err != nil {
		res.Err = fmt.Sprintf("load: %v", err)
		return
	}

	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		res.Err = fmt.Sprintf("gpu: %v", err)
		return
	}
	defer gpuPipe.FreeAll()

	res.GPULayers = gpuPipe.NumGPULayers
	res.Dp4a = gpu.HasDp4a()

	cfg := pipe.Model.Config
	chatPrompt := llm.FormatChat(cfg, "You are a helpful assistant.", prompt)

	genCfg := llm.DefaultGenerateConfig()
	genCfg.MaxTokens = maxTokens
	genCfg.Seed = 42
	genCfg.Sampler.Temperature = 0

	gpu.PerfReset()
	result, err := gpuPipe.GenerateDetailed(chatPrompt, genCfg)
	gpuUs := gpu.PerfGetGpuUs()
	dispatches := gpu.PerfGetDispatches()
	barriers := gpu.PerfGetBarriers()
	if err != nil {
		res.Err = fmt.Sprintf("gen: %v", err)
		return
	}

	res.Text = result.Text
	res.TokS = result.TokensPerSec
	res.PrefillMs = result.PrefillTimeMs
	res.GenerateMs = result.GenerateTimeMs
	res.TotalTokens = result.TotalTokens
	res.PromptTokens = result.PromptTokens

	genTokens := result.TotalTokens - result.PromptTokens
	if genTokens < 1 { genTokens = 1 }
	fmt.Fprintf(os.Stderr, "  dlgo %s: %.1f tok/s, prefill=%.0fms, gen=%.0fms, %d tok\n",
		name, res.TokS, res.PrefillMs, res.GenerateMs, res.TotalTokens)
	fmt.Fprintf(os.Stderr, "  [PERF] GPU fence-wait: %.1fms, dispatches: %d (%.0f/tok), barriers: %d (%.0f/tok)\n",
		gpuUs/1000.0, dispatches, float64(dispatches)/float64(genTokens), barriers, float64(barriers)/float64(genTokens))
	fmt.Fprintf(os.Stderr, "  [PERF] Host overhead: %.1fms (gen=%.1fms - gpu=%.1fms)\n",
		result.GenerateTimeMs - gpuUs/1000.0, result.GenerateTimeMs, gpuUs/1000.0)
}
