//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"os"

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
		fmt.Fprintf(os.Stderr, "usage: compare_worker <name> <gguf_path> <prompt> <output_json>\n")
		os.Exit(1)
	}
	name := os.Args[1]
	ggufPath := os.Args[2]
	prompt := os.Args[3]
	outPath := os.Args[4]

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

	pipe, err := llm.NewPipeline(ggufPath, 2048)
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
	genCfg.MaxTokens = 150
	genCfg.Seed = 42
	genCfg.Sampler.Temperature = 0

	result, err := gpuPipe.GenerateDetailed(chatPrompt, genCfg)
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

	fmt.Fprintf(os.Stderr, "  dlgo %s: %.1f tok/s, prefill=%.0fms, gen=%.0fms, %d tok\n",
		name, res.TokS, res.PrefillMs, res.GenerateMs, res.TotalTokens)
}
