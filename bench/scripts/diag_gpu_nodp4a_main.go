//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

type diagResult struct {
	Text   string `json:"text"`
	Tokens string `json:"tokens"`
	Err    string `json:"err,omitempty"`
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "usage: diag_gpu_nodp4a <model.gguf> <output.json>\n")
		os.Exit(1)
	}
	modelPath := os.Args[1]
	outPath := os.Args[2]

	os.Setenv("DLGO_NO_DP4A", "1")

	res := diagResult{}
	defer func() {
		data, _ := json.MarshalIndent(res, "", "  ")
		os.WriteFile(outPath, data, 0644)
	}()

	if err := gpu.Init(); err != nil {
		res.Err = fmt.Sprintf("gpu init: %v", err)
		return
	}
	defer gpu.Shutdown()

	pipe, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		res.Err = fmt.Sprintf("load: %v", err)
		return
	}

	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		res.Err = fmt.Sprintf("gpu pipeline: %v", err)
		return
	}
	defer gpuPipe.FreeAll()

	cfg := pipe.Model.Config
	prompt := llm.FormatChat(cfg, "You are a helpful assistant.", "Explain what a computer is in exactly two sentences.")

	genCfg := llm.DefaultGenerateConfig()
	genCfg.MaxTokens = 100
	genCfg.Seed = 42
	genCfg.Sampler = ops.SamplerConfig{
		Temperature:       0,
		RepetitionPenalty: 1.1,
	}

	result, err := gpuPipe.GenerateDetailed(prompt, genCfg)
	if err != nil {
		res.Err = fmt.Sprintf("gen: %v", err)
		return
	}

	res.Text = result.Text
	fmt.Fprintf(os.Stderr, "  GPU (no dp4a): %d tokens: %s\n", result.TotalTokens, res.Text)
}
