//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

type CoherencyResult struct {
	Name      string  `json:"name"`
	Pass      bool    `json:"pass"`
	Text      string  `json:"text"`
	TokS      float64 `json:"tok_s"`
	GenMs     float64 `json:"gen_ms"`
	GPULayers int     `json:"gpu_layers"`
	Dp4a      bool    `json:"dp4a"`
	Err       string  `json:"err,omitempty"`
}

func main() {
	if len(os.Args) < 4 {
		fmt.Fprintf(os.Stderr, "usage: coherency_worker <name> <gguf_path> <output_json>\n")
		os.Exit(1)
	}
	name := os.Args[1]
	ggufPath := os.Args[2]
	outPath := os.Args[3]

	res := CoherencyResult{Name: name}
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
	prompt := llm.FormatChat(cfg, "You are a helpful assistant.", "Write a Python function that checks if a number is prime. Include docstring and examples.")

	genCfg := llm.DefaultGenerateConfig()
	genCfg.MaxTokens = 256
	genCfg.Seed = 42
	genCfg.Sampler.Temperature = 0

	result, err := gpuPipe.GenerateDetailed(prompt, genCfg)
	if err != nil {
		res.Err = fmt.Sprintf("gen: %v", err)
		return
	}

	res.Text = result.Text
	res.TokS = result.TokensPerSec
	res.GenMs = result.GenerateTimeMs
	res.Pass = isCoherent(result.Text)

	fmt.Fprintf(os.Stderr, "  %s: %.1f tok/s gpu=%d dp4a=%v pass=%v\n",
		name, res.TokS, res.GPULayers, res.Dp4a, res.Pass)
	preview := strings.TrimSpace(strings.ReplaceAll(result.Text, "\n", " "))
	if len(preview) > 300 {
		preview = preview[:300] + "..."
	}
	fmt.Fprintf(os.Stderr, "  Output: %s\n", preview)
}

func isCoherent(text string) bool {
	if text == "" {
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
	if float64(nonASCII)/float64(len([]rune(t))) > 0.5 {
		return false
	}
	words := strings.Fields(t)
	if len(words) < 3 {
		return false
	}
	unique := make(map[string]bool)
	for _, w := range words {
		unique[strings.ToLower(w)] = true
	}
	return float64(len(unique))/float64(len(words)) >= 0.2
}
