//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

// Standalone worker: loads ONE model, benchmarks dlgo CPU+GPU, writes JSON to file, exits.
// All resources are fully released by the OS when this process terminates.
// Usage: bench_worker <gguf_path> <prompt> <gen_tokens> <output_json_path>

type WorkerResult struct {
	CPUPrefillMs float64 `json:"cpu_prefill_ms"`
	CPUGenTokS   float64 `json:"cpu_gen_tok_s"`
	GPUPrefillMs float64 `json:"gpu_prefill_ms"`
	GPUGenTokS   float64 `json:"gpu_gen_tok_s"`
	GPUText      string  `json:"gpu_text"`
	GPUError     string  `json:"gpu_error,omitempty"`
	Layers       int     `json:"layers"`
	Dim          int     `json:"dim"`
	Experts      int     `json:"experts"`
	ActiveExp    int     `json:"active_experts"`
	PromptTokens int     `json:"prompt_tokens"`
}

func main() {
	if len(os.Args) < 5 {
		fmt.Fprintf(os.Stderr, "usage: bench_worker <gguf_path> <prompt> <gen_tokens> <output_json>\n")
		os.Exit(1)
	}
	ggufPath := os.Args[1]
	prompt := os.Args[2]
	genTokens, _ := strconv.Atoi(os.Args[3])
	outPath := os.Args[4]
	if genTokens <= 0 {
		genTokens = 20
	}

	// Redirect stdout to stderr so GPU log messages don't pollute output.
	// JSON results go to the output file instead.
	origStdout := os.Stdout
	os.Stdout = os.Stderr
	_ = origStdout

	pipe, err := llm.NewPipeline(ggufPath, 512)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load fail: %v\n", err)
		os.Exit(1)
	}
	cfg := pipe.Model.Config
	m := pipe.Model

	tokens := pipe.Tokenizer.Encode(prompt)
	if len(tokens) > 0 && tokens[0] == int32(cfg.BOS) {
		tokens = tokens[1:]
	}

	res := WorkerResult{
		Layers:       cfg.NumLayers,
		Dim:          cfg.EmbeddingDim,
		Experts:      cfg.ExpertCount,
		ActiveExp:    cfg.ExpertUsedCount,
		PromptTokens: len(tokens),
	}

	// --- dlgo CPU ---
	func() {
		formatted := llm.FormatChat(cfg, "You are a helpful assistant.", prompt)
		genCfg := llm.DefaultGenerateConfig()
		genCfg.MaxTokens = genTokens
		genCfg.Seed = 42
		genCfg.Sampler.Temperature = 0

		cpuResult, err := pipe.GenerateDetailed(formatted, genCfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "cpu gen fail: %v\n", err)
			return
		}
		res.CPUPrefillMs = cpuResult.PrefillTimeMs
		res.CPUGenTokS = cpuResult.TokensPerSec
	}()

	// --- dlgo GPU ---
	func() {
		gpuPipe, err := gpu.NewGpuPipeline(pipe)
		if err != nil {
			res.GPUError = err.Error()
			return
		}
		defer gpuPipe.FreeAll()

		formatted := llm.FormatChat(cfg, "You are a helpful assistant.", prompt)
		gpuResult, err := gpuPipe.GenerateDetailed(formatted, llm.GenerateConfig{
			MaxTokens: genTokens,
			Sampler:   ops.SamplerConfig{Temperature: 0},
			Seed:      42,
		})
		if err != nil {
			res.GPUError = err.Error()
			return
		}
		res.GPUPrefillMs = gpuResult.PrefillTimeMs
		res.GPUGenTokS = gpuResult.TokensPerSec
		res.GPUText = gpuResult.Text
	}()

	_ = m
	data, _ := json.MarshalIndent(res, "", "  ")
	if err := os.WriteFile(outPath, data, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "write result: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Results written to %s\n", outPath)
}
