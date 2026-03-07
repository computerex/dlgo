//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

type ollamaGen struct {
	Response           string `json:"response"`
	TotalDuration      int64  `json:"total_duration"`
	LoadDuration       int64  `json:"load_duration"`
	PromptEvalDuration int64  `json:"prompt_eval_duration"`
	EvalDuration       int64  `json:"eval_duration"`
	EvalCount          int    `json:"eval_count"`
	PromptEvalCount    int    `json:"prompt_eval_count"`
}

func ollamaGenerate(model, prompt string, maxTok int) (*ollamaGen, error) {
	body := fmt.Sprintf(`{"model":"%s","prompt":"%s","stream":false,"options":{"temperature":0,"seed":42,"num_predict":%d}}`, model, prompt, maxTok)
	resp, err := http.Post("http://localhost:11434/api/generate", "application/json", strings.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var r ollamaGen
	json.NewDecoder(resp.Body).Decode(&r)
	return &r, nil
}

func main() {
	prompt := "Write a short story about a robot exploring a mysterious forest."
	maxTok := 128
	runs := 5

	type model struct {
		name   string
		ollama string
		gguf   string
	}
	models := []model{
		{"SmolLM2 360M Q8_0", "smollm2:360m", `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`},
		{"TinyLlama 1.1B Q4_0", "tinyllama", `C:\projects\evoke\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`},
	}

	fmt.Println("╔═══════════════════════════════════════════════════════════════╗")
	fmt.Println("║   Vulkan vs Vulkan: dlgo vs Ollama (multi-run average)       ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	for _, m := range models {
		fmt.Printf("═══ %s (%d runs) ═══\n", m.name, runs)

		// Warm up Ollama
		ollamaGenerate(m.ollama, "hi", 1)
		time.Sleep(time.Second)

		var ollamaTotalMs float64
		var ollamaTok int
		for r := 0; r < runs; r++ {
			res, err := ollamaGenerate(m.ollama, prompt, maxTok)
			if err != nil || res.EvalCount == 0 {
				fmt.Printf("  Ollama run %d failed\n", r+1)
				continue
			}
			ms := float64(res.EvalDuration) / 1e6
			tps := float64(res.EvalCount) / (ms / 1000.0)
			fmt.Printf("  Ollama run %d: %3.0fms  %5.1f tok/s  [%d tok]\n", r+1, ms, tps, res.EvalCount)
			ollamaTotalMs += ms
			ollamaTok += res.EvalCount
		}
		ollamaAvgTPS := float64(ollamaTok) / (ollamaTotalMs / 1000.0)

		// dlgo GPU
		if err := gpu.Init(); err != nil {
			fmt.Println("GPU init failed:", err)
			os.Exit(1)
		}
		pipe, err := llm.NewPipeline(m.gguf, 512)
		if err != nil {
			fmt.Println("Model load failed:", err)
			continue
		}
		gpuPipe, err := gpu.NewGpuPipeline(pipe)
		if err != nil {
			fmt.Println("GPU pipeline failed:", err)
			continue
		}

		// Warmup
		gpuPipe.KVCache.Reset()
		gpuPipe.GenerateDetailed(llm.FormatChat(pipe.Model.Config, "", "hi"), llm.GenerateConfig{MaxTokens: 1, Sampler: ops.SamplerConfig{Temperature: 0}})

		formatted := llm.FormatChat(pipe.Model.Config, "", prompt)
		var dlgoTotalMs float64
		var dlgoTok int
		for r := 0; r < runs; r++ {
			gpuPipe.KVCache.Reset()
			res, err := gpuPipe.GenerateDetailed(formatted, llm.GenerateConfig{
				MaxTokens: maxTok,
				Sampler:   ops.SamplerConfig{Temperature: 0},
				Seed:      42,
			})
			if err != nil {
				fmt.Printf("  dlgo run %d failed: %v\n", r+1, err)
				continue
			}
			tps := float64(res.TotalTokens) / (res.GenerateTimeMs / 1000.0)
			fmt.Printf("  dlgo  run %d: %3.0fms  %5.1f tok/s  [%d tok]\n", r+1, res.GenerateTimeMs, tps, res.TotalTokens)
			dlgoTotalMs += res.GenerateTimeMs
			dlgoTok += res.TotalTokens
		}
		dlgoAvgTPS := float64(dlgoTok) / (dlgoTotalMs / 1000.0)

		gpu.Shutdown()

		delta := (dlgoAvgTPS - ollamaAvgTPS) / ollamaAvgTPS * 100
		fmt.Printf("\n  Average: dlgo=%5.1f tok/s  Ollama Vulkan=%5.1f tok/s  delta=%+.1f%%\n\n", dlgoAvgTPS, ollamaAvgTPS, delta)
	}
}
