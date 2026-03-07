//go:build ignore

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

type OllamaReq struct {
	Model   string        `json:"model"`
	Msgs    []OllamaMsg   `json:"messages"`
	Stream  bool          `json:"stream"`
	Options OllamaOpts    `json:"options"`
}
type OllamaMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type OllamaOpts struct {
	Temperature float64 `json:"temperature"`
	NumPredict  int     `json:"num_predict"`
	Seed        int     `json:"seed"`
	NumGPU      int     `json:"num_gpu"`
}
type OllamaResp struct {
	Message            OllamaMsg     `json:"message"`
	Done               bool          `json:"done"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

type result struct {
	prefillMs, genMs, tps float64
	genTok                int
	text                  string
}

func ollamaBench(model, sys, prompt string, max, numGPU int) (*result, error) {
	req := OllamaReq{
		Model:  model,
		Msgs:   []OllamaMsg{{Role: "system", Content: sys}, {Role: "user", Content: prompt}},
		Stream: false,
		Options: OllamaOpts{Temperature: 0, NumPredict: max, Seed: 42, NumGPU: numGPU},
	}
	b, _ := json.Marshal(req)
	resp, err := http.Post("http://localhost:11434/api/chat", "application/json", bytes.NewBuffer(b))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var r OllamaResp
	if err := json.Unmarshal(raw, &r); err != nil {
		return nil, fmt.Errorf("json: %v body=%s", err, string(raw[:min(200, len(raw))]))
	}
	tps := 0.0
	if r.EvalDuration > 0 {
		tps = float64(r.EvalCount) / r.EvalDuration.Seconds()
	}
	return &result{float64(r.PromptEvalDuration.Milliseconds()), float64(r.EvalDuration.Milliseconds()), tps, r.EvalCount, r.Message.Content}, nil
}

func cpuBench(p *llm.Pipeline, sys, prompt string, max int) (*result, error) {
	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = max
	cfg.Seed = 42
	cfg.Sampler.Temperature = 0
	r, err := p.GenerateDetailed(llm.FormatChat(p.Model.Config, sys, prompt), cfg)
	if err != nil {
		return nil, err
	}
	return &result{r.PrefillTimeMs, r.GenerateTimeMs, r.TokensPerSec, r.TotalTokens, r.Text}, nil
}

func gpuBench(gp *gpu.GpuPipeline, sys, prompt string, max int, cpuModel *llm.Model, cpuTok *llm.Tokenizer) (*result, error) {
	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = max
	cfg.Seed = 42
	cfg.Sampler.Temperature = 0
	formatted := llm.FormatChat(cpuModel.Config, sys, prompt)
	r, err := gp.GenerateDetailed(formatted, cfg)
	if err != nil {
		return nil, err
	}
	return &result{r.PrefillTimeMs, r.GenerateTimeMs, r.TokensPerSec, r.TotalTokens, r.Text}, nil
}

func preview(s string, n int) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type pair struct {
	label, dlgoPath, ollamaName string
}

func main() {
	sys := "You are a helpful assistant."
	maxTok := 128
	prompt := "What is the capital of France?"

	models := []pair{
		{"SmolLM2 360M Q8_0", `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`, "dlgo-smollm2-360m"},
		{"TinyLlama 1.1B Q4_0", `C:\projects\evoke\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`, "dlgo-tinyllama"},
		{"Qwen 2.5 0.5B Q4_K_M", `C:\projects\evoke\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`, "dlgo-qwen25"},
		{"Gemma 3 1B Q4_K_M", `C:\projects\evoke\models\gemma-3-1b-it-Q4_K_M.gguf`, "dlgo-gemma3"},
	}

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           dlgo GPU vs Ollama GPU vs dlgo CPU — Benchmark                      ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════╝")

	// Phase 1: CPU benchmark (no GPU involved) to confirm no regression
	fmt.Println("\n═══ Phase 1: CPU dlgo vs Ollama CPU (regression check) ═══")
	for _, m := range models {
		fmt.Printf("\n─── %s ───\n", m.label)
		cpuPipe, err := llm.NewPipeline(m.dlgoPath, 512)
		if err != nil {
			fmt.Printf("  SKIP (load fail: %v)\n", err)
			continue
		}

		oRes, err := ollamaBench(m.ollamaName, sys, prompt, maxTok, 0)
		if err != nil {
			fmt.Printf("  Ollama CPU: FAIL %v\n", err)
		} else {
			fmt.Printf("  Ollama CPU:  gen=%5.0fms  %5.1f tok/s  prefill=%4.0fms  [%d tok]  → %s\n",
				oRes.genMs, oRes.tps, oRes.prefillMs, oRes.genTok, preview(oRes.text, 50))
		}

		dRes, err := cpuBench(cpuPipe, sys, prompt, maxTok)
		if err != nil {
			fmt.Printf("  dlgo CPU:    FAIL %v\n", err)
		} else {
			fmt.Printf("  dlgo CPU:    gen=%5.0fms  %5.1f tok/s  prefill=%4.0fms  [%d tok]  → %s\n",
				dRes.genMs, dRes.tps, dRes.prefillMs, dRes.genTok, preview(dRes.text, 50))
		}
		if oRes != nil && dRes != nil {
			genDelta := (dRes.tps - oRes.tps) / oRes.tps * 100
			fmt.Printf("  CPU delta:   gen %+.1f%%\n", genDelta)
		}
	}

	// Phase 2: Ollama GPU benchmark (target numbers)
	fmt.Println("\n═══ Phase 2: Ollama GPU (target) ═══")
	for _, m := range models {
		fmt.Printf("\n─── %s ───\n", m.label)
		oRes, err := ollamaBench(m.ollamaName, sys, prompt, maxTok, 99)
		if err != nil {
			fmt.Printf("  Ollama GPU: FAIL %v\n", err)
		} else {
			fmt.Printf("  Ollama GPU:  gen=%5.0fms  %5.1f tok/s  prefill=%4.0fms  [%d tok]  → %s\n",
				oRes.genMs, oRes.tps, oRes.prefillMs, oRes.genTok, preview(oRes.text, 50))
		}
	}

	// Phase 3: dlgo GPU benchmark
	fmt.Println("\n═══ Phase 3: dlgo GPU ═══")
	if err := gpu.Init(); err != nil {
		fmt.Printf("GPU init failed: %v\n", err)
		return
	}
	defer gpu.Shutdown()
	fmt.Printf("GPU: %s (%.0f MB VRAM)\n", gpu.DeviceName(), float64(gpu.VRAMBytes())/(1024*1024))

	for _, m := range models {
		fmt.Printf("\n─── %s ───\n", m.label)
		cpuPipe, err := llm.NewPipeline(m.dlgoPath, 512)
		if err != nil {
			fmt.Printf("  SKIP (load fail: %v)\n", err)
			continue
		}

		gpuPipe, err := gpu.NewGpuPipeline(cpuPipe)
		if err != nil {
			fmt.Printf("  GPU pipeline fail: %v\n", err)
			continue
		}

		gRes, err := gpuBench(gpuPipe, sys, prompt, maxTok, cpuPipe.Model, cpuPipe.Tokenizer)
		if err != nil {
			fmt.Printf("  dlgo GPU:   FAIL %v\n", err)
		} else {
			fmt.Printf("  dlgo GPU:    gen=%5.0fms  %5.1f tok/s  prefill=%4.0fms  [%d tok]  → %s\n",
				gRes.genMs, gRes.tps, gRes.prefillMs, gRes.genTok, preview(gRes.text, 50))
		}

		// Compare with Ollama GPU
		oGPU, err := ollamaBench(m.ollamaName, sys, prompt, maxTok, 99)
		if err != nil {
			fmt.Printf("  Ollama GPU:  FAIL %v\n", err)
		} else {
			fmt.Printf("  Ollama GPU:  gen=%5.0fms  %5.1f tok/s  prefill=%4.0fms  [%d tok]\n",
				oGPU.genMs, oGPU.tps, oGPU.prefillMs, oGPU.genTok)
		}
		if gRes != nil && oGPU != nil && oGPU.tps > 0 {
			genDelta := (gRes.tps - oGPU.tps) / oGPU.tps * 100
			fmt.Printf("  GPU delta vs Ollama: gen %+.1f%%\n", genDelta)
		}
	}
}
