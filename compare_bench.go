package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/computerex/dlgo/models/llm"
)

type OllamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []OllamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Options  OllamaOptions   `json:"options"`
}
type OllamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type OllamaOptions struct {
	Temperature float64 `json:"temperature"`
	NumPredict  int     `json:"num_predict"`
	Seed        int     `json:"seed"`
	NumGPU      int     `json:"num_gpu"`
}
type OllamaChatResponse struct {
	Message            OllamaMessage `json:"message"`
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

var prompts = []string{
	"What is the capital of France?",
	"Write a short poem about the ocean.",
	"Explain what gravity is in one sentence.",
}

type pair struct {
	label, dlgoPath, ollamaName, note string
}

var models = []pair{
	{"TinyLlama 1.1B", `C:\projects\evoke\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`, "tinyllama:1.1b", "Q4_0"},
	{"Qwen 2.5 0.5B", `C:\projects\evoke\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`, "qwen2.5:0.5b", "Q4_K_M"},
	{"Gemma 3 1B", `C:\projects\evoke\models\gemma-3-1b-it-Q4_K_M.gguf`, "gemma3:1b", "Q4_K_M"},
	{"SmolLM2 360M", `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`, "smollm2:360m", "dlgo=Q8_0 ollama=F16"},
	{"Gemma 3 270M", `C:\projects\evoke\models\gemma-3-270m-it-Q8_0.gguf`, "", "Q8_0 (dlgo only)"},
}

func ollamaBench(model, sys, prompt string, max int) (*result, error) {
	req := OllamaChatRequest{Model: model, Messages: []OllamaMessage{{Role: "system", Content: sys}, {Role: "user", Content: prompt}}, Stream: false, Options: OllamaOptions{Temperature: 0, NumPredict: max, Seed: 42, NumGPU: 0}}
	b, _ := json.Marshal(req)
	resp, err := http.Post("http://localhost:11434/api/chat", "application/json", bytes.NewBuffer(b))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var r OllamaChatResponse
	if err := json.Unmarshal(raw, &r); err != nil {
		return nil, err
	}
	tps := 0.0
	if r.EvalDuration > 0 {
		tps = float64(r.EvalCount) / r.EvalDuration.Seconds()
	}
	return &result{float64(r.PromptEvalDuration.Milliseconds()), float64(r.EvalDuration.Milliseconds()), tps, r.EvalCount, r.Message.Content}, nil
}

func dlgoBench(p *llm.Pipeline, sys, prompt string, max int) (*result, error) {
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

func preview(s string, n int) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

func main() {
	sys := "You are a helpful assistant."
	max := 128
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           dlgo vs Ollama — Full Comparison (CPU-only)                 ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════╝")
	fmt.Printf("MaxTokens=%d | Temp=0 | Seed=42\n\n", max)

	for _, m := range models {
		if _, err := os.Stat(m.dlgoPath); err != nil {
			continue
		}
		fmt.Printf("━━━ %s (%s) ━━━\n", m.label, m.note)

		if m.ollamaName != "" {
			var or [3]*result
			fmt.Printf("  Ollama (%s):\n", m.ollamaName)
			for i, p := range prompts {
				r, err := ollamaBench(m.ollamaName, sys, p, max)
				if err != nil {
					fmt.Printf("    FAIL %v\n", err)
					continue
				}
				or[i] = r
				tag := ""
				if i == 0 {
					tag = " (cold)"
				}
				fmt.Printf("    [%d] prefill:%4.0fms  gen:%5.0fms  %5.1f tok/s  [%d tok]%s  → %s\n",
					i+1, r.prefillMs, r.genMs, r.tps, r.genTok, tag, preview(r.text, 50))
			}

			var dr [3]*result
			p, err := llm.NewPipeline(m.dlgoPath, 512)
			if err != nil {
				fmt.Printf("  dlgo FAIL: %v\n\n", err)
				continue
			}
			fmt.Printf("  dlgo:\n")
			for i, prompt := range prompts {
				r, err := dlgoBench(p, sys, prompt, max)
				if err != nil {
					fmt.Printf("    FAIL %v\n", err)
					continue
				}
				dr[i] = r
				fmt.Printf("    [%d] prefill:%4.0fms  gen:%5.0fms  %5.1f tok/s  [%d tok]  → %s\n",
					i+1, r.prefillMs, r.genMs, r.tps, r.genTok, preview(r.text, 50))
			}

			var dPre, oPre, dTps, oTps float64
			n := 0
			for i := range prompts {
				if dr[i] != nil && or[i] != nil {
					dPre += dr[i].prefillMs
					oPre += or[i].prefillMs
					dTps += dr[i].tps
					oTps += or[i].tps
					n++
				}
			}
			if n > 0 {
				dPre /= float64(n)
				oPre /= float64(n)
				dTps /= float64(n)
				oTps /= float64(n)
				fmt.Printf("  ── Average: prefill dlgo=%.0fms ollama=%.0fms (%+.0f%%)  gen dlgo=%.1f ollama=%.1f tok/s (%+.0f%%)\n",
					dPre, oPre, (dPre-oPre)/oPre*100, dTps, oTps, (dTps-oTps)/oTps*100)
			}
		} else {
			p, err := llm.NewPipeline(m.dlgoPath, 512)
			if err != nil {
				fmt.Printf("  dlgo FAIL: %v\n\n", err)
				continue
			}
			fmt.Printf("  dlgo (no Ollama equivalent):\n")
			for i, prompt := range prompts {
				r, err := dlgoBench(p, sys, prompt, max)
				if err != nil {
					fmt.Printf("    FAIL %v\n", err)
					continue
				}
				fmt.Printf("    [%d] prefill:%4.0fms  gen:%5.0fms  %5.1f tok/s  [%d tok]  → %s\n",
					i+1, r.prefillMs, r.genMs, r.tps, r.genTok, preview(r.text, 50))
			}
		}
		fmt.Println()
	}
}
