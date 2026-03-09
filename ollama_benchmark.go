//go:build ignore

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

type ollamaReq struct {
	Model   string      `json:"model"`
	Msgs    []ollamaMsg `json:"messages"`
	Stream  bool        `json:"stream"`
	Options ollamaOpts  `json:"options"`
}
type ollamaMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type ollamaOpts struct {
	Temperature float64 `json:"temperature"`
	NumPredict  int     `json:"num_predict"`
	Seed        int     `json:"seed"`
	NumGPU      int     `json:"num_gpu"`
}
type ollamaResp struct {
	Message            ollamaMsg     `json:"message"`
	Done               bool          `json:"done"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

func ollamaGenerate(model, prompt string, maxTok, numGPU int) (float64, float64, int, int, error) {
	req := ollamaReq{
		Model:  model,
		Msgs:   []ollamaMsg{{Role: "user", Content: prompt}},
		Stream: false,
		Options: ollamaOpts{Temperature: 0, NumPredict: maxTok, Seed: 42, NumGPU: numGPU},
	}
	b, _ := json.Marshal(req)
	resp, err := http.Post("http://localhost:11434/api/chat", "application/json", bytes.NewBuffer(b))
	if err != nil {
		return 0, 0, 0, 0, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var r ollamaResp
	if err := json.Unmarshal(raw, &r); err != nil {
		return 0, 0, 0, 0, fmt.Errorf("json: %v", err)
	}
	gms := float64(r.EvalDuration) / 1e6
	return float64(r.PromptEvalDuration) / 1e6, gms, r.PromptEvalCount, r.EvalCount, nil
}

func main() {
	ggufPath := `C:\projects\evoke\models\Phi-4-mini-instruct-Q3_K_M.gguf`
	ollamaName := "dlgo-phi4-mini"
	prompt := "Explain what a compiler does in one paragraph."
	maxTok := 64

	fmt.Println("=== Phi-4-mini Q3_K_M Quick Benchmark ===")
	fmt.Println()

	// Ollama GPU
	fmt.Println("--- Ollama GPU ---")
	_, gms, _, genTok, err := ollamaGenerate(ollamaName, prompt, maxTok, 99)
	if err != nil {
		fmt.Printf("  Ollama GPU error: %v\n", err)
	} else {
		ollamaTokS := float64(genTok) / (gms / 1000)
		fmt.Printf("  Generation: %d tokens in %.1f ms = %.1f tok/s\n", genTok, gms, ollamaTokS)
	}

	// dlgo GPU
	fmt.Println("--- dlgo GPU ---")
	pipeline, err := llm.NewPipeline(ggufPath, 2048)
	if err != nil {
		fmt.Printf("  Load error: %v\n", err)
		return
	}
	gpuPipe, err := gpu.NewGpuPipeline(pipeline)
	if err != nil {
		fmt.Printf("  GPU upload error: %v\n", err)
		return
	}
	defer gpuPipe.FreeAll()

	cfg := llm.GenerateConfig{MaxTokens: maxTok}
	res, err := gpuPipe.GenerateDetailed(prompt, cfg)
	if err != nil {
		fmt.Printf("  Generate error: %v\n", err)
		return
	}
	fmt.Printf("  Generation: %d tokens in %.1f ms = %.1f tok/s\n",
		res.TotalTokens-res.PromptTokens, res.GenerateTimeMs, res.TokensPerSec)

	ollamaTokS := float64(genTok) / (gms / 1000)
	fmt.Printf("\n=== Comparison ===\n")
	fmt.Printf("  Ollama GPU: %.1f tok/s\n", ollamaTokS)
	fmt.Printf("  dlgo GPU:   %.1f tok/s\n", res.TokensPerSec)
	diff := (ollamaTokS - res.TokensPerSec) / ollamaTokS * 100
	fmt.Printf("  Gap: %.1f%%\n", diff)
}
