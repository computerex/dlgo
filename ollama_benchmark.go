package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

type GenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
	Options struct {
		Temperature float64 `json:"temperature"`
		NumPredict  int     `json:"num_predict"`
		Seed        int     `json:"seed"`
		NumGPU      int     `json:"num_gpu"`
	} `json:"options"`
}

type GenerateResponse struct {
	Model              string        `json:"model"`
	CreatedAt          time.Time     `json:"created_at"`
	Response           string        `json:"response"`
	Done               bool          `json:"done"`
	Context            []int         `json:"context,omitempty"`
	TotalDuration      time.Duration `json:"total_duration,omitempty"`
	LoadDuration       time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

func main() {
	model := "smollm2:1.7b"
	if len(os.Args) > 1 {
		model = os.Args[1]
	}

	prompt := "Write a short poem about the ocean."

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║               Ollama LLM Inference Benchmark                  ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Printf("\nUser: %q\nMax tokens: 64 (greedy)\nModel: %s\n\n", prompt, model)

	req := GenerateRequest{
		Model:  model,
		Prompt: prompt,
		Stream: false,
	}
	req.Options.Temperature = 0
	req.Options.NumPredict = 64
	req.Options.Seed = 42
	req.Options.NumGPU = 0

	reqBody, err := json.Marshal(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error marshaling request: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Running benchmark...\n")

	resp, err := http.Post("http://localhost:11434/api/generate", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error making request: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading response: %v\n", err)
		os.Exit(1)
	}

	var result GenerateResponse
	if err := json.Unmarshal(body, &result); err != nil {
		fmt.Fprintf(os.Stderr, "Error unmarshaling response: %v\n", err)
		os.Exit(1)
	}

	if !result.Done {
		fmt.Printf("Generation not complete\n")
		os.Exit(1)
	}

	tokensPerSec := float64(result.EvalCount) / result.EvalDuration.Seconds()
	totalTokens := result.PromptEvalCount + result.EvalCount

	preview := result.Response
	if len(preview) > 60 {
		preview = preview[:60] + "..."
	}

	fmt.Printf("%-30s  %5.1f tok/s  prefill:%5.0fms  gen:%5.0fms  [%d tok]\n",
		model, tokensPerSec,
		float64(result.PromptEvalDuration.Milliseconds()),
		float64(result.EvalDuration.Milliseconds()),
		totalTokens)
	fmt.Printf("  → %s\n\n", preview)
}