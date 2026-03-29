// Example: JSON grammar-constrained generation on GPU
//
// Usage: go run -tags "cgo vulkan" examples/json_grammar_gpu/main.go [model.gguf]
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/grammar"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	modelPath := `C:\models\Qwen3.5-0.8B-Q8_0.gguf`
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	fmt.Printf("Loading model: %s\n", modelPath)
	cpuPipe, err := llm.NewPipeline(modelPath, 2048)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}

	gpuPipe, err := gpu.NewGpuPipeline(cpuPipe)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create GPU pipeline: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("GPU pipeline ready (max seq len: %d)\n", gpuPipe.MaxSeqLen)

	// Test 1: JSON grammar on GPU
	fmt.Println("\n=== Test 1: JSON object (GPU + grammar) ===")
	gram, err := grammar.Parse(grammar.JSONGrammar)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Grammar parse error: %v\n", err)
		os.Exit(1)
	}

	prompt := llm.FormatChat(cpuPipe.Model.Config, "",
		"Return a JSON object with fields: name (string), age (number), city (string). "+
			"Example: {\"name\": \"Alice\", \"age\": 30, \"city\": \"NYC\"}. "+
			"Return ONLY the JSON, no other text.")

	cfg := llm.GenerateConfig{
		MaxTokens: 256,
		Sampler: ops.SamplerConfig{
			Temperature:       0.3,
			TopK:              40,
			TopP:              0.9,
			RepetitionPenalty: 1.1,
		},
		Seed:    42,
		Grammar: gram,
		Stream: func(token string) {
			fmt.Print(token)
		},
	}

	start := time.Now()
	result, err := gpuPipe.GenerateDetailed(prompt, cfg)
	elapsed := time.Since(start)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nGPU generation error: %v\n", err)
		os.Exit(1)
	}

	text := result.Text
	fmt.Printf("\n[%d tokens in %.2fs = %.1f tok/s]\n", result.TotalTokens, elapsed.Seconds(),
		float64(result.TotalTokens)/elapsed.Seconds())
	fmt.Printf("Generated: %s\n", text)

	var v interface{}
	if err := json.Unmarshal([]byte(text), &v); err != nil {
		fmt.Printf("FAIL: INVALID JSON: %v\n", err)
		os.Exit(1)
	}
	pretty, _ := json.MarshalIndent(v, "  ", "  ")
	fmt.Printf("PASS: Valid JSON!\n  %s\n", string(pretty))

	// Test 2: Nested JSON on GPU
	fmt.Println("\n=== Test 2: Nested JSON (GPU + grammar) ===")
	gram2, _ := grammar.Parse(grammar.JSONGrammar)
	prompt2 := llm.FormatChat(cpuPipe.Model.Config, "",
		"Return a JSON object representing a person with nested address. "+
			"Include: name, age, address (with street, city, zip). "+
			"Return ONLY the JSON.")

	cfg.Grammar = gram2
	start = time.Now()
	result2, err := gpuPipe.GenerateDetailed(prompt2, cfg)
	elapsed = time.Since(start)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nGPU generation error: %v\n", err)
		os.Exit(1)
	}

	text2 := result2.Text
	fmt.Printf("\n[%d tokens in %.2fs = %.1f tok/s]\n", result2.TotalTokens, elapsed.Seconds(),
		float64(result2.TotalTokens)/elapsed.Seconds())
	fmt.Printf("Generated: %s\n", text2)

	if err := json.Unmarshal([]byte(text2), &v); err != nil {
		fmt.Printf("FAIL: INVALID JSON: %v\n", err)
		os.Exit(1)
	}
	pretty, _ = json.MarshalIndent(v, "  ", "  ")
	fmt.Printf("PASS: Valid JSON!\n  %s\n", string(pretty))

	fmt.Println("\nAll GPU grammar tests passed!")
}
