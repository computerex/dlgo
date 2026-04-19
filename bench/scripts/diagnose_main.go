//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/computerex/dlgo/models/llm"
)

const llamaCLI = `C:\Users\mohd\Downloads\llama-vulkan\llama-cli.exe`
const testPrompt = "Explain what a computer is in exactly two sentences."
const systemPrompt = "You are a helpful assistant."

type modelSpec struct {
	name, path string
}

var models = []modelSpec{
	{"Gemma3-270M", `C:\models\gemma-3-270m-it-Q8_0.gguf`},
	{"Gemma2-2B", `C:\models\gemma-2-2b-it-Q4_K_M.gguf`},
	{"SmolLM2-1.7B", `C:\models\smollm2-1.7b-instruct-q4_k_m.gguf`},
	{"Phi-4-mini", `C:\models\Phi-4-mini-instruct-Q3_K_M.gguf`},
	{"Llama3.2-1B", `C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf`},
	{"Qwen3.5-0.8B", `C:\models\Qwen3.5-0.8B-Q8_0.gguf`},
	{"TinyLlama-1.1B", `C:\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`},
	{"Phi-2", `C:\models\phi-2.Q4_K_M.gguf`},
	{"Qwen2.5-0.5B", `C:\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`},
	{"SmolLM2-360M", `C:\models\smollm2-360m-instruct-q8_0.gguf`},
	{"Gemma3-1B", `C:\models\gemma-3-1b-it-Q4_K_M.gguf`},
}

func main() {
	fmt.Println("=== DIAGNOSIS: CPU vs GPU vs llama.cpp ===")
	fmt.Printf("Prompt: %q\n", testPrompt)
	fmt.Printf("System: %q\n\n", systemPrompt)

	// Step 1: For each model, print the formatted prompt
	for _, m := range models {
		if _, err := os.Stat(m.path); os.IsNotExist(err) {
			continue
		}
		pipe, err := llm.NewPipeline(m.path, 512)
		if err != nil {
			fmt.Printf("[%s] load error: %v\n", m.name, err)
			continue
		}
		cfg := pipe.Model.Config
		formatted := llm.FormatChat(cfg, systemPrompt, testPrompt)
		tokens := pipe.Tokenizer.Encode(formatted)
		fmt.Printf("═══ %s ═══\n", m.name)
		fmt.Printf("  Arch: %s, Template: %q\n", cfg.Architecture, cfg.ChatTemplate)
		fmt.Printf("  BOS: %d, EOS: %d, AddBOS: %v, StopTokens: %v\n",
			cfg.BOS, cfg.EOS, cfg.AddBOS, cfg.StopTokens)
		fmt.Printf("  Prompt tokens: %d\n", len(tokens))
		fmt.Printf("  Formatted prompt:\n---\n%s\n---\n\n", formatted)
		pipe = nil
	}

	// Step 2: Build CPU worker and GPU worker
	cpuWorker := filepath.Join(os.TempDir(), "diag_cpu_worker.exe")
	gpuWorker := filepath.Join(os.TempDir(), "diag_gpu_worker.exe")

	fmt.Println("Building CPU worker...")
	buildCPU := exec.Command("go", "build", "-a",
		"-ldflags", "-linkmode internal",
		"-o", cpuWorker, `bench/scripts/diag_cpu_worker_main.go`)
	buildCPU.Dir = `C:\projects\dlgo`
	buildCPU.Stdout = os.Stderr
	buildCPU.Stderr = os.Stderr
	if err := buildCPU.Run(); err != nil {
		fmt.Printf("FATAL: CPU worker build failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Building GPU worker...")
	buildGPU := exec.Command("go", "build", "-a", "-tags", "cgo vulkan",
		"-ldflags", "-linkmode internal",
		"-o", gpuWorker, `bench/scripts/diag_gpu_worker_main.go`)
	buildGPU.Dir = `C:\projects\dlgo`
	buildGPU.Stdout = os.Stderr
	buildGPU.Stderr = os.Stderr
	if err := buildGPU.Run(); err != nil {
		fmt.Printf("FATAL: GPU worker build failed: %v\n", err)
		os.Exit(1)
	}

	type diagResult struct {
		Text   string `json:"text"`
		Tokens string `json:"tokens"`
		Err    string `json:"err,omitempty"`
	}

	fmt.Println("\n\n=== RUNNING INFERENCE COMPARISONS ===\n")

	for i, m := range models {
		if _, err := os.Stat(m.path); os.IsNotExist(err) {
			continue
		}
		fmt.Printf("═══ [%d/%d] %s ═══\n", i+1, len(models), m.name)

		// Run CPU
		cpuOutFile := filepath.Join(os.TempDir(), fmt.Sprintf("diag_cpu_%d.json", i))
		fmt.Printf("  Running CPU...\n")
		cpuCmd := exec.Command(cpuWorker, m.path, cpuOutFile)
		cpuCmd.Stderr = os.Stderr
		cpuStart := time.Now()
		cpuCmd.Run()
		cpuTime := time.Since(cpuStart)

		var cpuRes diagResult
		if data, err := os.ReadFile(cpuOutFile); err == nil {
			json.Unmarshal(data, &cpuRes)
		}

		// Run GPU
		gpuOutFile := filepath.Join(os.TempDir(), fmt.Sprintf("diag_gpu_%d.json", i))
		fmt.Printf("  Running GPU...\n")
		gpuCmd := exec.Command(gpuWorker, m.path, gpuOutFile)
		gpuCmd.Stderr = os.Stderr
		gpuStart := time.Now()
		gpuCmd.Run()
		gpuTime := time.Since(gpuStart)

		var gpuRes diagResult
		if data, err := os.ReadFile(gpuOutFile); err == nil {
			json.Unmarshal(data, &gpuRes)
		}

		// Run llama.cpp
		fmt.Printf("  Running llama.cpp...\n")
		llamaText, llamaErr := runLlama(m.path)

		// Report
		fmt.Printf("\n  CPU  (%5.1fs): ", cpuTime.Seconds())
		if cpuRes.Err != "" {
			fmt.Printf("ERROR: %s\n", cpuRes.Err)
		} else {
			fmt.Printf("%s\n", truncate(cpuRes.Text, 200))
		}
		fmt.Printf("  GPU  (%5.1fs): ", gpuTime.Seconds())
		if gpuRes.Err != "" {
			fmt.Printf("ERROR: %s\n", gpuRes.Err)
		} else {
			fmt.Printf("%s\n", truncate(gpuRes.Text, 200))
		}
		fmt.Printf("  LLAMA(%5.0fs): ", 0.0)
		if llamaErr != "" {
			fmt.Printf("ERROR: %s\n", llamaErr)
		} else {
			fmt.Printf("%s\n", truncate(llamaText, 200))
		}

		cpuGPUMatch := cpuRes.Text == gpuRes.Text
		cpuLlamaMatch := wordSimilarity(cpuRes.Text, llamaText) > 0.5
		gpuLlamaMatch := wordSimilarity(gpuRes.Text, llamaText) > 0.5

		fmt.Printf("\n  CPU==GPU: %v | CPU~LLAMA: %v (%.0f%%) | GPU~LLAMA: %v (%.0f%%)\n\n",
			cpuGPUMatch,
			cpuLlamaMatch, wordSimilarity(cpuRes.Text, llamaText)*100,
			gpuLlamaMatch, wordSimilarity(gpuRes.Text, llamaText)*100)
	}
}

var reTiming = regexp.MustCompile(`Prompt:\s+([\d.]+)\s+t/s\s*\|\s*Generation:\s+([\d.]+)\s+t/s`)

func runLlama(modelPath string) (text string, errStr string) {
	args := []string{
		"-m", modelPath,
		"-p", fmt.Sprintf("[INST] %s %s [/INST]", systemPrompt, testPrompt),
		"-n", "150",
		"--temp", "0",
		"-c", "2048",
		"--no-display-prompt",
		"--log-disable",
	}
	cmd := exec.Command(llamaCLI, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Sprintf("exec: %v\n%s", err, truncate(string(out), 500))
	}
	text = extractLlamaText(string(out))
	return text, ""
}

func extractLlamaText(combined string) string {
	lines := strings.Split(combined, "\n")
	var textLines []string
	for _, line := range lines {
		if strings.HasPrefix(line, "llama_") || strings.HasPrefix(line, "llm_") ||
			strings.HasPrefix(line, "ggml_") || strings.HasPrefix(line, "srv_") ||
			strings.HasPrefix(line, "build:") || strings.HasPrefix(line, "system_info:") ||
			reTiming.MatchString(line) || strings.TrimSpace(line) == "" {
			continue
		}
		if strings.Contains(line, "tokens per second") {
			continue
		}
		textLines = append(textLines, line)
	}
	result := strings.TrimSpace(strings.Join(textLines, "\n"))
	if idx := strings.Index(result, "\nllama_"); idx >= 0 {
		result = result[:idx]
	}
	if idx := strings.Index(result, "\nllm_"); idx >= 0 {
		result = result[:idx]
	}
	return strings.TrimSpace(result)
}

func wordSimilarity(a, b string) float64 {
	wordsA := strings.Fields(strings.ToLower(a))
	wordsB := strings.Fields(strings.ToLower(b))
	if len(wordsA) == 0 || len(wordsB) == 0 {
		if len(wordsA) == 0 && len(wordsB) == 0 {
			return 1.0
		}
		return 0.0
	}
	setA := make(map[string]int)
	for _, w := range wordsA {
		setA[w]++
	}
	setB := make(map[string]int)
	for _, w := range wordsB {
		setB[w]++
	}
	allWords := make(map[string]bool)
	for w := range setA {
		allWords[w] = true
	}
	for w := range setB {
		allWords[w] = true
	}
	var dotProduct, magA, magB float64
	for w := range allWords {
		a := float64(setA[w])
		b := float64(setB[w])
		dotProduct += a * b
		magA += a * a
		magB += b * b
	}
	if magA == 0 || magB == 0 {
		return 0
	}
	return dotProduct / (math.Sqrt(magA) * math.Sqrt(magB))
}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

// unused but needed for compilation
var _ = sort.Strings
var _ = strconv.Atoi
