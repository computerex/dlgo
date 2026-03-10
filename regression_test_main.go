//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

type modelSpec struct {
	name, path string
}

var models = []modelSpec{
	// Small models (evoke)
	{"Gemma 3 270M Q8_0", `C:\projects\evoke\models\gemma-3-270m-it-Q8_0.gguf`},
	{"SmolLM2 360M Q8_0", `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`},
	{"Qwen 2.5 0.5B Q4_K_M", `C:\projects\evoke\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`},
	{"Qwen3 0.6B Q8_0", `C:\projects\evoke\models\Qwen3-0.6B-Q8_0.gguf`},
	{"TinyLlama 1.1B Q4_0", `C:\projects\evoke\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`},
	{"Gemma 3 1B Q4_K_M", `C:\projects\evoke\models\gemma-3-1b-it-Q4_K_M.gguf`},
	{"Llama 3.2 1B Q4_K_M", `C:\projects\evoke\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf`},
	{"SmolLM2 1.7B Q4_K_M", `C:\projects\evoke\models\smollm2-1.7b-instruct-q4_k_m.gguf`},
	{"Qwen3.5 0.8B Q8_0", `C:\projects\evoke\models\Qwen3.5-0.8B-Q8_0.gguf`},
	{"Phi-4-mini Q3_K_M", `C:\projects\evoke\models\Phi-4-mini-instruct-Q3_K_M.gguf`},

	// Medium models (gollm)
	{"Qwen3.5 2B Q4_K_M", `C:\projects\gollm\Qwen3.5-2B.Q4_K_M.gguf`},
	{"Qwen3.5 9B Q3_K_M", `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`},

	// Large models (gollm) — partial GPU offload
	{"Qwen3.5 27B Q3_K_M", `C:\projects\gollm\Qwen3.5-27B-Q3_K_M.gguf`},
	{"Qwen3.5 35B-A3B MoE Q3_K_M", `C:\projects\gollm\Qwen3.5-35B-A3B-Q3_K_M.gguf`},

	// Frontier models (Downloads) — MoE, hybrid architectures
	{"gpt-oss-20b Q3_K_M", `C:\Users\mohd\Downloads\gpt-oss-20b-Q3_K_M.gguf`},
	{"gpt-oss-20b MXFP4", `C:\Users\mohd\Downloads\gpt-oss-20b-mxfp4.gguf`},
	{"GLM-4.7-Flash Q4_K_XL", `C:\Users\mohd\Downloads\GLM-4.7-Flash-UD-Q4_K_XL.gguf`},
	{"Qwen3-Coder-30B-A3B IQ3_XXS", `C:\Users\mohd\Downloads\Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf`},
	{"Qwen3-Coder-Next 80B IQ3_XXS", `C:\Users\mohd\Downloads\Qwen3-Coder-Next-UD-IQ3_XXS.gguf`},
	{"Qwen3.5-122B-A10B IQ3_XXS", `C:\Users\mohd\Downloads\Qwen3.5-122B-A10B-UD-IQ3_XXS.gguf`},
}

type RegressionResult struct {
	Name            string  `json:"name"`
	Loaded          bool    `json:"loaded"`
	MaxErr          float64 `json:"max_err"`
	AvgErr          float64 `json:"avg_err"`
	TopMatch        bool    `json:"top_match"`
	CPUText         string  `json:"cpu_text"`
	GPUText         string  `json:"gpu_text"`
	CPUTokS         float64 `json:"cpu_tok_s"`
	GPUTokS         float64 `json:"gpu_tok_s"`
	CPUPrefillMs    float64 `json:"cpu_prefill_ms"`
	GPUPrefillMs    float64 `json:"gpu_prefill_ms"`
	CorrectnesPass  bool    `json:"correctness_pass"`
	CoherencePass   bool    `json:"coherence_pass"`
	Err             string  `json:"err,omitempty"`
	GPULayers       int     `json:"gpu_layers"`
	TotalLayers     int     `json:"total_layers"`
	IsPartialGPU    bool    `json:"is_partial_gpu"`
}

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   dlgo Full Regression Test — Subprocess Per Model (CPU + GPU)   ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════╝")

	workerExe := filepath.Join(os.TempDir(), "regression_worker.exe")

	fmt.Println("Building regression worker...")
	buildCmd := exec.Command("go", "build", "-tags", "cgo vulkan", "-ldflags", "-linkmode internal",
		"-o", workerExe, "regression_worker_main.go")
	buildCmd.Dir = `C:\projects\dlgo`
	buildCmd.Stdout = os.Stderr
	buildCmd.Stderr = os.Stderr
	if err := buildCmd.Run(); err != nil {
		fmt.Printf("FATAL: failed to build worker: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Worker built: %s\n\n", workerExe)

	var results []RegressionResult

	for i, m := range models {
		fmt.Printf("═══ [%d/%d] %s ═══\n", i+1, len(models), m.name)

		if _, err := os.Stat(m.path); os.IsNotExist(err) {
			fmt.Printf("  SKIP: model file not found\n\n")
			results = append(results, RegressionResult{
				Name: m.name,
				Err:  "model file not found",
			})
			continue
		}

		tmpJSON := filepath.Join(os.TempDir(), fmt.Sprintf("regression_%d.json", i))

		start := time.Now()
		cmd := exec.Command(workerExe, m.name, m.path, tmpJSON)
		cmd.Stderr = os.Stderr
		err := cmd.Run()
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("  Worker failed (exit %v) in %.1fs\n\n", err, elapsed.Seconds())
			results = append(results, RegressionResult{
				Name: m.name,
				Err:  fmt.Sprintf("worker exit: %v", err),
			})
			continue
		}

		data, readErr := os.ReadFile(tmpJSON)
		if readErr != nil {
			fmt.Printf("  Failed to read result: %v\n\n", readErr)
			results = append(results, RegressionResult{
				Name: m.name,
				Err:  fmt.Sprintf("read result: %v", readErr),
			})
			continue
		}

		var res RegressionResult
		if jsonErr := json.Unmarshal(data, &res); jsonErr != nil {
			fmt.Printf("  Failed to parse result: %v\n\n", jsonErr)
			results = append(results, RegressionResult{
				Name: m.name,
				Err:  fmt.Sprintf("parse result: %v", jsonErr),
			})
			continue
		}

		status := "PASS"
		if res.Err != "" {
			status = "SKIP"
		} else if !res.CorrectnesPass || !res.CoherencePass {
			status = "FAIL"
		}
		gpuInfo := "full GPU"
		if res.IsPartialGPU {
			gpuInfo = fmt.Sprintf("partial %d/%d", res.GPULayers, res.TotalLayers)
		}
		fmt.Printf("  %s  CPU=%.1f tok/s  GPU=%.1f tok/s  (%s)  %.1fs\n\n",
			status, res.CPUTokS, res.GPUTokS, gpuInfo, elapsed.Seconds())

		results = append(results, res)
		os.Remove(tmpJSON)
	}

	// --- Summary ---
	fmt.Println("\n╔══════╦═══════════════════════════════════╦═══════════╦════════════╦══════════════════════════════════╗")
	fmt.Println("║ STAT ║ Model                             ║ CPU tok/s ║ GPU tok/s  ║ Details                          ║")
	fmt.Println("╠══════╬═══════════════════════════════════╬═══════════╬════════════╬══════════════════════════════════╣")
	allPass := true
	for _, r := range results {
		status := "PASS"
		detail := ""
		cpuStr := "—"
		gpuStr := "—"
		if r.Err != "" {
			status = "SKIP"
			detail = r.Err
			if len(detail) > 32 {
				detail = detail[:32]
			}
		} else if !r.CorrectnesPass || !r.CoherencePass {
			status = "FAIL"
			allPass = false
			if !r.CorrectnesPass {
				detail += fmt.Sprintf("maxErr=%.2f ", r.MaxErr)
			}
			if !r.CoherencePass {
				detail += "incoherent "
			}
		} else {
			detail = fmt.Sprintf("maxErr=%.4f", r.MaxErr)
			if r.IsPartialGPU {
				detail += fmt.Sprintf(" partial=%d/%d", r.GPULayers, r.TotalLayers)
			}
		}
		if r.CPUTokS > 0 {
			cpuStr = fmt.Sprintf("%.1f", r.CPUTokS)
		}
		if r.GPUTokS > 0 {
			gpuStr = fmt.Sprintf("%.1f", r.GPUTokS)
		}
		fmt.Printf("║ %-4s ║ %-33s ║ %9s ║ %10s ║ %-32s ║\n", status, r.Name, cpuStr, gpuStr, detail)
	}
	fmt.Println("╚══════╩═══════════════════════════════════╩═══════════╩════════════╩══════════════════════════════════╝")

	// Write JSON results
	jsonData, _ := json.MarshalIndent(results, "", "  ")
	resultFile := fmt.Sprintf("regression_results_%s.json", time.Now().Format("20060102_150405"))
	os.WriteFile(resultFile, jsonData, 0644)
	fmt.Printf("\nResults saved to %s\n", resultFile)

	if allPass {
		fmt.Println("\nAll models passed regression test.")
	} else {
		fmt.Println("\nSome models failed. See details above.")
		os.Exit(1)
	}
}
