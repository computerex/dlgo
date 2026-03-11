//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type modelSpec struct {
	name, path string
}

var models = []modelSpec{
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
	{"Qwen3.5 2B Q4_K_M", `C:\projects\gollm\Qwen3.5-2B.Q4_K_M.gguf`},
	{"Qwen3.5 9B Q3_K_M", `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`},
	{"gpt-oss-20b Q3_K_M", `C:\Users\mohd\Downloads\gpt-oss-20b-Q3_K_M.gguf`},
	{"gpt-oss-20b MXFP4", `C:\Users\mohd\Downloads\gpt-oss-20b-mxfp4.gguf`},
	{"GLM-4.7-Flash Q4_K_XL", `C:\Users\mohd\Downloads\GLM-4.7-Flash-UD-Q4_K_XL.gguf`},
	{"Qwen3.5 35B-A3B MoE Q3_K_M", `C:\projects\gollm\Qwen3.5-35B-A3B-Q3_K_M.gguf`},
	{"Qwen3-Coder-30B-A3B IQ3_XXS", `C:\Users\mohd\Downloads\Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf`},
	{"Qwen3-Coder-Next IQ3_XXS", `C:\Users\mohd\Downloads\Qwen3-Coder-Next-UD-IQ3_XXS.gguf`},
	{"Qwen3.5-122B-A10B IQ3_XXS", `C:\Users\mohd\Downloads\Qwen3.5-122B-A10B-UD-IQ3_XXS.gguf`},
}

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
	fmt.Println("╔═══════════════════════════════════════════════════╗")
	fmt.Println("║   dlgo GPU Coherency Test — All Models           ║")
	fmt.Println("╚═══════════════════════════════════════════════════╝")

	workerExe := filepath.Join(os.TempDir(), "coherency_worker.exe")

	fmt.Println("Building coherency worker...")
	buildCmd := exec.Command("go", "build", "-a", "-tags", "cgo vulkan", "-ldflags", "-linkmode internal",
		"-o", workerExe, "coherency_worker_main.go")
	buildCmd.Dir = `C:\projects\dlgo`
	buildCmd.Stdout = os.Stderr
	buildCmd.Stderr = os.Stderr
	if err := buildCmd.Run(); err != nil {
		fmt.Printf("FATAL: failed to build worker: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Worker built: %s\n\n", workerExe)

	var results []CoherencyResult
	allPass := true

	for i, m := range models {
		fmt.Printf("═══ [%d/%d] %s ═══\n", i+1, len(models), m.name)

		if _, err := os.Stat(m.path); os.IsNotExist(err) {
			fmt.Printf("  SKIP: file not found\n\n")
			results = append(results, CoherencyResult{Name: m.name, Err: "not found"})
			continue
		}

		tmpJSON := filepath.Join(os.TempDir(), fmt.Sprintf("coherency_%d.json", i))
		start := time.Now()
		cmd := exec.Command(workerExe, m.name, m.path, tmpJSON)
		cmd.Stderr = os.Stderr
		err := cmd.Run()
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("  Worker failed (%v) in %.1fs\n\n", err, elapsed.Seconds())
			results = append(results, CoherencyResult{Name: m.name, Err: fmt.Sprintf("worker: %v", err)})
			allPass = false
			continue
		}

		data, readErr := os.ReadFile(tmpJSON)
		if readErr != nil {
			fmt.Printf("  Failed to read result: %v\n\n", readErr)
			results = append(results, CoherencyResult{Name: m.name, Err: fmt.Sprintf("read: %v", readErr)})
			continue
		}

		var res CoherencyResult
		if jsonErr := json.Unmarshal(data, &res); jsonErr != nil {
			results = append(results, CoherencyResult{Name: m.name, Err: fmt.Sprintf("parse: %v", jsonErr)})
			continue
		}

		status := "PASS"
		if res.Err != "" {
			status = "SKIP"
		} else if !res.Pass {
			status = "FAIL"
			allPass = false
		}
		fmt.Printf("  %s  %.1f tok/s  gpu=%d  %.1fs\n\n", status, res.TokS, res.GPULayers, elapsed.Seconds())
		results = append(results, res)
		os.Remove(tmpJSON)
	}

	fmt.Println("\n╔══════╦═══════════════════════════════════╦═══════════╦══════════════════════════════════════╗")
	fmt.Printf("║ %-4s ║ %-33s ║ %9s ║ %-36s ║\n", "STAT", "Model", "tok/s", "Output Preview")
	fmt.Println("╠══════╬═══════════════════════════════════╬═══════════╬══════════════════════════════════════╣")
	for _, r := range results {
		stat := "PASS"
		detail := ""
		tokStr := "—"
		if r.Err != "" {
			stat = "SKIP"
			detail = r.Err
		} else if !r.Pass {
			stat = "FAIL"
			allPass = false
			detail = "INCOHERENT"
		} else {
			detail = strings.TrimSpace(strings.ReplaceAll(r.Text, "\n", " "))
		}
		if r.TokS > 0 {
			tokStr = fmt.Sprintf("%.1f", r.TokS)
		}
		if len(detail) > 36 {
			detail = detail[:35] + "…"
		}
		fmt.Printf("║ %-4s ║ %-33s ║ %9s ║ %-36s ║\n", stat, r.Name, tokStr, detail)
	}
	fmt.Println("╚══════╩═══════════════════════════════════╩═══════════╩══════════════════════════════════════╝")

	jsonData, _ := json.MarshalIndent(results, "", "  ")
	resultFile := fmt.Sprintf("coherency_results_%s.json", time.Now().Format("20060102_150405"))
	os.WriteFile(resultFile, jsonData, 0644)
	fmt.Printf("\nResults saved to %s\n", resultFile)

	passed := 0
	failed := 0
	skipped := 0
	for _, r := range results {
		if r.Err != "" {
			skipped++
		} else if r.Pass {
			passed++
		} else {
			failed++
		}
	}
	fmt.Printf("\nTotal: %d passed, %d failed, %d skipped\n", passed, failed, skipped)

	if allPass {
		fmt.Println("All models passed coherency test.")
	} else {
		fmt.Println("Some models FAILED. See details above.")
		os.Exit(1)
	}
}
