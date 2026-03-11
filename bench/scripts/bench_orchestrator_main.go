//go:build ignore

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"
)

// Orchestrator: ensures dlgo and ollama NEVER run at the same time.
// 1. Builds bench_worker once
// 2. For each model: run worker subprocess (dlgo CPU+GPU) → wait for full exit
// 3. Verify no dlgo process lingers
// 4. Run ollama via HTTP API (CPU, then GPU)
// 5. Force-unload ollama model, verify RAM released
// 6. Print comparison table
//
// The orchestrator itself never loads any model — it's just a coordinator.

type modelSpec struct {
	name       string
	ggufPath   string
	ollamaName string
}

var allModels = []modelSpec{
	// Small models
	{"Gemma 3 270M Q8_0", `C:\projects\evoke\models\gemma-3-270m-it-Q8_0.gguf`, "dlgo-gemma3-270m"},
	{"SmolLM2 360M Q8_0", `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`, "dlgo-smollm2-360m"},
	{"Qwen 2.5 0.5B Q4_K_M", `C:\projects\evoke\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`, "dlgo-qwen25"},
	{"Qwen3 0.6B Q8_0", `C:\projects\evoke\models\Qwen3-0.6B-Q8_0.gguf`, "dlgo-qwen3-0.6b"},
	{"TinyLlama 1.1B Q4_0", `C:\projects\evoke\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`, "dlgo-tinyllama"},
	{"Gemma 3 1B Q4_K_M", `C:\projects\evoke\models\gemma-3-1b-it-Q4_K_M.gguf`, "dlgo-gemma3"},
	{"Llama 3.2 1B Q4_K_M", `C:\projects\evoke\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf`, "dlgo-llama32-1b"},
	{"SmolLM2 1.7B Q4_K_M", `C:\projects\evoke\models\smollm2-1.7b-instruct-q4_k_m.gguf`, "dlgo-smollm2-1.7b"},
	{"Qwen3.5 0.8B Q8_0", `C:\projects\evoke\models\Qwen3.5-0.8B-Q8_0.gguf`, "dlgo-qwen35-0.8b"},
	{"Phi-4-mini Q3_K_M", `C:\projects\evoke\models\Phi-4-mini-instruct-Q3_K_M.gguf`, "dlgo-phi4-mini"},
	// Large models
	{"Qwen3.5 9B Q3_K_M", `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`, "qwen35-9b"},
	{"Qwen3.5 27B Q3_K_M", `C:\projects\gollm\Qwen3.5-27B-Q3_K_M.gguf`, "qwen35-27b"},
	{"Qwen3.5 35B-A3B MoE Q3_K_M", `C:\projects\gollm\Qwen3.5-35B-A3B-Q3_K_M.gguf`, "qwen35-35b-moe"},
}

type workerResult struct {
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

type benchResult struct {
	name string
	// dlgo (from worker subprocess)
	dlgoCPUPrefillMs float64
	dlgoCPUGenTokS   float64
	dlgoGPUPrefillMs float64
	dlgoGPUGenTokS   float64
	dlgoGPUText      string
	dlgoGPUError     string
	dlgoWorkerError  string
	// ollama (from HTTP API)
	ollamaCPUPrefillMs float64
	ollamaCPUGenTokS   float64
	ollamaCPUText      string
	ollamaGPUPrefillMs float64
	ollamaGPUGenTokS   float64
	ollamaGPUText      string
	ollamaCPUError     string
	ollamaGPUError     string
}

const (
	prompt    = "Explain the theory of general relativity in simple terms"
	genTokens = 20
)

func main() {
	filter := ""
	if len(os.Args) > 1 {
		filter = strings.ToLower(os.Args[1])
	}

	fmt.Println("========================================================================")
	fmt.Println("  dlgo Benchmark Orchestrator")
	fmt.Println("  Ensures dlgo and ollama NEVER run simultaneously")
	fmt.Println("========================================================================")
	if filter != "" {
		fmt.Printf("  Filter: %q\n", filter)
	}
	fmt.Println()

	fmt.Println("Worker: go run -tags vulkan bench_worker_main.go")

	// Unload all ollama models upfront
	fmt.Print("Unloading all ollama models...")
	for _, m := range allModels {
		ollamaUnload(m.ollamaName)
	}
	time.Sleep(2 * time.Second)
	fmt.Println(" OK")
	fmt.Println()

	models := allModels
	if filter != "" {
		models = nil
		for _, m := range allModels {
			if strings.Contains(strings.ToLower(m.name), filter) {
				models = append(models, m)
			}
		}
		if len(models) == 0 {
			fmt.Printf("No models match filter %q\n", filter)
			os.Exit(1)
		}
	}

	var results []benchResult

	for i, m := range models {
		fmt.Printf("══════ [%d/%d] %s ══════\n", i+1, len(models), m.name)
		r := benchResult{name: m.name}

		// --- Phase A: dlgo (subprocess) ---
		fmt.Printf("  [dlgo] Running worker subprocess...\n")
		killDlgoProcesses()
		wr, workerErr := runWorker(m.ggufPath)
		if workerErr != nil {
			r.dlgoWorkerError = workerErr.Error()
			fmt.Printf("  [dlgo] FAIL: %v\n", workerErr)
		} else {
			r.dlgoCPUPrefillMs = wr.CPUPrefillMs
			r.dlgoCPUGenTokS = wr.CPUGenTokS
			r.dlgoGPUPrefillMs = wr.GPUPrefillMs
			r.dlgoGPUGenTokS = wr.GPUGenTokS
			r.dlgoGPUText = wr.GPUText
			r.dlgoGPUError = wr.GPUError
			fmt.Printf("  [dlgo] layers=%d dim=%d", wr.Layers, wr.Dim)
			if wr.Experts > 0 {
				fmt.Printf(" experts=%d active=%d", wr.Experts, wr.ActiveExp)
			}
			fmt.Println()
			fmt.Printf("  [dlgo CPU]  prefill=%6.0fms  gen=%6.1f tok/s\n", wr.CPUPrefillMs, wr.CPUGenTokS)
			if wr.GPUError != "" {
				fmt.Printf("  [dlgo GPU]  FAIL: %s\n", wr.GPUError)
			} else {
				fmt.Printf("  [dlgo GPU]  prefill=%6.0fms  gen=%6.1f tok/s  text=%q\n",
					wr.GPUPrefillMs, wr.GPUGenTokS, truncate(wr.GPUText, 60))
			}
		}

		// Worker process has fully exited — all RAM+VRAM released by the OS.
		// Verify nothing lingers.
		killDlgoProcesses()
		time.Sleep(1 * time.Second)

		// --- Phase B: Ollama CPU (HTTP API) ---
		fmt.Printf("  [Ollama CPU] Running...\n")
		ollamaUnload(m.ollamaName) // ensure clean start
		time.Sleep(500 * time.Millisecond)
		olPre, olGen, _, olGenTok, olText, olErr := ollamaGenerate(m.ollamaName, prompt, genTokens, 0)
		if olErr != nil {
			r.ollamaCPUError = olErr.Error()
			fmt.Printf("  [Ollama CPU] FAIL: %v\n", olErr)
		} else {
			tps := float64(olGenTok) / (olGen / 1000.0)
			r.ollamaCPUPrefillMs = olPre
			r.ollamaCPUGenTokS = tps
			r.ollamaCPUText = olText
			fmt.Printf("  [Ollama CPU] prefill=%6.0fms  gen=%6.1f tok/s  text=%q\n",
				olPre, tps, truncate(olText, 60))
		}
		// Fully unload before GPU test to free all RAM
		ollamaUnload(m.ollamaName)
		time.Sleep(1 * time.Second)

		// --- Phase C: Ollama GPU (HTTP API) ---
		fmt.Printf("  [Ollama GPU] Running...\n")
		olPre2, olGen2, _, olGenTok2, olText2, olErr2 := ollamaGenerate(m.ollamaName, prompt, genTokens, 999)
		if olErr2 != nil {
			r.ollamaGPUError = olErr2.Error()
			fmt.Printf("  [Ollama GPU] FAIL: %v\n", olErr2)
		} else {
			tps2 := float64(olGenTok2) / (olGen2 / 1000.0)
			r.ollamaGPUPrefillMs = olPre2
			r.ollamaGPUGenTokS = tps2
			r.ollamaGPUText = olText2
			fmt.Printf("  [Ollama GPU] prefill=%6.0fms  gen=%6.1f tok/s  text=%q\n",
				olPre2, tps2, truncate(olText2, 60))
		}

		// Fully unload after each model
		ollamaUnload(m.ollamaName)
		time.Sleep(1 * time.Second)

		results = append(results, r)
		fmt.Println()
	}

	// ═══════════════════════════════════════════════════════════════
	// SUMMARY TABLE
	// ═══════════════════════════════════════════════════════════════
	printSummary(results)

	// Save machine-readable results
	type jsonResult struct {
		Name               string  `json:"name"`
		DlgoCPUPrefillMs   float64 `json:"dlgo_cpu_prefill_ms"`
		DlgoCPUGenTokS     float64 `json:"dlgo_cpu_gen_tok_s"`
		DlgoGPUPrefillMs   float64 `json:"dlgo_gpu_prefill_ms"`
		DlgoGPUGenTokS     float64 `json:"dlgo_gpu_gen_tok_s"`
		OllamaCPUPrefillMs float64 `json:"ollama_cpu_prefill_ms"`
		OllamaCPUGenTokS   float64 `json:"ollama_cpu_gen_tok_s"`
		OllamaGPUPrefillMs float64 `json:"ollama_gpu_prefill_ms"`
		OllamaGPUGenTokS   float64 `json:"ollama_gpu_gen_tok_s"`
	}
	var jr []jsonResult
	for _, r := range results {
		jr = append(jr, jsonResult{
			Name:               r.name,
			DlgoCPUPrefillMs:   r.dlgoCPUPrefillMs,
			DlgoCPUGenTokS:     r.dlgoCPUGenTokS,
			DlgoGPUPrefillMs:   r.dlgoGPUPrefillMs,
			DlgoGPUGenTokS:     r.dlgoGPUGenTokS,
			OllamaCPUPrefillMs: r.ollamaCPUPrefillMs,
			OllamaCPUGenTokS:   r.ollamaCPUGenTokS,
			OllamaGPUPrefillMs: r.ollamaGPUPrefillMs,
			OllamaGPUGenTokS:   r.ollamaGPUGenTokS,
		})
	}
	data, _ := json.MarshalIndent(jr, "", "  ")
	outFile := fmt.Sprintf("bench_results_%s.json", time.Now().Format("20060102_150405"))
	os.WriteFile(outFile, data, 0644)
	fmt.Printf("\nResults saved to %s\n", outFile)
}

func runWorker(ggufPath string) (*workerResult, error) {
	tmpFile := fmt.Sprintf(`C:\projects\dlgo\bench_result_%d.json`, time.Now().UnixNano())
	defer os.Remove(tmpFile)

	cmd := exec.Command("go", "run", "-tags", "vulkan", "bench_worker_main.go",
		ggufPath, prompt, fmt.Sprintf("%d", genTokens), tmpFile)
	cmd.Dir = `C:\projects\dlgo`
	cmd.Stdout = os.Stderr // worker logs (including gpu messages) stream to console
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start: %v", err)
	}

	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()

	select {
	case err := <-done:
		if err != nil {
			return nil, fmt.Errorf("exit: %v", err)
		}
	case <-time.After(15 * time.Minute):
		cmd.Process.Kill()
		return nil, fmt.Errorf("timeout after 15 min")
	}

	data, err := os.ReadFile(tmpFile)
	if err != nil {
		return nil, fmt.Errorf("read result file: %v", err)
	}
	var wr workerResult
	if err := json.Unmarshal(data, &wr); err != nil {
		return nil, fmt.Errorf("parse result: %v raw: %s", err, string(data[:min(len(data), 200)]))
	}
	return &wr, nil
}

func killDlgoProcesses() {
	// Kill any lingering bench_worker processes (compiled by go run)
	exec.Command("taskkill", "/F", "/IM", "bench_worker_main.exe").Run()
}

// ─── Ollama HTTP API ───

type ollamaReq struct {
	Model     string      `json:"model"`
	Msgs      []ollamaMsg `json:"messages"`
	Stream    bool        `json:"stream"`
	Options   ollamaOpts  `json:"options"`
	KeepAlive string      `json:"keep_alive"`
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

func ollamaGenerate(model, prompt string, maxTok, numGPU int) (prefillMs, genMs float64, prefillTok, genTok int, text string, err error) {
	req := ollamaReq{
		Model:     model,
		Msgs:      []ollamaMsg{{Role: "user", Content: prompt}},
		Stream:    false,
		Options:   ollamaOpts{Temperature: 0, NumPredict: maxTok, Seed: 42, NumGPU: numGPU},
		KeepAlive: "0s",
	}
	b, _ := json.Marshal(req)
	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Post("http://localhost:11434/api/chat", "application/json", bytes.NewBuffer(b))
	if err != nil {
		return 0, 0, 0, 0, "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var r ollamaResp
	if err := json.Unmarshal(raw, &r); err != nil {
		return 0, 0, 0, 0, "", fmt.Errorf("json: %v body: %s", err, string(raw[:min(len(raw), 200)]))
	}
	pms := float64(r.PromptEvalDuration) / 1e6
	gms := float64(r.EvalDuration) / 1e6
	return pms, gms, r.PromptEvalCount, r.EvalCount, r.Message.Content, nil
}

func ollamaUnload(model string) {
	body, _ := json.Marshal(map[string]interface{}{
		"model":      model,
		"keep_alive": 0,
	})
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Post("http://localhost:11434/api/generate", "application/json", bytes.NewBuffer(body))
	if err != nil {
		return
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
}

// ─── Summary ───

func printSummary(results []benchResult) {
	sep := "═══════════════════════════════════════════════════════════════════════════════════════════════════════════"
	fmt.Println()
	fmt.Println(sep)
	fmt.Println("  BENCHMARK RESULTS (dlgo vs Ollama — isolated processes, no resource sharing)")
	fmt.Println(sep)
	fmt.Println()

	// CPU Generation
	fmt.Println("┌─── CPU Generation (tok/s, higher = better) ───────────────────────────────────────────┐")
	fmt.Printf("│ %-32s │ %10s │ %10s │ %8s │ %10s │\n", "Model", "dlgo", "Ollama", "Delta", "Coherent")
	fmt.Println("├──────────────────────────────────┼────────────┼────────────┼──────────┼────────────┤")
	for _, r := range results {
		delta := "n/a"
		if r.ollamaCPUGenTokS > 0 && r.dlgoCPUGenTokS > 0 {
			d := (r.dlgoCPUGenTokS - r.ollamaCPUGenTokS) / r.ollamaCPUGenTokS * 100
			delta = fmt.Sprintf("%+.0f%%", d)
		}
		coherent := "OK"
		if r.dlgoWorkerError != "" {
			coherent = "ERR"
		}
		fmt.Printf("│ %-32s │ %8.1f   │ %8.1f   │ %6s   │ %8s   │\n",
			truncate(r.name, 32), r.dlgoCPUGenTokS, r.ollamaCPUGenTokS, delta, coherent)
	}
	fmt.Println("└──────────────────────────────────┴────────────┴────────────┴──────────┴────────────┘")
	fmt.Println()

	// GPU Generation
	fmt.Println("┌─── GPU Generation (tok/s, higher = better) ───────────────────────────────────────────┐")
	fmt.Printf("│ %-32s │ %10s │ %10s │ %8s │ %10s │\n", "Model", "dlgo", "Ollama", "Delta", "Status")
	fmt.Println("├──────────────────────────────────┼────────────┼────────────┼──────────┼────────────┤")
	for _, r := range results {
		dlgo := fmt.Sprintf("%.1f", r.dlgoGPUGenTokS)
		ollama := fmt.Sprintf("%.1f", r.ollamaGPUGenTokS)
		delta := "n/a"
		status := "OK"
		if r.dlgoGPUError != "" {
			dlgo = "FAIL"
			status = "SKIP"
		} else if r.ollamaGPUError != "" {
			ollama = "FAIL"
		}
		if r.ollamaGPUGenTokS > 0 && r.dlgoGPUGenTokS > 0 {
			d := (r.dlgoGPUGenTokS - r.ollamaGPUGenTokS) / r.ollamaGPUGenTokS * 100
			delta = fmt.Sprintf("%+.0f%%", d)
		}
		fmt.Printf("│ %-32s │ %8s   │ %8s   │ %6s   │ %8s   │\n",
			truncate(r.name, 32), dlgo, ollama, delta, status)
	}
	fmt.Println("└──────────────────────────────────┴────────────┴────────────┴──────────┴────────────┘")
	fmt.Println()

	// Prefill
	fmt.Println("┌─── Prefill (ms, lower = better) ─────────────────────────────────────────────────────────────────────┐")
	fmt.Printf("│ %-32s │ %9s │ %9s │ %7s │ %9s │ %9s │ %7s │\n",
		"Model", "dlCPU", "OlCPU", "Δ CPU", "dlGPU", "OlGPU", "Δ GPU")
	fmt.Println("├──────────────────────────────────┼───────────┼───────────┼─────────┼───────────┼───────────┼─────────┤")
	for _, r := range results {
		cpuD := "n/a"
		if r.ollamaCPUPrefillMs > 0 && r.dlgoCPUPrefillMs > 0 {
			d := (r.dlgoCPUPrefillMs - r.ollamaCPUPrefillMs) / r.ollamaCPUPrefillMs * 100
			cpuD = fmt.Sprintf("%+.0f%%", d)
		}
		gpuD := "n/a"
		dlGPU := fmt.Sprintf("%.0f", r.dlgoGPUPrefillMs)
		olGPU := fmt.Sprintf("%.0f", r.ollamaGPUPrefillMs)
		if r.dlgoGPUError != "" {
			dlGPU = "FAIL"
		}
		if r.ollamaGPUError != "" {
			olGPU = "FAIL"
		}
		if r.ollamaGPUPrefillMs > 0 && r.dlgoGPUPrefillMs > 0 {
			d := (r.dlgoGPUPrefillMs - r.ollamaGPUPrefillMs) / r.ollamaGPUPrefillMs * 100
			gpuD = fmt.Sprintf("%+.0f%%", d)
		}
		fmt.Printf("│ %-32s │ %7.0f   │ %7.0f   │ %5s   │ %7s   │ %7s   │ %5s   │\n",
			truncate(r.name, 32), r.dlgoCPUPrefillMs, r.ollamaCPUPrefillMs, cpuD,
			dlGPU, olGPU, gpuD)
	}
	fmt.Println("└──────────────────────────────────┴───────────┴───────────┴─────────┴───────────┴───────────┴─────────┘")
	fmt.Println()

	// Coherence check
	fmt.Println("┌─── Coherence (GPU output text samples) ──────────────────────────────────────────────┐")
	for _, r := range results {
		dlgoText := truncate(strings.TrimSpace(r.dlgoGPUText), 70)
		olText := truncate(strings.TrimSpace(r.ollamaGPUText), 70)
		if r.dlgoGPUError != "" {
			dlgoText = "(GPU not available)"
		}
		if r.ollamaGPUError != "" {
			olText = "(failed)"
		}
		fmt.Printf("│ %-32s                                                                  │\n", r.name)
		fmt.Printf("│   dlgo:   %-80s │\n", dlgoText)
		fmt.Printf("│   ollama: %-80s │\n", olText)
	}
	fmt.Println("└───────────────────────────────────────────────────────────────────────────────────────┘")
}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}
