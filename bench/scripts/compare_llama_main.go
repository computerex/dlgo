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
	"strconv"
	"strings"
	"time"
)

const llamaCLI = `C:\projects\llama.cpp\build\bin\Release\llama-cli.exe`

type modelSpec struct {
	name, path string
}

var models = []modelSpec{
	{"Gemma 3 270M Q8_0", `C:\models\gemma-3-270m-it-Q8_0.gguf`},
	{"SmolLM2 360M Q8_0", `C:\models\smollm2-360m-instruct-q8_0.gguf`},
	{"Qwen 2.5 0.5B Q4_K_M", `C:\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`},
	{"Qwen3 0.6B Q8_0", `C:\models\Qwen3-0.6B-Q8_0.gguf`},
	{"Qwen3.5 0.8B Q8_0", `C:\models\Qwen3.5-0.8B-Q8_0.gguf`},
	{"TinyLlama 1.1B Q4_0", `C:\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`},
	{"Gemma 3 1B Q4_K_M", `C:\models\gemma-3-1b-it-Q4_K_M.gguf`},
	{"Llama 3.2 1B Q4_K_M", `C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf`},
	{"SmolLM2 1.7B Q4_K_M", `C:\models\smollm2-1.7b-instruct-q4_k_m.gguf`},
	{"Gemma 2 2B Q4_K_M", `C:\models\gemma-2-2b-it-Q4_K_M.gguf`},
	{"Gemma 2B Q4_K_M", `C:\models\gemma-2b.Q4_K_M.gguf`},
	{"Phi-2 Q4_K_M", `C:\models\phi-2.Q4_K_M.gguf`},
	{"Phi-4-mini Q3_K_M", `C:\models\Phi-4-mini-instruct-Q3_K_M.gguf`},
	{"Qwen3.5 2B Q4_K_M", `C:\models\Qwen3.5-2B.Q4_K_M.gguf`},
	{"Qwen3.5 9B Q3_K_M", `C:\models\Qwen3.5-9B-Q3_K_M.gguf`},
	{"Qwen3.5 27B Q3_K_M", `C:\models\Qwen3.5-27B-Q3_K_M.gguf`},
	{"Qwen3.5 35B-A3B Q3_K_M", `C:\models\Qwen3.5-35B-A3B-Q3_K_M.gguf`},
	{"Qwen3.6 35B-A3B IQ3_XXS", `C:\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`},
}

const testPrompt = "Explain what a computer is in exactly two sentences."

type CompareResult struct {
	Name         string  `json:"name"`
	Text         string  `json:"text"`
	TokS         float64 `json:"tok_s"`
	PrefillMs    float64 `json:"prefill_ms"`
	GenerateMs   float64 `json:"generate_ms"`
	TotalTokens  int     `json:"total_tokens"`
	PromptTokens int     `json:"prompt_tokens"`
	GPULayers    int     `json:"gpu_layers"`
	Dp4a         bool    `json:"dp4a"`
	Err          string  `json:"err,omitempty"`
}

type LlamaResult struct {
	Text       string  `json:"text"`
	PromptTokS float64 `json:"prompt_tok_s"`
	GenTokS    float64 `json:"gen_tok_s"`
	Err        string  `json:"err,omitempty"`
}

type ModelReport struct {
	Name       string       `json:"name"`
	Dlgo       CompareResult `json:"dlgo"`
	Llama      LlamaResult  `json:"llama"`
	Similarity float64      `json:"similarity"`
}

func runDlgo(workerExe, name, path string, idx int) CompareResult {
	tmpJSON := filepath.Join(os.TempDir(), fmt.Sprintf("compare_dlgo_%d.json", idx))
	defer os.Remove(tmpJSON)

	cmd := exec.Command(workerExe, name, path, testPrompt, tmpJSON)
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		return CompareResult{Name: name, Err: fmt.Sprintf("worker: %v", err)}
	}

	data, err := os.ReadFile(tmpJSON)
	if err != nil {
		return CompareResult{Name: name, Err: fmt.Sprintf("read: %v", err)}
	}

	var res CompareResult
	if err := json.Unmarshal(data, &res); err != nil {
		return CompareResult{Name: name, Err: fmt.Sprintf("parse: %v", err)}
	}
	return res
}

var reTiming = regexp.MustCompile(`Prompt:\s+([\d.]+)\s+t/s\s*\|\s*Generation:\s+([\d.]+)\s+t/s`)

func runLlama(name, path string) LlamaResult {
	cmd := exec.Command(llamaCLI,
		"-m", path,
		"-ngl", "999",
		"--temp", "0",
		"-s", "42",
		"-n", "150",
		"-p", testPrompt,
		"--no-display-prompt",
		"--single-turn",
		"-c", "2048",
	)

	var stdoutBuf, stderrBuf strings.Builder
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf
	err := cmd.Run()
	if err != nil {
		return LlamaResult{Err: fmt.Sprintf("exec: %v\nstderr: %s", err, stderrBuf.String()[:min(200, stderrBuf.Len())])}
	}

	stdout := stdoutBuf.String()
	combined := stdout + "\n" + stderrBuf.String()

	text := extractLlamaText(stdout)
	var promptTokS, genTokS float64
	if m := reTiming.FindStringSubmatch(combined); len(m) >= 3 {
		promptTokS, _ = strconv.ParseFloat(m[1], 64)
		genTokS, _ = strconv.ParseFloat(m[2], 64)
	}

	fmt.Fprintf(os.Stderr, "  llama %s: prompt=%.1f t/s, gen=%.1f t/s\n", name, promptTokS, genTokS)
	return LlamaResult{Text: text, PromptTokS: promptTokS, GenTokS: genTokS}
}

func extractLlamaText(combined string) string {
	lines := strings.Split(combined, "\n")

	// Find the prompt marker and the timing footer.
	promptIdx := -1
	timingIdx := len(lines)
	lastBreakdownIdx := -1
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, ">") && strings.Contains(trimmed, testPrompt[:20]) {
			promptIdx = i
		}
		if strings.Contains(line, "llama_memory_breakdown") {
			lastBreakdownIdx = i
		}
		if strings.Contains(line, "[ Prompt:") || strings.Contains(line, "Exiting...") {
			timingIdx = i
			break
		}
	}

	// Start collecting text after either the last breakdown line or the prompt.
	startIdx := promptIdx + 1
	if lastBreakdownIdx > startIdx {
		startIdx = lastBreakdownIdx + 1
	}
	if startIdx <= 0 {
		return ""
	}

	var textLines []string
	for i := startIdx; i < timingIdx; i++ {
		trimmed := strings.TrimSpace(lines[i])
		if strings.Contains(lines[i], "llama_memory_breakdown") {
			continue
		}
		if trimmed != "" {
			textLines = append(textLines, lines[i])
		}
	}
	return strings.TrimSpace(strings.Join(textLines, "\n"))
}

func wordSimilarity(a, b string) float64 {
	if a == "" || b == "" {
		return 0
	}
	wordsA := strings.Fields(strings.ToLower(a))
	wordsB := strings.Fields(strings.ToLower(b))
	if len(wordsA) == 0 || len(wordsB) == 0 {
		return 0
	}

	setA := make(map[string]bool)
	for _, w := range wordsA {
		setA[w] = true
	}
	setB := make(map[string]bool)
	for _, w := range wordsB {
		setB[w] = true
	}

	intersection := 0
	for w := range setA {
		if setB[w] {
			intersection++
		}
	}

	union := len(setA)
	for w := range setB {
		if !setA[w] {
			union++
		}
	}

	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

func truncate(s string, n int) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	runes := []rune(s)
	if len(runes) > n {
		return string(runes[:n-1]) + "…"
	}
	return s
}

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  dlgo vs llama.cpp — Coherence & Performance Comparison              ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════╝")
	fmt.Printf("Prompt: %q\n", testPrompt)
	fmt.Printf("Max tokens: 150, Temp: 0 (greedy), Seed: 42\n\n")

	workerExe := filepath.Join(os.TempDir(), "compare_llama_worker.exe")

	fmt.Println("Building dlgo worker...")
	buildCmd := exec.Command("go", "build", "-a", "-tags", "cgo vulkan", "-ldflags", "-linkmode internal",
		"-o", workerExe, `bench/scripts/compare_llama_worker_main.go`)
	buildCmd.Dir = `C:\projects\dlgo`
	buildCmd.Stdout = os.Stderr
	buildCmd.Stderr = os.Stderr
	if err := buildCmd.Run(); err != nil {
		fmt.Printf("FATAL: failed to build worker: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Worker built: %s\n\n", workerExe)

	var reports []ModelReport

	for i, m := range models {
		fmt.Printf("═══ [%d/%d] %s ═══\n", i+1, len(models), m.name)

		if _, err := os.Stat(m.path); os.IsNotExist(err) {
			fmt.Printf("  SKIP: file not found\n\n")
			reports = append(reports, ModelReport{
				Name:  m.name,
				Dlgo:  CompareResult{Name: m.name, Err: "not found"},
				Llama: LlamaResult{Err: "not found"},
			})
			continue
		}

		start := time.Now()
		dlgoRes := runDlgo(workerExe, m.name, m.path, i)
		dlgoTime := time.Since(start)

		start = time.Now()
		llamaRes := runLlama(m.name, m.path)
		llamaTime := time.Since(start)

		sim := wordSimilarity(dlgoRes.Text, llamaRes.Text)

		report := ModelReport{
			Name:       m.name,
			Dlgo:       dlgoRes,
			Llama:      llamaRes,
			Similarity: math.Round(sim*1000) / 1000,
		}
		reports = append(reports, report)

		status := "OK"
		if dlgoRes.Err != "" {
			status = "DLGO_ERR"
		}
		if llamaRes.Err != "" {
			status = "LLAMA_ERR"
		}

		fmt.Printf("  %s  dlgo=%.1f tok/s (%.1fs)  llama=%.1f t/s (%.1fs)  similarity=%.1f%%\n\n",
			status, dlgoRes.TokS, dlgoTime.Seconds(), llamaRes.GenTokS, llamaTime.Seconds(), sim*100)
	}

	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════════════════════════════════════════════════")
	fmt.Println("                              PERFORMANCE COMPARISON REPORT")
	fmt.Println("══════════════════════════════════════════════════════════════════════════════════════════════════════════")
	fmt.Println()

	header := fmt.Sprintf("%-30s │ %9s │ %9s │ %6s │ %9s │ %9s │ %5s",
		"Model", "dlgo t/s", "llama t/s", "ratio", "dlgo pp", "llama pp", "sim%")
	fmt.Println(header)
	fmt.Println(strings.Repeat("─", 30) + "─┼─" + strings.Repeat("─", 9) + "─┼─" +
		strings.Repeat("─", 9) + "─┼─" + strings.Repeat("─", 6) + "─┼─" +
		strings.Repeat("─", 9) + "─┼─" + strings.Repeat("─", 9) + "─┼─" +
		strings.Repeat("─", 5))

	for _, r := range reports {
		dlgoTokS := "—"
		llamaTokS := "—"
		ratio := "—"
		dlgoPP := "—"
		llamaPP := "—"
		simStr := "—"

		if r.Dlgo.Err == "" && r.Dlgo.TokS > 0 {
			dlgoTokS = fmt.Sprintf("%.1f", r.Dlgo.TokS)
			if r.Dlgo.PrefillMs > 0 && r.Dlgo.PromptTokens > 0 {
				ppTokS := float64(r.Dlgo.PromptTokens) / (r.Dlgo.PrefillMs / 1000.0)
				dlgoPP = fmt.Sprintf("%.0f", ppTokS)
			}
		}
		if r.Llama.Err == "" && r.Llama.GenTokS > 0 {
			llamaTokS = fmt.Sprintf("%.1f", r.Llama.GenTokS)
			if r.Llama.PromptTokS > 0 {
				llamaPP = fmt.Sprintf("%.0f", r.Llama.PromptTokS)
			}
		}
		if r.Dlgo.Err == "" && r.Llama.Err == "" && r.Dlgo.TokS > 0 && r.Llama.GenTokS > 0 {
			r := r.Dlgo.TokS / r.Llama.GenTokS
			ratio = fmt.Sprintf("%.2fx", r)
		}
		if r.Dlgo.Err == "" && r.Llama.Err == "" {
			simStr = fmt.Sprintf("%.0f%%", r.Similarity*100)
		}

		if r.Dlgo.Err != "" {
			dlgoTokS = "ERR"
		}
		if r.Llama.Err != "" {
			llamaTokS = "ERR"
		}

		fmt.Printf("%-30s │ %9s │ %9s │ %6s │ %9s │ %9s │ %5s\n",
			truncate(r.Name, 30), dlgoTokS, llamaTokS, ratio, dlgoPP, llamaPP, simStr)
	}

	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════════════════════════════════════════════════")
	fmt.Println("                              TEXT OUTPUT COMPARISON")
	fmt.Println("══════════════════════════════════════════════════════════════════════════════════════════════════════════")

	for _, r := range reports {
		if r.Dlgo.Err != "" && r.Llama.Err != "" {
			continue
		}
		fmt.Printf("\n── %s (similarity: %.0f%%) ──\n", r.Name, r.Similarity*100)
		fmt.Printf("  dlgo:  %s\n", truncate(r.Dlgo.Text, 200))
		if r.Dlgo.Err != "" {
			fmt.Printf("  dlgo:  [ERROR: %s]\n", r.Dlgo.Err)
		}
		fmt.Printf("  llama: %s\n", truncate(r.Llama.Text, 200))
		if r.Llama.Err != "" {
			fmt.Printf("  llama: [ERROR: %s]\n", r.Llama.Err)
		}
	}

	jsonData, _ := json.MarshalIndent(reports, "", "  ")
	resultFile := fmt.Sprintf("compare_results_%s.json", time.Now().Format("20060102_150405"))
	os.WriteFile(resultFile, jsonData, 0644)
	fmt.Printf("\n\nFull results saved to %s\n", resultFile)

	// Summary stats
	var dlgoTotal, llamaTotal float64
	var count int
	for _, r := range reports {
		if r.Dlgo.Err == "" && r.Llama.Err == "" && r.Dlgo.TokS > 0 && r.Llama.GenTokS > 0 {
			dlgoTotal += r.Dlgo.TokS
			llamaTotal += r.Llama.GenTokS
			count++
		}
	}
	if count > 0 {
		fmt.Printf("\nAverage generation speed (%d models):\n", count)
		fmt.Printf("  dlgo:      %.1f tok/s\n", dlgoTotal/float64(count))
		fmt.Printf("  llama.cpp: %.1f tok/s\n", llamaTotal/float64(count))
		fmt.Printf("  ratio:     %.2fx\n", (dlgoTotal/float64(count))/(llamaTotal/float64(count)))
	}
}
