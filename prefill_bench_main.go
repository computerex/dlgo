//go:build ignore

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

type OllamaReq struct {
	Model   string      `json:"model"`
	Msgs    []OMsg      `json:"messages"`
	Stream  bool        `json:"stream"`
	Options OOpts       `json:"options"`
}
type OMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type OOpts struct {
	Temperature float64 `json:"temperature"`
	NumPredict  int     `json:"num_predict"`
	Seed        int     `json:"seed"`
	NumGPU      int     `json:"num_gpu"`
}
type OResp struct {
	Message            OMsg          `json:"message"`
	Done               bool          `json:"done"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

type pair struct {
	label, ggufPath, ollamaName string
}

var models = []pair{
	{"SmolLM2 360M Q8_0", `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`, "dlgo-smollm2-360m"},
	{"TinyLlama 1.1B Q4_0", `C:\projects\evoke\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`, "dlgo-tinyllama"},
	{"SmolLM2 1.7B Q4_K_M", `C:\projects\evoke\models\smollm2-1.7b-instruct-q4_k_m.gguf`, "dlgo-smollm2-1.7b"},
}

var prompts = []struct {
	label  string
	text   string
}{
	{"short", "What is 2+2?"},
	{"medium", "Write a detailed explanation of how photosynthesis works in plants, including the light-dependent and light-independent reactions. Explain the role of chlorophyll."},
	{"long", strings.Repeat("The quick brown fox jumps over the lazy dog. ", 20) + "Summarize the above."},
}

func ollamaPrefill(model, prompt string, numGPU int) (prefillMs float64, prefillTok int, genTps float64, err error) {
	req := OllamaReq{
		Model:  model,
		Msgs:   []OMsg{{Role: "user", Content: prompt}},
		Stream: false,
		Options: OOpts{Temperature: 0, NumPredict: 1, Seed: 42, NumGPU: numGPU},
	}
	b, _ := json.Marshal(req)
	resp, err := http.Post("http://localhost:11434/api/chat", "application/json", bytes.NewBuffer(b))
	if err != nil {
		return 0, 0, 0, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var r OResp
	if err := json.Unmarshal(raw, &r); err != nil {
		return 0, 0, 0, fmt.Errorf("json: %v", err)
	}
	pms := float64(r.PromptEvalDuration) / 1e6
	tps := 0.0
	if r.EvalDuration > 0 {
		tps = float64(r.EvalCount) / r.EvalDuration.Seconds()
	}
	return pms, r.PromptEvalCount, tps, nil
}

func dlgoPrefillCPU(pipe *llm.Pipeline, prompt string) (prefillMs float64, prefillTok int, genTps float64) {
	formatted := llm.FormatChat(pipe.Model.Config, "", prompt)
	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = 1
	cfg.Seed = 42
	cfg.Sampler.Temperature = 0
	best := math.MaxFloat64
	var bestTok int
	var bestTps float64
	for i := 0; i < 3; i++ {
		r, err := pipe.GenerateDetailed(formatted, cfg)
		if err != nil {
			continue
		}
		if r.PrefillTimeMs < best {
			best = r.PrefillTimeMs
			bestTok = r.PromptTokens
			bestTps = r.TokensPerSec
		}
	}
	if best == math.MaxFloat64 {
		return 0, 0, 0
	}
	return best, bestTok, bestTps
}

func dlgoPrefillGPU(gpuPipe *gpu.GpuPipeline, cpuModel *llm.Model, prompt string) (prefillMs float64, prefillTok int, genTps float64) {
	formatted := llm.FormatChat(cpuModel.Config, "", prompt)
	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = 1
	cfg.Seed = 42
	cfg.Sampler.Temperature = 0
	best := math.MaxFloat64
	var bestTok int
	var bestTps float64
	for i := 0; i < 3; i++ {
		r, err := gpuPipe.GenerateDetailed(formatted, cfg)
		if err != nil {
			continue
		}
		if r.PrefillTimeMs < best {
			best = r.PrefillTimeMs
			bestTok = r.PromptTokens
			bestTps = r.TokensPerSec
		}
	}
	if best == math.MaxFloat64 {
		return 0, 0, 0
	}
	return best, bestTok, bestTps
}

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           dlgo vs Ollama — Prefill + Generation Benchmark            ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════╝")

	if err := gpu.Init(); err != nil {
		fmt.Printf("GPU init failed: %v\n", err)
		return
	}
	defer gpu.Shutdown()
	fmt.Printf("GPU: %s\n\n", gpu.DeviceName())

	type row struct {
		model, promptLen                          string
		promptTok                                 int
		ollamaCPUPrefill, dlgoCPUPrefill          float64
		ollamaGPUPrefill, dlgoGPUPrefill          float64
		ollamaCPUGenTps, dlgoCPUGenTps            float64
		ollamaGPUGenTps, dlgoGPUGenTps            float64
	}
	var rows []row

	for _, m := range models {
		fmt.Printf("═══ %s ═══\n", m.label)

		cpuPipe, err := llm.NewPipeline(m.ggufPath, 512)
		if err != nil {
			fmt.Printf("  SKIP: %v\n\n", err)
			continue
		}

		gpuPipe, gpuErr := gpu.NewGpuPipeline(cpuPipe)
		if gpuErr != nil {
			fmt.Printf("  GPU pipeline fail: %v\n", gpuErr)
		}

		// Warmup: run one prompt through both CPU and GPU to prime caches
		{
			warmPrompt := llm.FormatChat(cpuPipe.Model.Config, "", "Hello")
			wCfg := llm.DefaultGenerateConfig()
			wCfg.MaxTokens = 1
			wCfg.Seed = 42
			wCfg.Sampler.Temperature = 0
			cpuPipe.GenerateDetailed(warmPrompt, wCfg)
			if gpuPipe != nil {
				gpuPipe.GenerateDetailed(warmPrompt, wCfg)
			}
		}

		for _, p := range prompts {
			r := row{model: m.label, promptLen: p.label}

			// Ollama CPU (num_gpu=0)
			ocp, oct, ocg, err := ollamaPrefill(m.ollamaName, p.text, 0)
			if err != nil {
				fmt.Printf("  %s Ollama CPU: FAIL %v\n", p.label, err)
			} else {
				r.ollamaCPUPrefill = ocp
				r.promptTok = oct
				r.ollamaCPUGenTps = ocg
			}

			// dlgo CPU
			dcp, dct, dcg := dlgoPrefillCPU(cpuPipe, p.text)
			r.dlgoCPUPrefill = dcp
			if r.promptTok == 0 {
				r.promptTok = dct
			}
			r.dlgoCPUGenTps = dcg

			// Ollama GPU (num_gpu=99)
			ogp, _, ogg, err := ollamaPrefill(m.ollamaName, p.text, 99)
			if err != nil {
				fmt.Printf("  %s Ollama GPU: FAIL %v\n", p.label, err)
			} else {
				r.ollamaGPUPrefill = ogp
				r.ollamaGPUGenTps = ogg
			}

			// dlgo GPU
			if gpuPipe != nil {
				dgp, _, dgg := dlgoPrefillGPU(gpuPipe, cpuPipe.Model, p.text)
				r.dlgoGPUPrefill = dgp
				r.dlgoGPUGenTps = dgg
			}

			cpuDelta := ""
			if r.ollamaCPUPrefill > 0 && r.dlgoCPUPrefill > 0 {
				d := (r.dlgoCPUPrefill - r.ollamaCPUPrefill) / r.ollamaCPUPrefill * 100
				cpuDelta = fmt.Sprintf("%+.0f%%", d)
			}
			gpuDelta := ""
			if r.ollamaGPUPrefill > 0 && r.dlgoGPUPrefill > 0 {
				d := (r.dlgoGPUPrefill - r.ollamaGPUPrefill) / r.ollamaGPUPrefill * 100
				gpuDelta = fmt.Sprintf("%+.0f%%", d)
			}

			fmt.Printf("  %-6s [%3d tok]  CPU: ollama=%6.1fms  dlgo=%6.1fms  %6s  |  GPU: ollama=%6.1fms  dlgo=%6.1fms  %6s\n",
				p.label, r.promptTok, r.ollamaCPUPrefill, r.dlgoCPUPrefill, cpuDelta,
				r.ollamaGPUPrefill, r.dlgoGPUPrefill, gpuDelta)

			rows = append(rows, r)
		}
		fmt.Println()
	}

	// Summary table
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                           PREFILL BENCHMARK SUMMARY (lower ms = better)                              ║")
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ %-28s %5s %8s %8s %7s │ %8s %8s %7s ║\n",
		"Model / Prompt", "Tok", "OlCPU", "dlCPU", "Δ", "OlGPU", "dlGPU", "Δ")
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	for _, r := range rows {
		cpuD := ""
		if r.ollamaCPUPrefill > 0 && r.dlgoCPUPrefill > 0 {
			cpuD = fmt.Sprintf("%+.0f%%", (r.dlgoCPUPrefill-r.ollamaCPUPrefill)/r.ollamaCPUPrefill*100)
		}
		gpuD := ""
		if r.ollamaGPUPrefill > 0 && r.dlgoGPUPrefill > 0 {
			gpuD = fmt.Sprintf("%+.0f%%", (r.dlgoGPUPrefill-r.ollamaGPUPrefill)/r.ollamaGPUPrefill*100)
		}
		label := fmt.Sprintf("%s / %s", r.model, r.promptLen)
		if len(label) > 28 {
			label = label[:28]
		}
		fmt.Printf("║ %-28s %5d %7.1fms %7.1fms %6s │ %7.1fms %7.1fms %6s ║\n",
			label, r.promptTok, r.ollamaCPUPrefill, r.dlgoCPUPrefill, cpuD,
			r.ollamaGPUPrefill, r.dlgoGPUPrefill, gpuD)
	}
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Generation comparison
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           GENERATION SPEED (tok/s, higher = better)                 ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ %-28s %8s %8s │ %8s %8s ║\n", "Model / Prompt", "OlCPU", "dlCPU", "OlGPU", "dlGPU")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════╣")
	for _, r := range rows {
		label := fmt.Sprintf("%s / %s", r.model, r.promptLen)
		if len(label) > 28 {
			label = label[:28]
		}
		fmt.Printf("║ %-28s %7.1f  %7.1f  │ %7.1f  %7.1f  ║\n",
			label, r.ollamaCPUGenTps, r.dlgoCPUGenTps, r.ollamaGPUGenTps, r.dlgoGPUGenTps)
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
}
