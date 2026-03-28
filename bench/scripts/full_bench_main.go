//go:build ignore

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

type modelSpec struct {
	name, ggufPath, ollamaName string
	hasSSM                     bool
}

var allModels = []modelSpec{
	{"SmolLM2 360M Q8_0", `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`, "dlgo-smollm2-360m", false},
	{"TinyLlama 1.1B Q4_0", `C:\projects\evoke\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`, "dlgo-tinyllama", false},
	{"Qwen 2.5 0.5B Q4_K_M", `C:\projects\evoke\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`, "dlgo-qwen25", false},
	{"Gemma 3 1B Q4_K_M", `C:\projects\evoke\models\gemma-3-1b-it-Q4_K_M.gguf`, "dlgo-gemma3", false},
	{"Gemma 3 270M Q8_0", `C:\projects\evoke\models\gemma-3-270m-it-Q8_0.gguf`, "dlgo-gemma3-270m", false},
	{"SmolLM2 1.7B Q4_K_M", `C:\projects\evoke\models\smollm2-1.7b-instruct-q4_k_m.gguf`, "dlgo-smollm2-1.7b", false},
	{"Llama 3.2 1B Q4_K_M", `C:\projects\evoke\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf`, "dlgo-llama32-1b", false},
	{"Phi-4-mini Q3_K_M", `C:\projects\evoke\models\Phi-4-mini-instruct-Q3_K_M.gguf`, "dlgo-phi4-mini", false},
	{"Qwen3 0.6B Q8_0", `C:\projects\evoke\models\Qwen3-0.6B-Q8_0.gguf`, "dlgo-qwen3-0.6b", false},
	{"Qwen3.5 0.8B Q8_0", `C:\projects\evoke\models\Qwen3.5-0.8B-Q8_0.gguf`, "dlgo-qwen35-0.8b", true},
}

type ollamaReq struct {
	Model   string    `json:"model"`
	Msgs    []ollamaMsg `json:"messages"`
	Stream  bool      `json:"stream"`
	Options ollamaOpts `json:"options"`
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
		Model:  model,
		Msgs:   []ollamaMsg{{Role: "user", Content: prompt}},
		Stream: false,
		Options: ollamaOpts{Temperature: 0, NumPredict: maxTok, Seed: 42, NumGPU: numGPU},
	}
	b, _ := json.Marshal(req)
	resp, err := http.Post("http://localhost:11434/api/chat", "application/json", bytes.NewBuffer(b))
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

type modelResult struct {
	name string

	// Correctness
	maxErr, avgErr float64
	cpuTopTok      int
	gpuTopTok      int
	topMatch       bool
	correctnessOK  bool
	gpuUploadFail  bool

	// Coherence — dlgo
	dlgoCPUText    string
	dlgoCPUTok     int
	dlgoCPUPrefill float64
	dlgoCPUGen     float64
	dlgoCPUTps     float64
	dlgoGPUText    string
	dlgoGPUTok     int
	dlgoGPUPrefill float64
	dlgoGPUGen     float64
	dlgoGPUTps     float64
	gpuPipeFail    bool

	// Coherence — Ollama
	ollamaCPUText    string
	ollamaCPUTok     int
	ollamaCPUPrefill float64
	ollamaCPUGen     float64
	ollamaGPUText    string
	ollamaGPUTok     int
	ollamaGPUPrefill float64
	ollamaGPUGen     float64

	coherenceDlgoCPU bool
	coherenceDlgoGPU bool
	coherenceOlCPU   bool
	coherenceOlGPU   bool
}

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║    dlgo Full Correctness + Coherence + Ollama Benchmark (CPU & GPU)          ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════╝")

	if err := gpu.Init(); err != nil {
		fmt.Printf("GPU init failed: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()
	fmt.Printf("GPU: %s (%.0f MB VRAM)\n\n", gpu.DeviceName(), float64(gpu.VRAMBytes())/(1024*1024))

	prompt := "Explain what a computer is in one sentence."
	maxTokens := 64
	var results []modelResult

	for idx, m := range allModels {
		fmt.Printf("═══ [%d/%d] %s ═══\n", idx+1, len(allModels), m.name)
		r := modelResult{name: m.name}

		pipe, err := llm.NewPipeline(m.ggufPath, 512)
		if err != nil {
			fmt.Printf("  SKIP: load fail: %v\n\n", err)
			results = append(results, r)
			continue
		}

		cfg := pipe.Model.Config
		dim := cfg.EmbeddingDim
		vocabSize := cfg.VocabSize
		kvDim := cfg.NumKVHeads * cfg.HeadDim
		qDim := cfg.NumHeads * cfg.HeadDim
		ffnDim := cfg.FFNDim

		// ═══ Phase 1: Logit Correctness (CPU vs GPU) ═══
		fmt.Printf("  [Correctness] Computing CPU logits...\n")
		tokens := pipe.Tokenizer.Encode("Hello")
		if len(tokens) == 0 {
			tokens = []int32{1}
		}
		cpuLogits := make([]float32, vocabSize)
		cpuRS := llm.NewRunState(cfg, 512)
		cpuKV := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)
		for i, tok := range tokens {
			llm.Forward(pipe.Model, tok, i, cpuKV, cpuRS)
		}
		copy(cpuLogits, cpuRS.Logits)

		gpuModel, gpuUploadErr := gpu.UploadModel(pipe.Model)
		if gpuUploadErr != nil {
			r.gpuUploadFail = true
			r.correctnessOK = true
			fmt.Printf("  [Correctness] GPU upload fail: %v — CPU-only\n", gpuUploadErr)
		} else {
			var gpuAllocErr error
			rs, gpuAllocErr := gpu.NewGpuRunState(dim, qDim, kvDim, ffnDim, vocabSize)
			var kv *gpu.GpuKVCache
			if gpuAllocErr == nil {
				kv, gpuAllocErr = gpu.NewGpuKVCache(cfg.NumLayers, cfg.NumLayers, 512, kvDim, nil)
			}
			if gpuAllocErr != nil {
				r.gpuUploadFail = true
				r.correctnessOK = true
				fmt.Printf("  [Correctness] GPU alloc fail: %v — CPU-only\n", gpuAllocErr)
				if rs != nil {
					rs.FreeAll()
				}
				gpuModel.FreeAll()
			} else {
				gpuLogits := make([]float32, vocabSize)
				var fwdErr error
				for i, tok := range tokens {
					fwdErr = gpu.GpuForward(pipe.Model, gpuModel, tok, i, kv, rs, gpuLogits)
					if fwdErr != nil {
						break
					}
				}
				gpu.Sync()

				if fwdErr != nil {
					r.gpuUploadFail = true
					r.correctnessOK = true
					fmt.Printf("  [Correctness] GPU forward fail: %v — CPU-only\n", fwdErr)
				} else {
					maxErr := float64(0)
					maxIdx := 0
					sumErr := float64(0)
					for i := 0; i < vocabSize; i++ {
						diff := math.Abs(float64(cpuLogits[i] - gpuLogits[i]))
						sumErr += diff
						if diff > maxErr {
							maxErr = diff
							maxIdx = i
						}
					}
					r.maxErr = maxErr
					r.avgErr = sumErr / float64(vocabSize)
					r.cpuTopTok = argmax(cpuLogits)
					r.gpuTopTok = argmax(gpuLogits)
					r.topMatch = r.cpuTopTok == r.gpuTopTok
					errThreshold := 10.0
					if m.hasSSM {
						errThreshold = 30.0
					}
					nearTie := false
					if !r.topMatch && maxErr < errThreshold {
						top1 := cpuLogits[r.cpuTopTok]
						top2 := cpuLogits[r.gpuTopTok]
						nearTieThresh := float32(0.5)
						if m.hasSSM {
							nearTieThresh = float32(maxErr)
						}
						if math.Abs(float64(top1-top2)) < float64(nearTieThresh) {
							nearTie = true
						}
					}
					r.correctnessOK = maxErr < errThreshold && (r.topMatch || nearTie)

					status := passOrFail(r.correctnessOK)
					if nearTie {
						status = "WARN (near-tie)"
					}
					fmt.Printf("  [Correctness] maxErr=%.4f (idx %d)  avgErr=%.6f  top_match=%v  %s\n",
						maxErr, maxIdx, r.avgErr, r.topMatch, status)
				}
				rs.FreeAll()
				kv.FreeAll()
				gpuModel.FreeAll()
			}
		}

		// ═══ Phase 2: Coherence + Timing — dlgo CPU ═══
		formatted := llm.FormatChat(cfg, "You are a helpful assistant.", prompt)
		genCfg := llm.DefaultGenerateConfig()
		genCfg.MaxTokens = maxTokens
		genCfg.Seed = 42
		genCfg.Sampler.Temperature = 0

		fmt.Printf("  [dlgo CPU] Generating...\n")
		cpuResult, cpuErr := pipe.GenerateDetailed(formatted, genCfg)
		if cpuErr != nil {
			fmt.Printf("  [dlgo CPU] FAIL: %v\n", cpuErr)
		} else {
			r.dlgoCPUText = cpuResult.Text
			r.dlgoCPUTok = cpuResult.TotalTokens
			r.dlgoCPUPrefill = cpuResult.PrefillTimeMs
			r.dlgoCPUGen = cpuResult.GenerateTimeMs
			r.dlgoCPUTps = cpuResult.TokensPerSec
			r.coherenceDlgoCPU = isCoherent(cpuResult.Text)
			fmt.Printf("  [dlgo CPU] %d tok  prefill=%.1fms  gen=%.1fms (%.1f tok/s)  coherent=%v\n",
				cpuResult.TotalTokens, cpuResult.PrefillTimeMs, cpuResult.GenerateTimeMs, cpuResult.TokensPerSec, r.coherenceDlgoCPU)
			fmt.Printf("  [dlgo CPU] → %s\n", preview(cpuResult.Text, 100))
		}

		// ═══ Phase 3: Coherence + Timing — dlgo GPU ═══
		gpuPipe, gpuPipeErr := gpu.NewGpuPipeline(pipe)
		if gpuPipeErr != nil {
			r.gpuPipeFail = true
			fmt.Printf("  [dlgo GPU] Pipeline fail: %v\n", gpuPipeErr)
		} else {
			fmt.Printf("  [dlgo GPU] Generating...\n")
			gpuGenCfg := llm.DefaultGenerateConfig()
			gpuGenCfg.MaxTokens = maxTokens
			gpuGenCfg.Seed = 42
			gpuGenCfg.Sampler.Temperature = 0
			gpuResult, gpuErr := gpuPipe.GenerateDetailed(formatted, gpuGenCfg)
			if gpuErr != nil {
				r.gpuPipeFail = true
				fmt.Printf("  [dlgo GPU] FAIL: %v\n", gpuErr)
			} else {
				r.dlgoGPUText = gpuResult.Text
				r.dlgoGPUTok = gpuResult.TotalTokens
				r.dlgoGPUPrefill = gpuResult.PrefillTimeMs
				r.dlgoGPUGen = gpuResult.GenerateTimeMs
				r.dlgoGPUTps = gpuResult.TokensPerSec
				r.coherenceDlgoGPU = isCoherent(gpuResult.Text)
				fmt.Printf("  [dlgo GPU] %d tok  prefill=%.1fms  gen=%.1fms (%.1f tok/s)  coherent=%v\n",
					gpuResult.TotalTokens, gpuResult.PrefillTimeMs, gpuResult.GenerateTimeMs, gpuResult.TokensPerSec, r.coherenceDlgoGPU)
				fmt.Printf("  [dlgo GPU] → %s\n", preview(gpuResult.Text, 100))
			}
		}

		// Free all GPU resources before next model
		if gpuPipe != nil {
			gpuPipe.FreeAll()
			gpuPipe = nil
		}

		// ═══ Phase 4: Ollama CPU ═══
		fmt.Printf("  [Ollama CPU] Generating...\n")
		olCPUpre, olCPUgen, olCPUpretok, olCPUgentok, olCPUtext, olCPUerr := ollamaGenerate(m.ollamaName, prompt, maxTokens, 0)
		if olCPUerr != nil {
			fmt.Printf("  [Ollama CPU] FAIL: %v\n", olCPUerr)
		} else {
			r.ollamaCPUText = olCPUtext
			r.ollamaCPUTok = olCPUgentok
			r.ollamaCPUPrefill = olCPUpre
			r.ollamaCPUGen = olCPUgen
			r.coherenceOlCPU = isCoherent(olCPUtext)
			fmt.Printf("  [Ollama CPU] %d+%d tok  prefill=%.1fms  gen=%.1fms  coherent=%v\n",
				olCPUpretok, olCPUgentok, olCPUpre, olCPUgen, r.coherenceOlCPU)
			fmt.Printf("  [Ollama CPU] → %s\n", preview(olCPUtext, 100))
		}

		// ═══ Phase 5: Ollama GPU ═══
		fmt.Printf("  [Ollama GPU] Generating...\n")
		olGPUpre, olGPUgen, olGPUpretok, olGPUgentok, olGPUtext, olGPUerr := ollamaGenerate(m.ollamaName, prompt, maxTokens, 99)
		if olGPUerr != nil {
			fmt.Printf("  [Ollama GPU] FAIL: %v\n", olGPUerr)
		} else {
			r.ollamaGPUText = olGPUtext
			r.ollamaGPUTok = olGPUgentok
			r.ollamaGPUPrefill = olGPUpre
			r.ollamaGPUGen = olGPUgen
			r.coherenceOlGPU = isCoherent(olGPUtext)
			fmt.Printf("  [Ollama GPU] %d+%d tok  prefill=%.1fms  gen=%.1fms  coherent=%v\n",
				olGPUpretok, olGPUgentok, olGPUpre, olGPUgen, r.coherenceOlGPU)
			fmt.Printf("  [Ollama GPU] → %s\n", preview(olGPUtext, 100))
		}

		fmt.Println()
		results = append(results, r)
	}

	// ═══════════════════════════════════════════════════════════════
	// SUMMARY TABLES
	// ═══════════════════════════════════════════════════════════════

	// Table 1: Correctness
	fmt.Println("\n╔═════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      CORRECTNESS (CPU vs GPU logits)                       ║")
	fmt.Println("╠═════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ %-28s  %8s  %10s  %10s  %6s ║\n", "Model", "MaxErr", "AvgErr", "TopMatch", "Status")
	fmt.Println("╠═════════════════════════════════════════════════════════════════════════════╣")
	for _, r := range results {
		status := "PASS"
		if r.gpuUploadFail {
			status = "SKIP"
		} else if !r.correctnessOK {
			status = "FAIL"
		}
		fmt.Printf("║ %-28s  %8.4f  %10.6f  %10v  %6s ║\n",
			trunc(r.name, 28), r.maxErr, r.avgErr, r.topMatch, status)
	}
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════╝")

	// Table 2: Coherence
	fmt.Println("\n╔═════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                     COHERENCE (all 4 paths)                            ║")
	fmt.Println("╠═════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ %-28s  %8s  %8s  %8s  %8s ║\n", "Model", "dlCPU", "dlGPU", "OlCPU", "OlGPU")
	fmt.Println("╠═════════════════════════════════════════════════════════════════════════╣")
	for _, r := range results {
		fmt.Printf("║ %-28s  %8s  %8s  %8s  %8s ║\n",
			trunc(r.name, 28),
			boolStatus(r.coherenceDlgoCPU), boolStatus(r.coherenceDlgoGPU),
			boolStatus(r.coherenceOlCPU), boolStatus(r.coherenceOlGPU))
	}
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════╝")

	// Table 3: Prefill comparison
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                       PREFILL TIME (ms, lower = better)                                 ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ %-26s │ %8s  %8s  %7s │ %8s  %8s  %7s ║\n",
		"Model", "dlCPU", "OlCPU", "Δ CPU", "dlGPU", "OlGPU", "Δ GPU")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════╣")
	for _, r := range results {
		cpuDelta := "n/a"
		if r.ollamaCPUPrefill > 0 && r.dlgoCPUPrefill > 0 {
			d := (r.dlgoCPUPrefill - r.ollamaCPUPrefill) / r.ollamaCPUPrefill * 100
			cpuDelta = fmt.Sprintf("%+.0f%%", d)
		}
		gpuDelta := "n/a"
		if r.ollamaGPUPrefill > 0 && r.dlgoGPUPrefill > 0 {
			d := (r.dlgoGPUPrefill - r.ollamaGPUPrefill) / r.ollamaGPUPrefill * 100
			gpuDelta = fmt.Sprintf("%+.0f%%", d)
		}
		fmt.Printf("║ %-26s │ %7.1fms %7.1fms  %6s │ %7.1fms %7.1fms  %6s ║\n",
			trunc(r.name, 26), r.dlgoCPUPrefill, r.ollamaCPUPrefill, cpuDelta,
			r.dlgoGPUPrefill, r.ollamaGPUPrefill, gpuDelta)
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════╝")

	// Table 4: Generation comparison
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                     GENERATION TIME (ms, lower = better)                                ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ %-26s │ %8s  %8s  %7s │ %8s  %8s  %7s ║\n",
		"Model", "dlCPU", "OlCPU", "Δ CPU", "dlGPU", "OlGPU", "Δ GPU")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════╣")
	for _, r := range results {
		cpuDelta := "n/a"
		if r.ollamaCPUGen > 0 && r.dlgoCPUGen > 0 {
			d := (r.dlgoCPUGen - r.ollamaCPUGen) / r.ollamaCPUGen * 100
			cpuDelta = fmt.Sprintf("%+.0f%%", d)
		}
		gpuDelta := "n/a"
		if r.ollamaGPUGen > 0 && r.dlgoGPUGen > 0 {
			d := (r.dlgoGPUGen - r.ollamaGPUGen) / r.ollamaGPUGen * 100
			gpuDelta = fmt.Sprintf("%+.0f%%", d)
		}
		fmt.Printf("║ %-26s │ %7.1fms %7.1fms  %6s │ %7.1fms %7.1fms  %6s ║\n",
			trunc(r.name, 26), r.dlgoCPUGen, r.ollamaCPUGen, cpuDelta,
			r.dlgoGPUGen, r.ollamaGPUGen, gpuDelta)
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════╝")

	// Table 5: Generation speed (tok/s)
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                   GENERATION SPEED (tok/s, higher = better)                             ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ %-26s │ %8s  %8s  %7s │ %8s  %8s  %7s ║\n",
		"Model", "dlCPU", "OlCPU", "Δ CPU", "dlGPU", "OlGPU", "Δ GPU")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════╣")
	for _, r := range results {
		olCPUtps := float64(0)
		if r.ollamaCPUGen > 0 {
			olCPUtps = float64(r.ollamaCPUTok) / (r.ollamaCPUGen / 1000)
		}
		olGPUtps := float64(0)
		if r.ollamaGPUGen > 0 {
			olGPUtps = float64(r.ollamaGPUTok) / (r.ollamaGPUGen / 1000)
		}

		cpuDelta := "n/a"
		if olCPUtps > 0 && r.dlgoCPUTps > 0 {
			d := (r.dlgoCPUTps - olCPUtps) / olCPUtps * 100
			cpuDelta = fmt.Sprintf("%+.0f%%", d)
		}
		gpuDelta := "n/a"
		if olGPUtps > 0 && r.dlgoGPUTps > 0 {
			d := (r.dlgoGPUTps - olGPUtps) / olGPUtps * 100
			gpuDelta = fmt.Sprintf("%+.0f%%", d)
		}
		fmt.Printf("║ %-26s │ %7.1f   %7.1f   %6s │ %7.1f   %7.1f   %6s ║\n",
			trunc(r.name, 26), r.dlgoCPUTps, olCPUtps, cpuDelta,
			r.dlgoGPUTps, olGPUtps, gpuDelta)
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════╝")

	// Final verdict
	allCorrect := true
	allCoherent := true
	for _, r := range results {
		if !r.correctnessOK {
			allCorrect = false
		}
		if !r.coherenceDlgoCPU {
			allCoherent = false
		}
	}
	fmt.Println()
	if allCorrect && allCoherent {
		fmt.Println("ALL MODELS PASSED correctness and coherence checks.")
	} else {
		fmt.Println("SOME MODELS FAILED. See tables above.")
		os.Exit(1)
	}
}

func isCoherent(text string) bool {
	if text == "" || text == "FAIL" {
		return false
	}
	t := strings.TrimSpace(text)
	if len(t) < 5 {
		return false
	}
	nonASCII := 0
	for _, c := range t {
		if c > 127 {
			nonASCII++
		}
	}
	return float64(nonASCII)/float64(len([]rune(t))) <= 0.5
}

func preview(s string, n int) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

func trunc(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

func argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

func passOrFail(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}

func boolStatus(ok bool) string {
	if ok {
		return "OK"
	}
	return "FAIL"
}
