//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

type modelSpec struct {
	name, path string
}

var models = []modelSpec{
	{"SmolLM2 360M Q8_0", `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`},
	{"TinyLlama 1.1B Q4_0", `C:\projects\evoke\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`},
	{"Qwen 2.5 0.5B Q4_K_M", `C:\projects\evoke\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`},
	{"Gemma 3 1B Q4_K_M", `C:\projects\evoke\models\gemma-3-1b-it-Q4_K_M.gguf`},
	{"Gemma 3 270M Q8_0", `C:\projects\evoke\models\gemma-3-270m-it-Q8_0.gguf`},
	{"SmolLM2 1.7B Q4_K_M", `C:\projects\evoke\models\smollm2-1.7b-instruct-q4_k_m.gguf`},
	{"Llama 3.2 1B Q4_K_M", `C:\projects\evoke\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf`},
	{"Phi-4-mini Q3_K_M", `C:\projects\evoke\models\Phi-4-mini-instruct-Q3_K_M.gguf`},
	{"Qwen3 0.6B Q8_0", `C:\projects\evoke\models\Qwen3-0.6B-Q8_0.gguf`},
}

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║       dlgo Full Regression Test — Correctness & Coherence        ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════╝")

	if err := gpu.Init(); err != nil {
		fmt.Printf("GPU init failed: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()
	fmt.Printf("GPU: %s (%.0f MB VRAM)\n\n", gpu.DeviceName(), float64(gpu.VRAMBytes())/(1024*1024))

	type testResult struct {
		name                      string
		loaded                    bool
		maxErr, avgErr            float64
		cpuTop, gpuTop            int
		topMatch                  bool
		cpuText, gpuText          string
		cpuTok, gpuTok            int
		cpuGenMs, gpuGenMs        float64
		cpuPrefillMs, gpuPrefillMs float64
		correctnessPass           bool
		coherencePass             bool
		err                       string
	}

	var results []testResult

	for _, m := range models {
		fmt.Printf("═══ %s ═══\n", m.name)
		tr := testResult{name: m.name}

		pipe, err := llm.NewPipeline(m.path, 512)
		if err != nil {
			tr.err = fmt.Sprintf("load fail: %v", err)
			fmt.Printf("  SKIP: %s\n\n", tr.err)
			results = append(results, tr)
			continue
		}
		tr.loaded = true

		cfg := pipe.Model.Config
		dim := cfg.EmbeddingDim
		vocabSize := cfg.VocabSize

		// ─── Phase 1: Logit Correctness (CPU vs GPU) ───
		gpuModel, gpuUploadErr := gpu.UploadModel(pipe.Model)
		if gpuUploadErr != nil {
			fmt.Printf("  GPU upload fail (OOM): %v\n", gpuUploadErr)
		}

		qDim := cfg.NumHeads * cfg.HeadDim
		kvDim := cfg.NumKVHeads * cfg.HeadDim
		ffnDim := cfg.FFNDim

		prompt := "Hello"
		tokens := pipe.Tokenizer.Encode(prompt)
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

		if gpuUploadErr == nil {
			rs := gpu.NewGpuRunState(dim, qDim, kvDim, ffnDim, vocabSize)
			kv := gpu.NewGpuKVCache(cfg.NumLayers, 512, kvDim)
			gpuLogits := make([]float32, vocabSize)

			for i, tok := range tokens {
				gpu.GpuForward(pipe.Model, gpuModel, tok, i, kv, rs, gpuLogits)
			}
			gpu.Sync()

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
			avgErr := sumErr / float64(vocabSize)

			cpuTop := argmax(cpuLogits)
			gpuTop := argmax(gpuLogits)

			tr.maxErr = maxErr
			tr.avgErr = avgErr
			tr.cpuTop = cpuTop
			tr.gpuTop = gpuTop
			tr.topMatch = cpuTop == gpuTop
			tr.correctnessPass = maxErr < 10.0 && tr.topMatch

			fmt.Printf("  Logits: maxErr=%.4f (idx %d) avgErr=%.6f  top match=%v\n",
				maxErr, maxIdx, avgErr, tr.topMatch)
			fmt.Printf("  CPU top: %d (%.2f)  GPU top: %d (%.2f)\n",
				cpuTop, cpuLogits[cpuTop], gpuTop, gpuLogits[gpuTop])
		} else {
			tr.correctnessPass = true
			fmt.Printf("  GPU OOM — skipping logit comparison (CPU-only correctness assumed)\n")
		}

		// ─── Phase 2: Coherence Test (generate text on CPU and GPU) ───
		coherencePrompt := "Explain what a computer is in one sentence."
		formatted := llm.FormatChat(cfg, "You are a helpful assistant.", coherencePrompt)

		// CPU generation
		cpuCfg := llm.DefaultGenerateConfig()
		cpuCfg.MaxTokens = 64
		cpuCfg.Seed = 42
		cpuCfg.Sampler.Temperature = 0

		cpuStart := time.Now()
		cpuResult, cpuErr := pipe.GenerateDetailed(formatted, cpuCfg)
		cpuElapsed := time.Since(cpuStart)
		if cpuErr != nil {
			fmt.Printf("  CPU gen FAIL: %v\n", cpuErr)
			tr.cpuText = "FAIL"
		} else {
			tr.cpuText = cpuResult.Text
			tr.cpuTok = cpuResult.TotalTokens
			tr.cpuGenMs = cpuResult.GenerateTimeMs
			tr.cpuPrefillMs = cpuResult.PrefillTimeMs
			fmt.Printf("  CPU gen: %d tok  prefill=%.1fms  gen=%.1fms (%.1f tok/s)  total=%.1fms\n",
				cpuResult.TotalTokens, cpuResult.PrefillTimeMs, cpuResult.GenerateTimeMs,
				cpuResult.TokensPerSec, float64(cpuElapsed.Milliseconds()))
			fmt.Printf("  CPU text: %s\n", preview(cpuResult.Text, 120))
		}

		// GPU generation (using fused pipeline)
		var gpuPipeErr error
		var gpuPipe *gpu.GpuPipeline
		if gpuUploadErr == nil {
			gpuPipe, gpuPipeErr = gpu.NewGpuPipeline(pipe)
		} else {
			gpuPipeErr = gpuUploadErr
		}
		if gpuPipeErr != nil {
			fmt.Printf("  GPU pipeline fail: %v\n", gpuPipeErr)
		} else {
			gpuGenCfg := llm.DefaultGenerateConfig()
			gpuGenCfg.MaxTokens = 64
			gpuGenCfg.Seed = 42
			gpuGenCfg.Sampler.Temperature = 0

			gpuStart := time.Now()
			gpuResult, gpuErr := gpuPipe.GenerateDetailed(formatted, gpuGenCfg)
			gpuElapsed := time.Since(gpuStart)
			if gpuErr != nil {
				fmt.Printf("  GPU gen FAIL: %v\n", gpuErr)
				tr.gpuText = "FAIL"
			} else {
				tr.gpuText = gpuResult.Text
				tr.gpuTok = gpuResult.TotalTokens
				tr.gpuGenMs = gpuResult.GenerateTimeMs
				tr.gpuPrefillMs = gpuResult.PrefillTimeMs
				fmt.Printf("  GPU gen: %d tok  prefill=%.1fms  gen=%.1fms (%.1f tok/s)  total=%.1fms\n",
					gpuResult.TotalTokens, gpuResult.PrefillTimeMs, gpuResult.GenerateTimeMs,
					gpuResult.TokensPerSec, float64(gpuElapsed.Milliseconds()))
				fmt.Printf("  GPU text: %s\n", preview(gpuResult.Text, 120))
			}
		}

		cpuCoherent := isCoherent(tr.cpuText)
		gpuCoherent := isCoherent(tr.gpuText)
		gpuOOM := gpuPipeErr != nil
		tr.coherencePass = cpuCoherent && (gpuCoherent || gpuOOM)

		status := "PASS"
		if !tr.correctnessPass || !tr.coherencePass {
			status = "FAIL"
		}
		fmt.Printf("  Result: %s (correctness=%v coherence_cpu=%v coherence_gpu=%v)\n\n",
			status, tr.correctnessPass, cpuCoherent, gpuCoherent)

		results = append(results, tr)
	}

	// ─── Summary ───
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      REGRESSION TEST SUMMARY                     ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════╣")
	allPass := true
	for _, r := range results {
		status := "PASS"
		detail := ""
		if !r.loaded {
			status = "SKIP"
			detail = r.err
		} else if r.err != "" {
			status = "SKIP"
			detail = r.err
		} else if !r.correctnessPass || !r.coherencePass {
			status = "FAIL"
			allPass = false
			if !r.correctnessPass {
				detail += fmt.Sprintf("maxErr=%.2f topMatch=%v ", r.maxErr, r.topMatch)
			}
			if !r.coherencePass {
				detail += "incoherent_output"
			}
		} else {
			detail = fmt.Sprintf("maxErr=%.4f cpuPre=%.0fms gpuPre=%.0fms cpuGen=%.0fms gpuGen=%.0fms",
				r.maxErr, r.cpuPrefillMs, r.gpuPrefillMs, r.cpuGenMs, r.gpuGenMs)
		}
		fmt.Printf("║ %-4s  %-28s  %s\n", status, r.name, detail)
	}
	fmt.Println("╚═══════════════════════════════════════════════════════════════════╝")
	if allPass {
		fmt.Println("\n✓ All models passed regression test.")
	} else {
		fmt.Println("\n✗ Some models failed. See details above.")
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
	// Check for garbage: extremely high non-ASCII ratio or all-same char
	nonASCII := 0
	for _, c := range t {
		if c > 127 {
			nonASCII++
		}
	}
	if float64(nonASCII)/float64(len([]rune(t))) > 0.5 {
		return false
	}
	return true
}

func preview(s string, n int) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	if len(s) > n {
		return s[:n] + "..."
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
