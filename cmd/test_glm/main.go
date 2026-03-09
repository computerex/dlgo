package main

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func l2norm(x []float32) float32 {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	return float32(math.Sqrt(s))
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: test_glm <model.gguf> [--cpu]")
		os.Exit(1)
	}
	modelPath := os.Args[1]
	cpuOnly := false
	for _, a := range os.Args[2:] {
		if a == "--cpu" {
			cpuOnly = true
		}
	}

	nproc := runtime.NumCPU()
	runtime.GOMAXPROCS(nproc)
	debug.SetGCPercent(2000)

	fmt.Printf("Using %d CPU threads\n", nproc)

	maxSeqLen := 2048
	fmt.Printf("Loading pipeline (maxSeqLen=%d)...\n", maxSeqLen)
	pipe, err := llm.NewPipeline(modelPath, maxSeqLen)
	if err != nil {
		fmt.Printf("Error loading pipeline: %v\n", err)
		os.Exit(1)
	}
	defer pipe.Model.Close()
	cfg := pipe.Model.Config

	fmt.Printf("Model: %s (%d layers, %d dim, %d heads, vocab %d)\n",
		cfg.Architecture, cfg.NumLayers, cfg.EmbeddingDim, cfg.NumHeads, cfg.VocabSize)
	fmt.Printf("HeadDim=%d NumKVHeads=%d FFNDim=%d\n", cfg.HeadDim, cfg.NumKVHeads, cfg.FFNDim)
	fmt.Printf("RopeNeox=%v RopeFreqBase=%f RopeDim=%d\n", cfg.RopeNeox, cfg.RopeFreqBase, cfg.RopeDim)
	if cfg.RopeScaleType > 0 {
		fmt.Printf("RoPE scaling: type=%d factor=%.1f origMaxPos=%d betaFast=%.1f betaSlow=%.1f\n",
			cfg.RopeScaleType, cfg.RopeScaleFactor, cfg.RopeOrigMaxPos,
			cfg.RopeYaRNBetaFast, cfg.RopeYaRNBetaSlow)
	}
	if cfg.SlidingWindow > 0 {
		fmt.Printf("SlidingWindow=%d Pattern=%d\n", cfg.SlidingWindow, cfg.SlidingWindowPattern)
	}
	if cfg.ExpertCount > 0 {
		fmt.Printf("MoE: %d experts, %d active, ExpertFFNDim=%d SharedExpertFFNDim=%d GatingFunc=%d WeightsNorm=%v Scale=%.2f\n",
			cfg.ExpertCount, cfg.ExpertUsedCount, cfg.ExpertFFNDim, cfg.SharedExpertFFNDim,
			cfg.ExpertGatingFunc, cfg.ExpertWeightsNorm, cfg.ExpertWeightsScale)
	}
	if cfg.QLORARank > 0 {
		fmt.Printf("MLA: qLORARank=%d kvLORARank=%d qkNope=%d qkRope=%d vHeadDim=%d\n",
			cfg.QLORARank, cfg.KVLORARank, cfg.QKNopeDim, cfg.QKRopeDim, cfg.VHeadDim)
	}

	fmt.Printf("BOS=%d EOS=%d AddBOS=%v\n", cfg.BOS, cfg.EOS, cfg.AddBOS)
	// gpt-oss harmony template starts with <|start|>, no BOS needed
	if cfg.ChatTemplate == "harmony" {
		pipe.Tokenizer.AddBOS = false
	}
	prompt := llm.FormatChat(cfg, "", "What is 2+2?")
	fmt.Printf("Prompt: %q\n", prompt)

	var useGPU bool
	var gpuPipe *gpu.GpuPipeline
	if !cpuOnly {
		gpuPipe, err = gpu.NewGpuPipeline(pipe)
		if err != nil {
			fmt.Printf("[GPU] Not available: %v\n", err)
			fmt.Println("[CPU] Falling back to CPU-only inference")
		} else {
			useGPU = true
			defer gpuPipe.FreeAll()
		}
	} else {
		fmt.Println("[CPU] CPU-only mode requested")
	}

	maxTokens := 100

	if useGPU {
		skipCPU := false
		for _, a := range os.Args[2:] {
			if a == "--skip-cpu" {
				skipCPU = true
			}
		}

		if !skipCPU {
		fmt.Println("\n=== CPU vs GPU Per-Token Logits Comparison ===")
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("CPU comparison skipped (panic: %v)\n", r)
				}
			}()
			tokens := pipe.Tokenizer.Encode(prompt)

			// CPU forward
			rs := pipe.RunState
			pipe.KVCache.Reset()
			cpuTops := make([]int, len(tokens))
			cpuNorms := make([]float32, len(tokens))
			var cpuLogits []float32
			for i, t := range tokens {
				cpuLogits = llm.Forward(pipe.Model, t, i, pipe.KVCache, rs)
				cpuTops[i] = argmax(cpuLogits)
				cpuNorms[i] = l2norm(cpuLogits)
			}
			fmt.Printf("CPU final top: %d (%q) logit=%.4f norm=%.4f\n",
				cpuTops[len(tokens)-1], pipe.Tokenizer.DecodeToken(int32(cpuTops[len(tokens)-1])),
				cpuLogits[cpuTops[len(tokens)-1]], cpuNorms[len(tokens)-1])

			// Verify per-layer ForwardRange matches full forward
			{
				m := pipe.Model
				cfg := m.Config
				cpuRS2 := llm.NewRunState(cfg, pipe.MaxSeqLen)
				kvDim := cfg.NumKVHeads * cfg.HeadDim
				cpuKV2 := memory.NewMultiLayerKVCache(cfg.NumLayers, pipe.MaxSeqLen, kvDim)
				for l := 0; l < cfg.NumLayers; l++ {
					llm.ForwardRange(m, tokens[0], 0, l, l+1, cpuKV2, cpuRS2)
				}
				perLayerTop := argmax(cpuRS2.Logits)
				fullTop := cpuTops[0]
				fmt.Printf("Per-layer ForwardRange top=%d, Full forward top=%d, match=%v\n",
					perLayerTop, fullTop, perLayerTop == fullTop)
			}

			// GPU forward per-token comparison
			gpuPipe.KVCache.Reset()
			if gpuPipe.CPUKVCache != nil {
				gpuPipe.CPUKVCache.Reset()
			}
			gpuLogits := make([]float32, pipe.Model.Config.VocabSize)
			divergeAt := -1
			for i, t := range tokens {
				if gpuPipe.IsPartialGPU {
					gpu.GpuForwardPartial(pipe.Model, gpuPipe.GpuModel, t, i,
						gpuPipe.KVCache, gpuPipe.RunState, gpuLogits, gpuPipe.LayerConfs, gpuPipe)
				} else if err := gpu.GpuForward(pipe.Model, gpuPipe.GpuModel, t, i,
					gpuPipe.KVCache, gpuPipe.RunState, gpuLogits, gpuPipe); err != nil {
					fmt.Printf("  GpuForward error at pos %d: %v\n", i, err)
					break
				}
				gpuTop := argmax(gpuLogits)
				if gpuTop != cpuTops[i] && divergeAt < 0 {
					divergeAt = i
				}
				if i < 3 || i == len(tokens)-1 || (divergeAt >= 0 && i == divergeAt) {
					fmt.Printf("  pos %2d tok=%5d: CPU top=%5d norm=%.2f | GPU top=%5d norm=%.2f %s\n",
						i, t, cpuTops[i], cpuNorms[i], gpuTop, l2norm(gpuLogits),
						func() string {
							if gpuTop != cpuTops[i] {
								return " ** DIVERGE **"
							}
							return ""
						}())
				}
			}
			if divergeAt >= 0 {
				fmt.Printf("GPU diverges from CPU at position %d\n", divergeAt)
			} else {
				fmt.Println("GPU matches CPU for all prompt tokens!")
			}
		}()
		}

		// Run GPU prefill (resets KV cache internally)
		fmt.Printf("\n=== GPU Inference (%d max tokens) ===\n", maxTokens)
		result, err := gpuPipe.GenerateDetailed(prompt, llm.GenerateConfig{
			MaxTokens: maxTokens,
			Sampler:   ops.SamplerConfig{Temperature: 0},
			Stream: func(tok string) {
				fmt.Print(tok)
			},
		})
		if err != nil {
			fmt.Printf("\n[GPU] Error: %v\n", err)
		} else {
			fmt.Println()
			fmt.Printf("Prefill: %.1fms (%d tokens, %.1f tok/s)\n",
				result.PrefillTimeMs, result.PromptTokens,
				float64(result.PromptTokens)/result.PrefillTimeMs*1000)
			fmt.Printf("Generation: %d tokens in %.1fms (%.1f tok/s)\n",
				result.TotalTokens, result.GenerateTimeMs,
				float64(result.TotalTokens)/result.GenerateTimeMs*1000)
			fmt.Printf("Full output: %q\n", strings.TrimSpace(result.Text))
		}
	} else {
		fmt.Printf("\n=== CPU Inference (%d max tokens) ===\n", maxTokens)
		tokens := pipe.Tokenizer.Encode(prompt)
		fmt.Printf("Prompt tokens: %d\n", len(tokens))

		rs := pipe.RunState
		pipe.KVCache.Reset()

		prefillStart := time.Now()
		var logits []float32
		for i, t := range tokens {
			logits = llm.Forward(pipe.Model, t, i, pipe.KVCache, rs)
		}
		prefillMs := float64(time.Since(prefillStart).Microseconds()) / 1000.0
		fmt.Printf("Prefill: %.1fms (%d tokens, %.1f tok/s)\n",
			prefillMs, len(tokens), float64(len(tokens))/prefillMs*1000)

		pos := len(tokens)
		var generated []int32
		genStart := time.Now()
		for g := 0; g < maxTokens && pos < pipe.MaxSeqLen; g++ {
			maxIdx := argmax(logits)
			if maxIdx == int(pipe.Tokenizer.EOS) {
				break
			}
			for _, stop := range cfg.StopTokens {
				if int32(maxIdx) == stop {
					goto cpuDone
				}
			}
			generated = append(generated, int32(maxIdx))
			word := pipe.Tokenizer.DecodeToken(int32(maxIdx))
			fmt.Print(word)
			logits = llm.Forward(pipe.Model, int32(maxIdx), pos, pipe.KVCache, rs)
			pos++
		}
	cpuDone:
		genTime := time.Since(genStart)
		fmt.Println()
		fmt.Printf("Generation: %d tokens in %.1fms (%.1f tok/s)\n",
			len(generated), float64(genTime.Microseconds())/1000.0,
			float64(len(generated))/genTime.Seconds())
		fullOutput := pipe.Tokenizer.Decode(generated)
		fmt.Printf("Full output: %q\n", strings.TrimSpace(fullOutput))
	}

	printMemStats()
}

func printMemStats() {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	fmt.Printf("\nMemory: Heap=%.0fMB Sys=%.0fMB GC=%d\n",
		float64(ms.HeapAlloc)/(1024*1024),
		float64(ms.Sys)/(1024*1024),
		ms.NumGC)
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
