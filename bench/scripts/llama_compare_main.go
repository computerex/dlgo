//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

func l2(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

func diffL2(a, b []float32) float64 {
	var s float64
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		d := float64(a[i] - b[i])
		s += d * d
	}
	return math.Sqrt(s)
}

func argmax(v []float32) int32 {
	best := int32(0)
	for i := int32(1); i < int32(len(v)); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func main() {
	modelPath := `C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf`
	maxGen := 500

	if err := gpu.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "gpu init: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	pipe, err := llm.NewPipeline(modelPath, 4096)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gpu: %v\n", err)
		os.Exit(1)
	}
	defer gpuPipe.FreeAll()

	m := pipe.Model
	cfg := m.Config
	dim := cfg.EmbeddingDim

	prompt := llm.FormatChat(cfg, "You are a helpful assistant.",
		"Write a Python function for binary search and explain how it works step by step.")

	tokens := pipe.Tokenizer.Encode(prompt)
	fmt.Printf("Prompt tokens: %d, dim: %d, headDim: %d, layers: %d\n", len(tokens), dim, cfg.HeadDim, cfg.NumLayers)

	cpuKV := memory.NewMultiLayerKVCache(cfg.NumLayers, 4096, cfg.NumKVHeads*cfg.HeadDim)
	cpuRS := llm.NewRunState(cfg, 4096)

	gpuPipe.ResetState()
	rs := gpuPipe.RunState
	kv := gpuPipe.KVCache
	gm := gpuPipe.GpuModel

	gpuLogits := make([]float32, cfg.VocabSize)
	gpuX := make([]float32, dim)

	for i, tok := range tokens {
		llm.Forward(m, tok, i, cpuKV, cpuRS)
		gpu.GpuForward(m, gm, tok, i, kv, rs, gpuLogits, gpuPipe)
		gpu.Sync()
	}

	gpu.DownloadF32(rs.X, gpuX)
	cpuX := cpuRS.X[:dim]
	dNorm := diffL2(gpuX, cpuX)
	cNorm := l2(cpuX)
	relErr := 0.0
	if cNorm > 0 {
		relErr = 100 * dNorm / cNorm
	}
	fmt.Printf("Post-prefill X: CPU=%.4f GPU=%.4f diff=%.6f (%.4f%%)\n", cNorm, l2(gpuX), dNorm, relErr)

	cpuLogits := make([]float32, cfg.VocabSize)
	copy(cpuLogits, cpuRS.Logits)

	cpuTok := argmax(cpuLogits)
	gpuTok := argmax(gpuLogits[:cfg.VocabSize])
	fmt.Printf("First token: CPU=%d GPU=%d (same=%v)\n\n", cpuTok, gpuTok, cpuTok == gpuTok)

	fmt.Println("=== GENERATION (feeding CPU tokens) ===")
	pos := len(tokens)
	tok := cpuTok
	divergeCount := 0
	firstDivergenceStep := -1

	for step := 0; step < maxGen; step++ {
		if tok == cfg.EOS {
			fmt.Printf("\n[EOS at step %d]\n", step)
			break
		}
		for _, stop := range cfg.StopTokens {
			if tok == stop {
				fmt.Printf("\n[STOP at step %d]\n", step)
				goto done
			}
		}

		tokStr := pipe.Tokenizer.DecodeToken(tok)
		fmt.Print(tokStr)

		llm.Forward(m, tok, pos, cpuKV, cpuRS)
		gpu.GpuForward(m, gm, tok, pos, kv, rs, gpuLogits, gpuPipe)
		gpu.Sync()

		gpu.DownloadF32(rs.X, gpuX)
		copy(cpuX, cpuRS.X[:dim])
		dNorm = diffL2(gpuX, cpuX)
		cNorm = l2(cpuX)
		relErr = 0.0
		if cNorm > 0 {
			relErr = 100 * dNorm / cNorm
		}

		copy(cpuLogits, cpuRS.Logits)
		logDiff := diffL2(cpuLogits, gpuLogits[:cfg.VocabSize])

		cpuNext := argmax(cpuLogits)
		gpuNext := argmax(gpuLogits[:cfg.VocabSize])

		diverged := cpuNext != gpuNext
		if diverged {
			divergeCount++
			if firstDivergenceStep < 0 {
				firstDivergenceStep = step
			}
		}

		if step < 10 || step%50 == 0 || diverged {
			flag := ""
			if diverged {
				cpuStr := pipe.Tokenizer.DecodeToken(cpuNext)
				gpuStr := pipe.Tokenizer.DecodeToken(gpuNext)
				flag = fmt.Sprintf(" *** DIVERGED CPU=%q GPU=%q", cpuStr, gpuStr)
			}
			fmt.Printf("\nstep=%3d pos=%3d: X rel=%.2f%% logitDiff=%.1f%s\n",
				step, pos, relErr, logDiff, flag)
		}

		tok = cpuNext
		pos++
	}
done:
	fmt.Printf("\n\n=== SUMMARY ===\n")
	fmt.Printf("Total divergences: %d / %d steps\n", divergeCount, maxGen)
	if firstDivergenceStep >= 0 {
		fmt.Printf("First divergence at step %d (pos %d)\n", firstDivergenceStep, len(tokens)+firstDivergenceStep)
	} else {
		fmt.Printf("No token divergence in %d steps\n", maxGen)
	}
}
