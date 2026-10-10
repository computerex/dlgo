//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

func l2(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

func maxAbsDiff(a, b []float32) (float64, int) {
	maxD := float64(0)
	maxI := 0
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		d := math.Abs(float64(a[i] - b[i]))
		if d > maxD {
			maxD = d
			maxI = i
		}
	}
	return maxD, maxI
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

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: layer_cmp <gguf_path>")
		os.Exit(1)
	}

	if err := gpu.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "gpu init: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	pipe, err := llm.NewPipeline(os.Args[1], 4096)
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

	prompt := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
	tokens := pipe.Tokenizer.Encode(prompt)
	fmt.Printf("Tokens: %d, dim: %d\n", len(tokens), dim)

	// Process all prompt tokens through CPU
	pipe.KVCache.Reset()
	if pipe.RunState.SSMState != nil {
		pipe.RunState.SSMState.Reset()
	}
	for i, tok := range tokens {
		llm.Forward(m, tok, i, pipe.KVCache, pipe.RunState)
	}
	cpuLogits := make([]float32, cfg.VocabSize)
	copy(cpuLogits, pipe.RunState.Logits)

	// Process all prompt tokens through GPU (one at a time, unfused)
	gpuPipe.ResetState()
	rs := gpuPipe.RunState
	kv := gpuPipe.KVCache
	gm := gpuPipe.GpuModel

	// Process token by token through BOTH paths, comparing after each
	pipe.KVCache.Reset()
	if pipe.RunState.SSMState != nil {
		pipe.RunState.SSMState.Reset()
	}
	gpuPipe.ResetState()

	for ti, tok := range tokens {
		// CPU forward
		llm.Forward(m, int32(tok), ti, pipe.KVCache, pipe.RunState)

		// GPU forward (unfused)
		gpu.GpuForward(m, gm, int32(tok), ti, kv, rs, gpuPipe.LogitsBuf, gpuPipe)

		// Download GPU X
		gpuX := make([]float32, dim)
		gpu.Sync()
		gpu.DownloadF32(rs.X, gpuX)

		cpuX := pipe.RunState.X[:dim]
		dNorm := diffL2(gpuX, cpuX)
		cNorm := l2(cpuX)
		gNorm := l2(gpuX)
		md, mi := maxAbsDiff(gpuX, cpuX)
		relErr := 0.0
		if cNorm > 0 {
			relErr = 100 * dNorm / cNorm
		}
		fmt.Printf("pos=%2d: CPU=%.4f GPU=%.4f diff=%.6f (%.4f%%) maxAbs=%.6f@%d\n",
			ti, cNorm, gNorm, dNorm, relErr, md, mi)
	}

	// Final logit comparison
	gpuLogFinal := make([]float32, cfg.VocabSize)
	copy(gpuLogFinal, gpuPipe.LogitsBuf)
	fmt.Printf("\nFinal logit diff L2: %.4f (cpu=%.4f gpu=%.4f)\n",
		diffL2(cpuLogits, gpuLogFinal), l2(cpuLogits), l2(gpuLogFinal))
}
