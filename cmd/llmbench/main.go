// Command llmbench loads an LLM through the GPU pipeline and reports prefill and
// decode timings via GenerateDetailed. Used to A/B the unified-Affine8 weight
// path (DLGO_LLM_AFFINE8) against the per-quant path.
package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func main() {
	modelPath := os.Getenv("DLGO_BENCH_MODEL")
	if modelPath == "" && len(os.Args) > 1 {
		modelPath = os.Args[1]
	}
	if modelPath == "" {
		modelPath = `C:\models\Qwen3-4B-Q4_K_M.gguf`
	}
	maxTok := envInt("DLGO_BENCH_MAXTOK", 64)
	maxSeq := envInt("DLGO_BENCH_MAXSEQ", 8192)
	rep := envInt("DLGO_BENCH_REPEAT", 120)

	prompt := os.Getenv("DLGO_BENCH_PROMPT")
	if prompt == "" {
		sentence := "The quick brown fox jumps over the lazy dog near the riverbank. "
		prompt = strings.Repeat(sentence, rep) + "\n\nSummarize the passage above in one short sentence:"
	}

	if err := gpu.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "gpu init: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	pipe, err := llm.NewPipeline(modelPath, maxSeq)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	gp, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gpu pipeline: %v\n", err)
		os.Exit(1)
	}
	defer gp.FreeAll()

	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = maxTok
	cfg.Seed = 1234

	fmt.Printf("model=%s affine8=%v maxTok=%d maxSeq=%d\n",
		modelPath, os.Getenv("DLGO_LLM_AFFINE8"), maxTok, maxSeq)

	// Warmup (drop timing).
	if _, err := gp.GenerateDetailed("Hello.", cfg); err != nil {
		fmt.Fprintf(os.Stderr, "warmup: %v\n", err)
		os.Exit(1)
	}

	// Timed run.
	t0 := time.Now()
	res, err := gp.GenerateDetailed(prompt, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	wall := time.Since(t0)

	fmt.Printf("prompt_tokens=%d gen_tokens=%d prefill_ms=%.1f generate_ms=%.1f wall_ms=%.1f tok/s=%.2f\n",
		res.PromptTokens, res.TotalTokens, res.PrefillTimeMs, res.GenerateTimeMs,
		float64(wall.Milliseconds()), res.TokensPerSec)

	out := res.Text
	if len(out) > 160 {
		out = out[:160]
	}
	fmt.Printf("output: %q\n", out)
}
