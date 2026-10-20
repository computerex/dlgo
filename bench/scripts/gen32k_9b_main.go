//go:build ignore

// gen32k_9b_main is a long-form generation benchmark for the Qwen3.5-9B
// (Q3_K_M) backbone. It generates a large number of tokens (default 32768) on
// the Vulkan GPU backend and reports prefill + decode timings.
//
// Run:
//   go build -tags vulkan -ldflags "-linkmode internal" -o gen32k_9b.exe bench/scripts/gen32k_9b_main.go
//   .\gen32k_9b.exe -tokens 32768 -ctx 40960
//
// Flags:
//   -model   GGUF path (default the 9B Q3_K_M used by System One)
//   -tokens  tokens to generate (default 32768)
//   -ctx     requested context window (default 40960; clamped to model max/RAM)
//   -temp    sampling temperature (default 0 = greedy, deterministic)
//   -seed    RNG seed (default 42)
//   -mode    chat | chat_nothink | raw (default chat_nothink)
//   -force   suppress EOS and stop tokens so the full token budget is decoded
//            (default true; needed to actually exercise 32k decode steps)
//   -cpu     run on CPU instead of GPU
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := flag.String("model", `C:\Users\mohd\Downloads\Qwen3.5-9B-Q3_K_M.gguf`, "path to the GGUF model")
	maxTokens := flag.Int("tokens", 32768, "number of tokens to generate")
	ctxLen := flag.Int("ctx", 40960, "requested context window in tokens")
	temp := flag.Float64("temp", 0.0, "sampling temperature (0 = greedy)")
	seed := flag.Int64("seed", 42, "random seed")
	mode := flag.String("mode", "chat_nothink", "prompt mode: chat | chat_nothink | raw")
	force := flag.Bool("force", true, "suppress EOS/stop tokens to decode the full token budget")
	useCPU := flag.Bool("cpu", false, "run on CPU instead of GPU")
	progress := flag.Int("progress", 2048, "print a progress line every N tokens (0 = off)")
	flag.Parse()

	userPrompt := "Write an extremely long, detailed, and thorough essay about the complete history of artificial intelligence, from its philosophical origins through the deep learning revolution to modern large language models. Cover every era in depth, include many concrete examples, dates and people, and keep expanding the discussion."

	// ---- GPU init ----
	var gpuPipe *gpu.GpuPipeline
	if !*useCPU {
		if err := gpu.Init(); err != nil {
			fmt.Fprintf(os.Stderr, "gpu init: %v\n", err)
			os.Exit(1)
		}
		defer gpu.Shutdown()
	}

	// ---- Load model ----
	fmt.Printf("Loading %s (ctx=%d) ...\n", *modelPath, *ctxLen)
	loadStart := time.Now()
	pipe, err := llm.NewPipeline(*modelPath, *ctxLen)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	loadMs := float64(time.Since(loadStart).Microseconds()) / 1000.0

	// ---- Build runner ----
	// gpu.NewGpuPipeline releases the CPU-side scratch before uploading weights
	// (so a large context never doubles peak host memory) and rebuilds it if
	// creation fails. No manual FreeForGPU / GC dance is needed.
	if !*useCPU {
		gpuPipe, err = gpu.NewGpuPipeline(pipe)
		if err != nil {
			fmt.Fprintf(os.Stderr, "gpu pipeline: %v\n", err)
			os.Exit(1)
		}
		defer gpuPipe.FreeAll()
	}

	cfg := pipe.Model.Config
	var prompt string
	switch *mode {
	case "chat":
		prompt = llm.FormatChat(cfg, "You are a helpful assistant.", userPrompt)
	case "raw":
		prompt = userPrompt
	default: // chat_nothink
		noThink := false
		msgs := []llm.Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: userPrompt},
		}
		prompt = llm.FormatMessages(cfg, msgs, llm.FormatOptions{EnableThinking: &noThink})
	}

	genCfg := llm.DefaultGenerateConfig()
	genCfg.MaxTokens = *maxTokens
	genCfg.Seed = *seed
	genCfg.Sampler.Temperature = float32(*temp)

	// Force full-length decode: never emit EOS or structural stop tokens.
	if *force {
		eos := cfg.EOS
		found := false
		for _, s := range cfg.SuppressTokens {
			if s == eos {
				found = true
				break
			}
		}
		if !found {
			cfg.SuppressTokens = append(cfg.SuppressTokens, eos)
			pipe.Model.Config.SuppressTokens = cfg.SuppressTokens
		}
		pipe.Model.Config.StopTokens = nil
		pipe.Model.Config.EOS = -1
	}

	// Windowed decode-speed reporting. Stream fires once per generated token, so
	// we time each block of `progress` tokens to show how decode speed varies
	// with context length (it should fall as attention cost grows).
	promptTok := len(pipe.Tokenizer.Encode(prompt))
	winCount := 0
	var winStart time.Time
	genCfg.Stream = func(tok string) {
		if *progress <= 0 {
			return
		}
		if winCount == 0 {
			winStart = time.Now()
		}
		winCount++
		if winCount%*progress != 0 {
			return
		}
		el := time.Since(winStart).Seconds()
		fmt.Fprintf(os.Stderr, "  [decode] ctx~%6d  %6.1f tok/s (over last %d tokens)\n",
			promptTok+winCount, float64(*progress)/el, *progress)
		winStart = time.Now()
	}

	fmt.Println(strings.Repeat("=", 72))
	fmt.Printf("  Model:        %s\n", cfg.Architecture)
	fmt.Printf("  Layers:       %d   Dim: %d   Heads: %d/%d   Vocab: %d\n",
		cfg.NumLayers, cfg.EmbeddingDim, cfg.NumHeads, cfg.NumKVHeads, cfg.VocabSize)
	fmt.Printf("  Backend:      %s\n", backendName(*useCPU, gpuPipe))
	fmt.Printf("  MaxSeqLen:    %d (requested %d)\n", pipe.MaxSeqLen, *ctxLen)
	fmt.Printf("  Load time:    %.1f ms\n", loadMs)
	fmt.Printf("  Temperature:  %.2f   Seed: %d   Force-full: %v\n", *temp, *seed, *force)
	fmt.Printf("  Target:       %d tokens\n", *maxTokens)
	fmt.Println(strings.Repeat("=", 72))

	// Prefill timing is measured by GenerateDetailed; we additionally wrap the
	// whole call for wall time.
	wallStart := time.Now()
	result, err := run(pipe, gpuPipe, prompt, genCfg, *useCPU)
	wallMs := float64(time.Since(wallStart).Microseconds()) / 1000.0
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	prefillTokS := float64(0)
	if result.PrefillTimeMs > 0 {
		prefillTokS = float64(result.PromptTokens) / (result.PrefillTimeMs / 1000.0)
	}

	fmt.Println(strings.Repeat("=", 72))
	fmt.Println("  RESULTS")
	fmt.Println(strings.Repeat("-", 72))
	fmt.Printf("  Prompt tokens:     %d\n", result.PromptTokens)
	fmt.Printf("  Generated tokens:  %d\n", result.TotalTokens)
	fmt.Printf("  Prefill time:      %.1f ms  (%.1f tok/s)\n", result.PrefillTimeMs, prefillTokS)
	fmt.Printf("  Decode time:       %.1f ms\n", result.GenerateTimeMs)
	fmt.Printf("  Decode speed:      %.2f tok/s\n", result.TokensPerSec)
	fmt.Printf("  Total wall time:   %.1f ms  (%.1f s)\n", wallMs, wallMs/1000.0)
	fmt.Println(strings.Repeat("-", 72))

	preview := result.Text
	if len(preview) > 240 {
		preview = preview[:240] + " ..."
	}
	preview = strings.ReplaceAll(preview, "\n", " ")
	fmt.Printf("  Preview: %s\n", preview)
	fmt.Println(strings.Repeat("=", 72))
}

func backendName(useCPU bool, gpuPipe *gpu.GpuPipeline) string {
	if useCPU || gpuPipe == nil {
		return "CPU"
	}
	return "GPU (Vulkan)"
}

// genOut is the subset of generation results this benchmark reports. The CPU
// (llm.GenerateResult) and GPU (gpu.GenerateResult) result types are distinct,
// so results are normalised into this struct.
type genOut struct {
	Text           string
	TokensPerSec   float64
	PrefillTimeMs  float64
	GenerateTimeMs float64
	TotalTokens    int
	PromptTokens   int
}

func run(pipe *llm.Pipeline, gpuPipe *gpu.GpuPipeline, prompt string, cfg llm.GenerateConfig, useCPU bool) (*genOut, error) {
	if useCPU || gpuPipe == nil {
		r, err := pipe.GenerateDetailed(prompt, cfg)
		if err != nil {
			return nil, err
		}
		return &genOut{r.Text, r.TokensPerSec, r.PrefillTimeMs, r.GenerateTimeMs, r.TotalTokens, r.PromptTokens}, nil
	}
	r, err := gpuPipe.GenerateDetailed(prompt, cfg)
	if err != nil {
		return nil, err
	}
	return &genOut{r.Text, r.TokensPerSec, r.PrefillTimeMs, r.GenerateTimeMs, r.TotalTokens, r.PromptTokens}, nil
}
