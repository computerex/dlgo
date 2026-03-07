// Interactive example: multi-turn chat with an LLM and per-turn performance stats.
// Uses incremental KV-cache reuse for faster multi-turn chat.
//
// Usage:
//
//	go run . [--ctx N] [--max-tokens N] [--temp T] [--threads N] [--history-turns N] [model.gguf]
package main

import (
	"bufio"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

type turnResult struct {
	Text         string
	TokensPerSec float64
	PrefillMs    float64
	PrefillDelta int
	GenerateMs   float64
	PromptTokens int
	OutputTokens int
}

type chatRunner struct {
	pipe        *llm.Pipeline
	cachedToken []int32
}

func (r *chatRunner) resetCache() {
	r.pipe.KVCache.Reset()
	if r.pipe.RunState.SSMState != nil {
		r.pipe.RunState.SSMState.Reset()
	}
	r.cachedToken = nil
}

func commonPrefixLen(a, b []int32) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

func stopStrings() []string {
	return []string{
		"<|im_end|>",
		"<|endoftext|>",
		"<|end|>",
		"</s>",
		"<|assistant|>",
		"<end_of_turn>",
		"<|eot_id|>",
	}
}

func (r *chatRunner) generate(prompt string, cfg llm.GenerateConfig) (*turnResult, error) {
	tokens := r.pipe.Tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenizer produced no tokens")
	}
	if len(tokens) >= r.pipe.MaxSeqLen {
		return nil, fmt.Errorf("prompt too long: %d tokens (max %d)", len(tokens), r.pipe.MaxSeqLen)
	}

	common := commonPrefixLen(r.cachedToken, tokens)
	if common != len(r.cachedToken) {
		// Diverged prompt (history trim/edit): rebuild cache from scratch.
		r.resetCache()
		common = 0
	}

	prefillStart := time.Now()
	if llm.CanBatchPrefill(r.pipe.Model) && len(tokens)-common > 1 {
		llm.BatchPrefill(r.pipe.Model, tokens[common:], common, r.pipe.KVCache, r.pipe.RunState)
	} else {
		for i := common; i < len(tokens); i++ {
			if i == len(tokens)-1 {
				llm.Forward(r.pipe.Model, tokens[i], i, r.pipe.KVCache, r.pipe.RunState)
			} else {
				llm.ForwardNoLogits(r.pipe.Model, tokens[i], i, r.pipe.KVCache, r.pipe.RunState)
			}
		}
	}
	prefillMs := float64(time.Since(prefillStart).Microseconds()) / 1000.0

	r.cachedToken = append(r.cachedToken[:0], tokens...)

	rng := rand.New(rand.NewSource(cfg.Seed))
	if cfg.Seed < 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	stops := stopStrings()
	var recent []int32
	var forwardTokens []int32
	var visibleTokens int
	var outText strings.Builder

	genStart := time.Now()
	pos := len(tokens)

	for step := 0; step < cfg.MaxTokens; step++ {
		if pos >= r.pipe.MaxSeqLen-1 {
			break
		}

		next := int32(ops.SampleToken(r.pipe.RunState.Logits, cfg.Sampler, recent, rng))
		if next == r.pipe.Model.Config.EOS {
			break
		}
		stopTok := false
		for _, st := range r.pipe.Model.Config.StopTokens {
			if next == st {
				stopTok = true
				break
			}
		}
		if stopTok {
			break
		}

		tokText := r.pipe.Tokenizer.DecodeToken(next)
		outText.WriteString(tokText)
		visibleTokens++

		if cfg.Stream != nil {
			cfg.Stream(tokText)
		}

		full := outText.String()
		matchedStop := false
		for _, ss := range stops {
			if strings.HasSuffix(full, ss) {
				outText.Reset()
				outText.WriteString(strings.TrimSuffix(full, ss))
				matchedStop = true
				break
			}
		}
		if matchedStop {
			break
		}

		llm.Forward(r.pipe.Model, next, pos, r.pipe.KVCache, r.pipe.RunState)
		pos++
		forwardTokens = append(forwardTokens, next)
		recent = append(recent, next)
		if len(recent) > 64 {
			recent = recent[1:]
		}
	}
	genMs := float64(time.Since(genStart).Microseconds()) / 1000.0

	r.cachedToken = append(r.cachedToken, forwardTokens...)

	tokPerSec := 0.0
	if genMs > 0 {
		tokPerSec = float64(visibleTokens) / (genMs / 1000.0)
	}

	return &turnResult{
		Text:         outText.String(),
		TokensPerSec: tokPerSec,
		PrefillMs:    prefillMs,
		PrefillDelta: len(tokens) - common,
		GenerateMs:   genMs,
		PromptTokens: len(tokens),
		OutputTokens: visibleTokens,
	}, nil
}

func main() {
	ctx := flag.Int("ctx", 8192, "runtime context length (tokens)")
	maxTokens := flag.Int("max-tokens", 256, "max tokens per assistant response")
	temp := flag.Float64("temp", 0.7, "sampling temperature (0 = greedy)")
	threads := flag.Int("threads", 0, "worker threads (0 = auto, try 128 on this machine)")
	historyTurns := flag.Int("history-turns", 6, "number of recent user/assistant turns to keep")
	flag.Parse()

	modelPath := `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`
	if flag.NArg() > 0 {
		modelPath = flag.Arg(0)
	}

	if *ctx <= 0 {
		fmt.Fprintln(os.Stderr, "Error: --ctx must be > 0")
		os.Exit(1)
	}
	if *maxTokens <= 0 {
		fmt.Fprintln(os.Stderr, "Error: --max-tokens must be > 0")
		os.Exit(1)
	}
	if *temp < 0 {
		fmt.Fprintln(os.Stderr, "Error: --temp must be >= 0")
		os.Exit(1)
	}
	if *threads < 0 {
		fmt.Fprintln(os.Stderr, "Error: --threads must be >= 0")
		os.Exit(1)
	}
	if *historyTurns < 1 {
		fmt.Fprintln(os.Stderr, "Error: --history-turns must be >= 1")
		os.Exit(1)
	}
	if *threads > 0 {
		os.Setenv("DLGO_NUM_THREADS", fmt.Sprintf("%d", *threads))
	}

	pipe, err := llm.NewPipeline(modelPath, *ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	runner := &chatRunner{pipe: pipe}

	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = *maxTokens
	cfg.Sampler.Temperature = float32(*temp)

	system := "You are a helpful assistant."
	messages := []llm.Message{{Role: "system", Content: system}}

	fmt.Printf("Model: %s (%d layers, %d dim, %d heads, vocab %d, ctx %d)\n",
		pipe.Model.Config.Architecture,
		pipe.Model.Config.NumLayers,
		pipe.Model.Config.EmbeddingDim,
		pipe.Model.Config.NumHeads,
		pipe.Model.Config.VocabSize,
		pipe.Model.Config.ContextLength,
	)
	fmt.Printf("Runtime context (--ctx): %d tokens\n", pipe.MaxSeqLen)
	if *threads > 0 {
		fmt.Printf("Workers (--threads): %d\n", *threads)
	} else {
		fmt.Printf("Workers (--threads): auto (%d)\n", runtime.GOMAXPROCS(0))
	}
	fmt.Printf("Generation: max-tokens=%d temp=%.2f\n", cfg.MaxTokens, cfg.Sampler.Temperature)
	fmt.Printf("History window (--history-turns): %d turns\n", *historyTurns)
	fmt.Println("Interactive chat ready. Type 'exit' or 'quit' to leave.")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024), 1024*1024)
	for {
		fmt.Print("You> ")
		if !scanner.Scan() {
			fmt.Println()
			break
		}

		user := strings.TrimSpace(scanner.Text())
		if user == "" {
			continue
		}
		if strings.EqualFold(user, "exit") || strings.EqualFold(user, "quit") {
			break
		}

		messages = append(messages, llm.Message{Role: "user", Content: user})
		windowed := applyHistoryWindow(messages, *historyTurns)
		prompt := llm.FormatMessages(pipe.Model.Config, windowed)

		result, err := runner.generate(prompt, cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error generating response: %v\n", err)
			continue
		}

		responseRaw := result.Text
		responseDisplay := strings.TrimSpace(responseRaw)
		if responseDisplay == "" {
			responseDisplay = "(empty response)"
		}

		fmt.Println("AI>", responseDisplay)
		fmt.Printf("   [%.1f tok/s | prefill %.0f ms (%d delta tok) | gen %.0f ms | prompt %d tok | output %d tok]\n\n",
			result.TokensPerSec,
			result.PrefillMs,
			result.PrefillDelta,
			result.GenerateMs,
			result.PromptTokens,
			result.OutputTokens,
		)

		messages = append(messages, llm.Message{Role: "assistant", Content: responseRaw})
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Input error: %v\n", err)
		os.Exit(1)
	}
}

func applyHistoryWindow(messages []llm.Message, turns int) []llm.Message {
	if len(messages) <= 1 {
		return messages
	}
	system := messages[0]
	if system.Role != "system" {
		return messages
	}
	maxMsgs := turns * 2
	rest := messages[1:]
	if len(rest) <= maxMsgs {
		return messages
	}
	trimmed := make([]llm.Message, 0, 1+maxMsgs)
	trimmed = append(trimmed, system)
	trimmed = append(trimmed, rest[len(rest)-maxMsgs:]...)
	return trimmed
}
