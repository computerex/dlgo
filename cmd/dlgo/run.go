package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/computerex/dlgo/models/llm"
)

func cmdRun(args []string) {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: dlgo run <model.gguf> [flags]")
		fs.PrintDefaults()
	}

	ctx := fs.Int("ctx", 8192, "context length (tokens)")
	maxTokens := fs.Int("max-tokens", 512, "max tokens per response")
	temp := fs.Float64("temp", 0.7, "sampling temperature (0 = greedy)")
	topK := fs.Int("top-k", 40, "top-k sampling")
	topP := fs.Float64("top-p", 0.9, "top-p nucleus sampling")
	minP := fs.Float64("min-p", 0.0, "min-p sampling threshold")
	repeatPenalty := fs.Float64("repeat-penalty", 1.1, "repetition penalty")
	seed := fs.Int64("seed", -1, "random seed (-1 = random)")
	system := fs.String("system", "You are a helpful assistant.", "system prompt")
	threads := fs.Int("threads", 0, "worker threads (0 = auto)")
	historyTurns := fs.Int("history-turns", 6, "recent turns to keep in context")
	useGPU := fs.Bool("gpu", false, "use Vulkan GPU backend")
	noStream := fs.Bool("no-stream", false, "disable token streaming")

	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "Error: model path required")
		fmt.Fprintln(os.Stderr, "Usage: dlgo run <model.gguf> [flags]")
		os.Exit(1)
	}
	modelPath := fs.Arg(0)

	if *threads > 0 {
		os.Setenv("DLGO_NUM_THREADS", fmt.Sprintf("%d", *threads))
	}

	// Loading animation
	fmt.Printf("Loading %s ", modelPath)
	loadStart := time.Now()
	done := make(chan struct{})
	go func() {
		dots := []string{".", "..", "..."}
		i := 0
		for {
			select {
			case <-done:
				return
			case <-time.After(300 * time.Millisecond):
				fmt.Printf("\rLoading %s %s   ", modelPath, dots[i%len(dots)])
				i++
			}
		}
	}()

	pipe, err := llm.NewPipeline(modelPath, *ctx)
	close(done)
	loadTime := time.Since(loadStart)
	fmt.Printf("\rLoaded %s (%.1fs)                    \n",
		modelPath, loadTime.Seconds())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	runner, gpuName := setupRunner(pipe, *useGPU)

	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = *maxTokens
	cfg.Sampler.Temperature = float32(*temp)
	cfg.Sampler.TopK = *topK
	cfg.Sampler.TopP = float32(*topP)
	cfg.Sampler.MinP = float32(*minP)
	cfg.Sampler.RepetitionPenalty = float32(*repeatPenalty)
	cfg.Seed = *seed

	messages := []llm.Message{{Role: "system", Content: *system}}

	// Header
	fmt.Println()
	fmt.Printf("  Model:     %s\n", pipe.Model.Config.Architecture)
	fmt.Printf("  Params:    %d layers, %d dim, %d heads, vocab %d\n",
		pipe.Model.Config.NumLayers,
		pipe.Model.Config.EmbeddingDim,
		pipe.Model.Config.NumHeads,
		pipe.Model.Config.VocabSize)
	fmt.Printf("  Context:   %d tokens\n", pipe.MaxSeqLen)
	if *useGPU && gpuName != "" {
		fmt.Printf("  Backend:   GPU (%s)\n", gpuName)
	} else {
		numThreads := runtime.GOMAXPROCS(0)
		if *threads > 0 {
			numThreads = *threads
		}
		fmt.Printf("  Backend:   CPU (%d threads)\n", numThreads)
	}
	fmt.Printf("  Sampling:  temp=%.2f top-k=%d top-p=%.2f\n",
		cfg.Sampler.Temperature, cfg.Sampler.TopK, cfg.Sampler.TopP)
	fmt.Println()
	fmt.Println("Type /help for commands, or start chatting.")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024), 1024*1024)

	for {
		fmt.Print(">>> ")
		if !scanner.Scan() {
			fmt.Println()
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		if strings.HasPrefix(input, "/") {
			if handleCommand(input, pipe, messages, &cfg) {
				continue
			}
			if input == "/exit" || input == "/quit" || input == "/bye" {
				break
			}
		}

		if strings.EqualFold(input, "exit") || strings.EqualFold(input, "quit") {
			break
		}

		messages = append(messages, llm.Message{Role: "user", Content: input})
		windowed := applyHistoryWindow(messages, *historyTurns)
		prompt := llm.FormatMessages(pipe.Model.Config, windowed)

		origStream := cfg.Stream
		if !*noStream {
			cfg.Stream = func(tok string) {
				fmt.Print(tok)
			}
		} else {
			cfg.Stream = nil
		}

		genStart := time.Now()
		result, err := runner.generate(prompt, cfg)
		genDuration := time.Since(genStart)
		cfg.Stream = origStream

		if err != nil {
			fmt.Println()
			fmt.Fprintf(os.Stderr, "Error: %v\n\n", err)
			continue
		}

		response := strings.TrimSpace(result.Text)
		if *noStream {
			if response == "" {
				fmt.Print("(empty response)")
			}
			fmt.Println(response)
		} else {
			if response == "" {
				fmt.Print("(empty response)")
			}
			fmt.Println()
		}

		fmt.Printf("\n  %.1f tok/s | %d tokens | %.1fs\n\n",
			result.TokensPerSec,
			result.OutputTokens,
			genDuration.Seconds(),
		)

		messages = append(messages, llm.Message{Role: "assistant", Content: result.Text})
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Input error: %v\n", err)
		os.Exit(1)
	}
}

func handleCommand(input string, pipe *llm.Pipeline, messages []llm.Message, cfg *llm.GenerateConfig) bool {
	switch {
	case input == "/help":
		fmt.Println()
		fmt.Println("  /help          Show this help")
		fmt.Println("  /info          Show model info")
		fmt.Println("  /clear         Clear conversation history")
		fmt.Println("  /set temp N    Set temperature")
		fmt.Println("  /set tokens N  Set max tokens")
		fmt.Println("  /exit          Quit")
		fmt.Println()
		return true

	case input == "/info":
		fmt.Println()
		fmt.Printf("  Architecture:  %s\n", pipe.Model.Config.Architecture)
		fmt.Printf("  Layers:        %d\n", pipe.Model.Config.NumLayers)
		fmt.Printf("  Dimension:     %d\n", pipe.Model.Config.EmbeddingDim)
		fmt.Printf("  Heads:         %d (KV: %d)\n", pipe.Model.Config.NumHeads, pipe.Model.Config.NumKVHeads)
		fmt.Printf("  FFN dim:       %d\n", pipe.Model.Config.FFNDim)
		fmt.Printf("  Vocab:         %d\n", pipe.Model.Config.VocabSize)
		fmt.Printf("  Context:       %d\n", pipe.Model.Config.ContextLength)
		fmt.Printf("  Max tokens:    %d\n", cfg.MaxTokens)
		fmt.Printf("  Temperature:   %.2f\n", cfg.Sampler.Temperature)
		fmt.Println()
		return true

	case input == "/clear":
		if len(messages) > 1 {
			sys := messages[0]
			for i := range messages {
				messages[i] = llm.Message{}
			}
			messages = messages[:1]
			messages[0] = sys
		}
		pipe.KVCache.Reset()
		if pipe.RunState.SSMState != nil {
			pipe.RunState.SSMState.Reset()
		}
		fmt.Println("  Conversation cleared.")
		fmt.Println()
		return true

	case strings.HasPrefix(input, "/set temp "):
		var t float64
		if _, err := fmt.Sscanf(input, "/set temp %f", &t); err == nil {
			cfg.Sampler.Temperature = float32(t)
			fmt.Printf("  Temperature set to %.2f\n\n", t)
		}
		return true

	case strings.HasPrefix(input, "/set tokens "):
		var n int
		if _, err := fmt.Sscanf(input, "/set tokens %d", &n); err == nil && n > 0 {
			cfg.MaxTokens = n
			fmt.Printf("  Max tokens set to %d\n\n", n)
		}
		return true
	}
	return false
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

type turnResult struct {
	Text         string
	TokensPerSec float64
	PrefillMs    float64
	PrefillDelta int
	GenerateMs   float64
	PromptTokens int
	OutputTokens int
}

type generateRunner interface {
	generate(prompt string, cfg llm.GenerateConfig) (*turnResult, error)
}

type cpuRunner struct {
	pipe *llm.Pipeline
}

func (r *cpuRunner) generate(prompt string, cfg llm.GenerateConfig) (*turnResult, error) {
	result, err := r.pipe.GenerateDetailed(prompt, cfg)
	if err != nil {
		return nil, err
	}
	text := strings.TrimSpace(trimStopText(result.Text))
	return &turnResult{
		Text:         text,
		TokensPerSec: result.TokensPerSec,
		PrefillMs:    result.PrefillTimeMs,
		PrefillDelta: result.PromptTokens,
		GenerateMs:   result.GenerateTimeMs,
		PromptTokens: result.PromptTokens,
		OutputTokens: result.TotalTokens,
	}, nil
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

func trimStopText(text string) string {
	for {
		trimmed := strings.TrimRight(text, " \t\r\n")
		changed := false
		for _, ss := range stopStrings() {
			if strings.HasSuffix(trimmed, ss) {
				trimmed = strings.TrimSuffix(trimmed, ss)
				trimmed = strings.TrimRight(trimmed, " \t\r\n")
				changed = true
			}
		}
		if !changed {
			return trimmed
		}
		text = trimmed
	}
}
