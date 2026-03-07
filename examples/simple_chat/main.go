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
	"os"
	"runtime"
	"strings"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
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
	pipe *llm.Pipeline
}

type gpuChatRunner struct {
	cpuPipe *llm.Pipeline
	gpuPipe *gpu.GpuPipeline
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

func (r *gpuChatRunner) generate(prompt string, cfg llm.GenerateConfig) (*turnResult, error) {
	result, err := r.gpuPipe.GenerateDetailed(prompt, cfg)
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

func main() {
	ctx := flag.Int("ctx", 8192, "runtime context length (tokens)")
	maxTokens := flag.Int("max-tokens", 256, "max tokens per assistant response")
	temp := flag.Float64("temp", 0.7, "sampling temperature (0 = greedy)")
	topK := flag.Int("top-k", 40, "top-k sampling (0 = disabled)")
	topP := flag.Float64("top-p", 0.9, "top-p nucleus sampling (1.0 = disabled)")
	minP := flag.Float64("min-p", 0.0, "min-p sampling threshold (0 = disabled)")
	repeatPenalty := flag.Float64("repeat-penalty", 1.1, "repetition penalty (1.0 = disabled)")
	seed := flag.Int64("seed", -1, "random seed (-1 = random)")
	system := flag.String("system", "You are a helpful assistant.", "system prompt")
	threads := flag.Int("threads", 0, "worker threads (0 = auto, try 128 on this machine)")
	historyTurns := flag.Int("history-turns", 6, "number of recent user/assistant turns to keep")
	useGPU := flag.Bool("gpu", false, "use Vulkan GPU backend")
	stream := flag.Bool("stream", true, "stream tokens as they are generated")
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
	if *topK < 0 {
		fmt.Fprintln(os.Stderr, "Error: --top-k must be >= 0")
		os.Exit(1)
	}
	if *topP <= 0 || *topP > 1 {
		fmt.Fprintln(os.Stderr, "Error: --top-p must be in (0, 1]")
		os.Exit(1)
	}
	if *minP < 0 || *minP > 1 {
		fmt.Fprintln(os.Stderr, "Error: --min-p must be in [0, 1]")
		os.Exit(1)
	}
	if *repeatPenalty < 1.0 {
		fmt.Fprintln(os.Stderr, "Error: --repeat-penalty must be >= 1.0")
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
	cpuRunner := &chatRunner{pipe: pipe}
	var runner interface {
		generate(string, llm.GenerateConfig) (*turnResult, error)
	}
	runner = cpuRunner
	var gpuRunner *gpuChatRunner
	if *useGPU {
		if err := gpu.Init(); err != nil {
			fmt.Fprintf(os.Stderr, "Error initializing GPU: %v\n", err)
			os.Exit(1)
		}
		gp, err := gpu.NewGpuPipeline(pipe)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating GPU pipeline: %v\n", err)
			os.Exit(1)
		}
		gpuRunner = &gpuChatRunner{cpuPipe: pipe, gpuPipe: gp}
		runner = gpuRunner
	}

	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = *maxTokens
	cfg.Sampler.Temperature = float32(*temp)
	cfg.Sampler.TopK = *topK
	cfg.Sampler.TopP = float32(*topP)
	cfg.Sampler.MinP = float32(*minP)
	cfg.Sampler.RepetitionPenalty = float32(*repeatPenalty)
	cfg.Seed = *seed

	messages := []llm.Message{{Role: "system", Content: *system}}

	fmt.Printf("Model: %s (%d layers, %d dim, %d heads, vocab %d, ctx %d)\n",
		pipe.Model.Config.Architecture,
		pipe.Model.Config.NumLayers,
		pipe.Model.Config.EmbeddingDim,
		pipe.Model.Config.NumHeads,
		pipe.Model.Config.VocabSize,
		pipe.Model.Config.ContextLength,
	)
	fmt.Printf("Runtime context (--ctx): %d tokens\n", pipe.MaxSeqLen)
	if *useGPU {
		fmt.Printf("Backend (--gpu): Vulkan GPU (%s)\n", gpu.DeviceName())
	} else {
		fmt.Println("Backend (--gpu): CPU")
	}
	if *threads > 0 {
		fmt.Printf("Workers (--threads): %d\n", *threads)
	} else {
		fmt.Printf("Workers (--threads): auto (%d)\n", runtime.GOMAXPROCS(0))
	}
	fmt.Printf("Generation: max-tokens=%d temp=%.2f top-k=%d top-p=%.2f min-p=%.2f repeat-penalty=%.2f seed=%d\n",
		cfg.MaxTokens,
		cfg.Sampler.Temperature,
		cfg.Sampler.TopK,
		cfg.Sampler.TopP,
		cfg.Sampler.MinP,
		cfg.Sampler.RepetitionPenalty,
		cfg.Seed,
	)
	fmt.Printf("Streaming (--stream): %v\n", *stream)
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

		fmt.Print("AI> ")
		origStream := cfg.Stream
		if *stream {
			cfg.Stream = func(tok string) {
				fmt.Print(tok)
			}
		} else {
			cfg.Stream = nil
		}
		result, err := runner.generate(prompt, cfg)
		cfg.Stream = origStream
		if err != nil {
			fmt.Println()
			fmt.Fprintf(os.Stderr, "Error generating response: %v\n", err)
			continue
		}

		responseRaw := result.Text
		responseDisplay := strings.TrimSpace(responseRaw)
		if *stream {
			if responseDisplay == "" {
				fmt.Print("(empty response)")
			}
			fmt.Println()
		} else {
			if responseDisplay == "" {
				responseDisplay = "(empty response)"
			}
			fmt.Println(responseDisplay)
		}
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
