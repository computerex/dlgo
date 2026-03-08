//go:build ignore

package main

import (
	"fmt"
	"os"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	gguf := `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`
	pipe, err := llm.NewPipeline(gguf, 512)
	if err != nil {
		fmt.Println("ERROR:", err)
		return
	}

	cfg := pipe.Model.Config
	kvDim := cfg.NumKVHeads * cfg.HeadDim

	prompt := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 20) + "Summarize the above."
	formatted := llm.FormatChat(cfg, "", prompt)
	tokens := pipe.Tokenizer.Encode(formatted)
	fmt.Printf("Tokens: %d\n", len(tokens))

	rs := llm.NewRunState(cfg, 512)
	kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)
	bs := llm.NewBatchState(cfg, 512)

	// Warmup
	llm.ForwardBatch(pipe.Model, tokens, 0, kv, rs, bs)

	// Profile
	f, _ := os.Create("cpu_prefill.prof")
	pprof.StartCPUProfile(f)

	kv.Reset()
	start := time.Now()
	for i := 0; i < 5; i++ {
		kv.Reset()
		llm.ForwardBatch(pipe.Model, tokens, 0, kv, rs, bs)
	}
	elapsed := time.Since(start)

	pprof.StopCPUProfile()
	f.Close()

	fmt.Printf("5 iterations of %d tokens prefill: %v (avg %.1fms)\n", len(tokens), elapsed, float64(elapsed.Milliseconds())/5.0)
	fmt.Println("Profile saved to cpu_prefill.prof")
	fmt.Println("Run: go tool pprof -top cpu_prefill.prof")
}
