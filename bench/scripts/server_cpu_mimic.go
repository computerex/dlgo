//go:build ignore

package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	modelPath := os.Args[1]

	// Mimic server: maxSeqLen=0 → uses model's native context
	fmt.Fprintf(os.Stderr, "Loading model (maxSeqLen=0, native context)...\n")
	pipe, err := llm.NewPipeline(modelPath, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Loaded. MaxSeqLen=%d, Arch=%s\n", pipe.MaxSeqLen, pipe.Model.Config.Architecture)
	fmt.Fprintf(os.Stderr, "EOS=%d, StopTokens=%v, SuppressTokens=%v\n",
		pipe.Model.Config.EOS, pipe.Model.Config.StopTokens, pipe.Model.Config.SuppressTokens)

	// Mimic server: format prompt via FormatMessages (enable_thinking=nil → auto)
	msgs := []llm.Message{{Role: "user", Content: "Explain Python decorators in 3 sentences."}}
	prompt := llm.FormatMessages(pipe.Model.Config, msgs)
	fmt.Fprintf(os.Stderr, "Prompt: %q\n", prompt)

	tokens := pipe.Tokenizer.Encode(prompt)
	fmt.Fprintf(os.Stderr, "Prompt tokens: %d\n", len(tokens))

	// Test 1: mimic server exactly (temp=0.6, random seed)
	fmt.Fprintf(os.Stderr, "\n=== TEST 1: Server-like (temp=0.6, random seed) ===\n")
	runTest(pipe, tokens, 0.6, -1)

	// Test 2: e2e_think.go style (temp=0.7, seed=42)
	fmt.Fprintf(os.Stderr, "\n=== TEST 2: e2e_think style (temp=0.7, seed=42) ===\n")
	runTest(pipe, tokens, 0.7, 42)

	// Test 3: server style with seed=42
	fmt.Fprintf(os.Stderr, "\n=== TEST 3: Server-like (temp=0.6, seed=42) ===\n")
	runTest(pipe, tokens, 0.6, 42)

	// Test 4: maxSeqLen=2048 (like e2e_think.go)
	fmt.Fprintf(os.Stderr, "\n=== TEST 4: Load with maxSeqLen=2048 ===\n")
	pipe2, err := llm.NewPipeline(modelPath, 2048)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "MaxSeqLen=%d\n", pipe2.MaxSeqLen)
	tokens2 := pipe2.Tokenizer.Encode(prompt)
	runTestPipe(pipe2, tokens2, 0.6, -1)
}

func runTest(pipe *llm.Pipeline, tokens []int32, temp float32, seed int64) {
	runTestPipe(pipe, tokens, temp, seed)
}

func runTestPipe(pipe *llm.Pipeline, tokens []int32, temp float32, seed int64) {
	pipe.KVCache.Reset()
	if pipe.RunState.SSMState != nil {
		pipe.RunState.SSMState.Reset()
	}

	llm.ForwardBatch(pipe.Model, tokens, 0, pipe.KVCache, pipe.RunState, pipe.BatchState)

	pos := len(tokens)
	var recentTokens []int32

	llm.ApplySuppressTokens(pipe.RunState.Logits, pipe.Model.Config.SuppressTokens)

	// Show top 5 logits before sampling
	showTopLogits(pipe.RunState.Logits, pipe, 10)

	sampler := ops.DefaultSamplerConfig()
	sampler.Temperature = temp

	rng := rand.New(rand.NewSource(seed))
	if seed < 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	nextToken := int32(ops.SampleToken(pipe.RunState.Logits, sampler, recentTokens, rng))
	fmt.Fprintf(os.Stderr, "First token: id=%d text=%q\n", nextToken, pipe.Tokenizer.DecodeToken(nextToken))

	isStop := nextToken == pipe.Model.Config.EOS
	for _, st := range pipe.Model.Config.StopTokens {
		if nextToken == st {
			isStop = true
		}
	}
	if isStop {
		fmt.Fprintf(os.Stderr, "First token IS a stop token! Generation would stop with 0 output.\n")
		return
	}

	recentTokens = append(recentTokens, nextToken)

	// Generate up to 20 tokens
	for step := 1; step < 20; step++ {
		if pos >= pipe.MaxSeqLen-1 {
			fmt.Fprintf(os.Stderr, "Hit maxSeqLen\n")
			break
		}

		llm.Forward(pipe.Model, nextToken, pos, pipe.KVCache, pipe.RunState)
		pos++

		llm.ApplySuppressTokens(pipe.RunState.Logits, pipe.Model.Config.SuppressTokens)
		nextToken = int32(ops.SampleToken(pipe.RunState.Logits, sampler, recentTokens, rng))

		text := pipe.Tokenizer.DecodeToken(nextToken)
		fmt.Fprintf(os.Stderr, "  tok[%d]: id=%d text=%q\n", step, nextToken, text)

		isStop = nextToken == pipe.Model.Config.EOS
		for _, st := range pipe.Model.Config.StopTokens {
			if nextToken == st {
				isStop = true
			}
		}
		if isStop {
			fmt.Fprintf(os.Stderr, "  → STOP token at step %d\n", step)
			break
		}
		recentTokens = append(recentTokens, nextToken)
	}
}

func showTopLogits(logits []float32, pipe *llm.Pipeline, n int) {
	type kv struct {
		id    int
		logit float32
	}
	top := make([]kv, 0, n)
	for i, v := range logits {
		if math.IsInf(float64(v), -1) {
			continue
		}
		if len(top) < n {
			top = append(top, kv{i, v})
		} else {
			minIdx := 0
			for j := 1; j < len(top); j++ {
				if top[j].logit < top[minIdx].logit {
					minIdx = j
				}
			}
			if v > top[minIdx].logit {
				top[minIdx] = kv{i, v}
			}
		}
	}
	// Sort descending
	for i := 0; i < len(top); i++ {
		for j := i + 1; j < len(top); j++ {
			if top[j].logit > top[i].logit {
				top[i], top[j] = top[j], top[i]
			}
		}
	}
	fmt.Fprintf(os.Stderr, "Top %d logits:\n", n)
	for _, kv := range top {
		text := pipe.Tokenizer.DecodeToken(int32(kv.id))
		isEOS := int32(kv.id) == pipe.Model.Config.EOS
		fmt.Fprintf(os.Stderr, "  id=%6d logit=%8.3f text=%q%s\n", kv.id, kv.logit, text, map[bool]string{true: " *** EOS ***", false: ""}[isEOS])
	}
}
