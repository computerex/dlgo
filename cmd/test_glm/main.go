package main

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/computerex/dlgo/models/llm"
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
		fmt.Println("Usage: test_glm <model.gguf>")
		os.Exit(1)
	}
	nproc := runtime.NumCPU()
	runtime.GOMAXPROCS(nproc)
	fmt.Printf("Using %d CPU threads\n", nproc)

	fmt.Println("Loading pipeline (maxSeqLen=128)...")
	pipe, err := llm.NewPipeline(os.Args[1], 128)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	defer pipe.Model.Close()
	cfg := pipe.Model.Config

	fmt.Printf("Model: %s (%d layers, %d dim, %d heads, vocab %d)\n",
		cfg.Architecture, cfg.NumLayers, cfg.EmbeddingDim, cfg.NumHeads, cfg.VocabSize)
	fmt.Printf("MLA: qLORARank=%d kvLORARank=%d qkNope=%d qkRope=%d vHeadDim=%d\n",
		cfg.QLORARank, cfg.KVLORARank, cfg.QKNopeDim, cfg.QKRopeDim, cfg.VHeadDim)
	fmt.Printf("HeadDim=%d NumKVHeads=%d FFNDim=%d\n", cfg.HeadDim, cfg.NumKVHeads, cfg.FFNDim)
	fmt.Printf("RopeNeox=%v RopeFreqBase=%f RopeDim=%d\n", cfg.RopeNeox, cfg.RopeFreqBase, cfg.RopeDim)
	fmt.Printf("Experts=%d UsedExperts=%d ExpertFFNDim=%d SharedExpertFFNDim=%d\n",
		cfg.ExpertCount, cfg.ExpertUsedCount, cfg.ExpertFFNDim, cfg.SharedExpertFFNDim)

	tok := pipe.Tokenizer
	prompt := llm.FormatChat(cfg, "", "What is 2+2?")
	tokens := tok.Encode(prompt)
	fmt.Printf("Prompt: %q -> %d tokens\n", prompt, len(tokens))

	rs := pipe.RunState

	// Process tokens one-by-one and dump diagnostics on last prompt token
	var logits []float32
	for i, t := range tokens {
		logits = llm.Forward(pipe.Model, t, i, pipe.KVCache, rs)
		if i == len(tokens)-1 {
			fmt.Printf("\n=== Diagnostics after last prompt token (pos=%d) ===\n", i)
			fmt.Printf("X norm: %.6f\n", l2norm(rs.X))
			fmt.Printf("XNorm norm: %.6f\n", l2norm(rs.XNorm))
			fmt.Printf("AttnProj norm: %.6f\n", l2norm(rs.AttnProj))
			fmt.Printf("FFNOut norm: %.6f\n", l2norm(rs.FFNOut))
			fmt.Printf("FFNIn norm: %.6f\n", l2norm(rs.FFNIn))
			if len(rs.MLAQComp) > 0 {
				fmt.Printf("MLAQComp norm: %.6f\n", l2norm(rs.MLAQComp))
				fmt.Printf("MLAQAbsorbed norm: %.6f\n", l2norm(rs.MLAQAbsorbed))
				fmt.Printf("MLAAttnKV norm: %.6f\n", l2norm(rs.MLAAttnKV))
			}
			fmt.Printf("Logits[0:5]: %v\n", logits[:5])
			fmt.Printf("Logits max=%.4f min=%.4f\n", maxF(logits), minF(logits))
		}
	}

	maxGen := 30
	pos := len(tokens)
	var generated []int32
	fmt.Print("\nOutput: ")
	genStart := time.Now()
	for g := 0; g < maxGen && pos < pipe.MaxSeqLen; g++ {
		maxIdx := argmax(logits)
		if maxIdx == int(tok.EOS) {
			fmt.Print("[EOS]")
			break
		}
		generated = append(generated, int32(maxIdx))
		word := tok.DecodeToken(int32(maxIdx))
		fmt.Print(word)
		logits = llm.Forward(pipe.Model, int32(maxIdx), pos, pipe.KVCache, rs)
		pos++
	}
	genTime := time.Since(genStart)
	fmt.Println()
	fmt.Printf("Generated %d tokens in %v (%.1f tok/s)\n", len(generated), genTime, float64(len(generated))/genTime.Seconds())
	fullOutput := tok.Decode(generated)
	fmt.Printf("Full output: %q\n", strings.TrimSpace(fullOutput))
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

func maxF(x []float32) float32 {
	m := x[0]
	for _, v := range x[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func minF(x []float32) float32 {
	m := x[0]
	for _, v := range x[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func init() {
	// Verify KV cache size is correct for MLA
}
