package main

import (
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

// runParity feeds the same token sequence through the CPU and GPU per-token
// forward passes in lockstep and compares the logits after every position.
//
// Both paths see identical inputs and identical KV/conv state history, so a
// divergence at step k localises the defect: step 0 exercises only the
// projections, norms and MoE (attention is a no-op and the short-conv history
// is empty), while later steps additionally exercise the KV cache and the
// rolling short-conv window.
func runParity(p *llm.Pipeline, prompt string, chat bool) {
	cfg := p.Model.Config
	text := prompt
	if chat {
		text = llm.FormatChat(cfg, "", prompt)
	}
	ids := p.Tokenizer.Encode(text)
	if len(ids) == 0 {
		fmt.Fprintln(os.Stderr, "parity: prompt tokenized to zero tokens")
		os.Exit(1)
	}
	fmt.Printf("== parity over %d prompt tokens\n", len(ids))
	parityTok = p.Tokenizer

	// --- CPU pass first: NewGpuPipeline releases the CPU scratch buffers. ---
	cpuLogits := make([][]float32, len(ids))
	p.ResetSequenceState()
	for i, id := range ids {
		out := llm.Forward(p.Model, id, i, p.KVCache, p.RunState)
		cpuLogits[i] = append([]float32(nil), out...)
	}

	gp, err := gpu.NewGpuPipeline(p)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parity: gpu pipeline failed: %v\n", err)
		os.Exit(1)
	}
	defer gp.FreeAll()
	gp.ResetState()

	const topN = 5
	worst := 0.0
	fails := 0
	for i, id := range ids {
		if err := gpu.GpuForward(p.Model, gp.GpuModel, id, i, gp.KVCache, gp.RunState, gp.LogitsBuf, gp); err != nil {
			fmt.Fprintf(os.Stderr, "parity: gpu forward at %d failed: %v\n", i, err)
			os.Exit(1)
		}
		gpu.Sync()
		gpuLogits := append([]float32(nil), gp.LogitsBuf...)

		cpuTop := topTokens(cpuLogits[i], p, topN)
		gpuTop := topTokens(gpuLogits, p, topN)

		maxDiff := 0.0
		var sumAbs, sumRef float64
		for j := range cpuLogits[i] {
			d := math.Abs(float64(gpuLogits[j] - cpuLogits[i][j]))
			if d > maxDiff {
				maxDiff = d
			}
			sumAbs += d
			sumRef += math.Abs(float64(cpuLogits[i][j]))
		}
		if maxDiff > worst {
			worst = maxDiff
		}
		agree := cpuTop[0].id == gpuTop[0].id
		if !agree {
			fails++
		}
		fmt.Printf("tok %2d id=%-7d maxAbsDiff=%8.4f meanAbsDiff=%8.5f top1agree=%v\n",
			i, id, maxDiff, sumAbs/float64(len(cpuLogits[i])), agree)
		if !agree || maxDiff > 1.0 {
			fmt.Printf("        cpu top: %s\n", fmtTops(cpuTop))
			fmt.Printf("        gpu top: %s\n", fmtTops(gpuTop))
		}
	}
	fmt.Printf("== worst maxAbsDiff=%.4f, top1 disagreements=%d/%d\n", worst, fails, len(ids))
	if fails > 0 {
		fmt.Println("VERDICT: DIVERGED")
		os.Exit(2)
	}
	fmt.Println("VERDICT: logits agree at every position")
}

type tokScore struct {
	id int
	l  float32
}

func topTokens(logits []float32, p *llm.Pipeline, n int) []tokScore {
	out := make([]tokScore, 0, n)
	for i, v := range logits {
		out = append(out, tokScore{i, v})
	}
	sort.Slice(out, func(a, b int) bool { return out[a].l > out[b].l })
	if len(out) > n {
		out = out[:n]
	}
	return out
}

func fmtTops(ts []tokScore) string {
	s := ""
	for _, t := range ts {
		s += fmt.Sprintf("[%d %q %.3f] ", t.id, decodeTok(t.id), t.l)
	}
	return s
}

var parityTok *llm.Tokenizer

func decodeTok(id int) string {
	if parityTok == nil {
		return ""
	}
	return parityTok.DecodeToken(int32(id))
}
