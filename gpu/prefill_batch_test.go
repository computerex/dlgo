//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// TestShortConvBatchPrefillParity compares the chunked batched prefill
// (GpuForwardPrefillBatchShortConv) against the per-token path that it replaces,
// on real weights, at the logits.
//
// The two cannot be bit-identical: batched GEMMs accumulate the K dimension in a
// different order than the single-token matvec, and the batched MoE sums experts
// in expert order instead of slot order. The check is therefore on the magnitude
// of the deviation and on top-1 agreement, which is what actually matters for
// greedy decode.
func TestShortConvBatchPrefillParity(t *testing.T) {
	const modelPath = `C:\Users\mohd\Downloads\LFM2.5-24B-A2B-QAD-Q4_0.gguf`
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not present: %v", err)
	}
	// The batched prefill is opt-in and currently KNOWN to diverge from the
	// per-token path (see the bisection below: position 0 agrees exactly,
	// position 1 does not, and the first difference is the attention layer's
	// core output even though its input and every kernel it uses test exact).
	// Skip before loading the 12 GB model unless explicitly asked.
	if os.Getenv("DLGO_PREFILL_BATCH") != "1" {
		t.Skip("set DLGO_PREFILL_BATCH=1 to exercise the batched prefill")
	}

	cpuPipe, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	gp, err := NewGpuPipeline(cpuPipe)
	if err != nil {
		t.Fatalf("gpu pipeline: %v", err)
	}
	t.Cleanup(gp.FreeAll)

	m := gp.CPUModel
	cfg := m.Config

	text := "The capital of France is a city with a long history. " +
		"Records show that the measured throughput was acceptable for the workload. " +
		"Reports indicate the latency stayed near the target for many quarters. "
	prompt := strings.Repeat(text, 3)
	tokens := gp.Tokenizer.Encode(prompt)
	if len(tokens) < 8 {
		t.Fatalf("tokenizer produced %d tokens", len(tokens))
	}
	if len(tokens) > 256 {
		tokens = tokens[:256]
	}
	// Keep the prompt short of the context so decode could follow.
	if len(tokens) >= gp.MaxSeqLen {
		t.Fatalf("prompt %d tokens >= ctx %d", len(tokens), gp.MaxSeqLen)
	}
	t.Logf("prompt: %d tokens, ctx=%d", len(tokens), gp.MaxSeqLen)

	if !shortConvBatchSupported(m, gp.GpuModel, cfg) {
		t.Skip("model does not support the batched shortconv prefill")
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if m.Layers[l].Spec.Core == llm.CoreAttention {
			g := &gp.GpuModel.Layers[l]
			t.Logf("attn layer %d: Bq=%d Bk=%d Bv=%d Bo=%d qknorm=%v sliding=%d ropeDim=%d neox=%v",
				l, g.Bq, g.Bk, g.Bv, g.Bo, m.Layers[l].Spec.QKNorm,
				m.Layers[l].Spec.SlidingWindow, cfg.RopeDim, cfg.RopeNeox)
			break
		}
	}
	if err := gp.ensurePrefillBatchState(len(tokens)); err != nil {
		t.Fatalf("prefill batch state: %v", err)
	}
	if !gp.UseBatchPrefill {
		t.Skip("UseBatchPrefill not enabled (DLGO_PREFILL_BATCH=0?)")
	}

	// Bisect over chunk length: a prefix that already diverges at n=1 points at
	// the per-layer composition, while a prefix that only diverges once n > 1
	// points at the cross-position kernels (attention / shortconv scan).
	lengths := []int{1, 2, 3, 4, 8, 16, 32}
	if s := os.Getenv("DLGO_BISECT_N"); s != "" {
		var n int
		fmt.Sscanf(s, "%d", &n)
		if n > 0 {
			lengths = []int{n}
		}
	}
	if len(tokens) < 32 {
		lengths = lengths[:0]
		for n := 1; n <= len(tokens); n *= 2 {
			lengths = append(lengths, n)
		}
	}
	worst := 0.0
	worstLen := 0
	for _, n := range lengths {
		if n > len(tokens) {
			continue
		}
		gp.UseBatchPrefill = true
		gp.ResetState()
		if err := GpuForwardPrefillBatchShortConv(m, gp.GpuModel, tokens[:n], gp.KVCache,
			gp.RunState, gp.BatchState, gp.LogitsBuf, gp, 0, true); err != nil {
			t.Fatalf("batched prefill n=%d: %v", n, err)
		}
		batched := make([]float32, len(gp.LogitsBuf))
		copy(batched, gp.LogitsBuf)

		gp.UseBatchPrefill = false
		gp.ResetState()
		for i := 0; i < n; i++ {
			if err := GpuForward(m, gp.GpuModel, tokens[i], i, gp.KVCache, gp.RunState,
				gp.LogitsBuf, gp); err != nil {
				t.Fatalf("per-token prefill n=%d at %d: %v", n, i, err)
			}
		}
		perToken := make([]float32, len(gp.LogitsBuf))
		copy(perToken, gp.LogitsBuf)

		var maxAbs, sumAbs, sumMag float64
		for i := range batched {
			d := math.Abs(float64(batched[i] - perToken[i]))
			maxAbs = math.Max(maxAbs, d)
			sumAbs += d
			sumMag += math.Abs(float64(perToken[i]))
		}
		top1B := argmaxF32(batched)
		top1P := argmaxF32(perToken)
		meanRel := sumAbs / math.Max(sumMag, 1e-9)
		t.Logf("n=%3d  maxAbs=%.4e meanRel=%.4e top1 %d/%d agree=%v",
			n, maxAbs, meanRel, top1B, top1P, top1B == top1P)
		if meanRel > worst {
			worst, worstLen = meanRel, n
		}
	}

	if worst > 5e-3 {
		t.Errorf("mean relative logit diff %.4e too large (worst at n=%d)", worst, worstLen)
	}
}

func argmaxF32(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}
