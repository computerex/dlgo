//go:build cgo && vulkan

package gpu

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// TestPrefillMoEGroupedMatchesPerExpert compares the whole 40-layer chunked
// prefill with the grouped MoE against the per-expert loop, at identical chunking.
//
// It is a *smoke* check, deliberately: the numbers it reports (~1e-1 mean rel on
// this model) are dominated by discrete MoE routing flips, not by the two
// implementations. A ~4e-4 per-layer difference in the expert arithmetic is
// enough to flip a hair-thin top-4 tie a layer later, and one flipped expert moves
// the final logits by O(1) — so over 38 MoE layers the same two implementations
// can land 1e-3 or 1e-1 apart depending on where the ties happen to sit.
//
// The tight checks are:
//   - TestMoEGroupedVsPerExpert: every MoE layer, same activations, same routing,
//     grouped vs per-expert <= 1e-3 — this is the correctness proof;
//   - chunk-size invariance in TestShortConvBatchPrefillParity: exactly 0.
//
// What this test adds is the end-to-end assertion that matters practically: the
// top-1 token does not change, i.e. the grouped MoE answers the same question.
func TestPrefillMoEGroupedMatchesPerExpert(t *testing.T) {
	const modelPath = `C:\Users\mohd\Downloads\LFM2.5-24B-A2B-QAD-Q4_0.gguf`
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not present: %v", err)
	}
	if os.Getenv("DLGO_PREFILL_TEST") != "1" {
		t.Skip("set DLGO_PREFILL_TEST=1 to load the 12 GB model and run this")
	}
	if os.Getenv("DLGO_PREFILL_BATCH") == "0" {
		t.Skip("DLGO_PREFILL_BATCH=0 disables the batched prefill")
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
	text := "The capital of France is a city with a long history. " +
		"Records show that the measured throughput was acceptable for the workload. " +
		"Reports indicate the latency stayed near the target for many quarters. "
	tokens := gp.Tokenizer.Encode(strings.Repeat(text, 3))
	if len(tokens) > 128 {
		tokens = tokens[:128]
	}
	if err := gp.ensurePrefillBatchState(len(tokens)); err != nil {
		t.Fatalf("prefill batch state: %v", err)
	}
	if !gp.UseBatchPrefill {
		t.Skip("UseBatchPrefill not enabled")
	}
	// The grouped path needs its scratch; without it this test would silently
	// compare the per-expert loop against itself.
	if moeGroupedForce == 0 || gp.BatchState.MoEPairOut == 0 {
		t.Fatalf("grouped MoE scratch missing: the comparison would be trivial")
	}

	run := func(toks []int32) []float32 {
		gp.ResetState()
		if err := GpuForwardPrefillBatchShortConv(m, gp.GpuModel, toks, gp.KVCache,
			gp.RunState, gp.BatchState, gp.LogitsBuf, gp, 0, true); err != nil {
			t.Fatalf("prefill: %v", err)
		}
		out := make([]float32, len(gp.LogitsBuf))
		copy(out, gp.LogitsBuf)
		return out
	}

	prev := moeGroupedForce
	t.Cleanup(func() { moeGroupedForce = prev })

	for _, n := range []int{1, 2, 4, 8, 16, 32} {
		if n > len(tokens) {
			continue
		}
		chunk := tokens[:n]

		moeGroupedForce = 1
		grouped := run(chunk)
		moeGroupedForce = 0
		perExpert := run(chunk)

		var maxAbs, sumAbs, sumMag float64
		for i := range grouped {
			d := math.Abs(float64(grouped[i] - perExpert[i]))
			maxAbs = math.Max(maxAbs, d)
			sumAbs += d
			sumMag += math.Abs(float64(perExpert[i]))
		}
		meanRel := sumAbs / math.Max(sumMag, 1e-9)
		agree := argmaxF32Equal(grouped, perExpert)
		t.Logf("n=%3d grouped-vs-perexpert maxAbs=%.4e meanRel=%.4e top1 agree=%v",
			n, maxAbs, meanRel, agree)

		if !agree {
			t.Errorf("n=%d: grouped and per-expert MoE disagree on the top-1 token", n)
		}
	}
}
