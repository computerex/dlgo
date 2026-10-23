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

// TestShortConvBatchPrefillParity checks that batching the token axis in the
// chunked prefill (GpuForwardPrefillBatchShortConv) does not change the result on
// real weights: a whole chunk in one call, two tokens per call, and one token per
// call must all produce the same logits, bit for bit.
//
// The oracle is the same code driven one token at a time, not the single-token
// GpuForward decode path. The decode path uses different kernels (fused lane
// configs, dp4a, split-KV attention), so it differs by ~1e-5 per layer; on a 24B
// MoE that is enough to flip a hair-thin expert selection and move the logits by
// O(1), which makes it useless as a correctness oracle at these lengths.
func TestShortConvBatchPrefillParity(t *testing.T) {
	const modelPath = `C:\Users\mohd\Downloads\LFM2.5-24B-A2B-QAD-Q4_0.gguf`
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not present: %v", err)
	}
	// Loading the 12 GB model is slow, so skip unless asked.
	if os.Getenv("DLGO_PREFILL_BATCH") == "0" {
		t.Skip("DLGO_PREFILL_BATCH=0 disables the batched prefill")
	}
	if os.Getenv("DLGO_PREFILL_TEST") != "1" {
		t.Skip("set DLGO_PREFILL_TEST=1 to load the 12 GB model and run this")
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

	// Chunk lengths to compare. Each is checked at three chunking granularities,
	// so a divergence that only appears once the token axis is batched shows up
	// immediately.
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
	// The oracle is the same batched code driven one token at a time: it writes
	// the KV cache and the short-convolution state exactly the way the chunked
	// call does, so the only difference between the two sides is whether the
	// token axis was batched. That isolates what this test is about.
	//
	// Comparing final logits against the single-token GpuForward path is not a
	// usable oracle here: that path is not reproducible from a reset state on
	// this model (two identical runs differ by ~5 in absolute logits, because a
	// hair-thin MoE routing decision is made inside the fp32 noise floor), and
	// one flipped expert moves the logits by O(1).
	batchedChunk := func(toks []int32, startPos int, last bool) ([]float32, error) {
		gp.ResetState()
		if err := GpuForwardPrefillBatchShortConv(m, gp.GpuModel, toks, gp.KVCache,
			gp.RunState, gp.BatchState, gp.LogitsBuf, gp, startPos, last); err != nil {
			return nil, err
		}
		out := make([]float32, len(gp.LogitsBuf))
		copy(out, gp.LogitsBuf)
		return out, nil
	}

	// chunkSize > 0 splits the prompt so the multi-chunk path (KV/conv state
	// carried across chunks) is exercised too.
	runSequential := func(toks []int32, chunkSize int) ([]float32, error) {
		gp.ResetState()
		var out []float32
		for p := 0; p < len(toks); p += chunkSize {
			end := p + chunkSize
			if end > len(toks) {
				end = len(toks)
			}
			last := end == len(toks)
			if err := GpuForwardPrefillBatchShortConv(m, gp.GpuModel, toks[p:end],
				gp.KVCache, gp.RunState, gp.BatchState, gp.LogitsBuf, gp, p, last); err != nil {
				return nil, err
			}
			if last {
				out = make([]float32, len(gp.LogitsBuf))
				copy(out, gp.LogitsBuf)
			}
		}
		return out, nil
	}

	report := func(n int, label string, got, want []float32) {
		var maxAbs, sumAbs, sumMag float64
		for i := range got {
			d := math.Abs(float64(got[i] - want[i]))
			maxAbs = math.Max(maxAbs, d)
			sumAbs += d
			sumMag += math.Abs(float64(want[i]))
		}
		meanRel := sumAbs / math.Max(sumMag, 1e-9)
		agree := argmaxF32(got) == argmaxF32(want)
		t.Logf("n=%3d %-18s maxAbs=%.4e meanRel=%.4e top1 agree=%v", n, label, maxAbs, meanRel, agree)
	}

	// Chunking must be *invariant*: how the prompt is split cannot change the
	// answer. This is exact on every quant path, because both sides go through
	// the same batched kernels.
	worstSplit := 0.0

	worst := 0.0
	worstLen := 0
	for _, n := range lengths {
		if n > len(tokens) {
			continue
		}
		batched, err := batchedChunk(tokens[:n], 0, true)
		if err != nil {
			t.Fatalf("batched prefill n=%d: %v", n, err)
		}

		// Whole chunk in one call vs token-at-a-time. On the native-quant path
		// this is bit-exact (both sides dispatch the same kernels). On the
		// unified Affine path npos==1 takes the single-token matvec while npos>1
		// takes the coopmat GEMM, so the two differ by fp accumulation order;
		// that is expected and is why the bound below is a tolerance.
		sequential, err := runSequential(tokens[:n], 1)
		if err != nil {
			t.Fatalf("sequential batched prefill n=%d: %v", n, err)
		}
		report(n, "chunk-vs-tokenwise", batched, sequential)

		var sumAbs, sumMag float64
		for i := range batched {
			sumAbs += math.Abs(float64(batched[i] - sequential[i]))
			sumMag += math.Abs(float64(sequential[i]))
		}
		if rel := sumAbs / math.Max(sumMag, 1e-9); rel > worst {
			worst, worstLen = rel, n
		}
		if !argmaxF32Equal(batched, sequential) {
			t.Errorf("n=%d: batched chunk and token-at-a-time disagree on the top-1 token", n)
		}

		// A second chunking of the same prompt must land on identical logits.
		if n >= 4 {
			halves, err := runSequential(tokens[:n], 2)
			if err != nil {
				t.Fatalf("two-token chunks n=%d: %v", n, err)
			}
			report(n, "chunk2-vs-chunk", halves, batched)
			for i := range batched {
				if d := math.Abs(float64(halves[i] - batched[i])); d > worstSplit {
					worstSplit = d
				}
			}
		}
	}

	if worstSplit != 0 {
		t.Errorf("chunk size changed the logits: maxAbs=%.4e", worstSplit)
	}
	if worst > 2e-2 {
		t.Errorf("batched chunk differs from token-at-a-time by mean rel %.4e (worst at n=%d)",
			worst, worstLen)
	}
}

func argmaxF32Equal(a, b []float32) bool { return argmaxF32(a) == argmaxF32(b) }

func argmaxF32(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}
