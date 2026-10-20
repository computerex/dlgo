//go:build cgo && vulkan

package gpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestAttentionDecodeGQAParity checks the GQA-folded split-KV decode kernel
// (attention_decode_gqa.comp) against the tiled kernel (attention_tiled.comp),
// and separately against the per-q-head split kernel it supersedes
// (attention_decode_split.comp).
//
// The folded kernel is the point of interest: one workgroup owns a whole KV head
// and every q-head sharing it, so the KV cache is staged and read once per
// position instead of once per q-head, and the QK dot is no longer recomputed by
// all 8 subgroups. Shapes cover kv_mul 2, 4 and 8 so the fold is exercised at
// every supported grouping (kv_mul * head_dim/8 must fit 256 PV threads).
//
// AttentionDecodeUsed() asserts the intended kernel actually ran; without it a
// shape-gate regression would silently compare a kernel with itself and pass.
//
// Measured performance (end-to-end, not a microbenchmark): gen32k_9b -tokens
// 16384 -ctx 20480 -progress 2048 on Qwen3.5-9B, same binary, DLGO_ATTN_DECODE_GQA
// 1 vs 0 -> decode 76.03 vs 70.95 tok/s overall and 74.0 vs 64.4 tok/s at 16k
// context. The context-linear decode term drops ~45x: T/token 13.45 + 3.5e-6*N ms
// folded vs 12.95 + 1.57e-4*N ms per-q-head. The fold costs 1-2% below ~3k
// positions, where it halves the workgroup count without enough context to pay
// for it.
func TestAttentionDecodeGQAParity(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)
	t.Cleanup(func() {
		SetAttentionDecode(true)
		SetAttentionDecodeGQA(true)
	})

	const heads = 16

	shapes := []struct {
		name    string
		hd      int
		kvheads int
	}{
		{"hd256_kv4_qwen9b", 256, 4}, // kv_mul 4 — Qwen3.5-9B
		{"hd128_kv4", 128, 4},        // kv_mul 4, smaller Qwen3.5 variants
		{"hd256_kv8", 256, 8},        // kv_mul 2
		{"hd256_kv2", 256, 2},        // kv_mul 8 — PV covers all 256 threads
	}
	ctxs := []struct {
		name string
		ctx  int
	}{
		{"below_threshold_nosplit", 1000},
		{"at_threshold", 1024},
		{"split_tail_short", 4096},
		{"split_uneven", 5000},
		{"split_mid", 8192},
		{"split_tail", 16385},
	}

	for _, sh := range shapes {
		for _, tc := range ctxs {
			t.Run(sh.name+"_"+tc.name, func(t *testing.T) {
				hd, kvheads := sh.hd, sh.kvheads
				kvd := kvheads * hd
				scale := float32(1.0 / math.Sqrt(float64(hd)))
				rng := rand.New(rand.NewSource(int64(tc.ctx) + int64(hd)*31 + int64(kvheads)))
				alloc := func(bytes int) Buf {
					b := Alloc(uint64(bytes))
					if b == 0 {
						t.Fatal("alloc failed")
					}
					t.Cleanup(func() { Free(b) })
					return b
				}
				packed := func(count int) []byte {
					b := make([]byte, count*2)
					for i := 0; i < count; i++ {
						binary.LittleEndian.PutUint16(b[2*i:], packF32ToF16((rng.Float32()*2-1)*0.7))
					}
					return b
				}

				kb := alloc(tc.ctx * kvd * 2)
				vb := alloc(tc.ctx * kvd * 2)
				if err := Upload(kb, packed(tc.ctx*kvd)); err != nil {
					t.Fatal(err)
				}
				if err := Upload(vb, packed(tc.ctx*kvd)); err != nil {
					t.Fatal(err)
				}

				q := make([]float32, heads*hd)
				for i := range q {
					q[i] = (rng.Float32()*2 - 1) * 0.8
				}
				qb := alloc(len(q) * 4)
				if err := UploadF32(qb, q); err != nil {
					t.Fatal(err)
				}
				ob := alloc(len(q) * 4)

				const (
					modeTiled = iota
					modeSplit
					modeGQA
				)
				run := func(mode int) ([]float32, int) {
					SetAttentionDecode(mode != modeTiled)
					SetAttentionDecodeGQA(mode == modeGQA)
					BeginBatch()
					if err := AttentionF16(ob, qb, kb, vb, heads, kvheads, hd, kvd,
						tc.ctx, 0, scale, 0); err != nil {
						t.Fatal(err)
					}
					EndBatch()
					Sync()
					out := make([]float32, len(q))
					if err := DownloadF32(ob, out); err != nil {
						t.Fatal(err)
					}
					return out, AttentionDecodeUsed()
				}

				ref, refUsed := run(modeTiled)
				gotSplit, splitUsed := run(modeSplit)
				gotGQA, gqaUsed := run(modeGQA)
				SetAttentionDecode(true)
				SetAttentionDecodeGQA(true)

				// Below the split threshold both split kernels defer to tiled.
				wantSplit, wantGQA := 1, 2
				if tc.ctx < 2048 {
					wantSplit, wantGQA = 0, 0
				}
				if refUsed != 0 || splitUsed != wantSplit || gqaUsed != wantGQA {
					t.Fatalf("kernel selection: tiled=%d split=%d gqa=%d, want 0/%d/%d",
						refUsed, splitUsed, gqaUsed, wantSplit, wantGQA)
				}

				check := func(label string, got []float32) {
					maxAbs, maxRef := 0.0, 0.0
					nonfinite := 0
					for i := range ref {
						if a := math.Abs(float64(ref[i])); a > maxRef {
							maxRef = a
						}
						if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
							nonfinite++
						}
						if d := math.Abs(float64(ref[i] - got[i])); d > maxAbs {
							maxAbs = d
						}
					}
					rel := 0.0
					if maxRef > 0 {
						rel = maxAbs / maxRef
					}
					t.Logf("%s: hd=%d kv_mul=%d ctx=%d maxAbs=%.3e relAbs=%.3e nonfinite=%d/%d",
						label, hd, heads/kvheads, tc.ctx, maxAbs, rel, nonfinite, len(ref))
					if nonfinite > 0 {
						t.Fatalf("%s produced %d non-finite values", label, nonfinite)
					}
					if rel > 1e-4 {
						t.Fatalf("%s diverges from tiled kernel: relAbs=%.3e", label, rel)
					}
				}
				check("split", gotSplit)
				check("gqa", gotGQA)
			})
		}
	}
}

// TestAttentionDecodeSplitParity checks the split-KV decode attention kernel
// (attention_decode_split.comp + attention_decode_merge.comp) against the tiled
// kernel it replaces (attention_tiled.comp) at Qwen3.5 shapes and a spread of
// context lengths, exercising ns=1 (below the split threshold), the threshold
// itself, multi-split exact/uneven tails.
//
// Both kernels read the identical FP16-packed KV cache and use the same
// pre-scaled query, so the only difference is the order in which the softmax and
// the weighted-V sum are accumulated. Output must therefore agree to fp32
// rounding, not bit-exactly.
//
// Measured performance (do not trust a per-dispatch microbenchmark here — timing
// isolated sub-millisecond kernels on this machine is dominated by noise and GPU
// clock ramp). The authoritative numbers come from the end-to-end generation
// bench (bench/scripts/gen32k_9b_main.go, -tokens 16384 -ctx 20480 -progress
// 2048), where decode throughput at 16k context went from 35.6 to 64.0 tok/s and
// the whole run from 437 s to 232 s.
func TestAttentionDecodeSplitParity(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)
	t.Cleanup(func() { SetAttentionDecode(true) })

	const heads = 16

	shapes := []struct {
		hd      int
		kvheads int
	}{
		{256, 4}, // Qwen3.5-9B
		{128, 4}, // smaller Qwen3.5 variants
	}
	ctxs := []struct {
		name string
		ctx  int
	}{
		{"below_threshold_nosplit", 1000},
		{"at_threshold", 1024},
		{"split4_tail_short", 4096},
		{"split4_uneven", 5000},
		{"split4_mid", 8192},
		{"split4_tail", 16385},
	}

	for _, sh := range shapes {
		for _, tc := range ctxs {
			name := fmt.Sprintf("hd%d_%s", sh.hd, tc.name)
			t.Run(name, func(t *testing.T) {
				hd, kvheads := sh.hd, sh.kvheads
				kvd := kvheads * hd
				scale := float32(1.0 / math.Sqrt(float64(hd)))
				rng := rand.New(rand.NewSource(int64(tc.ctx) + int64(hd) + 7))
				alloc := func(bytes int) Buf {
					b := Alloc(uint64(bytes))
					if b == 0 {
						t.Fatal("alloc failed")
					}
					t.Cleanup(func() { Free(b) })
					return b
				}
				packed := func(count int) []byte {
					b := make([]byte, count*2)
					for i := 0; i < count; i++ {
						binary.LittleEndian.PutUint16(b[2*i:], packF32ToF16((rng.Float32()*2-1)*0.7))
					}
					return b
				}

				kb := alloc(tc.ctx * kvd * 2)
				vb := alloc(tc.ctx * kvd * 2)
				if err := Upload(kb, packed(tc.ctx*kvd)); err != nil {
					t.Fatal(err)
				}
				if err := Upload(vb, packed(tc.ctx*kvd)); err != nil {
					t.Fatal(err)
				}

				q := make([]float32, heads*hd)
				for i := range q {
					q[i] = (rng.Float32()*2 - 1) * 0.8
				}
				qb := alloc(len(q) * 4)
				if err := UploadF32(qb, q); err != nil {
					t.Fatal(err)
				}
				ob := alloc(len(q) * 4)

				run := func(split bool) []float32 {
					SetAttentionDecode(split)
					BeginBatch()
					if err := AttentionF16(ob, qb, kb, vb, heads, kvheads, hd, kvd,
						tc.ctx, 0, scale, 0); err != nil {
						t.Fatal(err)
					}
					EndBatch()
					Sync()
					out := make([]float32, len(q))
					if err := DownloadF32(ob, out); err != nil {
						t.Fatal(err)
					}
					return out
				}

				ref := run(false)
				got := run(true)
				SetAttentionDecode(true)

				maxAbs, maxRef := 0.0, 0.0
				nonfinite := 0
				for i := range ref {
					if a := math.Abs(float64(ref[i])); a > maxRef {
						maxRef = a
					}
					if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
						nonfinite++
					}
					if d := math.Abs(float64(ref[i] - got[i])); d > maxAbs {
						maxAbs = d
					}
				}
				rel := 0.0
				if maxRef > 0 {
					rel = maxAbs / maxRef
				}
				t.Logf("hd=%d ctx=%d: maxAbs=%.3e maxRef=%.3e relAbs=%.3e nonfinite=%d/%d",
					hd, tc.ctx, maxAbs, maxRef, rel, nonfinite, len(ref))
				if nonfinite > 0 {
					t.Fatalf("split decode produced %d non-finite values", nonfinite)
				}
				if rel > 1e-4 {
					t.Fatalf("split decode diverges from tiled kernel: relAbs=%.3e", rel)
				}
			})
		}
	}
}
