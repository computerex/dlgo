//go:build cgo && vulkan

package gpu

import (
	"math"
	"testing"
)

// TestBatchRoPEParity checks the position-batched RoPE (gpu_batch_rope, one
// workgroup in y per position) against the per-token RoPE that the decode path
// uses, for every position of the chunk and both theta layouts.
//
// RoPE is the identity at position 0 when startPos == 0, so a batched RoPE that
// ignores pos_id passes any test that only looks at the first position — which
// is exactly the signature the short-conv prefill bisection showed (n=1 exact,
// n=2 wrong at the first attention layer). This test looks at every position
// separately and reports the first one that differs.
func TestBatchRoPEParity(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("no GPU: %v", err)
	}
	t.Cleanup(Shutdown)

	const (
		numHeads   = 8
		numKVHeads = 2
		headDim    = 64
		npos       = 5
		startPos   = 3 // exercise a non-zero chunk offset too
	)
	qDim := numHeads * headDim
	kvDim := numKVHeads * headDim

	alloc := func(n int) Buf {
		b, err := AllocE(uint64(n * 4))
		if err != nil {
			t.Fatalf("alloc %d floats: %v", n, err)
		}
		t.Cleanup(func() { Free(b) })
		return b
	}

	// Deterministic pseudo-random activations.
	fill := func(v []float32, seed uint32) {
		for i := range v {
			seed = seed*1664525 + 1013904223
			v[i] = (float32(seed>>16)/65536.0 - 0.5) * 4.0
		}
	}

	// Synthetic cos/sin tables. Values depend on both position and pair index so
	// any indexing mistake (wrong row, wrong pair) shows up as a mismatch.
	tableRows := startPos + npos
	cosHost := make([]float32, tableRows*(headDim/2))
	sinHost := make([]float32, tableRows*(headDim/2))
	for p := 0; p < tableRows; p++ {
		for i := 0; i < headDim/2; i++ {
			a := float64(p*7+i) * 0.11
			cosHost[p*(headDim/2)+i] = float32(math.Cos(a))
			sinHost[p*(headDim/2)+i] = float32(math.Sin(a))
		}
	}

	cosTab := alloc(len(cosHost))
	sinTab := alloc(len(sinHost))
	if err := UploadF32(cosTab, cosHost); err != nil {
		t.Fatalf("upload cos: %v", err)
	}
	if err := UploadF32(sinTab, sinHost); err != nil {
		t.Fatalf("upload sin: %v", err)
	}

	cases := []struct {
		name    string
		ropeDim int
		neox    bool
	}{
		{"neox-full", headDim, true},
		{"neox-partial", headDim / 2, true},
		{"gpt-noninterleaved", headDim, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			qHost := make([]float32, npos*qDim)
			kHost := make([]float32, npos*kvDim)
			fill(qHost, 12345)
			fill(kHost, 999)

			qb := alloc(len(qHost))
			kb := alloc(len(kHost))
			if err := UploadF32(qb, qHost); err != nil {
				t.Fatalf("upload q: %v", err)
			}
			if err := UploadF32(kb, kHost); err != nil {
				t.Fatalf("upload k: %v", err)
			}

			BeginBatch()
			Barrier()
			if err := BatchRoPE(qb, kb, cosTab, sinTab, numHeads, numKVHeads,
				headDim, tc.ropeDim, startPos, tc.neox, npos); err != nil {
				t.Fatalf("batch rope: %v", err)
			}
			Sync()
			gotQ := make([]float32, len(qHost))
			gotK := make([]float32, len(kHost))
			if err := DownloadF32(qb, gotQ); err != nil {
				t.Fatalf("download q: %v", err)
			}
			if err := DownloadF32(kb, gotK); err != nil {
				t.Fatalf("download k: %v", err)
			}
			EndBatch()

			// Per-token reference: one row at a time through the decode path.
			rsQ := alloc(qDim)
			rsK := alloc(kvDim)
			wantQ := make([]float32, len(qHost))
			wantK := make([]float32, len(kHost))
			row := make([]float32, qDim)
			krow := make([]float32, kvDim)
			for p := 0; p < npos; p++ {
				BeginBatch()
				if err := UploadF32(rsQ, qHost[p*qDim:(p+1)*qDim]); err != nil {
					t.Fatalf("upload ref q: %v", err)
				}
				if err := UploadF32(rsK, kHost[p*kvDim:(p+1)*kvDim]); err != nil {
					t.Fatalf("upload ref k: %v", err)
				}
				Barrier()
				if err := RoPE(rsQ, rsK, cosTab, sinTab, numHeads, numKVHeads,
					headDim, tc.ropeDim, startPos+p, tc.neox); err != nil {
					t.Fatalf("ref rope: %v", err)
				}
				Sync()
				if err := DownloadF32(rsQ, row); err != nil {
					t.Fatalf("download ref q: %v", err)
				}
				if err := DownloadF32(rsK, krow); err != nil {
					t.Fatalf("download ref k: %v", err)
				}
				EndBatch()
				copy(wantQ[p*qDim:(p+1)*qDim], row)
				copy(wantK[p*kvDim:(p+1)*kvDim], krow)
			}

			report := func(name string, got, want []float32, perPos int) {
				firstBad := -1
				var maxAbs float64
				for i := range got {
					d := math.Abs(float64(got[i] - want[i]))
					if d > 1e-5 && firstBad < 0 {
						firstBad = i / perPos
					}
					if d > maxAbs {
						maxAbs = d
					}
				}
				for p := 0; p < npos; p++ {
					var pa float64
					for c := 0; c < perPos; c++ {
						i := p*perPos + c
						pa = math.Max(pa, math.Abs(float64(got[i]-want[i])))
					}
					t.Logf("%s pos %d: maxAbs=%.3e", name, p, pa)
				}
				t.Logf("%s overall maxAbs=%.3e firstWrongPos=%d", name, maxAbs, firstBad)
				if maxAbs > 1e-5 {
					t.Errorf("%s: batched RoPE differs (maxAbs=%.3e, first wrong pos=%d)",
						name, maxAbs, firstBad)
				}
			}
			report("q", gotQ, wantQ, qDim)
			report("k", gotK, wantK, kvDim)
		})
	}
}
