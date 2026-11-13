package cuda

import (
	"math"
	"testing"

	"github.com/computerex/dlgo/quant"
)

// TestBatchAttnProductionShape covers the Qwen3-4B attention shape exactly
// (32 query heads, 8 KV heads, head_dim 128), which is the maximum head_dim the
// prefill kernel supports and therefore the tightest shared-memory case. The
// smaller shapes in TestBatchAttnParity do not exercise it.
func TestBatchAttnProductionShape(t *testing.T) {
	affineInit(t)
	cases := []struct {
		name           string
		npos, startPos int
	}{
		{"tiny-10tok", 10, 0},
		{"one-full-tile", 64, 0},
		{"three-tiles", 192, 0},
		{"midchunk", 96, 1024},
	}
	const numHeads, numKVHeads, headDim = 32, 8, 128
	kvDim := numKVHeads * headDim
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			total := tc.startPos + tc.npos
			kf := affRandF32(total*kvDim, 1.0, 31)
			vf := affRandF32(total*kvDim, 1.0, 32)
			q := affRandF32(tc.npos*numHeads*headDim, 1.0, 33)
			scale := float32(1 / math.Sqrt(float64(headDim)))

			qb, _ := Alloc(uint64(len(q) * 4))
			defer Free(qb)
			kb, _ := Alloc(uint64(len(kf)+32*kvDim) * 2)
			defer Free(kb)
			vb, _ := Alloc(uint64(len(vf)+32*kvDim) * 2)
			defer Free(vb)
			ob, _ := Alloc(uint64(len(q) * 4))
			defer Free(ob)
			Upload(qb, affF32Bytes(q))
			Upload(kb, padKV(f16bytes(kf), kvDim))
			Upload(vb, padKV(f16bytes(vf), kvDim))
			Memset8(ob, 0, uint64(len(q)*4))

			if err := AttnPrefillF16(ob, qb, kb, vb, numHeads, numKVHeads, headDim,
				kvDim, tc.startPos, tc.npos, scale, 0); err != nil {
				t.Fatal(err)
			}
			Sync()
			raw := make([]byte, len(q)*4)
			Download(raw, ob)
			got := affBytesF32(raw)

			kc := make([]float32, len(kf))
			for i, v := range kf {
				kc[i] = f16f32(quant.F32ToF16(v))
			}
			vc := make([]float32, len(vf))
			for i, v := range vf {
				vc[i] = f16f32(quant.F32ToF16(v))
			}
			want := cpuAttn(q, kc, vc, numHeads, numKVHeads, headDim, kvDim,
				tc.startPos, tc.npos, float64(scale), 0)

			rel, nf := affRelL2(got, want)
			if nf > 0 || rel > 2e-3 {
				// Report the worst element to make a layout bug obvious.
				worst, wi := 0.0, 0
				for i := range want {
					if d := math.Abs(float64(got[i] - want[i])); d > worst {
						worst, wi = d, i
					}
				}
				t.Fatalf("%s relL2=%.3e nonfinite=%d worst=%.4g at %d (got %v want %v)",
					tc.name, rel, nf, worst, wi, got[wi], want[wi])
			}
			t.Logf("%s relL2=%.3e", tc.name, rel)
		})
	}
}
