package cuda

import (
	"math"
	"testing"
)

// TestDbgAttnOne isolates the prefill attention pipeline: the first query sees
// exactly one key (itself), so its output must be V[0] bit-for-bit.
func TestDbgAttnOne(t *testing.T) {
	affineInit(t)
	const numHeads, numKVHeads, headDim = 1, 1, 16
	const startPos, npos = 0, 1
	kvDim := numKVHeads * headDim
	q := make([]float32, npos*numHeads*headDim)
	for i := range q {
		q[i] = 0.25 * float32(i%7)
	}
	kf := make([]float32, (startPos+npos)*kvDim)
	vf := make([]float32, (startPos+npos)*kvDim)
	for i := range vf {
		vf[i] = 0.5 * float32(i%5)
	}
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

	if err := AttnPrefillF16(ob, qb, kb, vb, numHeads, numKVHeads, headDim, kvDim,
		startPos, npos, scale, 0); err != nil {
		t.Fatal(err)
	}
	Sync()
	raw := make([]byte, len(q)*4)
	Download(raw, ob)
	got := affBytesF32(raw)
	t.Logf("got = %v", got)
	t.Logf("want(v0) = %v", vf)
	for i := range got {
		if math.Abs(float64(got[i]-vf[i])) > 1e-3 {
			t.Fatalf("elem %d: got %v want %v", i, got[i], vf[i])
		}
	}
}
