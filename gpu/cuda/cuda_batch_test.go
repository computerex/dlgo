package cuda

import (
	"math"
	"testing"

	"github.com/computerex/dlgo/quant"
)

// ---------------------------------------------------------------------------
// Batched (prompt-processing) primitives: rmsnorm family, RoPE, KV store and the
// causal flash prefill attention. The attention reference is the plain quadratic
// float64 formulation, so any masking/indexing mistake shows up immediately.
// ---------------------------------------------------------------------------

func f16bytes(v []float32) []byte {
	b := make([]byte, len(v)*2)
	for i, x := range v {
		q := quant.F32ToF16(x)
		b[i*2] = byte(q)
		b[i*2+1] = byte(q >> 8)
	}
	return b
}

// f16f32 decodes an IEEE half to float32 (test-side only; the quant package
// exposes the encode direction).
func f16f32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	man := uint32(h) & 0x3ff
	switch {
	case exp == 0 && man == 0:
		return math.Float32frombits(sign << 31)
	case exp == 0:
		e := uint32(127 - 15 + 1)
		for man&0x400 == 0 {
			man <<= 1
			e--
		}
		return math.Float32frombits(sign<<31 | e<<23 | (man&0x3ff)<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign<<31 | 0xff<<23 | man<<13)
	default:
		return math.Float32frombits(sign<<31 | (exp+127-15)<<23 | man<<13)
	}
}

func TestBatchRMSNormParity(t *testing.T) {
	affineInit(t)
	const n, npos = 256, 5
	x := affRandF32(n*npos, 1.0, 21)
	w := affRandF32(n, 0.5, 22)
	const eps = 1e-6

	xb, err := Alloc(uint64(n * npos * 4))
	if err != nil {
		t.Fatal(err)
	}
	defer Free(xb)
	ob, err := Alloc(uint64(n * npos * 4))
	if err != nil {
		t.Fatal(err)
	}
	defer Free(ob)
	sum, err := Alloc(uint64(n * npos * 4))
	if err != nil {
		t.Fatal(err)
	}
	defer Free(sum)
	nb, err := Alloc(uint64(n * npos * 4))
	if err != nil {
		t.Fatal(err)
	}
	defer Free(nb)
	wb, err := Alloc(uint64(n * 4))
	if err != nil {
		t.Fatal(err)
	}
	defer Free(wb)
	for _, p := range []struct {
		buf Buffer
		v   []float32
	}{{xb, x}, {wb, w}} {
		if err := Upload(p.buf, affF32Bytes(p.v)); err != nil {
			t.Fatal(err)
		}
	}

	if err := BatchRMSNorm(ob, xb, wb, n, npos, eps); err != nil {
		t.Fatal(err)
	}
	if err := BatchAddRMSNorm(nb, sum, xb, xb, wb, n, npos, eps); err != nil {
		t.Fatal(err)
	}
	if err := Sync(); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, n*npos)
	raw := make([]byte, n*npos*4)
	if err := Download(raw, ob); err != nil {
		t.Fatal(err)
	}
	copy(got, affBytesF32(raw))
	gotN := make([]float32, n*npos)
	if err := Download(raw, nb); err != nil {
		t.Fatal(err)
	}
	copy(gotN, affBytesF32(raw))

	want := make([]float32, n*npos)
	wantAdd := make([]float32, n*npos)
	for p := 0; p < npos; p++ {
		var ss, ss2 float64
		for i := 0; i < n; i++ {
			ss += float64(x[p*n+i]) * float64(x[p*n+i])
			s := float64(x[p*n+i]) * 2
			ss2 += s * s
		}
		inv := 1 / math.Sqrt(ss/float64(n)+eps)
		inv2 := 1 / math.Sqrt(ss2/float64(n)+eps)
		for i := 0; i < n; i++ {
			want[p*n+i] = float32(float64(x[p*n+i]) * inv * float64(w[i]))
			s := float64(x[p*n+i]) * 2
			wantAdd[p*n+i] = float32(s * inv2 * float64(w[i]))
		}
	}
	if rel, nf := affRelL2(got, want); nf > 0 || rel > 1e-5 {
		t.Fatalf("batch rmsnorm relL2=%.3e nonfinite=%d", rel, nf)
	}
	if rel, nf := affRelL2(gotN, wantAdd); nf > 0 || rel > 1e-5 {
		t.Fatalf("batch add_rmsnorm relL2=%.3e nonfinite=%d", rel, nf)
	}
}

// cpuAttn is the exact causal attention reference over a linear f16 KV cache.
func cpuAttn(q, kc, vc []float32, numHeads, numKVHeads, headDim, kvDim, startPos, npos int,
	scale, softcap float64) []float32 {
	out := make([]float32, npos*numHeads*headDim)
	grp := numHeads / numKVHeads
	for tok := 0; tok < npos; tok++ {
		p := startPos + tok
		for h := 0; h < numHeads; h++ {
			kvh := h / grp
			sc := make([]float64, p+1)
			mx := -math.MaxFloat64
			for j := 0; j <= p; j++ {
				var acc float64
				for d := 0; d < headDim; d++ {
					acc += float64(q[(tok*numHeads+h)*headDim+d]) *
						float64(kc[j*kvDim+kvh*headDim+d])
				}
				acc *= scale
				if softcap > 0 {
					acc = softcap * math.Tanh(acc/softcap)
				}
				sc[j] = acc
				if acc > mx {
					mx = acc
				}
			}
			var den float64
			for j := range sc {
				sc[j] = math.Exp(sc[j] - mx)
				den += sc[j]
			}
			for d := 0; d < headDim; d++ {
				var acc float64
				for j := 0; j <= p; j++ {
					acc += sc[j] * float64(vc[j*kvDim+kvh*headDim+d])
				}
				out[(tok*numHeads+h)*headDim+d] = float32(acc / den)
			}
		}
	}
	return out
}

// padKV appends 32 zeroed rows to a KV-cache image. The prefill attention
// kernel rounds its last key tile UP to 32 keys, so those rows must be readable
// and finite -- the extra keys are causally masked, so their values are
// irrelevant, but a stray NaN in V would make 0*V a NaN and poison the row. A
// real KV cache is allocated with the whole context, so it always has the room.
func padKV(data []byte, kvDim int) []byte {
	out := make([]byte, len(data)+32*kvDim*2)
	copy(out, data)
	return out
}

func TestBatchAttnParity(t *testing.T) {
	affineInit(t)
	cases := []struct {
		name                    string
		numHeads, numKVHeads    int
		headDim, startPos, npos int
	}{
		{"mha-16k-tile", 4, 4, 64, 0, 70},
		{"gqa-odd-start", 8, 2, 64, 37, 45},
		{"single-kv-head", 6, 1, 32, 0, 64},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			kvDim := tc.numKVHeads * tc.headDim
			total := tc.startPos + tc.npos
			kf := affRandF32(total*kvDim, 1.0, 31)
			vf := affRandF32(total*kvDim, 1.0, 32)
			q := affRandF32(tc.npos*tc.numHeads*tc.headDim, 1.0, 33)
			scale := float32(1 / math.Sqrt(float64(tc.headDim)))

			qb, err := Alloc(uint64(len(q) * 4))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(qb)
			if err := Upload(qb, affF32Bytes(q)); err != nil {
				t.Fatal(err)
			}
			kb, err := Alloc(uint64(len(kf)+32*kvDim) * 2)
			if err != nil {
				t.Fatal(err)
			}
			defer Free(kb)
			vb, err := Alloc(uint64(len(vf)+32*kvDim) * 2)
			if err != nil {
				t.Fatal(err)
			}
			defer Free(vb)
			// f16 rounding on the host, matching the kernel's view of the cache.
			if err := Upload(kb, padKV(f16bytes(kf), kvDim)); err != nil {
				t.Fatal(err)
			}
			if err := Upload(vb, padKV(f16bytes(vf), kvDim)); err != nil {
				t.Fatal(err)
			}
			ob, err := Alloc(uint64(len(q) * 4))
			if err != nil {
				t.Fatal(err)
			}
			defer Free(ob)
			if err := Memset8(ob, 0, uint64(len(q)*4)); err != nil {
				t.Fatal(err)
			}

			if err := AttnPrefillF16(ob, qb, kb, vb, tc.numHeads, tc.numKVHeads,
				tc.headDim, kvDim, tc.startPos, tc.npos, scale, 0); err != nil {
				t.Fatal(err)
			}
			if err := Sync(); err != nil {
				t.Fatal(err)
			}
			raw := make([]byte, len(q)*4)
			if err := Download(raw, ob); err != nil {
				t.Fatal(err)
			}
			got := affBytesF32(raw)

			// Reference sees f16-rounded cache values, like the kernel does.
			kc := make([]float32, len(kf))
			for i, v := range kf {
				kc[i] = f16f32(quant.F32ToF16(v))
			}
			vc := make([]float32, len(vf))
			for i, v := range vf {
				vc[i] = f16f32(quant.F32ToF16(v))
			}
			want := cpuAttn(q, kc, vc, tc.numHeads, tc.numKVHeads, tc.headDim, kvDim,
				tc.startPos, tc.npos, float64(scale), 0)

			rel, nf := affRelL2(got, want)
			if nf > 0 || rel > 2e-3 {
				t.Fatalf("%s relL2=%.3e nonfinite=%d (want <2e-3)", tc.name, rel, nf)
			}
			t.Logf("%s relL2=%.3e", tc.name, rel)
		})
	}
}

func TestBatchRoPEKVStoreParity(t *testing.T) {
	affineInit(t)
	const numHeads, numKVHeads, headDim = 4, 2, 64
	const startPos, npos = 5, 9
	kvDim := numKVHeads * headDim
	half := headDim / 2

	q := affRandF32(npos*numHeads*headDim, 1.0, 41)
	k := affRandF32(npos*numKVHeads*headDim, 1.0, 42)
	cos := affRandF32((startPos+npos+4)*half, 1.0, 43)
	sin := affRandF32((startPos+npos+4)*half, 1.0, 44)

	qb, _ := Alloc(uint64(len(q) * 4))
	defer Free(qb)
	kb, _ := Alloc(uint64(len(k) * 4))
	defer Free(kb)
	cb, _ := Alloc(uint64(len(cos) * 4))
	defer Free(cb)
	sb, _ := Alloc(uint64(len(sin) * 4))
	defer Free(sb)
	for _, p := range []struct {
		b Buffer
		v []float32
	}{{qb, q}, {kb, k}, {cb, cos}, {sb, sin}} {
		if err := Upload(p.b, affF32Bytes(p.v)); err != nil {
			t.Fatal(err)
		}
	}
	if err := BatchRoPE(qb, kb, cb, sb, numHeads, numKVHeads, headDim, headDim,
		startPos, false, npos); err != nil {
		t.Fatal(err)
	}
	// KV store: cache is (startPos+npos) rows of kvDim f16.
	const cacheRows = startPos + npos
	kOut, _ := Alloc(uint64(cacheRows * kvDim * 2))
	defer Free(kOut)
	vOut, _ := Alloc(uint64(cacheRows * kvDim * 2))
	defer Free(vOut)
	if err := Memset8(kOut, 0, uint64(cacheRows*kvDim*2)); err != nil {
		t.Fatal(err)
	}
	if err := Memset8(vOut, 0, uint64(cacheRows*kvDim*2)); err != nil {
		t.Fatal(err)
	}
	if err := BatchKVStoreF16(kb, kb, kOut, vOut, startPos, kvDim, npos); err != nil {
		t.Fatal(err)
	}
	if err := Sync(); err != nil {
		t.Fatal(err)
	}

	rawQ := make([]byte, len(q)*4)
	if err := Download(rawQ, qb); err != nil {
		t.Fatal(err)
	}
	gotQ := affBytesF32(rawQ)
	rawK := make([]byte, len(k)*4)
	if err := Download(rawK, kb); err != nil {
		t.Fatal(err)
	}
	gotK := affBytesF32(rawK)

	// Reference RoPE (interleaved, neox=false), same table row = absolute pos.
	wantQ := append([]float32(nil), q...)
	wantK := append([]float32(nil), k...)
	for tok := 0; tok < npos; tok++ {
		pos := startPos + tok
		for h := 0; h < numHeads; h++ {
			base := (tok*numHeads + h) * headDim
			for t := 0; t < half; t++ {
				c := float64(cos[pos*half+t])
				s := float64(sin[pos*half+t])
				a := float64(q[base+2*t])
				b := float64(q[base+2*t+1])
				wantQ[base+2*t] = float32(a*c - b*s)
				wantQ[base+2*t+1] = float32(a*s + b*c)
			}
		}
		for h := 0; h < numKVHeads; h++ {
			base := (tok*numKVHeads + h) * headDim
			for t := 0; t < half; t++ {
				c := float64(cos[pos*half+t])
				s := float64(sin[pos*half+t])
				a := float64(k[base+2*t])
				b := float64(k[base+2*t+1])
				wantK[base+2*t] = float32(a*c - b*s)
				wantK[base+2*t+1] = float32(a*s + b*c)
			}
		}
	}
	if rel, nf := affRelL2(gotQ, wantQ); nf > 0 || rel > 1e-6 {
		t.Fatalf("batch rope q relL2=%.3e nonfinite=%d", rel, nf)
	}
	if rel, nf := affRelL2(gotK, wantK); nf > 0 || rel > 1e-6 {
		t.Fatalf("batch rope k relL2=%.3e nonfinite=%d", rel, nf)
	}

	// KV cache: row startPos+i must equal f16(k[i]) and rows before startPos stay 0.
	rawCache := make([]byte, cacheRows*kvDim*2)
	if err := Download(rawCache, kOut); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < npos; i++ {
		for c := 0; c < kvDim; c++ {
			off := ((startPos+i)*kvDim + c) * 2
			got := math.Float32frombits(0) // placeholder to keep types explicit
			_ = got
			bits := uint16(rawCache[off]) | uint16(rawCache[off+1])<<8
			want := quant.F32ToF16(gotK[i*kvDim+c])
			// GPU __float2half rounds to nearest; quant.F32ToF16 may truncate, so
			// allow a 1-ULP difference. The rows themselves must line up exactly.
			d := int(bits) - int(want)
			if d < -1 || d > 1 {
				t.Fatalf("kv cache row %d col %d = %#04x want %#04x", startPos+i, c, bits, want)
			}
		}
	}
	for i := 0; i < startPos; i++ {
		for c := 0; c < kvDim; c++ {
			off := (i*kvDim + c) * 2
			if rawCache[off] != 0 || rawCache[off+1] != 0 {
				t.Fatalf("kv cache row %d col %d should be untouched", i, c)
			}
		}
	}
}
