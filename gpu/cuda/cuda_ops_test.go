//go:build cgo && windows

package cuda

import (
	"math"
	"testing"

	"github.com/computerex/dlgo/quant"
)

func ensureInit(t *testing.T) {
	t.Helper()
	if err := Init(); err != nil {
		t.Skipf("cuda init failed: %v", err)
	}
}

func upBytes(t *testing.T, b []byte) Buffer {
	t.Helper()
	d, err := Alloc(uint64(len(b)))
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	if err := Upload(d, b); err != nil {
		t.Fatalf("upload: %v", err)
	}
	return d
}

func f32ToBytes(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		bits := math.Float32bits(x)
		b[i*4] = byte(bits)
		b[i*4+1] = byte(bits >> 8)
		b[i*4+2] = byte(bits >> 16)
		b[i*4+3] = byte(bits >> 24)
	}
	return b
}

func bytesToF32(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		bits := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		v[i] = math.Float32frombits(bits)
	}
	return v
}

// buildQ4KBytes makes rows blocks of Q4_K with d=1, dmin=0, all scales=63,
// and a deterministic nibble pattern. Reference comes from quant.DequantizeQ4_K
// so the test pins the CUDA layout to the canonical GGML format.
func buildQ4KBytes(rows int) []byte {
	const bb = 144
	b := make([]byte, rows*bb)
	for r := 0; r < rows; r++ {
		off := r * bb
		b[off+0] = 0x00 // d = 1.0h -> 0x3C00 little endian
		b[off+1] = 0x3C
		b[off+2] = 0x00 // dmin = 0
		b[off+3] = 0x00
		for i := 0; i < 12; i++ {
			b[off+4+i] = 0xFF
		}
		for i := 0; i < 128; i++ {
			// low nibble = 1, high nibble = 2
			b[off+16+i] = byte((2 << 4) | 1)
		}
	}
	return b
}

func buildQ6KBytes(rows int) []byte {
	const bb = 210
	b := make([]byte, rows*bb)
	for r := 0; r < rows; r++ {
		off := r * bb
		for i := 0; i < 192; i++ {
			b[off+i] = byte(0x21 + i*7)
		}
		for i := 0; i < 16; i++ {
			b[off+192+i] = byte(1 + (i % 3)) // positive scales 1..3
		}
		b[off+208] = 0x00 // d = 1.0h
		b[off+209] = 0x3C
	}
	return b
}

func dotRef(t *testing.T, wbytes []byte, ref []float32, rows, cols int, x []float32) []float32 {
	out := make([]float32, rows)
	for r := 0; r < rows; r++ {
		var s float64
		for c := 0; c < cols; c++ {
			s += float64(ref[r*cols+c]) * float64(x[c])
		}
		out[r] = float32(s)
	}
	return out
}

func relClose(got, want []float32, tol float64) bool {
	for i := range want {
		den := math.Abs(float64(want[i]))
		if den < 1e-3 {
			den = 1e-3
		}
		if math.Abs(float64(got[i]-want[i]))/den > tol {
			return false
		}
	}
	return true
}

func TestGEMVQ4KParity(t *testing.T) {
	ensureInit(t)
	rows, cols := 4, 256
	wb := buildQ4KBytes(rows)
	ref := quant.DequantizeQ4_K(wb, rows*cols)
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.1)) + 0.5
	}
	want := dotRef(t, wb, ref, rows, cols, x)

	wdev := upBytes(t, wb)
	defer Free(wdev)
	xdev := upBytes(t, f32ToBytes(x))
	defer Free(xdev)
	odev, _ := Alloc(uint64(rows * 4))
	defer Free(odev)

	if err := GEMVQ4K(wdev, xdev, odev, rows, cols); err != nil {
		t.Fatalf("gemv q4k: %v", err)
	}
	got := make([]float32, rows)
	ob := make([]byte, rows*4)
	Download(ob, odev)
	got = bytesToF32(ob)
	if !relClose(got, want, 3e-2) {
		t.Fatalf("Q4_K mismatch (dp4a)\n got=%v\nwant=%v", got, want)
	}
}

func TestGEMVQ6KParity(t *testing.T) {
	ensureInit(t)
	rows, cols := 4, 256
	wb := buildQ6KBytes(rows)
	ref := quant.DequantizeQ6_K(wb, rows*cols)
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(math.Cos(float64(i)*0.07)) - 0.3
	}
	want := dotRef(t, wb, ref, rows, cols, x)

	wdev := upBytes(t, wb)
	defer Free(wdev)
	xdev := upBytes(t, f32ToBytes(x))
	defer Free(xdev)
	odev, _ := Alloc(uint64(rows * 4))
	defer Free(odev)

	if err := GEMVQ6K(wdev, xdev, odev, rows, cols); err != nil {
		t.Fatalf("gemv q6k: %v", err)
	}
	ob := make([]byte, rows*4)
	Download(ob, odev)
	got := bytesToF32(ob)
	if !relClose(got, want, 1e-3) {
		t.Fatalf("Q6_K mismatch\n got=%v\nwant=%v", got, want)
	}
}

func TestGEMVF32Parity(t *testing.T) {
	ensureInit(t)
	rows, cols := 3, 300
	W := make([]float32, rows*cols)
	for i := range W {
		W[i] = float32(math.Sin(float64(i) * 0.03))
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(math.Cos(float64(i) * 0.11))
	}
	want := make([]float32, rows)
	for r := 0; r < rows; r++ {
		var s float64
		for c := 0; c < cols; c++ {
			s += float64(W[r*cols+c]) * float64(x[c])
		}
		want[r] = float32(s)
	}
	wdev := upBytes(t, f32ToBytes(W))
	defer Free(wdev)
	xdev := upBytes(t, f32ToBytes(x))
	defer Free(xdev)
	odev, _ := Alloc(uint64(rows * 4))
	defer Free(odev)
	if err := GEMVF32(wdev, xdev, odev, rows, cols); err != nil {
		t.Fatalf("gemv f32: %v", err)
	}
	ob := make([]byte, rows*4)
	Download(ob, odev)
	got := bytesToF32(ob)
	if !relClose(got, want, 1e-4) {
		t.Fatalf("F32 mismatch got=%v want=%v", got, want)
	}
}

// float32 -> IEEE half bits (round to nearest even is not required for the test;
// values are chosen small and exactly representable).
func f32ToHalf(x float32) uint16 {
	bits := math.Float32bits(x)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xff) - 127
	mant := bits & 0x7fffff
	if exp == -127 && mant == 0 {
		return sign
	}
	if exp < -14 {
		return sign // underflow to zero (values are ~O(1), not hit)
	}
	if exp > 15 {
		return sign | 0x7c00
	}
	e := uint16(exp + 15)
	m := uint16(mant >> 13)
	return sign | (e << 10) | m
}

func f16Bytes(v []float32) []byte {
	b := make([]byte, len(v)*2)
	for i, x := range v {
		h := f32ToHalf(x)
		b[i*2] = byte(h)
		b[i*2+1] = byte(h >> 8)
	}
	return b
}

func halfToF32(b []byte, i int) float32 {
	h := uint16(b[i*2]) | uint16(b[i*2+1])<<8
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	var bits uint32
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for (mant & 0x400) == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
		bits = sign | ((exp + 127 - 15) << 23) | (mant << 13)
	} else if exp == 31 {
		bits = sign | 0x7f800000 | (mant << 13)
	} else {
		bits = sign | ((exp + 127 - 15) << 23) | (mant << 13)
	}
	return math.Float32frombits(bits)
}

func TestAttentionF16Parity(t *testing.T) {
	ensureInit(t)
	cases := []struct {
		name                        string
		numHeads, numKVHeads, hd, n int
	}{
		{"gqa2-short", 2, 1, 64, 5},      // block-per-head path
		{"gqa4-1split", 4, 1, 64, 300},   // split path, single split
		{"gqa8x2-3split", 8, 2, 64, 700}, // split path, GQA fold + 3 splits
		{"qwen3ish-3split", 32, 8, 128, 600},
		{"qwen3ish-5split", 32, 8, 128, 1200},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			checkAttentionParity(t, c.numHeads, c.numKVHeads, c.hd, c.n)
		})
	}
}

func checkAttentionParity(t *testing.T, numHeads, numKVHeads, headDim, seqLen int) {
	kvDim := numKVHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	q := make([]float32, numHeads*headDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.13)) * 0.8
	}
	K := make([]float32, seqLen*kvDim)
	V := make([]float32, seqLen*kvDim)
	for i := range K {
		K[i] = float32(math.Cos(float64(i)*0.05)) * 1.5
		V[i] = float32(math.Sin(float64(i)*0.09)) * 1.2
	}
	kb := f16Bytes(K)
	vb := f16Bytes(V)

	// CPU reference (KV read back as the f16-rounded values).
	want := make([]float32, numHeads*headDim)
	kvMul := numHeads / numKVHeads
	for h := 0; h < numHeads; h++ {
		kvHead := h / kvMul
		scores := make([]float64, seqLen)
		maxs := -1e30
		for tt := 0; tt < seqLen; tt++ {
			var s float64
			for d := 0; d < headDim; d++ {
				k := float64(halfToF32(kb, tt*kvDim+kvHead*headDim+d))
				s += float64(q[h*headDim+d]) * k
			}
			s *= float64(scale)
			scores[tt] = s
			if s > maxs {
				maxs = s
			}
		}
		var sum float64
		for tt := 0; tt < seqLen; tt++ {
			scores[tt] = math.Exp(scores[tt] - maxs)
			sum += scores[tt]
		}
		for d := 0; d < headDim; d++ {
			var acc float64
			for tt := 0; tt < seqLen; tt++ {
				acc += scores[tt] / sum * float64(halfToF32(vb, tt*kvDim+kvHead*headDim+d))
			}
			want[h*headDim+d] = float32(acc)
		}
	}

	qdev := upBytes(t, f32ToBytes(q))
	defer Free(qdev)
	kdev := upBytes(t, kb)
	defer Free(kdev)
	vdev := upBytes(t, vb)
	defer Free(vdev)
	odev, _ := Alloc(uint64(numHeads * headDim * 4))
	defer Free(odev)

	if err := AttentionF16(qdev, kdev, vdev, odev, numHeads, numKVHeads, headDim, kvDim, seqLen, 0, scale, 0, 0); err != nil {
		t.Fatalf("attention: %v", err)
	}
	ob := make([]byte, numHeads*headDim*4)
	Download(ob, odev)
	got := bytesToF32(ob)
	if !relClose(got, want, 5e-3) {
		t.Fatalf("attention mismatch\n got=%v\nwant=%v", got, want)
	}
}

// TestAttentionSplitPartials inspects the split-kernel partials directly. With
// an all-zero K every score is equal, so the probabilities are all 1 and the
// per-split softmax denominator must be exactly the number of keys the block
// owns. That isolates the block-level reductions from the score math.
func TestAttentionSplitPartials(t *testing.T) {
	ensureInit(t)
	numHeads, numKVHeads, headDim, seqLen := 4, 2, 8, 300
	kvDim := numKVHeads * headDim
	kvMul := numHeads / numKVHeads
	splits := (seqLen + attnTile - 1) / attnTile
	per := (seqLen + splits - 1) / splits
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	q := make([]float32, numHeads*headDim)
	for i := range q {
		q[i] = 0.25
	}
	K := make([]float32, seqLen*kvDim) // zeros -> all scores 0
	V := make([]float32, seqLen*kvDim)
	for j := 0; j < seqLen; j++ {
		for d := 0; d < kvDim; d++ {
			V[j*kvDim+d] = float32(j) * 0.01
		}
	}
	qdev := upBytes(t, f32ToBytes(q))
	defer Free(qdev)
	kdev := upBytes(t, f16Bytes(K))
	defer Free(kdev)
	vdev := upBytes(t, f16Bytes(V))
	defer Free(vdev)
	odev, _ := Alloc(uint64(numHeads * headDim * 4))
	defer Free(odev)

	if err := AttentionF16(qdev, kdev, vdev, odev, numHeads, numKVHeads, headDim, kvDim, seqLen, 0, scale, 0, 0); err != nil {
		t.Fatalf("attention: %v", err)
	}
	ob := make([]byte, numHeads*headDim*4)
	Download(ob, odev)
	got := bytesToF32(ob)

	stride := kvMul * (headDim + 2)
	n := numKVHeads * splits * stride
	pb := make([]byte, n*4)
	if attnPartBuf == 0 {
		t.Fatalf("no partial buffer allocated")
	}
	Download(pb, attnPartBuf)
	part := bytesToF32(pb)
	t.Logf("splits=%d per=%d stride=%d partbytes=%d", splits, per, stride, len(part))
	for kh := 0; kh < numKVHeads; kh++ {
		for s := 0; s < splits; s++ {
			base := (kh*splits + s) * stride
			ms := part[base : base+kvMul]
			ls := part[base+kvMul : base+2*kvMul]
			t.Logf("kh=%d split=%d m=%v l=%v acc0..=%v", kh, s, ms, ls,
				part[base+2*kvMul:base+2*kvMul+4])
		}
	}
	// Expected: l == keys in split, out[d] == mean(V[j][kv*headDim+d] over split).
	for h := 0; h < numHeads; h++ {
		kvh := h / kvMul
		var acc float64
		for j := 0; j < seqLen; j++ {
			acc += float64(halfToF32(f16Bytes(V), j*kvDim+kvh*headDim+0))
		}
		t.Logf("head=%d out[0]=%.5f want%.5f", h, got[h*headDim], acc/float64(seqLen))
	}
}

// TestAttentionF16WindowParity covers the sliding-window (win_start) and ring
// (kv_cap) paths of the split kernel.
func TestAttentionF16WindowParity(t *testing.T) {
	ensureInit(t)
	numHeads, numKVHeads, headDim, seqLen := 8, 2, 64, 900
	winStart := 400
	kvCap := 1024
	kvDim := numKVHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	n := seqLen - winStart
	q := make([]float32, numHeads*headDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.17)) * 0.7
	}
	K := make([]float32, kvCap*kvDim)
	V := make([]float32, kvCap*kvDim)
	for i := range K {
		K[i] = float32(math.Cos(float64(i)*0.031)) * 1.3
		V[i] = float32(math.Sin(float64(i)*0.043)) * 1.1
	}
	kb := f16Bytes(K)
	vb := f16Bytes(V)

	want := make([]float32, numHeads*headDim)
	kvMul := numHeads / numKVHeads
	for h := 0; h < numHeads; h++ {
		kvHead := h / kvMul
		scores := make([]float64, n)
		maxs := -1e30
		for i := 0; i < n; i++ {
			row := (winStart + i) % kvCap
			var s float64
			for d := 0; d < headDim; d++ {
				s += float64(q[h*headDim+d]) *
					float64(halfToF32(kb, row*kvDim+kvHead*headDim+d))
			}
			s *= float64(scale)
			scores[i] = s
			if s > maxs {
				maxs = s
			}
		}
		var sum float64
		for i := 0; i < n; i++ {
			scores[i] = math.Exp(scores[i] - maxs)
			sum += scores[i]
		}
		for d := 0; d < headDim; d++ {
			var acc float64
			for i := 0; i < n; i++ {
				row := (winStart + i) % kvCap
				acc += scores[i] / sum *
					float64(halfToF32(vb, row*kvDim+kvHead*headDim+d))
			}
			want[h*headDim+d] = float32(acc)
		}
	}

	qdev := upBytes(t, f32ToBytes(q))
	defer Free(qdev)
	kdev := upBytes(t, kb)
	defer Free(kdev)
	vdev := upBytes(t, vb)
	defer Free(vdev)
	odev, _ := Alloc(uint64(numHeads * headDim * 4))
	defer Free(odev)
	if err := AttentionF16(qdev, kdev, vdev, odev, numHeads, numKVHeads, headDim, kvDim, seqLen, winStart, scale, 0, kvCap); err != nil {
		t.Fatalf("attention: %v", err)
	}
	ob := make([]byte, numHeads*headDim*4)
	Download(ob, odev)
	got := bytesToF32(ob)
	if !relClose(got, want, 5e-3) {
		t.Fatalf("attention(window) mismatch\n got=%v\nwant=%v", got, want)
	}
}
