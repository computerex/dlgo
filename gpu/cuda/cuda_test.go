//go:build cgo && windows

package cuda

import (
	"math"
	"math/rand"
	"testing"
)

func mustInit(t *testing.T) {
	t.Helper()
	if err := Init(); err != nil {
		t.Skipf("CUDA backend unavailable: %v", err)
	}
}

// requireKernels is a hook kept for callers; launches are stable now that Sync
// uses cuStreamSynchronize instead of the crash-prone cuCtxSynchronize.
func requireKernels(t *testing.T) { t.Helper() }

func upF32(t *testing.T, xs []float32) Buffer {
	t.Helper()
	b, err := Alloc(uint64(len(xs) * 4))
	if err != nil {
		t.Fatal(err)
	}
	if err := Upload(b, f32bytes(xs)); err != nil {
		t.Fatal(err)
	}
	return b
}

func newF32(t *testing.T, n int) Buffer {
	t.Helper()
	b, err := Alloc(uint64(n * 4))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func downF32(t *testing.T, b Buffer, n int) []float32 {
	t.Helper()
	raw := make([]byte, n*4)
	if err := Download(raw, b); err != nil {
		t.Fatal(err)
	}
	return bytesF32(raw)
}

func f32bytes(xs []float32) []byte {
	out := make([]byte, len(xs)*4)
	for i, v := range xs {
		u := math.Float32bits(v)
		out[i*4+0] = byte(u)
		out[i*4+1] = byte(u >> 8)
		out[i*4+2] = byte(u >> 16)
		out[i*4+3] = byte(u >> 24)
	}
	return out
}

func bytesF32(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		u := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		out[i] = math.Float32frombits(u)
	}
	return out
}

// h2f decodes an IEEE-754 binary16 to float32 (exact).
func h2f(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	var bits uint32
	switch {
	case exp == 0:
		if mant == 0 {
			bits = sign
		} else {
			e := int32(-1)
			m := mant
			for m&0x400 == 0 {
				m <<= 1
				e--
			}
			m &= 0x3ff
			exp32 := uint32(int32(127-15) + e + 1)
			bits = sign | (exp32 << 23) | (m << 13)
		}
	case exp == 0x1f:
		bits = sign | 0x7f800000 | (mant << 13)
	default:
		bits = sign | ((exp - 15 + 127) << 23) | (mant << 13)
	}
	return math.Float32frombits(bits)
}

func TestDeviceInfo(t *testing.T) {
	mustInit(t)
	t.Logf("device=%q SMs=%d totalVRAM=%d MiB free=%d MiB",
		DeviceName(), SMs(), TotalMem()/(1<<20), FreeMem()/(1<<20))
	if TotalMem() == 0 {
		t.Fatal("TotalMem()==0")
	}
}

// TestMemRoundTrip proves the device-memory subsystem end to end without any
// kernel launch: alloc, upload, download, verify, free.
func TestMemRoundTrip(t *testing.T) {
	mustInit(t)
	const n = 4096
	rng := rand.New(rand.NewSource(7))
	want := make([]float32, n)
	for i := range want {
		want[i] = float32(rng.Intn(20000)-10000) / 16
	}
	b, err := Alloc(uint64(n * 4))
	if err != nil {
		t.Fatal(err)
	}
	defer Free(b)
	if err := Upload(b, f32bytes(want)); err != nil {
		t.Fatal(err)
	}
	got := downF32(t, b, n)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("i=%d got %v want %v", i, got[i], want[i])
		}
	}

	// memset then verify zeros
	if err := Memset8(b, 0, uint64(n*4)); err != nil {
		t.Fatal(err)
	}
	zero := downF32(t, b, n)
	for i := range zero {
		if zero[i] != 0 {
			t.Fatalf("i=%d memset got %v want 0", i, zero[i])
		}
	}
}

func TestVectorAdd(t *testing.T) {
	mustInit(t)
	requireKernels(t)
	const n = 1000
	rng := rand.New(rand.NewSource(1))
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(rng.Intn(2000)-1000) / 8
		b[i] = float32(rng.Intn(2000)-1000) / 8
	}
	da, db, dout := upF32(t, a), upF32(t, b), newF32(t, n)
	defer Free(da)
	defer Free(db)
	defer Free(dout)

	if err := VectorAdd(da, db, dout, n); err != nil {
		t.Fatal(err)
	}
	got := downF32(t, dout, n)
	for i := range got {
		if got[i] != a[i]+b[i] {
			t.Fatalf("i=%d got %v want %v", i, got[i], a[i]+b[i])
		}
	}
}

func TestKVStoreF16(t *testing.T) {
	mustInit(t)
	requireKernels(t)
	const kvDim = 512
	rng := rand.New(rand.NewSource(2))
	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	for i := 0; i < kvDim; i++ {
		k[i] = float32(rng.Intn(2000)-1000) / 8
		v[i] = float32(rng.Intn(2000)-1000) / 8
	}
	halfDim := kvDim / 2
	pairsPerRow := halfDim

	for _, tc := range []struct{ pos, ringCap int }{{5, 0}, {70, 64}} {
		rows := tc.pos + 1
		if tc.ringCap > 0 {
			rows = tc.ringCap
		}
		dk, dv := upF32(t, k), upF32(t, v)
		kOut, _ := Alloc(uint64(rows * pairsPerRow * 4))
		vOut, _ := Alloc(uint64(rows * pairsPerRow * 4))

		if err := KVStoreF16(dk, dv, kOut, vOut, tc.pos, kvDim, tc.ringCap); err != nil {
			t.Fatal(err)
		}

		raw := make([]byte, rows*pairsPerRow*4)
		if err := Download(raw, kOut); err != nil {
			t.Fatal(err)
		}
		row := tc.pos
		if tc.ringCap > 0 {
			row = tc.pos % tc.ringCap
		}
		base := row * pairsPerRow * 4
		for j := 0; j < pairsPerRow; j++ {
			u := uint32(raw[base+j*4]) | uint32(raw[base+j*4+1])<<8 |
				uint32(raw[base+j*4+2])<<16 | uint32(raw[base+j*4+3])<<24
			lo := h2f(uint16(u & 0xffff))
			hi := h2f(uint16(u >> 16))
			if lo != k[j*2] || hi != k[j*2+1] {
				t.Fatalf("ring=%d slot %d: got (%v,%v) want (%v,%v)",
					tc.ringCap, j, lo, hi, k[j*2], k[j*2+1])
			}
		}
		Free(dk)
		Free(dv)
		Free(kOut)
		Free(vOut)
	}
}

func TestRMSNorm(t *testing.T) {
	mustInit(t)
	requireKernels(t)
	const n = 768
	rng := rand.New(rand.NewSource(3))
	x := make([]float32, n)
	w := make([]float32, n)
	for i := 0; i < n; i++ {
		x[i] = float32(rng.Intn(200)-100) / 16
		w[i] = 0.5 + float32(rng.Intn(100))/100
	}
	eps := float32(1e-5)
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	scale := float32(1 / math.Sqrt(ss/float64(n)+float64(eps)))

	dx, dw, dout := upF32(t, x), upF32(t, w), newF32(t, n)
	defer Free(dx)
	defer Free(dw)
	defer Free(dout)
	if err := RMSNorm(dx, dw, dout, n, eps); err != nil {
		t.Fatal(err)
	}
	got := downF32(t, dout, n)
	for i := range got {
		want := x[i] * scale * w[i]
		if math.Abs(float64(got[i]-want)) > 1e-4 {
			t.Fatalf("i=%d got %v want %v", i, got[i], want)
		}
	}
}

func TestSiluMul(t *testing.T) {
	mustInit(t)
	requireKernels(t)
	const n = 500
	rng := rand.New(rand.NewSource(4))
	g := make([]float32, n)
	u := make([]float32, n)
	for i := 0; i < n; i++ {
		g[i] = float32(rng.Intn(1600)-800) / 100
		u[i] = float32(rng.Intn(1600)-800) / 100
	}
	dg, du, dout := upF32(t, g), upF32(t, u), newF32(t, n)
	defer Free(dg)
	defer Free(du)
	defer Free(dout)
	if err := SiluMul(dg, du, dout, n); err != nil {
		t.Fatal(err)
	}
	got := downF32(t, dout, n)
	for i := range got {
		s := g[i] / (1 + float32(math.Exp(float64(-g[i]))))
		want := s * u[i]
		if math.Abs(float64(got[i]-want)) > 1e-4 {
			t.Fatalf("i=%d got %v want %v", i, got[i], want)
		}
	}
}

func TestSoftmaxRow(t *testing.T) {
	mustInit(t)
	requireKernels(t)
	const n = 300
	rng := rand.New(rand.NewSource(5))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.Intn(1000)) / 100
	}
	mx := float32(-1e30)
	for _, v := range x {
		if v > mx {
			mx = v
		}
	}
	var denom float64
	for _, v := range x {
		denom += math.Exp(float64(v - mx))
	}
	dx := upF32(t, x)
	defer Free(dx)
	if err := SoftmaxRow(dx, n); err != nil {
		t.Fatal(err)
	}
	got := downF32(t, dx, n)
	var sum float64
	for i := range got {
		want := float32(math.Exp(float64(x[i]-mx)) / denom)
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Fatalf("i=%d got %v want %v", i, got[i], want)
		}
		sum += float64(got[i])
	}
	if math.Abs(sum-1) > 1e-4 {
		t.Fatalf("sum=%v want 1", sum)
	}
}
