package quant

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

func relL2(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	if den == 0 {
		return 0
	}
	return math.Sqrt(num / den)
}

func TestAffine8BlockBytes(t *testing.T) {
	if Affine8BlockBytes != 36 {
		t.Fatalf("Affine8BlockBytes = %d, want 36", Affine8BlockBytes)
	}
	if Affine8BlockBytes%4 != 0 {
		t.Fatalf("Affine8BlockBytes must be 4-aligned for vectorized loads")
	}
}

// TestAffine8RoundTrip checks the canonical 8-bit affine quantizer reconstructs
// float data well within the tolerance of a 4-6 bit source format.
func TestAffine8RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1234))
	n := 3840 * 16
	x := make([]float32, n)
	for i := range x {
		// Weight-like: small gaussian, occasionally a heavier tail.
		x[i] = float32(rng.NormFloat64()) * 0.02
	}
	enc := QuantizeAffine8(x)
	if len(enc) != (n/Affine8GroupSize)*Affine8BlockBytes {
		t.Fatalf("encoded size %d", len(enc))
	}
	dec := DequantizeAffine8(enc, n)
	rl := relL2(x, dec)
	t.Logf("Affine8 round-trip relL2 = %.3e", rl)
	if rl > 1e-2 {
		t.Fatalf("relL2 %.3e too large for 8-bit affine", rl)
	}
}

func TestAffine8ConstantGroup(t *testing.T) {
	// A constant group must not divide by zero and must reconstruct exactly.
	x := make([]float32, 64)
	for i := range x {
		x[i] = 0.5
	}
	dec := DequantizeAffine8(QuantizeAffine8(x), len(x))
	for i, v := range dec {
		if math.Abs(float64(v-0.5)) > 1e-3 {
			t.Fatalf("const group idx %d = %v", i, v)
		}
	}
}

// TestRepackAffine8F32 exercises the full RepackAffine8 path (Dequantize dispatch
// + re-encode) using F32 as a trivially-constructed source.
func TestRepackAffine8F32(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	rows, cols := 128, 256
	n := rows * cols
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64()) * 0.03
	}
	raw := make([]byte, n*4)
	for i, v := range x {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	enc, err := RepackAffine8(raw, 0, n) // type 0 = F32
	if err != nil {
		t.Fatal(err)
	}
	dec := DequantizeAffine8(enc, n)
	rl := relL2(x, dec)
	t.Logf("RepackAffine8(F32) relL2 = %.3e", rl)
	if rl > 1e-2 {
		t.Fatalf("relL2 %.3e too large", rl)
	}
}

// TestRepackAffine8F16 exercises the F16 source path.
func TestRepackAffine8F16(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	n := 512
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64()) * 0.05
	}
	raw := make([]byte, n*2)
	for i, v := range x {
		binary.LittleEndian.PutUint16(raw[i*2:], F32ToF16(v))
	}
	enc, err := RepackAffine8(raw, 1, n) // type 1 = F16
	if err != nil {
		t.Fatal(err)
	}
	dec := DequantizeAffine8(enc, n)
	// Source itself is f16-rounded; tolerance is dominated by that + 8-bit affine.
	if rl := relL2(x, dec); rl > 2e-2 {
		t.Fatalf("relL2 %.3e too large", rl)
	}
}

func TestF32ToF16(t *testing.T) {
	cases := []struct {
		in   float32
		want uint16
	}{
		{0, 0x0000},
		{1, 0x3C00},
		{-1, 0xBC00},
		{0.5, 0x3800},
		{2, 0x4000},
	}
	for _, c := range cases {
		if got := F32ToF16(c.in); got != c.want {
			t.Errorf("F32ToF16(%v) = %#04x, want %#04x", c.in, got, c.want)
		}
	}
}
