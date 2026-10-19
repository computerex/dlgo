package quant

import (
	"math"
	"math/rand"
	"testing"
)

func TestAffine4BlockBytes(t *testing.T) {
	if Affine4BlockBytes != 20 {
		t.Fatalf("Affine4BlockBytes = %d, want 20", Affine4BlockBytes)
	}
	if Affine4BlockBytes%4 != 0 {
		t.Fatalf("Affine4BlockBytes must be 4-aligned")
	}
	if Affine4GroupSize != 32 {
		t.Fatalf("Affine4GroupSize = %d, want 32", Affine4GroupSize)
	}
}

func TestAffine4RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	n := 32 * 137
	x := make([]float32, n)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 0.3
	}
	enc := QuantizeAffine4(x)
	if len(enc) != (n/Affine4GroupSize)*Affine4BlockBytes {
		t.Fatalf("len(enc)=%d want %d", len(enc), (n/Affine4GroupSize)*Affine4BlockBytes)
	}
	dec := DequantizeAffine4(enc, n)
	var num, den, maxAbs float64
	for i := range x {
		d := float64(dec[i] - x[i])
		num += d * d
		den += float64(x[i]) * float64(x[i])
		if math.Abs(d) > maxAbs {
			maxAbs = math.Abs(d)
		}
	}
	rl := math.Sqrt(num / den)
	t.Logf("Affine4 round-trip relL2 = %.3e maxAbs = %.3e", rl, maxAbs)
	// A 4-bit quantizer (15 levels/group) gives ~5-7% relL2 on a uniform signal;
	// a materially larger number would indicate a layout/scale bug.
	if rl > 1.0e-1 {
		t.Fatalf("Affine4 round-trip relL2 %.3e too large", rl)
	}
}

func TestAffine4ConstantGroup(t *testing.T) {
	n := 64
	x := make([]float32, n)
	for i := range x {
		x[i] = -0.25 // constant group (range 0) -> d=0, m=-0.25
	}
	dec := DequantizeAffine4(QuantizeAffine4(x), n)
	for i := range dec {
		if math.Abs(float64(dec[i]-x[i])) > 1e-3 {
			t.Fatalf("constant group mismatch at %d: got %v want %v", i, dec[i], x[i])
		}
	}
}
