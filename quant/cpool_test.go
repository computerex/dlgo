//go:build amd64 && cgo

package quant

import (
	"math"
	"math/rand"
	"testing"
)

// packQ8_0Rows builds a Q8_0 weight buffer of nrows rows × cols floats
// (cols must be a multiple of 32). Each 32-value block is d(f16) + 32 int8.
func packQ8_0Rows(rng *rand.Rand, nrows, cols int) []byte {
	blocksPerRow := cols / 32
	bpr := blocksPerRow * BlockBytesQ8_0
	out := make([]byte, nrows*bpr)
	for r := 0; r < nrows; r++ {
		for b := 0; b < blocksPerRow; b++ {
			off := r*bpr + b*BlockBytesQ8_0
			// scale in [0.005, 0.05]
			scale := 0.005 + rng.Float64()*0.045
			h := float32ToFloat16Bits(float32(scale))
			out[off] = byte(h)
			out[off+1] = byte(h >> 8)
			for j := 0; j < 32; j++ {
				out[off+2+j] = byte(int8(rng.Intn(255) - 127))
			}
		}
	}
	return out
}

// float32ToFloat16Bits converts f32 to f16 bits (truncation is fine for a test).
func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xFF) - 127 + 15
	mant := bits & 0x7FFFFF
	if exp <= 0 {
		return sign
	}
	if exp >= 0x1F {
		return sign | 0x7C00
	}
	return sign | uint16(exp<<10) | uint16(mant>>13)
}

// TestCPoolBatchGEMMMatchesReference verifies the C thread-pool batch GEMM
// (cpool_qq_batch_gemm) produces bit-identical output to the single-threaded
// reference (qq_batch_gemm). If this passes, CPoolBatchHas can be enabled.
func TestCPoolBatchGEMMMatchesReference(t *testing.T) {
	CPoolInit()
	defer CPoolShutdown()

	const (
		nrows   = 4096
		cols    = 256
		nInputs = 32
		wType   = 8 // Q8_0
	)

	rng := rand.New(rand.NewSource(1234))
	w := packQ8_0Rows(rng, nrows, cols)
	bpr := (cols / 32) * BlockBytesQ8_0

	// Build nInputs random float vectors, quantize to Q8_0 (matching wType).
	q8Stride := Q8BufferSize(wType, cols)
	q8Flat := make([]byte, nInputs*q8Stride)
	x := make([]float32, cols)
	for p := 0; p < nInputs; p++ {
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		QuantizeForType(x, q8Flat[p*q8Stride:(p+1)*q8Stride], wType)
	}

	ref := make([]float32, nInputs*nrows)
	QQBatchGEMM(w, wType, q8Flat, q8Stride, nInputs, cols, ref, nrows, nrows, bpr)

	pool := make([]float32, nInputs*nrows)
	CPoolQQBatchGEMM(w, wType, q8Flat, q8Stride, nInputs, cols, pool, nrows, nrows, bpr)

	var maxAbs float64
	var mismatches int
	for i := range ref {
		d := math.Abs(float64(ref[i]) - float64(pool[i]))
		if d > maxAbs {
			maxAbs = d
		}
		if math.Float32bits(ref[i]) != math.Float32bits(pool[i]) {
			mismatches++
		}
	}
	t.Logf("nrows=%d cols=%d nInputs=%d: maxAbsDiff=%.3e bitMismatches=%d/%d",
		nrows, cols, nInputs, maxAbs, mismatches, len(ref))
	if mismatches != 0 {
		t.Errorf("pool batch GEMM differs from reference in %d of %d values (maxAbsDiff=%.3e)",
			mismatches, len(ref), maxAbs)
	}
}
