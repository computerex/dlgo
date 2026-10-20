package quant

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// serialRepack8 is the reference implementation (the pre-parallel body of
// RepackAffine8): dequantize the whole tensor, then requantize.
func serialRepack8(t *testing.T, data []byte, ggmlType uint32, n int) []byte {
	t.Helper()
	f32, err := Dequantize(data, ggmlType, n)
	if err != nil {
		t.Fatalf("dequant: %v", err)
	}
	return QuantizeAffine8(f32)
}

// f32Bytes encodes floats as little-endian F32 (GGML type 0 raw data).
func f32Bytes(x []float32) []byte {
	out := make([]byte, len(x)*4)
	for i, v := range x {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// f16Bytes encodes floats as little-endian IEEE-754 half precision (type 1).
func f16Bytes(x []float32) []byte {
	out := make([]byte, len(x)*2)
	for i, v := range x {
		binary.LittleEndian.PutUint16(out[i*2:], F32ToF16(v))
	}
	return out
}

// TestRepackAffine8Parallel verifies that the chunk-parallel RepackAffine8
// produces byte-identical output to the serial dequantize+requantize path, for
// element-wise types (F32/F16) and a 256-element-block type (Q8_K), at sizes
// both above (parallel path) and below (serial path) the internal threshold.
func TestRepackAffine8Parallel(t *testing.T) {
	// Deterministic, finite input values.
	mk := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*0.01))*0.7 + float32((i%7)-3)*0.05
		}
		return x
	}

	const big = 2 << 20 // 2,097,152 elements: multiple of 256, above the threshold
	xBig := mk(big)
	q8k := QuantizeQ8_K(xBig) // 256-element blocks → exercises block-aligned chunks

	cases := []struct {
		name string
		data []byte
		typ  uint32
		n    int
	}{
		{"F32_big", f32Bytes(xBig), 0, big},
		{"F16_big", f16Bytes(xBig), 1, big},
		{"Q8K_big", q8k, 15, big},
		{"F16_small_serial", f16Bytes(mk(4096)), 1, 4096},
		{"Q8K_small_serial", QuantizeQ8_K(mk(2048)), 15, 2048},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			want := serialRepack8(t, c.data, c.typ, c.n)
			got, err := RepackAffine8(c.data, c.typ, c.n)
			if err != nil {
				t.Fatalf("RepackAffine8: %v", err)
			}
			if len(got) != len(want) {
				t.Fatalf("length mismatch: got %d want %d", len(got), len(want))
			}
			if !bytes.Equal(got, want) {
				// Report the first differing byte for diagnosis.
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("byte %d differs: got %#x want %#x", i, got[i], want[i])
					}
				}
				t.Fatalf("outputs differ")
			}
		})
	}
}
