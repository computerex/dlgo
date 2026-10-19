package quant

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Affine4 is the byte-lean sibling of Affine8 (see affine8.go): the canonical
// weight format for the sub-8-bit sources, where Affine8's 1.125 B/weight would
// cost decode bandwidth. One block = 32 weights = 20 bytes = 0.625 B/weight,
// which is within 11% of the densest GGML schemes (Q4_K = 0.5625 B/w) and so
// keeps single-token decode close to the native kernels while still giving the
// runtime exactly ONE weight type / ONE matvec / ONE GEMM for all ≤5-bit
// sources (Q4_0/1, Q5_0/1, Q2_K, Q3_K, IQ*, TQ*, MXFP4).
//
// Layout (one block = 32 weights = 20 bytes, 4-byte aligned):
//
//	struct block_affine4 {
//	    uint32_t q32[4];   // 32 codes, 8 four-bit codes per uint32 (LE)
//	    float16_t d;       // per-block scale
//	    float16_t m;       // per-block offset (min)
//	};
//
// Reconstruction: w = d*float(q) + m, with q in [0,15].
const (
	Affine4GroupSize  = 32
	Affine4BlockBytes = 20 // 4*4 (q32) + 2 (d) + 2 (m)
)

// RepackAffine4 converts a quantized tensor of the given GGML type into the
// canonical Affine4 format. n is the element count (must be a multiple of 32).
func RepackAffine4(data []byte, ggmlType uint32, n int) ([]byte, error) {
	if n%Affine4GroupSize != 0 {
		return nil, fmt.Errorf("affine4: element count %d not a multiple of %d", n, Affine4GroupSize)
	}
	f32, err := Dequantize(data, ggmlType, n)
	if err != nil {
		return nil, err
	}
	return QuantizeAffine4(f32), nil
}

// QuantizeAffine4 encodes a float32 slice into canonical Affine4 format.
// len(x) must be a multiple of Affine4GroupSize.
func QuantizeAffine4(x []float32) []byte {
	ng := len(x) / Affine4GroupSize
	out := make([]byte, ng*Affine4BlockBytes)
	for g := 0; g < ng; g++ {
		base := g * Affine4GroupSize
		mn, mx := x[base], x[base]
		for i := 1; i < Affine4GroupSize; i++ {
			v := x[base+i]
			if v < mn {
				mn = v
			}
			if v > mx {
				mx = v
			}
		}
		d := (mx - mn) / 15.0
		var inv float32
		if d > 0 {
			inv = 1.0 / d
		}
		off := g * Affine4BlockBytes
		var qw [4]uint32
		for i := 0; i < Affine4GroupSize; i++ {
			q := uint32(0)
			if inv != 0 {
				r := math.Round(float64((x[base+i] - mn) * inv))
				if r < 0 {
					r = 0
				} else if r > 15 {
					r = 15
				}
				q = uint32(r)
			}
			qw[i>>3] |= q << (4 * uint(i&7))
		}
		for i := 0; i < 4; i++ {
			binary.LittleEndian.PutUint32(out[off+i*4:], qw[i])
		}
		binary.LittleEndian.PutUint16(out[off+16:], F32ToF16(d))
		binary.LittleEndian.PutUint16(out[off+18:], F32ToF16(mn))
	}
	return out
}

// DequantizeAffine4 reconstructs a float32 slice from canonical Affine4 bytes.
// Used by tests and CPU fallbacks.
func DequantizeAffine4(data []byte, n int) []float32 {
	out := make([]float32, n)
	ng := n / Affine4GroupSize
	for g := 0; g < ng; g++ {
		off := g * Affine4BlockBytes
		d := float16ToFloat32(binary.LittleEndian.Uint16(data[off+16:]))
		m := float16ToFloat32(binary.LittleEndian.Uint16(data[off+18:]))
		base := g * Affine4GroupSize
		for i := 0; i < Affine4GroupSize; i++ {
			qw := binary.LittleEndian.Uint32(data[off+(i>>3)*4:])
			q := float32((qw >> (4 * uint(i&7))) & 0xF)
			out[base+i] = d*q + m
		}
	}
	return out
}
