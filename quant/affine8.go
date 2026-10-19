package quant

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Affine8 is the canonical, kernel-friendly weight format used by the unified
// DiT/LLM GEMM path. The idea: pay the cost of understanding every GGUF block
// layout ONCE, at load time, by converting every quantized weight tensor into a
// single trivial format. After that the runtime has exactly one weight type and
// exactly one GEMM kernel — optimizations no longer have to be applied to each
// quantization scheme separately.
//
// Layout (one block = 32 weights = 36 bytes, every block 4-byte aligned):
//
//	struct block_affine8 {
//	    uint32_t q32[8];   // 32 codes, 4 codes little-endian per uint32
//	    float16_t d;       // per-block scale
//	    float16_t m;       // per-block offset (min)
//	};
//
// Reconstruction: w = d*float(q) + m, with q in [0,255].
//
// Design notes:
//   - Group size 32 matches the sub-block granularity of the K-quant schemes and
//     divides every DiT/LLM column dimension, so a GEMM k-tile of 32 aligns with
//     exactly one block → the (d,m) are uniform across a k-tile (hoistable).
//   - Storing the codes as uint32 gives the shader aligned, vectorized loads
//     instead of the per-element byte gathers that bottlenecked the per-quant
//     kernels.
//   - Affine (scale + offset) preserves the asymmetric min-offset of Q4_K/Q5_K,
//     and 8-bit depth is strictly finer than any source format we ship (≤6-bit),
//     so the conversion is effectively lossless.
const (
	Affine8GroupSize  = 32
	Affine8BlockBytes = 36 // 8*4 (q32) + 2 (d) + 2 (m)
)

// RepackAffine8 converts a quantized tensor of the given GGML type into the
// canonical Affine8 format. n is the element count (must be a multiple of the
// source block size and of 32).
func RepackAffine8(data []byte, ggmlType uint32, n int) ([]byte, error) {
	if n%Affine8GroupSize != 0 {
		return nil, fmt.Errorf("affine8: element count %d not a multiple of %d", n, Affine8GroupSize)
	}
	f32, err := Dequantize(data, ggmlType, n)
	if err != nil {
		return nil, err
	}
	return QuantizeAffine8(f32), nil
}

// QuantizeAffine8 encodes a float32 slice into the canonical Affine8 format.
// len(x) must be a multiple of Affine8GroupSize.
func QuantizeAffine8(x []float32) []byte {
	ng := len(x) / Affine8GroupSize
	out := make([]byte, ng*Affine8BlockBytes)
	for g := 0; g < ng; g++ {
		base := g * Affine8GroupSize
		mn, mx := x[base], x[base]
		for i := 1; i < Affine8GroupSize; i++ {
			v := x[base+i]
			if v < mn {
				mn = v
			}
			if v > mx {
				mx = v
			}
		}
		// scale = range / 255
		d := (mx - mn) / 255.0
		var inv float32
		if d > 0 {
			inv = 1.0 / d
		}
		off := g * Affine8BlockBytes
		var qw [8]uint32
		for i := 0; i < Affine8GroupSize; i++ {
			q := uint32(0)
			if inv != 0 {
				r := math.Round(float64((x[base+i] - mn) * inv))
				if r < 0 {
					r = 0
				} else if r > 255 {
					r = 255
				}
				q = uint32(r)
			}
			qw[i>>2] |= q << (8 * uint(i&3))
		}
		for i := 0; i < 8; i++ {
			binary.LittleEndian.PutUint32(out[off+i*4:], qw[i])
		}
		binary.LittleEndian.PutUint16(out[off+32:], F32ToF16(d))
		binary.LittleEndian.PutUint16(out[off+34:], F32ToF16(mn))
	}
	return out
}

// DequantizeAffine8 reconstructs a float32 slice from canonical Affine8 bytes.
// Used by tests and CPU fallbacks.
func DequantizeAffine8(data []byte, n int) []float32 {
	out := make([]float32, n)
	ng := n / Affine8GroupSize
	for g := 0; g < ng; g++ {
		off := g * Affine8BlockBytes
		d := float16ToFloat32(binary.LittleEndian.Uint16(data[off+32:]))
		m := float16ToFloat32(binary.LittleEndian.Uint16(data[off+34:]))
		base := g * Affine8GroupSize
		for i := 0; i < Affine8GroupSize; i++ {
			qw := binary.LittleEndian.Uint32(data[off+(i>>2)*4:])
			q := float32((qw >> (8 * uint(i&3))) & 0xFF)
			out[base+i] = d*q + m
		}
	}
	return out
}

// F32ToF16 converts a float32 to IEEE-754 half precision bits.
func F32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int32((b>>23)&0xFF) - 127 + 15
	mant := b & 0x7FFFFF

	if (b & 0x7FFFFFFF) == 0 {
		return sign // +/-0
	}
	if exp >= 0x1F {
		if (b & 0x7FFFFFFF) > 0x7F800000 {
			return sign | 0x7E00 // NaN
		}
		return sign | 0x7C00 // Inf
	}
	if exp <= 0 {
		if exp < -10 {
			return sign // underflow to zero
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		return sign | uint16(mant>>shift)
	}
	return sign | uint16(exp<<10) | uint16(mant>>13)
}
