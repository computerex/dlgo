package quant

import (
	"encoding/binary"
	"math"
)

// Q8_K quantization
// Block structure: 292 bytes per super-block of 256 values
// Layout:
//   d: float32 delta (4 bytes, offset 0)
//   qs[256]: signed 8-bit quants (256 bytes, offset 4)
//   bsums[16]: int16 sums of groups of 16 (32 bytes, offset 260) — unused for dequant
// Formula: y = d * qs[j]

const BlockSizeQ8_K = 256
const BlockBytesQ8_K = 292

// DequantizeQ8_K converts Q8_K quantized data to float32.
func DequantizeQ8_K(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeQ8_K

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesQ8_K

		// d is float32 (not float16!)
		d := math.Float32frombits(binary.LittleEndian.Uint32(data[off : off+4]))

		base := block * BlockSizeQ8_K
		for j := 0; j < 256; j++ {
			q := int8(data[off+4+j])
			result[base+j] = d * float32(q)
		}
	}
	return result
}

// QuantizeQ8_K quantizes float32 values into GGML Q8_K blocks.
// Each 256-value block uses one shared scale and 16 block sums.
func QuantizeQ8_K(x []float32) []byte {
	if len(x)%BlockSizeQ8_K != 0 {
		panic("QuantizeQ8_K: input length must be a multiple of 256")
	}
	out := make([]byte, (len(x)/BlockSizeQ8_K)*BlockBytesQ8_K)
	for block := 0; block < len(x)/BlockSizeQ8_K; block++ {
		base := block * BlockSizeQ8_K
		off := block * BlockBytesQ8_K

		maxAbs := float32(0)
		for i := 0; i < BlockSizeQ8_K; i++ {
			v := float32(math.Abs(float64(x[base+i])))
			if v > maxAbs {
				maxAbs = v
			}
		}
		d := float32(1.0)
		if maxAbs > 0 {
			d = maxAbs / 127.0
		}
		invD := float32(0)
		if d != 0 {
			invD = 1.0 / d
		}
		binary.LittleEndian.PutUint32(out[off:off+4], math.Float32bits(d))

		for g := 0; g < 16; g++ {
			sum := int16(0)
			for i := 0; i < 16; i++ {
				idx := base + g*16 + i
				q := int32(math.Round(float64(x[idx] * invD)))
				if q > 127 {
					q = 127
				}
				if q < -128 {
					q = -128
				}
				out[off+4+g*16+i] = byte(int8(q))
				sum += int16(q)
			}
			binary.LittleEndian.PutUint16(out[off+260+g*2:off+260+g*2+2], uint16(sum))
		}
	}
	return out
}
