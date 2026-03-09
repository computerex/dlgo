package quant

import "math"

// MXFP4 quantization (type 39)
// Block structure: 17 bytes per block of 32 values
// Layout:
//   e:     uint8 E8M0 shared exponent (1 byte)
//   qs[16]: 4-bit indices packed in nibble pairs (16 bytes)
// Each nibble indexes into kvalues_mxfp4[16].
// Formula: y = e8m0_to_half(e) * kvalues_mxfp4[nibble]
// Low nibbles → values[0..15], high nibbles → values[16..31].

const BlockSizeMXFP4 = 32
const BlockBytesMXFP4 = 17

var kvalues_mxfp4 = [16]int8{
	0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
}

func e8m0ToFloat32Half(e uint8) float32 {
	if e < 2 {
		bits := uint32(0x00200000) << e
		return math.Float32frombits(bits)
	}
	bits := uint32(e-1) << 23
	return math.Float32frombits(bits)
}

func DequantizeMXFP4(data []byte, n int) []float32 {
	result := make([]float32, n)
	numBlocks := n / BlockSizeMXFP4

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesMXFP4
		d := e8m0ToFloat32Half(data[off])
		base := block * BlockSizeMXFP4

		for j := 0; j < 16; j++ {
			qByte := data[off+1+j]
			result[base+j] = d * float32(kvalues_mxfp4[qByte&0xf])
			result[base+j+16] = d * float32(kvalues_mxfp4[qByte>>4])
		}
	}
	return result
}
