package quant

import (
	"encoding/binary"
	"math"
)

// DequantizeInto dequantizes data into an existing buffer, avoiding allocation.
// dst must have length >= n. This is the key for reusable-buffer SIMD matmul.
func DequantizeInto(dst []float32, data []byte, ggmlType uint32, n int) {
	switch ggmlType {
	case 0: // F32
		dequantizeF32Into(dst, data, n)
	case 1: // F16
		dequantizeF16Into(dst, data, n)
	case 2: // Q4_0
		dequantizeQ4_0Into(dst, data, n)
	case 3: // Q4_1
		dequantizeQ4_1Into(dst, data, n)
	case 6: // Q5_0
		dequantizeQ5_0Into(dst, data, n)
	case 7: // Q5_1
		dequantizeQ5_1Into(dst, data, n)
	case 8: // Q8_0
		dequantizeQ8_0Into(dst, data, n)
	case 9: // Q8_1
		dequantizeQ8_1Into(dst, data, n)
	case 10: // Q2_K
		dequantizeQ2_KInto(dst, data, n)
	case 11: // Q3_K
		dequantizeQ3_KInto(dst, data, n)
	case 12: // Q4_K
		dequantizeQ4_KInto(dst, data, n)
	case 13: // Q5_K
		dequantizeQ5_KInto(dst, data, n)
	case 14: // Q6_K
		dequantizeQ6_KInto(dst, data, n)
	case 16: // IQ2_XXS
		dequantizeIQ2XXSInto(dst, data, n)
	case 17: // IQ2_XS
		dequantizeIQ2XSInto(dst, data, n)
	case 18: // IQ3_XXS
		dequantizeIQ3XXSInto(dst, data, n)
	case 19: // IQ1_S
		dequantizeIQ1SInto(dst, data, n)
	case 20: // IQ4_NL
		dequantizeIQ4_NLInto(dst, data, n)
	case 21: // IQ3_S
		dequantizeIQ3SInto(dst, data, n)
	case 22: // IQ2_S
		dequantizeIQ2SInto(dst, data, n)
	case 23: // IQ4_XS
		dequantizeIQ4_XSInto(dst, data, n)
	case 29: // IQ1_M
		dequantizeIQ1MInto(dst, data, n)
	default:
		// Fallback: allocate and copy
		floats, _ := Dequantize(data, ggmlType, n)
		copy(dst, floats)
	}
}

// Fallback Into wrappers for IQ types not yet optimized
func dequantizeIQ2XXSInto(dst []float32, data []byte, n int) {
	copy(dst, DequantizeIQ2XXS(data, n))
}
func dequantizeIQ2XSInto(dst []float32, data []byte, n int) {
	copy(dst, DequantizeIQ2XS(data, n))
}
func dequantizeIQ1SInto(dst []float32, data []byte, n int) {
	copy(dst, DequantizeIQ1S(data, n))
}
func dequantizeIQ3SInto(dst []float32, data []byte, n int) {
	copy(dst, DequantizeIQ3S(data, n))
}
func dequantizeIQ2SInto(dst []float32, data []byte, n int) {
	copy(dst, DequantizeIQ2S(data, n))
}
func dequantizeIQ1MInto(dst []float32, data []byte, n int) {
	copy(dst, DequantizeIQ1M(data, n))
}

func dequantizeIQ3XXSInto(dst []float32, data []byte, n int) {
	numBlocks := n / BlockSizeIQ3XXS
	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ3XXS
		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)
		qsOff := off + 2
		scalesSignsOff := off + 2 + 64
		outBase := block * BlockSizeIQ3XXS
		gridOff := qsOff
		for ib32 := 0; ib32 < 8; ib32++ {
			aux32 := binary.LittleEndian.Uint32(data[scalesSignsOff+ib32*4:])
			db := d * (0.5 + float32(aux32>>28)) * 0.5
			for l := 0; l < 4; l++ {
				signIdx := (aux32 >> (7 * uint(l))) & 127
				signs := ksigns_iq2xs[signIdx]
				grid1 := iq3xxs_grid[data[gridOff+2*l]]
				grid2 := iq3xxs_grid[data[gridOff+2*l+1]]
				oIdx := outBase + ib32*32 + l*8
				for j := 0; j < 4; j++ {
					gridVal := float32(uint8(grid1 >> (8 * uint(j))))
					if signs&kmask_iq2xs[j] != 0 {
						dst[oIdx+j] = -db * gridVal
					} else {
						dst[oIdx+j] = db * gridVal
					}
				}
				for j := 0; j < 4; j++ {
					gridVal := float32(uint8(grid2 >> (8 * uint(j))))
					if signs&kmask_iq2xs[j+4] != 0 {
						dst[oIdx+4+j] = -db * gridVal
					} else {
						dst[oIdx+4+j] = db * gridVal
					}
				}
			}
			gridOff += 8
		}
	}
}

func dequantizeIQ4_XSInto(dst []float32, data []byte, n int) {
	numBlocks := n / BlockSizeIQ4_XS
	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ4_XS
		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)
		scalesH := uint16(data[off+2]) | uint16(data[off+3])<<8
		scalesLOff := off + 4
		qsOff := off + 8
		outOff := block * BlockSizeIQ4_XS
		for ib := 0; ib < 8; ib++ {
			lo := (data[scalesLOff+ib/2] >> uint(4*(ib%2))) & 0xf
			hi := (byte(scalesH>>uint(2*ib)) & 3) << 4
			ls := int(lo | hi)
			dl := d * float32(ls-32)
			for j := 0; j < 16; j++ {
				qByte := data[qsOff+ib*16+j]
				dst[outOff+j] = dl * float32(kvalues_iq4nl[qByte&0xf])
				dst[outOff+j+16] = dl * float32(kvalues_iq4nl[qByte>>4])
			}
			outOff += 32
		}
	}
}

func dequantizeIQ4_NLInto(dst []float32, data []byte, n int) {
	numBlocks := n / BlockSizeIQ4_NL
	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ4_NL
		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)
		base := block * BlockSizeIQ4_NL
		for j := 0; j < 16; j++ {
			qByte := data[off+2+j]
			dst[base+j] = d * float32(kvalues_iq4nl[qByte&0xf])
			dst[base+j+16] = d * float32(kvalues_iq4nl[qByte>>4])
		}
	}
}

// ── F32 ────────────────────────────────────────────────────────

func dequantizeF32Into(dst []float32, data []byte, n int) {
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		dst[i] = math.Float32frombits(bits)
	}
}

// ── F16 ────────────────────────────────────────────────────────

func dequantizeF16Into(dst []float32, data []byte, n int) {
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint16(data[i*2:])
		dst[i] = float16ToFloat32(bits)
	}
}

// ── Q4_0: 32 values per 18-byte block ──────────────────────────

func dequantizeQ4_0Into(dst []float32, data []byte, n int) {
	numBlocks := n / 32
	for block := 0; block < numBlocks; block++ {
		off := block * 18
		d := float16ToFloat32(uint16(data[off]) | uint16(data[off+1])<<8)
		base := block * 32
		for j := 0; j < 16; j++ {
			qByte := data[off+2+j]
			dst[base+j] = float32(int(qByte&0x0F)-8) * d
			dst[base+j+16] = float32(int(qByte>>4)-8) * d
		}
	}
}

// ── Q4_1: 32 values per 20-byte block ──────────────────────────

func dequantizeQ4_1Into(dst []float32, data []byte, n int) {
	numBlocks := n / 32
	for block := 0; block < numBlocks; block++ {
		off := block * 20
		d := float16ToFloat32(uint16(data[off]) | uint16(data[off+1])<<8)
		m := float16ToFloat32(uint16(data[off+2]) | uint16(data[off+3])<<8)
		base := block * 32
		for j := 0; j < 16; j++ {
			qByte := data[off+4+j]
			dst[base+j] = float32(qByte&0x0F)*d + m
			dst[base+j+16] = float32(qByte>>4)*d + m
		}
	}
}

// ── Q5_0: 32 values per 22-byte block ──────────────────────────

func dequantizeQ5_0Into(dst []float32, data []byte, n int) {
	numBlocks := n / 32
	for block := 0; block < numBlocks; block++ {
		off := block * 22
		d := float16ToFloat32(uint16(data[off]) | uint16(data[off+1])<<8)
		qh := uint32(data[off+2]) | uint32(data[off+3])<<8 |
			uint32(data[off+4])<<16 | uint32(data[off+5])<<24
		base := block * 32
		for j := 0; j < 32; j++ {
			var q int
			if j < 16 {
				q = int(data[off+6+j] & 0x0F)
			} else {
				q = int(data[off+6+j-16] >> 4)
			}
			if (qh>>uint(j))&1 != 0 {
				q |= 0x10
			}
			dst[base+j] = float32(q-16) * d
		}
	}
}

// ── Q5_1: 32 values per 24-byte block ──────────────────────────

func dequantizeQ5_1Into(dst []float32, data []byte, n int) {
	numBlocks := n / 32
	for block := 0; block < numBlocks; block++ {
		off := block * 24
		d := float16ToFloat32(uint16(data[off]) | uint16(data[off+1])<<8)
		m := float16ToFloat32(uint16(data[off+2]) | uint16(data[off+3])<<8)
		qh := uint32(data[off+4]) | uint32(data[off+5])<<8 |
			uint32(data[off+6])<<16 | uint32(data[off+7])<<24
		base := block * 32
		for j := 0; j < 16; j++ {
			qByte := data[off+8+j]
			x0 := int(qByte & 0x0F)
			xh0 := int((qh >> uint(j)) & 1)
			x0 |= xh0 << 4
			x1 := int(qByte >> 4)
			xh1 := int((qh >> uint(j+16)) & 1)
			x1 |= xh1 << 4
			dst[base+j] = float32(x0)*d + m
			dst[base+j+16] = float32(x1)*d + m
		}
	}
}

// ── Q8_0: 32 values per 34-byte block ──────────────────────────

func dequantizeQ8_0Into(dst []float32, data []byte, n int) {
	numBlocks := n / 32
	for block := 0; block < numBlocks; block++ {
		off := block * 34
		d := float16ToFloat32(uint16(data[off]) | uint16(data[off+1])<<8)
		base := block * 32
		for j := 0; j < 32; j++ {
			dst[base+j] = float32(int8(data[off+2+j])) * d
		}
	}
}

// ── Q8_1: 32 values per 36-byte block ──────────────────────────

func dequantizeQ8_1Into(dst []float32, data []byte, n int) {
	numBlocks := n / 32
	for block := 0; block < numBlocks; block++ {
		off := block * 36
		d := float16ToFloat32(uint16(data[off]) | uint16(data[off+1])<<8)
		base := block * 32
		for j := 0; j < 32; j++ {
			dst[base+j] = float32(int8(data[off+4+j])) * d
		}
	}
}

// ── Q2_K: 256 values per 84-byte block ─────────────────────────

func dequantizeQ2_KInto(dst []float32, data []byte, n int) {
	numBlocks := n / 256
	for block := 0; block < numBlocks; block++ {
		off := block * 84
		scOff := off
		qOff := off + 16
		d := float16ToFloat32(uint16(data[off+80]) | uint16(data[off+81])<<8)
		dmin := float16ToFloat32(uint16(data[off+82]) | uint16(data[off+83])<<8)
		outOff := block * 256
		is := 0
		for n128 := 0; n128 < 2; n128++ {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				sc := data[scOff+is]
				is++
				dl := d * float32(sc&0xF)
				ml := dmin * float32(sc>>4)
				for l := 0; l < 16; l++ {
					q := (data[qOff+l] >> shift) & 3
					dst[outOff] = dl*float32(q) - ml
					outOff++
				}
				sc = data[scOff+is]
				is++
				dl = d * float32(sc&0xF)
				ml = dmin * float32(sc>>4)
				for l := 0; l < 16; l++ {
					q := (data[qOff+l+16] >> shift) & 3
					dst[outOff] = dl*float32(q) - ml
					outOff++
				}
				shift += 2
			}
			qOff += 32
		}
	}
}

// ── Q3_K: 256 values per 110-byte block ────────────────────────

func dequantizeQ3_KInto(dst []float32, data []byte, n int) {
	numBlocks := n / 256
	const kmask1 = uint32(0x03030303)
	const kmask2 = uint32(0x0f0f0f0f)

	for block := 0; block < numBlocks; block++ {
		off := block * 110
		dAll := float16ToFloat32(uint16(data[off+108]) | uint16(data[off+109])<<8)
		hmOff := off
		qOff := off + 32
		scOff := off + 96

		var aux [4]uint32
		for i := 0; i < 12; i++ {
			byteIdx := i / 4
			shift := uint((i % 4) * 8)
			aux[byteIdx] |= uint32(data[scOff+i]) << shift
		}
		tmp := aux[2]
		aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
		aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
		aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
		aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)

		var scales [16]int8
		for i := 0; i < 16; i++ {
			scales[i] = int8(byte(aux[i/4] >> uint((i%4)*8)))
		}

		outOff := block * 256
		is := 0
		m := byte(1)
		for n128 := 0; n128 < 2; n128++ {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				dl := dAll * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					q2 := int((data[qOff+l] >> shift) & 3)
					hBit := 0
					if data[hmOff+l]&m == 0 {
						hBit = 4
					}
					dst[outOff] = dl * float32(q2-hBit)
					outOff++
				}
				dl = dAll * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					q2 := int((data[qOff+l+16] >> shift) & 3)
					hBit := 0
					if data[hmOff+l+16]&m == 0 {
						hBit = 4
					}
					dst[outOff] = dl * float32(q2-hBit)
					outOff++
				}
				shift += 2
				m <<= 1
			}
			qOff += 32
		}
	}
}

// ── Q4_K: 256 values per 144-byte block ────────────────────────

func dequantizeQ4_KInto(dst []float32, data []byte, n int) {
	numBlocks := n / 256
	for block := 0; block < numBlocks; block++ {
		blockOff := block * 144
		d := float16ToFloat32(uint16(data[blockOff]) | uint16(data[blockOff+1])<<8)
		dmin := float16ToFloat32(uint16(data[blockOff+2]) | uint16(data[blockOff+3])<<8)

		scalesOff := blockOff + 4
		var sc [8]float32
		var mn [8]float32
		for i := 0; i < 4; i++ {
			sc[i] = d * float32(data[scalesOff+i]&0x3F)
			mn[i] = dmin * float32(data[scalesOff+4+i]&0x3F)
		}
		for i := 0; i < 4; i++ {
			scHi := (data[scalesOff+i] >> 6) & 0x03
			mnHi := (data[scalesOff+4+i] >> 6) & 0x03
			scLo := data[scalesOff+8+i] & 0x0F
			mnLo := (data[scalesOff+8+i] >> 4) & 0x0F
			sc[4+i] = d * float32(scLo|scHi<<4)
			mn[4+i] = dmin * float32(mnLo|mnHi<<4)
		}

		qsOff := blockOff + 16
		outOff := block * 256
		is := 0
		for grp := 0; grp < 4; grp++ {
			d1 := sc[is]
			m1 := mn[is]
			d2 := sc[is+1]
			m2 := mn[is+1]
			qOff := qsOff + grp*32
			for l := 0; l < 32; l++ {
				dst[outOff] = d1*float32(data[qOff+l]&0x0F) - m1
				outOff++
			}
			for l := 0; l < 32; l++ {
				dst[outOff] = d2*float32(data[qOff+l]>>4) - m2
				outOff++
			}
			is += 2
		}
	}
}

// ── Q5_K: 256 values per 176-byte block ────────────────────────

func dequantizeQ5_KInto(dst []float32, data []byte, n int) {
	numBlocks := n / 256
	for block := 0; block < numBlocks; block++ {
		blockOff := block * 176
		d := float16ToFloat32(uint16(data[blockOff]) | uint16(data[blockOff+1])<<8)
		dmin := float16ToFloat32(uint16(data[blockOff+2]) | uint16(data[blockOff+3])<<8)

		scalesOff := blockOff + 4
		var sc [8]float32
		var mn [8]float32
		for i := 0; i < 4; i++ {
			sc[i] = d * float32(data[scalesOff+i]&0x3F)
			mn[i] = dmin * float32(data[scalesOff+4+i]&0x3F)
		}
		for i := 0; i < 4; i++ {
			scHi := (data[scalesOff+i] >> 6) & 0x03
			mnHi := (data[scalesOff+4+i] >> 6) & 0x03
			scLo := data[scalesOff+8+i] & 0x0F
			mnLo := (data[scalesOff+8+i] >> 4) & 0x0F
			sc[4+i] = d * float32(scLo|scHi<<4)
			mn[4+i] = dmin * float32(mnLo|mnHi<<4)
		}

		qhOff := blockOff + 16
		qsOff := blockOff + 48
		outOff := block * 256
		is := 0
		u1 := byte(1)
		u2 := byte(2)
		for grp := 0; grp < 4; grp++ {
			d1 := sc[is]
			m1 := mn[is]
			d2 := sc[is+1]
			m2 := mn[is+1]
			qlOff := qsOff + grp*32
			for l := 0; l < 32; l++ {
				q := int(data[qlOff+l] & 0x0F)
				if data[qhOff+l]&u1 != 0 {
					q |= 16
				}
				dst[outOff] = float32(q)*d1 - m1
				outOff++
			}
			for l := 0; l < 32; l++ {
				q := int(data[qlOff+l] >> 4)
				if data[qhOff+l]&u2 != 0 {
					q |= 16
				}
				dst[outOff] = float32(q)*d2 - m2
				outOff++
			}
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
}

// ── Q6_K: 256 values per 210-byte block ────────────────────────

func dequantizeQ6_KInto(dst []float32, data []byte, n int) {
	numBlocks := n / 256
	for block := 0; block < numBlocks; block++ {
		blockOff := block * 210
		d := float16ToFloat32(uint16(data[blockOff+208]) | uint16(data[blockOff+209])<<8)
		qlBase := blockOff
		qhBase := blockOff + 128
		scBase := blockOff + 192
		outBase := block * 256

		for n128 := 0; n128 < 2; n128++ {
			qlOff := qlBase + n128*64
			qhOff := qhBase + n128*32
			for l := 0; l < 32; l++ {
				qlByte0 := data[qlOff+l]
				qlByte32 := data[qlOff+l+32]
				qhByte := data[qhOff+l]

				q1 := (int(qlByte0&0x0F) | (int((qhByte>>0)&3) << 4)) - 32
				q2 := (int(qlByte32&0x0F) | (int((qhByte>>2)&3) << 4)) - 32
				q3 := (int(qlByte0>>4) | (int((qhByte>>4)&3) << 4)) - 32
				q4 := (int(qlByte32>>4) | (int((qhByte>>6)&3) << 4)) - 32

				is := 8*n128 + l/16
				sc0 := float32(int8(data[scBase+is]))
				sc2 := float32(int8(data[scBase+is+2]))
				sc4 := float32(int8(data[scBase+is+4]))
				sc6 := float32(int8(data[scBase+is+6]))

				pos := outBase + n128*128 + l
				dst[pos+0] = d * sc0 * float32(q1)
				dst[pos+32] = d * sc2 * float32(q2)
				dst[pos+64] = d * sc4 * float32(q3)
				dst[pos+96] = d * sc6 * float32(q4)
			}
		}
	}
}
