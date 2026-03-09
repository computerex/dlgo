package quant

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sync"
)

// HasZeroAllocDequantize returns true if the given quantization type has a
// dedicated DequantizeInto implementation that writes directly into a
// caller-provided buffer without heap allocation. Types that return false
// fall back to an allocating path that is unsafe when GC is disabled.
func HasZeroAllocDequantize(ggmlType uint32) bool {
	switch ggmlType {
	case 0, 1, 2, 3, 6, 7, 8, 9,
		10, 11, 12, 13, 14,
		16, 17, 18, 19, 20, 21, 22, 23, 29,
		34, 35, 39:
		return true
	default:
		return false
	}
}

var dequantFallbackOnce sync.Map // ggmlType -> warned

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
	case 34: // TQ1_0
		dequantizeTQ1_0Into(dst, data, n)
	case 35: // TQ2_0
		dequantizeTQ2_0Into(dst, data, n)
	case 39: // MXFP4
		dequantizeMXFP4Into(dst, data, n)
	default:
		if _, warned := dequantFallbackOnce.LoadOrStore(ggmlType, true); !warned {
			fmt.Fprintf(os.Stderr, "[dlgo] WARNING: quant type %d has no zero-alloc DequantizeInto — "+
				"using allocating fallback (will cause memory pressure if GC is disabled)\n", ggmlType)
		}
		floats, _ := Dequantize(data, ggmlType, n)
		copy(dst, floats)
	}
}

func dequantizeIQ2XXSInto(dst []float32, data []byte, n int) {
	numBlocks := n / BlockSizeIQ2XXS

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ2XXS

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)

		qsOff := off + 2 // start of qs (64 bytes of uint16 data)
		outBase := block * BlockSizeIQ2XXS

		for ib32 := 0; ib32 < 8; ib32++ {
			byteOff := qsOff + ib32*8
			aux1 := binary.LittleEndian.Uint32(data[byteOff+4:])

			db := d * (0.5 + float32(aux1>>28)) * 0.25

			for l := 0; l < 4; l++ {
				gridIdx := data[byteOff+l]
				grid := iq2xxs_grid[gridIdx]

				signIdx := (aux1 >> (7 * uint(l))) & 127
				signs := ksigns_iq2xs[signIdx]

				oIdx := outBase + ib32*32 + l*8
				for j := 0; j < 8; j++ {
					gridVal := float32(uint8(grid >> (8 * uint(j))))
					if signs&kmask_iq2xs[j] != 0 {
						dst[oIdx+j] = -db * gridVal
					} else {
						dst[oIdx+j] = db * gridVal
					}
				}
			}
		}
	}
}
func dequantizeIQ2XSInto(dst []float32, data []byte, n int) {
	numBlocks := n / BlockSizeIQ2XS

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ2XS

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)

		qsOff := off + 2
		scOff := off + 2 + 64
		outBase := block * BlockSizeIQ2XS

		for ib32 := 0; ib32 < 8; ib32++ {
			scByte := data[scOff+ib32]
			db0 := d * (0.5 + float32(scByte&0xf)) * 0.25
			db1 := d * (0.5 + float32(scByte>>4)) * 0.25

			for l := 0; l < 4; l++ {
				qIdx := qsOff + (ib32*4+l)*2
				qs := uint16(data[qIdx]) | uint16(data[qIdx+1])<<8

				gridIdx := qs & 511
				signIdx := qs >> 9
				grid := iq2xs_grid[gridIdx]
				signs := ksigns_iq2xs[signIdx]

				var db float32
				if l < 2 {
					db = db0
				} else {
					db = db1
				}

				oIdx := outBase + ib32*32 + l*8
				for j := 0; j < 8; j++ {
					gridVal := float32(uint8(grid >> (8 * uint(j))))
					if signs&kmask_iq2xs[j] != 0 {
						dst[oIdx+j] = -db * gridVal
					} else {
						dst[oIdx+j] = db * gridVal
					}
				}
			}
		}
	}
}
func dequantizeIQ1SInto(dst []float32, data []byte, n int) {
	numBlocks := n / BlockSizeIQ1S

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ1S

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)

		qsOff := off + 2
		qhOff := off + 2 + 32
		outBase := block * BlockSizeIQ1S

		for ib := 0; ib < 8; ib++ {
			qh := uint16(data[qhOff+ib*2]) | uint16(data[qhOff+ib*2+1])<<8

			dl := d * float32(2*int((qh>>12)&7)+1)
			var delta float32
			if qh&0x8000 != 0 {
				delta = -iq1sDelta
			} else {
				delta = iq1sDelta
			}

			for l := 0; l < 4; l++ {
				gridIdx := uint16(data[qsOff+l]) | ((qh >> uint(3*l) & 7) << 8)
				grid := iq1s_grid[gridIdx]

				oIdx := outBase + ib*32 + l*8
				for j := 0; j < 8; j++ {
					gridByte := uint8(grid >> (8 * uint(j)))
					gridVal := float32(int8(gridByte))
					dst[oIdx+j] = dl * (gridVal + delta)
				}
			}
			qsOff += 4
		}
	}
}
func dequantizeIQ3SInto(dst []float32, data []byte, n int) {
	numBlocks := n / BlockSizeIQ3S

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ3S

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)

		qsStart := off + 2
		qhStart := off + 2 + 64
		signStart := off + 2 + 64 + 8
		scStart := off + 2 + 64 + 8 + 32
		outBase := block * BlockSizeIQ3S

		qsOff := qsStart
		qhOff := qhStart
		signsOff := signStart

		for ib32 := 0; ib32 < 8; ib32 += 2 {
			scByte := data[scStart+ib32/2]
			db1 := d * float32(1+2*int(scByte&0xf))
			db2 := d * float32(1+2*int(scByte>>4))

			qh0 := data[qhOff]
			qh1 := data[qhOff+1]

			for l := 0; l < 4; l++ {
				gridIdx1 := uint16(data[qsOff+2*l]) | ((uint16(qh0) << (8 - 2*uint(l))) & 256)
				gridIdx2 := uint16(data[qsOff+2*l+1]) | ((uint16(qh0) << (7 - 2*uint(l))) & 256)
				grid1 := iq3s_grid[gridIdx1]
				grid2 := iq3s_grid[gridIdx2]
				signByte := data[signsOff+l]

				oIdx := outBase + ib32*32 + l*8
				for j := 0; j < 4; j++ {
					gridVal := float32(uint8(grid1 >> (8 * uint(j))))
					if signByte&kmask_iq2xs[j] != 0 {
						dst[oIdx+j] = -db1 * gridVal
					} else {
						dst[oIdx+j] = db1 * gridVal
					}
				}
				for j := 0; j < 4; j++ {
					gridVal := float32(uint8(grid2 >> (8 * uint(j))))
					if signByte&kmask_iq2xs[j+4] != 0 {
						dst[oIdx+4+j] = -db1 * gridVal
					} else {
						dst[oIdx+4+j] = db1 * gridVal
					}
				}
			}
			qsOff += 8
			signsOff += 4

			for l := 0; l < 4; l++ {
				gridIdx1 := uint16(data[qsOff+2*l]) | ((uint16(qh1) << (8 - 2*uint(l))) & 256)
				gridIdx2 := uint16(data[qsOff+2*l+1]) | ((uint16(qh1) << (7 - 2*uint(l))) & 256)
				grid1 := iq3s_grid[gridIdx1]
				grid2 := iq3s_grid[gridIdx2]
				signByte := data[signsOff+l]

				oIdx := outBase + (ib32+1)*32 + l*8
				for j := 0; j < 4; j++ {
					gridVal := float32(uint8(grid1 >> (8 * uint(j))))
					if signByte&kmask_iq2xs[j] != 0 {
						dst[oIdx+j] = -db2 * gridVal
					} else {
						dst[oIdx+j] = db2 * gridVal
					}
				}
				for j := 0; j < 4; j++ {
					gridVal := float32(uint8(grid2 >> (8 * uint(j))))
					if signByte&kmask_iq2xs[j+4] != 0 {
						dst[oIdx+4+j] = -db2 * gridVal
					} else {
						dst[oIdx+4+j] = db2 * gridVal
					}
				}
			}
			qhOff += 2
			qsOff += 8
			signsOff += 4
		}
	}
}
func dequantizeIQ2SInto(dst []float32, data []byte, n int) {
	numBlocks := n / BlockSizeIQ2S

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ2S

		dBits := uint16(data[off]) | uint16(data[off+1])<<8
		d := float16ToFloat32(dBits)

		qsOff := off + 2
		qhOff := off + 2 + 64
		scOff := off + 2 + 64 + 8
		outBase := block * BlockSizeIQ2S

		gridLowOff := qsOff
		signsOff := qsOff + 32

		for ib32 := 0; ib32 < 8; ib32++ {
			scByte := data[scOff+ib32]
			db0 := d * (0.5 + float32(scByte&0xf)) * 0.25
			db1 := d * (0.5 + float32(scByte>>4)) * 0.25
			qhByte := data[qhOff+ib32]

			for l := 0; l < 4; l++ {
				lowByte := uint16(data[gridLowOff])
				highBits := (uint16(qhByte) << (8 - 2*uint(l))) & 0x300
				gridIdx := lowByte | highBits
				grid := iq2s_grid[gridIdx]

				signByte := data[signsOff]

				var db float32
				if l < 2 {
					db = db0
				} else {
					db = db1
				}

				oIdx := outBase + ib32*32 + l*8
				for j := 0; j < 8; j++ {
					gridVal := float32(uint8(grid >> (8 * uint(j))))
					if signByte&kmask_iq2xs[j] != 0 {
						dst[oIdx+j] = -db * gridVal
					} else {
						dst[oIdx+j] = db * gridVal
					}
				}
				gridLowOff++
				signsOff++
			}
		}
	}
}
func dequantizeIQ1MInto(dst []float32, data []byte, n int) {
	numBlocks := n / BlockSizeIQ1M

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesIQ1M

		qsStart := off
		qhStart := off + 32
		scStart := off + 32 + 16

		sc := [4]uint16{
			uint16(data[scStart]) | uint16(data[scStart+1])<<8,
			uint16(data[scStart+2]) | uint16(data[scStart+3])<<8,
			uint16(data[scStart+4]) | uint16(data[scStart+5])<<8,
			uint16(data[scStart+6]) | uint16(data[scStart+7])<<8,
		}

		scaleU16 := (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
		d := float16ToFloat32(scaleU16)

		outBase := block * BlockSizeIQ1M

		qsOff := qsStart
		qhOff := qhStart

		for ib := 0; ib < 8; ib++ {
			scIdx := ib / 2
			scShift := 6 * (ib % 2)
			dl1 := d * float32(2*int((sc[scIdx]>>uint(scShift))&0x7)+1)
			dl2 := d * float32(2*int((sc[scIdx]>>uint(scShift+3))&0x7)+1)

			qh0 := data[qhOff]
			qh1 := data[qhOff+1]

			var idx [4]uint16
			var delta [4]float32

			idx[0] = uint16(data[qsOff]) | (uint16(qh0)<<8)&0x700
			idx[1] = uint16(data[qsOff+1]) | (uint16(qh0)<<4)&0x700
			idx[2] = uint16(data[qsOff+2]) | (uint16(qh1)<<8)&0x700
			idx[3] = uint16(data[qsOff+3]) | (uint16(qh1)<<4)&0x700

			if qh0&0x08 != 0 {
				delta[0] = -iq1mDelta
			} else {
				delta[0] = iq1mDelta
			}
			if qh0&0x80 != 0 {
				delta[1] = -iq1mDelta
			} else {
				delta[1] = iq1mDelta
			}
			if qh1&0x08 != 0 {
				delta[2] = -iq1mDelta
			} else {
				delta[2] = iq1mDelta
			}
			if qh1&0x80 != 0 {
				delta[3] = -iq1mDelta
			} else {
				delta[3] = iq1mDelta
			}

			for l := 0; l < 2; l++ {
				grid := iq1s_grid[idx[l]]
				oIdx := outBase + ib*32 + l*8
				for j := 0; j < 8; j++ {
					gridByte := uint8(grid >> (8 * uint(j)))
					gridVal := float32(int8(gridByte))
					dst[oIdx+j] = dl1 * (gridVal + delta[l])
				}
			}
			for l := 2; l < 4; l++ {
				grid := iq1s_grid[idx[l]]
				oIdx := outBase + ib*32 + l*8
				for j := 0; j < 8; j++ {
					gridByte := uint8(grid >> (8 * uint(j)))
					gridVal := float32(int8(gridByte))
					dst[oIdx+j] = dl2 * (gridVal + delta[l])
				}
			}

			qsOff += 4
			qhOff += 2
		}
	}
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

func dequantizeTQ1_0Into(dst []float32, data []byte, n int) {
	numBlocks := n / BlockSizeTQ1_0
	pow3 := [6]uint8{1, 3, 9, 27, 81, 243}
	outOff := 0

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesTQ1_0

		dBits := uint16(data[off+52]) | uint16(data[off+53])<<8
		d := float16ToFloat32(dBits)

		qsLen := 48
		qsChunk32 := qsLen - (qsLen % 32)
		for j := 0; j < qsChunk32; j += 32 {
			for nn := 0; nn < 5; nn++ {
				for m := 0; m < 32; m++ {
					q := data[off+j+m]
					q = uint8((uint16(q) * uint16(pow3[nn])) >> 0)
					xi := int16((uint16(q) * 3) >> 8)
					dst[outOff] = float32(xi-1) * d
					outOff++
				}
			}
		}
		for j := qsChunk32; j < qsLen; j += 16 {
			for nn := 0; nn < 5; nn++ {
				for m := 0; m < 16; m++ {
					q := data[off+j+m]
					q = uint8((uint16(q) * uint16(pow3[nn])) >> 0)
					xi := int16((uint16(q) * 3) >> 8)
					dst[outOff] = float32(xi-1) * d
					outOff++
				}
			}
		}

		qhOff := off + 48
		for nn := 0; nn < 4; nn++ {
			for j := 0; j < 4; j++ {
				q := data[qhOff+j]
				q = uint8((uint16(q) * uint16(pow3[nn])) >> 0)
				xi := int16((uint16(q) * 3) >> 8)
				dst[outOff] = float32(xi-1) * d
				outOff++
			}
		}
	}
}

func dequantizeTQ2_0Into(dst []float32, data []byte, n int) {
	numBlocks := n / BlockSizeTQ2_0
	outOff := 0

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesTQ2_0

		dBits := uint16(data[off+64]) | uint16(data[off+65])<<8
		d := float16ToFloat32(dBits)

		for j := 0; j < 64; j += 32 {
			for l := 0; l < 4; l++ {
				shift := uint(l * 2)
				for m := 0; m < 32; m++ {
					q := int8((data[off+j+m] >> shift) & 3)
					dst[outOff] = float32(q-1) * d
					outOff++
				}
			}
		}
	}
}

func dequantizeMXFP4Into(dst []float32, data []byte, n int) {
	numBlocks := n / BlockSizeMXFP4
	outOff := 0

	for block := 0; block < numBlocks; block++ {
		off := block * BlockBytesMXFP4
		d := e8m0ToFloat32Half(data[off])

		for j := 0; j < 16; j++ {
			qByte := data[off+1+j]
			dst[outOff+j] = d * float32(kvalues_mxfp4[qByte&0xf])
			dst[outOff+j+16] = d * float32(kvalues_mxfp4[qByte>>4])
		}
		outOff += 32
	}
}
