//go:build cgo && vulkan

package gpu

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/computerex/dlgo/quant"
)

// packF32ToF16 converts a float32 to IEEE-754 binary16 (round-to-nearest-even).
func packF32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xFF)
	mant := bits & 0x7FFFFF

	if exp == 0xFF { // Inf/NaN
		return sign | 0x7C00 | uint16(mant>>13)
	}
	e := exp - 127 + 15
	if e >= 0x1F {
		return sign | 0x7C00 // overflow -> Inf
	}
	if e <= 0 {
		if e < -10 {
			return sign // underflow -> 0
		}
		// subnormal
		mant |= 0x800000
		shift := uint(14 - e)
		m := mant >> shift
		// round to nearest even
		if (mant>>(shift-1))&1 == 1 && ((mant&((1<<(shift-1))-1)) != 0 || m&1 == 1) {
			m++
		}
		return sign | uint16(m)
	}
	// normal
	m := mant >> 13
	if mant&0x1000 != 0 && ((mant&0xFFF) != 0 || m&1 == 1) {
		m++
		if m == 0x400 {
			m = 0
			e++
			if e >= 0x1F {
				return sign | 0x7C00
			}
		}
	}
	return sign | uint16(e<<10) | uint16(m)
}

// packQ3KRow packs `cols` float32 values (multiple of 256) into Q3_K super-blocks
// using a simple, self-consistent encoding (not an optimal quantizer; just a
// valid bit layout so the GPU dequant can be checked against quant.DequantizeQ3_K).
//
// Encoding per 256-element super-block:
//   - choose d so the max |value| maps into the 3-bit range [-4..3]
//   - per 16-element group pick a 6-bit scale (biased by 32)
//   - 3-bit quant q = round(value / (d*(scale-32))) clamped to [-4,3]
//   - store low 2 bits in qs, high bit in hmask (hmask bit set => q>=0)
func packQ3KRow(vals []float32) []byte {
	const blockSize = 256
	const blockBytes = 110
	nBlocks := len(vals) / blockSize
	out := make([]byte, nBlocks*blockBytes)

	for b := 0; b < nBlocks; b++ {
		off := b * blockBytes
		hm := out[off : off+32]
		qs := out[off+32 : off+96]
		sc := out[off+96 : off+108]

		block := vals[b*blockSize : (b+1)*blockSize]

		// Scale selection: use a global super-block scale d = max|v| / 32 so
		// that with a per-group scale of ~32 (i.e. scale-32 = 0 -> degenerate).
		// Instead give each group a fixed usable scale factor of 8 (scale=40),
		// and set d so quant values land in range.
		var maxAbs float32
		for _, v := range block {
			if a := float32(math.Abs(float64(v))); a > maxAbs {
				maxAbs = a
			}
		}
		// group scale delta = 8 (6-bit value 40 -> 40-32 = 8)
		const scaleDelta = 8.0
		var d float32
		if maxAbs > 0 {
			d = maxAbs / (scaleDelta * 4.0) // max quant magnitude 4
		} else {
			d = 1.0
		}

		// Write d as f16 at offset 108.
		d16 := packF32ToF16(d)
		out[off+108] = byte(d16)
		out[off+109] = byte(d16 >> 8)

		// All 16 scales = 40 (=> scale-32 = 8).
		// Pack 16 six-bit values of 40 into the 12-byte layout used by
		// quant/q3_k.go. We reverse that packing.
		packQ3KScales(sc, 40)

		// Quantize each element.
		for i := 0; i < blockSize; i++ {
			dl := d * scaleDelta
			q := 0
			if dl != 0 {
				q = int(math.Round(float64(block[i] / dl)))
			}
			if q > 3 {
				q = 3
			}
			if q < -4 {
				q = -4
			}
			// 3-bit value: low 2 bits -> qs, high bit -> hmask.
			// r3 = q + 4  in [0..7]; low2 = r3 & 3; hi = r3>>2.
			r3 := q + 4
			low2 := r3 & 3
			hi := (r3 >> 2) & 1

			// qs byte/field mapping (inverse of shader):
			//   byte = (i/128)*32 + (i%32); field = (i/32)%4.
			qi := (i/128)*32 + (i % 32)
			shift := ((i / 32) % 4) * 2
			qs[qi] |= byte(low2 << uint(shift))

			// hmask byte = i%32; bit = i/32.
			if hi != 0 {
				hm[i%32] |= byte(1 << uint(i/32))
			}
		}
	}
	return out
}

// packQ3KScales writes 16 six-bit scale values into the 12-byte `scales` array
// using the reverse of quant/q3_k.go's unpacking. All scales equal here, so this
// is simple, but the bit packing must be exact enough for DequantizeQ3_K.
func packQ3KScales(sc []byte, value byte) {
	for i := range sc {
		sc[i] = 0
	}
	// From q3_k.go:
	//   aux[2] = ((a0>>4)&kmask2) | (((a2>>4)&kmask1)<<4)
	//   aux[3] = ((a1>>4)&kmask2) | (((a2>>6)&kmask1)<<4)
	//   aux[0] = (a0&kmask2)      | (((a2>>0)&kmask1)<<4)
	//   aux[1] = (a1&kmask2)      | (((a2>>2)&kmask1)<<4)
	// where a0..a2 are the three little-endian u32 words of sc[0..11].
	// scales[i] = (aux[i/4] >> ((i%4)*8)) & 0xFF.
	//
	// We want scales[i] = value (6-bit, <64) for all i.
	// Set the low nibble of each aux byte to (value & 0x0F) and the high 2 bits
	// to (value >> 4). Work in u32 words directly.
	v := uint32(value)
	loNib := v & 0x0F
	hi2 := (v >> 4) & 0x03

	// Every nibble of a0 and a1 must equal loNib (both aux low and high nibbles
	// draw their low 4 bits from a0/a1 across the four aux words).
	byteVal := loNib | (loNib << 4)
	lowWord := byteVal | (byteVal << 8) | (byteVal << 16) | (byteVal << 24)

	// a2 contributes the top 2 bits of each 6-bit scale; all four 2-bit fields
	// per byte must equal hi2 (used at shifts 0,2,4,6 by aux[0..3]).
	a2 := hi2 | (hi2 << 2) | (hi2 << 4) | (hi2 << 6)

	a0 := lowWord
	a1 := lowWord

	words := [3]uint32{a0, a1, a2}
	for i := 0; i < 12; i++ {
		sc[i] = byte(words[i/4] >> uint((i%4)*8))
	}
}

// TestGemmA3KTiledGPU validates the opt-in Q3_K tiled GEMM against a CPU
// reference that dequantizes the same Q3_K bytes (quant.DequantizeQ3_K).
func TestGemmA3KTiledGPU(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer Shutdown()

	// Force the opt-in path for this test's lifetime.
	SetGemmQ3K(true)
	defer SetGemmQ3K(false)

	const rows = 128
	const cols = 256
	const npos = 16

	rng := rand.New(rand.NewSource(20261008))

	// Build random weights (per row: multiple of 256 cols).
	wF32 := make([]float32, rows*cols)
	for i := range wF32 {
		wF32[i] = (rng.Float32()*2 - 1) * 4
	}

	// Pack all rows into one Q3_K buffer (row-major, cols/256 blocks per row).
	q3k := make([]byte, 0, rows*(cols/256)*110)
	for r := 0; r < rows; r++ {
		q3k = append(q3k, packQ3KRow(wF32[r*cols:(r+1)*cols])...)
	}

	// CPU reference: dequantize the same bytes and matmul.
	wDeq := quant.DequantizeQ3_K(q3k, rows*cols)
	x := make([]float32, npos*cols)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	cpuOut := make([]float32, npos*rows)
	for p := 0; p < npos; p++ {
		for r := 0; r < rows; r++ {
			var sum float64
			for c := 0; c < cols; c++ {
				sum += float64(wDeq[r*cols+c]) * float64(x[p*cols+c])
			}
			cpuOut[p*rows+r] = float32(sum)
		}
	}

	// Upload and run.
	wBuf := Alloc(uint64(len(q3k)))
	xBuf := Alloc(uint64(len(x)) * 4)
	outBuf := Alloc(uint64(npos*rows) * 4)
	if wBuf == 0 || xBuf == 0 || outBuf == 0 {
		t.Fatal("alloc failed")
	}
	defer Free(wBuf)
	defer Free(xBuf)
	defer Free(outBuf)

	if err := Upload(wBuf, q3k); err != nil {
		t.Fatalf("upload weights: %v", err)
	}
	if err := UploadF32(xBuf, x); err != nil {
		t.Fatalf("upload x: %v", err)
	}

	BeginBatch()
	if err := BatchMatVec(outBuf, wBuf, xBuf, rows, cols, npos, 11); err != nil { // 11 = QTYPE_Q3_K
		EndBatch()
		t.Fatalf("BatchMatVec: %v", err)
	}
	EndBatch()

	gpuOut := make([]float32, npos*rows)
	if err := DownloadF32(outBuf, gpuOut); err != nil {
		t.Fatalf("download: %v", err)
	}

	var maxErr, maxRef float64
	for i := range cpuOut {
		diff := math.Abs(float64(gpuOut[i]) - float64(cpuOut[i]))
		if diff > maxErr {
			maxErr = diff
		}
		if a := math.Abs(float64(cpuOut[i])); a > maxRef {
			maxRef = a
		}
	}
	relErr := maxErr
	if maxRef > 0 {
		relErr = maxErr / maxRef
	}
	t.Logf("Q3_K tiled GEMM: maxAbsErr=%.6f maxRef=%.4f relErr=%.3e", maxErr, maxRef, relErr)

	if relErr > 1e-4 {
		t.Errorf("Q3_K GEMM relative error too high: %.3e (want < 1e-4)", relErr)
		for i := 0; i < len(cpuOut) && i < 6; i++ {
			t.Logf("  [%d] CPU=%.6f GPU=%.6f", i, cpuOut[i], gpuOut[i])
		}
	}
}

// TestGemmA3KBatchTimingGPU compares the per-token matvec fallback against the
// opt-in Q3_K tiled GEMM on realistic projection shapes (FFN gate/up:
// 4096 x 12288) at several token counts. This isolates K1's speedup on the
// kernel from the rest of the model.
func TestGemmA3KBatchTimingGPU(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer Shutdown()

	const rows = 12288 // ffn_gate/up output dim
	const cols = 4096  // hidden dim (multiple of 256)

	rng := rand.New(rand.NewSource(7))
	wF32 := make([]float32, rows*cols)
	for i := range wF32 {
		wF32[i] = (rng.Float32()*2 - 1) * 4
	}
	q3k := make([]byte, 0, rows*(cols/256)*110)
	for r := 0; r < rows; r++ {
		q3k = append(q3k, packQ3KRow(wF32[r*cols:(r+1)*cols])...)
	}

	wBuf := Alloc(uint64(len(q3k)))
	if wBuf == 0 {
		t.Fatal("alloc weights failed")
	}
	defer Free(wBuf)
	if err := Upload(wBuf, q3k); err != nil {
		t.Fatalf("upload: %v", err)
	}
	outBuf := Alloc(uint64(rows) * 4 * 600)
	if outBuf == 0 {
		t.Fatal("alloc out failed")
	}
	defer Free(outBuf)
	xBuf := Alloc(uint64(cols) * 4 * 600)
	if xBuf == 0 {
		t.Fatal("alloc x failed")
	}
	defer Free(xBuf)
	x := make([]float32, cols*600)

	runOnce := func(npos int) {
		if err := UploadF32(xBuf, x[:npos*cols]); err != nil {
			t.Fatalf("upload x: %v", err)
		}
		BeginBatch()
		if err := BatchMatVec(outBuf, wBuf, xBuf, rows, cols, npos, 11); err != nil {
			EndBatch()
			t.Fatalf("BatchMatVec: %v", err)
		}
		EndBatch()
		Sync()
	}

	bench := func(npos int, reps int) float64 {
		runOnce(npos) // warmup
		start := time.Now()
		for i := 0; i < reps; i++ {
			runOnce(npos)
		}
		return float64(time.Since(start).Microseconds()) / 1000.0 / float64(reps)
	}

	for _, npos := range []int{64, 128, 256, 512} {
		SetGemmQ3K(false)
		offMs := bench(npos, 3)
		SetGemmQ3K(true)
		onMs := bench(npos, 3)
		SetGemmQ3K(false)
		t.Logf("npos=%4d  matvec=%.3f ms  gemm=%.3f ms  speedup=%.2fx", npos, offMs, onMs, offMs/onMs)
	}
}
