package quant

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"
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

// ggmlBlockElems returns the number of elements per source quantization block
// for a GGML type (1 for element-wise F32/F16/BF16, 32 for the Q4_0-family,
// 256 for the K-quants / IQ / TQ schemes). Used to keep parallel repack chunk
// boundaries aligned to whole source blocks.
func ggmlBlockElems(ggmlType uint32) int {
	switch ggmlType {
	case 0, 1, 30: // F32, F16, BF16
		return 1
	case 2, 3, 6, 7, 8, 9, 20, 39: // Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/IQ4_NL/MXFP4
		return 32
	default: // K-quants, I-quants, TQ: 256-element super-blocks
		return 256
	}
}

// repackChunkElems is the granularity at which RepackAffine8 splits a tensor for
// parallel conversion. It must be a multiple of every source block size (1, 32,
// 256) and of Affine8GroupSize (32); 256 satisfies all of them.
const repackChunkElems = 256

// RepackAffine8 converts a quantized tensor of the given GGML type into the
// canonical Affine8 format. n is the element count (must be a multiple of the
// source block size and of 32).
//
// The conversion — dequantize to float32 then requantize — is the dominant cost
// of loading a quantized model onto the GPU (~40 s for a 6 GB DiT, single
// threaded, compute-bound). It is therefore split into block-aligned chunks and
// run across all cores. Each chunk writes a disjoint, group-aligned byte range
// of the output, so no locking is needed; the result is byte-identical to the
// serial path (tested by TestRepackAffine8Parallel).
func RepackAffine8(data []byte, ggmlType uint32, n int) ([]byte, error) {
	if n%Affine8GroupSize != 0 {
		return nil, fmt.Errorf("affine8: element count %d not a multiple of %d", n, Affine8GroupSize)
	}

	// Serial path for small tensors (goroutine overhead dominates), a single CPU,
	// an unknown type, or a tensors whose element count is not a whole number of
	// source blocks (RepackAffine8 requires n%32==0; a 256-block type additionally
	// needs n%256==0 for the chunk byte offsets to land on block boundaries).
	workers := runtime.GOMAXPROCS(0)
	be := ggmlBlockElems(ggmlType)
	const minParallelElems = 1 << 20 // 1M elements (~<=4 MB); below this, serial wins
	if workers <= 1 || n < minParallelElems || BytesForN(ggmlType, repackChunkElems) == 0 || n%be != 0 {
		f32, err := Dequantize(data, ggmlType, n)
		if err != nil {
			return nil, err
		}
		return QuantizeAffine8(f32), nil
	}

	out := make([]byte, (n/Affine8GroupSize)*Affine8BlockBytes)

	// Aim for a few chunks per worker so the biggest tensors load-balance well.
	chunk := (n + workers*4 - 1) / (workers * 4)
	chunk = (chunk + repackChunkElems - 1) / repackChunkElems * repackChunkElems
	if chunk < repackChunkElems {
		chunk = repackChunkElems
	}

	type job struct{ start, count int }
	jobs := make([]job, 0, n/chunk+1)
	for e := 0; e < n; e += chunk {
		c := chunk
		if e+c > n {
			c = n - e
		}
		jobs = append(jobs, job{e, c})
	}

	var wg sync.WaitGroup
	sem := make(chan struct{}, workers)
	errs := make([]error, len(jobs))
	for i := range jobs {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, j job) {
			defer wg.Done()
			defer func() { <-sem }()
			byteStart := BytesForN(ggmlType, j.start)
			byteEnd := BytesForN(ggmlType, j.start+j.count)
			if byteEnd <= byteStart || byteEnd > len(data) {
				errs[i] = fmt.Errorf("affine8: chunk [%d,%d) out of data range (%d bytes)", j.start, j.start+j.count, len(data))
				return
			}
			f32, err := Dequantize(data[byteStart:byteEnd], ggmlType, j.count)
			if err != nil {
				errs[i] = err
				return
			}
			outOff := (j.start / Affine8GroupSize) * Affine8BlockBytes
			quantizeAffine8Into(out[outOff:], f32)
		}(i, jobs[i])
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// QuantizeAffine8 encodes a float32 slice into the canonical Affine8 format.
// len(x) must be a multiple of Affine8GroupSize.
func QuantizeAffine8(x []float32) []byte {
	ng := len(x) / Affine8GroupSize
	out := make([]byte, ng*Affine8BlockBytes)
	quantizeAffine8Into(out, x)
	return out
}

// quantizeAffine8Into writes the Affine8 encoding of x into out, which must be
// exactly len(x)/Affine8GroupSize*Affine8BlockBytes bytes. Split out so the
// parallel repack can write each chunk straight into its slice of the output.
func quantizeAffine8Into(out []byte, x []float32) {
	ng := len(x) / Affine8GroupSize
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
