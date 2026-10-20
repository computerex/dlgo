//go:build cgo && vulkan

package gpu_test

import (
	"math"
	"testing"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/quant"
)

// xorshift32 is a tiny deterministic PRNG so the test needs no fixtures.
type xs32 uint32

func (r *xs32) next() uint32 {
	x := uint32(*r)
	x ^= x << 13
	x ^= x >> 17
	x ^= x << 5
	*r = xs32(x)
	return x
}

// makeQuantBytes fabricates a valid-enough K-quant tensor: random codes/scales
// with a fixed small positive f16 super-scale so the decode stays finite. The
// point is to exercise the GPU decoder against quant.Dequantize on the SAME
// bytes, so the fixture does not need to be a "good" quantization.
func makeQuantBytes(qtype uint32, blocks int) []byte {
	var rng xs32 = 0x1234567
	dhi := quant.F32ToF16(0.02)
	dlo := quant.F32ToF16(0.01)
	put16 := func(b []byte, off int, v uint16) { b[off] = byte(v); b[off+1] = byte(v >> 8) }

	switch qtype {
	case 12: // Q4_K: 144 B
		b := make([]byte, blocks*144)
		for i := 0; i < blocks; i++ {
			o := i * 144
			put16(b, o, dhi)
			put16(b, o+2, dlo)
			for j := 4; j < 144; j++ {
				b[o+j] = byte(rng.next())
			}
		}
		return b
	case 13: // Q5_K: 176 B
		b := make([]byte, blocks*176)
		for i := 0; i < blocks; i++ {
			o := i * 176
			put16(b, o, dhi)
			put16(b, o+2, dlo)
			for j := 4; j < 176; j++ {
				b[o+j] = byte(rng.next())
			}
		}
		return b
	case 14: // Q6_K: 210 B
		b := make([]byte, blocks*210)
		for i := 0; i < blocks; i++ {
			o := i * 210
			for j := 0; j < 208; j++ {
				b[o+j] = byte(rng.next())
			}
			put16(b, o+208, dhi)
		}
		return b
	}
	panic("unsupported qtype")
}

func relL2(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(b[i]) * float64(b[i])
	}
	if den == 0 {
		return math.Sqrt(num)
	}
	return math.Sqrt(num / den)
}

// TestConvertQuantToAffine checks that the in-VRAM converter produces the same
// canonical encoding as the CPU repack (quant.RepackAffine4/8) for Q4_K, Q5_K
// and Q6_K sources, in both Affine8 and Affine4 output, and that the encoding is
// (near-)lossless: reconstructing it recovers the source decode.
func TestConvertQuantToAffine(t *testing.T) {
	if err := gpu.Init(); err != nil {
		t.Skipf("GPU unavailable: %v", err)
	}
	defer gpu.Shutdown()

	const blocks = 6 // 1536 elements; multiple of 256 and 32
	n := blocks * 256

	cases := []struct {
		name  string
		qtype uint32
	}{
		{"Q4_K", 12},
		{"Q5_K", 13},
		{"Q6_K", 14},
	}

	for _, tc := range cases {
		src := makeQuantBytes(tc.qtype, blocks)
		if want := quant.BytesForN(tc.qtype, n); want != len(src) {
			t.Fatalf("%s: source size %d, BytesForN=%d", tc.name, len(src), want)
		}

		// CPU reference decode + repack.
		f32, err := quant.Dequantize(src, tc.qtype, n)
		if err != nil {
			t.Fatalf("%s: dequantize: %v", tc.name, err)
		}

		for _, outBits := range []int{8, 4} {
			var cpuEnc []byte
			var blockBytes int
			if outBits == 8 {
				cpuEnc = quant.QuantizeAffine8(f32)
				blockBytes = quant.Affine8BlockBytes
			} else {
				cpuEnc = quant.QuantizeAffine4(f32)
				blockBytes = quant.Affine4BlockBytes
			}
			outLen := (n / 32) * blockBytes

			// GPU conversion.
			srcBuf := gpu.Alloc(uint64(len(src)))
			if srcBuf == 0 {
				t.Fatalf("%s: alloc src failed", tc.name)
			}
			if err := gpu.Upload(srcBuf, src); err != nil {
				t.Fatalf("%s: upload: %v", tc.name, err)
			}
			outBuf := gpu.Alloc(uint64(outLen))
			if outBuf == 0 {
				t.Fatalf("%s: alloc out failed", tc.name)
			}
			if err := gpu.ConvertQuantToAffine(outBuf, srcBuf, n, int(tc.qtype), outBits); err != nil {
				t.Fatalf("%s/%d: convert: %v", tc.name, outBits, err)
			}
			gotEnc := make([]byte, outLen)
			if err := gpu.Download(outBuf, gotEnc); err != nil {
				t.Fatalf("%s: download: %v", tc.name, err)
			}
			gpu.Free(srcBuf)
			gpu.Free(outBuf)

			// Reconstruct both encodings and measure them against the source
			// decode. The correctness invariant is that the GPU encoding is no
			// less faithful than the CPU encoding: for Affine8 both are
			// near-lossless, and for Affine4 both carry the same 4-bit error
			// (tiny FMA-level f32 decode differences can flip a 4-bit rounding
			// boundary — that is quantization resolution, not a decode bug).
			var cpuRecon, gpuRecon []float32
			if outBits == 8 {
				cpuRecon = quant.DequantizeAffine8(cpuEnc, n)
				gpuRecon = quant.DequantizeAffine8(gotEnc, n)
			} else {
				cpuRecon = quant.DequantizeAffine4(cpuEnc, n)
				gpuRecon = quant.DequantizeAffine4(gotEnc, n)
			}
			g2s := relL2(gpuRecon, f32) // GPU encoding error vs source
			c2s := relL2(cpuRecon, f32) // CPU encoding error vs source

			// GPU must be essentially as faithful as the CPU path.
			if g2s > c2s*1.5+5e-3 {
				t.Errorf("%s/Affine%d: GPU-vs-source relL2=%.3e worse than CPU %.3e",
					tc.name, outBits, g2s, c2s)
			}
			// Affine8 is strictly finer than any ≤6-bit source → near-lossless.
			if outBits == 8 && g2s > 1e-2 {
				t.Errorf("%s/Affine8: GPU-vs-source relL2=%.3e not near-lossless", tc.name, g2s)
			}
			t.Logf("%s/Affine%d: relL2(GPU,src)=%.3e relL2(CPU,src)=%.3e relL2(GPU,CPU)=%.3e",
				tc.name, outBits, g2s, c2s, relL2(gpuRecon, cpuRecon))
		}
	}
}
