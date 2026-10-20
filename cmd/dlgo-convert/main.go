// dlgo-convert rewrites a GGUF model's quantized weight tensors into the
// canonical Affine8 format (GGML type 60), producing a model file that loads
// without any per-run repack.
//
// Why: dlgo's GPU kernels run every weight through the single unified Affine8
// GEMM. Historically the Q4_K/Q6_K/... tensors were repacked to Affine8 in main
// memory at every load (~40 s serial, ~12 s parallel for the 6 GB DiT). Doing
// that conversion once, up front, means the runtime only ever sees Affine8 —
// which is exactly what lets the per-quant GEMM/matvec kernels be deleted (the
// model no longer contains the formats they implement), and makes each load a
// pure upload of ~6 GB instead of a repack.
//
// The per-quant *decode* stays in Go (quant.RepackAffine8) — one dispatch table,
// load-time only — rather than being duplicated as GLSL dequant shaders.
//
// Usage:
//
//	dlgo-convert -i z-image-turbo-Q4_K_M.gguf -o z-image-turbo-affine8.gguf
//
// The output is a valid GGUF v3 file with the same metadata, tensor names, shapes
// and order; only the data of the converted tensors changes (their type becomes
// 60). Element-wise tensors (F32/F16/BF16) are copied verbatim so norm/RMSNorm
// weights keep their exact layout. Load the result with any dlgo tool (the DiT
// loader raw-uploads type-60 tensors into the unified Affine8 kernel).
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/computerex/dlgo/format/gguf"
	"github.com/computerex/dlgo/mmap"
	"github.com/computerex/dlgo/quant"
)

// affinizeTypes are the block-quantized formats RepackAffine8 can convert. F32 (0),
// F16 (1) and BF16 (30) are left untouched: they are element-wise and are the
// formats the loader reads as raw float32 (norms/biases), so their layout must
// not change.
var affinizeTypes = map[uint32]bool{
	2: true, 3: true, 6: true, 7: true, 8: true, 9: true, // Q4_0..Q8_1
	10: true, 11: true, 12: true, 13: true, 14: true, 15: true, // Q2_K..Q8_K
	16: true, 17: true, 18: true, 19: true, 20: true, 21: true, 22: true, 23: true, // IQ
	29: true, 34: true, 35: true, 39: true, // IQ1_M, TQ, MXFP4
}

// useAffine4 mirrors gpu.canonicalQuant4: ≤5-bit sources become the byte-lean
// Affine4 (0.625 B/weight, within 11% of Q4_K); ≥5.5-bit sources (Q5_K/Q6_K/
// Q8_*) keep the finer Affine8 (1.125 B/weight). This keeps the converted file
// close to the source size while collapsing every format onto two kernels.
func useAffine4(t uint32) bool {
	switch t {
	case 6, 7, 8, 9, 13, 14, 15: // Q5_0, Q5_1, Q8_0, Q8_1, Q5_K, Q6_K, Q8_K
		return false
	}
	return true
}

func main() {
	in := flag.String("i", "", "input GGUF file")
	out := flag.String("o", "", "output GGUF file (canonical Affine8)")
	align := flag.Uint("align", 32, "GGUF tensor-data alignment")
	flag.Parse()
	if *in == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "Usage: dlgo-convert -i <input.gguf> -o <output.gguf>")
		flag.PrintDefaults()
		os.Exit(1)
	}

	if err := run(*in, *out, uint32(*align)); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func run(in, out string, align uint32) error {
	start := time.Now()
	gf, err := gguf.Open(in)
	if err != nil {
		return fmt.Errorf("open %s: %w", in, err)
	}
	mf, err := mmap.Open(in)
	if err != nil {
		return fmt.Errorf("mmap %s: %w", in, err)
	}
	defer mf.Close()

	fmt.Printf("Read %s: %d tensors, %d metadata keys\n", in, len(gf.Tensors), len(gf.Metadata))

	writeTensors := make([]gguf.WriteTensor, len(gf.Tensors))
	converted, copied := 0, 0
	var srcBytes, dstBytes int64

	for i, t := range gf.Tensors {
		numel := int64(1)
		for _, d := range t.Dimensions {
			numel *= d
		}
		srcSize := quant.BytesForN(uint32(t.Type), int(numel))
		off := gf.DataOffset + int64(t.Offset)
		if off < 0 || off+int64(srcSize) > int64(len(mf.Data)) {
			return fmt.Errorf("tensor %q: data range [%d,%d) out of file", t.Name, off, off+int64(srcSize))
		}
		src := mf.Data[off : off+int64(srcSize)]
		srcBytes += int64(srcSize)

		otype := uint32(t.Type)
		var data []byte
		if affinizeTypes[uint32(t.Type)] && numel%int64(quant.Affine8GroupSize) == 0 {
			var enc []byte
			var rerr error
			if useAffine4(uint32(t.Type)) {
				// Byte-lean canonical for ≤5-bit sources (0.625 B/weight).
				enc, rerr = quant.RepackAffine4(src, uint32(t.Type), int(numel))
				otype = 61
			} else {
				// ≥5.5-bit sources (Q5_K/Q6_K/Q8_K) get Affine8.
				enc, rerr = quant.RepackAffine8(src, uint32(t.Type), int(numel))
				otype = 60
			}
			if rerr != nil {
				return fmt.Errorf("tensor %q (type %d): repack: %w", t.Name, uint32(t.Type), rerr)
			}
			data = enc
			converted++
		} else {
			// Copy verbatim (element-wise types keep their exact layout).
			data = make([]byte, len(src))
			copy(data, src)
			copied++
		}
		dstBytes += int64(len(data))

		writeTensors[i] = gguf.WriteTensor{
			Name:       t.Name,
			Dimensions: t.Dimensions,
			Type:       gguf.GGMLType(otype),
			Data:       data,
		}
		if (i+1)%64 == 0 {
			fmt.Printf("  converted %d/%d tensors...\n", i+1, len(gf.Tensors))
		}
	}

	fmt.Printf("Repacked %d tensors to canonical Affine4/Affine8, copied %d unchanged (%.1f MB -> %.1f MB)\n",
		converted, copied, float64(srcBytes)/(1<<20), float64(dstBytes)/(1<<20))

	if err := gguf.WriteGGUF(out, gf.Metadata, writeTensors, align); err != nil {
		return fmt.Errorf("write %s: %w", out, err)
	}

	// Count the worker threads used, for the log reader's benefit.
	fmt.Printf("Wrote %s in %v (%d cores)\n", out, time.Since(start), runtime.GOMAXPROCS(0))
	return nil
}
