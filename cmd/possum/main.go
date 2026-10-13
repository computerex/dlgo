package main

// possum: print sum of v.position_embd.weight (all elements) and per-row sums sanity.

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/format/gguf"
	"github.com/computerex/dlgo/mmap"
	"github.com/computerex/dlgo/quant"
)

func main() {
	path := os.Args[1]
	gf, err := gguf.Open(path)
	if err != nil {
		panic(err)
	}
	mf, err := mmap.Open(path)
	if err != nil {
		panic(err)
	}
	for _, ti := range gf.Tensors {
		if ti.Name != "v.position_embd.weight" {
			continue
		}
		total := int64(1)
		for _, d := range ti.Dimensions {
			total *= d
		}
		data := mf.Slice(gf.DataOffset+int64(ti.Offset), int64(quant.BytesForN(uint32(ti.Type), int(total))))
		out, err := quant.Dequantize(data, uint32(ti.Type), int(total))
		if err != nil {
			panic(err)
		}
		var sum float64
		for _, v := range out {
			sum += float64(v)
		}
		fmt.Printf("pos table dims=%v type=%d total=%d sum=%.6f mean=%.6f\n",
			ti.Dimensions, ti.Type, total, sum, sum/float64(total))
		// rows: assume [embd, npos] ne-order -> row p = out[p*embd:(p+1)*embd]
		embd := int(ti.Dimensions[0])
		npos := int(ti.Dimensions[1])
		var s0, s1, s48, s49 float64
		for _, v := range out[0:embd] {
			s0 += float64(v)
		}
		for _, v := range out[embd : 2*embd] {
			s1 += float64(v)
		}
		for _, v := range out[48*embd : 49*embd] {
			s48 += float64(v)
		}
		for _, v := range out[49*embd : 50*embd] {
			s49 += float64(v)
		}
		fmt.Printf("rowsums: r0=%.4f r1=%.4f r48=%.4f r49=%.4f (embd=%d npos=%d)\n", s0, s1, s48, s49, embd, npos)
		// patch bias
		for _, ti2 := range gf.Tensors {
			if ti2.Name == "v.patch_embd.bias" {
				t2 := int64(1)
				for _, d := range ti2.Dimensions {
					t2 *= d
				}
				d2 := mf.Slice(gf.DataOffset+int64(ti2.Offset), int64(quant.BytesForN(uint32(ti2.Type), int(t2))))
				o2, err := quant.Dequantize(d2, uint32(ti2.Type), int(t2))
				if err != nil {
					panic(err)
				}
				var sb float64
				for _, v := range o2 {
					sb += float64(v)
				}
				fmt.Printf("patch_bias sum=%.6f (x2304=%.4f)\n", sb, sb*2304)
			}
		}
	}
}
