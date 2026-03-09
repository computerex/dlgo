package main

import (
	"fmt"
	"os"
	"sort"

	"github.com/computerex/dlgo/format/gguf"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: gguf_info <model.gguf>\n")
		os.Exit(1)
	}
	model, err := gguf.Open(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "open: %v\n", err)
		os.Exit(1)
	}

	typeCounts := map[uint32]int{}
	for _, t := range model.Tensors {
		typeCounts[uint32(t.Type)]++
	}

	type kv struct {
		t uint32
		n int
	}
	var sorted []kv
	for t, n := range typeCounts {
		sorted = append(sorted, kv{t, n})
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].t < sorted[j].t })

	typeNames := map[uint32]string{
		0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
		8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
		13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
		18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
		23: "IQ4_XS", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
	}

	fmt.Printf("Tensors: %d total\n", len(model.Tensors))
	fmt.Printf("Quant types:\n")
	for _, kv := range sorted {
		name := typeNames[kv.t]
		if name == "" {
			name = fmt.Sprintf("type_%d", kv.t)
		}
		fmt.Printf("  %-10s (id=%2d): %d tensors\n", name, kv.t, kv.n)
	}

	if len(os.Args) > 2 && os.Args[2] == "-names" {
		fmt.Printf("\nTensor names:\n")
		for _, t := range model.Tensors {
			fmt.Printf("  %s  type=%d  dims=%v\n", t.Name, t.Type, t.Dimensions)
		}
	}
	if len(os.Args) > 2 && os.Args[2] == "-meta" {
		fmt.Printf("\nMetadata:\n")
		for k, v := range model.Metadata {
			fmt.Printf("  %s = %v\n", k, v)
		}
	}
}
