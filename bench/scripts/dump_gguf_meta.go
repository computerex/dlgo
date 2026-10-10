//go:build ignore

package main

import (
	"fmt"
	"os"
	"sort"

	"github.com/computerex/dlgo/format/gguf"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: dump_gguf_meta <path.gguf>")
		os.Exit(1)
	}
	gf, err := gguf.Open(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "open: %v\n", err)
		os.Exit(1)
	}
	md := gf.Metadata
	keys := make([]string, 0, len(md))
	for k := range md {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Printf("%-60s = %v\n", k, md[k])
	}

	fmt.Println("\n--- Tensor info ---")
	for i, t := range gf.Tensors {
		if i < 5 || i > len(gf.Tensors)-3 {
			fmt.Printf("  [%d] %s  shape=%v  type=%d\n", i, t.Name, t.Dimensions, t.Type)
		}
		if i == 5 {
			fmt.Printf("  ... (%d tensors total) ...\n", len(gf.Tensors))
		}
	}
}
