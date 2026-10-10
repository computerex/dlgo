//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/format/gguf"
)

func main() {
	gf, err := gguf.Open(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	for k, v := range gf.Metadata {
		if k == "general.architecture" || k == "general.name" {
			fmt.Printf("%s = %v\n", k, v)
		}
	}
	arch, _ := gf.Metadata["general.architecture"].(string)
	keys := []string{
		arch + ".context_length",
		arch + ".rope.freq_base",
		arch + ".rope.dimension_count",
		arch + ".rope.scaling.type",
		arch + ".rope.scaling.factor",
		arch + ".rope.scaling.original_context_length",
		arch + ".expert_weights_scale",
		arch + ".expert_weights_norm",
		arch + ".expert_count",
		arch + ".expert_used_count",
		arch + ".block_count",
		arch + ".embedding_length",
		arch + ".attention.head_count",
		arch + ".attention.head_count_kv",
		arch + ".attention.key_length",
	}
	for _, k := range keys {
		if v, ok := gf.Metadata[k]; ok {
			fmt.Printf("%s = %v\n", k, v)
		}
	}
}
