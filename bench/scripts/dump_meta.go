//go:build ignore

package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/computerex/dlgo/format/gguf"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: dump_meta.go <model.gguf>\n")
		os.Exit(1)
	}
	gf, err := gguf.Open(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "open: %v\n", err)
		os.Exit(1)
	}

	keys := make([]string, 0, len(gf.Metadata))
	for k := range gf.Metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		v := gf.Metadata[k]
		if strings.Contains(k, "sliding") || strings.Contains(k, "pattern") ||
			strings.Contains(k, "window") || strings.Contains(k, "rope") ||
			strings.Contains(k, "softcap") || strings.Contains(k, "scale") ||
			strings.Contains(k, "head") || strings.Contains(k, "layer") ||
			strings.Contains(k, "attention") || strings.Contains(k, "embed") ||
			strings.Contains(k, "eos") || strings.Contains(k, "bos") ||
			strings.Contains(k, "stop") || strings.Contains(k, "token") ||
			strings.Contains(k, "expert") || strings.Contains(k, "moe") ||
			strings.Contains(k, "block_count") || strings.Contains(k, "arch") {
			fmt.Printf("  %s = %v\n", k, v)
		}
	}
}
