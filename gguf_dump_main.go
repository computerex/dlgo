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
		fmt.Println("Usage: gguf_dump <path.gguf>")
		os.Exit(1)
	}
	for _, path := range os.Args[1:] {
		dumpGGUF(path)
	}
}

func dumpGGUF(path string) {
	fmt.Printf("\n\n========== %s ==========\n", path)
	gf, err := gguf.Open(path)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("=== GGUF Metadata ===")
	keys := make([]string, 0, len(gf.Metadata))
	for k := range gf.Metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := gf.Metadata[k]
		if strings.Contains(k, "tokens") || strings.Contains(k, "scores") || strings.Contains(k, "token_type") || strings.Contains(k, "merges") || strings.Contains(k, "chat_template") {
			switch vv := v.(type) {
			case []interface{}:
				fmt.Printf("  %s = [%d items]\n", k, len(vv))
			case string:
				if len(vv) > 100 {
					fmt.Printf("  %s = %q... (%d chars)\n", k, vv[:100], len(vv))
				} else {
					fmt.Printf("  %s = %q\n", k, vv)
				}
			default:
				fmt.Printf("  %s = %v\n", k, v)
			}
			continue
		}
		fmt.Printf("  %s = %v\n", k, v)
	}

	fmt.Printf("\n=== Tensors (first 20 + SSM-related) ===\n")
	count := 0
	for _, t := range gf.Tensors {
		isSSM := strings.Contains(t.Name, "ssm") || strings.Contains(t.Name, "gate")
		isFirst := count < 20
		isLayer0 := strings.HasPrefix(t.Name, "blk.0.") || strings.HasPrefix(t.Name, "blk.3.")
		if isSSM || isFirst || isLayer0 {
			fmt.Printf("  %s  dims=%v  type=%d\n", t.Name, t.Dimensions, t.Type)
		}
		count++
	}
	fmt.Printf("Total tensors: %d\n", count)
}
