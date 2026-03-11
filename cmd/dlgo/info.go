package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/computerex/dlgo/format/gguf"
)

func cmdInfo(args []string) {
	fs := flag.NewFlagSet("info", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: dlgo info <model.gguf>")
	}
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "Error: model path required")
		fmt.Fprintln(os.Stderr, "Usage: dlgo info <model.gguf>")
		os.Exit(1)
	}

	path := fs.Arg(0)
	f, err := gguf.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("File:           %s\n", path)
	fmt.Printf("GGUF version:   %d\n", f.Version)
	fmt.Printf("Tensors:        %d\n", f.TensorCount)
	fmt.Printf("Metadata keys:  %d\n", f.MetadataCount)
	fmt.Println()

	// Print key metadata
	keys := []string{
		"general.architecture",
		"general.name",
		"general.quantization_version",
		"general.file_type",
	}
	for _, k := range keys {
		if v, ok := f.Metadata[k]; ok {
			fmt.Printf("%-30s %v\n", k, v)
		}
	}

	arch := ""
	if v, ok := f.Metadata["general.architecture"]; ok {
		arch = fmt.Sprint(v)
	}

	if arch != "" {
		archKeys := []string{
			arch + ".context_length",
			arch + ".embedding_length",
			arch + ".block_count",
			arch + ".attention.head_count",
			arch + ".attention.head_count_kv",
			arch + ".feed_forward_length",
			arch + ".vocab_size",
		}
		for _, k := range archKeys {
			if v, ok := f.Metadata[k]; ok {
				fmt.Printf("%-30s %v\n", k, v)
			}
		}
	}

	// Quantization type distribution
	fmt.Println()
	fmt.Println("Tensor types:")
	typeCounts := map[gguf.GGMLType]int{}
	for _, t := range f.Tensors {
		typeCounts[t.Type]++
	}
	for typ, count := range typeCounts {
		fmt.Printf("  type %-6d  %d tensors\n", typ, count)
	}
}
