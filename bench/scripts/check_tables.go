//go:build ignore

package main

import (
    "fmt"
    "os"

    "github.com/computerex/dlgo/models/llm"
)

func main() {
    pipe, err := llm.NewPipeline(os.Args[1], 4096)
    if err != nil {
        fmt.Fprintf(os.Stderr, "error: %v\n", err)
        os.Exit(1)
    }
    cos, _ := pipe.RunState.RoPETables()
    if cos == nil {
        fmt.Println("RoPE tables: nil (NOT set)")
    } else {
        fmt.Printf("RoPE tables: %d entries, cos[0]=%.6f\n", len(cos), cos[0])
        // YaRN mscale for factor=32 should make cos[0] ~ 1.347 (not 1.0)
        if cos[0] > 1.01 {
            fmt.Println("  -> YaRN tables ACTIVE (mscale > 1)")
        } else {
            fmt.Println("  -> PLAIN tables (no YaRN)")
        }
    }
}