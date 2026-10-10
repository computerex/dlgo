//go:build ignore

package main

import (
    "fmt"
    "os"
    "math"

    "github.com/computerex/dlgo/models/llm"
    "github.com/computerex/dlgo/ops"
)

func main() {
    m, err := llm.LoadModel(os.Args[1])
    if err != nil {
        fmt.Fprintf(os.Stderr, "error: %v\n", err)
        os.Exit(1)
    }
    cfg := m.Config
    fmt.Printf("RopeScaleType=%d, Factor=%.1f, OrigMaxPos=%d\n", cfg.RopeScaleType, cfg.RopeScaleFactor, cfg.RopeOrigMaxPos)
    fmt.Printf("FreqBase=%.1f, FreqBaseSWA=%.1f\n", cfg.RopeFreqBase, cfg.RopeFreqBaseSWA)
    fmt.Printf("BetaFast=%.1f, BetaSlow=%.1f, ExtFactor=%.1f, AttnFactor=%.1f\n", cfg.RopeYaRNBetaFast, cfg.RopeYaRNBetaSlow, cfg.RopeYaRNExtFactor, cfg.RopeYaRNAttnFactor)
    fmt.Printf("HeadDim=%d, RopeDim=%d\n", cfg.HeadDim, cfg.RopeDim)
    
    dim := cfg.RopeDim
    if dim <= 0 { dim = cfg.HeadDim }
    
    plain_cos, _ := ops.RoPEFrequencyTable(10, dim, cfg.RopeFreqBase)
    yarn_cos, _ := ops.YaRNRoPEFrequencyTable(10, dim, cfg.RopeFreqBase,
        cfg.RopeScaleFactor, cfg.RopeOrigMaxPos,
        cfg.RopeYaRNBetaFast, cfg.RopeYaRNBetaSlow,
        cfg.RopeYaRNExtFactor, cfg.RopeYaRNAttnFactor)
    
    half := dim / 2
    maxDiff := float64(0)
    for i := 0; i < half && i < 5; i++ {
        for pos := 0; pos < 10; pos++ {
            p := plain_cos[pos*half+i]
            y := yarn_cos[pos*half+i]
            diff := math.Abs(float64(p) - float64(y))
            if diff > maxDiff { maxDiff = diff }
            if pos == 5 {
                fmt.Printf("  dim=%d pos=%d: plain=%.6f yarn=%.6f diff=%.6f\n", i, pos, p, y, diff)
            }
        }
    }
    fmt.Printf("Max diff between plain and YaRN RoPE: %.6f\n", maxDiff)
}