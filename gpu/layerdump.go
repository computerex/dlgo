//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math"
	"os"
)

// Per-layer hidden-state dump for comparing two forward paths layer by layer.
//
//	DLGO_LAYER_DUMP=1 lfmcheck ... | grep 'layer-dump'
//
// Prints the L2 norm and the head of the hidden row for each layer so the first
// layer where two implementations diverge is obvious. Diagnostic only.
// DLGO_DUMP_INLINE=off restricts it to the per-layer boundary checkpoints;
// =batch or =token adds the per-stage checkpoints inside the attention core for
// just one of the two paths.
var layerDumpEnabled = os.Getenv("DLGO_LAYER_DUMP") == "1"

// inlineDumpMode selects the extra per-stage checkpoints inside the attention
// core: "both" (default), "batch", "token", or "off". They exist to locate a
// divergence inside one layer, and they are heavier than the boundary dumps.
var inlineDumpMode = func() string {
	if layerDumpEnabled && os.Getenv("DLGO_DUMP_INLINE") == "0" {
		return "off"
	}
	if m := os.Getenv("DLGO_DUMP_INLINE"); m == "batch" || m == "token" {
		return m
	}
	return "both"
}()

func inlineDumpOn(side string) bool {
	return layerDumpEnabled && (inlineDumpMode == "both" || inlineDumpMode == side)
}

// layerDumpPath identifies which forward path is emitting the dump.
var layerDumpPath string

// layerDumpBatchRow dumps the last row of an [npos x n] batch buffer in the
// same format as layerDumpRow, so a batched checkpoint can be compared directly
// against the per-token path's dump of the same checkpoint. Submits and drains
// the pending batch first (a download needs finished work) and re-opens it.
func layerDumpBatchRow(tag string, l int, x Buf, npos, n int) {
	if !layerDumpEnabled {
		return
	}
	EndBatch()
	Sync()
	all := make([]float32, npos*n)
	err := DownloadF32(x, all)
	BeginBatch()
	if err != nil {
		fmt.Fprintf(os.Stderr, "[layer-dump] %s L%d batched download failed: %v\n", tag, l, err)
		return
	}
	dumpVec(tag, l, all[(npos-1)*n:npos*n])
}

// layerDumpRow downloads and prints a single-token buffer at a checkpoint.
func layerDumpRow(tag string, l int, x Buf, n int) {
	if !layerDumpEnabled {
		return
	}
	all := make([]float32, n)
	if err := DownloadF32(x, all); err != nil {
		fmt.Fprintf(os.Stderr, "[layer-dump] %s L%d download failed: %v\n", tag, l, err)
		return
	}
	dumpVec(tag, l, all)
}

// dumpVec prints the norm and the leading elements of one hidden row.
func dumpVec(tag string, l int, buf []float32) {
	var ss float64
	for _, v := range buf {
		ss += float64(v) * float64(v)
	}
	fmt.Fprintf(os.Stderr, "[layer-dump] %s L%02d norm=%.6f x=%v %v %v %v\n",
		tag, l, math.Sqrt(ss), buf[0], buf[1], buf[2], buf[3])
}
