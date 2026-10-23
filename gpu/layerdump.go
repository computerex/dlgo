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
var layerDumpEnabled = os.Getenv("DLGO_LAYER_DUMP") == "1"

// layerDumpPath identifies which forward path is emitting the dump.
var layerDumpPath string

func layerDumpRow(tag string, l int, x Buf, n int) {
	if !layerDumpEnabled {
		return
	}
	buf := make([]float32, n)
	if err := DownloadF32(x, buf); err != nil {
		fmt.Fprintf(os.Stderr, "[layer-dump] %s L%d download failed: %v\n", tag, l, err)
		return
	}
	var ss float64
	for _, v := range buf {
		ss += float64(v) * float64(v)
	}
	fmt.Fprintf(os.Stderr, "[layer-dump] %s L%02d norm=%.6f x=%v %v %v %v\n",
		tag, l, math.Sqrt(ss), buf[0], buf[1], buf[2], buf[3])
}
