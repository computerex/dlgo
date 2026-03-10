//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
)

// Diag provides structured GPU diagnostic logging that can be enabled at
// runtime via environment variables. When disabled (the default), all
// checkpoint functions reduce to a single boolean check with zero allocations
// and zero GPU synchronization.
//
// Enable: DLGO_GPU_DIAG=1           (all layers, all positions)
//         DLGO_GPU_DIAG=L0          (layer 0 only)
//         DLGO_GPU_DIAG=L0,L1       (layers 0 and 1)
//         DLGO_GPU_DIAG=L0:P0       (layer 0, position 0 only)
//         DLGO_GPU_DIAG=L0-L5       (layers 0 through 5)
//         DLGO_GPU_DIAG=1:P0-P3     (all layers, positions 0-3)
//
// Verbosity: DLGO_GPU_DIAG_V=1  (default: print first 8 elements)
//            DLGO_GPU_DIAG_V=2  (print first 16 elements + norms)
//            DLGO_GPU_DIAG_V=3  (print first 32 elements + full norms + comparisons)
var GpuDiag diagState

type diagState struct {
	Enabled    bool
	Verbosity  int
	layers     map[int]bool // nil = all layers
	allLayers  bool
	posMin     int
	posMax     int // -1 = all positions
	allPos     bool
	initOnce   sync.Once
}

func (d *diagState) Init() {
	d.initOnce.Do(func() {
		env := os.Getenv("DLGO_GPU_DIAG")
		if env == "" {
			return
		}
		d.Enabled = true
		d.allLayers = true
		d.allPos = true
		d.posMin = 0
		d.posMax = -1
		d.Verbosity = 1

		if v := os.Getenv("DLGO_GPU_DIAG_V"); v != "" {
			if n, err := strconv.Atoi(v); err == nil {
				d.Verbosity = n
			}
		}

		if env == "1" {
			return
		}

		parts := strings.Split(env, ":")
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if part == "1" {
				continue
			}
			if strings.HasPrefix(part, "P") {
				d.parsePositions(part)
			} else if strings.HasPrefix(part, "L") {
				d.parseLayers(part)
			}
		}
	})
}

func (d *diagState) parseLayers(s string) {
	for _, seg := range strings.Split(s, ",") {
		seg = strings.TrimSpace(seg)
		if !strings.HasPrefix(seg, "L") {
			continue
		}
		seg = seg[1:]
		if idx := strings.Index(seg, "-"); idx >= 0 {
			lo, _ := strconv.Atoi(seg[:idx])
			hiStr := seg[idx+1:]
			if strings.HasPrefix(hiStr, "L") {
				hiStr = hiStr[1:]
			}
			hi, _ := strconv.Atoi(hiStr)
			if d.layers == nil {
				d.layers = make(map[int]bool)
			}
			d.allLayers = false
			for i := lo; i <= hi; i++ {
				d.layers[i] = true
			}
		} else {
			n, _ := strconv.Atoi(seg)
			if d.layers == nil {
				d.layers = make(map[int]bool)
			}
			d.allLayers = false
			d.layers[n] = true
		}
	}
}

func (d *diagState) parsePositions(s string) {
	s = strings.TrimPrefix(s, "P")
	d.allPos = false
	if idx := strings.Index(s, "-"); idx >= 0 {
		d.posMin, _ = strconv.Atoi(s[:idx])
		hiStr := s[idx+1:]
		if strings.HasPrefix(hiStr, "P") {
			hiStr = hiStr[1:]
		}
		d.posMax, _ = strconv.Atoi(hiStr)
	} else {
		n, _ := strconv.Atoi(s)
		d.posMin = n
		d.posMax = n
	}
}

// Active returns true if diagnostics should fire for this layer+position.
// This is the hot-path guard: a single boolean check when disabled.
func (d *diagState) Active(layer, pos int) bool {
	if !d.Enabled {
		return false
	}
	if !d.allLayers && !d.layers[layer] {
		return false
	}
	if !d.allPos {
		if pos < d.posMin || (d.posMax >= 0 && pos > d.posMax) {
			return false
		}
	}
	return true
}

// SnapshotBuf downloads a GPU buffer to CPU and returns it. Only call when Active() is true.
func (d *diagState) SnapshotBuf(buf Buf, n int) []float32 {
	EndBatch()
	Sync()
	data := make([]float32, n)
	DownloadF32(buf, data)
	BeginBatch()
	return data
}

func vecL2(v []float32) float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return float32(math.Sqrt(s))
}

func vecMean(v []float32) float32 {
	if len(v) == 0 {
		return 0
	}
	var s float64
	for _, x := range v {
		s += float64(x)
	}
	return float32(s / float64(len(v)))
}

func vecMinMax(v []float32) (float32, float32) {
	if len(v) == 0 {
		return 0, 0
	}
	mn, mx := v[0], v[0]
	for _, x := range v[1:] {
		if x < mn {
			mn = x
		}
		if x > mx {
			mx = x
		}
	}
	return mn, mx
}

func (d *diagState) preview(v []float32) string {
	n := 8
	if d.Verbosity >= 2 {
		n = 16
	}
	if d.Verbosity >= 3 {
		n = 32
	}
	if n > len(v) {
		n = len(v)
	}
	return fmt.Sprintf("%v", v[:n])
}

func (d *diagState) stats(v []float32) string {
	if d.Verbosity < 2 {
		return ""
	}
	mn, mx := vecMinMax(v)
	return fmt.Sprintf(" norm=%.4f mean=%.6f min=%.4f max=%.4f", vecL2(v), vecMean(v), mn, mx)
}

// --- Checkpoint functions ---

// LogEmbed logs the token embedding.
func (d *diagState) LogEmbed(tag string, pos int, x []float32) {
	if !d.Enabled {
		return
	}
	fmt.Printf("[DIAG %s pos=%d] Embed: %s%s\n", tag, pos, d.preview(x), d.stats(x))
}

// LogBuf logs a named GPU buffer at a layer/position checkpoint.
func (d *diagState) LogBuf(tag string, layer, pos int, name string, buf Buf, n int) {
	data := d.SnapshotBuf(buf, n)
	fmt.Printf("[DIAG %s L%d pos=%d] %s: %s%s\n", tag, layer, pos, name, d.preview(data), d.stats(data))
}

// LogSlice logs a named CPU float slice at a layer/position checkpoint.
func (d *diagState) LogSlice(tag string, layer, pos int, name string, data []float32) {
	fmt.Printf("[DIAG %s L%d pos=%d] %s: %s%s\n", tag, layer, pos, name, d.preview(data), d.stats(data))
}

// LogBufPair logs the same named buffer from GPU alongside a CPU reference for comparison.
func (d *diagState) LogBufPair(layer, pos int, name string, gpuBuf Buf, cpuData []float32, n int) {
	gpuData := d.SnapshotBuf(gpuBuf, n)
	fmt.Printf("[DIAG GPU L%d pos=%d] %s: %s%s\n", layer, pos, name, d.preview(gpuData), d.stats(gpuData))
	fmt.Printf("[DIAG CPU L%d pos=%d] %s: %s%s\n", layer, pos, name, d.preview(cpuData), d.stats(cpuData))
	if d.Verbosity >= 3 && len(gpuData) > 0 && len(cpuData) > 0 {
		maxDiff := float32(0)
		maxIdx := 0
		for i := 0; i < len(gpuData) && i < len(cpuData); i++ {
			diff := gpuData[i] - cpuData[i]
			if diff < 0 {
				diff = -diff
			}
			if diff > maxDiff {
				maxDiff = diff
				maxIdx = i
			}
		}
		fmt.Printf("[DIAG CMP L%d pos=%d] %s: maxAbsDiff=%.8f at [%d] (gpu=%.8f cpu=%.8f)\n",
			layer, pos, name, maxDiff, maxIdx, gpuData[maxIdx], cpuData[maxIdx])
	}
}

// LogScalar logs a named scalar value.
func (d *diagState) LogScalar(tag string, layer, pos int, name string, val float32) {
	fmt.Printf("[DIAG %s L%d pos=%d] %s: %.8f\n", tag, layer, pos, name, val)
}

// LogInfo logs a free-form diagnostic message.
func (d *diagState) LogInfo(tag string, layer, pos int, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Printf("[DIAG %s L%d pos=%d] %s\n", tag, layer, pos, msg)
}

// LogMoE logs MoE routing decisions.
func (d *diagState) LogMoE(tag string, layer, pos int, routerLogits []float32, indices []int, weights []float32, gatingFunc int) {
	gatingNames := map[int]string{0: "softmax", 1: "softmax", 2: "sigmoid", 3: "softmax_weight"}
	gname := gatingNames[gatingFunc]
	if gname == "" {
		gname = fmt.Sprintf("unknown(%d)", gatingFunc)
	}
	fmt.Printf("[DIAG %s L%d pos=%d] MoE gating=%s indices=%v weights=%v\n",
		tag, layer, pos, gname, indices, weights)
	if d.Verbosity >= 2 {
		fmt.Printf("[DIAG %s L%d pos=%d] MoE routerLogits: %s%s\n",
			tag, layer, pos, d.preview(routerLogits), d.stats(routerLogits))
	}
}
