package llm

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
)

// CPUDiag provides structured CPU forward pass diagnostic logging, mirroring
// the GPU diagnostic system. Enable with DLGO_CPU_DIAG=1 or the same layer/
// position filter syntax as DLGO_GPU_DIAG.
//
// Enable: DLGO_CPU_DIAG=1           (all layers, all positions)
//         DLGO_CPU_DIAG=L0          (layer 0 only)
//         DLGO_CPU_DIAG=L0:P0       (layer 0, position 0 only)
//         DLGO_CPU_DIAG=L0-L5       (layers 0 through 5)
//
// Verbosity: DLGO_CPU_DIAG_V=1  (default: print first 8 elements)
//            DLGO_CPU_DIAG_V=2  (print first 16 elements + norms)
//            DLGO_CPU_DIAG_V=3  (print first 32 elements + full stats)
var CPUDiag cpuDiagState

type cpuDiagState struct {
	Enabled    bool
	Verbosity  int
	layers     map[int]bool
	allLayers  bool
	posMin     int
	posMax     int
	allPos     bool
	initOnce   sync.Once
}

func (d *cpuDiagState) Init() {
	d.initOnce.Do(func() {
		env := os.Getenv("DLGO_CPU_DIAG")
		if env == "" {
			return
		}
		d.Enabled = true
		d.allLayers = true
		d.allPos = true
		d.posMin = 0
		d.posMax = -1
		d.Verbosity = 1

		if v := os.Getenv("DLGO_CPU_DIAG_V"); v != "" {
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

func (d *cpuDiagState) parseLayers(s string) {
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

func (d *cpuDiagState) parsePositions(s string) {
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

func (d *cpuDiagState) Active(layer, pos int) bool {
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

func diagVecL2(v []float32) float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return float32(math.Sqrt(s))
}

func diagVecMean(v []float32) float32 {
	if len(v) == 0 {
		return 0
	}
	var s float64
	for _, x := range v {
		s += float64(x)
	}
	return float32(s / float64(len(v)))
}

func diagVecMinMax(v []float32) (float32, float32) {
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

func (d *cpuDiagState) preview(v []float32) string {
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

func (d *cpuDiagState) stats(v []float32) string {
	if d.Verbosity < 2 {
		return ""
	}
	mn, mx := diagVecMinMax(v)
	return fmt.Sprintf(" norm=%.4f mean=%.6f min=%.4f max=%.4f", diagVecL2(v), diagVecMean(v), mn, mx)
}

// LogSlice logs a named CPU float slice.
func (d *cpuDiagState) LogSlice(layer, pos int, name string, data []float32) {
	fmt.Printf("[DIAG CPU L%d pos=%d] %s: %s%s\n", layer, pos, name, d.preview(data), d.stats(data))
}

// LogScalar logs a named scalar.
func (d *cpuDiagState) LogScalar(layer, pos int, name string, val float32) {
	fmt.Printf("[DIAG CPU L%d pos=%d] %s: %.8f\n", layer, pos, name, val)
}

// LogInfo logs a free-form diagnostic message.
func (d *cpuDiagState) LogInfo(layer, pos int, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Printf("[DIAG CPU L%d pos=%d] %s\n", layer, pos, msg)
}

// DiagLayer and DiagPos track the current layer/position for functions
// that don't have direct access to them (e.g. MoE FFN dispatch).
// Only set when CPUDiag is enabled.
var DiagLayer, DiagPos int

// LogMoE logs MoE routing decisions.
func (d *cpuDiagState) LogMoE(layer, pos int, routerLogits []float32, indices []int, weights []float32, gatingFunc int) {
	gatingNames := map[int]string{0: "softmax", 1: "softmax", 2: "sigmoid", 3: "softmax_weight"}
	gname := gatingNames[gatingFunc]
	if gname == "" {
		gname = fmt.Sprintf("unknown(%d)", gatingFunc)
	}
	fmt.Printf("[DIAG CPU L%d pos=%d] MoE gating=%s indices=%v weights=%v\n",
		layer, pos, gname, indices, weights)
	if d.Verbosity >= 2 {
		fmt.Printf("[DIAG CPU L%d pos=%d] MoE routerLogits: %s%s\n",
			layer, pos, d.preview(routerLogits), d.stats(routerLogits))
	}
}
