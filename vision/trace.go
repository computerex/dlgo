package vision

import (
	"fmt"
	"os"
	"time"
)

// traceEnabled reports whether stage-sum tracing is on (DLGO_VISION_TRACE).
// When enabled, Encode prints the sum of all elements after each pipeline
// stage so it can be diffed against llama.cpp's common_debug_cb_eval output.
func traceEnabled() bool {
	return os.Getenv("DLGO_VISION_TRACE") != ""
}

func traceSum(name string, x [][]float32) {
	if !traceEnabled() {
		return
	}
	var sum float64
	for _, r := range x {
		for _, v := range r {
			sum += float64(v)
		}
	}
	fmt.Printf("TRACE %-18s sum=%.6f n=%d\n", name, sum, len(x))
}

func traceMark(name string, stage *time.Time) {
	if !traceEnabled() {
		return
	}
	fmt.Printf("TRACE %-18s %s\n", name, time.Since(*stage))
	*stage = time.Now()
}
