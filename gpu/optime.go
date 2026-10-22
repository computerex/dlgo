//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"os"
	"time"
)

// Per-op timing for the non-fused single-token forward path.
//
// That path records a whole token into one command buffer, so the only way to
// attribute time to an op class is to submit and wait around it. Submitting per
// class removes cross-class overlap, so read the numbers as ratios, not as the
// token time you would measure with the profiler off. Same caveat as the DiT
// -debug profiler.
//
//	DLGO_GPU_TIMING=1 lfmcheck -gpu ...
var opTimeEnabled = os.Getenv("DLGO_GPU_TIMING") == "1"

const (
	opTimeNorm = iota
	opTimeQKV
	opTimeNormRope
	opTimeCore
	opTimeOutProj
	opTimeMoE
	opTimeFFN
	opTimeResidual
	opTimeLogits
	opTimeClasses
)

var opTimeNames = [opTimeClasses]string{
	"norm",
	"qkv/in-proj",
	"norm+rope+kvstore",
	"attention/shortconv",
	"out-proj",
	"moe-ffn",
	"dense-ffn",
	"residual",
	"logits",
}

var (
	opTimeTot       [opTimeClasses]time.Duration
	opTimeTokens    int
	opTimeNextPrint = 2
)

// opTimeBegin starts the clock for one op class. It only drains prior work; the
// submission that makes the measurement meaningful happens in opTimeEnd, which
// mirrors the ssmTimeBegin/End pairing.
func opTimeBegin() (int64, bool) {
	if !opTimeEnabled {
		return 0, false
	}
	Sync()
	return time.Now().UnixNano(), true
}

// opTimeEnd closes the batch, waits for it, accounts the elapsed GPU time and
// reopens the batch for the next class.
func opTimeEnd(idx int, start int64, on bool) {
	if !on {
		return
	}
	EndBatch()
	Sync()
	opTimeTot[idx] += time.Duration(time.Now().UnixNano() - start)
	BeginBatch()
}

// opTimeTick is called once per forwarded token. It prints a windowed summary at
// 2, 4, 8, ... tokens so a prefill run shows how the split shifts with position
// (the attention term grows, the weight-bound terms do not).
func opTimeTick() {
	if !opTimeEnabled {
		return
	}
	opTimeTokens++
	if opTimeTokens < opTimeNextPrint {
		return
	}
	var total time.Duration
	for _, d := range opTimeTot {
		total += d
	}
	fmt.Fprintf(os.Stderr, "[dlgo/gpu][op-time] tokens %d..%d: %s for %d token(s)\n",
		opTimeTokens-opTimeNextPrint+1, opTimeTokens, total.Round(time.Microsecond),
		opTimeNextPrint)
	for i := 0; i < opTimeClasses; i++ {
		pct := 0.0
		if total > 0 {
			pct = 100 * float64(opTimeTot[i]) / float64(total)
		}
		fmt.Fprintf(os.Stderr, "[dlgo/gpu][op-time]   %-22s %12s  %5.1f%%\n",
			opTimeNames[i], opTimeTot[i].Round(time.Microsecond), pct)
		opTimeTot[i] = 0
	}
	opTimeNextPrint *= 2
}
