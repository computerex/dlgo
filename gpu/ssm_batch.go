//go:build cgo && vulkan

package gpu

import "os"

// Opt-in until model E2E and state validation complete. Reference disables it.
var ssmBatchEnabled = os.Getenv("DLGO_SSM_BATCH") == "1"

func ssmBatchSupported(heads, kd, vd, keyDim, npos int) bool {
	return ssmBatchEnabled && !ssmReference && !cpuSSMDelta && !cpuSSMFull && heads > 0 && kd > 0 && kd <= 128 && vd > 0 && vd <= 128 && keyDim > 0 && keyDim%kd == 0 && keyDim/kd <= heads && npos > 1 && npos <= 512
}

func ssmBatchPrefill(raw, processed, alpha, beta, z, y, state, convState, convW, a, bias, norm Buf, heads, kd, vd, keyDim, convK, npos int, eps float32) error {
	return ssmBatchPrefillTimed(raw, processed, alpha, beta, z, y, state, convState, convW, a, bias, norm, heads, kd, vd, keyDim, convK, npos, eps, false)
}

func ssmBatchPrefillTimed(raw, processed, alpha, beta, z, y, state, convState, convW, a, bias, norm Buf, heads, kd, vd, keyDim, convK, npos int, eps float32, timed bool) error {
	started, on := ssmTimeBegin(timed)
	if err := ssmBatchStage(0, raw, processed, convState, convW, 0, heads, kd, vd, keyDim, convK, npos, eps, false); err != nil {
		return err
	}
	ssmTimeEnd(5, started, on)
	Barrier()
	// Models without a dt-bias tensor still need a valid descriptor at binding
	// 3; bind any live buffer (alpha) and keep has_bias=0 so it is never read.
	biasDesc := bias
	if biasDesc == 0 {
		biasDesc = alpha
	}
	started, on = ssmTimeBegin(timed)
	if err := ssmBatchStage(1, alpha, beta, a, biasDesc, processed, heads, kd, vd, keyDim, convK, npos, eps, bias != 0); err != nil {
		return err
	}
	ssmTimeEnd(6, started, on)
	Barrier()
	started, on = ssmTimeBegin(timed)
	if err := ssmBatchStage(2, state, processed, alpha, beta, y, heads, kd, vd, keyDim, convK, npos, eps, false); err != nil {
		return err
	}
	ssmTimeEnd(7, started, on)
	Barrier()
	started, on = ssmTimeBegin(timed)
	err := ssmBatchStage(3, y, z, norm, 0, 0, heads, kd, vd, keyDim, convK, npos, eps, false)
	ssmTimeEnd(8, started, on)
	return err
}
