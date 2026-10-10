//go:build cgo && vulkan

package gpu

import "os"

// DLGO_SSM_REFERENCE=1 retains the four-dispatch compensated reference.
// Read once at startup, like the other GPU path switches.
var ssmReference = os.Getenv("DLGO_SSM_REFERENCE") == "1"

// ssmRecurrence preserves reference summation order while avoiding three
// dispatches/barriers and intermediate state/prediction traffic per token.
// Larger key columns use the unbounded reference instead of col[128].
func ssmRecurrence(state, qkv, alpha, beta, pred, y Buf, heads, kd, vd, keyDim int) error {
	if !ssmReference && kd > 0 && kd <= 128 && vd > 0 && vd <= 128 && keyDim > 0 && keyDim%kd == 0 {
		return SSMDeltaRule(state, qkv, alpha, beta, y, heads, kd, vd, keyDim)
	}
	return ssmRecurrenceReference(state, qkv, alpha, beta, pred, y, heads, kd, vd, keyDim)
}

func ssmRecurrenceReference(state, qkv, alpha, beta, pred, y Buf, heads, kd, vd, keyDim int) error {
	if err := SSMDecay(state, alpha, heads, kd, vd); err != nil {
		return err
	}
	Barrier()
	if err := SSMPredict(state, qkv, pred, heads, kd, vd, keyDim); err != nil {
		return err
	}
	Barrier()
	if err := SSMUpdate(state, qkv, beta, pred, heads, kd, vd, keyDim); err != nil {
		return err
	}
	Barrier()
	return SSMOutput(state, qkv, y, heads, kd, vd, keyDim)
}
