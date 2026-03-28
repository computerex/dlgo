package gpu

// allocChecker is a helper that records the first VRAM allocation failure.
// After the first failure, subsequent calls to alloc are no-ops (return 0)
// so callers can batch many allocations and check a.err once at the end.
type allocChecker struct {
	err error
}

// allocFn is the allocation function used by allocChecker.
// Defaults to AllocE; overridden in tests to inject failures.
var allocFn = AllocE

func (a *allocChecker) alloc(sizeBytes uint64) Buf {
	if a.err != nil {
		return 0
	}
	buf, err := allocFn(sizeBytes)
	if err != nil {
		a.err = err
		return 0
	}
	return buf
}
