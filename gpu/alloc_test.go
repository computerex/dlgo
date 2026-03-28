//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
)

// --- Test helpers ---

// fakeAllocN returns an allocFn that succeeds for the first n calls,
// then fails all subsequent ones.
func fakeAllocN(n int) func(uint64) (Buf, error) {
	var count int64
	return func(size uint64) (Buf, error) {
		i := atomic.AddInt64(&count, 1)
		if i <= int64(n) {
			// Return a fake non-zero handle. Use the counter so each buf is unique.
			return Buf(i * 0x1000), nil
		}
		return 0, fmt.Errorf("fake OOM at allocation #%d (requested %d bytes)", i, size)
	}
}

// fakeAllocAlways returns an allocFn that always succeeds.
func fakeAllocAlways() func(uint64) (Buf, error) {
	var count int64
	return func(size uint64) (Buf, error) {
		i := atomic.AddInt64(&count, 1)
		return Buf(i * 0x1000), nil
	}
}

// fakeAllocNever returns an allocFn that always fails.
func fakeAllocNever() func(uint64) (Buf, error) {
	return func(size uint64) (Buf, error) {
		return 0, fmt.Errorf("fake OOM (all allocations disabled)")
	}
}

// withFakeAlloc sets allocFn to f for the duration of the test,
// restoring the original on cleanup.
func withFakeAlloc(t *testing.T, f func(uint64) (Buf, error)) {
	t.Helper()
	orig := allocFn
	allocFn = f
	t.Cleanup(func() { allocFn = orig })
}

// countNonZeroBufs counts how many Buf values in a slice are non-zero.
func countNonZeroBufs(bufs []Buf) int {
	n := 0
	for _, b := range bufs {
		if b != 0 {
			n++
		}
	}
	return n
}

// --- allocChecker tests ---

func TestAllocChecker_AllSucceed(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	a := allocChecker{}
	b1 := a.alloc(100)
	b2 := a.alloc(200)
	b3 := a.alloc(300)

	if a.err != nil {
		t.Fatalf("expected no error, got: %v", a.err)
	}
	if b1 == 0 || b2 == 0 || b3 == 0 {
		t.Fatalf("expected non-zero bufs, got %d %d %d", b1, b2, b3)
	}
	// Each buf should be unique
	if b1 == b2 || b2 == b3 || b1 == b3 {
		t.Fatalf("expected unique bufs, got %d %d %d", b1, b2, b3)
	}
}

func TestAllocChecker_FirstFails(t *testing.T) {
	withFakeAlloc(t, fakeAllocNever())

	a := allocChecker{}
	b1 := a.alloc(100)
	b2 := a.alloc(200)

	if a.err == nil {
		t.Fatal("expected error")
	}
	if b1 != 0 || b2 != 0 {
		t.Fatalf("expected zero bufs on failure, got %d %d", b1, b2)
	}
}

func TestAllocChecker_MiddleFails(t *testing.T) {
	withFakeAlloc(t, fakeAllocN(2))

	a := allocChecker{}
	b1 := a.alloc(100)
	b2 := a.alloc(200)
	b3 := a.alloc(300) // should fail
	b4 := a.alloc(400) // should be short-circuited

	if a.err == nil {
		t.Fatal("expected error after 3rd allocation")
	}
	if b1 == 0 || b2 == 0 {
		t.Fatal("first two allocations should have succeeded")
	}
	if b3 != 0 || b4 != 0 {
		t.Fatalf("failed/skipped allocations should return 0, got b3=%d b4=%d", b3, b4)
	}
	if !strings.Contains(a.err.Error(), "fake OOM") {
		t.Fatalf("unexpected error: %v", a.err)
	}
}

func TestAllocChecker_ZeroSize(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	a := allocChecker{}
	// Zero-size allocation should succeed (AllocE returns 0, nil for size 0)
	// But our fakeAllocAlways returns nonzero. Let's test that allocChecker
	// properly delegates to the fn.
	b := a.alloc(0)
	if a.err != nil {
		t.Fatalf("zero-size alloc should not error, got: %v", a.err)
	}
	// fakeAllocAlways returns non-zero even for 0, which is fine for testing
	_ = b
}

func TestAllocChecker_ErrorRecordsFirstOnly(t *testing.T) {
	// Fail on call 2 and 3 — error should be from call 2
	calls := 0
	withFakeAlloc(t, func(size uint64) (Buf, error) {
		calls++
		if calls >= 2 {
			return 0, fmt.Errorf("OOM#%d", calls)
		}
		return Buf(calls * 0x1000), nil
	})

	a := allocChecker{}
	a.alloc(100)
	a.alloc(200) // fails
	a.alloc(300) // short-circuit, never called

	if a.err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(a.err.Error(), "OOM#2") {
		t.Fatalf("expected first failure error, got: %v", a.err)
	}
	// call 3 should have been short-circuited, not reaching the fn
	if calls != 2 {
		t.Fatalf("expected allocFn called exactly 2 times, got %d", calls)
	}
}

// --- NewGpuRunState tests ---

func TestNewGpuRunState_Success(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	rs, err := NewGpuRunState(128, 128, 64, 256, 1000)
	if err != nil {
		t.Fatalf("expected success, got: %v", err)
	}
	if rs == nil {
		t.Fatal("expected non-nil RunState")
	}
	// Verify all 14 buffers are non-zero
	bufs := []Buf{rs.X, rs.XNorm, rs.Q, rs.K, rs.V, rs.AttnOut, rs.AttnProj,
		rs.FFNIn, rs.FFNNorm, rs.Gate, rs.Up, rs.Hidden, rs.FFNOut, rs.Logits}
	for i, b := range bufs {
		if b == 0 {
			t.Errorf("buf[%d] is zero", i)
		}
	}
}

func TestNewGpuRunState_OOMAtFirst(t *testing.T) {
	withFakeAlloc(t, fakeAllocNever())

	rs, err := NewGpuRunState(128, 128, 64, 256, 1000)
	if err == nil {
		t.Fatal("expected error")
	}
	if rs != nil {
		t.Fatal("expected nil RunState on OOM")
	}
	if !strings.Contains(err.Error(), "NewGpuRunState") {
		t.Fatalf("error should mention NewGpuRunState: %v", err)
	}
}

func TestNewGpuRunState_OOMPartial(t *testing.T) {
	// Fail at allocation 5 (out of 14), verify cleanup
	withFakeAlloc(t, fakeAllocN(4))

	rs, err := NewGpuRunState(128, 128, 64, 256, 1000)
	if err == nil {
		t.Fatal("expected error for partial OOM")
	}
	if rs != nil {
		t.Fatal("expected nil RunState on partial OOM")
	}
	if !strings.Contains(err.Error(), "NewGpuRunState") {
		t.Fatalf("error should mention NewGpuRunState: %v", err)
	}
}

func TestNewGpuRunState_OOMAtLast(t *testing.T) {
	// 14 allocations total; fail at the 14th (Logits)
	withFakeAlloc(t, fakeAllocN(13))

	rs, err := NewGpuRunState(128, 128, 64, 256, 1000)
	if err == nil {
		t.Fatal("expected error at last allocation")
	}
	if rs != nil {
		t.Fatal("expected nil RunState")
	}
}

// --- AllocSSMScratch tests ---

func TestAllocSSMScratch_Success(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	rs := &GpuRunState{}
	err := rs.AllocSSMScratch(1024, 512, 32)
	if err != nil {
		t.Fatalf("expected success: %v", err)
	}
	if rs.SSMQKV == 0 || rs.SSMZ == 0 || rs.SSMAlpha == 0 || rs.SSMBeta == 0 || rs.SSMY == 0 {
		t.Fatal("expected all SSM scratch bufs non-zero")
	}
}

func TestAllocSSMScratch_OOM(t *testing.T) {
	withFakeAlloc(t, fakeAllocN(2))

	rs := &GpuRunState{}
	err := rs.AllocSSMScratch(1024, 512, 32)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "AllocSSMScratch") {
		t.Fatalf("error should mention AllocSSMScratch: %v", err)
	}
}

// --- AllocGatedQScratch tests ---

func TestAllocGatedQScratch_Success(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	rs := &GpuRunState{}
	err := rs.AllocGatedQScratch(512)
	if err != nil {
		t.Fatalf("expected success: %v", err)
	}
	if rs.QFull == 0 || rs.QGate == 0 {
		t.Fatal("expected GatedQ bufs non-zero")
	}
}

func TestAllocGatedQScratch_OOM(t *testing.T) {
	withFakeAlloc(t, fakeAllocNever())

	rs := &GpuRunState{}
	err := rs.AllocGatedQScratch(512)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "AllocGatedQScratch") {
		t.Fatalf("error should mention AllocGatedQScratch: %v", err)
	}
}

func TestAllocGatedQScratch_SecondFails(t *testing.T) {
	withFakeAlloc(t, fakeAllocN(1))

	rs := &GpuRunState{}
	err := rs.AllocGatedQScratch(512)
	if err == nil {
		t.Fatal("expected error on second alloc")
	}
}

// --- NewGpuBatchState tests ---

func TestNewGpuBatchState_Success(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	bs, err := NewGpuBatchState(32, 128, 128, 64, 256)
	if err != nil {
		t.Fatalf("expected success: %v", err)
	}
	if bs == nil {
		t.Fatal("expected non-nil BatchState")
	}
	if bs.Npos != 32 {
		t.Fatalf("expected Npos=32, got %d", bs.Npos)
	}
	bufs := []Buf{bs.X, bs.XNorm, bs.Q, bs.K, bs.V, bs.AttnOut, bs.AttnProj,
		bs.FFNIn, bs.FFNNorm, bs.Gate, bs.Up, bs.Hidden, bs.FFNOut}
	for i, b := range bufs {
		if b == 0 {
			t.Errorf("buf[%d] is zero", i)
		}
	}
}

func TestNewGpuBatchState_OOM(t *testing.T) {
	withFakeAlloc(t, fakeAllocNever())

	bs, err := NewGpuBatchState(32, 128, 128, 64, 256)
	if err == nil {
		t.Fatal("expected error")
	}
	if bs != nil {
		t.Fatal("expected nil BatchState on OOM")
	}
	if !strings.Contains(err.Error(), "NewGpuBatchState") {
		t.Fatalf("error should mention NewGpuBatchState: %v", err)
	}
}

func TestNewGpuBatchState_OOMPartial(t *testing.T) {
	// 13 allocations; fail at 7th
	withFakeAlloc(t, fakeAllocN(6))

	bs, err := NewGpuBatchState(32, 128, 128, 64, 256)
	if err == nil {
		t.Fatal("expected error for partial OOM")
	}
	if bs != nil {
		t.Fatal("expected nil BatchState on partial OOM")
	}
}

// --- AllocGatedQBatch tests ---

func TestAllocGatedQBatch_Success(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	bs := &GpuBatchState{}
	err := bs.AllocGatedQBatch(32, 512)
	if err != nil {
		t.Fatalf("expected success: %v", err)
	}
	if bs.QFull == 0 || bs.QGate == 0 {
		t.Fatal("expected non-zero bufs")
	}
}

func TestAllocGatedQBatch_OOM(t *testing.T) {
	withFakeAlloc(t, fakeAllocNever())

	bs := &GpuBatchState{}
	err := bs.AllocGatedQBatch(32, 512)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "AllocGatedQBatch") {
		t.Fatalf("error should mention AllocGatedQBatch: %v", err)
	}
}

// --- AllocSSMBatch tests ---

func TestAllocSSMBatch_Success(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	bs := &GpuBatchState{}
	err := bs.AllocSSMBatch(32, 1024, 512, 32)
	if err != nil {
		t.Fatalf("expected success: %v", err)
	}
	if bs.SSMQKV == 0 || bs.SSMZ == 0 || bs.SSMAlpha == 0 || bs.SSMBeta == 0 || bs.SSMY == 0 {
		t.Fatal("expected all SSM batch bufs non-zero")
	}
}

func TestAllocSSMBatch_OOM(t *testing.T) {
	withFakeAlloc(t, fakeAllocN(3))

	bs := &GpuBatchState{}
	err := bs.AllocSSMBatch(32, 1024, 512, 32)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "AllocSSMBatch") {
		t.Fatalf("error should mention AllocSSMBatch: %v", err)
	}
}

// --- NewGpuKVCache tests ---

func TestNewGpuKVCache_Success(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	kv, err := NewGpuKVCache(4, 4, 512, 128, nil)
	if err != nil {
		t.Fatalf("expected success: %v", err)
	}
	if kv == nil {
		t.Fatal("expected non-nil KVCache")
	}
	if len(kv.KeyBufs) != 4 || len(kv.ValBufs) != 4 {
		t.Fatalf("expected 4 layers, got key=%d val=%d", len(kv.KeyBufs), len(kv.ValBufs))
	}
	for l := 0; l < 4; l++ {
		if kv.KeyBufs[l] == 0 || kv.ValBufs[l] == 0 {
			t.Errorf("layer %d: expected non-zero K/V bufs", l)
		}
	}
}

func TestNewGpuKVCache_PartialGPULayers(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	// 8 total layers, only 4 on GPU
	kv, err := NewGpuKVCache(8, 4, 512, 128, nil)
	if err != nil {
		t.Fatalf("expected success: %v", err)
	}
	for l := 0; l < 4; l++ {
		if kv.KeyBufs[l] == 0 {
			t.Errorf("layer %d: expected allocated", l)
		}
	}
	for l := 4; l < 8; l++ {
		if kv.KeyBufs[l] != 0 {
			t.Errorf("layer %d: expected zero (not on GPU)", l)
		}
	}
}

func TestNewGpuKVCache_NeedsKVMask(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	// 4 layers, only layers 0 and 2 need KV
	needsKV := []bool{true, false, true, false}
	kv, err := NewGpuKVCache(4, 4, 512, 128, needsKV)
	if err != nil {
		t.Fatalf("expected success: %v", err)
	}
	if kv.KeyBufs[0] == 0 || kv.KeyBufs[2] == 0 {
		t.Error("layers 0,2 should have KV")
	}
	if kv.KeyBufs[1] != 0 || kv.KeyBufs[3] != 0 {
		t.Error("layers 1,3 should NOT have KV")
	}
}

func TestNewGpuKVCache_OOMFirstLayer(t *testing.T) {
	withFakeAlloc(t, fakeAllocNever())

	kv, err := NewGpuKVCache(4, 4, 512, 128, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if kv != nil {
		t.Fatal("expected nil KVCache on OOM")
	}
	if !strings.Contains(err.Error(), "NewGpuKVCache") {
		t.Fatalf("error should mention NewGpuKVCache: %v", err)
	}
}

func TestNewGpuKVCache_OOMMiddleLayer(t *testing.T) {
	// Each layer uses 2 allocs (key + val). 4 layers = 8 allocs.
	// Fail at alloc 5 (layer 2 key buf).
	withFakeAlloc(t, fakeAllocN(4))

	kv, err := NewGpuKVCache(4, 4, 512, 128, nil)
	if err == nil {
		t.Fatal("expected error at middle layer")
	}
	if kv != nil {
		t.Fatal("expected nil on OOM")
	}
	if !strings.Contains(err.Error(), "layer 2") {
		t.Fatalf("expected error at layer 2, got: %v", err)
	}
}

func TestNewGpuKVCache_OOMLastLayer(t *testing.T) {
	// 4 layers × 2 allocs = 8. Fail at 8th (layer 3 val buf).
	withFakeAlloc(t, fakeAllocN(7))

	kv, err := NewGpuKVCache(4, 4, 512, 128, nil)
	if err == nil {
		t.Fatal("expected error at last layer")
	}
	if kv != nil {
		t.Fatal("expected nil on OOM")
	}
	if !strings.Contains(err.Error(), "layer 3") {
		t.Fatalf("expected error at layer 3, got: %v", err)
	}
}

func TestNewGpuKVCache_ZeroLayers(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	kv, err := NewGpuKVCache(0, 0, 512, 128, nil)
	if err != nil {
		t.Fatalf("zero layers should succeed: %v", err)
	}
	if kv == nil {
		t.Fatal("expected non-nil cache even with zero layers")
	}
}

// --- NewPagedKVPool tests ---

func TestNewPagedKVPool_Success(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	pool, err := NewPagedKVPool(4, 128, 16, 64)
	if err != nil {
		t.Fatalf("expected success: %v", err)
	}
	if pool == nil {
		t.Fatal("expected non-nil pool")
	}
	if pool.NLayers != 4 || pool.KVDim != 128 || pool.BlockSize != 16 || pool.TotalBlocks != 64 {
		t.Fatal("pool fields mismatch")
	}
	for l := 0; l < 4; l++ {
		if pool.KeyPools[l] == 0 || pool.ValPools[l] == 0 {
			t.Errorf("layer %d: expected non-zero pool bufs", l)
		}
	}
}

func TestNewPagedKVPool_OOM(t *testing.T) {
	// 4 layers × 2 = 8 allocs. Fail at 3rd.
	withFakeAlloc(t, fakeAllocN(2))

	pool, err := NewPagedKVPool(4, 128, 16, 64)
	if err == nil {
		t.Fatal("expected error")
	}
	if pool != nil {
		t.Fatal("expected nil pool on OOM")
	}
	if !strings.Contains(err.Error(), "NewPagedKVPool") {
		t.Fatalf("error should mention NewPagedKVPool: %v", err)
	}
}

func TestNewPagedKVPool_OOMFirstAlloc(t *testing.T) {
	withFakeAlloc(t, fakeAllocNever())

	pool, err := NewPagedKVPool(2, 64, 8, 32)
	if err == nil {
		t.Fatal("expected error")
	}
	if pool != nil {
		t.Fatal("expected nil")
	}
}

// --- FreeBatchState nil safety ---

func TestFreeBatchState_Nil(t *testing.T) {
	var bs *GpuBatchState
	// Should not panic
	bs.Free()
}

func TestFreeRunState_Nil(t *testing.T) {
	var rs *GpuRunState
	// Should not panic
	rs.FreeAll()
}

func TestFreeKVCache_Nil(t *testing.T) {
	var kv *GpuKVCache
	// Should not panic
	kv.FreeAll()
}

func TestFreePagedKVPool_Nil(t *testing.T) {
	var p *PagedKVPool
	// Should not panic
	p.FreeAll()
}

func TestFreeGpuModel_Nil(t *testing.T) {
	var gm *GpuModel
	// Should not panic
	gm.FreeAll()
}

// --- AllocE tests ---

func TestAllocE_ZeroSize(t *testing.T) {
	// Real AllocE with size 0 should return (0, nil)
	buf, err := AllocE(0)
	if err != nil {
		t.Fatalf("AllocE(0) should succeed: %v", err)
	}
	if buf != 0 {
		t.Fatalf("AllocE(0) should return 0, got %d", buf)
	}
}

// --- Exhaustive field coverage tests ---
// These verify that every buffer field in the struct is actually
// allocated (non-zero) on success, catching any field that was
// missed in the allocation function.

func TestNewGpuRunState_AllFieldsPopulated(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	rs, err := NewGpuRunState(128, 128, 64, 256, 1000)
	if err != nil {
		t.Fatal(err)
	}
	fields := map[string]Buf{
		"X": rs.X, "XNorm": rs.XNorm, "Q": rs.Q, "K": rs.K, "V": rs.V,
		"AttnOut": rs.AttnOut, "AttnProj": rs.AttnProj,
		"FFNIn": rs.FFNIn, "FFNNorm": rs.FFNNorm,
		"Gate": rs.Gate, "Up": rs.Up, "Hidden": rs.Hidden,
		"FFNOut": rs.FFNOut, "Logits": rs.Logits,
	}
	for name, buf := range fields {
		if buf == 0 {
			t.Errorf("field %s not allocated", name)
		}
	}
}

func TestNewGpuBatchState_AllFieldsPopulated(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	bs, err := NewGpuBatchState(32, 128, 128, 64, 256)
	if err != nil {
		t.Fatal(err)
	}
	fields := map[string]Buf{
		"X": bs.X, "XNorm": bs.XNorm, "Q": bs.Q, "K": bs.K, "V": bs.V,
		"AttnOut": bs.AttnOut, "AttnProj": bs.AttnProj,
		"FFNIn": bs.FFNIn, "FFNNorm": bs.FFNNorm,
		"Gate": bs.Gate, "Up": bs.Up, "Hidden": bs.Hidden,
		"FFNOut": bs.FFNOut,
	}
	for name, buf := range fields {
		if buf == 0 {
			t.Errorf("field %s not allocated", name)
		}
	}
}

func TestAllocSSMScratch_AllFieldsPopulated(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	rs := &GpuRunState{}
	if err := rs.AllocSSMScratch(1024, 512, 32); err != nil {
		t.Fatal(err)
	}
	fields := map[string]Buf{
		"SSMQKV": rs.SSMQKV, "SSMZ": rs.SSMZ,
		"SSMAlpha": rs.SSMAlpha, "SSMBeta": rs.SSMBeta, "SSMY": rs.SSMY,
	}
	for name, buf := range fields {
		if buf == 0 {
			t.Errorf("field %s not allocated", name)
		}
	}
}

func TestAllocGatedQScratch_AllFieldsPopulated(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	rs := &GpuRunState{}
	if err := rs.AllocGatedQScratch(512); err != nil {
		t.Fatal(err)
	}
	if rs.QFull == 0 {
		t.Error("QFull not allocated")
	}
	if rs.QGate == 0 {
		t.Error("QGate not allocated")
	}
}

func TestAllocSSMBatch_AllFieldsPopulated(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	bs := &GpuBatchState{}
	if err := bs.AllocSSMBatch(32, 1024, 512, 32); err != nil {
		t.Fatal(err)
	}
	fields := map[string]Buf{
		"SSMQKV": bs.SSMQKV, "SSMZ": bs.SSMZ,
		"SSMAlpha": bs.SSMAlpha, "SSMBeta": bs.SSMBeta, "SSMY": bs.SSMY,
	}
	for name, buf := range fields {
		if buf == 0 {
			t.Errorf("field %s not allocated", name)
		}
	}
}

func TestAllocGatedQBatch_AllFieldsPopulated(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	bs := &GpuBatchState{}
	if err := bs.AllocGatedQBatch(32, 512); err != nil {
		t.Fatal(err)
	}
	if bs.QFull == 0 {
		t.Error("QFull not allocated")
	}
	if bs.QGate == 0 {
		t.Error("QGate not allocated")
	}
}

// --- Allocation count tests ---
// Verify the exact number of allocations each function makes,
// to catch accidentally added/removed allocations.

func TestNewGpuRunState_ExactAllocCount(t *testing.T) {
	// NewGpuRunState should make exactly 14 allocations
	var count int64
	withFakeAlloc(t, func(size uint64) (Buf, error) {
		i := atomic.AddInt64(&count, 1)
		return Buf(i * 0x1000), nil
	})

	_, err := NewGpuRunState(128, 128, 64, 256, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if count != 14 {
		t.Fatalf("expected 14 allocations, got %d", count)
	}
}

func TestNewGpuBatchState_ExactAllocCount(t *testing.T) {
	// NewGpuBatchState should make exactly 13 allocations
	var count int64
	withFakeAlloc(t, func(size uint64) (Buf, error) {
		i := atomic.AddInt64(&count, 1)
		return Buf(i * 0x1000), nil
	})

	_, err := NewGpuBatchState(32, 128, 128, 64, 256)
	if err != nil {
		t.Fatal(err)
	}
	if count != 13 {
		t.Fatalf("expected 13 allocations, got %d", count)
	}
}

func TestAllocSSMScratch_ExactAllocCount(t *testing.T) {
	var count int64
	withFakeAlloc(t, func(size uint64) (Buf, error) {
		i := atomic.AddInt64(&count, 1)
		return Buf(i * 0x1000), nil
	})

	rs := &GpuRunState{}
	_ = rs.AllocSSMScratch(1024, 512, 32)
	if count != 5 {
		t.Fatalf("expected 5 allocations, got %d", count)
	}
}

func TestAllocGatedQScratch_ExactAllocCount(t *testing.T) {
	var count int64
	withFakeAlloc(t, func(size uint64) (Buf, error) {
		i := atomic.AddInt64(&count, 1)
		return Buf(i * 0x1000), nil
	})

	rs := &GpuRunState{}
	_ = rs.AllocGatedQScratch(512)
	if count != 2 {
		t.Fatalf("expected 2 allocations, got %d", count)
	}
}

func TestNewGpuKVCache_ExactAllocCount(t *testing.T) {
	var count int64
	withFakeAlloc(t, func(size uint64) (Buf, error) {
		i := atomic.AddInt64(&count, 1)
		return Buf(i * 0x1000), nil
	})

	_, err := NewGpuKVCache(4, 4, 512, 128, nil)
	if err != nil {
		t.Fatal(err)
	}
	// 4 layers × 2 (key+val) = 8
	if count != 8 {
		t.Fatalf("expected 8 allocations, got %d", count)
	}
}

func TestNewGpuKVCache_NeedsKVAllocCount(t *testing.T) {
	var count int64
	withFakeAlloc(t, func(size uint64) (Buf, error) {
		i := atomic.AddInt64(&count, 1)
		return Buf(i * 0x1000), nil
	})

	needsKV := []bool{true, false, true, false}
	_, err := NewGpuKVCache(4, 4, 512, 128, needsKV)
	if err != nil {
		t.Fatal(err)
	}
	// Only 2 layers need KV × 2 = 4
	if count != 4 {
		t.Fatalf("expected 4 allocations with needsKV mask, got %d", count)
	}
}

func TestNewPagedKVPool_ExactAllocCount(t *testing.T) {
	var count int64
	withFakeAlloc(t, func(size uint64) (Buf, error) {
		i := atomic.AddInt64(&count, 1)
		return Buf(i * 0x1000), nil
	})

	_, err := NewPagedKVPool(3, 64, 8, 32)
	if err != nil {
		t.Fatal(err)
	}
	// 3 layers × 2 = 6
	if count != 6 {
		t.Fatalf("expected 6 allocations, got %d", count)
	}
}

// --- Error message quality tests ---

func TestOOM_ErrorMessages_ContainContext(t *testing.T) {
	withFakeAlloc(t, fakeAllocNever())

	_, err := NewGpuRunState(128, 128, 64, 256, 1000)
	if err == nil {
		t.Fatal("expected error")
	}
	// Error should wrap the function name for traceability
	if !strings.Contains(err.Error(), "NewGpuRunState") {
		t.Errorf("RunState error should mention function name: %v", err)
	}

	_, err = NewGpuKVCache(4, 4, 512, 128, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "layer") {
		t.Errorf("KVCache error should mention layer: %v", err)
	}

	_, err = NewGpuBatchState(32, 128, 128, 64, 256)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "npos=32") {
		t.Errorf("BatchState error should mention npos: %v", err)
	}
}

// --- Concurrency safety of allocChecker ---
// allocChecker is inherently single-threaded (no goroutines),
// but verify the test hook mechanism works safely.

func TestAllocFn_Restoration(t *testing.T) {
	original := allocFn
	withFakeAlloc(t, fakeAllocNever())
	// After withFakeAlloc's cleanup runs, allocFn should be restored.
	// We can't check cleanup during the test, but verify it's set correctly.
	if allocFn == nil {
		t.Fatal("allocFn should not be nil")
	}
	// Restoration happens on t.Cleanup, which runs after the test function.
	_ = original
}

// --- Boundary conditions ---

func TestNewGpuKVCache_gpuLayersExceedsTotalLayers(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	// Request 8 GPU layers but only 4 total
	kv, err := NewGpuKVCache(4, 8, 512, 128, nil)
	if err != nil {
		t.Fatalf("should handle gpuLayers > totalLayers: %v", err)
	}
	// Should only allocate 4 (min of 4, 8)
	for l := 0; l < 4; l++ {
		if kv.KeyBufs[l] == 0 {
			t.Errorf("layer %d should be allocated", l)
		}
	}
}

func TestNewGpuKVCache_EmptyNeedsKV(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	// All layers marked as not needing KV
	needsKV := []bool{false, false, false, false}
	kv, err := NewGpuKVCache(4, 4, 512, 128, needsKV)
	if err != nil {
		t.Fatalf("should succeed with no KV needed: %v", err)
	}
	for l := 0; l < 4; l++ {
		if kv.KeyBufs[l] != 0 || kv.ValBufs[l] != 0 {
			t.Errorf("layer %d should have zero bufs", l)
		}
	}
}

// --- Regression test: allocation failure should not leave partial state ---

func TestNewGpuRunState_NoPartialState(t *testing.T) {
	// Fail at allocation 7 (middle). The returned RunState should be nil,
	// and FreeAll should have been called to clean up the first 6 allocations.
	withFakeAlloc(t, fakeAllocN(6))

	rs, err := NewGpuRunState(128, 128, 64, 256, 1000)
	if err == nil {
		t.Fatal("expected error")
	}
	if rs != nil {
		// This is the critical check — we MUST NOT return partial state
		// that would later cause Vulkan device lost when used with zero bufs.
		t.Fatal("CRITICAL: partial RunState returned — this causes Vulkan device lost!")
	}
}

func TestNewGpuBatchState_NoPartialState(t *testing.T) {
	withFakeAlloc(t, fakeAllocN(5))

	bs, err := NewGpuBatchState(32, 128, 128, 64, 256)
	if err == nil {
		t.Fatal("expected error")
	}
	if bs != nil {
		t.Fatal("CRITICAL: partial BatchState returned — this causes Vulkan device lost!")
	}
}

func TestNewGpuKVCache_NoPartialState(t *testing.T) {
	withFakeAlloc(t, fakeAllocN(3))

	kv, err := NewGpuKVCache(4, 4, 512, 128, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if kv != nil {
		t.Fatal("CRITICAL: partial KVCache returned — this causes Vulkan device lost!")
	}
}

func TestNewPagedKVPool_NoPartialState(t *testing.T) {
	withFakeAlloc(t, fakeAllocN(1))

	pool, err := NewPagedKVPool(4, 128, 16, 64)
	if err == nil {
		t.Fatal("expected error")
	}
	if pool != nil {
		t.Fatal("CRITICAL: partial PagedKVPool returned — this causes Vulkan device lost!")
	}
}

// --- Integration-level: multiple alloc phases ---

func TestFullAllocationSequence_Success(t *testing.T) {
	// Simulate the pipeline's full allocation sequence:
	// 1. NewGpuRunState (14 allocs)
	// 2. NewGpuKVCache  (8 allocs for 4 layers)
	// 3. AllocSSMScratch (5 allocs)
	// 4. AllocGatedQScratch (2 allocs)
	// Total: 29 allocs
	withFakeAlloc(t, fakeAllocAlways())

	rs, err := NewGpuRunState(128, 128, 64, 256, 1000)
	if err != nil {
		t.Fatalf("RunState: %v", err)
	}
	kv, err := NewGpuKVCache(4, 4, 512, 128, nil)
	if err != nil {
		t.Fatalf("KVCache: %v", err)
	}
	if err := rs.AllocSSMScratch(1024, 512, 32); err != nil {
		t.Fatalf("SSMScratch: %v", err)
	}
	if err := rs.AllocGatedQScratch(128); err != nil {
		t.Fatalf("GatedQScratch: %v", err)
	}

	// All should be populated
	if rs.X == 0 || rs.SSMQKV == 0 || rs.QFull == 0 {
		t.Fatal("expected all fields populated")
	}
	if kv.KeyBufs[0] == 0 {
		t.Fatal("expected KV populated")
	}
}

func TestFullAllocationSequence_OOMAtKV(t *testing.T) {
	// RunState succeeds (14 allocs), then KVCache fails at 17th total alloc
	withFakeAlloc(t, fakeAllocN(16))

	rs, err := NewGpuRunState(128, 128, 64, 256, 1000)
	if err != nil {
		t.Fatalf("RunState should succeed: %v", err)
	}
	if rs == nil {
		t.Fatal("RunState should not be nil")
	}

	_, err = NewGpuKVCache(4, 4, 512, 128, nil)
	if err == nil {
		t.Fatal("KVCache should fail")
	}
}

func TestFullAllocationSequence_OOMAtSSM(t *testing.T) {
	// RunState (14) + KVCache (8) = 22 succeed, then SSMScratch fails
	withFakeAlloc(t, fakeAllocN(22))

	rs, err := NewGpuRunState(128, 128, 64, 256, 1000)
	if err != nil {
		t.Fatalf("RunState: %v", err)
	}
	_, err = NewGpuKVCache(4, 4, 512, 128, nil)
	if err != nil {
		t.Fatalf("KVCache: %v", err)
	}
	err = rs.AllocSSMScratch(1024, 512, 32)
	if err == nil {
		t.Fatal("SSMScratch should fail")
	}
	if !strings.Contains(err.Error(), "AllocSSMScratch") {
		t.Fatalf("error should mention AllocSSMScratch: %v", err)
	}
}

// --- Batch prefill allocation at runtime ---

func TestBatchStateReallocation_LargerSize(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	// First allocation at batch size 32
	bs1, err := NewGpuBatchState(32, 128, 128, 64, 256)
	if err != nil {
		t.Fatal(err)
	}
	if bs1.Npos != 32 {
		t.Fatalf("expected Npos=32, got %d", bs1.Npos)
	}

	// "Free" and reallocate at batch size 64
	bs1.Free()
	bs2, err := NewGpuBatchState(64, 128, 128, 64, 256)
	if err != nil {
		t.Fatal(err)
	}
	if bs2.Npos != 64 {
		t.Fatalf("expected Npos=64, got %d", bs2.Npos)
	}
}

func TestBatchStateWithSSMAndGatedQ_Success(t *testing.T) {
	withFakeAlloc(t, fakeAllocAlways())

	bs, err := NewGpuBatchState(32, 128, 128, 64, 256)
	if err != nil {
		t.Fatal(err)
	}
	if err := bs.AllocGatedQBatch(32, 128); err != nil {
		t.Fatal(err)
	}
	if err := bs.AllocSSMBatch(32, 1024, 512, 32); err != nil {
		t.Fatal(err)
	}
	// Verify both SSM and GatedQ fields populated
	if bs.QFull == 0 || bs.SSMQKV == 0 {
		t.Fatal("expected all batch fields populated")
	}
}

func TestBatchStateWithSSMAndGatedQ_OOMAtGatedQ(t *testing.T) {
	// 13 allocs for BatchState, then GatedQ fails
	withFakeAlloc(t, fakeAllocN(13))

	bs, err := NewGpuBatchState(32, 128, 128, 64, 256)
	if err != nil {
		t.Fatal(err)
	}
	err = bs.AllocGatedQBatch(32, 128)
	if err == nil {
		t.Fatal("expected OOM at GatedQ")
	}
}
