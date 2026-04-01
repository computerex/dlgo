//go:build cgo && vulkan

package gpu

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// f32ToBF16 converts a float32 to BF16 (truncate lower 16 bits).
func f32ToBF16(f float32) uint16 {
	bits := math.Float32bits(f)
	return uint16(bits >> 16)
}

// bf16ToF32 converts a BF16 to float32.
func bf16ToF32(b uint16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}

// packBF16Pairs packs BF16 values into uint32 pairs (little-endian: low 16 = first, high 16 = second).
func packBF16Pairs(vals []float32) []uint32 {
	n := len(vals)
	packed := make([]uint32, (n+1)/2)
	for i := 0; i < n; i += 2 {
		lo := f32ToBF16(vals[i])
		var hi uint16
		if i+1 < n {
			hi = f32ToBF16(vals[i+1])
		}
		packed[i/2] = uint32(lo) | (uint32(hi) << 16)
	}
	return packed
}

// cpuBF16MatVec computes out[r] = dot(W[r,:], x) where W is BF16.
func cpuBF16MatVec(w []float32, x []float32, rows, cols int) []float32 {
	out := make([]float32, rows)
	for r := 0; r < rows; r++ {
		var sum float64
		for c := 0; c < cols; c++ {
			// Convert to BF16 and back to simulate quantization loss
			wBF16 := bf16ToF32(f32ToBF16(w[r*cols+c]))
			sum += float64(wBF16) * float64(x[c])
		}
		out[r] = float32(sum)
	}
	return out
}

func TestBF16MatVec(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer Shutdown()

	t.Logf("GPU: %s, VRAM: %.0f MB", DeviceName(), float64(VRAMBytes())/(1024*1024))

	// Small matrix: 64 rows x 256 cols (well within safe limits)
	const rows = 64
	const cols = 256
	const npos = 1

	rng := rand.New(rand.NewSource(42))

	// Generate random weights and input
	wF32 := make([]float32, rows*cols)
	x := make([]float32, cols)
	for i := range wF32 {
		wF32[i] = rng.Float32()*2 - 1 // [-1, 1]
	}
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	// CPU reference (with BF16 quantization applied)
	cpuOut := cpuBF16MatVec(wF32, x, rows, cols)

	// Pack weights as BF16
	wBF16 := packBF16Pairs(wF32)

	// Allocate GPU buffers
	wBufSize := uint64(len(wBF16)) * 4 // uint32 = 4 bytes
	wBuf := Alloc(wBufSize)
	if wBuf == 0 {
		t.Fatal("Failed to allocate weight buffer")
	}
	defer Free(wBuf)

	xBuf := Alloc(uint64(cols) * 4)
	if xBuf == 0 {
		t.Fatal("Failed to allocate input buffer")
	}
	defer Free(xBuf)

	outBuf := Alloc(uint64(rows) * 4)
	if outBuf == 0 {
		t.Fatal("Failed to allocate output buffer")
	}
	defer Free(outBuf)

	// Upload
	wBytes := unsafe.Slice((*byte)(unsafe.Pointer(&wBF16[0])), len(wBF16)*4)
	if err := Upload(wBuf, wBytes); err != nil {
		t.Fatalf("Upload weights: %v", err)
	}
	if err := UploadF32(xBuf, x); err != nil {
		t.Fatalf("Upload input: %v", err)
	}

	// Run GPU matvec
	BeginBatch()
	if err := BatchMatVec(outBuf, wBuf, xBuf, rows, cols, npos, 30); err != nil { // 30 = QTYPE_BF16
		EndBatch()
		t.Fatalf("BatchMatVec: %v", err)
	}
	EndBatch()

	// Download result
	gpuOut := make([]float32, rows)
	if err := DownloadF32(outBuf, gpuOut); err != nil {
		t.Fatalf("Download: %v", err)
	}

	// Compare
	var maxErr float64
	var sumSqErr float64
	for r := 0; r < rows; r++ {
		diff := float64(gpuOut[r]) - float64(cpuOut[r])
		sumSqErr += diff * diff
		if abs := math.Abs(diff); abs > maxErr {
			maxErr = abs
		}
	}
	rmse := math.Sqrt(sumSqErr / float64(rows))
	t.Logf("BF16 MatVec: RMSE=%.6f, MaxErr=%.6f", rmse, maxErr)

	if rmse > 0.01 {
		t.Errorf("RMSE too high: %.6f (expected < 0.01)", rmse)
		// Print first few mismatches
		for r := 0; r < rows && r < 5; r++ {
			t.Logf("  row %d: CPU=%.6f GPU=%.6f diff=%.6f", r, cpuOut[r], gpuOut[r], gpuOut[r]-cpuOut[r])
		}
	}
}

func TestBF16BatchMatVec(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer Shutdown()

	// Test batch (npos > 1) to validate groups_y dispatch
	const rows = 32
	const cols = 256
	const npos = 4

	rng := rand.New(rand.NewSource(123))

	wF32 := make([]float32, rows*cols)
	xAll := make([]float32, npos*cols)
	for i := range wF32 {
		wF32[i] = rng.Float32()*2 - 1
	}
	for i := range xAll {
		xAll[i] = rng.Float32()*2 - 1
	}

	// CPU reference for each position
	cpuOut := make([]float32, npos*rows)
	for p := 0; p < npos; p++ {
		ref := cpuBF16MatVec(wF32, xAll[p*cols:(p+1)*cols], rows, cols)
		copy(cpuOut[p*rows:(p+1)*rows], ref)
	}

	// Pack weights
	wBF16 := packBF16Pairs(wF32)

	wBuf := Alloc(uint64(len(wBF16)) * 4)
	if wBuf == 0 {
		t.Fatal("Failed to allocate weight buffer")
	}
	defer Free(wBuf)

	xBuf := Alloc(uint64(npos*cols) * 4)
	if xBuf == 0 {
		t.Fatal("Failed to allocate input buffer")
	}
	defer Free(xBuf)

	outBuf := Alloc(uint64(npos*rows) * 4)
	if outBuf == 0 {
		t.Fatal("Failed to allocate output buffer")
	}
	defer Free(outBuf)

	wBytes := unsafe.Slice((*byte)(unsafe.Pointer(&wBF16[0])), len(wBF16)*4)
	if err := Upload(wBuf, wBytes); err != nil {
		t.Fatal(err)
	}
	if err := UploadF32(xBuf, xAll); err != nil {
		t.Fatal(err)
	}

	BeginBatch()
	if err := BatchMatVec(outBuf, wBuf, xBuf, rows, cols, npos, 30); err != nil {
		EndBatch()
		t.Fatal(err)
	}
	EndBatch()

	gpuOut := make([]float32, npos*rows)
	if err := DownloadF32(outBuf, gpuOut); err != nil {
		t.Fatal(err)
	}

	var maxErr float64
	var sumSqErr float64
	for i := range cpuOut {
		diff := float64(gpuOut[i]) - float64(cpuOut[i])
		sumSqErr += diff * diff
		if abs := math.Abs(diff); abs > maxErr {
			maxErr = abs
		}
	}
	rmse := math.Sqrt(sumSqErr / float64(len(cpuOut)))
	t.Logf("BF16 BatchMatVec (npos=%d): RMSE=%.6f, MaxErr=%.6f", npos, rmse, maxErr)

	if rmse > 0.01 {
		t.Errorf("RMSE too high: %.6f", rmse)
	}
}
