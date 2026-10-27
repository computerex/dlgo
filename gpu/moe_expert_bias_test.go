//go:build cgo && vulkan

package gpu

import (
	"math"
	"testing"
)

// TestMoEExpertBiasAdd exercises the position-batched MoE expert-bias kernel in
// both of its modes, which the grouped and per-expert paths select between:
//
//	use_index == false: one expert covers every row, bias_base = expert*expDim
//	use_index == true : each row's expert comes from a GPU pair->expert table
//
// gpt-oss is the model this exists for (ffn_{gate,up,down}_exps.bias); without
// it the batched prefill cannot reproduce the per-token MoE.
func TestMoEExpertBiasAdd(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}

	const (
		expDim  = 8
		nRows   = 5
		nExpert = 4
		n       = nRows * expDim
	)

	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i) * 0.25
	}
	bias := make([]float32, nExpert*expDim)
	for e := 0; e < nExpert; e++ {
		for i := 0; i < expDim; i++ {
			bias[e*expDim+i] = float32(100*(e+1) + i)
		}
	}

	dataBuf := Alloc(uint64(n * 4))
	biasBuf := Alloc(uint64(len(bias) * 4))
	idxBuf := Alloc(uint64(nRows * 4))
	if dataBuf == 0 || biasBuf == 0 || idxBuf == 0 {
		t.Fatalf("alloc failed")
	}
	defer Free(dataBuf)
	defer Free(biasBuf)
	defer Free(idxBuf)
	if err := UploadF32(biasBuf, bias); err != nil {
		t.Fatalf("upload bias: %v", err)
	}

	// --- Mode 1: one expert per dispatch (the per-expert loop) ---
	const e = 2
	if err := UploadF32(dataBuf, data); err != nil {
		t.Fatalf("upload data: %v", err)
	}
	if err := MoEExpertBiasAdd(dataBuf, biasBuf, idxBuf, n, expDim, false, e*expDim); err != nil {
		t.Fatalf("bias add (single expert): %v", err)
	}
	got := make([]float32, n)
	if err := DownloadF32(dataBuf, got); err != nil {
		t.Fatalf("download: %v", err)
	}
	for r := 0; r < nRows; r++ {
		for i := 0; i < expDim; i++ {
			want := data[r*expDim+i] + bias[e*expDim+i]
			if math.Abs(float64(got[r*expDim+i]-want)) > 1e-6 {
				t.Fatalf("single-expert row %d col %d: got %v want %v", r, i, got[r*expDim+i], want)
			}
		}
	}

	// --- Mode 2: per-row expert from the GPU table (the grouped path) ---
	rowExpert := []int32{0, 1, 3, 2, 1}
	if err := UploadI32(idxBuf, rowExpert); err != nil {
		t.Fatalf("upload idx: %v", err)
	}
	if err := UploadF32(dataBuf, data); err != nil {
		t.Fatalf("re-upload data: %v", err)
	}
	if err := MoEExpertBiasAdd(dataBuf, biasBuf, idxBuf, n, expDim, true, 0); err != nil {
		t.Fatalf("bias add (indexed): %v", err)
	}
	if err := DownloadF32(dataBuf, got); err != nil {
		t.Fatalf("download: %v", err)
	}
	for r := 0; r < nRows; r++ {
		re := int(rowExpert[r])
		for i := 0; i < expDim; i++ {
			want := data[r*expDim+i] + bias[re*expDim+i]
			if math.Abs(float64(got[r*expDim+i]-want)) > 1e-6 {
				t.Fatalf("indexed row %d col %d (expert %d): got %v want %v",
					r, i, re, got[r*expDim+i], want)
			}
		}
	}
}
