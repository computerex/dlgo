//go:build !cgo || !windows

// Package cuda (stub) — CUDA backend is only built on Windows with cgo.
package cuda

import "errors"

// Buffer is a device (VRAM) allocation handle.
type Buffer = uint64

// Kernels mirrors the real backend's handle set (unused in the stub).
type Kernels struct {
	VectorAdd, KVStoreF16, RMSNorm, SiluMul, SoftmaxRow          uint64
	LayerNorm, RMSNormHeads, Add, Scale, AddRMSNorm, RoPE        uint64
	AttentionF16                                                 uint64
	AttentionSplitF16, AttentionCombine                          uint64
	GemvQ4K, GemvQ6K, GemvF32, GemvF16                           uint64
}

var errNoCUDA = errors.New("cuda: backend not built (requires windows + cgo)")

func Init() error        { return errNoCUDA }
func Shutdown()          {}
func Available() bool    { return false }
func LastError() string  { return errNoCUDA.Error() }
func DeviceName() string { return "" }
func SMs() int           { return 0 }
func TotalMem() uint64   { return 0 }
func FreeMem() uint64    { return 0 }
func Allocated() uint64  { return 0 }

func Alloc(bytes uint64) (Buffer, error)             { return 0, errNoCUDA }
func AllocHost(bytes uint64) (uintptr, error)        { return 0, errNoCUDA }
func Free(b Buffer) error                            { return errNoCUDA }
func FreeHost(p uintptr) error                       { return errNoCUDA }
func Upload(dst Buffer, src []byte) error            { return errNoCUDA }
func UploadAt(dst Buffer, off uint64, src []byte) error { return errNoCUDA }
func Download(dst []byte, src Buffer) error          { return errNoCUDA }
func Memset8(dst Buffer, value byte, n uint64) error { return errNoCUDA }
func Sync() error                                    { return errNoCUDA }

func VectorAdd(a, b, out Buffer, n int) error                               { return errNoCUDA }
func KVStoreF16(kIn, vIn, kOut, vOut Buffer, pos, kvDim, ringCap int) error { return errNoCUDA }
func RMSNorm(x, w, out Buffer, n int, eps float32) error                    { return errNoCUDA }
func SiluMul(gate, up, out Buffer, n int) error                             { return errNoCUDA }
func SoftmaxRow(x Buffer, n int) error                                      { return errNoCUDA }
func LayerNorm(x, w, b, out Buffer, n int, eps float32) error               { return errNoCUDA }
func RMSNormHeads(data, w Buffer, numHeads, headDim int, eps float32) error { return errNoCUDA }
func Add(a, b, out Buffer, n int) error                                     { return errNoCUDA }
func Scale(buf Buffer, s float32, n int) error                              { return errNoCUDA }
func AddRMSNorm(a, b, w, sum, out Buffer, n int, eps float32) error         { return errNoCUDA }
func RoPE(q, k, cosT, sinT Buffer, numHeads, numKVHeads, headDim, ropeDim, pos int, neox bool) error {
	return errNoCUDA
}
func AttentionF16(q, kc, vc, out Buffer, numHeads, numKVHeads, headDim, kvDim, seqLen, winStart int, scale, softcap float32, kvCap int) error {
	return errNoCUDA
}

// CUDA graphs are only available in the real (windows+cgo) backend.
func GraphReady() bool { return false }
func GraphCapture() error {
	return errNoCUDA
}
func GraphCaptureEnd() error { return errNoCUDA }
func GraphReplay(pos, seqLen, winStart, kvCap int) error {
	return errNoCUDA
}
func GraphDiscard() {}
func GEMVQ4K(W, x, out Buffer, rows, cols int) error { return errNoCUDA }
func GEMVQ6K(W, x, out Buffer, rows, cols int) error { return errNoCUDA }
func GEMVF32(W, x, out Buffer, rows, cols int) error { return errNoCUDA }
func GEMVF16(W, x, out Buffer, rows, cols int) error { return errNoCUDA }
