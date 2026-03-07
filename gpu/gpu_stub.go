//go:build !vulkan || !cgo

package gpu

import "fmt"

type Buf = uint64

var errNoGPU = fmt.Errorf("gpu: not compiled with vulkan support (use -tags vulkan)")

func Init() error           { return errNoGPU }
func Shutdown()             {}
func IsInitialized() bool   { return false }
func DeviceName() string    { return "none" }
func VRAMBytes() uint64     { return 0 }
func Alloc(uint64) Buf      { return 0 }
func Free(Buf)              {}
func Upload(Buf, []byte) error             { return errNoGPU }
func UploadF32(Buf, []float32) error       { return errNoGPU }
func Download(Buf, []byte) error           { return errNoGPU }
func DownloadF32(Buf, []float32) error     { return errNoGPU }
func MatVec(out, w, x Buf, rows, cols int, qtype uint32) error { return errNoGPU }
func RMSNorm(out, x, w Buf, n int, eps float32) error          { return errNoGPU }
func Softmax(Buf, int) error               { return errNoGPU }
func RoPE(q, k Buf, nh, nkv, hd, pos int, fb float32) error   { return errNoGPU }
func SwiGLU(out, gate, up Buf, n int) error { return errNoGPU }
func GeGLU(out, gate, up Buf, n int) error  { return errNoGPU }
func GELU(Buf, int) error                   { return errNoGPU }
func Add(out, a, b Buf, n int) error        { return errNoGPU }
func Scale(Buf, float32, int) error         { return errNoGPU }
func KVStore(kc, vc, k, v Buf, pos, kvDim int) error { return errNoGPU }
func Sync()                                 {}
func BeginBatch()                            {}
func EndBatch()                              {}
