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
func RMSNormHeads(data, w Buf, nh, hd int, eps float32) error { return errNoGPU }
func Softmax(Buf, int) error               { return errNoGPU }
func RoPE(q, k Buf, nh, nkv, hd, rd, pos int, fb float32, neox bool) error { return errNoGPU }
func SwiGLU(out, gate, up Buf, n int) error { return errNoGPU }
func GeGLU(out, gate, up Buf, n int) error  { return errNoGPU }
func GELU(Buf, int) error                   { return errNoGPU }
func Add(out, a, b Buf, n int) error        { return errNoGPU }
func Scale(Buf, float32, int) error         { return errNoGPU }
func Attention(out, q, kc, vc Buf, nh, nkv, hd, kvd, sl int, s float32) error { return errNoGPU }
func KVStore(kc, vc, k, v Buf, pos, kvDim int) error { return errNoGPU }
func Sync()                                 {}
func HasDp4a() bool                          { return false }
func QuantizeQ8_1(q, f Buf, n int) error     { return errNoGPU }
func MatVecOffsetDp4a(o Buf, oo int, w Buf, wo int, q Buf, r, c int, t uint32) error { return errNoGPU }
func MoEMatVecDp4a(o, w, q, i Buf, r, c int, t uint32, es, bo, si, nu int) error    { return errNoGPU }
func MoEAccumulate(o, e, w, b, i Buf, d, n int, h bool) error                      { return errNoGPU }
func SwiGLUAt(o, g, u Buf, oo, go_, uo, n int) error                               { return errNoGPU }
func SwiGLU_OAI_At(o, g, u Buf, oo, go_, uo, n int, a, l float32) error            { return errNoGPU }
func QuantizeQ8_1At(q Buf, qo int, f Buf, fo, n int) error                         { return errNoGPU }
func BeginBatch()                            {}
func EndBatch()                              {}
func Barrier()                               {}
func AddRMSNorm(no, so, a, b, w Buf, n int, eps float32) error { return errNoGPU }

type LayerConf struct{}

func NewLayerConf() *LayerConf                                            { return nil }
func (lc *LayerConf) SetScratch(x, xn, q, k, v, ao, ap, fn, fi, g, u, h, fo Buf) {}
func (lc *LayerConf) SetAttn(an Buf, wq, wk, wv, wo *GpuTensor, bq, bk, bv, qn, kn Buf) {}
func (lc *LayerConf) SetFFN(fn Buf, gate, up, down *GpuTensor, pan, pfn Buf) {}
func (lc *LayerConf) SetFFNMoE(fn Buf, pan Buf)                          {}
func (lc *LayerConf) SetKV(kc, vc Buf)                                   {}
func (lc *LayerConf) SetCoreType(ct int)                                 {}
func (lc *LayerConf) SetAttnNormOnly(an Buf)                             {}
func (lc *LayerConf) SetConfig(d, hd, nh, nkv, kd int, e, f float32, rd int, n bool, ft, rt int) {}
func (lc *LayerConf) SetDP4A(q Buf) {}
func ForwardLayer(lc *LayerConf, pos, sl int, s float32, nan Buf) error   { return errNoGPU }
