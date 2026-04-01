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
func VRAMFreeBytes() uint64 { return 0 }
func AllocatedBytes() uint64 { return 0 }
func Alloc(uint64) Buf                 { return 0 }
func AllocE(n uint64) (Buf, error)     { return 0, errNoGPU }
func Free(Buf)              {}
func ResetBufferTable()     {}
func Upload(Buf, []byte) error             { return errNoGPU }
func UploadF32(Buf, []float32) error       { return errNoGPU }
func Download(Buf, []byte) error           { return errNoGPU }
func DownloadF32(Buf, []float32) error     { return errNoGPU }
func MatVec(out, w, x Buf, rows, cols int, qtype uint32) error { return errNoGPU }
func RMSNorm(out, x, w Buf, n int, eps float32) error          { return errNoGPU }
func LayerNorm(out, x, w, b Buf, n int, eps float32) error     { return errNoGPU }
func RMSNormHeads(data, w Buf, nh, hd int, eps float32) error { return errNoGPU }
func Softmax(Buf, int) error               { return errNoGPU }
func RoPE(q, k Buf, nh, nkv, hd, rd, pos int, fb float32, neox bool) error { return errNoGPU }
func SwiGLU(out, gate, up Buf, n int) error { return errNoGPU }
func GeGLU(out, gate, up Buf, n int) error  { return errNoGPU }
func GELU(Buf, int) error                   { return errNoGPU }
func Add(out, a, b Buf, n int) error        { return errNoGPU }
func Scale(Buf, float32, int) error         { return errNoGPU }
func Attention(out, q, kc, vc Buf, nh, nkv, hd, kvd, sl, sp int, s float32) error { return errNoGPU }
func KVStore(kc, vc, k, v Buf, pos, kvDim int) error { return errNoGPU }
func Sync()                                 {}
func HasDp4a() bool                          { return false }
func QuantizeQ8_1(q, f Buf, n int) error     { return errNoGPU }
func MatVecOffsetDp4a(o Buf, oo int, w Buf, wo int, q Buf, r, c int, t uint32) error { return errNoGPU }
func MoEMatVecDp4a(o, w, q, i Buf, r, c int, t uint32, es, bo, si, nu int) error    { return errNoGPU }
func MoEAccumulate(o, e, w, b, i Buf, d, n int, h bool) error                      { return errNoGPU }
func SwiGLU_OAI_Bias_MoE(o, g, u, gb, ub, i Buf, tn int, a, l float32, ed int) error { return errNoGPU }
func MoEBiasAdd(d, b, i Buf, ed, nu int) error                                    { return errNoGPU }
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
func (lc *LayerConf) SetAttn(an Buf, wq, wk, wv, wo *GpuTensor, bq, bk, bv, bo Buf, qn, kn Buf) {}
func (lc *LayerConf) SetAttnSinks(s Buf) {}
func (lc *LayerConf) SetSlidingWindow(w int) {}
func (lc *LayerConf) SetFFN(fn Buf, gate, up, down *GpuTensor, pan, pfn Buf)                     {}
func (lc *LayerConf) SetFFNMoE(fn Buf, pan Buf)                          {}
func (lc *LayerConf) SetKV(kc, vc Buf)                                   {}
func (lc *LayerConf) SetCoreType(ct int)                                 {}
func (lc *LayerConf) SetAttnNormOnly(an Buf)                             {}
func (lc *LayerConf) SetConfig(d, hd, nh, nkv, kd int, e, f float32, rd int, n bool, ft, rt int) {}
func (lc *LayerConf) SetDP4A(q Buf) {}
func ForwardLayer(lc *LayerConf, pos, sl int, s float32, nan Buf) error   { return errNoGPU }

// VAE-specific stubs
func Conv2dF32(out, in, weight, bias Buf, inCh, H, W, kH, kW, padH, padW, stride, outH, outW, outCh int) error { return errNoGPU }
func GroupNorm(out, in, weight, bias Buf, channels, spatialSize, numGroups int, eps float32) error { return errNoGPU }
func SiLU(data Buf, n int) error { return errNoGPU }
func UpsampleNearest(out, in Buf, channels, H, W int) error { return errNoGPU }
func SpatialAttention(out, q, k, v Buf, channels, spatial int, scale float32) error { return errNoGPU }

type MoEFFNConf struct{}

func NewMoEFFNConf() *MoEFFNConf                                                       { return nil }
func (mc *MoEFFNConf) SetScratch(fn, fo, ml, ti, tw, q8, gs, us, q8d, os Buf)          {}
func (mc *MoEFFNConf) SetRouter(w Buf, r, c, t int, b Buf)                             {}
func (mc *MoEFFNConf) SetExperts(gw Buf, gt, gs, gb int, uw Buf, ut, us, ub int, dw Buf, dt, ds int) {}
func (mc *MoEFFNConf) SetBiases(gb, ub, db Buf)                                        {}
func (mc *MoEFFNConf) SetConfig(d, ed, ne, nu, gf int, wn bool, ws float32, oai bool, a, l float32) {}
func ForwardMoEFFN_C(mc *MoEFFNConf) error                                             { return errNoGPU }
