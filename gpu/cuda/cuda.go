//go:build cgo && windows

// Package cuda is dlgo's CUDA backend (Stage 1 core).
//
// It talks to the CUDA Driver API in nvcuda.dll at run time (no cudart link),
// loads nvcc-compiled code embedded at build time, and exposes the same kind of
// primitives the Vulkan backend does. Kernels are added incrementally.
//
// Thread model: a CUDA primary context is bound to the OS thread that makes it
// current, and Go goroutines migrate between OS threads. To avoid binding the
// context on an ever-growing set of threads (which crashes this driver after a
// couple of calls), every CUDA call is funnelled through one worker goroutine
// locked to a single OS thread for the life of the process. Public functions
// marshal arguments in Go, then run the cgo call on that worker.
package cuda

/*
#cgo windows CFLAGS: -O2 -I${SRCDIR} -DDLGO_CUDA_BUILD=13
#include <stdlib.h>
#include "cuda_shim.h"
*/
import "C"

import (
	_ "embed"
	"fmt"
	"math"
	"os"
	"strconv"
	"sync"
	"unsafe"
)

//go:embed kernels.cubin
var kernelImage string

// Buffer is a device (VRAM) allocation handle.
type Buffer = uint64

// Kernels holds resolved CUfunction handles for the loaded module.
type Kernels struct {
	VectorAdd, KVStoreF16, RMSNorm, SiluMul, SoftmaxRow, Probe7, HalfProbe uint64
	LayerNorm, RMSNormHeads, Add, Scale, AddRMSNorm, RoPE, AttentionF16    uint64
	AttentionSplitF16, AttentionCombine                                    uint64
	AttnSplitV2D2, AttnSplitV2D4                                          uint64
	GemvQ4K, GemvQ6K, GemvF32, GemvF16                                     uint64
	QuantizeX, GemvQ4KDp4a                                                 uint64
	GemvQ4KBc                                                              uint64
	F32ToHalf, GemvAffine4, GemvAffine8, GemmAffine4F16, GemmAffine8F16    uint64
	BatchRMSNorm, BatchAddRMSNorm, BatchRMSNormHeads, BatchRoPE            uint64
	BatchKVStoreF16, AttnPrefillF16                                        uint64
	CopyRegion                                                             uint64
}

var (
	kernels Kernels
	module  uint64
)

// cudaQ4KMode selects the Q4_K GEMV path. Default ("" or "dp4a") uses the
// block-cooperative int8 dp4a kernel; "float" forces the vectorized float path.
var cudaQ4KMode = os.Getenv("DLGO_CUDA_Q4K")

// do runs f on the caller's goroutine. Each CUDA entry point re-binds the
// primary context (see ensure_ctx in cuda_shim.c), so no thread pinning is
// needed here.

// do runs f inline. (An earlier version funnelled everything through a locked
// worker goroutine; that destabilised the driver, so calls run on the caller's
// thread and each entry re-binds the context.)
func do(f func()) { f() }

// Init initializes the driver, retains the primary context, and loads the
// embedded kernel image â€” all on the worker thread.
func Init() error {
	var err error
	do(func() {
		if C.dlcuda_init() != 0 {
			err = fmt.Errorf("cuda: init failed: %s", C.GoString(C.dlcuda_last_error()))
			return
		}
		if module != 0 {
			return
		}
		img := C.CString(kernelImage)
		module = uint64(C.dlcuda_load_module(img))
		C.free(unsafe.Pointer(img))
		if module == 0 {
			err = fmt.Errorf("cuda: load module failed: %s", C.GoString(C.dlcuda_last_error()))
			return
		}
		for _, kv := range []struct {
			n string
			p *uint64
		}{
			{"dl_vector_add", &kernels.VectorAdd},
			{"dl_kv_store_f16", &kernels.KVStoreF16},
			{"dl_rmsnorm_f32", &kernels.RMSNorm},
			{"dl_silu_mul", &kernels.SiluMul},
			{"dl_softmax_row", &kernels.SoftmaxRow},
			{"dl_probe7", &kernels.Probe7},
			{"dl_half_probe", &kernels.HalfProbe},
			{"dl_layernorm_f32", &kernels.LayerNorm},
			{"dl_rmsnorm_heads", &kernels.RMSNormHeads},
			{"dl_add", &kernels.Add},
			{"dl_scale", &kernels.Scale},
			{"dl_add_rmsnorm", &kernels.AddRMSNorm},
			{"dl_rope", &kernels.RoPE},
			{"dl_attention_f16", &kernels.AttentionF16},
			{"dl_attention_split_f16", &kernels.AttentionSplitF16},
			{"dl_attn_split_v2_d2", &kernels.AttnSplitV2D2},
			{"dl_attn_split_v2_d4", &kernels.AttnSplitV2D4},
			{"dl_attention_combine", &kernels.AttentionCombine},
			{"dl_gemv_q4k", &kernels.GemvQ4K},
			{"dl_gemv_q6k", &kernels.GemvQ6K},
			{"dl_gemv_f32", &kernels.GemvF32},
			{"dl_gemv_f16", &kernels.GemvF16},
			{"dl_quantize_x32", &kernels.QuantizeX},
			{"dl_gemv_q4k_dp4a", &kernels.GemvQ4KDp4a},
			{"dl_gemv_q4k_bc", &kernels.GemvQ4KBc},
			{"dl_f32_to_half", &kernels.F32ToHalf},
			{"dl_gemv_affine4", &kernels.GemvAffine4},
			{"dl_gemv_affine8", &kernels.GemvAffine8},
			{"dl_gemm_affine4_f16", &kernels.GemmAffine4F16},
			{"dl_gemm_affine8_f16", &kernels.GemmAffine8F16},
			{"dl_batch_rmsnorm", &kernels.BatchRMSNorm},
			{"dl_batch_add_rmsnorm", &kernels.BatchAddRMSNorm},
			{"dl_batch_rmsnorm_heads", &kernels.BatchRMSNormHeads},
			{"dl_batch_rope", &kernels.BatchRoPE},
			{"dl_batch_kv_store_f16", &kernels.BatchKVStoreF16},
			{"dl_attn_prefill_f16", &kernels.AttnPrefillF16},
			{"dl_copy_region", &kernels.CopyRegion},
		} {
			cn := C.CString(kv.n)
			h := uint64(C.dlcuda_get_func(C.ulonglong(module), cn))
			C.free(unsafe.Pointer(cn))
			if h == 0 {
				err = fmt.Errorf("cuda: missing kernel %q", kv.n)
				return
			}
			*kv.p = h
		}
	})
	return err
}

// Shutdown releases the driver context association.
func Shutdown() { do(func() { C.dlcuda_shutdown() }) }

// Available reports whether the driver initialized successfully.
func Available() bool { return C.dlcuda_available() != 0 }

// LastError returns the driver's last human-readable error string.
func LastError() string { return C.GoString(C.dlcuda_last_error()) }

// DeviceName returns the CUDA device name.
func DeviceName() string { return C.GoString(C.dlcuda_device_name()) }

// SMs returns the multiprocessor count.
func SMs() int { return int(C.dlcuda_sm_count()) }

// TotalMem returns total device-local VRAM in bytes.
func TotalMem() uint64 {
	var v uint64
	do(func() { v = uint64(C.dlcuda_total_mem()) })
	return v
}

// FreeMem returns current free device-local VRAM in bytes.
func FreeMem() uint64 {
	var v uint64
	do(func() { v = uint64(C.dlcuda_free_mem()) })
	return v
}

// Allocated returns the total live device allocation made through Alloc.
func Allocated() uint64 { return uint64(C.dlcuda_allocated()) }

// Alloc allocates bytes of device-local memory.
func Alloc(bytes uint64) (Buffer, error) {
	var p uint64
	var err error
	do(func() {
		p = uint64(C.dlcuda_alloc(C.ulonglong(bytes)))
		if p == 0 {
			err = fmt.Errorf("cuda: alloc %d bytes failed: %s", bytes, C.GoString(C.dlcuda_last_error()))
		}
	})
	return p, err
}

// AllocHost allocates page-locked host memory (the future RAM tier).
func AllocHost(bytes uint64) (uintptr, error) {
	var p uintptr
	var err error
	do(func() {
		p = uintptr(C.dlcuda_alloc_host(C.ulonglong(bytes)))
		if p == 0 {
			err = fmt.Errorf("cuda: alloc host failed: %s", C.GoString(C.dlcuda_last_error()))
		}
	})
	return p, err
}

// Free releases a device allocation.
func Free(b Buffer) error {
	var err error
	do(func() {
		if C.dlcuda_free(C.ulonglong(b)) != 0 {
			err = fmt.Errorf("cuda: free failed: %s", C.GoString(C.dlcuda_last_error()))
		}
	})
	return err
}

// FreeHost releases a pinned host allocation.
func FreeHost(p uintptr) error {
	var err error
	do(func() {
		if C.dlcuda_free_host(C.ulonglong(p)) != 0 {
			err = fmt.Errorf("cuda: free host failed: %s", C.GoString(C.dlcuda_last_error()))
		}
	})
	return err
}

// Upload copies src into device buffer dst (H2D, synchronous).
func Upload(dst Buffer, src []byte) error {
	if len(src) == 0 {
		return nil
	}
	var err error
	do(func() {
		if C.dlcuda_upload(C.ulonglong(dst), unsafe.Pointer(&src[0]), C.ulonglong(len(src))) != 0 {
			err = fmt.Errorf("cuda: upload failed: %s", C.GoString(C.dlcuda_last_error()))
		}
	})
	return err
}

// Download copies device buffer src into dst (D2H, synchronous).
func Download(dst []byte, src Buffer) error {
	if len(dst) == 0 {
		return nil
	}
	var err error
	do(func() {
		if C.dlcuda_download(unsafe.Pointer(&dst[0]), C.ulonglong(src), C.ulonglong(len(dst))) != 0 {
			err = fmt.Errorf("cuda: download failed: %s", C.GoString(C.dlcuda_last_error()))
		}
	})
	return err
}

// Memset8 fills n bytes at dst with value.
func Memset8(dst Buffer, value byte, n uint64) error {
	var err error
	do(func() {
		if C.dlcuda_memset8(C.ulonglong(dst), C.int(value), C.ulonglong(n)) != 0 {
			err = fmt.Errorf("cuda: memset failed: %s", C.GoString(C.dlcuda_last_error()))
		}
	})
	return err
}

// Sync blocks until all prior work on the context completes.
func Sync() error {
	var err error
	do(func() {
		if C.dlcuda_sync() != 0 {
			err = fmt.Errorf("cuda: sync failed: %s", C.GoString(C.dlcuda_last_error()))
		}
	})
	return err
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

func cdiv(a, b int) int { return (a + b - 1) / b }

func pow2Block(n, max int) int {
	b := 32
	for b < n && b < max {
		b <<= 1
	}
	if b > max {
		b = max
	}
	return b
}

func le64(b []byte, v uint64) {
	for i := 0; i < 8; i++ {
		b[i] = byte(v >> (8 * i))
	}
}
func le32(b []byte, v uint32) {
	for i := 0; i < 4; i++ {
		b[i] = byte(v >> (8 * i))
	}
}
func u64b(v uint64) []byte  { b := make([]byte, 8); le64(b, v); return b }
func i32b(v int) []byte     { b := make([]byte, 4); le32(b, uint32(int32(v))); return b }
func f32b(v float32) []byte { b := make([]byte, 4); le32(b, math.Float32bits(v)); return b }
func ptrb(b Buffer) []byte  { return u64b(uint64(b)) }

// launch marshals args into a reusable C buffer and dispatches.
//
// Hundreds of kernels fire per decoded token, so per-launch C.malloc/free calls
// dominated the CPU submission path on WDDM. We keep one growable C buffer and
// lay the launch out as: [void* param-array][arg0 bytes][arg1 bytes]...  The
// param-array entries point into the trailing value region (all C memory, so no
// Go pointers are handed to cgo). cuLaunchKernel copies the params synchronously
// at launch, so the buffer is safe to reuse immediately afterwards.
var (
	launchMu  sync.Mutex
	launchBuf unsafe.Pointer
	launchCap int
)

func launch(fn uint64, gx, bx int, args [][]byte) error {
	return launchN(fn, gx, 1, bx, 0, args)
}

// launchN dispatches with an explicit grid.y and a dynamic shared-memory size.
func launchN(fn uint64, gx, gy, bx, shared int, args [][]byte) error {
	n := len(args)
	if n == 0 {
		return fmt.Errorf("cuda: launch with no params")
	}
	need := n * 8
	for _, a := range args {
		need += len(a)
	}

	launchMu.Lock()
	defer launchMu.Unlock()

	if need > launchCap {
		if launchBuf != nil {
			C.free(launchBuf)
		}
		launchBuf = C.malloc(C.size_t(need))
		if launchBuf == nil {
			launchCap = 0
			return fmt.Errorf("cuda: malloc launch buffer failed")
		}
		launchCap = need
	}

	base := launchBuf
	valRegion := unsafe.Add(base, uintptr(n*8))
	off := 0
	for i, a := range args {
		dst := unsafe.Add(valRegion, uintptr(off))
		copy(unsafe.Slice((*byte)(dst), len(a)), a)
		*(*unsafe.Pointer)(unsafe.Add(base, uintptr(i*8))) = dst
		off += len(a)
	}

	var err error
	do(func() {
		if C.dlcuda_launch(C.ulonglong(fn),
			C.uint(gx), C.uint(gy), 1,
			C.uint(bx), 1, 1,
			C.uint(shared),
			(*unsafe.Pointer)(base)) != 0 {
			err = fmt.Errorf("cuda: launch failed: %s", C.GoString(C.dlcuda_last_error()))
		}
	})
	return err
}

// VectorAdd computes out[i] = a[i] + b[i] for n float32 elements.
func VectorAdd(a, b, out Buffer, n int) error {
	block := 256
	return launch(kernels.VectorAdd, cdiv(n, block), block, [][]byte{
		ptrb(a), ptrb(b), ptrb(out), i32b(n),
	})
}

// KVStoreF16 quantizes float32 K/V to packed half2 in the KV cache.
func KVStoreF16(kIn, vIn, kOut, vOut Buffer, pos, kvDim, ringCap int) error {
	halfDim := kvDim / 2
	block := 256
	grid := cdiv(halfDim, block)
	if grid == 0 {
		grid = 1
	}
	return launch(kernels.KVStoreF16, grid, block, [][]byte{
		ptrb(kIn), ptrb(vIn), ptrb(kOut), ptrb(vOut),
		i32b(pos), i32b(kvDim), i32b(ringCap), graphScArg(),
	})
}

// RMSNorm computes out[i] = x[i] * w[i] * rsqrt(mean(x^2)+eps).
func RMSNorm(x, w, out Buffer, n int, eps float32) error {
	block := pow2Block(n, 1024)
	return launch(kernels.RMSNorm, 1, block, [][]byte{
		ptrb(x), ptrb(w), ptrb(out), i32b(n), f32b(eps),
	})
}

// SiluMul computes out[i] = silu(gate[i]) * up[i].
func SiluMul(gate, up, out Buffer, n int) error {
	block := 256
	return launch(kernels.SiluMul, cdiv(n, block), block, [][]byte{
		ptrb(gate), ptrb(up), ptrb(out), i32b(n),
	})
}

// SoftmaxRow applies an in-place softmax over n float32 elements.
func SoftmaxRow(x Buffer, n int) error {
	block := pow2Block(n, 1024)
	return launch(kernels.SoftmaxRow, 1, block, [][]byte{
		ptrb(x), i32b(n),
	})
}

// Probe7 is a diagnostic kernel: 7 args, trivial body (writes c[0]).
func Probe7(a, b, c, d Buffer, n1, n2, n3 int) error {
	return launch(kernels.Probe7, 1, 32, [][]byte{
		ptrb(a), ptrb(b), ptrb(c), ptrb(d), i32b(n1), i32b(n2), i32b(n3),
	})
}

// HalfProbe is a diagnostic kernel: out[i] = half_bits(in[i]).
func HalfProbe(in, out Buffer, n int) error {
	block := 256
	return launch(kernels.HalfProbe, cdiv(n, block), block, [][]byte{
		ptrb(in), ptrb(out), i32b(n),
	})
}

// UploadAt copies src into device buffer dst at a byte offset (H2D, sync).
func UploadAt(dst Buffer, offsetBytes uint64, src []byte) error {
	if len(src) == 0 {
		return nil
	}
	var err error
	do(func() {
		if C.dlcuda_upload_at(C.ulonglong(dst), C.ulonglong(offsetBytes),
			unsafe.Pointer(&src[0]), C.ulonglong(len(src))) != 0 {
			err = fmt.Errorf("cuda: upload_at failed: %s", C.GoString(C.dlcuda_last_error()))
		}
	})
	return err
}

// LayerNorm computes out = (x-mean)*rsqrt(var+eps)*w + b.
func LayerNorm(x, w, b, out Buffer, n int, eps float32) error {
	block := pow2Block(n, 1024)
	return launch(kernels.LayerNorm, 1, block, [][]byte{
		ptrb(x), ptrb(w), ptrb(b), ptrb(out), i32b(n), f32b(eps),
	})
}

// RMSNormHeads applies per-head RMSNorm in place. grid = numHeads.
func RMSNormHeads(data, w Buffer, numHeads, headDim int, eps float32) error {
	block := pow2Block(headDim, 1024)
	return launch(kernels.RMSNormHeads, numHeads, block, [][]byte{
		ptrb(data), ptrb(w), i32b(headDim), f32b(eps),
	})
}

// Add computes out = a + b elementwise.
func Add(a, b, out Buffer, n int) error {
	block := 256
	return launch(kernels.Add, cdiv(n, block), block, [][]byte{
		ptrb(a), ptrb(b), ptrb(out), i32b(n),
	})
}

// Scale multiplies buf by s in place.
func Scale(buf Buffer, s float32, n int) error {
	block := 256
	return launch(kernels.Scale, cdiv(n, block), block, [][]byte{
		ptrb(buf), f32b(s), i32b(n),
	})
}

// AddRMSNorm computes sum=a+b and out=sum*w*rsqrt(mean(sum^2)+eps).
func AddRMSNorm(a, b, w, sum, out Buffer, n int, eps float32) error {
	block := pow2Block(n, 1024)
	return launch(kernels.AddRMSNorm, 1, block, [][]byte{
		ptrb(a), ptrb(b), ptrb(w), ptrb(sum), ptrb(out), i32b(n), f32b(eps),
	})
}

// RoPE applies rotary embedding to q/k using precomputed [pos][half_dim] tables.
func RoPE(q, k, cosT, sinT Buffer, numHeads, numKVHeads, headDim, ropeDim, pos int, neox bool) error {
	rd := ropeDim
	if rd <= 0 {
		rd = headDim
	}
	halfDim := rd / 2
	if halfDim <= 0 {
		return nil
	}
	neoxi := 0
	if neox {
		neoxi = 1
	}
	heads := numHeads
	if numKVHeads > heads {
		heads = numKVHeads
	}
	return launch(kernels.RoPE, heads, halfDim, [][]byte{
		ptrb(q), ptrb(k), ptrb(cosT), ptrb(sinT),
		i32b(numHeads), i32b(numKVHeads), i32b(headDim), i32b(ropeDim),
		i32b(pos), i32b(neoxi), graphScArg(),
	})
}

// Attention split-KV tuning. Must match ATTN_TILE / ATTN_MAXSPLITS in kernels.cu.
const (
	attnThreads   = 256
	attnTile      = 256
	attnMaxSplits = 64
)

// cudaAttnSplits overrides the split count (experiments); 0 = derive from ctx.
var cudaAttnSplits = func() int {
	if v := os.Getenv("DLGO_CUDA_ATTN_SPLITS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return 0
}()

var (
	attnPartMu   sync.Mutex
	attnPartBuf  Buffer
	attnPartSize uint64
	// attnPartWant is the largest partials buffer any call has needed. A graph
	// capture cannot allocate (cuMemAlloc is not capturable), so GraphCapture
	// pre-reserves this much.
	attnPartWant uint64
)

// attentionPart returns a cached device scratch buffer for attention partials.
func attentionPart(size uint64) (Buffer, error) {
	attnPartMu.Lock()
	defer attnPartMu.Unlock()
	if attnPartBuf != 0 && attnPartSize >= size {
		return attnPartBuf, nil
	}
	if attnPartBuf != 0 {
		Free(attnPartBuf)
		attnPartBuf, attnPartSize = 0, 0
	}
	b, err := Alloc(size)
	if err != nil {
		return 0, err
	}
	attnPartBuf, attnPartSize = b, size
	return b, nil
}

// ---- CUDA graphs: capture the per-token layer loop, replay per token --------
//
// Decode fires ~13 kernels x 36 layers per token, and each launch costs ~10us on
// this WDDM driver (~4.9 ms of every token; see TestLaunchOverhead). Capturing
// that sequence into a CUDA graph and replaying it collapses the dispatch to one
// submission. The per-token scalars (pos / seq_len / win_start / kv_cap) are read
// from graphBuf by rope/kv_store/attention instead of being baked into the
// recorded launch, so one graph serves every decode step.
var (
	graphBuf  Buffer // int32[4]: pos, seqLen, winStart, kvCap
	graphMode bool   // true while capturing: kernels read graphBuf
	graphExec uint64
)

// graphScArg is the device-scalar launch argument: the scalars buffer while
// capturing, NULL otherwise (kernels then use their immediate argument).
func graphScArg() []byte {
	if graphMode && graphBuf != 0 {
		return ptrb(graphBuf)
	}
	return u64b(0)
}

// PrimeAttentionPartials reserves the split-KV partials buffer for a known
// (heads, kvHeads, headDim) shape.
//
// A graph capture cannot allocate device memory, and AttentionF16 is what
// normally records the worst-case partials size. The batched prefill path never
// calls AttentionF16, so without priming, the first captured decode step tries
// to allocate inside the capture and dies with
// CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED.
func PrimeAttentionPartials(numHeads, numKVHeads, headDim int) error {
	if numKVHeads <= 0 || headDim <= 0 || numHeads%numKVHeads != 0 {
		return nil
	}
	kvMul := numHeads / numKVHeads
	need := uint64(numKVHeads*attnMaxSplits*kvMul*(headDim+2)) * 4
	if need > attnPartWant {
		attnPartWant = need
	}
	_, err := attentionPart(need)
	return err
}

// GraphReady reports whether a captured graph is available for replay.
func GraphReady() bool { return graphExec != 0 }

// GraphCapture begins capturing subsequent launches into a graph.
func GraphCapture() error {
	if graphBuf == 0 {
		b, err := Alloc(16)
		if err != nil {
			return err
		}
		graphBuf = b
	}
	// Device allocations are not capturable, so reserve the attention partials
	// (the only lazily allocated buffer on the decode path) up front.
	if attnPartWant > 0 {
		if _, err := attentionPart(attnPartWant); err != nil {
			return err
		}
	}
	GraphDiscard()
	graphMode = true
	var rc C.int
	do(func() { rc = C.dlcuda_graph_begin() })
	if rc != 0 {
		graphMode = false
		return fmt.Errorf("cuda: graph begin failed: %s", LastError())
	}
	return nil
}

// GraphCaptureEnd closes the capture and keeps the executable for replay.
func GraphCaptureEnd() error {
	var exec uint64
	do(func() { exec = uint64(C.dlcuda_graph_end()) })
	graphMode = false
	if exec == 0 {
		return fmt.Errorf("cuda: graph capture failed: %s", LastError())
	}
	graphExec = exec
	return nil
}

// GraphReplay uploads fresh scalars and replays the captured graph.
func GraphReplay(pos, seqLen, winStart, kvCap int) error {
	if graphExec == 0 || graphBuf == 0 {
		return fmt.Errorf("cuda: no graph captured")
	}
	sc := make([]byte, 16)
	le32(sc[0:], uint32(int32(pos)))
	le32(sc[4:], uint32(int32(seqLen)))
	le32(sc[8:], uint32(int32(winStart)))
	le32(sc[12:], uint32(int32(kvCap)))
	if err := UploadAt(graphBuf, 0, sc); err != nil {
		return err
	}
	var rc C.int
	do(func() {
		rc = C.dlcuda_upload_at_async(C.ulonglong(graphBuf), 0, unsafe.Pointer(&sc[0]), C.ulonglong(len(sc)))
	})
	if rc != 0 {
		return fmt.Errorf("cuda: scalar upload failed: %s", LastError())
	}
	do(func() { rc = C.dlcuda_graph_launch(C.ulonglong(graphExec)) })
	if rc != 0 {
		return fmt.Errorf("cuda: graph launch failed: %s", LastError())
	}
	return nil
}

// GraphDiscard drops the captured graph.
func GraphDiscard() {
	if graphExec != 0 {
		exec := graphExec
		graphExec = 0
		do(func() { C.dlcuda_graph_destroy(C.ulonglong(exec)) })
	}
}

// AttentionF16 is single-query causal GQA attention over an f16 KV cache.
//
// Long contexts use the split-KV / GQA-folded kernel: one block per (KV head,
// key-range split) serves all numHeads/numKVHeads query heads that share that
// KV head, so each KV row is read once instead of once per query head, and the
// key range is split across blocks to keep the GPU busy. Short contexts use the
// block-per-head two-pass kernel, which has lower fixed cost.
func AttentionF16(q, kc, vc, out Buffer, numHeads, numKVHeads, headDim, kvDim,
	seqLen, winStart int, scale, softcap float32, kvCap int) error {
	n := seqLen - winStart
	if n <= 0 {
		// Nothing visible: zero the output.
		return Memset8(out, 0, uint64(numHeads*headDim*4))
	}
	// Remember the worst-case partials size for graph capture (which cannot
	// allocate device memory).
	if numKVHeads > 0 && numHeads%numKVHeads == 0 && headDim > 0 {
		kvMul := numHeads / numKVHeads
		if need := uint64(numKVHeads*attnMaxSplits*kvMul*(headDim+2)) * 4; need > attnPartWant {
			attnPartWant = need
		}
	}
	// In graph-capture mode the key count/pos must be read from device memory at
	// replay time, so use the split kernel even for short contexts.
	if (n > attnTile || graphMode) && numKVHeads > 0 && numHeads%numKVHeads == 0 && headDim%2 == 0 {
		kvMul := numHeads / numKVHeads
		tph := attnThreads / kvMul
		if kvMul <= 8 && attnThreads%kvMul == 0 && tph >= 32 && tph%32 == 0 && kvMul*headDim <= 512 {
			// Split count. Measured (RTX 4070 Ti SUPER, 32 heads / 8 KV heads /
			// head_dim 128): this kernel is parallelism-limited, not
			// bandwidth-limited -- 8 -> 16 -> 32 splits each bought ~1.7-1.8x at
			// 12k ctx, and 64 splits another 1.8x (0.2135 -> 0.1191 ms/layer,
			// 230 -> 413 GB/s). 128 is no better and hurts short contexts.
			//
			// It MUST NOT depend on n. A captured CUDA graph bakes the grid into
			// the replay, so a split count derived from the live key count makes
			// the graph replay whatever tiny grid was live at capture time:
			// capturing at n=1 and replaying at n=4114 cost 10x decode
			// (111 -> 11 tok/s).
			splits := cudaAttnSplits
			if splits <= 0 {
				splits = attnMaxSplits
			}
			if splits > attnMaxSplits {
				splits = attnMaxSplits
			}
			stride := kvMul * (headDim + 2)
			part, err := attentionPart(uint64(numKVHeads*splits*stride) * 4)
			if err != nil {
				return err
			}
			// v2 (one warp per query head, lane-parallel 8-byte K/V loads, no
			// shared memory) covers head_dim 64 and 128; other shapes keep the
			// v1 key-per-thread kernel.
			kfn := kernels.AttentionSplitF16
			threads := attnThreads
			shared := 0
			switch {
			case kvMul >= 2 && headDim == 128:
				kfn, threads = kernels.AttnSplitV2D4, kvMul*32
			case kvMul >= 2 && headDim == 64:
				kfn, threads = kernels.AttnSplitV2D2, kvMul*32
			default:
				nw := attnThreads / 32
				shared = (kvMul*headDim + kvMul*attnTile + kvMul*nw + 5*kvMul) * 4
			}
			if err := launchN(kfn, numKVHeads, splits, threads, shared, [][]byte{
				ptrb(q), ptrb(kc), ptrb(vc), ptrb(part),
				i32b(numHeads), i32b(numKVHeads), i32b(headDim), i32b(kvDim),
				i32b(seqLen), f32b(scale), f32b(softcap), i32b(winStart),
				i32b(kvCap), i32b(splits), graphScArg(),
			}); err != nil {
				return err
			}
			block := headDim
			if block < 32 {
				block = 32
			}
			if block > 1024 {
				block = 1024
			}
			return launch(kernels.AttentionCombine, numHeads, block, [][]byte{
				ptrb(part), ptrb(out),
				i32b(numHeads), i32b(numKVHeads), i32b(headDim), i32b(splits),
			})
		}
	}
	// Block-per-head two-pass kernel: grid = numHeads, 256 threads/block.
	return launch(kernels.AttentionF16, numHeads, 256, [][]byte{
		ptrb(q), ptrb(kc), ptrb(vc), ptrb(out),
		i32b(numHeads), i32b(numKVHeads), i32b(headDim), i32b(kvDim),
		i32b(seqLen), f32b(scale), f32b(softcap), i32b(winStart), i32b(kvCap),
	})
}

// GEMV typed/quantized: out[row] = sum_c dequant(W[row,c]) * x[c].
// One warp per row; 16 rows per 512-thread block (must match GEMV_ROWS_PER_BLOCK
// in kernels.cu).
const gemvRowsPerBlock = 16

func gemv(kern uint64, W, x, out Buffer, rows, cols int) error {
	if rows <= 0 {
		return nil
	}
	return launch(kern, cdiv(rows, gemvRowsPerBlock), gemvRowsPerBlock*32, [][]byte{
		ptrb(W), ptrb(x), ptrb(out), i32b(rows), i32b(cols),
	})
}

func GEMVQ4K(W, x, out Buffer, rows, cols int) error {
	// Fast path: block-cooperative int8 dp4a (one block per row). Falls back to
	// the vectorized float kernel if Q8 scratch is unavailable or cols not a
	// multiple of 256. Set DLGO_CUDA_Q4K=float to force the float path.
	if cudaQ4KMode != "float" && cols%256 == 0 && cols >= 256 {
		xq, dx, sumxq, ok := ensureQ8(cols)
		if ok {
			if err := launch(kernels.QuantizeX, cols/32, 32, [][]byte{
				ptrb(x), i32b(cols), ptrb(xq), ptrb(dx), ptrb(sumxq),
			}); err != nil {
				return err
			}
			return launch(kernels.GemvQ4KBc, rows, 256, [][]byte{
				ptrb(W), ptrb(xq), ptrb(dx), ptrb(sumxq), ptrb(out), i32b(rows), i32b(cols),
			})
		}
	}
	// Float Q4_K path: one warp per row (GEMV_ROWS_PER_BLOCK rows/block).
	return gemv(kernels.GemvQ4K, W, x, out, rows, cols)
}
func GEMVQ6K(W, x, out Buffer, rows, cols int) error {
	return gemv(kernels.GemvQ6K, W, x, out, rows, cols)
}
func GEMVF32(W, x, out Buffer, rows, cols int) error {
	return gemv(kernels.GemvF32, W, x, out, rows, cols)
}
func GEMVF16(W, x, out Buffer, rows, cols int) error {
	return gemv(kernels.GemvF16, W, x, out, rows, cols)
}

// ---------------------------------------------------------------------------
// Canonical Affine4 / Affine8 (see quant/affine4.go, quant/affine8.go): one
// weight type covers every source quantization scheme, so one GEMV serves decode
// and one GEMM serves prefill.
// ---------------------------------------------------------------------------

// F32ToHalf converts n float32 values to float16.
func F32ToHalf(x, y Buffer, n int) error {
	if n <= 0 {
		return nil
	}
	const block = 256
	return launch(kernels.F32ToHalf, cdiv(n, block), block, [][]byte{
		ptrb(x), ptrb(y), i32b(n),
	})
}

// GEMVAffine4 is the single-token form for canonical Affine4 weights.
func GEMVAffine4(W, x, out Buffer, rows, cols int) error {
	return gemv(kernels.GemvAffine4, W, x, out, rows, cols)
}

// GEMVAffine8 is the single-token form for canonical Affine8 weights.
func GEMVAffine8(W, x, out Buffer, rows, cols int) error {
	return gemv(kernels.GemvAffine8, W, x, out, rows, cols)
}

// gemmTileTok / gemmTileRow must match DLG_TOK / DLG_ROW in kernels.cu.
const (
	gemmTileTok = 128
	gemmTileRow = 128
	gemmBlock   = 256
)

// GemmAffine4F16 computes out[tok][row] = sum_c X[tok][c]*W[row][c] over npos
// tokens with canonical Affine4 weights. X must already be float16 ([npos][cols],
// see F32ToHalf); out is float32 [npos][rows] â€” token-major, which is the layout
// every batch layer kernel expects.
func GemmAffine4F16(W, xh, out Buffer, rows, cols, npos int) error {
	return gemmAffine(kernels.GemmAffine4F16, W, xh, out, rows, cols, npos)
}

// GemmAffine8F16 is the Affine8 form of GemmAffine4F16.
func GemmAffine8F16(W, xh, out Buffer, rows, cols, npos int) error {
	return gemmAffine(kernels.GemmAffine8F16, W, xh, out, rows, cols, npos)
}

func gemmAffine(kern uint64, W, xh, out Buffer, rows, cols, npos int) error {
	if rows <= 0 || cols <= 0 || npos <= 0 {
		return nil
	}
	if cols%32 != 0 {
		return fmt.Errorf("cuda: gemm cols %d is not a multiple of the affine group size 32", cols)
	}
	// grid.x = row tiles (vary fastest, so a scheduling wave keeps its X token
	// tiles hot in L2), grid.y = token tiles.
	return launchN(kern, cdiv(rows, gemmTileRow), cdiv(npos, gemmTileTok), gemmBlock, 0, [][]byte{
		ptrb(W), ptrb(xh), ptrb(out), i32b(rows), i32b(cols), i32b(npos),
	})
}

// ---------------------------------------------------------------------------
// Batched (prompt-processing) layer primitives. Every buffer is token-major:
// x[pos][elem], so grid.y is the position and one dispatch covers the whole
// prompt chunk.
// ---------------------------------------------------------------------------

// BatchRMSNorm computes out[pos] = rmsnorm(x[pos]) * w for every position.
func BatchRMSNorm(out, x, w Buffer, n, npos int, eps float32) error {
	if npos <= 0 {
		return nil
	}
	return launch(kernels.BatchRMSNorm, npos, 256, [][]byte{
		ptrb(out), ptrb(x), ptrb(w), i32b(n), i32b(npos), f32b(eps),
	})
}

// BatchAddRMSNorm writes sumOut = a+b per position and normOut = rmsnorm(sumOut)*w.
func BatchAddRMSNorm(normOut, sumOut, a, b, w Buffer, n, npos int, eps float32) error {
	if npos <= 0 {
		return nil
	}
	return launch(kernels.BatchAddRMSNorm, npos, 256, [][]byte{
		ptrb(normOut), ptrb(sumOut), ptrb(a), ptrb(b), ptrb(w), i32b(n), i32b(npos), f32b(eps),
	})
}

// BatchRMSNormHeads applies the per-head RMS norm (QK-norm) to every position.
func BatchRMSNormHeads(data, w Buffer, numHeads, headDim, npos int, eps float32) error {
	if npos <= 0 || numHeads <= 0 {
		return nil
	}
	return launchN(kernels.BatchRMSNormHeads, numHeads, npos, headDim, 0, [][]byte{
		ptrb(data), ptrb(w), i32b(numHeads), i32b(headDim), i32b(npos), f32b(eps),
	})
}

// BatchRoPE applies RoPE to every position of a chunk, with position
// startPos+pos for the token at chunk offset pos.
func BatchRoPE(q, k, cosT, sinT Buffer, numHeads, numKVHeads, headDim, ropeDim, startPos int,
	neox bool, npos int) error {
	if npos <= 0 {
		return nil
	}
	rd := ropeDim
	if rd <= 0 || rd > headDim {
		rd = headDim
	}
	nh := numHeads
	if numKVHeads > nh {
		nh = numKVHeads
	}
	neo := 0
	if neox {
		neo = 1
	}
	return launchN(kernels.BatchRoPE, nh, npos, rd/2, 0, [][]byte{
		ptrb(q), ptrb(k), ptrb(cosT), ptrb(sinT),
		i32b(numHeads), i32b(numKVHeads), i32b(headDim), i32b(ropeDim),
		i32b(startPos), i32b(neo), i32b(npos),
	})
}

// BatchKVStoreF16 appends npos rows of float32 k/v to the f16 KV cache starting
// at absolute position startPos.
func BatchKVStoreF16(kIn, vIn, kOut, vOut Buffer, startPos, kvDim, npos int) error {
	if npos <= 0 {
		return nil
	}
	const block = 256
	return launchN(kernels.BatchKVStoreF16, cdiv(kvDim, block), npos, block, 0, [][]byte{
		ptrb(kIn), ptrb(vIn), ptrb(kOut), ptrb(vOut), i32b(startPos), i32b(kvDim), i32b(npos),
	})
}

// AttnPrefillF16 is causal flash attention over a prompt chunk: one block per
// (head, 64-query tile), keys streamed in 32-row tiles from the f16 KV cache.
// Requires headDim <= 128 and headDim%16 == 0.
func AttnPrefillF16(out, q, kCache, vCache Buffer, numHeads, numKVHeads, headDim, kvDim,
	startPos, npos int, scale, softcap float32) error {
	if npos <= 0 {
		return nil
	}
	if headDim%16 != 0 || headDim > 128 {
		return fmt.Errorf("cuda: prefill attention needs headDim<=128 and a multiple of 16 (got %d)", headDim)
	}
	return launchN(kernels.AttnPrefillF16, numHeads, cdiv(npos, 64), 128, 0, [][]byte{
		ptrb(out), ptrb(q), ptrb(kCache), ptrb(vCache),
		i32b(numHeads), i32b(numKVHeads), i32b(headDim), i32b(kvDim),
		i32b(startPos), i32b(npos), f32b(scale), f32b(softcap),
	})
}

// ---------------------------------------------------------------------------
// Persistent int8 quantization scratch for the dp4a GEMV (xq, dx, sumxq). x is
// re-quantized every matvec call but the buffer is reused across calls/rows.
// Layout: [xq: cols int8][dx: cols/32 f32][sumxq: cols/32 i32].
// ---------------------------------------------------------------------------
var (
	q8Buf      Buffer
	q8CapBytes int
	q8Xq       Buffer
	q8Dx       Buffer
	q8Sum      Buffer
	q8Cols     int
)

func ensureQ8(cols int) (xq, dx, sumxq Buffer, ok bool) {
	ng := cols / 32
	xqBytes := (cols + 3) &^ 3
	dxOff := xqBytes
	sumOff := dxOff + ng*4
	total := sumOff + ng*4

	if total > q8CapBytes || q8Buf == 0 {
		if q8Buf != 0 {
			_ = Free(q8Buf)
		}
		b, err := Alloc(uint64(total))
		if err != nil {
			q8Buf = 0
			q8CapBytes = 0
			return 0, 0, 0, false
		}
		q8Buf = b
		q8CapBytes = total
		base := uint64(b)
		q8Xq = Buffer(base)
		q8Dx = Buffer(base + uint64(dxOff))
		q8Sum = Buffer(base + uint64(sumOff))
		q8Cols = cols
	} else if q8Xq == 0 {
		return 0, 0, 0, false
	}
	return q8Xq, q8Dx, q8Sum, true
}

// CopyRegionBytes copies n bytes from src+srcOff to dst+dstOff on the device.
func CopyRegionBytes(dst Buffer, dstOff uint64, src Buffer, srcOff uint64, n uint64) error {
	if n == 0 {
		return nil
	}
	words := cdiv(int((n+3)/4), 1)
	return launch(kernels.CopyRegion, cdiv(words, 256), 256, [][]byte{
		ptrb(dst + dstOff), ptrb(src + srcOff), u64b(n),
	})
}
