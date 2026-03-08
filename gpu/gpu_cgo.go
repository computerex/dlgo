//go:build cgo && vulkan

package gpu

/*
#cgo CFLAGS: -I${SRCDIR}/csrc
#cgo windows CFLAGS: -IC:/VulkanSDK/1.4.341.1/Include
#cgo windows LDFLAGS: -LC:/VulkanSDK/1.4.341.1/Lib -lvulkan-1

#include "vulkan_gpu.c"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Buf is a handle to a GPU buffer.
type Buf = uint64

// Init initializes the Vulkan compute backend.
func Init() error {
	rc := C.gpu_init()
	if rc != C.GPU_OK {
		switch rc {
		case C.GPU_ERR_NO_VULKAN:
			return fmt.Errorf("gpu: vulkan runtime not found")
		case C.GPU_ERR_NO_DEVICE:
			return fmt.Errorf("gpu: no vulkan-capable GPU found")
		case C.GPU_ERR_INIT_FAIL:
			return fmt.Errorf("gpu: vulkan initialization failed")
		default:
			return fmt.Errorf("gpu: init error %d", rc)
		}
	}
	return nil
}

// Shutdown releases all GPU resources.
func Shutdown() { C.gpu_shutdown() }

// IsInitialized returns true if the GPU backend is ready.
func IsInitialized() bool { return C.gpu_is_initialized() != 0 }

// DeviceName returns the GPU device name.
func DeviceName() string { return C.GoString(C.gpu_device_name()) }

// VRAMBytes returns total device-local VRAM in bytes.
func VRAMBytes() uint64 { return uint64(C.gpu_vram_bytes()) }

// Alloc allocates a GPU buffer of the given size.
func Alloc(sizeBytes uint64) Buf {
	return uint64(C.gpu_alloc(C.uint64_t(sizeBytes), C.GPU_BUF_STORAGE))
}

// Free releases a GPU buffer.
func Free(buf Buf) { C.gpu_free(C.GpuBuf(buf)) }

// Upload copies data from CPU to GPU.
func Upload(dst Buf, src []byte) error {
	if len(src) == 0 {
		return nil
	}
	rc := C.gpu_upload(C.GpuBuf(dst), unsafe.Pointer(&src[0]), C.uint64_t(len(src)), 0)
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: upload failed (%d)", rc)
	}
	return nil
}

// UploadF32 copies float32 data from CPU to GPU.
func UploadF32(dst Buf, src []float32) error {
	if len(src) == 0 {
		return nil
	}
	size := len(src) * 4
	rc := C.gpu_upload(C.GpuBuf(dst), unsafe.Pointer(&src[0]), C.uint64_t(size), 0)
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: upload failed (%d)", rc)
	}
	return nil
}

// Download copies data from GPU to CPU.
func Download(src Buf, dst []byte) error {
	if len(dst) == 0 {
		return nil
	}
	rc := C.gpu_download(unsafe.Pointer(&dst[0]), C.GpuBuf(src), C.uint64_t(len(dst)), 0)
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: download failed (%d)", rc)
	}
	return nil
}

// DownloadF32 copies float32 data from GPU to CPU.
func DownloadF32(src Buf, dst []float32) error {
	if len(dst) == 0 {
		return nil
	}
	size := len(dst) * 4
	rc := C.gpu_download(unsafe.Pointer(&dst[0]), C.GpuBuf(src), C.uint64_t(size), 0)
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: download failed (%d)", rc)
	}
	return nil
}

// MatVec performs quantized matrix-vector multiply on GPU.
func MatVec(out, weights, x Buf, rows, cols int, qtype uint32) error {
	rc := C.gpu_matvec(C.GpuBuf(out), C.GpuBuf(weights), C.GpuBuf(x),
		C.int(rows), C.int(cols), C.int(qtype))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: matvec failed (%d)", rc)
	}
	return nil
}

// RMSNorm performs RMS normalization on GPU.
func RMSNorm(out, x, weight Buf, n int, eps float32) error {
	rc := C.gpu_rmsnorm(C.GpuBuf(out), C.GpuBuf(x), C.GpuBuf(weight), C.int(n), C.float(eps))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: rmsnorm failed (%d)", rc)
	}
	return nil
}

// RMSNormHeads performs per-head in-place RMS normalization on GPU.
func RMSNormHeads(data, weight Buf, numHeads, headDim int, eps float32) error {
	rc := C.gpu_rmsnorm_heads(C.GpuBuf(data), C.GpuBuf(weight), C.int(numHeads), C.int(headDim), C.float(eps))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: rmsnorm_heads failed (%d)", rc)
	}
	return nil
}

// Softmax performs in-place softmax on GPU.
func Softmax(buf Buf, n int) error {
	rc := C.gpu_softmax(C.GpuBuf(buf), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: softmax failed (%d)", rc)
	}
	return nil
}

// RoPE applies rotary position embedding on GPU.
func RoPE(q, k Buf, numHeads, numKVHeads, headDim, ropeDim, pos int, freqBase float32, neox bool) error {
	n := 0
	if neox {
		n = 1
	}
	rc := C.gpu_rope(C.GpuBuf(q), C.GpuBuf(k),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(ropeDim), C.int(pos), C.float(freqBase), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: rope failed (%d)", rc)
	}
	return nil
}

// SwiGLU performs SwiGLU activation on GPU.
func SwiGLU(out, gate, up Buf, n int) error {
	rc := C.gpu_swiglu(C.GpuBuf(out), C.GpuBuf(gate), C.GpuBuf(up), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: swiglu failed (%d)", rc)
	}
	return nil
}

// GeGLU performs GeGLU activation on GPU.
func GeGLU(out, gate, up Buf, n int) error {
	rc := C.gpu_geglu(C.GpuBuf(out), C.GpuBuf(gate), C.GpuBuf(up), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: geglu failed (%d)", rc)
	}
	return nil
}

// GELU performs in-place GELU activation on GPU.
func GELU(buf Buf, n int) error {
	rc := C.gpu_gelu(C.GpuBuf(buf), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: gelu failed (%d)", rc)
	}
	return nil
}

// Add performs element-wise addition on GPU.
func Add(out, a, b Buf, n int) error {
	rc := C.gpu_add(C.GpuBuf(out), C.GpuBuf(a), C.GpuBuf(b), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: add failed (%d)", rc)
	}
	return nil
}

// Scale performs in-place scaling on GPU.
func Scale(buf Buf, s float32, n int) error {
	rc := C.gpu_scale(C.GpuBuf(buf), C.float(s), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: scale failed (%d)", rc)
	}
	return nil
}

// Attention performs fused multi-head attention entirely on GPU.
func Attention(out, q, kCache, vCache Buf, numHeads, numKVHeads, headDim, kvDim, seqLen int, scale float32) error {
	rc := C.gpu_attention(C.GpuBuf(out), C.GpuBuf(q), C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(kvDim), C.int(seqLen), C.float(scale))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: attention failed (%d)", rc)
	}
	return nil
}

// KVStore copies K and V vectors into cache buffers at the given position.
func KVStore(kCache, vCache, k, v Buf, pos, kvDim int) error {
	rc := C.gpu_kv_store(C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.GpuBuf(k), C.GpuBuf(v), C.int(pos), C.int(kvDim))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: kv_store failed (%d)", rc)
	}
	return nil
}

// Sync waits for all GPU operations to complete.
func Sync() { C.gpu_sync() }

// BeginBatch starts recording GPU operations into a single command buffer.
// All subsequent GPU calls are batched until EndBatch.
func BeginBatch() { C.gpu_begin_batch() }

// EndBatch submits all batched operations at once and waits for completion.
func EndBatch() { C.gpu_end_batch() }

// Barrier inserts a compute memory barrier so subsequent dispatches see prior writes.
func Barrier() { C.gpu_barrier() }

// AddRMSNorm performs fused Add + RMSNorm: sumOut = a+b, normOut = RMSNorm(sumOut, weight).
func AddRMSNorm(normOut, sumOut, a, b, weight Buf, n int, eps float32) error {
	rc := C.gpu_add_rmsnorm(C.GpuBuf(normOut), C.GpuBuf(sumOut),
		C.GpuBuf(a), C.GpuBuf(b), C.GpuBuf(weight), C.int(n), C.float(eps))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: add_rmsnorm failed (%d)", rc)
	}
	return nil
}

// LayerConf holds all buffer handles and parameters for one transformer layer.
// Set up once per model, reused for every token.
type LayerConf struct {
	c C.GpuLayerConf
}

// NewLayerConf creates a LayerConf from the model's layer data.
func NewLayerConf() *LayerConf { return &LayerConf{} }

func (lc *LayerConf) SetScratch(x, xNorm, q, k, v, attnOut, attnProj Buf,
	ffnNorm, ffnIn, gate, up, hidden, ffnOut Buf) {
	lc.c.x = C.GpuBuf(x)
	lc.c.x_norm = C.GpuBuf(xNorm)
	lc.c.q = C.GpuBuf(q)
	lc.c.k = C.GpuBuf(k)
	lc.c.v = C.GpuBuf(v)
	lc.c.attn_out = C.GpuBuf(attnOut)
	lc.c.attn_proj = C.GpuBuf(attnProj)
	lc.c.ffn_norm = C.GpuBuf(ffnNorm)
	lc.c.ffn_in = C.GpuBuf(ffnIn)
	lc.c.gate = C.GpuBuf(gate)
	lc.c.up = C.GpuBuf(up)
	lc.c.hidden = C.GpuBuf(hidden)
	lc.c.ffn_out = C.GpuBuf(ffnOut)
}

func (lc *LayerConf) SetAttn(attnNorm Buf, wq, wk, wv, wo *GpuTensor,
	bq, bk, bv Buf, qNorm, kNorm Buf) {
	lc.c.attn_norm_w = C.GpuBuf(attnNorm)
	lc.c.wq = C.GpuBuf(wq.Buf)
	lc.c.wq_rows = C.int(wq.Rows)
	lc.c.wq_cols = C.int(wq.Cols)
	lc.c.wq_type = C.int(wq.Type)
	lc.c.wk = C.GpuBuf(wk.Buf)
	lc.c.wk_rows = C.int(wk.Rows)
	lc.c.wk_cols = C.int(wk.Cols)
	lc.c.wk_type = C.int(wk.Type)
	lc.c.wv = C.GpuBuf(wv.Buf)
	lc.c.wv_rows = C.int(wv.Rows)
	lc.c.wv_cols = C.int(wv.Cols)
	lc.c.wv_type = C.int(wv.Type)
	lc.c.wo = C.GpuBuf(wo.Buf)
	lc.c.wo_rows = C.int(wo.Rows)
	lc.c.wo_cols = C.int(wo.Cols)
	lc.c.wo_type = C.int(wo.Type)
	lc.c.bq = C.GpuBuf(bq)
	lc.c.bk = C.GpuBuf(bk)
	lc.c.bv = C.GpuBuf(bv)
	lc.c.q_norm_w = C.GpuBuf(qNorm)
	lc.c.k_norm_w = C.GpuBuf(kNorm)
}

func (lc *LayerConf) SetFFN(ffnNorm Buf, gate, up, down *GpuTensor,
	postAttnNorm, postFFNNorm Buf) {
	lc.c.ffn_norm_w = C.GpuBuf(ffnNorm)
	if gate != nil {
		lc.c.ffn_gate_w = C.GpuBuf(gate.Buf)
		lc.c.gate_rows = C.int(gate.Rows)
		lc.c.gate_cols = C.int(gate.Cols)
		lc.c.gate_type = C.int(gate.Type)
	}
	lc.c.ffn_up_w = C.GpuBuf(up.Buf)
	lc.c.up_rows = C.int(up.Rows)
	lc.c.up_cols = C.int(up.Cols)
	lc.c.up_type = C.int(up.Type)
	lc.c.ffn_down_w = C.GpuBuf(down.Buf)
	lc.c.down_rows = C.int(down.Rows)
	lc.c.down_cols = C.int(down.Cols)
	lc.c.down_type = C.int(down.Type)
	lc.c.post_attn_norm_w = C.GpuBuf(postAttnNorm)
	lc.c.post_ffn_norm_w = C.GpuBuf(postFFNNorm)
}

func (lc *LayerConf) SetKV(kCache, vCache Buf) {
	lc.c.k_cache = C.GpuBuf(kCache)
	lc.c.v_cache = C.GpuBuf(vCache)
}

func (lc *LayerConf) SetConfig(dim, headDim, numHeads, numKVHeads, kvDim int,
	rmsEps, ropeFreqBase float32, ropeDim int, ropeNeox bool,
	ffnType, residualType int) {
	lc.c.dim = C.int(dim)
	lc.c.head_dim = C.int(headDim)
	lc.c.num_heads = C.int(numHeads)
	lc.c.num_kv_heads = C.int(numKVHeads)
	lc.c.kv_dim = C.int(kvDim)
	lc.c.rms_eps = C.float(rmsEps)
	lc.c.rope_freq_base = C.float(ropeFreqBase)
	lc.c.rope_dim = C.int(ropeDim)
	if ropeNeox {
		lc.c.rope_neox = 1
	}
	lc.c.ffn_type = C.int(ffnType)
	lc.c.residual_type = C.int(residualType)
}

func (lc *LayerConf) SetDP4A(q8_1Scratch Buf) {
	lc.c.q8_1_scratch = C.GpuBuf(q8_1Scratch)
	if q8_1Scratch != 0 {
		lc.c.use_dp4a = 1
	}
}

// ForwardLayer records all dispatches for one transformer layer.
func ForwardLayer(lc *LayerConf, pos, seqLen int, scale float32, nextAttnNorm Buf) error {
	rc := C.gpu_forward_layer(&lc.c, C.int(pos), C.int(seqLen), C.float(scale),
		C.GpuBuf(nextAttnNorm))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: forward_layer failed (%d)", rc)
	}
	return nil
}

// ForwardLayerBatch records all dispatches for npos tokens through one layer.
func ForwardLayerBatch(lc *LayerConf, npos, startPos int, scale float32, nextAttnNorm Buf) error {
	rc := C.gpu_forward_layer_batch(&lc.c, C.int(npos), C.int(startPos), C.float(scale),
		C.GpuBuf(nextAttnNorm))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: forward_layer_batch failed (%d)", rc)
	}
	return nil
}

// BatchRMSNorm performs RMS normalization over npos positions.
func BatchRMSNorm(out, x, weight Buf, n, npos int, eps float32) error {
	rc := C.gpu_batch_rmsnorm(C.GpuBuf(out), C.GpuBuf(x), C.GpuBuf(weight),
		C.int(n), C.int(npos), C.float(eps))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: batch_rmsnorm failed (%d)", rc)
	}
	return nil
}

// BatchMatVec performs batched matrix-vector multiply for npos positions.
func BatchMatVec(out, weights, x Buf, rows, cols, npos int, qtype uint32) error {
	rc := C.gpu_batch_matvec(C.GpuBuf(out), C.GpuBuf(weights), C.GpuBuf(x),
		C.int(rows), C.int(cols), C.int(npos), C.int(qtype))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: batch_matvec failed (%d)", rc)
	}
	return nil
}

// CopyRegion copies a region between GPU buffers.
func CopyRegion(dst Buf, dstOff uint64, src Buf, srcOff, size uint64) error {
	rc := C.gpu_copy_region(C.GpuBuf(dst), C.uint64_t(dstOff),
		C.GpuBuf(src), C.uint64_t(srcOff), C.uint64_t(size))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: copy_region failed (%d)", rc)
	}
	return nil
}
