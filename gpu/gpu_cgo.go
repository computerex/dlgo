//go:build cgo && vulkan

package gpu

/*
#cgo CFLAGS: -I${SRCDIR}/csrc
#cgo windows CFLAGS: -IC:/VulkanSDK/1.4.341.1/Include
#cgo windows LDFLAGS: -LC:/VulkanSDK/1.4.341.1/Lib -lvulkan-1

// Force Go to track shaders_spirv.h for build cache invalidation.
// Without this, Go only tracks vulkan_gpu.c but not headers it includes,
// so shader changes can be silently missed by the build cache.
#include "shaders_spirv.h"

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

// VRAMFreeBytes returns currently available device-local VRAM in bytes.
// Uses VK_EXT_memory_budget when available, otherwise falls back to 90% of total.
func VRAMFreeBytes() uint64 { return uint64(C.gpu_vram_free_bytes()) }

// AllocatedBytes returns cumulative VRAM allocated by this process (our own counter).
func AllocatedBytes() uint64 { return uint64(C.gpu_allocated_bytes()) }

// Alloc allocates a GPU buffer of the given size.
// Returns 0 on failure — callers that cannot tolerate OOM should use AllocE.
func Alloc(sizeBytes uint64) Buf {
	return uint64(C.gpu_alloc(C.uint64_t(sizeBytes), C.GPU_BUF_STORAGE))
}

// AllocE allocates a GPU buffer, returning an error on failure.
// This is the safe variant that should be used for all allocations where
// a failure must be handled gracefully (like llama.cpp does) rather than
// silently proceeding with a zero buffer.
func AllocE(sizeBytes uint64) (Buf, error) {
	buf := uint64(C.gpu_alloc(C.uint64_t(sizeBytes), C.GPU_BUF_STORAGE))
	if buf == 0 && sizeBytes > 0 {
		return 0, fmt.Errorf("gpu: VRAM allocation failed for %d bytes (%.1f MB)", sizeBytes, float64(sizeBytes)/(1024*1024))
	}
	return buf, nil
}

// Free releases a GPU buffer.
func Free(buf Buf) { C.gpu_free(C.GpuBuf(buf)) }

// ResetBufferTable compacts the buffer ID table after freeing all buffers,
// allowing new allocations to reuse slots from the beginning.
func ResetBufferTable() { C.gpu_reset_buffer_table() }

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

// UploadI32 copies int32 data from CPU to GPU.
func UploadI32(dst Buf, src []int32) error {
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

// ZeroFill writes zeros to a GPU buffer.
func ZeroFill(dst Buf, sizeBytes uint64) {
	zeros := make([]byte, sizeBytes)
	Upload(dst, zeros)
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

// MatVecOffset performs matrix-vector multiply with byte offsets into the output and weight buffers.
// Used for MoE expert projections from packed expert tensors.
func MatVecOffset(out Buf, outOffBytes int, weights Buf, weightsOffBytes int, x Buf, rows, cols int, qtype uint32) error {
	rc := C.gpu_matvec_offset(C.GpuBuf(out), C.int(outOffBytes),
		C.GpuBuf(weights), C.int(weightsOffBytes),
		C.GpuBuf(x), C.int(rows), C.int(cols), C.int(qtype))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: matvec_offset failed (%d)", rc)
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

// LayerNorm performs layer normalization on GPU.
func LayerNorm(out, x, weight, bias Buf, n int, eps float32) error {
	rc := C.gpu_layernorm(C.GpuBuf(out), C.GpuBuf(x), C.GpuBuf(weight), C.GpuBuf(bias), C.int(n), C.float(eps))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: layernorm failed (%d)", rc)
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

// RoPE applies rotary position embedding on GPU using precomputed cos/sin tables.
func RoPE(q, k, cosTable, sinTable Buf, numHeads, numKVHeads, headDim, ropeDim, pos int, neox bool) error {
	n := 0
	if neox {
		n = 1
	}
	rc := C.gpu_rope(C.GpuBuf(q), C.GpuBuf(k), C.GpuBuf(cosTable), C.GpuBuf(sinTable),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(ropeDim), C.int(pos), C.int(n))
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

// SwiGLU_OAI performs OpenAI SwiGLU variant with clamping and alpha-scaled sigmoid on GPU.
func SwiGLU_OAI(out, gate, up Buf, n int, alpha, limit float32) error {
	rc := C.gpu_swiglu_oai(C.GpuBuf(out), C.GpuBuf(gate), C.GpuBuf(up), C.int(n), C.float(alpha), C.float(limit))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: swiglu_oai failed (%d)", rc)
	}
	return nil
}

// AddOffset performs out[i] += bias[offset + i] on GPU.
func AddOffset(out, bias Buf, n, offset int) error {
	rc := C.gpu_add_offset(C.GpuBuf(out), C.GpuBuf(bias), C.int(n), C.int(offset))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: add_offset failed (%d)", rc)
	}
	return nil
}

// ScaleAdd performs out[i] += scale * in[i] on GPU.
func ScaleAdd(out, in Buf, n int, scale float32) error {
	rc := C.gpu_scale_add(C.GpuBuf(out), C.GpuBuf(in), C.int(n), C.float(scale))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: scale_add failed (%d)", rc)
	}
	return nil
}

// SwiGLU_OAI_Bias performs fused bias addition + SwiGLU-OAI activation on GPU.
func SwiGLU_OAI_Bias(out, gate, up, gateBias, upBias Buf, n int, alpha, limit float32, gateBiasOff, upBiasOff int) error {
	rc := C.gpu_swiglu_oai_bias(C.GpuBuf(out), C.GpuBuf(gate), C.GpuBuf(up),
		C.GpuBuf(gateBias), C.GpuBuf(upBias),
		C.int(n), C.float(alpha), C.float(limit),
		C.int(gateBiasOff), C.int(upBiasOff))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: swiglu_oai_bias failed (%d)", rc)
	}
	return nil
}

// MoETopK performs router gating + top-K selection entirely on GPU.
// weightsNorm: if true, normalize selected weights by their sum.
// weightsScale: if non-zero and != 1.0, multiply weights by this factor.
func MoETopK(logits, outIndices, outWeights Buf, nExperts, k, gatingFunc int, weightsNorm bool, weightsScale float32) error {
	wn := 0
	if weightsNorm {
		wn = 1
	}
	rc := C.gpu_moe_topk(C.GpuBuf(logits), C.GpuBuf(outIndices), C.GpuBuf(outWeights),
		C.int(nExperts), C.int(k), C.int(gatingFunc),
		C.int(wn), C.float(weightsScale))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: moe_topk failed (%d)", rc)
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
// startPos specifies the sliding window start position (0 for full attention).
func Attention(out, q, kCache, vCache Buf, numHeads, numKVHeads, headDim, kvDim, seqLen, startPos int, scale float32) error {
	rc := C.gpu_attention(C.GpuBuf(out), C.GpuBuf(q), C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(kvDim), C.int(seqLen), C.float(scale), C.int(startPos))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: attention failed (%d)", rc)
	}
	return nil
}

// AttentionSinks performs attention with learned sink logits per KV head.
// startPos specifies the sliding window start position (0 for full attention).
func AttentionSinks(out, q, kCache, vCache, sinks Buf, numHeads, numKVHeads, headDim, kvDim, seqLen, startPos int, scale float32) error {
	rc := C.gpu_attention_sinks(C.GpuBuf(out), C.GpuBuf(q), C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.GpuBuf(sinks), C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(kvDim), C.int(seqLen), C.float(scale), C.int(startPos))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: attention_sinks failed (%d)", rc)
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

// KVStoreF16 converts float32 K/V to packed half2 and stores in FP16 KV cache.
func KVStoreF16(kCache, vCache, k, v Buf, pos, kvDim int) error {
	rc := C.gpu_kv_store_f16(C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.GpuBuf(k), C.GpuBuf(v), C.int(pos), C.int(kvDim))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: kv_store_f16 failed (%d)", rc)
	}
	return nil
}

// AttentionF16 performs tiled attention with FP16 KV cache (online softmax, no seq_len limit).
func AttentionF16(out, q, kCache, vCache Buf, numHeads, numKVHeads, headDim, kvDim, seqLen, startPos int, scale, softcap float32) error {
	rc := C.gpu_attention_f16(C.GpuBuf(out), C.GpuBuf(q), C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(kvDim), C.int(seqLen), C.float(scale), C.float(softcap), C.int(startPos))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: attention_f16 failed (%d)", rc)
	}
	return nil
}

// AttentionTiledF32 performs tiled causal attention with FP32 KV cache (no seq_len limit).
func AttentionTiledF32(out, q, kCache, vCache Buf, numHeads, numKVHeads, headDim, kvDim, seqLen, startPos int, scale, softcap float32) error {
	rc := C.gpu_attention_tiled_f32(C.GpuBuf(out), C.GpuBuf(q), C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(kvDim), C.int(seqLen), C.float(scale), C.float(softcap), C.int(startPos))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: attention_tiled_f32 failed (%d)", rc)
	}
	return nil
}

// PagedKVStore writes K and V into block pool buffers at the effective position.
// effectivePos = physical_block * block_size + slot_in_block (computed on CPU).
func PagedKVStore(kPool, vPool, k, v Buf, effectivePos, kvDim int) error {
	rc := C.gpu_paged_kv_store(C.GpuBuf(kPool), C.GpuBuf(vPool),
		C.GpuBuf(k), C.GpuBuf(v), C.int(effectivePos), C.int(kvDim))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: paged_kv_store failed (%d)", rc)
	}
	return nil
}

// PagedAttention performs multi-head attention with block-table-indexed KV access.
// blockTableBuf is a GPU buffer of int32 physical block IDs.
func PagedAttention(out, q, kPool, vPool, blockTable Buf,
	numHeads, numKVHeads, headDim, kvDim, seqLen int, scale float32, blockSize int) error {
	rc := C.gpu_paged_attention(C.GpuBuf(out), C.GpuBuf(q),
		C.GpuBuf(kPool), C.GpuBuf(vPool), C.GpuBuf(blockTable),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(kvDim),
		C.int(seqLen), C.float(scale), C.int(blockSize))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: paged_attention failed (%d)", rc)
	}
	return nil
}

// Sync waits for all GPU operations to complete.
func Sync() { C.gpu_sync() }

// HasDp4a returns true if the GPU supports integer dot product (dp4a) acceleration.
func HasDp4a() bool { return C.gpu_has_dp4a() != 0 }

// QuantizeQ8_1 quantizes an f32 buffer to Q8_1 format for dp4a operations.
func QuantizeQ8_1(q8_1Buf, f32Buf Buf, nElements int) error {
	rc := C.gpu_quantize_q8_1(C.GpuBuf(q8_1Buf), C.GpuBuf(f32Buf), C.int(nElements))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: quantize_q8_1 failed (%d)", rc)
	}
	return nil
}

// MatVecOffsetDp4a performs dp4a integer dot product matvec with byte offsets into packed tensors.
func MatVecOffsetDp4a(out Buf, outOff int, weights Buf, weightsOff int, q8_1 Buf,
	rows, cols int, qtype uint32) error {
	rc := C.gpu_matvec_offset_dp4a(C.GpuBuf(out), C.int(outOff),
		C.GpuBuf(weights), C.int(weightsOff),
		C.GpuBuf(q8_1), C.int(rows), C.int(cols), C.int(qtype))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: matvec_offset_dp4a failed (%d)", rc)
	}
	return nil
}

// MoEMatVecDp4a dispatches batched dp4a MoE matvec for all active experts.
// Expert indices stay on GPU — no CPU download needed.
// Output is interleaved: out[slot * rows + row] for each expert slot.
func MoEMatVecDp4a(out, weights, q8_1, indices Buf,
	rows, cols int, qtype uint32,
	expertStride, baseOffset, sharedInput, nUsed int) error {
	rc := C.gpu_moe_matvec_dp4a(
		C.GpuBuf(out), C.GpuBuf(weights),
		C.GpuBuf(q8_1), C.GpuBuf(indices),
		C.int(rows), C.int(cols), C.int(qtype),
		C.int(expertStride), C.int(baseOffset),
		C.int(sharedInput), C.int(nUsed))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: moe_matvec_dp4a failed (%d)", rc)
	}
	return nil
}

// SwiGLUAt dispatches SwiGLU with byte offsets into the buffers.
func SwiGLUAt(out, gate, up Buf, outOff, gateOff, upOff, n int) error {
	rc := C.gpu_swiglu_at(C.GpuBuf(out), C.GpuBuf(gate), C.GpuBuf(up),
		C.int(outOff), C.int(gateOff), C.int(upOff), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: swiglu_at failed (%d)", rc)
	}
	return nil
}

// SwiGLU_OAI_At dispatches SwiGLU_OAI with byte offsets.
func SwiGLU_OAI_At(out, gate, up Buf, outOff, gateOff, upOff, n int, alpha, limit float32) error {
	rc := C.gpu_swiglu_oai_at(C.GpuBuf(out), C.GpuBuf(gate), C.GpuBuf(up),
		C.int(outOff), C.int(gateOff), C.int(upOff),
		C.int(n), C.float(alpha), C.float(limit))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: swiglu_oai_at failed (%d)", rc)
	}
	return nil
}

// QuantizeQ8_1At quantizes f32 to Q8_1 with byte offsets for packed MoE buffers.
func QuantizeQ8_1At(q8Buf Buf, q8Off int, f32Buf Buf, f32Off, nElements int) error {
	rc := C.gpu_quantize_q8_1_at(C.GpuBuf(q8Buf), C.int(q8Off), C.GpuBuf(f32Buf), C.int(f32Off), C.int(nElements))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: quantize_q8_1_at failed (%d)", rc)
	}
	return nil
}

// MoEAccumulate performs weighted accumulation of expert outputs using GPU-side weights and indices.
func MoEAccumulate(out, expOuts, weights, bias, indices Buf,
	dim, nUsed int, hasBias bool) error {
	hb := 0
	if hasBias {
		hb = 1
	}
	rc := C.gpu_moe_accumulate(
		C.GpuBuf(out), C.GpuBuf(expOuts), C.GpuBuf(weights),
		C.GpuBuf(bias), C.GpuBuf(indices),
		C.int(dim), C.int(nUsed), C.int(hb))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: moe_accumulate failed (%d)", rc)
	}
	return nil
}

// SwiGLU_OAI_Bias_MoE performs fused SwiGLU+bias with per-expert bias using GPU-side indices.
func SwiGLU_OAI_Bias_MoE(out, gate, up, gateBias, upBias, indices Buf,
	totalN int, alpha, limit float32, expDim int) error {
	rc := C.gpu_swiglu_oai_bias_moe(
		C.GpuBuf(out), C.GpuBuf(gate), C.GpuBuf(up),
		C.GpuBuf(gateBias), C.GpuBuf(upBias), C.GpuBuf(indices),
		C.int(totalN), C.float(alpha), C.float(limit), C.int(expDim))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: swiglu_oai_bias_moe failed (%d)", rc)
	}
	return nil
}

// MoEBiasAdd adds per-expert biases to data using GPU-side expert indices.
func MoEBiasAdd(data, bias, indices Buf, expDim, nUsed int) error {
	rc := C.gpu_moe_bias_add(C.GpuBuf(data), C.GpuBuf(bias), C.GpuBuf(indices),
		C.int(expDim), C.int(nUsed))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: moe_bias_add failed (%d)", rc)
	}
	return nil
}

// MoEFFNConf holds all parameters for a fused C-side MoE FFN dispatch.
type MoEFFNConf struct {
	c C.GpuMoEConf
}

func NewMoEFFNConf() *MoEFFNConf { return &MoEFFNConf{} }

func (mc *MoEFFNConf) SetScratch(ffnNorm, ffnOut, moeLogits, topkIdx, topkW,
	q8Input, gateScratch, upScratch, q8DownPacked, outScratch Buf) {
	mc.c.ffn_norm = C.GpuBuf(ffnNorm)
	mc.c.ffn_out = C.GpuBuf(ffnOut)
	mc.c.moe_logits = C.GpuBuf(moeLogits)
	mc.c.moe_topk_idx = C.GpuBuf(topkIdx)
	mc.c.moe_topk_w = C.GpuBuf(topkW)
	mc.c.q8_input = C.GpuBuf(q8Input)
	mc.c.gate_scratch = C.GpuBuf(gateScratch)
	mc.c.up_scratch = C.GpuBuf(upScratch)
	mc.c.q8_down_packed = C.GpuBuf(q8DownPacked)
	mc.c.out_scratch = C.GpuBuf(outScratch)
}

func (mc *MoEFFNConf) SetRouter(w Buf, rows, cols, rtype int, bias Buf) {
	mc.c.router_w = C.GpuBuf(w)
	mc.c.router_rows = C.int(rows)
	mc.c.router_cols = C.int(cols)
	mc.c.router_type = C.int(rtype)
	mc.c.router_bias = C.GpuBuf(bias)
}

func (mc *MoEFFNConf) SetExperts(gateW Buf, gateType, gateStride, gateBase int,
	upW Buf, upType, upStride, upBase int,
	downW Buf, downType, downStride int) {
	mc.c.gate_w = C.GpuBuf(gateW)
	mc.c.gate_type = C.int(gateType)
	mc.c.gate_stride = C.int(gateStride)
	mc.c.gate_base = C.int(gateBase)
	mc.c.up_w = C.GpuBuf(upW)
	mc.c.up_type = C.int(upType)
	mc.c.up_stride = C.int(upStride)
	mc.c.up_base = C.int(upBase)
	mc.c.down_w = C.GpuBuf(downW)
	mc.c.down_type = C.int(downType)
	mc.c.down_stride = C.int(downStride)
}

func (mc *MoEFFNConf) SetBiases(gateBias, upBias, downBias Buf) {
	mc.c.gate_bias = C.GpuBuf(gateBias)
	mc.c.up_bias = C.GpuBuf(upBias)
	mc.c.down_bias = C.GpuBuf(downBias)
}

func (mc *MoEFFNConf) SetConfig(dim, expDim, nExperts, nUsed, gatingFunc int,
	weightsNorm bool, weightsScale float32,
	isOAI bool, alpha, limit float32) {
	mc.c.dim = C.int(dim)
	mc.c.exp_dim = C.int(expDim)
	mc.c.n_experts = C.int(nExperts)
	mc.c.n_used = C.int(nUsed)
	mc.c.gating_func = C.int(gatingFunc)
	if weightsNorm {
		mc.c.weights_norm = 1
	}
	mc.c.weights_scale = C.float(weightsScale)
	if isOAI {
		mc.c.is_oai = 1
	}
	mc.c.alpha = C.float(alpha)
	mc.c.limit = C.float(limit)
}

// ForwardMoEFFN_C runs the entire MoE FFN in a single CGo call.
func ForwardMoEFFN_C(mc *MoEFFNConf) error {
	rc := C.gpu_forward_moe_ffn(&mc.c)
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: forward_moe_ffn failed (%d)", rc)
	}
	return nil
}

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

func (lc *LayerConf) SetAttnNormOnly(attnNorm Buf) {
	lc.c.attn_norm_w = C.GpuBuf(attnNorm)
}

func (lc *LayerConf) SetAttn(attnNorm Buf, wq, wk, wv, wo *GpuTensor,
	bq, bk, bv, bo Buf, qNorm, kNorm Buf) {
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
	lc.c.bo = C.GpuBuf(bo)
	lc.c.bv = C.GpuBuf(bv)
	lc.c.q_norm_w = C.GpuBuf(qNorm)
	lc.c.k_norm_w = C.GpuBuf(kNorm)
}

func (lc *LayerConf) SetAttnSinks(sinks Buf) {
	lc.c.attn_sinks = C.GpuBuf(sinks)
}

func (lc *LayerConf) SetSlidingWindow(w int) {
	lc.c.sliding_window = C.int(w)
}

func (lc *LayerConf) SetAttnLogitSoftcap(v float32) {
	lc.c.attn_logit_softcap = C.float(v)
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

// SetFFNMoE configures MoE layer norms without FFN weight tensors.
// The C side will compute pre-FFN residual+norm then return early (ffn_type=3).
func (lc *LayerConf) SetFFNMoE(ffnNorm Buf, postAttnNorm Buf) {
	lc.c.ffn_norm_w = C.GpuBuf(ffnNorm)
	lc.c.post_attn_norm_w = C.GpuBuf(postAttnNorm)
}

func (lc *LayerConf) SetKV(kCache, vCache Buf) {
	lc.c.k_cache = C.GpuBuf(kCache)
	lc.c.v_cache = C.GpuBuf(vCache)
}

func (lc *LayerConf) SetConfig(dim, headDim, numHeads, numKVHeads, kvDim int,
	rmsEps float32, ropeDim int, ropeNeox bool,
	ropeCosTable, ropeSinTable Buf,
	ffnType, residualType int) {
	lc.c.dim = C.int(dim)
	lc.c.head_dim = C.int(headDim)
	lc.c.num_heads = C.int(numHeads)
	lc.c.num_kv_heads = C.int(numKVHeads)
	lc.c.kv_dim = C.int(kvDim)
	lc.c.rms_eps = C.float(rmsEps)
	lc.c.rope_dim = C.int(ropeDim)
	if ropeNeox {
		lc.c.rope_neox = 1
	}
	lc.c.rope_cos_table = C.GpuBuf(ropeCosTable)
	lc.c.rope_sin_table = C.GpuBuf(ropeSinTable)
	lc.c.ffn_type = C.int(ffnType)
	lc.c.residual_type = C.int(residualType)
}

func (lc *LayerConf) SetCoreType(coreType int) {
	lc.c.core_type = C.int(coreType)
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

// ---------------------------------------------------------------------------
// Diffusion-specific operations
// ---------------------------------------------------------------------------

// BroadcastMul multiplies data[i] *= scale[i % dim] in-place.
func BroadcastMul(data, scale Buf, totalN, dim int) error {
	rc := C.gpu_broadcast_mul(C.GpuBuf(data), C.GpuBuf(scale), C.int(totalN), C.int(dim))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: broadcast_mul failed (%d)", rc)
	}
	return nil
}

// TanhGateResidual computes out[i] = residual[i] + data[i] * tanh(gate[i % dim]).
func TanhGateResidual(out, residual, data, gate Buf, totalN, dim int) error {
	rc := C.gpu_tanh_gate_residual(C.GpuBuf(out), C.GpuBuf(residual), C.GpuBuf(data),
		C.GpuBuf(gate), C.int(totalN), C.int(dim))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: tanh_gate_residual failed (%d)", rc)
	}
	return nil
}

// RoPE3D applies 3D RoPE using a precomputed cos/sin table.
func RoPE3D(vec, pe Buf, nPos, nHeads, headDim, peOffset, peStride int) error {
	rc := C.gpu_rope_3d(C.GpuBuf(vec), C.GpuBuf(pe), C.int(nPos), C.int(nHeads),
		C.int(headDim), C.int(peOffset), C.int(peStride))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: rope_3d failed (%d)", rc)
	}
	return nil
}

// AttentionFullF32 performs bidirectional multi-head attention (no causal mask).
func AttentionFullF32(out, q, k, v Buf, numHeads, numKVHeads, headDim, kvDim, seqLen int, scale float32) error {
	rc := C.gpu_attention_full_f32(C.GpuBuf(out), C.GpuBuf(q), C.GpuBuf(k), C.GpuBuf(v),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(kvDim),
		C.int(seqLen), C.float(scale))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: attention_full_f32 failed (%d)", rc)
	}
	return nil
}

// RMSNormHeadsBatch applies per-head RMSNorm across npos positions.
func RMSNormHeadsBatch(data, weight Buf, numHeads, headDim, npos int, eps float32) error {
	rc := C.gpu_rmsnorm_heads_batch(C.GpuBuf(data), C.GpuBuf(weight),
		C.int(numHeads), C.int(headDim), C.int(npos), C.float(eps))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: rmsnorm_heads_batch failed (%d)", rc)
	}
	return nil
}

// SSMConv1dSiLU performs single-step causal conv1d with state + SiLU.
func SSMConv1dSiLU(qkv, convState, convW Buf, channels, convK int) error {
	rc := C.gpu_ssm_conv1d_silu(C.GpuBuf(qkv), C.GpuBuf(convState), C.GpuBuf(convW),
		C.int(channels), C.int(convK))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: ssm_conv1d_silu failed (%d)", rc)
	}
	return nil
}

// SSMPreprocess computes decay/lr and L2-normalizes Q/K per head.
func SSMPreprocess(alpha, beta, ssma, dtBias, qkv Buf, numHeads, headKDim, keyDim int,
	rmsEps float32, hasDtBias bool) error {
	dtb := 0
	if hasDtBias {
		dtb = 1
	}
	rc := C.gpu_ssm_preprocess(C.GpuBuf(alpha), C.GpuBuf(beta), C.GpuBuf(ssma),
		C.GpuBuf(dtBias), C.GpuBuf(qkv), C.int(numHeads), C.int(headKDim),
		C.int(keyDim), C.float(rmsEps), C.int(dtb))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: ssm_preprocess failed (%d)", rc)
	}
	return nil
}

// SSMDeltaRule performs the delta rule state update and output computation.
func SSMDeltaRule(state, qkv, alpha, beta, y Buf, numHeads, headKDim, headVDim, keyDim int) error {
	rc := C.gpu_ssm_delta_rule(C.GpuBuf(state), C.GpuBuf(qkv), C.GpuBuf(alpha),
		C.GpuBuf(beta), C.GpuBuf(y), C.int(numHeads), C.int(headKDim), C.int(headVDim), C.int(keyDim))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: ssm_delta_rule failed (%d)", rc)
	}
	return nil
}

// SSMNormGate applies per-head RMSNorm + SiLU gate.
func SSMNormGate(y, z, normW Buf, numHeads, headVDim int, eps float32) error {
	rc := C.gpu_ssm_norm_gate(C.GpuBuf(y), C.GpuBuf(z), C.GpuBuf(normW),
		C.int(numHeads), C.int(headVDim), C.float(eps))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: ssm_norm_gate failed (%d)", rc)
	}
	return nil
}

// DeinterleaveQGate splits QFull into Q and QGate.
func DeinterleaveQGate(qfull, q, qgate Buf, numHeads, headDim int) error {
	rc := C.gpu_deinterleave_qgate(C.GpuBuf(qfull), C.GpuBuf(q), C.GpuBuf(qgate),
		C.int(numHeads), C.int(headDim))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: deinterleave_qgate failed (%d)", rc)
	}
	return nil
}

// SigmoidGate applies out[i] *= sigmoid(gate[i]).
func SigmoidGate(out, gate Buf, n int) error {
	rc := C.gpu_sigmoid_gate(C.GpuBuf(out), C.GpuBuf(gate), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: sigmoid_gate failed (%d)", rc)
	}
	return nil
}

// ---------------------------------------------------------------------------
// VAE-specific operations
// ---------------------------------------------------------------------------

// Conv2dF32 performs 2D convolution with F32 weights.
func Conv2dF32(out, in, weight, bias Buf, inCh, H, W, kH, kW, padH, padW, stride, outH, outW, outCh int) error {
	rc := C.gpu_conv2d_f32(C.GpuBuf(out), C.GpuBuf(in), C.GpuBuf(weight), C.GpuBuf(bias),
		C.int(inCh), C.int(H), C.int(W), C.int(kH), C.int(kW),
		C.int(padH), C.int(padW), C.int(stride), C.int(outH), C.int(outW), C.int(outCh))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: conv2d_f32 failed (%d)", rc)
	}
	return nil
}

// GroupNorm performs group normalization.
func GroupNorm(out, in, weight, bias Buf, channels, spatialSize, numGroups int, eps float32) error {
	rc := C.gpu_group_norm(C.GpuBuf(out), C.GpuBuf(in), C.GpuBuf(weight), C.GpuBuf(bias),
		C.int(channels), C.int(spatialSize), C.int(numGroups), C.float(eps))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: group_norm failed (%d)", rc)
	}
	return nil
}

// SiLU performs in-place SiLU activation.
func SiLU(data Buf, n int) error {
	rc := C.gpu_silu(C.GpuBuf(data), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: silu failed (%d)", rc)
	}
	return nil
}

// UpsampleNearest performs 2× nearest-neighbor upsampling.
func UpsampleNearest(out, in Buf, channels, H, W int) error {
	rc := C.gpu_upsample_nearest(C.GpuBuf(out), C.GpuBuf(in), C.int(channels), C.int(H), C.int(W))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: upsample_nearest failed (%d)", rc)
	}
	return nil
}

// SpatialAttention performs single-head spatial self-attention for channel-major [C, spatial] layout.
func SpatialAttention(out, q, k, v Buf, channels, spatial int, scale float32) error {
	rc := C.gpu_spatial_attention(C.GpuBuf(out), C.GpuBuf(q), C.GpuBuf(k), C.GpuBuf(v),
		C.int(channels), C.int(spatial), C.float(scale))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: spatial_attention failed (%d)", rc)
	}
	return nil
}

// BatchRoPE applies RoPE to Q and K for npos positions starting at startPos.
func BatchRoPE(q, k, cosTable, sinTable Buf, numHeads, numKVHeads, headDim, ropeDim, startPos int,
	neox bool, npos int) error {
	n := 0
	if neox {
		n = 1
	}
	rc := C.gpu_batch_rope(C.GpuBuf(q), C.GpuBuf(k), C.GpuBuf(cosTable), C.GpuBuf(sinTable),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(ropeDim),
		C.int(startPos), C.int(n), C.int(npos))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: batch_rope failed (%d)", rc)
	}
	return nil
}

// BatchKVStore bulk-copies K and V for npos positions into the cache starting at startPos.
func BatchKVStore(kCache, vCache, k, v Buf, startPos, kvDim, npos int) error {
	rc := C.gpu_batch_kv_store(C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.GpuBuf(k), C.GpuBuf(v),
		C.int(startPos), C.int(kvDim), C.int(npos))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: batch_kv_store failed (%d)", rc)
	}
	return nil
}

// BatchKVStoreF16 converts float32 K/V to packed half2 and stores npos positions in FP16 KV cache.
func BatchKVStoreF16(kCache, vCache, k, v Buf, startPos, kvDim, npos int) error {
	rc := C.gpu_batch_kv_store_f16(C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.GpuBuf(k), C.GpuBuf(v),
		C.int(startPos), C.int(kvDim), C.int(npos))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: batch_kv_store_f16 failed (%d)", rc)
	}
	return nil
}

// BatchAttention performs causal attention for npos positions.
// startSeqLen is the sequence length for the first position (startPos + 1).
func BatchAttention(out, q, kCache, vCache Buf,
	numHeads, numKVHeads, headDim, kvDim, startSeqLen int, scale float32, npos int) error {
	rc := C.gpu_batch_attention(C.GpuBuf(out), C.GpuBuf(q),
		C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(kvDim),
		C.int(startSeqLen), C.float(scale), C.int(npos))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: batch_attention failed (%d)", rc)
	}
	return nil
}

// BatchAttentionF16 performs tiled causal attention for npos positions with FP16 KV cache.
func BatchAttentionF16(out, q, kCache, vCache Buf,
	numHeads, numKVHeads, headDim, kvDim, startSeqLen int, scale, softcap float32, npos int) error {
	rc := C.gpu_batch_attention_f16(C.GpuBuf(out), C.GpuBuf(q),
		C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(kvDim),
		C.int(startSeqLen), C.float(scale), C.float(softcap), C.int(npos))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: batch_attention_f16 failed (%d)", rc)
	}
	return nil
}

// BatchAttentionTiledF32 performs tiled causal attention for npos positions with FP32 KV cache.
func BatchAttentionTiledF32(out, q, kCache, vCache Buf,
	numHeads, numKVHeads, headDim, kvDim, startSeqLen int, scale, softcap float32, npos int) error {
	rc := C.gpu_batch_attention_tiled_f32(C.GpuBuf(out), C.GpuBuf(q),
		C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(kvDim),
		C.int(startSeqLen), C.float(scale), C.float(softcap), C.int(npos))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: batch_attention_tiled_f32 failed (%d)", rc)
	}
	return nil
}

// BatchAddBias adds a per-position bias using a scratch buffer for expansion.
func BatchAddBias(dst, bias, scratch Buf, elemsPerPos, npos int) error {
	rc := C.gpu_batch_add_bias2(C.GpuBuf(dst), C.GpuBuf(bias), C.GpuBuf(scratch),
		C.int(elemsPerPos), C.int(npos))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: batch_add_bias failed (%d)", rc)
	}
	return nil
}
