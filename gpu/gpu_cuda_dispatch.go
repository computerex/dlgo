//go:build cgo && vulkan

// CUDA backend dispatch for the gpu package.
//
// The gpu package's primitives are implemented in C (Vulkan). This file lets a
// caller opt into the CUDA backend at runtime with DLGO_GPU_BACKEND=cuda. When
// active, each primitive the dense forward path uses is routed to gpu/cuda; the
// higher-level Go orchestration (forward.go, pipeline.go, ...) is unchanged
// because it only calls these primitives.
//
// Only one backend is ever initialized: if CUDA is selected and initializes,
// the Vulkan C runtime is never brought up, so buffers are plain CUDA device
// pointers (uint64) throughout. Functions that are not routed here would call
// into the (uninitialized) Vulkan runtime and fail loudly rather than silently
// corrupt — they are simply not on the dense path.

package gpu

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/gpu/cuda"
)

// cudaActive reports whether the CUDA backend is the active backend.
var cudaActive bool

// cudaBackendName is the device name when CUDA is active.
var cudaBackendName string

// tryInitCUDA initializes the CUDA backend if DLGO_GPU_BACKEND selects it.
// Returns true if CUDA is now the active backend.
func tryInitCUDA() bool {
	switch os.Getenv("DLGO_GPU_BACKEND") {
	case "cuda", "CUDA", "Cuda":
	default:
		return false
	}
	if err := cuda.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "[gpu] CUDA init failed, falling back to Vulkan: %v\n", err)
		return false
	}
	cudaActive = true
	cudaBackendName = cuda.DeviceName()
	fmt.Fprintf(os.Stderr, "[gpu] CUDA backend active: %s\n", cudaBackendName)
	return true
}

// BackendName returns the active GPU backend ("cuda" or "vulkan").
func BackendName() string {
	if cudaActive {
		return "cuda"
	}
	return "vulkan"
}

// ---------------------------------------------------------------------------
// Memory primitives
// ---------------------------------------------------------------------------

func cudaAlloc(size uint64) Buf {
	if size == 0 {
		return 0
	}
	p, err := cuda.Alloc(size)
	if err != nil {
		return 0
	}
	return Buf(p)
}

func cudaAllocE(size uint64) (Buf, error) {
	if size == 0 {
		return 0, nil
	}
	p, err := cuda.Alloc(size)
	if err != nil {
		return 0, err
	}
	return Buf(p), nil
}

func cudaFree(b Buf) {
	if b != 0 {
		_ = cuda.Free(uint64(b))
	}
}

func cudaUpload(dst Buf, src []byte) error { return cuda.Upload(uint64(dst), src) }
func cudaUploadAt(dst Buf, byteOff uint64, src []byte) error {
	return cuda.UploadAt(uint64(dst), byteOff, src)
}
func cudaDownload(src Buf, dst []byte) error { return cuda.Download(dst, uint64(src)) }

// Thin wrappers so gpu_cgo.go (a different file) can use the CUDA driver
// without importing the cuda package itself.
func cudaShutdown()                    { cuda.Shutdown() }
func cudaTotalMem() uint64             { return cuda.TotalMem() }
func cudaFreeMem() uint64              { return cuda.FreeMem() }
func cudaAllocated() uint64            { return cuda.Allocated() }
func cudaMemsetZero(dst Buf, n uint64) { _ = cuda.Memset8(uint64(dst), 0, n) }
func cudaStreamSync()                  { _ = cuda.Sync() }

// ---------------------------------------------------------------------------
// Compute primitives
// ---------------------------------------------------------------------------

func cudaMatVec(out, weights, x Buf, rows, cols int, qtype uint32) error {
	switch qtype {
	case 12: // Q4_K
		return cuda.GEMVQ4K(uint64(weights), uint64(x), uint64(out), rows, cols)
	case 14: // Q6_K
		return cuda.GEMVQ6K(uint64(weights), uint64(x), uint64(out), rows, cols)
	case 0: // F32
		return cuda.GEMVF32(uint64(weights), uint64(x), uint64(out), rows, cols)
	case 1: // F16
		return cuda.GEMVF16(uint64(weights), uint64(x), uint64(out), rows, cols)
	case 60: // canonical Affine8 (unified quant path)
		return cuda.GEMVAffine8(uint64(weights), uint64(x), uint64(out), rows, cols)
	case 61: // canonical Affine4 (unified quant path)
		return cuda.GEMVAffine4(uint64(weights), uint64(x), uint64(out), rows, cols)
	default:
		return fmt.Errorf("gpu/cuda: matvec qtype %d not implemented", qtype)
	}
}

func cudaRMSNorm(out, x, w Buf, n int, eps float32) error {
	return cuda.RMSNorm(uint64(x), uint64(w), uint64(out), n, eps)
}

func cudaLayerNorm(out, x, w, b Buf, n int, eps float32) error {
	return cuda.LayerNorm(uint64(x), uint64(w), uint64(b), uint64(out), n, eps)
}

func cudaRMSNormHeads(data, w Buf, numHeads, headDim int, eps float32) error {
	return cuda.RMSNormHeads(uint64(data), uint64(w), numHeads, headDim, eps)
}

func cudaAdd(out, a, b Buf, n int) error {
	return cuda.Add(uint64(a), uint64(b), uint64(out), n)
}

func cudaScale(buf Buf, s float32, n int) error { return cuda.Scale(uint64(buf), s, n) }

func cudaAddRMSNorm(normOut, sumOut, a, b, w Buf, n int, eps float32) error {
	return cuda.AddRMSNorm(uint64(a), uint64(b), uint64(w), uint64(sumOut), uint64(normOut), n, eps)
}

func cudaRoPE(q, k, cosT, sinT Buf, numHeads, numKVHeads, headDim, ropeDim, pos int, neox bool) error {
	return cuda.RoPE(uint64(q), uint64(k), uint64(cosT), uint64(sinT),
		numHeads, numKVHeads, headDim, ropeDim, pos, neox)
}

func cudaSwiGLU(out, gate, up Buf, n int) error {
	return cuda.SiluMul(uint64(gate), uint64(up), uint64(out), n)
}

func cudaKVStoreF16(kCache, vCache, k, v Buf, pos, kvDim int) error {
	return cuda.KVStoreF16(uint64(k), uint64(v), uint64(kCache), uint64(vCache), pos, kvDim, 0)
}

func cudaAttentionF16(out, q, kCache, vCache Buf, numHeads, numKVHeads, headDim, kvDim, seqLen, winStart int, scale, softcap float32) error {
	return cuda.AttentionF16(uint64(q), uint64(kCache), uint64(vCache), uint64(out),
		numHeads, numKVHeads, headDim, kvDim, seqLen, winStart, scale, softcap, 0)
}

// CUDA-graph capture/replay of the per-token layer loop (removes the ~10us/launch
// WDDM dispatch cost, which is ~4.9 ms of an 8.9 ms token).
func cudaGraphReady() bool { return cuda.GraphReady() }

func cudaGraphCapture() error {
	Sync()
	return cuda.GraphCapture()
}

func cudaGraphCaptureEnd() error { return cuda.GraphCaptureEnd() }

func cudaGraphReplay(pos, seqLen, winStart, kvCap int) error {
	return cuda.GraphReplay(pos, seqLen, winStart, kvCap)
}

func cudaGraphDiscard() { cuda.GraphDiscard() }

// ---------------------------------------------------------------------------
// Batched (prompt-processing) prefill primitives.
//
// The batched GEMM dequantizes each weight tile into shared memory for every
// k-step, so the activation matrix is converted to float16 ONCE per GEMM instead
// of once per output tile. The staging buffer is grow-only: a chunk's shape
// repeats for every layer and every chunk of the prompt.
// ---------------------------------------------------------------------------

var (
	cudaBatchX    Buf
	cudaBatchXCap uint64
)

// cudaBatchXBuf returns a float16 activation buffer of at least bytes. Like the
// rest of the package's GPU state this assumes a single pipeline at a time.
func cudaBatchXBuf(bytes uint64) (Buf, error) {
	if bytes <= cudaBatchXCap {
		return cudaBatchX, nil
	}
	if cudaBatchX != 0 {
		cudaFree(cudaBatchX)
		cudaBatchX, cudaBatchXCap = 0, 0
	}
	b, err := cudaAllocE(bytes)
	if err != nil {
		return 0, err
	}
	cudaBatchX, cudaBatchXCap = b, bytes
	return b, nil
}

// cudaFreeBatchScratch releases the batched-prefill staging buffer.
func cudaFreeBatchScratch() {
	if cudaBatchX != 0 {
		cudaFree(cudaBatchX)
	}
	cudaBatchX, cudaBatchXCap = 0, 0
}

// cudaBatchMatVec is the npos-token form of the projection:
// out[pos][row] = sum_c x[pos][c] * W[row][c], both token-major like the C batch
// layer, so it drops straight into the same residual/norm chain.
func cudaBatchMatVec(out, weights, x Buf, rows, cols, npos int, qtype uint32) error {
	if npos <= 0 || rows <= 0 || cols <= 0 {
		return nil
	}
	if cols%32 != 0 {
		return fmt.Errorf("gpu/cuda: batched matvec needs cols%%32==0 (got %d)", cols)
	}
	xh, err := cudaBatchXBuf(uint64(npos * cols * 2))
	if err != nil {
		return err
	}
	if err := cuda.F32ToHalf(uint64(x), uint64(xh), npos*cols); err != nil {
		return err
	}
	switch qtype {
	case 61: // canonical Affine4
		return cuda.GemmAffine4F16(uint64(weights), uint64(xh), uint64(out), rows, cols, npos)
	case 60: // canonical Affine8
		return cuda.GemmAffine8F16(uint64(weights), uint64(xh), uint64(out), rows, cols, npos)
	default:
		return fmt.Errorf("gpu/cuda: batched matvec qtype %d unsupported (need Affine4=61 or Affine8=60)", qtype)
	}
}

func cudaBatchRMSNorm(out, x, w Buf, n, npos int, eps float32) error {
	return cuda.BatchRMSNorm(uint64(out), uint64(x), uint64(w), n, npos, eps)
}

func cudaBatchAddRMSNorm(normOut, sumOut, a, b, w Buf, n, npos int, eps float32) error {
	return cuda.BatchAddRMSNorm(uint64(normOut), uint64(sumOut), uint64(a), uint64(b),
		uint64(w), n, npos, eps)
}

func cudaBatchRMSNormHeads(data, w Buf, numHeads, headDim, npos int, eps float32) error {
	return cuda.BatchRMSNormHeads(uint64(data), uint64(w), numHeads, headDim, npos, eps)
}

func cudaBatchRoPE(q, k, cosT, sinT Buf, numHeads, numKVHeads, headDim, ropeDim, startPos int,
	neox bool, npos int) error {
	return cuda.BatchRoPE(uint64(q), uint64(k), uint64(cosT), uint64(sinT),
		numHeads, numKVHeads, headDim, ropeDim, startPos, neox, npos)
}

func cudaBatchKVStoreF16(kIn, vIn, kOut, vOut Buf, startPos, kvDim, npos int) error {
	return cuda.BatchKVStoreF16(uint64(kIn), uint64(vIn), uint64(kOut), uint64(vOut),
		startPos, kvDim, npos)
}

func cudaBatchAttentionF16(out, q, kCache, vCache Buf, numHeads, numKVHeads, headDim, kvDim,
	startPos, npos int, scale, softcap float32) error {
	return cuda.AttnPrefillF16(uint64(out), uint64(q), uint64(kCache), uint64(vCache),
		numHeads, numKVHeads, headDim, kvDim, startPos, npos, scale, softcap)
}

// cudaPrimeAttentionPartials reserves the split-KV attention scratch up front so
// that a later CUDA-graph capture never has to allocate.
func cudaPrimeAttentionPartials(numHeads, numKVHeads, headDim int) error {
	return cuda.PrimeAttentionPartials(numHeads, numKVHeads, headDim)
}

// cudaCopyRegion is the device-to-device form of CopyRegion.
func cudaCopyRegion(dst Buf, dstOff uint64, src Buf, srcOff, size uint64) error {
	return cuda.CopyRegionBytes(uint64(dst), dstOff, uint64(src), srcOff, size)
}
