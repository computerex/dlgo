//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/quant"
)

// GpuTensor mirrors core.QuantizedTensor but with data on the GPU.
type GpuTensor struct {
	Buf  Buf
	Type uint32
	Rows int
	Cols int
}

// UploadTensor copies a QuantizedTensor's raw data to GPU memory.
func UploadTensor(qt *core.QuantizedTensor) (*GpuTensor, error) {
	if qt == nil {
		return nil, nil
	}

	var data []byte
	var size uint64

	if qt.FP32Data != nil {
		size = uint64(len(qt.FP32Data) * 4)
		buf := Alloc(size)
		if buf == 0 {
			return nil, fmt.Errorf("gpu: alloc failed for tensor %dx%d", qt.Rows, qt.Cols)
		}
		if err := UploadF32(buf, qt.FP32Data); err != nil {
			Free(buf)
			return nil, err
		}
		return &GpuTensor{Buf: buf, Type: 0, Rows: qt.Rows, Cols: qt.Cols}, nil
	}

	data = qt.Data
	size = uint64(len(data))
	buf := Alloc(size)
	if buf == 0 {
		return nil, fmt.Errorf("gpu: alloc failed for tensor %dx%d (%d bytes)", qt.Rows, qt.Cols, size)
	}
	if err := Upload(buf, data); err != nil {
		Free(buf)
		return nil, err
	}
	return &GpuTensor{Buf: buf, Type: qt.Type, Rows: qt.Rows, Cols: qt.Cols}, nil
}

// uploadBytes allocates and uploads a raw byte slice as a GpuTensor.
func uploadBytes(data []byte, typ uint32, rows, cols int) (*GpuTensor, error) {
	buf := Alloc(uint64(len(data)))
	if buf == 0 {
		return nil, fmt.Errorf("gpu: alloc failed for tensor %dx%d (%d bytes)", rows, cols, len(data))
	}
	if err := Upload(buf, data); err != nil {
		Free(buf)
		return nil, err
	}
	return &GpuTensor{Buf: buf, Type: typ, Rows: rows, Cols: cols}, nil
}

// Affine8BytesFor returns the canonical Affine8 byte size for a rows×cols weight.
func Affine8BytesFor(rows, cols int) uint64 {
	return uint64(rows) * uint64(cols) / quant.Affine8GroupSize * quant.Affine8BlockBytes
}

// UploadTensorAffine8 repacks a QuantizedTensor into the canonical Affine8 format
// (see quant/affine8.go) and uploads it. After this, the weight tensor — whatever
// its on-disk quantization scheme — is served by the single unified GEMM kernel
// (gpu_gemm_affine8); no per-quant code is needed in the hot path.
func UploadTensorAffine8(qt *core.QuantizedTensor) (*GpuTensor, error) {
	if qt == nil {
		return nil, nil
	}
	if qt.FP32Data != nil {
		return uploadBytes(quant.QuantizeAffine8(qt.FP32Data), QuantTypeAffine8, qt.Rows, qt.Cols)
	}
	enc, err := quant.RepackAffine8(qt.Data, qt.Type, qt.Rows*qt.Cols)
	if err != nil {
		return nil, fmt.Errorf("affine8 repack %dx%d: %w", qt.Rows, qt.Cols, err)
	}
	return uploadBytes(enc, QuantTypeAffine8, qt.Rows, qt.Cols)
}

// Affine4BytesFor returns the canonical Affine4 byte size for a rows×cols weight.
func Affine4BytesFor(rows, cols int) uint64 {
	return uint64(rows) * uint64(cols) / quant.Affine4GroupSize * quant.Affine4BlockBytes
}

// UploadTensorAffine4 repacks a QuantizedTensor into the byte-lean canonical
// Affine4 format (see quant/affine4.go) and uploads it. Used for sub-8-bit
// sources where Affine8's extra bytes would cost decode bandwidth.
func UploadTensorAffine4(qt *core.QuantizedTensor) (*GpuTensor, error) {
	if qt == nil {
		return nil, nil
	}
	if qt.FP32Data != nil {
		return uploadBytes(quant.QuantizeAffine4(qt.FP32Data), QuantTypeAffine4, qt.Rows, qt.Cols)
	}
	enc, err := quant.RepackAffine4(qt.Data, qt.Type, qt.Rows*qt.Cols)
	if err != nil {
		return nil, fmt.Errorf("affine4 repack %dx%d: %w", qt.Rows, qt.Cols, err)
	}
	return uploadBytes(enc, QuantTypeAffine4, qt.Rows, qt.Cols)
}

// gpuNativeQuant reports whether the GPU has a native per-quant kernel for the
// scheme (a matvec_*/coopmat kernel selected in gpu_matvec / gpu_batch_matvec).
// Any scheme NOT listed here would otherwise fall through to the CPU
// QMatVecMulParallel fallback; uploadProjWeight repacks those to Affine8 at load
// so they stay on the GPU. (quant.Dequantize — hence RepackAffine8 — covers
// every GGML type, so this fallback is always available.)
func gpuNativeQuant(qtype uint32) bool {
	switch qtype {
	case 0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 29, 30, 34, 39:
		return true
	}
	return false
}

// llmAffine8Quant reports whether a scheme should ALSO get an Affine8 copy for
// the batched prefill path on top of its native weights. True for schemes with
// NO cooperative-matrix GEMM: in batched prefill they otherwise fall back to a
// per-token matvec that re-reads the whole weight matrix once per prompt token
// (the SystemOne 9B Q3_K FFN bottleneck). Q4_K/Q5_K/Q6_K/BF16 already have
// byte-optimal coopmat GEMMs and are left untouched (decode bytes matter).
func llmAffine8Quant(qtype uint32) bool {
	switch qtype {
	case 12, 13, 14, 30: // Q4_K, Q5_K, Q6_K, BF16
		return false
	}
	return true
}

// llmUseBatchAffine8 reports whether to build Affine8 copies of the prefill
// projection/FFN weights. Opt-in via DLGO_LLM_AFFINE8=1.
//
// These copies are consumed ONLY by the batched prefill path
// (BuildBatchLayerConfs → gpu_forward_layer_batch → the one unified coopmat
// GEMM). Single-token decode keeps the original per-quant weights, so decode
// speed is unchanged; the cost is the extra Affine8 VRAM (one copy of the
// affected tensors).
func llmUseBatchAffine8() bool {
	return os.Getenv("DLGO_LLM_AFFINE8") == "1"
}

// uploadProjWeight uploads a projection/FFN weight tensor in its ORIGINAL
// quantization when the GPU has a native kernel for it; otherwise it repacks to
// the canonical Affine8 format so the scheme still runs on the GPU instead of
// falling back to the CPU (no unsupported quants).
func uploadProjWeight(qt *core.QuantizedTensor) (*GpuTensor, error) {
	if qt == nil {
		return nil, nil
	}
	// Canonical formats use 32-element blocks; each weight row must be a whole
	// number of blocks for the row-major matvec/GEMM indexing to be correct.
	shapeOK := qt.Cols%quant.Affine8GroupSize == 0
	// Unified canonical path (DLGO_UNIFIED_QUANT=1): EVERY projected weight is
	// repacked to Affine4 (sub-8-bit sources, byte-lean decode) or Affine8
	// (≥5.5-bit sources). The whole model then runs on exactly two matvec kernels
	// (matvec_affine4/8) + two GEMMs (gemm_affine4/8_coopmat) — no per-quant code.
	if llmUnifiedQuant() && qt.FP32Data == nil && shapeOK {
		if gt, err := uploadCanonical(qt); err == nil {
			return gt, nil
		}
	}
	// Forced Affine8 (DLGO_DECODE_AFFINE8=1): A/B knob pinning everything to
	// Affine8 regardless of source bit width.
	if llmDecodeAffine8() && qt.FP32Data == nil && shapeOK {
		if gt, err := UploadTensorAffine8(qt); err == nil {
			return gt, nil
		}
	}
	// Otherwise: native per-quant kernel when one exists, else Affine8 fallback
	// (so there are no unsupported quants).
	if qt.FP32Data == nil && !gpuNativeQuant(qt.Type) && shapeOK {
		if gt, err := UploadTensorAffine8(qt); err == nil {
			return gt, nil
		}
	}
	return UploadTensor(qt)
}

// canonicalQuant4 reports whether a GGML type should be stored as the byte-lean
// Affine4 (≤ ~5 bits/weight) rather than Affine8. Keeping sub-5-bit sources on
// Affine4 holds decode weight bytes close to the native schemes while still
// collapsing every source onto one format family.
func canonicalQuant4(qtype uint32) bool {
	switch qtype {
	case 6, 7, 8, 9, 13, 14, 15, 30, 0, 1: // Q5_0,Q5_1,Q8_0,Q8_1,Q5_K,Q6_K,Q8_K,BF16,F32,F16
		return false
	}
	return true // Q2_K,Q3_K,Q4_0,Q4_1,Q4_K,IQ*,TQ*,MXFP4
}

// uploadCanonical repacks a weight tensor to its canonical format (Affine4 or
// Affine8, chosen by source bit-width).
func uploadCanonical(qt *core.QuantizedTensor) (*GpuTensor, error) {
	if canonicalQuant4(qt.Type) {
		return UploadTensorAffine4(qt)
	}
	return UploadTensorAffine8(qt)
}

// llmUnifiedQuant reports whether to repack every projected weight to a canonical
// Affine format (Affine4 for ≤5-bit sources, else Affine8) for BOTH decode and
// prefill. This is the DEFAULT dense-path configuration: the whole model runs on
// the two unified matvec kernels + two unified GEMMs, so no per-quant kernel is
// on the hot path. Set DLGO_UNIFIED_QUANT=0 to fall back to the legacy per-quant
// kernels (used to A/B and to exercise the per-quant code paths in tests).
func llmUnifiedQuant() bool { return os.Getenv("DLGO_UNIFIED_QUANT") != "0" }

// llmDecodeAffine8 reports whether to force all projected weights onto the
// canonical Affine8 format for BOTH decode and prefill (opt-in via
// DLGO_DECODE_AFFINE8=1) — the A/B knob for the Affine8-only configuration.
func llmDecodeAffine8() bool { return os.Getenv("DLGO_DECODE_AFFINE8") == "1" }

// uploadAffine8Copy repacks+uploads an Affine8 copy of qt for the batched
// prefill path, or returns nil when it is not warranted (flag off, native
// coopmat scheme, FP32, or unsupported shape).
func uploadAffine8Copy(qt *core.QuantizedTensor) *GpuTensor {
	if qt == nil || !llmUseBatchAffine8() || qt.FP32Data != nil || !llmAffine8Quant(qt.Type) {
		return nil
	}
	if (qt.Rows*qt.Cols)%quant.Affine8GroupSize != 0 {
		return nil
	}
	gt, err := UploadTensorAffine8(qt)
	if err != nil {
		return nil
	}
	return gt
}

// uploadWeightPair returns (base, affine8copy) for a projection/FFN weight:
// base is the native tensor (or Affine8 when the scheme has no native kernel);
// the copy is the optional Affine8 prefill variant, built only when the flag is
// on and the scheme has no coopmat GEMM. When base is already Affine8 the copy
// is nil — prefill then uses base via pickA8.
func uploadWeightPair(qt *core.QuantizedTensor) (*GpuTensor, *GpuTensor, error) {
	base, err := uploadProjWeight(qt)
	if err != nil || base == nil || base.Type == QuantTypeAffine8 || base.Type == QuantTypeAffine4 {
		return base, nil, err
	}
	return base, uploadAffine8Copy(qt), nil
}

// UploadF32Slice uploads a float32 slice to a new GPU buffer.
func UploadF32Slice(data []float32) (Buf, error) {
	if len(data) == 0 {
		return 0, nil
	}
	buf := Alloc(uint64(len(data) * 4))
	if buf == 0 {
		return 0, fmt.Errorf("gpu: alloc failed for %d floats", len(data))
	}
	if err := UploadF32(buf, data); err != nil {
		Free(buf)
		return 0, err
	}
	return buf, nil
}

// BytesPerRow returns the byte size of one row for the tensor's quant type.
func (gt *GpuTensor) BytesPerRow() int {
	if gt.Type == 0 {
		return gt.Cols * 4
	}
	return quant.BytesForN(gt.Type, gt.Cols)
}

// GpuLayer holds GPU buffers for one transformer layer's weights.
type GpuLayer struct {
	AttnNorm     Buf
	AttnNormBias Buf
	Wq           *GpuTensor
	Wk           *GpuTensor
	Wv           *GpuTensor
	Wo           *GpuTensor
	Bq, Bk, Bv   Buf
	Bo           Buf
	AttnQNorm    Buf
	AttnKNorm    Buf
	PostAttnNorm Buf
	FFNNorm      Buf
	FFNGate      *GpuTensor
	FFNUp        *GpuTensor
	FFNDown      *GpuTensor
	FFNUpBias    Buf
	FFNDownBias  Buf
	PostFFNNorm  Buf

	// Affine8 copies of the projection/FFN weights used ONLY by the batched
	// prefill path (BuildBatchLayerConfs → the one unified coopmat GEMM). nil
	// unless DLGO_LLM_AFFINE8=1 and the source quant has no coopmat GEMM.
	// Single-token decode keeps using the per-quant fields above.
	WqA8, WkA8, WvA8, WoA8 *GpuTensor
	GateA8, UpA8, DownA8   *GpuTensor

	// MoE expert weights (packed, GPU-supported)
	FFNRouter     *GpuTensor // [expertCount × dim] router/gating network
	FFNRouterBias Buf        // [expertCount] optional bias
	FFNGateExps   *GpuTensor // [expertCount*expertFFNDim × dim] packed gate
	FFNUpExps     *GpuTensor // [expertCount*expertFFNDim × dim] packed up
	FFNGateUpExps *GpuTensor // [expertCount*2*expertFFNDim × dim] fused gate+up
	FFNDownExps   *GpuTensor // [expertCount*dim × expertFFNDim] packed down
	// MoE expert biases (packed float32 on GPU)
	FFNGateExpsBias Buf // [expertCount*expertFFNDim] packed expert gate bias
	FFNUpExpsBias   Buf // [expertCount*expertFFNDim] packed expert up bias
	FFNDownExpsBias Buf // [expertCount*dim] packed expert down bias

	// MoE shared expert weights
	FFNGateShared   *GpuTensor // [sharedFFNDim × dim]
	FFNUpShared     *GpuTensor // [sharedFFNDim × dim]
	FFNDownShared   *GpuTensor // [dim × sharedFFNDim]
	FFNRouterShared Buf        // [dim] shared expert gate (float32)
	IsMoE           bool
	MoEOnGPU        bool // true if expert weights are on GPU

	AttnSinks Buf // [num_heads] learned sink logits (0 if not used)

	OnGPU   bool // true if this layer's weights are on GPU
	CPUAttn bool // true if attention matmuls need CPU fallback (unsupported qtype)

	// SSM (Gated Delta Net) weights and per-layer state on GPU
	SSMInProj  *GpuTensor // [qkvDim × dim]
	SSMGate    *GpuTensor // [valueDim × dim] (AttnGate)
	SSMAlpha   *GpuTensor // [numHeads × dim]
	SSMBeta    *GpuTensor // [numHeads × dim]
	SSMConv1dW Buf        // [channels × convK] float32
	SSMA       Buf        // [numHeads] float32
	SSMDtBias  Buf        // [numHeads] float32 (may be 0)
	SSMNorm    Buf        // [headVDim] float32 (shared across heads)
	SSMOut     *GpuTensor // [dim × valueDim]
	SSMState   Buf        // [numHeads × headKDim × headVDim] float32 (persistent)
	SSMConvBuf Buf        // [convK × channels] float32 (persistent)

	// Affine8 copies of the SSM projection weights, used ONLY by batched prefill
	// (same rationale as the Wq/WoA8/etc. fields). nil unless DLGO_LLM_AFFINE8=1
	// and the source quant has no coopmat GEMM (e.g. Qwen3.5's attn_gate=Q3_K).
	// Single-token decode keeps the per-quant fields above.
	SSMInProjA8, SSMGateA8, SSMAlphaA8, SSMBetaA8, SSMOutA8 *GpuTensor
}

// GpuModel holds all model weights on the GPU.
type GpuModel struct {
	TokenEmbed     *GpuTensor
	OutputNorm     Buf
	OutputNormBias Buf
	Output         *GpuTensor
	OutputBias     Buf
	Layers         []GpuLayer
	NumGPULayers   int // actual number of layers uploaded to GPU (may be < requested)
}

// GpuRunState holds GPU buffers for intermediate activations during inference.
type GpuRunState struct {
	X        Buf // [dim]
	XNorm    Buf // [dim]
	Q        Buf // [qDim]
	K        Buf // [kvDim]
	V        Buf // [kvDim]
	AttnOut  Buf // [qDim]
	AttnProj Buf // [dim]
	FFNIn    Buf // [dim]
	FFNNorm  Buf // [dim]
	Gate     Buf // [ffnDim]
	Up       Buf // [ffnDim]
	Hidden   Buf // [ffnDim]
	FFNOut   Buf // [dim]
	Logits   Buf // [vocabSize]

	// SSM (Gated Delta Net) scratch buffers
	SSMQKV   Buf // [qkvDim] SSM in-projection output
	SSMZ     Buf // [valueDim] gate projection output
	SSMAlpha Buf // [numHeads] alpha scratch
	SSMBeta  Buf // [numHeads] beta scratch
	SSMY     Buf // [valueDim] SSM output
	SSMPred  Buf // [valueDim] prediction scratch for decomposed delta rule

	// GatedQ attention scratch buffers
	QFull Buf // [2*qDim] fused Q+gate output
	QGate Buf // [qDim] attention gate values

	// MoE scratch buffers (allocated on first use)
	MoELogits  Buf // [expertCount] router logits
	MoETopKIdx Buf // [expertUsedCount] top-K indices (float-encoded)
	MoETopKW   Buf // [expertUsedCount] top-K weights
	// Per-expert parallel buffers (one per active expert for batched dispatch)
	MoEGates    []Buf // [expertUsedCount][expertFFNDim]
	MoEUps      []Buf // [expertUsedCount][expertFFNDim]
	MoEHiddens  []Buf // [expertUsedCount][expertFFNDim]
	MoEOuts     []Buf // [expertUsedCount][dim]
	MoEShGate   Buf   // [sharedFFNDim] shared expert gate
	MoEShUp     Buf   // [sharedFFNDim] shared expert up
	MoEShHidden Buf   // [sharedFFNDim] shared expert hidden
	MoEShOut    Buf   // [dim] shared expert output

	// dp4a acceleration for MoE expert projections
	MoEQ8_1Scratch    Buf   // Q8_1 quantized input (dim) for dp4a gate/up projections
	MoEQ8_1DownPacked Buf   // packed Q8_1 buffer [nUsed * q8_blocks(expDim) * 36]
	MoEQ8_1DownBufs   []Buf // per-expert Q8_1 scratch (expDim) for dp4a down projections (fallback)
	MoEUseDp4a        bool  // true if dp4a should be used for MoE expert projections

	// Fused MoE dp4a buffers (interleaved [nUsed * dim/expDim])
	MoEGateScratch   Buf // [nUsed * expDim] interleaved gate outputs
	MoEUpScratch     Buf // [nUsed * expDim] interleaved up outputs
	MoEHiddenScratch Buf // [nUsed * expDim] interleaved hidden outputs
	MoEOutScratch    Buf // [nUsed * dim] interleaved down outputs
	MoEWeightsBuf    Buf // [nUsed] weights for accumulation

	// CPU scratch buffers used for correctness fallbacks when a quant type
	// has no GPU kernel yet (for example Q3_K on Vulkan).
	ScratchIn  []float32
	ScratchOut []float32
	ScratchAux []float32
	Pool       *blas.Pool
}

// NewGpuRunState allocates all GPU activation buffers.
// Returns an error if any VRAM allocation fails (OOM).
func NewGpuRunState(dim, qDim, kvDim, ffnDim, vocabSize int) (*GpuRunState, error) {
	rs := &GpuRunState{Pool: blas.DefaultPool()}
	a := allocChecker{}
	rs.X = a.alloc(uint64(dim * 4))
	rs.XNorm = a.alloc(uint64(dim * 4))
	rs.Q = a.alloc(uint64(qDim * 4))
	rs.K = a.alloc(uint64(kvDim * 4))
	rs.V = a.alloc(uint64(kvDim * 4))
	rs.AttnOut = a.alloc(uint64(qDim * 4))
	rs.AttnProj = a.alloc(uint64(dim * 4))
	rs.FFNIn = a.alloc(uint64(dim * 4))
	rs.FFNNorm = a.alloc(uint64(dim * 4))
	rs.Gate = a.alloc(uint64(ffnDim * 4))
	rs.Up = a.alloc(uint64(ffnDim * 4))
	rs.Hidden = a.alloc(uint64(ffnDim * 4))
	rs.FFNOut = a.alloc(uint64(dim * 4))
	rs.Logits = a.alloc(uint64(vocabSize * 4))
	if a.err != nil {
		rs.FreeAll()
		return nil, fmt.Errorf("gpu: NewGpuRunState: %w", a.err)
	}
	return rs, nil
}

// AllocSSMScratch allocates GPU scratch buffers for SSM layers.
func (rs *GpuRunState) AllocSSMScratch(qkvDim, valueDim, numHeads int) error {
	a := allocChecker{}
	rs.SSMQKV = a.alloc(uint64(qkvDim * 4))
	rs.SSMZ = a.alloc(uint64(valueDim * 4))
	rs.SSMAlpha = a.alloc(uint64(numHeads * 4))
	rs.SSMBeta = a.alloc(uint64(numHeads * 4))
	rs.SSMY = a.alloc(uint64(valueDim * 4))
	rs.SSMPred = a.alloc(uint64(valueDim * 4))
	if a.err != nil {
		return fmt.Errorf("gpu: AllocSSMScratch: %w", a.err)
	}
	return nil
}

// AllocGatedQScratch allocates GPU scratch buffers for GatedQ attention.
func (rs *GpuRunState) AllocGatedQScratch(qDim int) error {
	a := allocChecker{}
	rs.QFull = a.alloc(uint64(2 * qDim * 4))
	rs.QGate = a.alloc(uint64(qDim * 4))
	if a.err != nil {
		return fmt.Errorf("gpu: AllocGatedQScratch: %w", a.err)
	}
	return nil
}

// GpuBatchState holds batch-sized GPU buffers for prefill.
type GpuBatchState struct {
	X        Buf
	XNorm    Buf
	Q        Buf
	K        Buf
	V        Buf
	AttnOut  Buf
	AttnProj Buf
	FFNIn    Buf
	FFNNorm  Buf
	Gate     Buf
	Up       Buf
	Hidden   Buf
	FFNOut   Buf
	Npos     int

	// GatedQ batch buffers
	QFull Buf // [npos * 2*qDim]
	QGate Buf // [npos * qDim]

	// SSM batch buffers (input projections batched, recurrence per-position)
	SSMQKV   Buf // [npos * qkvDim]
	SSMZ     Buf // [npos * valueDim]
	SSMAlpha Buf // [npos * numHeads]
	SSMBeta  Buf // [npos * numHeads]
	SSMY     Buf // [npos * valueDim]
}

// NewGpuBatchState allocates batch-sized GPU activation buffers.
func NewGpuBatchState(npos, dim, qDim, kvDim, ffnDim int) (*GpuBatchState, error) {
	bs := &GpuBatchState{Npos: npos}
	a := allocChecker{}
	bs.X = a.alloc(uint64(npos * dim * 4))
	bs.XNorm = a.alloc(uint64(npos * dim * 4))
	bs.Q = a.alloc(uint64(npos * qDim * 4))
	bs.K = a.alloc(uint64(npos * kvDim * 4))
	bs.V = a.alloc(uint64(npos * kvDim * 4))
	bs.AttnOut = a.alloc(uint64(npos * qDim * 4))
	bs.AttnProj = a.alloc(uint64(npos * dim * 4))
	bs.FFNIn = a.alloc(uint64(npos * dim * 4))
	bs.FFNNorm = a.alloc(uint64(npos * dim * 4))
	bs.Gate = a.alloc(uint64(npos * ffnDim * 4))
	bs.Up = a.alloc(uint64(npos * ffnDim * 4))
	bs.Hidden = a.alloc(uint64(npos * ffnDim * 4))
	bs.FFNOut = a.alloc(uint64(npos * dim * 4))
	if a.err != nil {
		bs.Free()
		return nil, fmt.Errorf("gpu: NewGpuBatchState(npos=%d): %w", npos, a.err)
	}
	return bs, nil
}

// AllocGatedQBatch allocates batch-sized GatedQ scratch buffers.
func (bs *GpuBatchState) AllocGatedQBatch(npos, qDim int) error {
	a := allocChecker{}
	bs.QFull = a.alloc(uint64(npos * 2 * qDim * 4))
	bs.QGate = a.alloc(uint64(npos * qDim * 4))
	if a.err != nil {
		return fmt.Errorf("gpu: AllocGatedQBatch: %w", a.err)
	}
	return nil
}

// AllocSSMBatch allocates batch-sized SSM scratch buffers.
func (bs *GpuBatchState) AllocSSMBatch(npos, qkvDim, valueDim, numHeads int) error {
	a := allocChecker{}
	bs.SSMQKV = a.alloc(uint64(npos * qkvDim * 4))
	bs.SSMZ = a.alloc(uint64(npos * valueDim * 4))
	bs.SSMAlpha = a.alloc(uint64(npos * numHeads * 4))
	bs.SSMBeta = a.alloc(uint64(npos * numHeads * 4))
	bs.SSMY = a.alloc(uint64(npos * valueDim * 4))
	if a.err != nil {
		return fmt.Errorf("gpu: AllocSSMBatch: %w", a.err)
	}
	return nil
}

// FreeBatchState releases all batch GPU buffers.
func (bs *GpuBatchState) Free() {
	if bs == nil {
		return
	}
	Free(bs.X)
	Free(bs.XNorm)
	Free(bs.Q)
	Free(bs.K)
	Free(bs.V)
	Free(bs.AttnOut)
	Free(bs.AttnProj)
	Free(bs.FFNIn)
	Free(bs.FFNNorm)
	Free(bs.Gate)
	Free(bs.Up)
	Free(bs.Hidden)
	Free(bs.FFNOut)
	Free(bs.QFull)
	Free(bs.QGate)
	Free(bs.SSMQKV)
	Free(bs.SSMZ)
	Free(bs.SSMAlpha)
	Free(bs.SSMBeta)
	Free(bs.SSMY)
}

// GpuKVCache holds GPU-resident KV cache for all layers.
type GpuKVCache struct {
	KeyBufs []Buf // [nLayers] each is [maxSeqLen * kvDim/2] packed uint32 (FP16)
	ValBufs []Buf
	KVDim   int
	MaxSeq  int
	Len     int
	// BytesPerElem is 4 when the F32 KV cache is enabled (DLGO_F32_KV=1),
	// otherwise 2 (packed FP16). State snapshot/restore needs the element
	// size to compute byte spans of the used region.
	BytesPerElem int
}

// NewGpuKVCache allocates GPU buffers for KV cache.
// gpuLayers controls how many layers get GPU-allocated KV buffers.
// needsKV is a per-layer mask: only layers where needsKV[l] is true get buffers.
func NewGpuKVCache(totalLayers, gpuLayers, maxSeqLen, kvDim int, needsKV []bool) (*GpuKVCache, error) {
	bytesPerElem := 2 // FP16: packed half2
	if os.Getenv("DLGO_F32_KV") == "1" {
		bytesPerElem = 4 // F32
		fmt.Println("[dlgo/gpu] F32 KV cache enabled (DLGO_F32_KV=1)")
	}
	c := &GpuKVCache{
		KeyBufs:      make([]Buf, totalLayers),
		ValBufs:      make([]Buf, totalLayers),
		KVDim:        kvDim,
		MaxSeq:       maxSeqLen,
		BytesPerElem: bytesPerElem,
	}
	size := uint64(maxSeqLen * kvDim * bytesPerElem)
	a := allocChecker{}
	for l := 0; l < gpuLayers && l < totalLayers; l++ {
		if needsKV != nil && !needsKV[l] {
			continue
		}
		c.KeyBufs[l] = a.alloc(size)
		c.ValBufs[l] = a.alloc(size)
		if a.err != nil {
			for j := 0; j <= l; j++ {
				freeBuf(c.KeyBufs[j])
				freeBuf(c.ValBufs[j])
			}
			return nil, fmt.Errorf("gpu: NewGpuKVCache(layer %d/%d, maxSeq=%d, kvDim=%d): %w", l, totalLayers, maxSeqLen, kvDim, a.err)
		}
	}
	return c, nil
}

func (c *GpuKVCache) Reset() { c.Len = 0 }

func freeTensor(gt *GpuTensor) {
	if gt != nil && gt.Buf != 0 {
		Free(gt.Buf)
	}
}

func freeBuf(b Buf) {
	if b != 0 {
		Free(b)
	}
}

// FreeModel releases all GPU buffers held by a GpuModel.
func (gm *GpuModel) FreeAll() {
	if gm == nil {
		return
	}
	freeTensor(gm.TokenEmbed)
	freeBuf(gm.OutputNorm)
	freeBuf(gm.OutputNormBias)
	freeTensor(gm.Output)
	freeBuf(gm.OutputBias)
	for i := range gm.Layers {
		gl := &gm.Layers[i]
		freeBuf(gl.AttnNorm)
		freeBuf(gl.AttnNormBias)
		freeTensor(gl.Wq)
		freeTensor(gl.Wk)
		freeTensor(gl.Wv)
		freeTensor(gl.Wo)
		freeBuf(gl.Bq)
		freeBuf(gl.Bk)
		freeBuf(gl.Bv)
		freeBuf(gl.Bo)
		freeBuf(gl.AttnQNorm)
		freeBuf(gl.AttnKNorm)
		freeBuf(gl.PostAttnNorm)
		freeBuf(gl.FFNNorm)
		freeTensor(gl.FFNGate)
		freeTensor(gl.FFNUp)
		freeTensor(gl.FFNDown)
		freeTensor(gl.WqA8)
		freeTensor(gl.WkA8)
		freeTensor(gl.WvA8)
		freeTensor(gl.WoA8)
		freeTensor(gl.GateA8)
		freeTensor(gl.UpA8)
		freeTensor(gl.DownA8)
		freeBuf(gl.FFNUpBias)
		freeBuf(gl.FFNDownBias)
		freeBuf(gl.PostFFNNorm)
		freeBuf(gl.AttnSinks)

		// MoE expert weights (packed, multi-GB for large MoE models)
		freeTensor(gl.FFNRouter)
		freeBuf(gl.FFNRouterBias)
		freeTensor(gl.FFNGateExps)
		freeTensor(gl.FFNUpExps)
		freeTensor(gl.FFNGateUpExps)
		freeTensor(gl.FFNDownExps)
		freeBuf(gl.FFNGateExpsBias)
		freeBuf(gl.FFNUpExpsBias)
		freeBuf(gl.FFNDownExpsBias)

		// MoE shared expert weights
		freeTensor(gl.FFNGateShared)
		freeTensor(gl.FFNUpShared)
		freeTensor(gl.FFNDownShared)
		freeBuf(gl.FFNRouterShared)

		// SSM (Gated Delta Net) layer buffers
		freeTensor(gl.SSMInProj)
		freeTensor(gl.SSMGate)
		freeTensor(gl.SSMAlpha)
		freeTensor(gl.SSMBeta)
		freeBuf(gl.SSMConv1dW)
		freeBuf(gl.SSMA)
		freeBuf(gl.SSMDtBias)
		freeBuf(gl.SSMNorm)
		freeTensor(gl.SSMOut)
		freeTensor(gl.SSMInProjA8)
		freeTensor(gl.SSMGateA8)
		freeTensor(gl.SSMAlphaA8)
		freeTensor(gl.SSMBetaA8)
		freeTensor(gl.SSMOutA8)
		freeBuf(gl.SSMState)
		freeBuf(gl.SSMConvBuf)
	}
}

// FreeAll releases all GPU buffers held by a GpuRunState.
func (rs *GpuRunState) FreeAll() {
	if rs == nil {
		return
	}
	freeBuf(rs.X)
	freeBuf(rs.XNorm)
	freeBuf(rs.Q)
	freeBuf(rs.K)
	freeBuf(rs.V)
	freeBuf(rs.AttnOut)
	freeBuf(rs.AttnProj)
	freeBuf(rs.FFNIn)
	freeBuf(rs.FFNNorm)
	freeBuf(rs.Gate)
	freeBuf(rs.Up)
	freeBuf(rs.Hidden)
	freeBuf(rs.FFNOut)
	freeBuf(rs.Logits)

	// SSM scratch buffers
	freeBuf(rs.SSMQKV)
	freeBuf(rs.SSMZ)
	freeBuf(rs.SSMAlpha)
	freeBuf(rs.SSMBeta)
	freeBuf(rs.SSMY)
	freeBuf(rs.SSMPred)

	// GatedQ scratch buffers
	freeBuf(rs.QFull)
	freeBuf(rs.QGate)

	// MoE scratch buffers
	freeBuf(rs.MoELogits)
	freeBuf(rs.MoETopKIdx)
	freeBuf(rs.MoETopKW)
	for _, b := range rs.MoEGates {
		freeBuf(b)
	}
	for _, b := range rs.MoEUps {
		freeBuf(b)
	}
	for _, b := range rs.MoEHiddens {
		freeBuf(b)
	}
	for _, b := range rs.MoEOuts {
		freeBuf(b)
	}
	freeBuf(rs.MoEShGate)
	freeBuf(rs.MoEShUp)
	freeBuf(rs.MoEShHidden)
	freeBuf(rs.MoEShOut)

	// dp4a MoE scratch buffers
	freeBuf(rs.MoEQ8_1Scratch)
	freeBuf(rs.MoEQ8_1DownPacked)
	for _, b := range rs.MoEQ8_1DownBufs {
		freeBuf(b)
	}

	// Fused MoE dp4a buffers
	freeBuf(rs.MoEGateScratch)
	freeBuf(rs.MoEUpScratch)
	freeBuf(rs.MoEHiddenScratch)
	freeBuf(rs.MoEOutScratch)
	freeBuf(rs.MoEWeightsBuf)
}

// FreeAll releases all GPU buffers held by a GpuKVCache.
func (c *GpuKVCache) FreeAll() {
	if c == nil {
		return
	}
	for _, b := range c.KeyBufs {
		freeBuf(b)
	}
	for _, b := range c.ValBufs {
		freeBuf(b)
	}
}
