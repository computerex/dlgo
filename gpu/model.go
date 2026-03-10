//go:build cgo && vulkan

package gpu

import (
	"fmt"

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
	Bq, Bk, Bv  Buf
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
	FFNGateShared *GpuTensor // [sharedFFNDim × dim]
	FFNUpShared   *GpuTensor // [sharedFFNDim × dim]
	FFNDownShared *GpuTensor // [dim × sharedFFNDim]
	FFNRouterShared Buf      // [dim] shared expert gate (float32)
	IsMoE         bool
	MoEOnGPU      bool       // true if expert weights are on GPU

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
}

// GpuModel holds all model weights on the GPU.
type GpuModel struct {
	TokenEmbed   *GpuTensor
	OutputNorm   Buf
	OutputNormBias Buf
	Output       *GpuTensor
	OutputBias   Buf
	Layers       []GpuLayer
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

	// GatedQ attention scratch buffers
	QFull Buf // [2*qDim] fused Q+gate output
	QGate Buf // [qDim] attention gate values

	// MoE scratch buffers (allocated on first use)
	MoELogits   Buf // [expertCount] router logits
	MoEGate     Buf // [expertFFNDim] expert gate projection
	MoEUp       Buf // [expertFFNDim] expert up projection
	MoEHidden   Buf // [expertFFNDim] SwiGLU hidden
	MoEExpertOut Buf // [dim] per-expert output
	MoEShGate   Buf // [sharedFFNDim] shared expert gate
	MoEShUp     Buf // [sharedFFNDim] shared expert up
	MoEShHidden Buf // [sharedFFNDim] shared expert hidden
	MoEShOut    Buf // [dim] shared expert output

	// CPU scratch buffers used for correctness fallbacks when a quant type
	// has no GPU kernel yet (for example Q3_K on Vulkan).
	ScratchIn  []float32
	ScratchOut []float32
	ScratchAux []float32
	Pool       *blas.Pool
}

// NewGpuRunState allocates all GPU activation buffers.
func NewGpuRunState(dim, qDim, kvDim, ffnDim, vocabSize int) *GpuRunState {
	return &GpuRunState{
		X:        Alloc(uint64(dim * 4)),
		XNorm:    Alloc(uint64(dim * 4)),
		Q:        Alloc(uint64(qDim * 4)),
		K:        Alloc(uint64(kvDim * 4)),
		V:        Alloc(uint64(kvDim * 4)),
		AttnOut:  Alloc(uint64(qDim * 4)),
		AttnProj: Alloc(uint64(dim * 4)),
		FFNIn:    Alloc(uint64(dim * 4)),
		FFNNorm:  Alloc(uint64(dim * 4)),
		Gate:     Alloc(uint64(ffnDim * 4)),
		Up:       Alloc(uint64(ffnDim * 4)),
		Hidden:   Alloc(uint64(ffnDim * 4)),
		FFNOut:   Alloc(uint64(dim * 4)),
		Logits:   Alloc(uint64(vocabSize * 4)),
		Pool:     blas.DefaultPool(),
	}
}

// AllocSSMScratch allocates GPU scratch buffers for SSM layers.
func (rs *GpuRunState) AllocSSMScratch(qkvDim, valueDim, numHeads int) {
	rs.SSMQKV = Alloc(uint64(qkvDim * 4))
	rs.SSMZ = Alloc(uint64(valueDim * 4))
	rs.SSMAlpha = Alloc(uint64(numHeads * 4))
	rs.SSMBeta = Alloc(uint64(numHeads * 4))
	rs.SSMY = Alloc(uint64(valueDim * 4))
}

// AllocGatedQScratch allocates GPU scratch buffers for GatedQ attention.
func (rs *GpuRunState) AllocGatedQScratch(qDim int) {
	rs.QFull = Alloc(uint64(2 * qDim * 4))
	rs.QGate = Alloc(uint64(qDim * 4))
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
func NewGpuBatchState(npos, dim, qDim, kvDim, ffnDim int) *GpuBatchState {
	return &GpuBatchState{
		X:        Alloc(uint64(npos * dim * 4)),
		XNorm:    Alloc(uint64(npos * dim * 4)),
		Q:        Alloc(uint64(npos * qDim * 4)),
		K:        Alloc(uint64(npos * kvDim * 4)),
		V:        Alloc(uint64(npos * kvDim * 4)),
		AttnOut:  Alloc(uint64(npos * qDim * 4)),
		AttnProj: Alloc(uint64(npos * dim * 4)),
		FFNIn:    Alloc(uint64(npos * dim * 4)),
		FFNNorm:  Alloc(uint64(npos * dim * 4)),
		Gate:     Alloc(uint64(npos * ffnDim * 4)),
		Up:       Alloc(uint64(npos * ffnDim * 4)),
		Hidden:   Alloc(uint64(npos * ffnDim * 4)),
		FFNOut:   Alloc(uint64(npos * dim * 4)),
		Npos:     npos,
	}
}

// AllocGatedQBatch allocates batch-sized GatedQ scratch buffers.
func (bs *GpuBatchState) AllocGatedQBatch(npos, qDim int) {
	bs.QFull = Alloc(uint64(npos * 2 * qDim * 4))
	bs.QGate = Alloc(uint64(npos * qDim * 4))
}

// AllocSSMBatch allocates batch-sized SSM scratch buffers.
func (bs *GpuBatchState) AllocSSMBatch(npos, qkvDim, valueDim, numHeads int) {
	bs.SSMQKV = Alloc(uint64(npos * qkvDim * 4))
	bs.SSMZ = Alloc(uint64(npos * valueDim * 4))
	bs.SSMAlpha = Alloc(uint64(npos * numHeads * 4))
	bs.SSMBeta = Alloc(uint64(npos * numHeads * 4))
	bs.SSMY = Alloc(uint64(npos * valueDim * 4))
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
	KeyBufs []Buf // [nLayers] each is [maxSeqLen * kvDim] floats
	ValBufs []Buf
	KVDim   int
	MaxSeq  int
	Len     int
}

// NewGpuKVCache allocates GPU buffers for KV cache.
// gpuLayers controls how many layers get GPU-allocated KV buffers.
// Layers beyond gpuLayers get zero-valued (nil) buffers.
func NewGpuKVCache(totalLayers, gpuLayers, maxSeqLen, kvDim int) *GpuKVCache {
	c := &GpuKVCache{
		KeyBufs: make([]Buf, totalLayers),
		ValBufs: make([]Buf, totalLayers),
		KVDim:   kvDim,
		MaxSeq:  maxSeqLen,
	}
	size := uint64(maxSeqLen * kvDim * 4)
	for l := 0; l < gpuLayers && l < totalLayers; l++ {
		c.KeyBufs[l] = Alloc(size)
		c.ValBufs[l] = Alloc(size)
	}
	return c
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
		freeBuf(gl.FFNUpBias)
		freeBuf(gl.FFNDownBias)
		freeBuf(gl.PostFFNNorm)
		freeBuf(gl.AttnSinks)
		freeTensor(gl.FFNGateShared)
		freeTensor(gl.FFNUpShared)
		freeTensor(gl.FFNDownShared)

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

	// GatedQ scratch buffers
	freeBuf(rs.QFull)
	freeBuf(rs.QGate)
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
