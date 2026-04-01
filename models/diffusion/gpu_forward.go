//go:build cgo && vulkan

package diffusion

import (
	"fmt"
	"log"
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/ops"
)

// GpuDiTLayer holds GPU buffers for one JointTransformerBlock.
type GpuDiTLayer struct {
	AttnQKV  *gpu.GpuTensor
	AttnOut  *gpu.GpuTensor
	QNorm    gpu.Buf
	KNorm    gpu.Buf
	AttnNorm1 gpu.Buf
	AttnNorm2 gpu.Buf

	FFNGate  *gpu.GpuTensor
	FFNDown  *gpu.GpuTensor
	FFNUp    *gpu.GpuTensor
	FFNNorm1 gpu.Buf
	FFNNorm2 gpu.Buf

	AdaLNWeight *gpu.GpuTensor
	AdaLNBias   gpu.Buf
}

// GpuDiTModel holds the GPU representation of the DiT model.
type GpuDiTModel struct {
	Config  ZImageConfig
	Layers  []GpuDiTLayer // context_refiner + noise_refiner + main
	NCtxRef int           // number of context refiner layers
	NNoise  int           // number of noise refiner layers
	NMain   int           // number of main layers

	// Embeddings (kept on CPU for now since they run once per step)
	// The heavy per-layer MatVecs are on GPU.
}

// GpuDiTRunState holds GPU activation buffers for DiT inference.
type GpuDiTRunState struct {
	X       gpu.Buf // [maxSeq * hidden]
	XNorm   gpu.Buf // [maxSeq * hidden]
	QKV     gpu.Buf // [maxSeq * qkvDim]
	Q       gpu.Buf // [maxSeq * qDim]
	K       gpu.Buf // [maxSeq * kvDim]
	V       gpu.Buf // [maxSeq * kvDim]
	AttnOut gpu.Buf // [maxSeq * qDim]
	Proj    gpu.Buf // [maxSeq * hidden]
	Gate    gpu.Buf // [maxSeq * ffnDim]
	Up      gpu.Buf // [maxSeq * ffnDim]
	Hidden  gpu.Buf // [maxSeq * ffnDim]
	FFNOut  gpu.Buf // [maxSeq * hidden]
	Residual gpu.Buf // [maxSeq * hidden]

	// Small buffers
	Mod     gpu.Buf // [4*hidden]
	ScaleBuf gpu.Buf // [hidden] for (1+scale) computation
	GateBuf  gpu.Buf // [hidden] for tanh(gate) or gate values

	PE gpu.Buf // [peLen] precomputed positional embeddings

	MaxSeqLen int
}

// UploadDiTModel uploads DiT layer weights to GPU.
func UploadDiTModel(m *DiTModel) (*GpuDiTModel, error) {
	cfg := m.Config
	gm := &GpuDiTModel{
		Config:  cfg,
		NCtxRef: len(m.ContextRefiner),
		NNoise:  len(m.NoiseRefiner),
		NMain:   len(m.MainLayers),
	}

	totalLayers := gm.NCtxRef + gm.NNoise + gm.NMain
	gm.Layers = make([]GpuDiTLayer, totalLayers)

	// Estimate VRAM needed
	var totalBytes uint64
	allCPULayers := make([]*DiTLayer, 0, totalLayers)
	for i := range m.ContextRefiner {
		allCPULayers = append(allCPULayers, &m.ContextRefiner[i])
	}
	for i := range m.NoiseRefiner {
		allCPULayers = append(allCPULayers, &m.NoiseRefiner[i])
	}
	for i := range m.MainLayers {
		allCPULayers = append(allCPULayers, &m.MainLayers[i])
	}

	for _, l := range allCPULayers {
		totalBytes += uint64(len(l.AttnQKV.Data))
		totalBytes += uint64(len(l.AttnOut.Data))
		totalBytes += uint64(len(l.FFNGate.Data))
		totalBytes += uint64(len(l.FFNDown.Data))
		totalBytes += uint64(len(l.FFNUp.Data))
		if l.AdaLNWeight != nil {
			totalBytes += uint64(len(l.AdaLNWeight.Data))
		}
		// F32 norm weights + bias
		totalBytes += uint64(cfg.HeadDim*4) * 2   // QNorm, KNorm
		totalBytes += uint64(cfg.HiddenSize*4) * 4 // AttnNorm1,2, FFNNorm1,2
		if l.AdaLNBias != nil {
			totalBytes += uint64(len(l.AdaLNBias) * 4)
		}
	}

	freeVRAM := gpu.VRAMFreeBytes()
	log.Printf("[diffusion/gpu] Model weights: %.1f MB, Free VRAM: %.1f MB",
		float64(totalBytes)/(1024*1024), float64(freeVRAM)/(1024*1024))
	if totalBytes > freeVRAM*9/10 {
		return nil, fmt.Errorf("not enough VRAM: need %.1f MB, have %.1f MB free",
			float64(totalBytes)/(1024*1024), float64(freeVRAM)/(1024*1024))
	}

	for i, l := range allCPULayers {
		var err error
		gl := &gm.Layers[i]

		gl.AttnQKV, err = gpu.UploadTensor(l.AttnQKV)
		if err != nil {
			return nil, fmt.Errorf("layer %d AttnQKV: %w", i, err)
		}
		gl.AttnOut, err = gpu.UploadTensor(l.AttnOut)
		if err != nil {
			return nil, fmt.Errorf("layer %d AttnOut: %w", i, err)
		}
		gl.FFNGate, err = gpu.UploadTensor(l.FFNGate)
		if err != nil {
			return nil, fmt.Errorf("layer %d FFNGate: %w", i, err)
		}
		gl.FFNDown, err = gpu.UploadTensor(l.FFNDown)
		if err != nil {
			return nil, fmt.Errorf("layer %d FFNDown: %w", i, err)
		}
		gl.FFNUp, err = gpu.UploadTensor(l.FFNUp)
		if err != nil {
			return nil, fmt.Errorf("layer %d FFNUp: %w", i, err)
		}

		gl.QNorm, err = gpu.UploadF32Slice(l.QNorm)
		if err != nil {
			return nil, fmt.Errorf("layer %d QNorm: %w", i, err)
		}
		gl.KNorm, err = gpu.UploadF32Slice(l.KNorm)
		if err != nil {
			return nil, fmt.Errorf("layer %d KNorm: %w", i, err)
		}
		gl.AttnNorm1, err = gpu.UploadF32Slice(l.AttnNorm1)
		if err != nil {
			return nil, fmt.Errorf("layer %d AttnNorm1: %w", i, err)
		}
		gl.AttnNorm2, err = gpu.UploadF32Slice(l.AttnNorm2)
		if err != nil {
			return nil, fmt.Errorf("layer %d AttnNorm2: %w", i, err)
		}
		gl.FFNNorm1, err = gpu.UploadF32Slice(l.FFNNorm1)
		if err != nil {
			return nil, fmt.Errorf("layer %d FFNNorm1: %w", i, err)
		}
		gl.FFNNorm2, err = gpu.UploadF32Slice(l.FFNNorm2)
		if err != nil {
			return nil, fmt.Errorf("layer %d FFNNorm2: %w", i, err)
		}

		if l.AdaLNWeight != nil {
			gl.AdaLNWeight, err = gpu.UploadTensor(l.AdaLNWeight)
			if err != nil {
				return nil, fmt.Errorf("layer %d AdaLN: %w", i, err)
			}
		}
		if l.AdaLNBias != nil {
			gl.AdaLNBias, err = gpu.UploadF32Slice(l.AdaLNBias)
			if err != nil {
				return nil, fmt.Errorf("layer %d AdaLNBias: %w", i, err)
			}
		}
	}

	log.Printf("[diffusion/gpu] Uploaded %d layers to GPU (%.1f MB)",
		totalLayers, float64(gpu.AllocatedBytes())/(1024*1024))
	return gm, nil
}

// NewGpuDiTRunState allocates GPU activation buffers.
func NewGpuDiTRunState(cfg ZImageConfig, maxSeqLen int) (*GpuDiTRunState, error) {
	hidden := cfg.HiddenSize
	ffnDim := cfg.FFNHiddenDim()
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	qkvDim := qDim + 2*kvDim

	rs := &GpuDiTRunState{MaxSeqLen: maxSeqLen}
	var err error

	alloc := func(name string, nFloats int) (gpu.Buf, error) {
		b, e := gpu.AllocE(uint64(nFloats) * 4)
		if e != nil {
			return 0, fmt.Errorf("alloc %s: %w", name, e)
		}
		return b, nil
	}

	if rs.X, err = alloc("X", maxSeqLen*hidden); err != nil {
		return nil, err
	}
	if rs.XNorm, err = alloc("XNorm", maxSeqLen*hidden); err != nil {
		return nil, err
	}
	if rs.QKV, err = alloc("QKV", maxSeqLen*qkvDim); err != nil {
		return nil, err
	}
	if rs.Q, err = alloc("Q", maxSeqLen*qDim); err != nil {
		return nil, err
	}
	if rs.K, err = alloc("K", maxSeqLen*kvDim); err != nil {
		return nil, err
	}
	if rs.V, err = alloc("V", maxSeqLen*kvDim); err != nil {
		return nil, err
	}
	if rs.AttnOut, err = alloc("AttnOut", maxSeqLen*qDim); err != nil {
		return nil, err
	}
	if rs.Proj, err = alloc("Proj", maxSeqLen*hidden); err != nil {
		return nil, err
	}
	if rs.Gate, err = alloc("Gate", maxSeqLen*ffnDim); err != nil {
		return nil, err
	}
	if rs.Up, err = alloc("Up", maxSeqLen*ffnDim); err != nil {
		return nil, err
	}
	if rs.Hidden, err = alloc("Hidden", maxSeqLen*ffnDim); err != nil {
		return nil, err
	}
	if rs.FFNOut, err = alloc("FFNOut", maxSeqLen*hidden); err != nil {
		return nil, err
	}
	if rs.Residual, err = alloc("Residual", maxSeqLen*hidden); err != nil {
		return nil, err
	}
	if rs.Mod, err = alloc("Mod", 4*hidden); err != nil {
		return nil, err
	}
	if rs.ScaleBuf, err = alloc("ScaleBuf", hidden); err != nil {
		return nil, err
	}
	if rs.GateBuf, err = alloc("GateBuf", hidden); err != nil {
		return nil, err
	}

	log.Printf("[diffusion/gpu] RunState allocated: %.1f MB for maxSeq=%d",
		float64(gpu.AllocatedBytes())/(1024*1024), maxSeqLen)
	return rs, nil
}

// GpuDiTForward runs the DiT forward pass with GPU-accelerated layer computations.
// Embeddings and final layer run on CPU. The 34 transformer layers run on GPU.
func GpuDiTForward(m *DiTModel, gm *GpuDiTModel, rs *DiTRunState, grs *GpuDiTRunState,
	x []float32, timestep float32, context []float32, contextLen, H, W int) []float32 {

	cfg := m.Config
	hidden := cfg.HiddenSize
	patchSize := cfg.PatchSize
	hPatches := H / patchSize
	wPatches := W / patchSize
	nImgTokens := hPatches * wPatches
	patchDim := patchSize * patchSize * cfg.InChannels

	// === CPU: Pre-processing (runs once per step, not performance-critical) ===

	// 1. Patchify
	imgPatches := patchify(x, cfg.InChannels, H, W, patchSize)

	// 2. Timestep embedding
	sinEmb := timestepEmbedding(timestep, cfg.AdaLNEmbedDim)
	blas.QMatVecMulParallel(rs.TEmbMid, m.TEmbedMLP0Weight, sinEmb, rs.pool)
	addBias(rs.TEmbMid, m.TEmbedMLP0Bias)
	ops.SiLU(rs.TEmbMid)
	blas.QMatVecMulParallel(rs.TEmb, m.TEmbedMLP2Weight, rs.TEmbMid, rs.pool)
	addBias(rs.TEmb, m.TEmbedMLP2Bias)

	// 3. Caption embedding
	txtNormed := make([]float32, contextLen*cfg.CapFeatDim)
	for i := 0; i < contextLen; i++ {
		ops.RMSNorm(txtNormed[i*cfg.CapFeatDim:(i+1)*cfg.CapFeatDim],
			context[i*cfg.CapFeatDim:(i+1)*cfg.CapFeatDim],
			m.CapEmbedNormWeight, cfg.NormEps)
	}
	txt := make([]float32, contextLen*hidden)
	blas.QBatchGEMMParallel(txt, m.CapEmbedLinWeight, txtNormed, contextLen, rs.pool)
	addBiasBatch(txt, m.CapEmbedLinBias, contextLen, hidden)

	// 4. Image embedding
	img := make([]float32, nImgTokens*hidden)
	blas.QBatchGEMMParallel(img, m.XEmbedWeight, imgPatches, nImgTokens, rs.pool)
	addBiasBatch(img, m.XEmbedBias, nImgTokens, hidden)

	// 5. Pad text and image
	txtPadLen := boundMod(contextLen, cfg.SeqMultiOf)
	nTxtPadded := contextLen + txtPadLen
	if txtPadLen > 0 {
		txtPadded := make([]float32, nTxtPadded*hidden)
		copy(txtPadded, txt)
		for i := contextLen; i < nTxtPadded; i++ {
			copy(txtPadded[i*hidden:(i+1)*hidden], m.CapPadToken)
		}
		txt = txtPadded
	}
	imgPadLen := boundMod(nImgTokens, cfg.SeqMultiOf)
	nImgPadded := nImgTokens + imgPadLen
	if imgPadLen > 0 {
		imgPadded := make([]float32, nImgPadded*hidden)
		copy(imgPadded, img)
		for i := nImgTokens; i < nImgPadded; i++ {
			copy(imgPadded[i*hidden:(i+1)*hidden], m.XPadToken)
		}
		img = imgPadded
	}

	// 6. Positional embeddings
	var pe []float32
	if rs.cachedPEH == H && rs.cachedPEW == W && rs.cachedPECtxLen == contextLen && rs.cachedPE != nil {
		pe = rs.cachedPE
	} else {
		pe = GenZImagePE(H, W, cfg.PatchSize, 1, contextLen, cfg.SeqMultiOf, cfg.Theta, cfg.AxesDim)
		rs.cachedPE = pe
		rs.cachedPEH = H
		rs.cachedPEW = W
		rs.cachedPECtxLen = contextLen
	}

	// Upload PE to GPU
	if err := gpu.UploadF32(grs.PE, pe); err != nil {
		// Allocate PE buffer on first use or if size changed
		if grs.PE != 0 {
			gpu.Free(grs.PE)
		}
		var allocErr error
		grs.PE, allocErr = gpu.AllocE(uint64(len(pe)) * 4)
		if allocErr != nil {
			log.Printf("[diffusion/gpu] PE alloc failed: %v", allocErr)
			return nil
		}
		gpu.UploadF32(grs.PE, pe)
	}

	peStride := cfg.HeadDim * 2

	// === GPU: Layer processing ===

	// 7. Context refiner: text tokens only
	if err := gpu.UploadF32(grs.X, txt[:nTxtPadded*hidden]); err != nil {
		log.Printf("[diffusion/gpu] upload txt: %v", err)
		return nil
	}
	for i := 0; i < gm.NCtxRef; i++ {
		gpuForwardBlock(gm, grs, &gm.Layers[i], nTxtPadded, hidden, pe, 0, nil, peStride)
	}

	// Download text back
	gpu.DownloadF32(grs.X, txt[:nTxtPadded*hidden])

	// 8. Noise refiner: image tokens only
	if err := gpu.UploadF32(grs.X, img[:nImgPadded*hidden]); err != nil {
		log.Printf("[diffusion/gpu] upload img: %v", err)
		return nil
	}
	for i := 0; i < gm.NNoise; i++ {
		gpuForwardBlock(gm, grs, &gm.Layers[gm.NCtxRef+i], nImgPadded, hidden, nil, nTxtPadded, rs.TEmb, peStride)
	}

	// Download image back
	gpu.DownloadF32(grs.X, img[:nImgPadded*hidden])

	// 9. Concatenate and upload for main layers
	totalSeq := nTxtPadded + nImgPadded
	combined := rs.X[:totalSeq*hidden]
	copy(combined[:nTxtPadded*hidden], txt)
	copy(combined[nTxtPadded*hidden:], img)

	if err := gpu.UploadF32(grs.X, combined); err != nil {
		log.Printf("[diffusion/gpu] upload combined: %v", err)
		return nil
	}

	// 10. Main layers
	for i := 0; i < gm.NMain; i++ {
		gpuForwardBlock(gm, grs, &gm.Layers[gm.NCtxRef+gm.NNoise+i], totalSeq, hidden, nil, 0, rs.TEmb, peStride)
	}

	// Download combined result
	gpu.DownloadF32(grs.X, combined)

	// === CPU: Final layer + post-processing ===

	// 11. Final layer
	out := forwardFinalLayer(m, rs, combined, totalSeq, rs.TEmb)

	// 12–14. Extract and unpatchify
	imgStart := nTxtPadded
	imgOut := out[imgStart*patchDim : (imgStart+nImgTokens)*patchDim]
	result := unpatchify(imgOut, cfg.OutChannels, H, W, patchSize)
	for i := range result {
		result[i] = -result[i]
	}

	return result
}

// gpuForwardBlock runs one transformer block on GPU.
// adaLNInput is CPU float32 slice (small, computed once per step) or nil.
func gpuForwardBlock(gm *GpuDiTModel, grs *GpuDiTRunState, gl *GpuDiTLayer,
	seqLen, hidden int, pe []float32, peOffset int, adaLNInput []float32, peStride int) {

	cfg := gm.Config
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	qDim := numHeads * headDim
	kvDim := numKVHeads * headDim
	qkvDim := qDim + 2*kvDim
	ffnDim := cfg.FFNHiddenDim()
	eps := cfg.NormEps

	hasAdaLN := gl.AdaLNWeight != nil && adaLNInput != nil

	// adaLN: upload small input to GPU, run matvec, download 4 mod vectors.
	var scaleMSA, gateMSA, scaleMLPMod, gateMLPMod []float32
	if hasAdaLN {
		modSize := 4 * hidden
		// Upload adaLNInput to ScaleBuf (reuse, it's large enough for 256 floats)
		gpu.UploadF32(grs.ScaleBuf, adaLNInput)

		gpu.BeginBatch()
		gpu.BatchMatVec(grs.Mod, gl.AdaLNWeight.Buf, grs.ScaleBuf,
			gl.AdaLNWeight.Rows, gl.AdaLNWeight.Cols, 1, gl.AdaLNWeight.Type)
		gpu.EndBatch()

		// Download mod and add bias on CPU
		mod := make([]float32, modSize)
		gpu.DownloadF32(grs.Mod, mod)
		if gl.AdaLNBias != 0 {
			bias := make([]float32, modSize)
			gpu.DownloadF32(gl.AdaLNBias, bias)
			for i := range mod {
				mod[i] += bias[i]
			}
		}

		scaleMSA = mod[0*hidden : 1*hidden]
		gateMSA = mod[1*hidden : 2*hidden]
		scaleMLPMod = mod[2*hidden : 3*hidden]
		gateMLPMod = mod[3*hidden : 4*hidden]
	}

	// --- Attention path ---

	gpu.BeginBatch()
	gpu.BatchRMSNorm(grs.XNorm, grs.X, gl.AttnNorm1, hidden, seqLen, eps)
	gpu.EndBatch()

	// Apply (1+scaleMSA) modulation if adaLN
	if hasAdaLN {
		// Upload (1+scale) to GPU
		oneScale := make([]float32, hidden)
		for j := range oneScale {
			oneScale[j] = 1.0 + scaleMSA[j]
		}
		gpu.UploadF32(grs.ScaleBuf, oneScale)

		gpu.BeginBatch()
		gpu.BroadcastMul(grs.XNorm, grs.ScaleBuf, seqLen*hidden, hidden)
		gpu.EndBatch()
	}

	// QKV projection
	gpu.BeginBatch()
	gpu.BatchMatVec(grs.QKV, gl.AttnQKV.Buf, grs.XNorm, gl.AttnQKV.Rows, gl.AttnQKV.Cols, seqLen, gl.AttnQKV.Type)
	gpu.EndBatch()

	// Split QKV on CPU (simple data reshuffling, not compute-bound)
	qkvData := make([]float32, seqLen*qkvDim)
	gpu.DownloadF32(grs.QKV, qkvData)
	qData := make([]float32, seqLen*qDim)
	kData := make([]float32, seqLen*kvDim)
	vData := make([]float32, seqLen*kvDim)
	for i := 0; i < seqLen; i++ {
		copy(qData[i*qDim:(i+1)*qDim], qkvData[i*qkvDim:i*qkvDim+qDim])
		copy(kData[i*kvDim:(i+1)*kvDim], qkvData[i*qkvDim+qDim:i*qkvDim+qDim+kvDim])
		copy(vData[i*kvDim:(i+1)*kvDim], qkvData[i*qkvDim+qDim+kvDim:i*qkvDim+qkvDim])
	}
	gpu.UploadF32(grs.Q, qData)
	gpu.UploadF32(grs.K, kData)
	gpu.UploadF32(grs.V, vData)

	// QK norm (per-head RMSNorm)
	if cfg.QKNorm {
		gpu.BeginBatch()
		gpu.RMSNormHeads(grs.Q, gl.QNorm, numHeads*seqLen, headDim, eps)
		gpu.Barrier()
		gpu.RMSNormHeads(grs.K, gl.KNorm, numKVHeads*seqLen, headDim, eps)
		gpu.EndBatch()
	}

	// 3D RoPE
	gpu.BeginBatch()
	gpu.RoPE3D(grs.Q, grs.PE, seqLen, numHeads, headDim, peOffset, peStride)
	gpu.Barrier()
	gpu.RoPE3D(grs.K, grs.PE, seqLen, numKVHeads, headDim, peOffset, peStride)
	gpu.EndBatch()

	// Full bidirectional attention
	attnScale := float32(1.0 / math.Sqrt(float64(headDim)))
	gpu.BeginBatch()
	gpu.AttentionFullF32(grs.AttnOut, grs.Q, grs.K, grs.V,
		numHeads, numKVHeads, headDim, kvDim, seqLen, attnScale)
	gpu.EndBatch()

	// Output projection
	gpu.BeginBatch()
	gpu.BatchMatVec(grs.Proj, gl.AttnOut.Buf, grs.AttnOut, gl.AttnOut.Rows, gl.AttnOut.Cols, seqLen, gl.AttnOut.Type)
	gpu.EndBatch()

	// Post-attention norm2: standard per-position RMSNorm
	gpu.BeginBatch()
	gpu.BatchRMSNorm(grs.Proj, grs.Proj, gl.AttnNorm2, hidden, seqLen, eps)
	gpu.EndBatch()

	// Gate and residual: X = residual + proj * tanh(gateMSA)
	if hasAdaLN {
		gpu.UploadF32(grs.GateBuf, gateMSA)
		gpu.BeginBatch()
		// First copy X to Residual
		gpu.CopyRegion(grs.Residual, 0, grs.X, 0, uint64(seqLen*hidden*4))
		gpu.Barrier()
		gpu.TanhGateResidual(grs.X, grs.Residual, grs.Proj, grs.GateBuf, seqLen*hidden, hidden)
		gpu.EndBatch()
	} else {
		gpu.BeginBatch()
		gpu.Add(grs.X, grs.X, grs.Proj, seqLen*hidden)
		gpu.EndBatch()
	}

	// --- FFN path ---

	gpu.BeginBatch()
	gpu.BatchRMSNorm(grs.XNorm, grs.X, gl.FFNNorm1, hidden, seqLen, eps)
	gpu.EndBatch()

	if hasAdaLN {
		oneScale := make([]float32, hidden)
		for j := range oneScale {
			oneScale[j] = 1.0 + scaleMLPMod[j]
		}
		gpu.UploadF32(grs.ScaleBuf, oneScale)
		gpu.BeginBatch()
		gpu.BroadcastMul(grs.XNorm, grs.ScaleBuf, seqLen*hidden, hidden)
		gpu.EndBatch()
	}

	// SwiGLU FFN: gate = SiLU(W1 @ x) * (W3 @ x), out = W2 @ gate
	gpu.BeginBatch()
	gpu.BatchMatVec(grs.Gate, gl.FFNGate.Buf, grs.XNorm, gl.FFNGate.Rows, gl.FFNGate.Cols, seqLen, gl.FFNGate.Type)
	gpu.Barrier()
	gpu.BatchMatVec(grs.Up, gl.FFNUp.Buf, grs.XNorm, gl.FFNUp.Rows, gl.FFNUp.Cols, seqLen, gl.FFNUp.Type)
	gpu.EndBatch()

	// SwiGLU activation on GPU
	gpu.BeginBatch()
	gpu.SwiGLU(grs.Hidden, grs.Gate, grs.Up, seqLen*ffnDim)
	gpu.EndBatch()

	// Down projection
	gpu.BeginBatch()
	gpu.BatchMatVec(grs.FFNOut, gl.FFNDown.Buf, grs.Hidden, gl.FFNDown.Rows, gl.FFNDown.Cols, seqLen, gl.FFNDown.Type)
	gpu.EndBatch()

	// Post-FFN norm2
	gpu.BeginBatch()
	gpu.BatchRMSNorm(grs.FFNOut, grs.FFNOut, gl.FFNNorm2, hidden, seqLen, eps)
	gpu.EndBatch()

	// Gate and residual for FFN
	if hasAdaLN {
		gpu.UploadF32(grs.GateBuf, gateMLPMod)
		gpu.BeginBatch()
		gpu.CopyRegion(grs.Residual, 0, grs.X, 0, uint64(seqLen*hidden*4))
		gpu.Barrier()
		gpu.TanhGateResidual(grs.X, grs.Residual, grs.FFNOut, grs.GateBuf, seqLen*hidden, hidden)
		gpu.EndBatch()
	} else {
		gpu.BeginBatch()
		gpu.Add(grs.X, grs.X, grs.FFNOut, seqLen*hidden)
		gpu.EndBatch()
	}
}
