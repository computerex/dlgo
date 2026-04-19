//go:build cgo && vulkan

package diffusion

import (
	"fmt"
	"log"
	"math"
	"sort"
	"time"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/ops"
)

// GpuDebugProfile accumulates per-operation timing across layers.
type GpuDebugProfile struct {
	Enabled   bool
	StepNum   int
	Ops       map[string]time.Duration // operation name → total time
	OpCounts  map[string]int           // operation name → call count
	LayerOps  []layerTiming            // per-layer total
	stepStart time.Time
}

type layerTiming struct {
	Name     string
	SeqLen   int
	Total    time.Duration
	Details  []opTiming
}

type opTiming struct {
	Name string
	Dur  time.Duration
}

func NewGpuDebugProfile() *GpuDebugProfile {
	return &GpuDebugProfile{
		Ops:      make(map[string]time.Duration),
		OpCounts: make(map[string]int),
	}
}

func (p *GpuDebugProfile) StartStep(step int) {
	p.StepNum = step
	p.stepStart = time.Now()
	// Reset per-step accumulators
	p.Ops = make(map[string]time.Duration)
	p.OpCounts = make(map[string]int)
	p.LayerOps = p.LayerOps[:0]
}

func (p *GpuDebugProfile) AddOp(name string, d time.Duration) {
	p.Ops[name] += d
	p.OpCounts[name]++
}

// flushAndTime flushes the current batch, measures wall time, and starts a new batch.
func (p *GpuDebugProfile) flushAndTime(opName string, start time.Time) time.Duration {
	gpu.EndBatch()
	d := time.Since(start)
	p.AddOp(opName, d)
	gpu.BeginBatch()
	return d
}

// PrintStepSummary prints a breakdown of where time was spent in this step.
func (p *GpuDebugProfile) PrintStepSummary() {
	total := time.Since(p.stepStart)
	log.Printf("[debug] ===== Step %d Summary (total: %v) =====", p.StepNum, total)

	// Sort ops by total time descending
	type kv struct {
		Name  string
		Total time.Duration
		Count int
	}
	var sorted []kv
	for k, v := range p.Ops {
		sorted = append(sorted, kv{k, v, p.OpCounts[k]})
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Total > sorted[j].Total })

	var measured time.Duration
	for _, s := range sorted {
		pct := float64(s.Total) / float64(total) * 100
		avg := s.Total / time.Duration(s.Count)
		log.Printf("[debug]   %-25s %10v (%5.1f%%)  x%-4d  avg=%v",
			s.Name, s.Total, pct, s.Count, avg)
		measured += s.Total
	}
	overhead := total - measured
	log.Printf("[debug]   %-25s %10v (%5.1f%%)", "overhead/host", overhead,
		float64(overhead)/float64(total)*100)

	// Print all layers with detail
	if len(p.LayerOps) > 0 {
		log.Printf("[debug] --- Per-layer detail (all %d layers) ---", len(p.LayerOps))
		for i, lt := range p.LayerOps {
			// Print first 3 layers in detail, then just summary
			if i < 3 {
				log.Printf("[debug]   %s (seq=%d, total=%v):", lt.Name, lt.SeqLen, lt.Total)
				for _, op := range lt.Details {
					pct := float64(op.Dur) / float64(lt.Total) * 100
					log.Printf("[debug]     %-23s %10v (%5.1f%%)", op.Name, op.Dur, pct)
				}
			} else if i == 3 {
				log.Printf("[debug]   ... (remaining layers similar, showing totals) ...")
			}
			if i >= 3 {
				log.Printf("[debug]   %-20s total=%v", lt.Name, lt.Total)
			}
		}
	}

	// Compute and print key metrics
	attnTotal := p.Ops["attention"]
	attnCount := p.OpCounts["attention"]
	qkvTotal := p.Ops["attn_qkv_proj"]
	outProjTotal := p.Ops["attn_out_proj"]
	ffnGateUp := p.Ops["ffn_gate_up"]
	ffnDown := p.Ops["ffn_down_proj"]
	log.Printf("[debug] --- Key Metrics ---")
	log.Printf("[debug]   Total matmul time: %v (qkv=%v + out=%v + ffn_gu=%v + ffn_d=%v)",
		qkvTotal+outProjTotal+ffnGateUp+ffnDown, qkvTotal, outProjTotal, ffnGateUp, ffnDown)
	if attnCount > 0 {
		log.Printf("[debug]   Average attention: %v x%d = %v total",
			attnTotal/time.Duration(attnCount), attnCount, attnTotal)
	}
	log.Printf("[debug]   Submits per step: ~%d (debug mode adds %d extra submits)",
		p.OpCounts["attention"]*12, p.OpCounts["attention"]*11) // rough estimate
	log.Printf("[debug] =================================================")
}

var gpuDebugProfile *GpuDebugProfile

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
	OnesBuf  gpu.Buf // [hidden] pre-filled with 1.0s

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
// Uses a pool allocator (single vkAllocateMemory) to avoid hitting WDDM
// per-process allocation limits on Windows.
func NewGpuDiTRunState(cfg ZImageConfig, maxSeqLen int) (*GpuDiTRunState, error) {
	hidden := cfg.HiddenSize
	ffnDim := cfg.FFNHiddenDim()
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	qkvDim := qDim + 2*kvDim

	rs := &GpuDiTRunState{MaxSeqLen: maxSeqLen}
	var err error

	// Compute total pool size (each buffer aligned to 256 bytes)
	align := func(n int) uint64 { return (uint64(n)*4 + 255) & ^uint64(255) }
	poolSize := uint64(0)
	poolSize += align(maxSeqLen * hidden)  // X
	poolSize += align(maxSeqLen * hidden)  // XNorm
	poolSize += align(maxSeqLen * qkvDim)  // QKV
	poolSize += align(maxSeqLen * qDim)    // Q
	poolSize += align(maxSeqLen * kvDim)   // K
	poolSize += align(maxSeqLen * kvDim)   // V
	poolSize += align(maxSeqLen * qDim)    // AttnOut
	poolSize += align(maxSeqLen * hidden)  // Proj
	poolSize += align(maxSeqLen * ffnDim)  // Gate
	poolSize += align(maxSeqLen * ffnDim)  // Up
	poolSize += align(maxSeqLen * ffnDim)  // Hidden
	poolSize += align(maxSeqLen * hidden)  // FFNOut
	poolSize += align(maxSeqLen * hidden)  // Residual
	poolSize += align(4 * hidden)          // Mod
	poolSize += align(hidden)              // ScaleBuf
	poolSize += align(hidden)              // GateBuf
	poolSize += align(hidden)              // OnesBuf
	poolSize += 1024 * 1024 // 1MB headroom for alignment rounding

	if err := gpu.PoolCreate(poolSize); err != nil {
		return nil, fmt.Errorf("pool create (%.1f MB): %w", float64(poolSize)/(1024*1024), err)
	}

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
	// OnesBuf: pre-filled with 1.0s for adaLN scale computation on GPU
	if rs.OnesBuf, err = alloc("OnesBuf", hidden); err != nil {
		return nil, err
	}
	{
		ones := make([]float32, hidden)
		for j := range ones {
			ones[j] = 1.0
		}
		gpu.UploadF32(rs.OnesBuf, ones)
	}

	// Seal pool: stop suballocating, but keep memory alive.
	// Future allocations (PE, VAE, etc.) use normal vkAllocateMemory.
	gpu.PoolSeal()

	log.Printf("[diffusion/gpu] RunState allocated: %.1f MB for maxSeq=%d (pool: %.1f MB)",
		float64(gpu.AllocatedBytes())/(1024*1024), maxSeqLen, float64(poolSize)/(1024*1024))
	return rs, nil
}

// GpuDiTForward runs the DiT forward pass with GPU-accelerated layer computations.
// Embeddings and final layer run on CPU. The 34 transformer layers run on GPU.
func GpuDiTForward(m *DiTModel, gm *GpuDiTModel, rs *DiTRunState, grs *GpuDiTRunState,
	x []float32, timestep float32, context []float32, contextLen, H, W int, debug bool) []float32 {

	cfg := m.Config
	hidden := cfg.HiddenSize
	patchSize := cfg.PatchSize
	hPatches := H / patchSize
	wPatches := W / patchSize
	nImgTokens := hPatches * wPatches
	patchDim := patchSize * patchSize * cfg.InChannels

	// === CPU: Pre-processing (runs once per step, not performance-critical) ===
	cpuPreStart := time.Now()

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
	if debug && gpuDebugProfile == nil {
		gpuDebugProfile = NewGpuDebugProfile()
		gpuDebugProfile.Enabled = true
	}
	dp := gpuDebugProfile
	if dp != nil && dp.Enabled {
		dp.StepNum++
		dp.StartStep(dp.StepNum)
		dp.AddOp("cpu_preprocess", time.Since(cpuPreStart))
	}

	// 7. Context refiner: text tokens only
	t0 := time.Now()
	if err := gpu.UploadF32(grs.X, txt[:nTxtPadded*hidden]); err != nil {
		log.Printf("[diffusion/gpu] upload txt: %v", err)
		return nil
	}
	if dp != nil && dp.Enabled {
		dp.AddOp("upload_txt", time.Since(t0))
	}
	for i := 0; i < gm.NCtxRef; i++ {
		gpuForwardBlock(gm, grs, &gm.Layers[i], nTxtPadded, hidden, pe, 0, nil, peStride, fmt.Sprintf("ctx_ref[%d]", i), dp)
	}

	// Download text back
	t0 = time.Now()
	gpu.DownloadF32(grs.X, txt[:nTxtPadded*hidden])
	if dp != nil && dp.Enabled {
		dp.AddOp("download_txt", time.Since(t0))
	}

	// 8. Noise refiner: image tokens only
	t0 = time.Now()
	if err := gpu.UploadF32(grs.X, img[:nImgPadded*hidden]); err != nil {
		log.Printf("[diffusion/gpu] upload img: %v", err)
		return nil
	}
	if dp != nil && dp.Enabled {
		dp.AddOp("upload_img", time.Since(t0))
	}
	for i := 0; i < gm.NNoise; i++ {
		gpuForwardBlock(gm, grs, &gm.Layers[gm.NCtxRef+i], nImgPadded, hidden, nil, nTxtPadded, rs.TEmb, peStride, fmt.Sprintf("noise_ref[%d]", i), dp)
	}

	// Download image back
	t0 = time.Now()
	gpu.DownloadF32(grs.X, img[:nImgPadded*hidden])
	if dp != nil && dp.Enabled {
		dp.AddOp("download_img", time.Since(t0))
	}

	// 9. Concatenate and upload for main layers
	t0 = time.Now()
	totalSeq := nTxtPadded + nImgPadded
	combined := make([]float32, totalSeq*hidden)
	copy(combined[:nTxtPadded*hidden], txt)
	copy(combined[nTxtPadded*hidden:], img)

	if err := gpu.UploadF32(grs.X, combined); err != nil {
		log.Printf("[diffusion/gpu] upload combined: %v", err)
		return nil
	}
	if dp != nil && dp.Enabled {
		dp.AddOp("upload_combined", time.Since(t0))
	}

	// 10. Main layers
	for i := 0; i < gm.NMain; i++ {
		gpuForwardBlock(gm, grs, &gm.Layers[gm.NCtxRef+gm.NNoise+i], totalSeq, hidden, nil, 0, rs.TEmb, peStride, fmt.Sprintf("main[%d]", i), dp)
	}

	// Download combined result
	t0 = time.Now()
	gpu.DownloadF32(grs.X, combined)
	if dp != nil && dp.Enabled {
		dp.AddOp("download_combined", time.Since(t0))
	}

	// === CPU: Final layer + post-processing ===

	// 11. Final layer
	t0 = time.Now()
	out := forwardFinalLayer(m, rs, combined, totalSeq, rs.TEmb)
	if dp != nil && dp.Enabled {
		dp.AddOp("cpu_final_layer", time.Since(t0))
	}

	// 12–14. Extract and unpatchify
	t0 = time.Now()
	imgStart := nTxtPadded
	imgOut := out[imgStart*patchDim : (imgStart+nImgTokens)*patchDim]
	result := unpatchify(imgOut, cfg.OutChannels, H, W, patchSize)
	for i := range result {
		result[i] = -result[i]
	}
	if dp != nil && dp.Enabled {
		dp.AddOp("cpu_unpatchify", time.Since(t0))
		dp.PrintStepSummary()
	}

	return result
}

// gpuForwardBlock runs one transformer block on GPU.
// adaLNInput is CPU float32 slice (small, computed once per step) or nil.
func gpuForwardBlock(gm *GpuDiTModel, grs *GpuDiTRunState, gl *GpuDiTLayer,
	seqLen, hidden int, pe []float32, peOffset int, adaLNInput []float32, peStride int,
	layerName string, dp *GpuDebugProfile) {

	cfg := gm.Config
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	qDim := numHeads * headDim
	kvDim := numKVHeads * headDim
	ffnDim := cfg.FFNHiddenDim()
	eps := cfg.NormEps

	hasAdaLN := gl.AdaLNWeight != nil && adaLNInput != nil

	h4 := uint64(hidden * 4) // bytes per hidden-dim slice

	debug := dp != nil && dp.Enabled
	layerStart := time.Now()
	var details []opTiming

	// Helper: flush batch to measure a specific operation
	measure := func(name string) {
		if !debug {
			return
		}
		gpu.EndBatch()
		d := time.Since(layerStart)
		dp.AddOp(name, d)
		details = append(details, opTiming{name, d})
		gpu.BeginBatch()
		layerStart = time.Now()
	}

	gpu.BeginBatch()

	if hasAdaLN {
		gpu.UploadF32(grs.ScaleBuf, adaLNInput)
		gpu.BatchMatVec(grs.Mod, gl.AdaLNWeight.Buf, grs.ScaleBuf,
			gl.AdaLNWeight.Rows, gl.AdaLNWeight.Cols, 1, gl.AdaLNWeight.Type)
		if gl.AdaLNBias != 0 {
			gpu.Barrier()
			gpu.Add(grs.Mod, grs.Mod, gl.AdaLNBias, 4*hidden)
		}
		gpu.Barrier()
		gpu.CopyRegion(grs.ScaleBuf, 0, grs.Mod, 0, h4)
		gpu.Barrier()
		gpu.Add(grs.ScaleBuf, grs.ScaleBuf, grs.OnesBuf, hidden)
		gpu.Barrier()
	}
	measure("adaln_mod")

	// --- Attention norm ---
	gpu.BatchRMSNorm(grs.XNorm, grs.X, gl.AttnNorm1, hidden, seqLen, eps)
	if hasAdaLN {
		gpu.Barrier()
		gpu.BroadcastMul(grs.XNorm, grs.ScaleBuf, seqLen*hidden, hidden)
	}
	gpu.Barrier()
	measure("attn_norm")

	// --- QKV projection ---
	gpu.BatchMatVec(grs.QKV, gl.AttnQKV.Buf, grs.XNorm, gl.AttnQKV.Rows, gl.AttnQKV.Cols, seqLen, gl.AttnQKV.Type)
	gpu.Barrier()
	measure("attn_qkv_proj")

	// --- Split + QKNorm + RoPE ---
	gpu.SplitQKV(grs.Q, grs.K, grs.V, grs.QKV, qDim, kvDim, seqLen)
	gpu.Barrier()
	if cfg.QKNorm {
		gpu.RMSNormHeads(grs.Q, gl.QNorm, numHeads*seqLen, headDim, eps)
		gpu.RMSNormHeads(grs.K, gl.KNorm, numKVHeads*seqLen, headDim, eps)
		gpu.Barrier()
	}
	gpu.RoPE3D(grs.Q, grs.PE, seqLen, numHeads, headDim, peOffset, peStride)
	gpu.RoPE3D(grs.K, grs.PE, seqLen, numKVHeads, headDim, peOffset, peStride)
	gpu.Barrier()
	measure("split_qknorm_rope")

	// --- Attention ---
	attnScale := float32(1.0 / math.Sqrt(float64(headDim)))
	gpu.AttentionFullF32(grs.AttnOut, grs.Q, grs.K, grs.V,
		numHeads, numKVHeads, headDim, kvDim, seqLen, attnScale)
	gpu.Barrier()
	measure("attention")

	// --- Attention output projection ---
	gpu.BatchMatVec(grs.Proj, gl.AttnOut.Buf, grs.AttnOut, gl.AttnOut.Rows, gl.AttnOut.Cols, seqLen, gl.AttnOut.Type)
	gpu.Barrier()
	measure("attn_out_proj")

	// --- Attention norm2 + gate residual ---
	gpu.BatchRMSNorm(grs.Proj, grs.Proj, gl.AttnNorm2, hidden, seqLen, eps)
	gpu.Barrier()

	if hasAdaLN {
		gpu.CopyRegion(grs.GateBuf, 0, grs.Mod, h4, h4)
		gpu.CopyRegion(grs.Residual, 0, grs.X, 0, uint64(seqLen*hidden*4))
		gpu.Barrier()
		gpu.TanhGateResidual(grs.X, grs.Residual, grs.Proj, grs.GateBuf, seqLen*hidden, hidden)
	} else {
		gpu.Add(grs.X, grs.X, grs.Proj, seqLen*hidden)
	}
	gpu.Barrier()
	measure("attn_residual")

	// --- FFN path ---
	if hasAdaLN {
		gpu.CopyRegion(grs.ScaleBuf, 0, grs.Mod, 2*h4, h4)
		gpu.Barrier()
		gpu.Add(grs.ScaleBuf, grs.ScaleBuf, grs.OnesBuf, hidden)
		gpu.Barrier()
	}
	gpu.BatchRMSNorm(grs.XNorm, grs.X, gl.FFNNorm1, hidden, seqLen, eps)
	if hasAdaLN {
		gpu.Barrier()
		gpu.BroadcastMul(grs.XNorm, grs.ScaleBuf, seqLen*hidden, hidden)
	}
	gpu.Barrier()
	measure("ffn_norm")

	// --- FFN gate + up projections ---
	gpu.BatchMatVec(grs.Gate, gl.FFNGate.Buf, grs.XNorm, gl.FFNGate.Rows, gl.FFNGate.Cols, seqLen, gl.FFNGate.Type)
	gpu.BatchMatVec(grs.Up, gl.FFNUp.Buf, grs.XNorm, gl.FFNUp.Rows, gl.FFNUp.Cols, seqLen, gl.FFNUp.Type)
	gpu.Barrier()
	measure("ffn_gate_up")

	// --- SwiGLU ---
	gpu.SwiGLU(grs.Hidden, grs.Gate, grs.Up, seqLen*ffnDim)
	gpu.Barrier()
	measure("ffn_swiglu")

	// --- FFN down projection ---
	gpu.BatchMatVec(grs.FFNOut, gl.FFNDown.Buf, grs.Hidden, gl.FFNDown.Rows, gl.FFNDown.Cols, seqLen, gl.FFNDown.Type)
	gpu.Barrier()
	measure("ffn_down_proj")

	// --- FFN norm2 + gate residual ---
	gpu.BatchRMSNorm(grs.FFNOut, grs.FFNOut, gl.FFNNorm2, hidden, seqLen, eps)
	gpu.Barrier()

	if hasAdaLN {
		gpu.CopyRegion(grs.GateBuf, 0, grs.Mod, 3*h4, h4)
		gpu.CopyRegion(grs.Residual, 0, grs.X, 0, uint64(seqLen*hidden*4))
		gpu.Barrier()
		gpu.TanhGateResidual(grs.X, grs.Residual, grs.FFNOut, grs.GateBuf, seqLen*hidden, hidden)
	} else {
		gpu.Add(grs.X, grs.X, grs.FFNOut, seqLen*hidden)
	}

	if !debug {
		gpu.EndBatch()
	} else {
		measure("ffn_residual")
		// Finalize: we already ended the batch in the last measure() call,
		// and started a new one. End the empty batch.
		gpu.EndBatch()

		layerTotal := time.Duration(0)
		for _, d := range details {
			layerTotal += d.Dur
		}
		dp.LayerOps = append(dp.LayerOps, layerTiming{
			Name: layerName, SeqLen: seqLen, Total: layerTotal, Details: details,
		})
	}
}
