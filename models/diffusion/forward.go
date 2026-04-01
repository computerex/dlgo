package diffusion

import (
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/ops"
)

// DiTRunState holds pre-allocated buffers for DiT inference.
type DiTRunState struct {
	cfg  ZImageConfig
	pool *blas.Pool

	// Per-token activation buffers (all [maxSeqLen * hiddenSize] or similar)
	X      []float32 // Current hidden states [seqLen * hidden]
	XNorm  []float32 // Normalized hidden states
	QKV    []float32 // QKV projections [seqLen * 3*qDim]
	Q      []float32 // Q split [seqLen * qDim]
	K      []float32 // K split [seqLen * kvDim]
	V      []float32 // V split [seqLen * kvDim]
	AttnOut []float32 // Attention output [seqLen * qDim]
	Proj    []float32 // Output projection [seqLen * hidden]
	Gate    []float32 // FFN gate [seqLen * ffnDim]
	Up      []float32 // FFN up [seqLen * ffnDim]
	Hidden  []float32 // FFN hidden [seqLen * ffnDim]
	FFNOut  []float32 // FFN output [seqLen * hidden]
	Mod     []float32 // AdaLN modulation [4*hidden]
	Residual []float32 // Skip connection [seqLen * hidden]
	SiLUBuf []float32 // Temp for adaLN SiLU [adaLNEmbedDim]

	// Attention scratch buffer
	Scores []float32 // [maxSeqLen]

	// Final layer buffers
	FinalOut   []float32 // [maxSeqLen * patchDim]
	OnesWeight []float32 // [hidden] filled with 1.0 for LayerNorm
	ZeroBias   []float32 // [hidden] filled with 0.0 for LayerNorm (no affine)
	TanhGate   []float32 // [hidden] precomputed tanh(gate) per layer

	// PE cache (geometry-dependent, reused across diffusion steps)
	cachedPE   []float32
	cachedPEH  int
	cachedPEW  int
	cachedPECtxLen int

	// Temporary per-token buffers
	TEmb    []float32 // Timestep embedding [adaLNEmbedDim=256]
	TEmbMid []float32 // Timestep embedding intermediate [1024]
}

// NewDiTRunState allocates all buffers for inference.
func NewDiTRunState(cfg ZImageConfig, maxSeqLen int) *DiTRunState {
	hidden := cfg.HiddenSize
	ffnDim := cfg.FFNHiddenDim()
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	qkvDim := qDim + 2*kvDim

	patchDim := cfg.PatchSize * cfg.PatchSize * cfg.OutChannels

	onesWeight := make([]float32, hidden)
	for i := range onesWeight {
		onesWeight[i] = 1.0
	}

	return &DiTRunState{
		cfg:     cfg,
		pool:    blas.DefaultPool(),
		X:       make([]float32, maxSeqLen*hidden),
		XNorm:   make([]float32, maxSeqLen*hidden),
		QKV:     make([]float32, maxSeqLen*qkvDim),
		Q:       make([]float32, maxSeqLen*qDim),
		K:       make([]float32, maxSeqLen*kvDim),
		V:       make([]float32, maxSeqLen*kvDim),
		AttnOut: make([]float32, maxSeqLen*qDim),
		Proj:    make([]float32, maxSeqLen*hidden),
		Gate:    make([]float32, maxSeqLen*ffnDim),
		Up:      make([]float32, maxSeqLen*ffnDim),
		Hidden:  make([]float32, maxSeqLen*ffnDim),
		FFNOut:  make([]float32, maxSeqLen*hidden),
		Mod:     make([]float32, 4*hidden),
		Residual: make([]float32, maxSeqLen*hidden),
		SiLUBuf: make([]float32, cfg.AdaLNEmbedDim),
		Scores:  make([]float32, maxSeqLen),
		FinalOut:   make([]float32, maxSeqLen*patchDim),
		OnesWeight: onesWeight,
		ZeroBias:   make([]float32, hidden),
		TanhGate:   make([]float32, hidden),
		TEmb:    make([]float32, cfg.AdaLNEmbedDim),
		TEmbMid: make([]float32, 1024),
	}
}

// DiTForward runs the full Z-Image DiT forward pass.
// x: input latent [1, inCh, H, W] as flat [inCh*H*W]
// timestep: scalar timestep value
// context: text embeddings [contextLen, capFeatDim] as flat
// contextLen: number of text tokens

// Returns: output latent [1, outCh, H, W] as flat [outCh*H*W]
func DiTForward(m *DiTModel, rs *DiTRunState, x []float32, timestep float32,
	context []float32, contextLen, H, W int) []float32 {

	cfg := m.Config
	hidden := cfg.HiddenSize
	patchSize := cfg.PatchSize
	if H%patchSize != 0 || W%patchSize != 0 {
		panic("DiTForward: H and W must be multiples of patchSize")
	}
	hPatches := H / patchSize
	wPatches := W / patchSize
	nImgTokens := hPatches * wPatches
	patchDim := patchSize * patchSize * cfg.InChannels

	// 1. Patchify input: [C, H, W] → [nImgTokens, patchDim]
	imgPatches := patchify(x, cfg.InChannels, H, W, patchSize)

	// 2. Timestep embedding: sinusoidal(timestep) → MLP
	sinEmb := timestepEmbedding(timestep, cfg.AdaLNEmbedDim)
	blas.QMatVecMulParallel(rs.TEmbMid, m.TEmbedMLP0Weight, sinEmb, rs.pool)
	addBias(rs.TEmbMid, m.TEmbedMLP0Bias)
	ops.SiLU(rs.TEmbMid)
	blas.QMatVecMulParallel(rs.TEmb, m.TEmbedMLP2Weight, rs.TEmbMid, rs.pool)
	addBias(rs.TEmb, m.TEmbedMLP2Bias)

	// 3. Caption embedding: RMSNorm(context) → Linear → txt tokens
	txtNormed := make([]float32, contextLen*cfg.CapFeatDim)
	for i := 0; i < contextLen; i++ {
		ops.RMSNorm(txtNormed[i*cfg.CapFeatDim:(i+1)*cfg.CapFeatDim],
			context[i*cfg.CapFeatDim:(i+1)*cfg.CapFeatDim],
			m.CapEmbedNormWeight, cfg.NormEps)
	}
	txt := make([]float32, contextLen*hidden)
	blas.QBatchGEMMParallel(txt, m.CapEmbedLinWeight, txtNormed, contextLen, rs.pool)
	addBiasBatch(txt, m.CapEmbedLinBias, contextLen, hidden)

	// 4. Image embedding: Linear(patches) → img tokens
	img := make([]float32, nImgTokens*hidden)
	blas.QBatchGEMMParallel(img, m.XEmbedWeight, imgPatches, nImgTokens, rs.pool)
	addBiasBatch(img, m.XEmbedBias, nImgTokens, hidden)

	// 5. Pad text and image to multiples of seqMultiOf
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

	// 6. Generate positional embeddings (cached across steps)
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
	// 7. Context refiner: process text tokens only (no adaLN)
	for i := range m.ContextRefiner {
		forwardBlock(m, rs, &m.ContextRefiner[i], txt, nTxtPadded, pe, 0, nil)
	}

	// 8. Noise refiner: process image tokens only (with adaLN from timestep)
	for i := range m.NoiseRefiner {
		forwardBlock(m, rs, &m.NoiseRefiner[i], img, nImgPadded, pe, nTxtPadded, rs.TEmb)
	}

	// 9. Concatenate text + image for main layers — reuse rs.X
	totalSeq := nTxtPadded + nImgPadded
	combined := rs.X[:totalSeq*hidden]
	copy(combined[:nTxtPadded*hidden], txt)
	copy(combined[nTxtPadded*hidden:], img)

	// 10. Main layers: process combined sequence (with adaLN)
	for i := range m.MainLayers {
		forwardBlock(m, rs, &m.MainLayers[i], combined, totalSeq, pe, 0, rs.TEmb)
	}

	// 11. Final layer
	out := forwardFinalLayer(m, rs, combined, totalSeq, rs.TEmb)

	// 12. Extract image tokens (skip text + text pad)
	imgStart := nTxtPadded
	imgOut := out[imgStart*patchDim : (imgStart+nImgTokens)*patchDim]

	// 13. Unpatchify
	result := unpatchify(imgOut, cfg.OutChannels, H, W, patchSize)

	// 14. Negate output (matches sd.cpp: out = ggml_ext_scale(out, -1.f))
	for i := range result {
		result[i] = -result[i]
	}

	return result
}

// forwardBlock runs a single JointTransformerBlock.
func forwardBlock(m *DiTModel, rs *DiTRunState, layer *DiTLayer,
	x []float32, seqLen int, pe []float32, peOffset int, adaLNInput []float32) {

	cfg := m.Config
	hidden := cfg.HiddenSize
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	qDim := numHeads * headDim
	kvDim := numKVHeads * headDim

	hasAdaLN := layer.AdaLNWeight != nil && adaLNInput != nil

	var scaleMSA, gateMSA, scaleMLPMod, gateMLPMod []float32

	if hasAdaLN {
		// Compute adaLN modulation: Linear(input) → split into 4
		// Note: sd.cpp and GGUF weights skip SiLU here (unlike Python reference).
		// The GGUF conversion fuses the SiLU absence into the weight naming (.0 vs .1).
		modSize := 4 * hidden
		mod := rs.Mod[:modSize]
		blas.QMatVecMulParallel(mod, layer.AdaLNWeight, adaLNInput, rs.pool)
		addBias(mod, layer.AdaLNBias)

		scaleMSA = mod[0*hidden : 1*hidden]
		gateMSA = mod[1*hidden : 2*hidden]
		scaleMLPMod = mod[2*hidden : 3*hidden]
		gateMLPMod = mod[3*hidden : 4*hidden]
	}

	// === Self-Attention ===

	// Save residual
	residual := rs.Residual[:seqLen*hidden]
	copy(residual, x)

	// Pre-attention norm1 + modulation
	xNorm := rs.XNorm[:seqLen*hidden]
	for i := 0; i < seqLen; i++ {
		ops.RMSNorm(xNorm[i*hidden:(i+1)*hidden],
			x[i*hidden:(i+1)*hidden],
			layer.AttnNorm1, cfg.NormEps)
	}
	if hasAdaLN {
		// modulate: x = x * (1 + scale)
		for i := 0; i < seqLen; i++ {
			for j := 0; j < hidden; j++ {
				xNorm[i*hidden+j] *= (1.0 + scaleMSA[j])
			}
		}
	}

	// QKV projection
	qkvDim := qDim + 2*kvDim
	qkv := rs.QKV[:seqLen*qkvDim]
	blas.QBatchGEMMParallel(qkv, layer.AttnQKV, xNorm, seqLen, rs.pool)

	// Split Q, K, V
	q := rs.Q[:seqLen*qDim]
	k := rs.K[:seqLen*kvDim]
	v := rs.V[:seqLen*kvDim]
	for i := 0; i < seqLen; i++ {
		copy(q[i*qDim:(i+1)*qDim], qkv[i*qkvDim:i*qkvDim+qDim])
		copy(k[i*kvDim:(i+1)*kvDim], qkv[i*qkvDim+qDim:i*qkvDim+qDim+kvDim])
		copy(v[i*kvDim:(i+1)*kvDim], qkv[i*qkvDim+qDim+kvDim:i*qkvDim+qkvDim])
	}

	// QK norm (RMSNorm per head)
	if cfg.QKNorm {
		for i := 0; i < seqLen; i++ {
			for h := 0; h < numHeads; h++ {
				qHead := q[i*qDim+h*headDim : i*qDim+(h+1)*headDim]
				ops.RMSNormInPlace(qHead, layer.QNorm, cfg.NormEps)
			}
			for h := 0; h < numKVHeads; h++ {
				kHead := k[i*kvDim+h*headDim : i*kvDim+(h+1)*headDim]
				ops.RMSNormInPlace(kHead, layer.KNorm, cfg.NormEps)
			}
		}
	}

	// Apply 3D RoPE
	ApplyRoPE3D(q, pe, seqLen, numHeads, headDim, peOffset)
	ApplyRoPE3D(k, pe, seqLen, numKVHeads, headDim, peOffset)

	// Scaled dot-product attention (scale = 1/sqrt(headDim))
	// NOTE: sd.cpp's kv_scale=1/128 in ggml_ext_attention_ext cancels out completely —
	// it's only for FP16 numerical stability. The actual scale is always 1/sqrt(d_head).
	attnOut := rs.AttnOut[:seqLen*qDim]
	attnScale := float32(1.0 / math.Sqrt(float64(headDim)))
	scaledMultiHeadAttention(attnOut, q, k, v, seqLen, numHeads, numKVHeads, headDim, attnScale, rs.Scores)

	// Output projection
	proj := rs.Proj[:seqLen*hidden]
	blas.QBatchGEMMParallel(proj, layer.AttnOut, attnOut, seqLen, rs.pool)

	// Post-attention norm2
	for i := 0; i < seqLen; i++ {
		ops.RMSNormInPlace(proj[i*hidden:(i+1)*hidden], layer.AttnNorm2, cfg.NormEps)
	}

	// Gate and residual
	if hasAdaLN {
		// Precompute tanh(gateMSA) — values are position-independent
		tanhGate := rs.TanhGate[:hidden]
		for j := 0; j < hidden; j++ {
			tanhGate[j] = float32(math.Tanh(float64(gateMSA[j])))
		}
		for i := 0; i < seqLen; i++ {
			for j := 0; j < hidden; j++ {
				x[i*hidden+j] = residual[i*hidden+j] + proj[i*hidden+j]*tanhGate[j]
			}
		}
	} else {
		for i := 0; i < seqLen*hidden; i++ {
			x[i] = residual[i] + proj[i]
		}
	}

	// === Feed-Forward Network ===

	// Save residual
	copy(residual, x)

	// Pre-FFN norm1 + modulation
	ffnDim := cfg.FFNHiddenDim()
	for i := 0; i < seqLen; i++ {
		ops.RMSNorm(xNorm[i*hidden:(i+1)*hidden],
			x[i*hidden:(i+1)*hidden],
			layer.FFNNorm1, cfg.NormEps)
	}
	if hasAdaLN {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < hidden; j++ {
				xNorm[i*hidden+j] *= (1.0 + scaleMLPMod[j])
			}
		}
	}

	// SwiGLU: gate = SiLU(W1 @ x) * (W3 @ x), out = W2 @ gate
	gate := rs.Gate[:seqLen*ffnDim]
	up := rs.Up[:seqLen*ffnDim]
	blas.QDualBatchGEMMParallel(gate, layer.FFNGate, up, layer.FFNUp, xNorm, seqLen, rs.pool)

	ffnHidden := rs.Hidden[:seqLen*ffnDim]
	for i := 0; i < seqLen; i++ {
		ops.SwiGLU(ffnHidden[i*ffnDim:(i+1)*ffnDim],
			gate[i*ffnDim:(i+1)*ffnDim],
			up[i*ffnDim:(i+1)*ffnDim], ffnDim)
	}

	ffnOut := rs.FFNOut[:seqLen*hidden]
	blas.QBatchGEMMParallel(ffnOut, layer.FFNDown, ffnHidden, seqLen, rs.pool)

	// Post-FFN norm2
	for i := 0; i < seqLen; i++ {
		ops.RMSNormInPlace(ffnOut[i*hidden:(i+1)*hidden], layer.FFNNorm2, cfg.NormEps)
	}

	// Gate and residual
	if hasAdaLN {
		tanhGate := rs.TanhGate[:hidden]
		for j := 0; j < hidden; j++ {
			tanhGate[j] = float32(math.Tanh(float64(gateMLPMod[j])))
		}
		for i := 0; i < seqLen; i++ {
			for j := 0; j < hidden; j++ {
				x[i*hidden+j] = residual[i*hidden+j] + ffnOut[i*hidden+j]*tanhGate[j]
			}
		}
	} else {
		for i := 0; i < seqLen*hidden; i++ {
			x[i] = residual[i] + ffnOut[i]
		}
	}
}

// forwardFinalLayer applies the final DiT layer.
func forwardFinalLayer(m *DiTModel, rs *DiTRunState, x []float32, seqLen int, tEmb []float32) []float32 {
	cfg := m.Config
	hidden := cfg.HiddenSize
	patchDim := cfg.PatchSize * cfg.PatchSize * cfg.OutChannels

	// AdaLN modulation: SiLU(tEmb) → Linear → scale
	// Reuse SiLUBuf for the SiLU(tEmb)
	copy(rs.SiLUBuf, tEmb)
	ops.SiLU(rs.SiLUBuf)

	// Reuse Mod[:hidden] for scale
	scale := rs.Mod[:hidden]
	blas.QMatVecMulParallel(scale, m.FinalAdaLNWeight, rs.SiLUBuf, rs.pool)
	addBias(scale, m.FinalAdaLNBias)

	// LayerNorm (no affine) — reuse XNorm for normed output
	normed := rs.XNorm[:seqLen*hidden]
	for i := 0; i < seqLen; i++ {
		ops.LayerNorm(normed[i*hidden:(i+1)*hidden],
			x[i*hidden:(i+1)*hidden],
			rs.OnesWeight, rs.ZeroBias, 1e-6)
	}

	// Modulate: x = x * (1 + scale)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < hidden; j++ {
			normed[i*hidden+j] *= (1.0 + scale[j])
		}
	}

	// Linear projection to patch space — reuse FinalOut
	out := rs.FinalOut[:seqLen*patchDim]
	blas.QBatchGEMMParallel(out, m.FinalLinWeight, normed, seqLen, rs.pool)
	addBiasBatch(out, m.FinalLinBias, seqLen, patchDim)

	return out
}

// scaledMultiHeadAttention implements multi-head attention with custom scale.
// Supports GQA when numKVHeads < numHeads.
func scaledMultiHeadAttention(out, q, k, v []float32,
	seqLen, numHeads, numKVHeads, headDim int, scale float32, scores []float32) {

	qDim := numHeads * headDim
	kvDim := numKVHeads * headDim
	headsPerKV := numHeads / numKVHeads

	scores = scores[:seqLen]

	for h := 0; h < numHeads; h++ {
		kvH := h / headsPerKV
		qOff := h * headDim
		kvOff := kvH * headDim

		for qi := 0; qi < seqLen; qi++ {
			qSrc := q[qi*qDim+qOff : qi*qDim+qOff+headDim]
			outSlice := out[qi*qDim+qOff : qi*qDim+qOff+headDim]
			ops.Clear(outSlice)

			for ki := 0; ki < seqLen; ki++ {
				kSrc := k[ki*kvDim+kvOff : ki*kvDim+kvOff+headDim]
				scores[ki] = ops.DotProduct(qSrc, kSrc, headDim) * scale
			}

			ops.Softmax(scores)

			for ki := 0; ki < seqLen; ki++ {
				vSrc := v[ki*kvDim+kvOff : ki*kvDim+kvOff+headDim]
				ops.AddScaled(outSlice, scores[ki], vSrc, headDim)
			}
		}
	}
}

// patchify extracts 2D patches from a NCHW tensor.
// input: [C, H, W] flat, output: [nPatches, patchSize*patchSize*C] flat
func patchify(input []float32, C, H, W, patchSize int) []float32 {
	hPatches := H / patchSize
	wPatches := W / patchSize
	patchDim := patchSize * patchSize * C
	nPatches := hPatches * wPatches
	out := make([]float32, nPatches*patchDim)

	for ph := 0; ph < hPatches; ph++ {
		for pw := 0; pw < wPatches; pw++ {
			pIdx := ph*wPatches + pw
			for c := 0; c < C; c++ {
				for kh := 0; kh < patchSize; kh++ {
					for kw := 0; kw < patchSize; kw++ {
						ih := ph*patchSize + kh
						iw := pw*patchSize + kw
						// Channel-last within patch (matching sd.cpp patch_last=false)
						outIdx := pIdx*patchDim + kh*patchSize*C + kw*C + c
						inIdx := c*H*W + ih*W + iw
						out[outIdx] = input[inIdx]
					}
				}
			}
		}
	}
	return out
}

// unpatchify reconstructs a NCHW tensor from patches.
// input: [nPatches, patchDim] flat, output: [C, H, W] flat
func unpatchify(input []float32, C, H, W, patchSize int) []float32 {
	hPatches := H / patchSize
	wPatches := W / patchSize
	patchDim := patchSize * patchSize * C
	out := make([]float32, C*H*W)

	for ph := 0; ph < hPatches; ph++ {
		for pw := 0; pw < wPatches; pw++ {
			pIdx := ph*wPatches + pw
			for c := 0; c < C; c++ {
				for kh := 0; kh < patchSize; kh++ {
					for kw := 0; kw < patchSize; kw++ {
						ih := ph*patchSize + kh
						iw := pw*patchSize + kw
						// Channel-last within patch (matching sd.cpp patch_last=false)
						inIdx := pIdx*patchDim + kh*patchSize*C + kw*C + c
						outIdx := c*H*W + ih*W + iw
						out[outIdx] = input[inIdx]
					}
				}
			}
		}
	}
	return out
}

// timestepEmbedding computes sinusoidal timestep embedding.
func timestepEmbedding(timestep float32, dim int) []float32 {
	halfDim := dim / 2
	emb := make([]float32, dim)
	logTimescale := -math.Log(10000.0) / float64(halfDim)
	for i := 0; i < halfDim; i++ {
		freq := math.Exp(float64(i) * logTimescale)
		angle := float64(timestep) * freq
		emb[i] = float32(math.Cos(angle))
		emb[i+halfDim] = float32(math.Sin(angle))
	}
	return emb
}

// addBias adds bias to a vector in-place. Nil-safe.
func addBias(x, bias []float32) {
	if bias == nil {
		return
	}
	for i := range bias {
		x[i] += bias[i]
	}
}

// addBiasBatch adds bias to each of nPos vectors of size dim.
func addBiasBatch(x, bias []float32, nPos, dim int) {
	if bias == nil {
		return
	}
	for i := 0; i < nPos; i++ {
		for j := 0; j < dim; j++ {
			x[i*dim+j] += bias[j]
		}
	}
}


