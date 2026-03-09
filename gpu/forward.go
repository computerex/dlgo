//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/models/llm"
)

// BuildLayerConfs creates reusable fused-layer configurations from the model,
// run state, and KV cache. Call once after model upload; reuse for every token.
func BuildLayerConfs(m *llm.Model, gm *GpuModel, rs *GpuRunState, kv *GpuKVCache) []*LayerConf {
	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim

	confs := make([]*LayerConf, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		gl := &gm.Layers[l]
		lc := NewLayerConf()

		lc.SetScratch(rs.X, rs.XNorm, rs.Q, rs.K, rs.V, rs.AttnOut, rs.AttnProj,
			rs.FFNNorm, rs.FFNIn, rs.Gate, rs.Up, rs.Hidden, rs.FFNOut)

		if layer.Spec.Core == llm.CoreSSM || layer.Spec.GatedQ {
			lc.SetCoreType(1)
			lc.SetAttnNormOnly(gl.AttnNorm)
		} else {
			lc.SetAttn(gl.AttnNorm, gl.Wq, gl.Wk, gl.Wv, gl.Wo,
				gl.Bq, gl.Bk, gl.Bv, gl.AttnQNorm, gl.AttnKNorm)
			lc.SetKV(kv.KeyBufs[l], kv.ValBufs[l])
		}

		if gl.IsMoE {
			// MoE layer: provide norm weights for pre-FFN residual,
			// but set ffn_type=3 to skip GPU FFN dispatch
			ffnNorm := gl.FFNNorm
			postAttnNorm := gl.PostAttnNorm
			if layer.Spec.Residual == llm.ResPostAttnFFN {
				ffnNorm = gl.PostAttnNorm
				postAttnNorm = 0
			}
			lc.SetFFNMoE(ffnNorm, postAttnNorm)
		} else {
			var ffnGate *GpuTensor
			if gl.FFNGate != nil {
				ffnGate = gl.FFNGate
			}
			ffnNorm := gl.FFNNorm
			postAttnNorm := gl.PostAttnNorm
			if layer.Spec.Residual == llm.ResPostAttnFFN {
				ffnNorm = gl.PostAttnNorm
				postAttnNorm = 0
			}
			lc.SetFFN(ffnNorm, ffnGate, gl.FFNUp, gl.FFNDown,
				postAttnNorm, gl.PostFFNNorm)
		}

		ffnType := 0
		switch layer.Spec.FFN {
		case llm.FFNSwiGLU:
			ffnType = 0
		case llm.FFNGeGLU:
			ffnType = 1
		case llm.FFNPlain:
			ffnType = 2
		case llm.FFNMoE:
			ffnType = 3
		}
		resType := 0
		if layer.Spec.Residual == llm.ResParallel {
			resType = 1
		}

		lc.SetConfig(dim, headDim, numHeads, numKVHeads, kvDim,
			cfg.RMSNormEps, cfg.RopeFreqBase, cfg.RopeDim, cfg.RopeNeox,
			ffnType, resType)

		confs[l] = lc
	}
	return confs
}

// GpuForwardFused performs a single-token forward pass using pre-built layer
// configurations. One CGo call per layer instead of ~20+.
func GpuForwardFused(m *llm.Model, gm *GpuModel, token int32, pos int,
	kv *GpuKVCache, rs *GpuRunState, logitsBuf []float32, layerConfs []*LayerConf) {
	GpuForwardFusedSSM(m, gm, token, pos, kv, rs, logitsBuf, layerConfs, nil)
}

func GpuForwardFusedSSM(m *llm.Model, gm *GpuModel, token int32, pos int,
	kv *GpuKVCache, rs *GpuRunState, logitsBuf []float32, layerConfs []*LayerConf,
	pipe *GpuPipeline) {
	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim

	if layerConfs == nil {
		layerConfs = BuildLayerConfs(m, gm, rs, kv)
	}

	xCPU := make([]float32, dim)
	_ = m.TokenEmbed.DequantizeRow(int(token), xCPU)
	if cfg.EmbedScale != 0 {
		for i := range xCPU {
			xCPU[i] *= cfg.EmbedScale
		}
	}
	seqLen := pos + 1
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// SSM parameters
	var ssmNumHeads, ssmHeadKDim, ssmHeadVDim, ssmKeyDim, ssmQKVDim, ssmConvK int
	if pipe != nil && pipe.HasSSM {
		ssmNumHeads = cfg.SSMTimeStepRank
		ssmHeadVDim = cfg.SSMInnerSize / ssmNumHeads
		ssmHeadKDim = cfg.SSMStateSize
		ssmKVGroups := cfg.SSMGroupCount
		if ssmKVGroups <= 0 {
			ssmKVGroups = ssmNumHeads
		}
		ssmKeyDim = ssmKVGroups * ssmHeadKDim
		ssmQKVDim = ssmKeyDim*2 + ssmNumHeads*ssmHeadVDim
		ssmConvK = cfg.SSMConvKernel
	}

	BeginBatch()
	UploadF32(rs.X, xCPU)

	if m.Layers[0].Spec.Norm == llm.NormRMS {
		Barrier()
		RMSNorm(rs.XNorm, rs.X, gm.Layers[0].AttnNorm, dim, cfg.RMSNormEps)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		gl := &gm.Layers[l]

		if layer.Spec.Core == llm.CoreSSM && pipe != nil && pipe.HasSSM {
			Barrier()
			MatVec(rs.SSMQKV, gl.SSMInProj.Buf, rs.XNorm, gl.SSMInProj.Rows, gl.SSMInProj.Cols, gl.SSMInProj.Type)
			MatVec(rs.SSMZ, gl.SSMGate.Buf, rs.XNorm, gl.SSMGate.Rows, gl.SSMGate.Cols, gl.SSMGate.Type)
			MatVec(rs.SSMAlpha, gl.SSMAlpha.Buf, rs.XNorm, gl.SSMAlpha.Rows, gl.SSMAlpha.Cols, gl.SSMAlpha.Type)
			MatVec(rs.SSMBeta, gl.SSMBeta.Buf, rs.XNorm, gl.SSMBeta.Rows, gl.SSMBeta.Cols, gl.SSMBeta.Type)
			Barrier()
			SSMConv1dSiLU(rs.SSMQKV, gl.SSMConvBuf, gl.SSMConv1dW, ssmQKVDim, ssmConvK)
			Barrier()
			hasDtBias := gl.SSMDtBias != 0
			SSMPreprocess(rs.SSMAlpha, rs.SSMBeta, gl.SSMA, gl.SSMDtBias, rs.SSMQKV,
				ssmNumHeads, ssmHeadKDim, ssmKeyDim, cfg.RMSNormEps, hasDtBias)
			Barrier()
			SSMDeltaRule(gl.SSMState, rs.SSMQKV, rs.SSMAlpha, rs.SSMBeta, rs.SSMY,
				ssmNumHeads, ssmHeadKDim, ssmHeadVDim, ssmKeyDim)
			Barrier()
			SSMNormGate(rs.SSMY, rs.SSMZ, gl.SSMNorm, ssmNumHeads, ssmHeadVDim, cfg.RMSNormEps)
			Barrier()
			MatVec(rs.AttnProj, gl.SSMOut.Buf, rs.SSMY, gl.SSMOut.Rows, gl.SSMOut.Cols, gl.SSMOut.Type)
		} else if layer.Spec.GatedQ && pipe != nil && pipe.HasGatedQ {
			Barrier()
			MatVec(rs.QFull, gl.Wq.Buf, rs.XNorm, gl.Wq.Rows, gl.Wq.Cols, gl.Wq.Type)
			MatVec(rs.K, gl.Wk.Buf, rs.XNorm, gl.Wk.Rows, gl.Wk.Cols, gl.Wk.Type)
			MatVec(rs.V, gl.Wv.Buf, rs.XNorm, gl.Wv.Rows, gl.Wv.Cols, gl.Wv.Type)
			Barrier()
			DeinterleaveQGate(rs.QFull, rs.Q, rs.QGate, numHeads, headDim)
			if gl.Bq != 0 {
				Add(rs.Q, rs.Q, gl.Bq, numHeads*headDim)
			}
			if gl.Bk != 0 {
				Add(rs.K, rs.K, gl.Bk, kvDim)
			}
			if gl.Bv != 0 {
				Add(rs.V, rs.V, gl.Bv, kvDim)
			}
			if layer.Spec.QKNorm {
				Barrier()
				RMSNormHeads(rs.Q, gl.AttnQNorm, numHeads, headDim, cfg.RMSNormEps)
				RMSNormHeads(rs.K, gl.AttnKNorm, numKVHeads, headDim, cfg.RMSNormEps)
			}
			Barrier()
			RoPE(rs.Q, rs.K, numHeads, numKVHeads, headDim, cfg.RopeDim, pos, cfg.RopeFreqBase, cfg.RopeNeox)
			KVStore(kv.KeyBufs[l], kv.ValBufs[l], rs.K, rs.V, pos, kvDim)
			Barrier()
			Attention(rs.AttnOut, rs.Q, kv.KeyBufs[l], kv.ValBufs[l],
				numHeads, numKVHeads, headDim, kvDim, seqLen, scale)
			Barrier()
			SigmoidGate(rs.AttnOut, rs.QGate, numHeads*headDim)
			Barrier()
			MatVec(rs.AttnProj, gl.Wo.Buf, rs.AttnOut, gl.Wo.Rows, gl.Wo.Cols, gl.Wo.Type)
		}

		var nextAttnNorm Buf
		if l < cfg.NumLayers-1 {
			nextAttnNorm = gm.Layers[l+1].AttnNorm
		}
		ForwardLayer(layerConfs[l], pos, seqLen, scale, nextAttnNorm)

		if gl.IsMoE && pipe != nil && pipe.HasMoE {
			// ForwardLayer computed pre-FFN residual+norm then returned early.
			// Download ffn_norm, run MoE FFN on CPU, upload result, do residual.
			Sync()
			cpuRS := pipe.CPURunState
			DownloadF32(rs.FFNNorm, cpuRS.FFNNorm)
			llm.ForwardMoEFFN(layer, cpuRS, cpuRS.FFNNorm, cfg, cpuRS.Pool)
			BeginBatch()
			UploadF32(rs.FFNOut, cpuRS.FFNOut)
			Barrier()
			if nextAttnNorm != 0 {
				AddRMSNorm(rs.XNorm, rs.X, rs.FFNIn, rs.FFNOut, nextAttnNorm, dim, cfg.RMSNormEps)
			} else {
				Add(rs.X, rs.FFNIn, rs.FFNOut, dim)
			}
		}
	}

	Barrier()
	RMSNorm(rs.X, rs.X, gm.OutputNorm, dim, cfg.RMSNormEps)
	Barrier()
	output := gm.Output
	if output == nil {
		output = gm.TokenEmbed
	}
	MatVec(rs.Logits, output.Buf, rs.X, output.Rows, output.Cols, output.Type)
	DownloadF32(rs.Logits, logitsBuf)
}

// BuildBatchLayerConfs creates layer configs that point to batch-sized scratch
// buffers while sharing the same weight/norm buffers as single-token configs.
func BuildBatchLayerConfs(m *llm.Model, gm *GpuModel, bs *GpuBatchState, kv *GpuKVCache) []*LayerConf {
	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim

	confs := make([]*LayerConf, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		gl := &gm.Layers[l]
		lc := NewLayerConf()

		lc.SetScratch(bs.X, bs.XNorm, bs.Q, bs.K, bs.V, bs.AttnOut, bs.AttnProj,
			bs.FFNNorm, bs.FFNIn, bs.Gate, bs.Up, bs.Hidden, bs.FFNOut)

		if layer.Spec.Core == llm.CoreSSM || layer.Spec.GatedQ {
			lc.SetCoreType(1)
			lc.SetAttnNormOnly(gl.AttnNorm)
		} else {
			lc.SetAttn(gl.AttnNorm, gl.Wq, gl.Wk, gl.Wv, gl.Wo,
				gl.Bq, gl.Bk, gl.Bv, gl.AttnQNorm, gl.AttnKNorm)
			lc.SetKV(kv.KeyBufs[l], kv.ValBufs[l])
		}

		if gl.IsMoE {
			ffnNorm := gl.FFNNorm
			postAttnNorm := gl.PostAttnNorm
			if layer.Spec.Residual == llm.ResPostAttnFFN {
				ffnNorm = gl.PostAttnNorm
				postAttnNorm = 0
			}
			lc.SetFFNMoE(ffnNorm, postAttnNorm)
		} else {
			var ffnGate *GpuTensor
			if gl.FFNGate != nil {
				ffnGate = gl.FFNGate
			}
			ffnNorm := gl.FFNNorm
			postAttnNorm := gl.PostAttnNorm
			if layer.Spec.Residual == llm.ResPostAttnFFN {
				ffnNorm = gl.PostAttnNorm
				postAttnNorm = 0
			}
			lc.SetFFN(ffnNorm, ffnGate, gl.FFNUp, gl.FFNDown,
				postAttnNorm, gl.PostFFNNorm)
		}

		ffnType := 0
		switch layer.Spec.FFN {
		case llm.FFNSwiGLU:
			ffnType = 0
		case llm.FFNGeGLU:
			ffnType = 1
		case llm.FFNPlain:
			ffnType = 2
		case llm.FFNMoE:
			ffnType = 3
		}
		resType := 0
		if layer.Spec.Residual == llm.ResParallel {
			resType = 1
		}

		lc.SetConfig(dim, headDim, numHeads, numKVHeads, kvDim,
			cfg.RMSNormEps, cfg.RopeFreqBase, cfg.RopeDim, cfg.RopeNeox,
			ffnType, resType)

		confs[l] = lc
	}
	return confs
}

// GpuForwardPrefillBatch processes all prompt tokens in a single batched pass.
func GpuForwardPrefillBatch(m *llm.Model, gm *GpuModel, tokens []int32,
	kv *GpuKVCache, rs *GpuRunState, bs *GpuBatchState, logitsBuf []float32,
	batchLayerConfs []*LayerConf) {

	npos := len(tokens)
	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	xBatch := make([]float32, npos*dim)
	for i, tok := range tokens {
		_ = m.TokenEmbed.DequantizeRow(int(tok), xBatch[i*dim:(i+1)*dim])
		if cfg.EmbedScale != 0 {
			for j := 0; j < dim; j++ {
				xBatch[i*dim+j] *= cfg.EmbedScale
			}
		}
	}

	BeginBatch()
	UploadF32(bs.X, xBatch)

	if m.Layers[0].Spec.Norm == llm.NormRMS {
		Barrier()
		BatchRMSNorm(bs.XNorm, bs.X, gm.Layers[0].AttnNorm, dim, npos, cfg.RMSNormEps)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		var nextAttnNorm Buf
		if l < cfg.NumLayers-1 {
			nextAttnNorm = gm.Layers[l+1].AttnNorm
		}
		ForwardLayerBatch(batchLayerConfs[l], npos, 0, scale, nextAttnNorm)
	}

	Barrier()
	CopyRegion(rs.X, 0, bs.X, uint64((npos-1)*dim*4), uint64(dim*4))
	Barrier()
	RMSNorm(rs.X, rs.X, gm.OutputNorm, dim, cfg.RMSNormEps)
	Barrier()
	output := gm.Output
	if output == nil {
		output = gm.TokenEmbed
	}
	MatVec(rs.Logits, output.Buf, rs.X, output.Rows, output.Cols, output.Type)
	DownloadF32(rs.Logits, logitsBuf)
}

// GpuForwardPrefillBatchHybrid performs batched prefill for hybrid SSM+attention models.
// Attention layers (including GatedQ) and SSM matmuls are batched; SSM recurrence is per-position.
func GpuForwardPrefillBatchHybrid(m *llm.Model, gm *GpuModel, tokens []int32,
	kv *GpuKVCache, rs *GpuRunState, bs *GpuBatchState, logitsBuf []float32,
	batchLayerConfs []*LayerConf, pipe *GpuPipeline) {

	npos := len(tokens)
	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim
	qDim := numHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	var ssmNumHeads, ssmHeadKDim, ssmHeadVDim, ssmKeyDim, ssmQKVDim, ssmConvK int
	if pipe.HasSSM {
		ssmNumHeads = cfg.SSMTimeStepRank
		ssmHeadVDim = cfg.SSMInnerSize / ssmNumHeads
		ssmHeadKDim = cfg.SSMStateSize
		ssmKVGroups := cfg.SSMGroupCount
		if ssmKVGroups <= 0 {
			ssmKVGroups = ssmNumHeads
		}
		ssmKeyDim = ssmKVGroups * ssmHeadKDim
		ssmQKVDim = ssmKeyDim*2 + ssmNumHeads*ssmHeadVDim
		ssmConvK = cfg.SSMConvKernel
	}

	xBatch := make([]float32, npos*dim)
	for i, tok := range tokens {
		_ = m.TokenEmbed.DequantizeRow(int(tok), xBatch[i*dim:(i+1)*dim])
		if cfg.EmbedScale != 0 {
			for j := 0; j < dim; j++ {
				xBatch[i*dim+j] *= cfg.EmbedScale
			}
		}
	}

	BeginBatch()
	UploadF32(bs.X, xBatch)

	if m.Layers[0].Spec.Norm == llm.NormRMS {
		Barrier()
		BatchRMSNorm(bs.XNorm, bs.X, gm.Layers[0].AttnNorm, dim, npos, cfg.RMSNormEps)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		gl := &gm.Layers[l]

		if layer.Spec.Core == llm.CoreSSM && pipe.HasSSM {
			// Batch the 4 SSM input matmuls
			Barrier()
			BatchMatVec(bs.SSMQKV, gl.SSMInProj.Buf, bs.XNorm, gl.SSMInProj.Rows, gl.SSMInProj.Cols, npos, gl.SSMInProj.Type)
			BatchMatVec(bs.SSMZ, gl.SSMGate.Buf, bs.XNorm, gl.SSMGate.Rows, gl.SSMGate.Cols, npos, gl.SSMGate.Type)
			BatchMatVec(bs.SSMAlpha, gl.SSMAlpha.Buf, bs.XNorm, gl.SSMAlpha.Rows, gl.SSMAlpha.Cols, npos, gl.SSMAlpha.Type)
			BatchMatVec(bs.SSMBeta, gl.SSMBeta.Buf, bs.XNorm, gl.SSMBeta.Rows, gl.SSMBeta.Cols, npos, gl.SSMBeta.Type)
			Barrier()

			// Per-position: copy from batch to single-token buffers, run recurrence, copy Y back
			for p := 0; p < npos; p++ {
				off := uint64(p)
				CopyRegion(rs.SSMQKV, 0, bs.SSMQKV, off*uint64(ssmQKVDim*4), uint64(ssmQKVDim*4))
				CopyRegion(rs.SSMZ, 0, bs.SSMZ, off*uint64(ssmNumHeads*ssmHeadVDim*4), uint64(ssmNumHeads*ssmHeadVDim*4))
				CopyRegion(rs.SSMAlpha, 0, bs.SSMAlpha, off*uint64(ssmNumHeads*4), uint64(ssmNumHeads*4))
				CopyRegion(rs.SSMBeta, 0, bs.SSMBeta, off*uint64(ssmNumHeads*4), uint64(ssmNumHeads*4))
				Barrier()
				SSMConv1dSiLU(rs.SSMQKV, gl.SSMConvBuf, gl.SSMConv1dW, ssmQKVDim, ssmConvK)
				Barrier()
				hasDtBias := gl.SSMDtBias != 0
				SSMPreprocess(rs.SSMAlpha, rs.SSMBeta, gl.SSMA, gl.SSMDtBias, rs.SSMQKV,
					ssmNumHeads, ssmHeadKDim, ssmKeyDim, cfg.RMSNormEps, hasDtBias)
				Barrier()
				SSMDeltaRule(gl.SSMState, rs.SSMQKV, rs.SSMAlpha, rs.SSMBeta, rs.SSMY,
					ssmNumHeads, ssmHeadKDim, ssmHeadVDim, ssmKeyDim)
				Barrier()
				SSMNormGate(rs.SSMY, rs.SSMZ, gl.SSMNorm, ssmNumHeads, ssmHeadVDim, cfg.RMSNormEps)
				Barrier()
				CopyRegion(bs.SSMY, off*uint64(ssmNumHeads*ssmHeadVDim*4), rs.SSMY, 0, uint64(ssmNumHeads*ssmHeadVDim*4))
				Barrier()
			}

			// Batch SSMOut matmul
			BatchMatVec(bs.AttnProj, gl.SSMOut.Buf, bs.SSMY, gl.SSMOut.Rows, gl.SSMOut.Cols, npos, gl.SSMOut.Type)

		} else if layer.Spec.GatedQ && pipe.HasGatedQ {
			// GatedQ attention: fully batched Q/K/V, deinterleave, bias, QKnorm, RoPE, KV store, attention, sigmoid gate
			Barrier()
			BatchMatVec(bs.QFull, gl.Wq.Buf, bs.XNorm, gl.Wq.Rows, gl.Wq.Cols, npos, gl.Wq.Type)
			BatchMatVec(bs.K, gl.Wk.Buf, bs.XNorm, gl.Wk.Rows, gl.Wk.Cols, npos, gl.Wk.Type)
			BatchMatVec(bs.V, gl.Wv.Buf, bs.XNorm, gl.Wv.Rows, gl.Wv.Cols, npos, gl.Wv.Type)
			Barrier()
			DeinterleaveQGate(bs.QFull, bs.Q, bs.QGate, numHeads*npos, headDim)
			if gl.Bq != 0 {
				Barrier()
				BatchAddBias(bs.Q, gl.Bq, bs.AttnOut, qDim, npos)
			}
			if gl.Bk != 0 {
				BatchAddBias(bs.K, gl.Bk, bs.AttnOut, kvDim, npos)
			}
			if gl.Bv != 0 {
				BatchAddBias(bs.V, gl.Bv, bs.AttnOut, kvDim, npos)
			}
			if layer.Spec.QKNorm {
				Barrier()
				RMSNormHeads(bs.Q, gl.AttnQNorm, numHeads*npos, headDim, cfg.RMSNormEps)
				RMSNormHeads(bs.K, gl.AttnKNorm, numKVHeads*npos, headDim, cfg.RMSNormEps)
			}
			Barrier()
			BatchRoPE(bs.Q, bs.K, numHeads, numKVHeads, headDim, cfg.RopeDim, 0,
				cfg.RopeFreqBase, cfg.RopeNeox, npos)
			BatchKVStore(kv.KeyBufs[l], kv.ValBufs[l], bs.K, bs.V, 0, kvDim, npos)
			BatchAttention(bs.AttnOut, bs.Q, kv.KeyBufs[l], kv.ValBufs[l],
				numHeads, numKVHeads, headDim, kvDim, 1, scale, npos)
			Barrier()
			SigmoidGate(bs.AttnOut, bs.QGate, qDim*npos)
			Barrier()
			BatchMatVec(bs.AttnProj, gl.Wo.Buf, bs.AttnOut, gl.Wo.Rows, gl.Wo.Cols, npos, gl.Wo.Type)

		} else {
			// Standard attention (non-GatedQ, non-SSM) - handled by ForwardLayerBatch directly
			var nextAttnNorm Buf
			if l < cfg.NumLayers-1 {
				nextAttnNorm = gm.Layers[l+1].AttnNorm
			}
			ForwardLayerBatch(batchLayerConfs[l], npos, 0, scale, nextAttnNorm)
			continue
		}

		// FFN + residual via ForwardLayerBatch with core_type=1 (skips attention)
		var nextAttnNorm Buf
		if l < cfg.NumLayers-1 {
			nextAttnNorm = gm.Layers[l+1].AttnNorm
		}
		ForwardLayerBatch(batchLayerConfs[l], npos, 0, scale, nextAttnNorm)

		if gl.IsMoE && pipe.HasMoE {
			// ForwardLayerBatch computed pre-FFN residual+norm for all positions
			// then returned early (ffn_type=3). Run MoE FFN per-position on CPU.
			Sync()
			cpuRS := pipe.CPURunState
			ffnNormBatch := make([]float32, npos*dim)
			DownloadF32(bs.FFNNorm, ffnNormBatch)
			ffnOutBatch := make([]float32, npos*dim)
			for p := 0; p < npos; p++ {
				posInput := ffnNormBatch[p*dim : (p+1)*dim]
				llm.ForwardMoEFFN(layer, cpuRS, posInput, cfg, cpuRS.Pool)
				copy(ffnOutBatch[p*dim:(p+1)*dim], cpuRS.FFNOut)
			}
			BeginBatch()
			UploadF32(bs.FFNOut, ffnOutBatch)
			Barrier()
			Add(bs.X, bs.FFNIn, bs.FFNOut, dim*npos)
			if nextAttnNorm != 0 {
				Barrier()
				BatchRMSNorm(bs.XNorm, bs.X, nextAttnNorm, dim, npos, cfg.RMSNormEps)
			}
		}
	}

	Barrier()
	CopyRegion(rs.X, 0, bs.X, uint64((npos-1)*dim*4), uint64(dim*4))
	Barrier()
	RMSNorm(rs.X, rs.X, gm.OutputNorm, dim, cfg.RMSNormEps)
	Barrier()
	output := gm.Output
	if output == nil {
		output = gm.TokenEmbed
	}
	MatVec(rs.Logits, output.Buf, rs.X, output.Rows, output.Cols, output.Type)
	DownloadF32(rs.Logits, logitsBuf)
}

// GpuForward performs a single-token forward pass entirely on GPU.
// This is the general path with error handling and CPU fallback for
// unsupported quant types.
func GpuForward(m *llm.Model, gm *GpuModel, token int32, pos int,
	kv *GpuKVCache, rs *GpuRunState, logitsBuf []float32, pipe ...*GpuPipeline) error {
	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim

	xCPU := make([]float32, dim)
	_ = m.TokenEmbed.DequantizeRow(int(token), xCPU)
	if cfg.EmbedScale != 0 {
		for i := range xCPU {
			xCPU[i] *= cfg.EmbedScale
		}
	}
	seqLen := pos + 1
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// SSM parameters
	var p *GpuPipeline
	if len(pipe) > 0 {
		p = pipe[0]
	}
	var ssmNumHeads, ssmHeadKDim, ssmHeadVDim, ssmKeyDim, ssmQKVDim, ssmConvK int
	if p != nil && p.HasSSM {
		ssmNumHeads = cfg.SSMTimeStepRank
		ssmHeadVDim = cfg.SSMInnerSize / ssmNumHeads
		ssmHeadKDim = cfg.SSMStateSize
		ssmKVGroups := cfg.SSMGroupCount
		if ssmKVGroups <= 0 {
			ssmKVGroups = ssmNumHeads
		}
		ssmKeyDim = ssmKVGroups * ssmHeadKDim
		ssmQKVDim = ssmKeyDim*2 + ssmNumHeads*ssmHeadVDim
		ssmConvK = cfg.SSMConvKernel
	}

	BeginBatch()
	if err := UploadF32(rs.X, xCPU); err != nil {
		return err
	}

	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		spec := &layer.Spec
		gl := &gm.Layers[l]

		Barrier()
		if spec.Norm == llm.NormRMS {
			if err := RMSNorm(rs.XNorm, rs.X, gl.AttnNorm, dim, cfg.RMSNormEps); err != nil {
				return fmt.Errorf("layer %d attn rmsnorm: %w", l, err)
			}
		}

		if spec.Core == llm.CoreSSM && p != nil && p.HasSSM {
			Barrier()
			MatVec(rs.SSMQKV, gl.SSMInProj.Buf, rs.XNorm, gl.SSMInProj.Rows, gl.SSMInProj.Cols, gl.SSMInProj.Type)
			MatVec(rs.SSMZ, gl.SSMGate.Buf, rs.XNorm, gl.SSMGate.Rows, gl.SSMGate.Cols, gl.SSMGate.Type)
			MatVec(rs.SSMAlpha, gl.SSMAlpha.Buf, rs.XNorm, gl.SSMAlpha.Rows, gl.SSMAlpha.Cols, gl.SSMAlpha.Type)
			MatVec(rs.SSMBeta, gl.SSMBeta.Buf, rs.XNorm, gl.SSMBeta.Rows, gl.SSMBeta.Cols, gl.SSMBeta.Type)
			Barrier()
			SSMConv1dSiLU(rs.SSMQKV, gl.SSMConvBuf, gl.SSMConv1dW, ssmQKVDim, ssmConvK)
			Barrier()
			hasDtBias := gl.SSMDtBias != 0
			SSMPreprocess(rs.SSMAlpha, rs.SSMBeta, gl.SSMA, gl.SSMDtBias, rs.SSMQKV,
				ssmNumHeads, ssmHeadKDim, ssmKeyDim, cfg.RMSNormEps, hasDtBias)
			Barrier()
			SSMDeltaRule(gl.SSMState, rs.SSMQKV, rs.SSMAlpha, rs.SSMBeta, rs.SSMY,
				ssmNumHeads, ssmHeadKDim, ssmHeadVDim, ssmKeyDim)
			Barrier()
			SSMNormGate(rs.SSMY, rs.SSMZ, gl.SSMNorm, ssmNumHeads, ssmHeadVDim, cfg.RMSNormEps)
			Barrier()
			if err := gpuMatVec(rs.AttnProj, gl.SSMOut, layer.SSMOut, rs.SSMY, rs); err != nil {
				return fmt.Errorf("layer %d ssm out: %w", l, err)
			}
		} else if spec.GatedQ && p != nil && p.HasGatedQ {
			Barrier()
			if err := gpuMatVec(rs.QFull, gl.Wq, layer.Wq, rs.XNorm, rs); err != nil {
				return fmt.Errorf("layer %d wq: %w", l, err)
			}
			if err := gpuMatVec(rs.K, gl.Wk, layer.Wk, rs.XNorm, rs); err != nil {
				return fmt.Errorf("layer %d wk: %w", l, err)
			}
			if err := gpuMatVec(rs.V, gl.Wv, layer.Wv, rs.XNorm, rs); err != nil {
				return fmt.Errorf("layer %d wv: %w", l, err)
			}
			Barrier()
			DeinterleaveQGate(rs.QFull, rs.Q, rs.QGate, numHeads, headDim)
			if gl.Bq != 0 {
				Add(rs.Q, rs.Q, gl.Bq, numHeads*headDim)
			}
			if gl.Bk != 0 {
				Add(rs.K, rs.K, gl.Bk, kvDim)
			}
			if gl.Bv != 0 {
				Add(rs.V, rs.V, gl.Bv, kvDim)
			}
			if spec.QKNorm {
				Barrier()
				RMSNormHeads(rs.Q, gl.AttnQNorm, numHeads, headDim, cfg.RMSNormEps)
				RMSNormHeads(rs.K, gl.AttnKNorm, numKVHeads, headDim, cfg.RMSNormEps)
			}
			Barrier()
			RoPE(rs.Q, rs.K, numHeads, numKVHeads, headDim, cfg.RopeDim, pos, cfg.RopeFreqBase, cfg.RopeNeox)
			KVStore(kv.KeyBufs[l], kv.ValBufs[l], rs.K, rs.V, pos, kvDim)
			Barrier()
			Attention(rs.AttnOut, rs.Q, kv.KeyBufs[l], kv.ValBufs[l],
				numHeads, numKVHeads, headDim, kvDim, seqLen, scale)
			Barrier()
			SigmoidGate(rs.AttnOut, rs.QGate, numHeads*headDim)
			Barrier()
			if err := gpuMatVec(rs.AttnProj, gl.Wo, layer.Wo, rs.AttnOut, rs); err != nil {
				return fmt.Errorf("layer %d wo: %w", l, err)
			}
		} else if spec.Core == llm.CoreAttention {
			Barrier()
			if err := gpuMatVec(rs.Q, gl.Wq, layer.Wq, rs.XNorm, rs); err != nil {
				return fmt.Errorf("layer %d wq: %w", l, err)
			}
			if err := gpuMatVec(rs.K, gl.Wk, layer.Wk, rs.XNorm, rs); err != nil {
				return fmt.Errorf("layer %d wk: %w", l, err)
			}
			if err := gpuMatVec(rs.V, gl.Wv, layer.Wv, rs.XNorm, rs); err != nil {
				return fmt.Errorf("layer %d wv: %w", l, err)
			}

			Barrier()
			if gl.Bq != 0 {
				if err := addBuf(rs.Q, gl.Bq, numHeads*headDim); err != nil {
					return fmt.Errorf("layer %d bq: %w", l, err)
				}
			}
			if gl.Bk != 0 {
				if err := addBuf(rs.K, gl.Bk, kvDim); err != nil {
					return fmt.Errorf("layer %d bk: %w", l, err)
				}
			}
			if gl.Bv != 0 {
				if err := addBuf(rs.V, gl.Bv, kvDim); err != nil {
					return fmt.Errorf("layer %d bv: %w", l, err)
				}
			}

			if spec.QKNorm {
				Barrier()
				if err := RMSNormHeads(rs.Q, gl.AttnQNorm, numHeads, headDim, cfg.RMSNormEps); err != nil {
					return fmt.Errorf("layer %d qnorm: %w", l, err)
				}
				if err := RMSNormHeads(rs.K, gl.AttnKNorm, numKVHeads, headDim, cfg.RMSNormEps); err != nil {
					return fmt.Errorf("layer %d knorm: %w", l, err)
				}
			}

			Barrier()
			if err := RoPE(rs.Q, rs.K, numHeads, numKVHeads, headDim, cfg.RopeDim, pos, cfg.RopeFreqBase, cfg.RopeNeox); err != nil {
				return fmt.Errorf("layer %d rope: %w", l, err)
			}

			if err := KVStore(kv.KeyBufs[l], kv.ValBufs[l], rs.K, rs.V, pos, kvDim); err != nil {
				return fmt.Errorf("layer %d kvstore: %w", l, err)
			}

			Barrier()
			if err := Attention(rs.AttnOut, rs.Q, kv.KeyBufs[l], kv.ValBufs[l],
				numHeads, numKVHeads, headDim, kvDim, seqLen, scale); err != nil {
				return fmt.Errorf("layer %d attention: %w", l, err)
			}

			Barrier()
			if err := gpuMatVec(rs.AttnProj, gl.Wo, layer.Wo, rs.AttnOut, rs); err != nil {
				return fmt.Errorf("layer %d wo: %w", l, err)
			}
		}

		if gl.IsMoE && p != nil && p.HasMoE {
			// MoE: compute pre-FFN residual+norm on GPU, FFN on CPU
			ffnNormW := gl.FFNNorm
			if spec.Residual == llm.ResPostAttnFFN {
				ffnNormW = gl.PostAttnNorm
			}
			Barrier()
			if err := AddRMSNorm(rs.FFNNorm, rs.FFNIn, rs.X, rs.AttnProj, ffnNormW, dim, cfg.RMSNormEps); err != nil {
				return fmt.Errorf("layer %d moe add+rmsnorm: %w", l, err)
			}
			Sync()
			cpuRS := p.CPURunState
			if err := DownloadF32(rs.FFNNorm, cpuRS.FFNNorm); err != nil {
				return fmt.Errorf("layer %d moe download: %w", l, err)
			}
			llm.ForwardMoEFFN(layer, cpuRS, cpuRS.FFNNorm, cfg, cpuRS.Pool)
			BeginBatch()
			if err := UploadF32(rs.FFNOut, cpuRS.FFNOut); err != nil {
				return fmt.Errorf("layer %d moe upload: %w", l, err)
			}
			Barrier()
			if err := Add(rs.X, rs.FFNIn, rs.FFNOut, dim); err != nil {
				return fmt.Errorf("layer %d moe residual: %w", l, err)
			}
		} else {
		switch spec.Residual {
		case llm.ResStandard:
			Barrier()
			if gl.PostAttnNorm != 0 {
				if err := RMSNorm(rs.AttnProj, rs.AttnProj, gl.PostAttnNorm, dim, cfg.RMSNormEps); err != nil {
					return fmt.Errorf("layer %d post-attn norm: %w", l, err)
				}
				Barrier()
			}
			if err := AddRMSNorm(rs.FFNNorm, rs.FFNIn, rs.X, rs.AttnProj, gl.FFNNorm, dim, cfg.RMSNormEps); err != nil {
				return fmt.Errorf("layer %d add+rmsnorm: %w", l, err)
			}
			Barrier()
			if err := gpuForwardFFN(layer, gl, rs, rs.FFNNorm, dim, cfg); err != nil {
				return fmt.Errorf("layer %d ffn: %w", l, err)
			}
			Barrier()
			if gl.PostFFNNorm != 0 {
				if err := RMSNorm(rs.FFNOut, rs.FFNOut, gl.PostFFNNorm, dim, cfg.RMSNormEps); err != nil {
					return fmt.Errorf("layer %d post-ffn norm: %w", l, err)
				}
				Barrier()
			}
			if err := Add(rs.X, rs.FFNIn, rs.FFNOut, dim); err != nil {
				return fmt.Errorf("layer %d residual add: %w", l, err)
			}

		case llm.ResPostAttnFFN:
			Barrier()
			if err := AddRMSNorm(rs.FFNNorm, rs.FFNIn, rs.X, rs.AttnProj, gl.PostAttnNorm, dim, cfg.RMSNormEps); err != nil {
				return fmt.Errorf("layer %d add+rmsnorm: %w", l, err)
			}
			Barrier()
			if err := gpuForwardFFN(layer, gl, rs, rs.FFNNorm, dim, cfg); err != nil {
				return fmt.Errorf("layer %d ffn: %w", l, err)
			}
			Barrier()
			if err := Add(rs.X, rs.FFNIn, rs.FFNOut, dim); err != nil {
				return fmt.Errorf("layer %d residual add: %w", l, err)
			}

		case llm.ResParallel:
			Barrier()
			if err := gpuForwardFFN(layer, gl, rs, rs.XNorm, dim, cfg); err != nil {
				return fmt.Errorf("layer %d ffn: %w", l, err)
			}
			Barrier()
			if err := Add(rs.X, rs.X, rs.AttnProj, dim); err != nil {
				return fmt.Errorf("layer %d parallel add attn: %w", l, err)
			}
			Barrier()
			if err := Add(rs.X, rs.X, rs.FFNOut, dim); err != nil {
				return fmt.Errorf("layer %d parallel add ffn: %w", l, err)
			}
		}
		} // end else (non-MoE)
	}

	Barrier()
	if err := RMSNorm(rs.X, rs.X, gm.OutputNorm, dim, cfg.RMSNormEps); err != nil {
		return fmt.Errorf("output norm: %w", err)
	}
	Barrier()
	output := gm.Output
	outputCPU := m.Output
	if output == nil {
		output = gm.TokenEmbed
		outputCPU = m.TokenEmbed
	}
	if err := gpuMatVec(rs.Logits, output, outputCPU, rs.X, rs); err != nil {
		if output == gm.TokenEmbed {
			return fmt.Errorf("output matvec (token embed): %w", err)
		}
		return fmt.Errorf("output matvec: %w", err)
	}

	if err := DownloadF32(rs.Logits, logitsBuf); err != nil {
		return err
	}
	return nil
}

func addBuf(dst, src Buf, n int) error {
	return Add(dst, dst, src, n)
}

func gpuForwardFFN(layer *llm.Layer, gl *GpuLayer, rs *GpuRunState, input Buf, dim int, cfg llm.ModelConfig) error {
	switch layer.Spec.FFN {
	case llm.FFNSwiGLU:
		if supportsGPUQType(gl.FFNGate.Type) && supportsGPUQType(gl.FFNUp.Type) {
			if err := MatVec(rs.Gate, gl.FFNGate.Buf, input, gl.FFNGate.Rows, gl.FFNGate.Cols, gl.FFNGate.Type); err != nil {
				return err
			}
			if err := MatVec(rs.Up, gl.FFNUp.Buf, input, gl.FFNUp.Rows, gl.FFNUp.Cols, gl.FFNUp.Type); err != nil {
				return err
			}
		} else if err := gpuDualMatVec(rs.Gate, gl.FFNGate, layer.FFNGate, rs.Up, gl.FFNUp, layer.FFNUp, input, rs); err != nil {
			return err
		}
		Barrier()
		if err := SwiGLU(rs.Hidden, rs.Gate, rs.Up, gl.FFNGate.Rows); err != nil {
			return err
		}
		Barrier()
		return gpuMatVec(rs.FFNOut, gl.FFNDown, layer.FFNDown, rs.Hidden, rs)

	case llm.FFNGeGLU:
		if supportsGPUQType(gl.FFNGate.Type) && supportsGPUQType(gl.FFNUp.Type) {
			if err := MatVec(rs.Gate, gl.FFNGate.Buf, input, gl.FFNGate.Rows, gl.FFNGate.Cols, gl.FFNGate.Type); err != nil {
				return err
			}
			if err := MatVec(rs.Up, gl.FFNUp.Buf, input, gl.FFNUp.Rows, gl.FFNUp.Cols, gl.FFNUp.Type); err != nil {
				return err
			}
		} else if err := gpuDualMatVec(rs.Gate, gl.FFNGate, layer.FFNGate, rs.Up, gl.FFNUp, layer.FFNUp, input, rs); err != nil {
			return err
		}
		Barrier()
		if err := GeGLU(rs.Hidden, rs.Gate, rs.Up, gl.FFNGate.Rows); err != nil {
			return err
		}
		Barrier()
		return gpuMatVec(rs.FFNOut, gl.FFNDown, layer.FFNDown, rs.Hidden, rs)

	case llm.FFNPlain:
		if err := gpuMatVec(rs.Up, gl.FFNUp, layer.FFNUp, input, rs); err != nil {
			return err
		}
		Barrier()
		if err := GELU(rs.Up, gl.FFNUp.Rows); err != nil {
			return err
		}
		Barrier()
		return gpuMatVec(rs.FFNOut, gl.FFNDown, layer.FFNDown, rs.Up, rs)
	}
	return nil
}

func supportsGPUQType(qtype uint32) bool {
	switch qtype {
	case 0, 1, 2, 6, 8, 11, 12, 13, 14:
		return true
	default:
		return false
	}
}

func ensureScratch(buf []float32, n int) []float32 {
	if cap(buf) < n {
		return make([]float32, n)
	}
	return buf[:n]
}

func gpuMatVec(out Buf, gpuW *GpuTensor, cpuW *core.QuantizedTensor, xBuf Buf, rs *GpuRunState) error {
	if gpuW == nil || cpuW == nil {
		return fmt.Errorf("missing tensor")
	}
	if supportsGPUQType(gpuW.Type) {
		return MatVec(out, gpuW.Buf, xBuf, gpuW.Rows, gpuW.Cols, gpuW.Type)
	}
	EndBatch()
	rs.ScratchIn = ensureScratch(rs.ScratchIn, cpuW.Cols)
	rs.ScratchOut = ensureScratch(rs.ScratchOut, cpuW.Rows)
	if err := DownloadF32(xBuf, rs.ScratchIn); err != nil {
		return err
	}
	blas.QMatVecMulParallel(rs.ScratchOut, cpuW, rs.ScratchIn, rs.Pool)
	if err := UploadF32(out, rs.ScratchOut); err != nil {
		return err
	}
	BeginBatch()
	return nil
}

func gpuDualMatVec(out1 Buf, gpuW1 *GpuTensor, cpuW1 *core.QuantizedTensor, out2 Buf, gpuW2 *GpuTensor, cpuW2 *core.QuantizedTensor, xBuf Buf, rs *GpuRunState) error {
	if gpuW1 == nil || gpuW2 == nil || cpuW1 == nil || cpuW2 == nil {
		return fmt.Errorf("missing tensor")
	}
	if supportsGPUQType(gpuW1.Type) && supportsGPUQType(gpuW2.Type) {
		if err := MatVec(out1, gpuW1.Buf, xBuf, gpuW1.Rows, gpuW1.Cols, gpuW1.Type); err != nil {
			return err
		}
		return MatVec(out2, gpuW2.Buf, xBuf, gpuW2.Rows, gpuW2.Cols, gpuW2.Type)
	}
	EndBatch()
	rs.ScratchIn = ensureScratch(rs.ScratchIn, cpuW1.Cols)
	rs.ScratchOut = ensureScratch(rs.ScratchOut, cpuW1.Rows)
	rs.ScratchAux = ensureScratch(rs.ScratchAux, cpuW2.Rows)
	if err := DownloadF32(xBuf, rs.ScratchIn); err != nil {
		return err
	}
	blas.QDualMatVecMulParallel(rs.ScratchOut, cpuW1, rs.ScratchAux, cpuW2, rs.ScratchIn, rs.Pool)
	if err := UploadF32(out1, rs.ScratchOut); err != nil {
		return err
	}
	if err := UploadF32(out2, rs.ScratchAux); err != nil {
		return err
	}
	BeginBatch()
	return nil
}
