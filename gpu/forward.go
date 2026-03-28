//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math"
	"os"
	"sync"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
	"github.com/computerex/dlgo/quant"
)

var moeDebugOnce sync.Once

// BuildLayerConfs creates reusable fused-layer configurations from the model,
// run state, and KV cache. Call once after model upload; reuse for every token.
func BuildLayerConfs(m *llm.Model, gm *GpuModel, pipe *GpuPipeline, rs *GpuRunState, kv *GpuKVCache) []*LayerConf {
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

		if !gl.OnGPU {
			confs[l] = nil
			continue
		}

		lc := NewLayerConf()

		lc.SetScratch(rs.X, rs.XNorm, rs.Q, rs.K, rs.V, rs.AttnOut, rs.AttnProj,
			rs.FFNNorm, rs.FFNIn, rs.Gate, rs.Up, rs.Hidden, rs.FFNOut)

		if layer.Spec.Core == llm.CoreSSM || layer.Spec.GatedQ || layer.Spec.Core == llm.CoreMLA {
			lc.SetCoreType(1)
			lc.SetAttnNormOnly(gl.AttnNorm)
		} else {
		lc.SetAttn(gl.AttnNorm, gl.Wq, gl.Wk, gl.Wv, gl.Wo,
			gl.Bq, gl.Bk, gl.Bv, gl.Bo, gl.AttnQNorm, gl.AttnKNorm)
		lc.SetKV(kv.KeyBufs[l], kv.ValBufs[l])
		if gl.AttnSinks != 0 {
			lc.SetAttnSinks(gl.AttnSinks)
		}
		if layer.Spec.SlidingWindow > 0 {
			lc.SetSlidingWindow(layer.Spec.SlidingWindow)
		}
		if cfg.AttnLogitSoftcap > 0 {
			lc.SetAttnLogitSoftcap(cfg.AttnLogitSoftcap)
		}
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
	case llm.FFNMoE, llm.FFNMoESwiOAI:
		ffnType = 3
	}
	resType := 0
	if layer.Spec.Residual == llm.ResParallel {
		resType = 1
	}

	lc.SetConfig(dim, headDim, numHeads, numKVHeads, kvDim,
		cfg.RMSNormEps, cfg.RopeDim, cfg.RopeNeox,
		pipe.RoPECosTable, pipe.RoPESinTable,
		ffnType, resType)

	confs[l] = lc
	}
	return confs
}

// GpuForwardFused performs a single-token forward pass using pre-built layer
// configurations. One CGo call per layer instead of ~20+.
func GpuForwardFused(m *llm.Model, gm *GpuModel, pipe *GpuPipeline, token int32, pos int,
	kv *GpuKVCache, rs *GpuRunState, logitsBuf []float32, layerConfs []*LayerConf) {
	GpuForwardFusedSSM(m, gm, token, pos, kv, rs, logitsBuf, layerConfs, pipe)
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
		layerConfs = BuildLayerConfs(m, gm, pipe, rs, kv)
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
			RoPE(rs.Q, rs.K, pipe.RoPECosTable, pipe.RoPESinTable, numHeads, numKVHeads, headDim, cfg.RopeDim, pos, cfg.RopeNeox)
			KVStore(kv.KeyBufs[l], kv.ValBufs[l], rs.K, rs.V, pos, kvDim)
			Barrier()
			gqWinStart := 0
			if w := m.Layers[l].Spec.SlidingWindow; w > 0 && seqLen > w {
				gqWinStart = seqLen - w
			}
			Attention(rs.AttnOut, rs.Q, kv.KeyBufs[l], kv.ValBufs[l],
				numHeads, numKVHeads, headDim, kvDim, seqLen, gqWinStart, scale)
			Barrier()
			SigmoidGate(rs.AttnOut, rs.QGate, numHeads*headDim)
			Barrier()
			MatVec(rs.AttnProj, gl.Wo.Buf, rs.AttnOut, gl.Wo.Rows, gl.Wo.Cols, gl.Wo.Type)
		} else if layer.Spec.Core == llm.CoreMLA && pipe != nil && pipe.HasMLA {
			Sync()
			cpuRS := pipe.CPURunState
			DownloadF32(rs.X, cpuRS.X)
			ops.RMSNorm(cpuRS.XNorm, cpuRS.X, layer.AttnNorm, cfg.RMSNormEps)
			llm.ForwardMLA(layer, cpuRS, pipe.CPUKVCache, l, pos, cfg, cpuRS.Pool)
			BeginBatch()
			UploadF32(rs.AttnProj, cpuRS.AttnProj)
			Barrier()
		}

		var nextAttnNorm Buf
		if l < cfg.NumLayers-1 {
			nextAttnNorm = gm.Layers[l+1].AttnNorm
		}
		ForwardLayer(layerConfs[l], pos, seqLen, scale, nextAttnNorm)

	if gl.IsMoE && pipe != nil && pipe.HasMoE {
		if gl.MoEOnGPU {
			GpuForwardMoEFFN(gl, layer, rs, cfg, false, 0, 0)
			Barrier()
			if nextAttnNorm != 0 {
				AddRMSNorm(rs.XNorm, rs.X, rs.FFNIn, rs.FFNOut, nextAttnNorm, dim, cfg.RMSNormEps)
			} else {
				Add(rs.X, rs.FFNIn, rs.FFNOut, dim)
			}
		} else {
			Sync()
			cpuRS := pipe.CPURunState
			DownloadF32(rs.FFNIn, cpuRS.FFNIn)
			ops.RMSNorm(cpuRS.FFNNorm, cpuRS.FFNIn, layer.FFNNorm, cfg.RMSNormEps)
			llm.ForwardMoEFFNDispatch(layer, cpuRS, cpuRS.FFNNorm, cfg, cpuRS.Pool)
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
	if cfg.FinalLogitSoftcap > 0 {
		cap := float64(cfg.FinalLogitSoftcap)
		for i := range logitsBuf {
			logitsBuf[i] = float32(cap * math.Tanh(float64(logitsBuf[i])/cap))
		}
	}

}

// GpuForwardPartial performs a single-token forward pass where only the first
// NumGPULayers are on GPU. Remaining layers run on CPU. Handles the GPU->CPU
// activation transfer at the boundary.
func GpuForwardPartial(m *llm.Model, gm *GpuModel, token int32, pos int,
	kv *GpuKVCache, rs *GpuRunState, logitsBuf []float32, layerConfs []*LayerConf,
	pipe *GpuPipeline) {

	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim
	numGPU := pipe.NumGPULayers

	// --- Phase 1: GPU layers (same as fused path, but only up to numGPU) ---
	xCPU := make([]float32, dim)
	_ = m.TokenEmbed.DequantizeRow(int(token), xCPU)
	if cfg.EmbedScale != 0 {
		for i := range xCPU {
			xCPU[i] *= cfg.EmbedScale
		}
	}
	seqLen := pos + 1
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

	BeginBatch()
	UploadF32(rs.X, xCPU)

	if m.Layers[0].Spec.Norm == llm.NormRMS {
		Barrier()
		RMSNorm(rs.XNorm, rs.X, gm.Layers[0].AttnNorm, dim, cfg.RMSNormEps)
	}

	for l := 0; l < numGPU; l++ {
		layer := &m.Layers[l]
		gl := &gm.Layers[l]

		if layer.Spec.Core == llm.CoreSSM && pipe.HasSSM {
			Barrier()
			MatVec(rs.SSMQKV, gl.SSMInProj.Buf, rs.XNorm, gl.SSMInProj.Rows, gl.SSMInProj.Cols, gl.SSMInProj.Type)
			MatVec(rs.SSMZ, gl.SSMGate.Buf, rs.XNorm, gl.SSMGate.Rows, gl.SSMGate.Cols, gl.SSMGate.Type)
			MatVec(rs.SSMAlpha, gl.SSMAlpha.Buf, rs.XNorm, gl.SSMAlpha.Rows, gl.SSMAlpha.Cols, gl.SSMAlpha.Type)
			MatVec(rs.SSMBeta, gl.SSMBeta.Buf, rs.XNorm, gl.SSMBeta.Rows, gl.SSMBeta.Cols, gl.SSMBeta.Type)
			Barrier()
			SSMConv1dSiLU(rs.SSMQKV, gl.SSMConvBuf, gl.SSMConv1dW, ssmQKVDim, ssmConvK)
			Barrier()
			SSMPreprocess(rs.SSMAlpha, rs.SSMBeta, gl.SSMA, gl.SSMDtBias, rs.SSMQKV,
				ssmNumHeads, ssmHeadKDim, ssmKeyDim, cfg.RMSNormEps, gl.SSMDtBias != 0)
			Barrier()
			SSMDeltaRule(gl.SSMState, rs.SSMQKV, rs.SSMAlpha, rs.SSMBeta, rs.SSMY,
				ssmNumHeads, ssmHeadKDim, ssmHeadVDim, ssmKeyDim)
			Barrier()
			SSMNormGate(rs.SSMY, rs.SSMZ, gl.SSMNorm, ssmNumHeads, ssmHeadVDim, cfg.RMSNormEps)
			Barrier()
			MatVec(rs.AttnProj, gl.SSMOut.Buf, rs.SSMY, gl.SSMOut.Rows, gl.SSMOut.Cols, gl.SSMOut.Type)
		} else if layer.Spec.GatedQ && pipe.HasGatedQ {
			Barrier()
			MatVec(rs.QFull, gl.Wq.Buf, rs.XNorm, gl.Wq.Rows, gl.Wq.Cols, gl.Wq.Type)
			MatVec(rs.K, gl.Wk.Buf, rs.XNorm, gl.Wk.Rows, gl.Wk.Cols, gl.Wk.Type)
			MatVec(rs.V, gl.Wv.Buf, rs.XNorm, gl.Wv.Rows, gl.Wv.Cols, gl.Wv.Type)
			Barrier()
			DeinterleaveQGate(rs.QFull, rs.Q, rs.QGate, numHeads, headDim)
			if gl.Bq != 0 { Add(rs.Q, rs.Q, gl.Bq, numHeads*headDim) }
			if gl.Bk != 0 { Add(rs.K, rs.K, gl.Bk, kvDim) }
			if gl.Bv != 0 { Add(rs.V, rs.V, gl.Bv, kvDim) }
			if layer.Spec.QKNorm {
				Barrier()
				RMSNormHeads(rs.Q, gl.AttnQNorm, numHeads, headDim, cfg.RMSNormEps)
				RMSNormHeads(rs.K, gl.AttnKNorm, numKVHeads, headDim, cfg.RMSNormEps)
			}
			Barrier()
			RoPE(rs.Q, rs.K, pipe.RoPECosTable, pipe.RoPESinTable, numHeads, numKVHeads, headDim, cfg.RopeDim, pos, cfg.RopeNeox)
			KVStore(kv.KeyBufs[l], kv.ValBufs[l], rs.K, rs.V, pos, kvDim)
			Barrier()
			gqWinStart2 := 0
			if w := m.Layers[l].Spec.SlidingWindow; w > 0 && seqLen > w {
				gqWinStart2 = seqLen - w
			}
			Attention(rs.AttnOut, rs.Q, kv.KeyBufs[l], kv.ValBufs[l],
				numHeads, numKVHeads, headDim, kvDim, seqLen, gqWinStart2, scale)
			Barrier()
			SigmoidGate(rs.AttnOut, rs.QGate, numHeads*headDim)
			Barrier()
			MatVec(rs.AttnProj, gl.Wo.Buf, rs.AttnOut, gl.Wo.Rows, gl.Wo.Cols, gl.Wo.Type)
		} else if layer.Spec.Core == llm.CoreMLA && pipe != nil && pipe.HasMLA {
			Sync()
			cpuRS := pipe.CPURunState
			DownloadF32(rs.X, cpuRS.X)
			ops.RMSNorm(cpuRS.XNorm, cpuRS.X, layer.AttnNorm, cfg.RMSNormEps)
			llm.ForwardMLA(layer, cpuRS, pipe.CPUKVCache, l, pos, cfg, cpuRS.Pool)
			BeginBatch()
			UploadF32(rs.AttnProj, cpuRS.AttnProj)
			Barrier()
		}

		var nextAttnNorm Buf
		if l < numGPU-1 {
			nextAttnNorm = gm.Layers[l+1].AttnNorm
		}
		ForwardLayer(layerConfs[l], pos, seqLen, scale, nextAttnNorm)

		if gl.IsMoE && pipe.HasMoE {
			if gl.MoEOnGPU {
				GpuForwardMoEFFN(gl, layer, rs, cfg, false, 0, 0)
				Barrier()
				if nextAttnNorm != 0 {
					AddRMSNorm(rs.XNorm, rs.X, rs.FFNIn, rs.FFNOut, nextAttnNorm, dim, cfg.RMSNormEps)
				} else {
					Add(rs.X, rs.FFNIn, rs.FFNOut, dim)
				}
			} else {
				Sync()
				cpuRS := pipe.CPURunState
				DownloadF32(rs.FFNIn, cpuRS.FFNIn)
				ops.RMSNorm(cpuRS.FFNNorm, cpuRS.FFNIn, layer.FFNNorm, cfg.RMSNormEps)
				llm.ForwardMoEFFNDispatch(layer, cpuRS, cpuRS.FFNNorm, cfg, cpuRS.Pool)
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
	}

	// --- Phase 2: GPU -> CPU transition ---
	EndBatch()
	Sync()
	DownloadF32(rs.X, pipe.CPURunState.X)

	// --- Phase 3: CPU layers ---
	llm.ForwardFromLayer(m, numGPU, pos, pipe.CPUKVCache, pipe.CPURunState)

	// Logits are now in CPURunState.Logits — copy to output buffer
	copy(logitsBuf, pipe.CPURunState.Logits)
}

// BuildBatchLayerConfs creates layer configs that point to batch-sized scratch
// buffers while sharing the same weight/norm buffers as single-token configs.
func BuildBatchLayerConfs(m *llm.Model, gm *GpuModel, pipe *GpuPipeline, bs *GpuBatchState, kv *GpuKVCache) []*LayerConf {
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

		if !gl.OnGPU {
			confs[l] = nil
			continue
		}

		lc := NewLayerConf()

		lc.SetScratch(bs.X, bs.XNorm, bs.Q, bs.K, bs.V, bs.AttnOut, bs.AttnProj,
			bs.FFNNorm, bs.FFNIn, bs.Gate, bs.Up, bs.Hidden, bs.FFNOut)

		if layer.Spec.Core == llm.CoreSSM || layer.Spec.GatedQ || layer.Spec.Core == llm.CoreMLA {
			lc.SetCoreType(1)
			lc.SetAttnNormOnly(gl.AttnNorm)
		} else {
		lc.SetAttn(gl.AttnNorm, gl.Wq, gl.Wk, gl.Wv, gl.Wo,
			gl.Bq, gl.Bk, gl.Bv, gl.Bo, gl.AttnQNorm, gl.AttnKNorm)
		lc.SetKV(kv.KeyBufs[l], kv.ValBufs[l])
		if gl.AttnSinks != 0 {
			lc.SetAttnSinks(gl.AttnSinks)
		}
		if layer.Spec.SlidingWindow > 0 {
			lc.SetSlidingWindow(layer.Spec.SlidingWindow)
		}
		if cfg.AttnLogitSoftcap > 0 {
			lc.SetAttnLogitSoftcap(cfg.AttnLogitSoftcap)
		}
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
		case llm.FFNMoE, llm.FFNMoESwiOAI:
			ffnType = 3
		}
		resType := 0
		if layer.Spec.Residual == llm.ResParallel {
			resType = 1
		}

		lc.SetConfig(dim, headDim, numHeads, numKVHeads, kvDim,
			cfg.RMSNormEps, cfg.RopeDim, cfg.RopeNeox,
			pipe.RoPECosTable, pipe.RoPESinTable,
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
	batchLayerConfs []*LayerConf, pipe *GpuPipeline, startPos int, lastChunk bool) {

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
			BatchRoPE(bs.Q, bs.K, pipe.RoPECosTable, pipe.RoPESinTable, numHeads, numKVHeads, headDim, cfg.RopeDim, startPos,
				cfg.RopeNeox, npos)
			BatchKVStore(kv.KeyBufs[l], kv.ValBufs[l], bs.K, bs.V, startPos, kvDim, npos)
			BatchAttention(bs.AttnOut, bs.Q, kv.KeyBufs[l], kv.ValBufs[l],
				numHeads, numKVHeads, headDim, kvDim, startPos+1, scale, npos)
			Barrier()
			SigmoidGate(bs.AttnOut, bs.QGate, qDim*npos)
			Barrier()
			BatchMatVec(bs.AttnProj, gl.Wo.Buf, bs.AttnOut, gl.Wo.Rows, gl.Wo.Cols, npos, gl.Wo.Type)

		} else if layer.Spec.Core == llm.CoreMLA && pipe.HasMLA {
			// MLA: run per-position on CPU (sequential KV cache dependency)
			// Download raw X and recompute RMSNorm on CPU with f64 precision
			Sync()
			cpuRS := pipe.CPURunState
			xBatch := make([]float32, npos*dim)
			DownloadF32(bs.X, xBatch)
			attnProjBatch := make([]float32, npos*dim)
			for p := 0; p < npos; p++ {
				copy(cpuRS.X, xBatch[p*dim:(p+1)*dim])
				ops.RMSNorm(cpuRS.XNorm, cpuRS.X, layer.AttnNorm, cfg.RMSNormEps)
				llm.ForwardMLA(layer, cpuRS, pipe.CPUKVCache, l, p, cfg, cpuRS.Pool)
				copy(attnProjBatch[p*dim:(p+1)*dim], cpuRS.AttnProj)
			}
			BeginBatch()
			UploadF32(bs.AttnProj, attnProjBatch)
			Barrier()
		}

		// FFN + residual via ForwardLayerBatch
		var nextAttnNorm Buf
		if l < cfg.NumLayers-1 {
			nextAttnNorm = gm.Layers[l+1].AttnNorm
		}
		ForwardLayerBatch(batchLayerConfs[l], npos, startPos, scale, nextAttnNorm)

		if gl.IsMoE && pipe.HasMoE {
			// Batch prefill MoE: download raw FFNIn, recompute norm on CPU with f64 precision
			Sync()
			cpuRS := pipe.CPURunState
			ffnInBatch := make([]float32, npos*dim)
			DownloadF32(bs.FFNIn, ffnInBatch)
			ffnOutBatch := make([]float32, npos*dim)
			for p := 0; p < npos; p++ {
				posFFNIn := ffnInBatch[p*dim : (p+1)*dim]
				ops.RMSNorm(cpuRS.FFNNorm, posFFNIn, layer.FFNNorm, cfg.RMSNormEps)
				llm.ForwardMoEFFNDispatch(layer, cpuRS, cpuRS.FFNNorm, cfg, cpuRS.Pool)
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

	if lastChunk {
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

	if p != nil && p.AllCPUAttn {
		cpuRS := p.CPURunState
		llm.ForwardRange(m, token, pos, 0, cfg.NumLayers, p.CPUKVCache, cpuRS)
		copy(logitsBuf, cpuRS.Logits)
		return nil
	}

	GpuDiag.Init()
	diagOn := GpuDiag.Enabled

	BeginBatch()
	if err := UploadF32(rs.X, xCPU); err != nil {
		return err
	}

	if diagOn && GpuDiag.Active(-1, pos) {
		GpuDiag.LogEmbed("GPU", pos, xCPU)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		spec := &layer.Spec
		gl := &gm.Layers[l]
		diagL := diagOn && GpuDiag.Active(l, pos)

		Barrier()
		if spec.Norm == llm.NormRMS {
			if err := RMSNorm(rs.XNorm, rs.X, gl.AttnNorm, dim, cfg.RMSNormEps); err != nil {
				return fmt.Errorf("layer %d attn rmsnorm: %w", l, err)
			}
		} else if spec.Norm == llm.NormLayer {
			if err := LayerNorm(rs.XNorm, rs.X, gl.AttnNorm, gl.AttnNormBias, dim, cfg.RMSNormEps); err != nil {
				return fmt.Errorf("layer %d attn layernorm: %w", l, err)
			}
		}

		if diagL {
			GpuDiag.LogBuf("GPU", l, pos, "XNorm", rs.XNorm, dim)
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
			if diagL {
				GpuDiag.LogBuf("GPU", l, pos, "SSM AttnProj", rs.AttnProj, dim)
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
			if diagL {
				GpuDiag.LogBuf("GPU", l, pos, "GatedQ Q+bias", rs.Q, numHeads*headDim)
				GpuDiag.LogBuf("GPU", l, pos, "GatedQ K+bias", rs.K, kvDim)
			}
			if spec.QKNorm {
				Barrier()
				RMSNormHeads(rs.Q, gl.AttnQNorm, numHeads, headDim, cfg.RMSNormEps)
				RMSNormHeads(rs.K, gl.AttnKNorm, numKVHeads, headDim, cfg.RMSNormEps)
			}
			Barrier()
			RoPE(rs.Q, rs.K, p.RoPECosTable, p.RoPESinTable, numHeads, numKVHeads, headDim, cfg.RopeDim, pos, cfg.RopeNeox)
			if diagL {
				GpuDiag.LogBuf("GPU", l, pos, "GatedQ Q post-RoPE", rs.Q, numHeads*headDim)
				GpuDiag.LogBuf("GPU", l, pos, "GatedQ K post-RoPE", rs.K, kvDim)
			}
			KVStore(kv.KeyBufs[l], kv.ValBufs[l], rs.K, rs.V, pos, kvDim)
			Barrier()
			gqWinStart3 := 0
			if w := m.Layers[l].Spec.SlidingWindow; w > 0 && seqLen > w {
				gqWinStart3 = seqLen - w
			}
			Attention(rs.AttnOut, rs.Q, kv.KeyBufs[l], kv.ValBufs[l],
				numHeads, numKVHeads, headDim, kvDim, seqLen, gqWinStart3, scale)
			if diagL {
				GpuDiag.LogBuf("GPU", l, pos, "GatedQ AttnOut", rs.AttnOut, numHeads*headDim)
			}
			Barrier()
			SigmoidGate(rs.AttnOut, rs.QGate, numHeads*headDim)
			Barrier()
			if err := gpuMatVec(rs.AttnProj, gl.Wo, layer.Wo, rs.AttnOut, rs); err != nil {
				return fmt.Errorf("layer %d wo: %w", l, err)
			}
			if gl.Bo != 0 {
				Barrier()
				if err := addBuf(rs.AttnProj, gl.Bo, dim); err != nil {
					return fmt.Errorf("layer %d bo: %w", l, err)
				}
			}
			if diagL {
				GpuDiag.LogBuf("GPU", l, pos, "GatedQ AttnProj", rs.AttnProj, dim)
			}
		} else if spec.Core == llm.CoreMLA && p != nil && p.HasMLA {
			Sync()
			cpuRS := p.CPURunState
			if err := DownloadF32(rs.X, cpuRS.X); err != nil {
				return fmt.Errorf("layer %d mla download x: %w", l, err)
			}
			ops.RMSNorm(cpuRS.XNorm, cpuRS.X, layer.AttnNorm, cfg.RMSNormEps)
			llm.ForwardMLA(layer, cpuRS, p.CPUKVCache, l, pos, cfg, cpuRS.Pool)
			BeginBatch()
			if err := UploadF32(rs.AttnProj, cpuRS.AttnProj); err != nil {
				return fmt.Errorf("layer %d mla upload attn_proj: %w", l, err)
			}
			Barrier()
		} else if spec.Core == llm.CoreAttention {
			if gl.CPUAttn && p != nil {
				EndBatch()
				Sync()
				cpuRS := p.CPURunState
				if err := DownloadF32(rs.X, cpuRS.X); err != nil {
					return fmt.Errorf("layer %d cpu fallback download: %w", l, err)
				}
				llm.ForwardRange(m, token, pos, l, l+1, p.CPUKVCache, cpuRS)
				if l+1 == cfg.NumLayers {
					copy(logitsBuf, cpuRS.Logits)
					return nil
				}
				BeginBatch()
				if err := UploadF32(rs.X, cpuRS.X); err != nil {
					return fmt.Errorf("layer %d cpu fallback upload: %w", l, err)
				}
				Barrier()
				continue
			}
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

			if diagL {
				GpuDiag.LogBuf("GPU", l, pos, "Q+bias", rs.Q, numHeads*headDim)
				GpuDiag.LogBuf("GPU", l, pos, "K+bias", rs.K, kvDim)
				GpuDiag.LogBuf("GPU", l, pos, "V+bias", rs.V, kvDim)
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
			if err := RoPE(rs.Q, rs.K, p.RoPECosTable, p.RoPESinTable, numHeads, numKVHeads, headDim, cfg.RopeDim, pos, cfg.RopeNeox); err != nil {
				return fmt.Errorf("layer %d rope: %w", l, err)
			}

			if diagL {
				GpuDiag.LogBuf("GPU", l, pos, "Q post-RoPE", rs.Q, numHeads*headDim)
				GpuDiag.LogBuf("GPU", l, pos, "K post-RoPE", rs.K, kvDim)
			}

			if err := KVStore(kv.KeyBufs[l], kv.ValBufs[l], rs.K, rs.V, pos, kvDim); err != nil {
				return fmt.Errorf("layer %d kvstore: %w", l, err)
			}

			Barrier()
			winStart := 0
			if w := m.Layers[l].Spec.SlidingWindow; w > 0 && seqLen > w {
				winStart = seqLen - w
			}
			if err := Attention(rs.AttnOut, rs.Q, kv.KeyBufs[l], kv.ValBufs[l],
				numHeads, numKVHeads, headDim, kvDim, seqLen, winStart, scale); err != nil {
				return fmt.Errorf("layer %d attention: %w", l, err)
			}

			if diagL {
				GpuDiag.LogBuf("GPU", l, pos, "AttnOut", rs.AttnOut, numHeads*headDim)
			}

			Barrier()
			if err := gpuMatVec(rs.AttnProj, gl.Wo, layer.Wo, rs.AttnOut, rs); err != nil {
				return fmt.Errorf("layer %d wo: %w", l, err)
			}
			if gl.Bo != 0 {
				Barrier()
				if err := addBuf(rs.AttnProj, gl.Bo, dim); err != nil {
					return fmt.Errorf("layer %d bo: %w", l, err)
				}
			}

			if diagL {
				GpuDiag.LogBuf("GPU", l, pos, "AttnProj", rs.AttnProj, dim)
			}
		}

		if gl.IsMoE && p != nil && p.HasMoE {
			ffnNormW := gl.FFNNorm
			if spec.Residual == llm.ResPostAttnFFN {
				ffnNormW = gl.PostAttnNorm
			}
			Barrier()
			if err := AddRMSNorm(rs.FFNNorm, rs.FFNIn, rs.X, rs.AttnProj, ffnNormW, dim, cfg.RMSNormEps); err != nil {
				return fmt.Errorf("layer %d moe add+rmsnorm: %w", l, err)
			}
			if diagL {
				GpuDiag.LogBuf("GPU", l, pos, "FFNNorm(MoE)", rs.FFNNorm, dim)
				GpuDiag.LogBuf("GPU", l, pos, "FFNIn(MoE)", rs.FFNIn, dim)
			}
			if gl.MoEOnGPU {
				if err := GpuForwardMoEFFN(gl, layer, rs, cfg, diagL, l, pos); err != nil {
					return fmt.Errorf("layer %d gpu moe: %w", l, err)
				}
				if diagL {
					GpuDiag.LogBuf("GPU", l, pos, "FFNOut(MoE)", rs.FFNOut, dim)
				}
				Barrier()
				if err := Add(rs.X, rs.FFNIn, rs.FFNOut, dim); err != nil {
					return fmt.Errorf("layer %d moe residual: %w", l, err)
				}
			} else {
				EndBatch()
				cpuRS := p.CPURunState
				if err := DownloadF32(rs.FFNIn, cpuRS.FFNIn); err != nil {
					return fmt.Errorf("layer %d moe download ffnin: %w", l, err)
				}
				cpuFFNNormW := layer.FFNNorm
				if spec.Residual == llm.ResPostAttnFFN {
					cpuFFNNormW = layer.PostAttnNorm
				}
				ops.RMSNorm(cpuRS.FFNNorm, cpuRS.FFNIn, cpuFFNNormW, cfg.RMSNormEps)
				llm.ForwardMoEFFNDispatch(layer, cpuRS, cpuRS.FFNNorm, cfg, cpuRS.Pool)
				BeginBatch()
				if err := UploadF32(rs.FFNOut, cpuRS.FFNOut); err != nil {
					return fmt.Errorf("layer %d moe upload: %w", l, err)
				}
				if diagL {
					GpuDiag.LogSlice("GPU", l, pos, "FFNOut(CPU-MoE)", cpuRS.FFNOut[:dim])
				}
				Barrier()
				if err := Add(rs.X, rs.FFNIn, rs.FFNOut, dim); err != nil {
					return fmt.Errorf("layer %d moe residual: %w", l, err)
				}
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

		if diagL {
			GpuDiag.LogBuf("GPU", l, pos, "X(end-of-layer)", rs.X, dim)
		}
	}

	Barrier()
	if gm.OutputNormBias != 0 {
		if err := LayerNorm(rs.X, rs.X, gm.OutputNorm, gm.OutputNormBias, dim, cfg.RMSNormEps); err != nil {
			return fmt.Errorf("output layernorm: %w", err)
		}
	} else {
		if err := RMSNorm(rs.X, rs.X, gm.OutputNorm, dim, cfg.RMSNormEps); err != nil {
			return fmt.Errorf("output norm: %w", err)
		}
	}

	if diagOn && GpuDiag.Active(-1, pos) {
		GpuDiag.LogBuf("GPU", -1, pos, "X(final-norm)", rs.X, dim)
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
	if cfg.FinalLogitSoftcap > 0 {
		cap := float64(cfg.FinalLogitSoftcap)
		for i := range logitsBuf {
			logitsBuf[i] = float32(cap * math.Tanh(float64(logitsBuf[i])/cap))
		}
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
		if gl.FFNUpBias != 0 {
			Barrier()
			if err := AddOffset(rs.Up, gl.FFNUpBias, gl.FFNUp.Rows, 0); err != nil {
				return err
			}
		}
		Barrier()
		if err := GELU(rs.Up, gl.FFNUp.Rows); err != nil {
			return err
		}
		Barrier()
		if err := gpuMatVec(rs.FFNOut, gl.FFNDown, layer.FFNDown, rs.Up, rs); err != nil {
			return err
		}
		if gl.FFNDownBias != 0 {
			Barrier()
			if err := AddOffset(rs.FFNOut, gl.FFNDownBias, gl.FFNDown.Rows, 0); err != nil {
				return err
			}
		}
		return nil
	}
	return nil
}

func supportsGPUQType(qtype uint32) bool {
	switch qtype {
	case 0, 1, 2, 6, 8, 11, 12, 13, 14: // F32, F16, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K
		return true
	case 10, 16, 18, 19, 21, 22, 23, 29, 34: // Q2_K, IQ2_XXS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, TQ1_0
		return true
	case 20: // IQ4_NL
		return true
	case 39: // MXFP4
		return true
	default:
		return false
	}
}


func hasDp4aQType(qtype uint32) bool {
	switch qtype {
	case 2, 6, 8, 10, 39: // Q4_0, Q5_0, Q8_0, Q3_K, MXFP4
		return true
	default:
		return false
	}
}

// moeBlockStride returns the expert stride in block units for MoE dp4a shaders.
// The stride = rows * blocks_per_row = rows * (cols / block_size).
func moeBlockStride(rows, cols int, qtype uint32) int {
	bs := quantBlockSize(qtype)
	if bs == 0 {
		return 0
	}
	return rows * (cols / bs)
}

func quantBlockSize(qtype uint32) int {
	switch qtype {
	case 2, 6, 8, 20, 39: // Q4_0, Q5_0, Q8_0, IQ4_NL, MXFP4
		return 32
	case 10, 12, 13, 14: // Q3_K, Q4_K, Q5_K, Q6_K
		return 256
	default:
		return 0
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

// GpuForwardMoEFFN runs the Mixture-of-Experts FFN entirely on GPU.
// Router logits are downloaded for top-K selection, then expert projections
// run on GPU using offset matmuls into packed weight tensors.
func GpuForwardMoEFFN(gl *GpuLayer, layer *llm.Layer, rs *GpuRunState, cfg llm.ModelConfig, diagL bool, diagLayer, diagPos int) error {
	dim := cfg.EmbeddingDim
	expDim := cfg.ExpertFFNDim
	nUsed := cfg.ExpertUsedCount

	// Allocate MoE scratch buffers on first use (per-expert for batched dispatch)
	if rs.MoELogits == 0 {
		a := allocChecker{}
		rs.MoELogits = a.alloc(uint64(cfg.ExpertCount * 4))
		rs.MoETopKIdx = a.alloc(uint64(nUsed * 4))
		rs.MoETopKW = a.alloc(uint64(nUsed * 4))
		rs.MoEGates = make([]Buf, nUsed)
		rs.MoEUps = make([]Buf, nUsed)
		rs.MoEHiddens = make([]Buf, nUsed)
		rs.MoEOuts = make([]Buf, nUsed)
		for e := 0; e < nUsed; e++ {
			rs.MoEGates[e] = a.alloc(uint64(expDim * 4))
			rs.MoEUps[e] = a.alloc(uint64(expDim * 4))
			rs.MoEHiddens[e] = a.alloc(uint64(expDim * 4))
			rs.MoEOuts[e] = a.alloc(uint64(dim * 4))
		}
		shDim := cfg.SharedExpertFFNDim
		if shDim == 0 {
			shDim = expDim
		}
		rs.MoEShGate = a.alloc(uint64(shDim * 4))
		rs.MoEShUp = a.alloc(uint64(shDim * 4))
		rs.MoEShHidden = a.alloc(uint64(shDim * 4))
		rs.MoEShOut = a.alloc(uint64(dim * 4))

		if rs.MoEUseDp4a {
			q8BlocksDim := (dim + 31) / 32
			rs.MoEQ8_1Scratch = a.alloc(uint64(q8BlocksDim) * 36)

			q8BlocksExp := (expDim + 31) / 32
			rs.MoEQ8_1DownPacked = a.alloc(uint64(nUsed) * uint64(q8BlocksExp) * 36)
			rs.MoEQ8_1DownBufs = make([]Buf, nUsed)
			for e := 0; e < nUsed; e++ {
				rs.MoEQ8_1DownBufs[e] = a.alloc(uint64(q8BlocksExp) * 36)
			}

			rs.MoEGateScratch = a.alloc(uint64(nUsed * expDim * 4))
			rs.MoEUpScratch = a.alloc(uint64(nUsed * expDim * 4))
			rs.MoEHiddenScratch = a.alloc(uint64(nUsed * expDim * 4))
			rs.MoEOutScratch = a.alloc(uint64(nUsed * dim * 4))
			rs.MoEWeightsBuf = a.alloc(uint64(nUsed * 4))
		}
		if a.err != nil {
			return fmt.Errorf("gpu: MoE scratch alloc: %w", a.err)
		}
	}

	fused := gl.FFNGateUpExps != nil
	var gateGpu, upGpu *GpuTensor
	if fused {
		gateGpu = gl.FFNGateUpExps
		upGpu = gl.FFNGateUpExps
	} else {
		gateGpu = gl.FFNGateExps
		upGpu = gl.FFNUpExps
	}
	bpr := quant.BytesForN(gateGpu.Type, gateGpu.Cols)
	downBpr := quant.BytesForN(gl.FFNDownExps.Type, gl.FFNDownExps.Cols)

	hasBias := gl.FFNGateExpsBias != 0 && gl.FFNUpExpsBias != 0
	isOAI := layer.Spec.FFN == llm.FFNMoESwiOAI

	useFusedDp4a := rs.MoEUseDp4a && rs.MoEQ8_1Scratch != 0 &&
		rs.MoEGateScratch != 0 &&
		hasDp4aQType(gateGpu.Type) && hasDp4aQType(uint32(gl.FFNDownExps.Type)) &&
		!diagL

	moeDebugOnce.Do(func() {
		fmt.Printf("[dlgo/gpu] MoE dp4a check: useFused=%v dp4a=%v q8=%v gate=%v gateT=%v downT=%v hasBias=%v diag=%v fused=%v isOAI=%v weightsNorm=%v weightsScale=%.2f gatingFunc=%d gateType=%d upType=%d downType=%d gateCols=%d dim=%d expDim=%d nExperts=%d nUsed=%d\n",
			useFusedDp4a, rs.MoEUseDp4a, rs.MoEQ8_1Scratch != 0, rs.MoEGateScratch != 0,
			hasDp4aQType(gateGpu.Type), hasDp4aQType(uint32(gl.FFNDownExps.Type)),
			hasBias, diagL, fused, isOAI, cfg.ExpertWeightsNorm, cfg.ExpertWeightsScale, cfg.ExpertGatingFunc,
			gateGpu.Type, upGpu.Type, gl.FFNDownExps.Type, gateGpu.Cols, dim, expDim, cfg.ExpertCount, nUsed)
	})
	if os.Getenv("DLGO_NO_FUSED_MOE") == "1" {
		useFusedDp4a = false
	}
	if useFusedDp4a {
		return gpuForwardMoEFFNFused(gl, layer, rs, cfg, dim, expDim, nUsed,
			gateGpu, upGpu, bpr, downBpr, fused, isOAI)
	}

	// Fallback: per-expert dispatch with Sync for top-K indices
	// 1. Router + top-K entirely on GPU
	Barrier()
	MatVec(rs.MoELogits, gl.FFNRouter.Buf, rs.FFNNorm,
		gl.FFNRouter.Rows, gl.FFNRouter.Cols, gl.FFNRouter.Type)
	if gl.FFNRouterBias != 0 {
		Barrier()
		Add(rs.MoELogits, rs.MoELogits, gl.FFNRouterBias, cfg.ExpertCount)
	}
	Barrier()
	MoETopK(rs.MoELogits, rs.MoETopKIdx, rs.MoETopKW,
		cfg.ExpertCount, nUsed, cfg.ExpertGatingFunc, false, 0)

	Sync()
	idxBuf := make([]float32, nUsed)
	weights := make([]float32, nUsed)
	DownloadF32(rs.MoETopKIdx, idxBuf)
	DownloadF32(rs.MoETopKW, weights)

	indices := make([]int, nUsed)
	for i := range idxBuf {
		indices[i] = int(idxBuf[i])
	}

	if diagL {
		routerLogits := make([]float32, cfg.ExpertCount)
		DownloadF32(rs.MoELogits, routerLogits)
		GpuDiag.LogMoE("GPU", diagLayer, diagPos, routerLogits, indices, weights, cfg.ExpertGatingFunc)
	}

	if cfg.ExpertWeightsNorm {
		var wSum float32
		for _, w := range weights {
			wSum += w
		}
		if wSum < 1e-12 {
			wSum = 1e-12
		}
		invSum := 1.0 / wSum
		for i := range weights {
			weights[i] *= invSum
		}
	}
	if cfg.ExpertWeightsScale > 0 && cfg.ExpertWeightsScale != 1.0 {
		for i := range weights {
			weights[i] *= cfg.ExpertWeightsScale
		}
	}

	useDp4a := rs.MoEUseDp4a && rs.MoEQ8_1Scratch != 0 &&
		hasDp4aQType(gateGpu.Type) && hasDp4aQType(uint32(gl.FFNDownExps.Type))

	BeginBatch()
	ZeroFill(rs.FFNOut, uint64(dim*4))

	if useDp4a {
		QuantizeQ8_1(rs.MoEQ8_1Scratch, rs.FFNNorm, dim)
	}
	Barrier()

	for e := 0; e < nUsed; e++ {
		idx := indices[e]
		if idx < 0 {
			continue
		}
		var gateOff, upOff int
		if fused {
			gateOff = idx * 2 * expDim * bpr
			upOff = (idx*2 + 1) * expDim * bpr
		} else {
			gateOff = idx * expDim * bpr
			upOff = idx * expDim * bpr
		}
		if useDp4a {
			MatVecOffsetDp4a(rs.MoEGates[e], 0, gateGpu.Buf, gateOff, rs.MoEQ8_1Scratch,
				expDim, gateGpu.Cols, gateGpu.Type)
			MatVecOffsetDp4a(rs.MoEUps[e], 0, upGpu.Buf, upOff, rs.MoEQ8_1Scratch,
				expDim, upGpu.Cols, upGpu.Type)
		} else {
			MatVecOffset(rs.MoEGates[e], 0, gateGpu.Buf, gateOff, rs.FFNNorm,
				expDim, gateGpu.Cols, gateGpu.Type)
			MatVecOffset(rs.MoEUps[e], 0, upGpu.Buf, upOff, rs.FFNNorm,
				expDim, upGpu.Cols, upGpu.Type)
		}
	}

	Barrier()
	for e := 0; e < nUsed; e++ {
		idx := indices[e]
		if idx < 0 {
			continue
		}
		if isOAI && hasBias {
			SwiGLU_OAI_Bias(rs.MoEHiddens[e], rs.MoEGates[e], rs.MoEUps[e],
				gl.FFNGateExpsBias, gl.FFNUpExpsBias,
				expDim, 1.702, 7.0, idx*expDim, idx*expDim)
		} else if isOAI {
			SwiGLU_OAI(rs.MoEHiddens[e], rs.MoEGates[e], rs.MoEUps[e], expDim, 1.702, 7.0)
		} else {
			SwiGLU(rs.MoEHiddens[e], rs.MoEGates[e], rs.MoEUps[e], expDim)
		}
	}

	Barrier()
	for e := 0; e < nUsed; e++ {
		idx := indices[e]
		if idx < 0 {
			continue
		}
		downOff := idx * dim * downBpr
		if useDp4a {
			QuantizeQ8_1(rs.MoEQ8_1DownBufs[e], rs.MoEHiddens[e], expDim)
			Barrier()
			MatVecOffsetDp4a(rs.MoEOuts[e], 0, gl.FFNDownExps.Buf, downOff, rs.MoEQ8_1DownBufs[e],
				dim, gl.FFNDownExps.Cols, uint32(gl.FFNDownExps.Type))
			Barrier()
		} else {
			MatVecOffset(rs.MoEOuts[e], 0, gl.FFNDownExps.Buf, downOff, rs.MoEHiddens[e],
				dim, gl.FFNDownExps.Cols, gl.FFNDownExps.Type)
		}
	}

	if gl.FFNDownExpsBias != 0 {
		Barrier()
		for e := 0; e < nUsed; e++ {
			idx := indices[e]
			if idx < 0 {
				continue
			}
			AddOffset(rs.MoEOuts[e], gl.FFNDownExpsBias, dim, idx*dim)
		}
	}

	for e := 0; e < nUsed; e++ {
		if indices[e] < 0 {
			continue
		}
		Barrier()
		ScaleAdd(rs.FFNOut, rs.MoEOuts[e], dim, weights[e])
	}

	// 3. Shared expert (if present)
	if gl.FFNGateShared != nil {
		Barrier()
		MatVec(rs.MoEShGate, gl.FFNGateShared.Buf, rs.FFNNorm,
			gl.FFNGateShared.Rows, gl.FFNGateShared.Cols, gl.FFNGateShared.Type)
		MatVec(rs.MoEShUp, gl.FFNUpShared.Buf, rs.FFNNorm,
			gl.FFNUpShared.Rows, gl.FFNUpShared.Cols, gl.FFNUpShared.Type)
		Barrier()
		shDim := gl.FFNGateShared.Rows
		SwiGLU(rs.MoEShHidden, rs.MoEShGate, rs.MoEShUp, shDim)
		Barrier()
		MatVec(rs.MoEShOut, gl.FFNDownShared.Buf, rs.MoEShHidden,
			gl.FFNDownShared.Rows, gl.FFNDownShared.Cols, gl.FFNDownShared.Type)

		if gl.FFNRouterShared != 0 {
			Sync()
			shInput := make([]float32, dim)
			DownloadF32(rs.FFNNorm, shInput)
			shGateW := make([]float32, dim)
			DownloadF32(gl.FFNRouterShared, shGateW)
			var dot float32
			for i := 0; i < dim; i++ {
				dot += shGateW[i] * shInput[i]
			}
			gate := float32(1.0 / (1.0 + math.Exp(-float64(dot))))
			BeginBatch()
			Scale(rs.MoEShOut, gate, dim)
		}
		Barrier()
		Add(rs.FFNOut, rs.FFNOut, rs.MoEShOut, dim)
	}

	return nil
}

// gpuForwardMoEFFNFused runs the MoE FFN entirely on GPU using dp4a MoE shader variants.
// No Sync() or CPU round-trip for expert indices — everything stays on GPU.
// Uses C-side fused dispatch (single CGo call) to eliminate per-step CGo overhead.
func gpuForwardMoEFFNFused(gl *GpuLayer, layer *llm.Layer, rs *GpuRunState, cfg llm.ModelConfig,
	dim, expDim, nUsed int, gateGpu, upGpu *GpuTensor, bpr, downBpr int, fused, isOAI bool) error {

	gateStride := moeBlockStride(expDim, gateGpu.Cols, gateGpu.Type)
	downStride := moeBlockStride(dim, gl.FFNDownExps.Cols, uint32(gl.FFNDownExps.Type))

	gateBaseOff := 0
	upBaseOff := 0
	upStride := gateStride
	if fused {
		gateStride = 2 * gateStride
		upStride = gateStride
		upBaseOff = moeBlockStride(expDim, gateGpu.Cols, gateGpu.Type)
	}

	mc := NewMoEFFNConf()
	mc.SetScratch(rs.FFNNorm, rs.FFNOut, rs.MoELogits, rs.MoETopKIdx, rs.MoETopKW,
		rs.MoEQ8_1Scratch, rs.MoEGateScratch, rs.MoEUpScratch, rs.MoEQ8_1DownPacked, rs.MoEOutScratch)
	mc.SetRouter(gl.FFNRouter.Buf, gl.FFNRouter.Rows, gl.FFNRouter.Cols, int(gl.FFNRouter.Type), gl.FFNRouterBias)
	mc.SetExperts(gateGpu.Buf, int(gateGpu.Type), gateStride, gateBaseOff,
		upGpu.Buf, int(upGpu.Type), upStride, upBaseOff,
		gl.FFNDownExps.Buf, int(gl.FFNDownExps.Type), downStride)
	mc.SetBiases(gl.FFNGateExpsBias, gl.FFNUpExpsBias, gl.FFNDownExpsBias)
	mc.SetConfig(dim, expDim, cfg.ExpertCount, nUsed, cfg.ExpertGatingFunc,
		cfg.ExpertWeightsNorm, float32(cfg.ExpertWeightsScale),
		isOAI, 1.702, 7.0)

	if err := ForwardMoEFFN_C(mc); err != nil {
		return err
	}

	// 8. Shared expert (if present)
	if gl.FFNGateShared != nil {
		Barrier()
		MatVec(rs.MoEShGate, gl.FFNGateShared.Buf, rs.FFNNorm,
			gl.FFNGateShared.Rows, gl.FFNGateShared.Cols, gl.FFNGateShared.Type)
		MatVec(rs.MoEShUp, gl.FFNUpShared.Buf, rs.FFNNorm,
			gl.FFNUpShared.Rows, gl.FFNUpShared.Cols, gl.FFNUpShared.Type)
		Barrier()
		shDim := gl.FFNGateShared.Rows
		SwiGLU(rs.MoEShHidden, rs.MoEShGate, rs.MoEShUp, shDim)
		Barrier()
		MatVec(rs.MoEShOut, gl.FFNDownShared.Buf, rs.MoEShHidden,
			gl.FFNDownShared.Rows, gl.FFNDownShared.Cols, gl.FFNDownShared.Type)

		if gl.FFNRouterShared != 0 {
			Sync()
			shInput := make([]float32, dim)
			DownloadF32(rs.FFNNorm, shInput)
			shGateW := make([]float32, dim)
			DownloadF32(gl.FFNRouterShared, shGateW)
			var dot float32
			for i := 0; i < dim; i++ {
				dot += shGateW[i] * shInput[i]
			}
			gate := float32(1.0 / (1.0 + math.Exp(-float64(dot))))
			BeginBatch()
			Scale(rs.MoEShOut, gate, dim)
		}
		Barrier()
		Add(rs.FFNOut, rs.FFNOut, rs.MoEShOut, dim)
	}

	return nil
}

// topKIndices returns the indices and values of the top-K elements.
func topKIndices(logits []float32, k int) ([]int, []float32) {
	indices := make([]int, k)
	weights := make([]float32, k)
	for i := range indices {
		indices[i] = -1
	}
	for i, v := range logits {
		minIdx := 0
		for j := 1; j < k; j++ {
			if weights[j] < weights[minIdx] {
				minIdx = j
			}
		}
		if v > weights[minIdx] || indices[minIdx] < 0 {
			indices[minIdx] = i
			weights[minIdx] = v
		}
	}
	return indices, weights
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

