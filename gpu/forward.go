//go:build cgo && vulkan

package gpu

import (
	"math"

	"github.com/computerex/dlgo/models/llm"
)

// GpuForward performs a single-token forward pass entirely on GPU.
// Only the final logits are downloaded to CPU for sampling.
func GpuForward(m *llm.Model, gm *GpuModel, token int32, pos int,
	kv *GpuKVCache, rs *GpuRunState, logitsBuf []float32) {
	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim

	// Embedding: dequantize row on CPU, upload to GPU
	// (Embedding lookup is cheap, not worth a GPU kernel)
	xCPU := make([]float32, dim)
	_ = m.TokenEmbed.DequantizeRow(int(token), xCPU)
	if cfg.EmbedScale != 0 {
		for i := range xCPU {
			xCPU[i] *= cfg.EmbedScale
		}
	}
	UploadF32(rs.X, xCPU)

	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		spec := &layer.Spec
		gl := &gm.Layers[l]

		// BATCH 1: pre-attention GPU ops (norm, Q/K/V projections, RoPE, KV store)
		BeginBatch()

		if spec.Norm == llm.NormRMS {
			RMSNorm(rs.XNorm, rs.X, gl.AttnNorm, dim, cfg.RMSNormEps)
		}

		if spec.Core == llm.CoreAttention {
			MatVec(rs.Q, gl.Wq.Buf, rs.XNorm, gl.Wq.Rows, gl.Wq.Cols, gl.Wq.Type)
			MatVec(rs.K, gl.Wk.Buf, rs.XNorm, gl.Wk.Rows, gl.Wk.Cols, gl.Wk.Type)
			MatVec(rs.V, gl.Wv.Buf, rs.XNorm, gl.Wv.Rows, gl.Wv.Cols, gl.Wv.Type)

			if gl.Bq != 0 {
				addBuf(rs.Q, gl.Bq, numHeads*headDim)
			}
			if gl.Bk != 0 {
				addBuf(rs.K, gl.Bk, kvDim)
			}
			if gl.Bv != 0 {
				addBuf(rs.V, gl.Bv, kvDim)
			}

			RoPE(rs.Q, rs.K, numHeads, numKVHeads, headDim, pos, cfg.RopeFreqBase)
			KVStore(kv.KeyBufs[l], kv.ValBufs[l], rs.K, rs.V, pos, kvDim)
		}

		EndBatch() // submit+wait so we can download for CPU attention

		// CPU attention fallback (needs Q and KV cache data on CPU)
		if spec.Core == llm.CoreAttention {
			seqLen := pos + 1
			scale := float32(1.0 / math.Sqrt(float64(headDim)))
			gpuAttentionCPUFallback(rs, kv, l, numHeads, numKVHeads, headDim, kvDim, seqLen, scale)
		}

		// BATCH 2: post-attention GPU ops (output proj, residual, FFN)
		BeginBatch()

		if spec.Core == llm.CoreAttention {
			MatVec(rs.AttnProj, gl.Wo.Buf, rs.AttnOut, gl.Wo.Rows, gl.Wo.Cols, gl.Wo.Type)
		}

		switch spec.Residual {
		case llm.ResStandard:
			if gl.PostAttnNorm != 0 {
				RMSNorm(rs.AttnProj, rs.AttnProj, gl.PostAttnNorm, dim, cfg.RMSNormEps)
			}
			Add(rs.FFNIn, rs.X, rs.AttnProj, dim)
			RMSNorm(rs.FFNNorm, rs.FFNIn, gl.FFNNorm, dim, cfg.RMSNormEps)
			gpuForwardFFN(layer, gl, rs, rs.FFNNorm, dim, cfg)
			if gl.PostFFNNorm != 0 {
				RMSNorm(rs.FFNOut, rs.FFNOut, gl.PostFFNNorm, dim, cfg.RMSNormEps)
			}
			Add(rs.X, rs.FFNIn, rs.FFNOut, dim)

		case llm.ResParallel:
			gpuForwardFFN(layer, gl, rs, rs.XNorm, dim, cfg)
			Add(rs.X, rs.X, rs.AttnProj, dim)
			Add(rs.X, rs.X, rs.FFNOut, dim)
		}

		EndBatch()
	}

	// Final: norm + logits
	BeginBatch()
	RMSNorm(rs.X, rs.X, gm.OutputNorm, dim, cfg.RMSNormEps)
	output := gm.Output
	if output == nil {
		output = gm.TokenEmbed
	}
	MatVec(rs.Logits, output.Buf, rs.X, output.Rows, output.Cols, output.Type)
	EndBatch()

	// Download logits to CPU for sampling
	DownloadF32(rs.Logits, logitsBuf)
}

func addBuf(dst, src Buf, n int) {
	Add(dst, dst, src, n)
}

func gpuForwardFFN(layer *llm.Layer, gl *GpuLayer, rs *GpuRunState, input Buf, dim int, cfg llm.ModelConfig) {
	switch layer.Spec.FFN {
	case llm.FFNSwiGLU:
		MatVec(rs.Gate, gl.FFNGate.Buf, input, gl.FFNGate.Rows, gl.FFNGate.Cols, gl.FFNGate.Type)
		MatVec(rs.Up, gl.FFNUp.Buf, input, gl.FFNUp.Rows, gl.FFNUp.Cols, gl.FFNUp.Type)
		SwiGLU(rs.Hidden, rs.Gate, rs.Up, gl.FFNGate.Rows)
		MatVec(rs.FFNOut, gl.FFNDown.Buf, rs.Hidden, gl.FFNDown.Rows, gl.FFNDown.Cols, gl.FFNDown.Type)

	case llm.FFNGeGLU:
		MatVec(rs.Gate, gl.FFNGate.Buf, input, gl.FFNGate.Rows, gl.FFNGate.Cols, gl.FFNGate.Type)
		MatVec(rs.Up, gl.FFNUp.Buf, input, gl.FFNUp.Rows, gl.FFNUp.Cols, gl.FFNUp.Type)
		GeGLU(rs.Hidden, rs.Gate, rs.Up, gl.FFNGate.Rows)
		MatVec(rs.FFNOut, gl.FFNDown.Buf, rs.Hidden, gl.FFNDown.Rows, gl.FFNDown.Cols, gl.FFNDown.Type)

	case llm.FFNPlain:
		MatVec(rs.Up, gl.FFNUp.Buf, input, gl.FFNUp.Rows, gl.FFNUp.Cols, gl.FFNUp.Type)
		GELU(rs.Up, gl.FFNUp.Rows)
		MatVec(rs.FFNOut, gl.FFNDown.Buf, rs.Up, gl.FFNDown.Rows, gl.FFNDown.Cols, gl.FFNDown.Type)
	}
}

// gpuAttentionCPUFallback: downloads Q and KV cache for attention, computes on CPU, uploads result.
// This is the initial fallback; will be replaced by a fused GPU attention kernel.
func gpuAttentionCPUFallback(rs *GpuRunState, kv *GpuKVCache, layer int,
	numHeads, numKVHeads, headDim, kvDim, seqLen int, scale float32) {
	qDim := numHeads * headDim
	kvMul := numHeads / numKVHeads

	// Download Q
	qCPU := make([]float32, qDim)
	DownloadF32(rs.Q, qCPU)

	// Download K/V cache up to seqLen
	kCPU := make([]float32, seqLen*kvDim)
	vCPU := make([]float32, seqLen*kvDim)
	DownloadF32(kv.KeyBufs[layer], kCPU)
	DownloadF32(kv.ValBufs[layer], vCPU)

	// Compute attention on CPU
	attnOut := make([]float32, qDim)
	for h := 0; h < numHeads; h++ {
		kvH := h / kvMul
		qHead := qCPU[h*headDim : (h+1)*headDim]
		outHead := attnOut[h*headDim : (h+1)*headDim]

		scores := make([]float32, seqLen)
		maxScore := float32(-1e30)
		for t := 0; t < seqLen; t++ {
			kHead := kCPU[t*kvDim+kvH*headDim : t*kvDim+(kvH+1)*headDim]
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qHead[d] * kHead[d]
			}
			scores[t] = dot * scale
			if scores[t] > maxScore {
				maxScore = scores[t]
			}
		}

		// Softmax
		var sum float32
		for t := range scores {
			scores[t] = float32(math.Exp(float64(scores[t] - maxScore)))
			sum += scores[t]
		}
		for t := range scores {
			scores[t] /= sum
		}

		// Weighted sum
		for t := 0; t < seqLen; t++ {
			vHead := vCPU[t*kvDim+kvH*headDim : t*kvDim+(kvH+1)*headDim]
			for d := 0; d < headDim; d++ {
				outHead[d] += scores[t] * vHead[d]
			}
		}
	}

	// Upload attention output back to GPU
	UploadF32(rs.AttnOut, attnOut)
}
