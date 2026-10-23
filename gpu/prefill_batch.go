//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math"

	"github.com/computerex/dlgo/models/llm"
)

// Chunked prefill for hybrid short-convolution + MoE models (LFM2/LFM2MoE).
//
// These models cannot use gpuForwardPrefillBatchHybrid: that function batches
// attention but runs the MoE per position on the CPU, which is impossible once
// FreeForGPU has released the CPU state — and it has no short-convolution core at
// all. So this path does every step on the GPU in Go, one chunk of tokens at a
// time, with the batched MoE (gpuForwardMoEBatch) supplying the piece the fused C
// layer dispatcher lacks.
//
// Why it matters: the per-token path is latency-bound, not bandwidth-bound (see
// gpu/optime.go — ~180 GB/s of 672 GB/s at ~100 W of a 220 W cap). Batching
// multiplies the work behind each dispatch by npos and divides the weight traffic
// by the same factor, which is where the prefill win comes from.
//
// shortConvBatchSupported reports whether every layer can run here.
func shortConvBatchSupported(m *llm.Model, gm *GpuModel, cfg llm.ModelConfig) bool {
	if !cfg.HasShortConv() {
		return false
	}
	for l := range m.Layers {
		gl := &gm.Layers[l]
		spec := m.Layers[l].Spec
		switch spec.Core {
		case llm.CoreShortConv:
			if gl.ShortConvInProj == nil || gl.ShortConvOutProj == nil || gl.ShortConvConv == 0 {
				return false
			}
		case llm.CoreAttention:
			if gl.Wq == nil || gl.Wk == nil || gl.Wv == nil || gl.Wo == nil {
				return false
			}
			if gl.CPUAttn || gl.AttnSinks != 0 || spec.GatedQ {
				return false
			}
		default:
			return false
		}
		if spec.Norm != llm.NormRMS {
			return false
		}
		switch m.Layers[l].Spec.FFN {
		case llm.FFNMoE:
			if !moeBatchSupported(&m.Layers[l], gl, cfg) {
				return false
			}
		case llm.FFNSwiGLU, llm.FFNGeGLU:
			if gl.FFNGate == nil || gl.FFNUp == nil || gl.FFNDown == nil {
				return false
			}
		default:
			return false
		}
	}
	return true
}

// GpuForwardPrefillBatchShortConv runs one chunk of prompt tokens through the
// whole model with the token axis batched. Consumes/produces the same KV and
// run state as the per-token path, so decode can continue afterwards.
func GpuForwardPrefillBatchShortConv(m *llm.Model, gm *GpuModel, tokens []int32,
	kv *GpuKVCache, rs *GpuRunState, bs *GpuBatchState, logitsBuf []float32,
	pipe *GpuPipeline, startPos int, lastChunk bool) error {

	npos := len(tokens)
	if npos <= 0 {
		return nil
	}
	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim
	qDim := numHeads * headDim
	eps := cfg.RMSNormEps
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	BeginBatch()

	// Embed the chunk.
	xBatch := make([]float32, npos*dim)
	for i, tok := range tokens {
		if err := m.TokenEmbed.DequantizeRow(int(tok), xBatch[i*dim:(i+1)*dim]); err != nil {
			return fmt.Errorf("prefill batch embed: %w", err)
		}
		if cfg.EmbedScale != 0 {
			for j := 0; j < dim; j++ {
				xBatch[i*dim+j] *= cfg.EmbedScale
			}
		}
	}
	if err := UploadF32(bs.X, xBatch); err != nil {
		return fmt.Errorf("prefill batch upload: %w", err)
	}
	Barrier()
	if err := BatchRMSNorm(bs.XNorm, bs.X, gm.Layers[0].AttnNorm, dim, npos, eps); err != nil {
		return fmt.Errorf("prefill batch first norm: %w", err)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		spec := &layer.Spec
		gl := &gm.Layers[l]

		if layerDumpEnabled {
			EndBatch()
			Sync()
			if err := CopyRegion(rs.Q, 0, bs.XNorm, uint64((npos-1)*dim*4), uint64(dim*4)); err != nil {
				return fmt.Errorf("layer %d xnorm dump copy: %w", l, err)
			}
			BeginBatch()
			EndBatch()
			Sync()
			layerDumpRow("xn-batched", l, rs.Q, dim)
			BeginBatch()
		}

		// ---- core ----
		Barrier()
		switch spec.Core {
		case llm.CoreShortConv:
			scK := cfg.ShortConvLCache
			if scK < 1 {
				scK = 3
			}
			if err := BatchMatVec(bs.ShortConvBCX, gl.ShortConvInProj.Buf, bs.XNorm,
				3*dim, dim, npos, gl.ShortConvInProj.Type); err != nil {
				return fmt.Errorf("layer %d shortconv in_proj: %w", l, err)
			}
			Barrier()
			if err := ShortConv(bs.ShortConvBCX, bs.MoEOut, gl.ShortConvState,
				gl.ShortConvConv, dim, scK, npos); err != nil {
				return fmt.Errorf("layer %d shortconv conv: %w", l, err)
			}
			Barrier()
			if err := BatchMatVec(bs.AttnProj, gl.ShortConvOutProj.Buf, bs.MoEOut,
				dim, dim, npos, gl.ShortConvOutProj.Type); err != nil {
				return fmt.Errorf("layer %d shortconv out_proj: %w", l, err)
			}

		case llm.CoreAttention:
			if err := BatchMatVec(bs.Q, gl.Wq.Buf, bs.XNorm, qDim, dim, npos, gl.Wq.Type); err != nil {
				return fmt.Errorf("layer %d wq: %w", l, err)
			}
			if err := BatchMatVec(bs.K, gl.Wk.Buf, bs.XNorm, kvDim, dim, npos, gl.Wk.Type); err != nil {
				return fmt.Errorf("layer %d wk: %w", l, err)
			}
			if err := BatchMatVec(bs.V, gl.Wv.Buf, bs.XNorm, kvDim, dim, npos, gl.Wv.Type); err != nil {
				return fmt.Errorf("layer %d wv: %w", l, err)
			}
			if inlineDumpOn("batch") {
				layerDumpBatchRow("qkv-batched", l, bs.Q, npos, qDim)
				layerDumpBatchRow("qkv-batched", l, bs.V, npos, kvDim)
			}
			if gl.Bq != 0 || gl.Bk != 0 || gl.Bv != 0 {
				Barrier()
				// The bias scratch is only needed when the kernel cannot read the
				// bias directly; AttnOut is dead at this point.
				if gl.Bq != 0 {
					if err := BatchAddBias(bs.Q, gl.Bq, bs.AttnOut, qDim, npos); err != nil {
						return fmt.Errorf("layer %d bq: %w", l, err)
					}
				}
				if gl.Bk != 0 {
					if err := BatchAddBias(bs.K, gl.Bk, bs.AttnOut, kvDim, npos); err != nil {
						return fmt.Errorf("layer %d bk: %w", l, err)
					}
				}
				if gl.Bv != 0 {
					if err := BatchAddBias(bs.V, gl.Bv, bs.AttnOut, kvDim, npos); err != nil {
						return fmt.Errorf("layer %d bv: %w", l, err)
					}
				}
			}
			if spec.QKNorm {
				Barrier()
				if err := BatchRMSNormHeads(bs.Q, gl.AttnQNorm, numHeads, headDim, eps, npos); err != nil {
					return fmt.Errorf("layer %d qnorm: %w", l, err)
				}
				Barrier()
				if err := BatchRMSNormHeads(bs.K, gl.AttnKNorm, numKVHeads, headDim, eps, npos); err != nil {
					return fmt.Errorf("layer %d knorm: %w", l, err)
				}
				if inlineDumpOn("batch") {
					layerDumpBatchRow("qknorm-batched", l, bs.Q, npos, qDim)
				}
			}
			Barrier()
			ropeCos, ropeSin := ropeTablesForLayer(pipe, *spec)
			if err := BatchRoPE(bs.Q, bs.K, ropeCos, ropeSin, numHeads, numKVHeads,
				headDim, cfg.RopeDim, startPos, cfg.RopeNeox, npos); err != nil {
				return fmt.Errorf("layer %d rope: %w", l, err)
			}
			if inlineDumpOn("batch") {
				layerDumpBatchRow("rope-batched", l, bs.Q, npos, qDim)
				layerDumpBatchRow("rope-batched", l, bs.K, npos, kvDim)
			}
			Barrier()
			if err := BatchKVStoreF16(kv.KeyBufs[l], kv.ValBufs[l], bs.K, bs.V,
				startPos, kvDim, npos); err != nil {
				return fmt.Errorf("layer %d kvstore: %w", l, err)
			}
			Barrier()
			if err := BatchAttentionF16(bs.AttnOut, bs.Q, kv.KeyBufs[l], kv.ValBufs[l],
				numHeads, numKVHeads, headDim, kvDim, startPos+1, scale, 0, npos); err != nil {
				return fmt.Errorf("layer %d attention: %w", l, err)
			}
			if inlineDumpOn("batch") {
				layerDumpBatchRow("attn-batched", l, bs.AttnOut, npos, qDim)
			}
			Barrier()
			if err := BatchMatVec(bs.AttnProj, gl.Wo.Buf, bs.AttnOut, dim, qDim, npos, gl.Wo.Type); err != nil {
				return fmt.Errorf("layer %d wo: %w", l, err)
			}

		default:
			return fmt.Errorf("layer %d: core type %d not supported by the batched prefill", l, spec.Core)
		}
		if layerDumpEnabled {
			EndBatch()
			Sync()
			if err := CopyRegion(rs.AttnProj, 0, bs.AttnProj, uint64((npos-1)*dim*4), uint64(dim*4)); err != nil {
				return fmt.Errorf("layer %d core dump copy: %w", l, err)
			}
			BeginBatch()
			EndBatch()
			Sync()
			layerDumpRow("core-batched", l, rs.AttnProj, dim)
			BeginBatch()
		}

		// ---- residual + FFN norm: FFNIn = X + AttnProj, FFNNorm = norm(FFNIn) ----
		ffnNormW := gl.FFNNorm
		if spec.Residual == llm.ResPostAttnFFN {
			ffnNormW = gl.PostAttnNorm
		}
		Barrier()
		if err := BatchAddRMSNorm(bs.FFNNorm, bs.FFNIn, bs.X, bs.AttnProj,
			ffnNormW, dim, eps, npos); err != nil {
			return fmt.Errorf("layer %d add+rmsnorm: %w", l, err)
		}

		// ---- FFN ----
		Barrier()
		switch spec.FFN {
		case llm.FFNMoE:
			if err := gpuForwardMoEBatch(layer, gl, bs, cfg, npos); err != nil {
				return fmt.Errorf("layer %d moe: %w", l, err)
			}
		case llm.FFNSwiGLU, llm.FFNGeGLU:
			if err := BatchMatVec(bs.Gate, gl.FFNGate.Buf, bs.FFNNorm,
				gl.FFNGate.Rows, dim, npos, gl.FFNGate.Type); err != nil {
				return fmt.Errorf("layer %d ffn gate: %w", l, err)
			}
			if err := BatchMatVec(bs.Up, gl.FFNUp.Buf, bs.FFNNorm,
				gl.FFNUp.Rows, dim, npos, gl.FFNUp.Type); err != nil {
				return fmt.Errorf("layer %d ffn up: %w", l, err)
			}
			Barrier()
			n := gl.FFNGate.Rows * npos
			if spec.FFN == llm.FFNGeGLU {
				if err := GeGLU(bs.Hidden, bs.Gate, bs.Up, n); err != nil {
					return fmt.Errorf("layer %d geglu: %w", l, err)
				}
			} else {
				if err := SwiGLU(bs.Hidden, bs.Gate, bs.Up, n); err != nil {
					return fmt.Errorf("layer %d swiglu: %w", l, err)
				}
			}
			Barrier()
			if err := BatchMatVec(bs.FFNOut, gl.FFNDown.Buf, bs.Hidden,
				dim, gl.FFNUp.Rows, npos, gl.FFNDown.Type); err != nil {
				return fmt.Errorf("layer %d ffn down: %w", l, err)
			}
		default:
			return fmt.Errorf("layer %d: ffn type %d not supported by the batched prefill", l, spec.FFN)
		}

		// ---- residual + next attention norm ----
		Barrier()
		if err := Add(bs.X, bs.FFNIn, bs.FFNOut, dim*npos); err != nil {
			return fmt.Errorf("layer %d residual: %w", l, err)
		}
		if l+1 < cfg.NumLayers {
			Barrier()
			if err := BatchRMSNorm(bs.XNorm, bs.X, gm.Layers[l+1].AttnNorm, dim, npos, eps); err != nil {
				return fmt.Errorf("layer %d next attn norm: %w", l, err)
			}
		}
		if layerDumpEnabled {
			EndBatch()
			Sync()
			if err := CopyRegion(rs.X, 0, bs.X, uint64((npos-1)*dim*4), uint64(dim*4)); err != nil {
				return fmt.Errorf("layer %d dump copy: %w", l, err)
			}
			BeginBatch()
			EndBatch()
			Sync()
			layerDumpRow("batched", l, rs.X, dim)
			BeginBatch()
		}
	}

	if !lastChunk {
		// The caller uploads the next chunk's embeddings into bs.X; submit first so
		// that upload cannot overwrite this chunk's activations mid-flight.
		EndBatch()
		Sync()
		return nil
	}

	// Final norm + vocabulary projection for the last position only.
	Barrier()
	if err := CopyRegion(rs.X, 0, bs.X, uint64((npos-1)*dim*4), uint64(dim*4)); err != nil {
		return fmt.Errorf("prefill batch copy last row: %w", err)
	}
	Barrier()
	if gm.OutputNormBias != 0 {
		if err := LayerNorm(rs.X, rs.X, gm.OutputNorm, gm.OutputNormBias, dim, eps); err != nil {
			return fmt.Errorf("prefill batch output norm: %w", err)
		}
	} else {
		if err := RMSNorm(rs.X, rs.X, gm.OutputNorm, dim, eps); err != nil {
			return fmt.Errorf("prefill batch output norm: %w", err)
		}
	}
	output := gm.Output
	if output == nil {
		output = gm.TokenEmbed
	}
	Barrier()
	if err := MatVec(rs.Logits, output.Buf, rs.X, output.Rows, output.Cols, output.Type); err != nil {
		return fmt.Errorf("prefill batch logits: %w", err)
	}
	if err := DownloadF32(rs.Logits, logitsBuf); err != nil {
		return fmt.Errorf("prefill batch logits download: %w", err)
	}
	if cfg.FinalLogitSoftcap > 0 {
		capv := float64(cfg.FinalLogitSoftcap)
		for i := range logitsBuf {
			logitsBuf[i] = float32(capv * math.Tanh(float64(logitsBuf[i])/capv))
		}
	}
	return nil
}
