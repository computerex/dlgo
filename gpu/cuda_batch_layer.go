//go:build cgo && vulkan

package gpu

// CUDA batched prefill: the Go equivalent of the C gpu_forward_layer_batch
// (gpu/csrc/vulkan_gpu.c), which is Vulkan-only because it is a command-buffer
// recording. Under CUDA the same sequence runs as ordinary kernel launches:
//
//	QKV GEMMs -> QK-norm -> RoPE -> KV store -> flash attention -> O GEMM
//	-> (add + rmsnorm) -> gate/up GEMMs -> SwiGLU -> down GEMM -> (add + rmsnorm)
//
// The two things that make this faster than driving the single-token path npos
// times are the Affine4/Affine8 tensor-core GEMM (weights are read once per
// chunk instead of once per token) and the query-tiled prefill attention (K/V
// traffic is amortised over 64 queries instead of re-read per query).

import (
	"fmt"
	"math"
	"os"

	"github.com/computerex/dlgo/models/llm"
)

// debugBatchPrefill prints the activation magnitudes of the batched prefill
// (DLGO_BATCH_DEBUG=1): the f16 activation staging has a hard 65504 range.
var debugBatchPrefill = os.Getenv("DLGO_BATCH_DEBUG") == "1"

// cudaBatchPrefillSupported reports whether the CUDA batched stack can execute
// this model EXACTLY. GpuForwardPrefillBatch cannot return an error, so anything
// not implemented here must be rejected here â€” otherwise the prompt would be
// silently mis-computed.
func cudaBatchPrefillSupported(m *llm.Model, gm *GpuModel, isPartial bool) bool {
	if !cudaActive || isPartial || gm == nil {
		return false
	}
	cfg := m.Config
	if cfg.NumLayers <= 0 || cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 {
		return false
	}
	if cfg.HeadDim%16 != 0 || cfg.HeadDim > 128 {
		return false
	}
	if cfg.NumHeads%cfg.NumKVHeads != 0 {
		return false
	}
	// Every GEMM k-tile is one 32-weight affine block.
	for _, d := range []int{cfg.EmbeddingDim, cfg.HeadDim * cfg.NumHeads,
		cfg.HeadDim * cfg.NumKVHeads, cfg.FFNDim} {
		if d <= 0 || d%32 != 0 {
			return false
		}
	}
	for l := range m.Layers {
		sp := m.Layers[l].Spec
		gl := &gm.Layers[l]
		if !gl.OnGPU || gl.IsMoE {
			return false
		}
		if sp.SlidingWindow > 0 || sp.GatedQ {
			return false
		}
		if sp.Core == llm.CoreSSM || sp.Core == llm.CoreMLA || sp.Core == llm.CoreShortConv {
			return false
		}
		if sp.FFN != llm.FFNSwiGLU || sp.Residual == llm.ResParallel {
			return false
		}
		if gl.Bq != 0 || gl.Bk != 0 || gl.Bv != 0 || gl.Bo != 0 ||
			gl.AttnSinks != 0 || gl.PostFFNNorm != 0 {
			return false
		}
		// Only the canonical formats have a CUDA batched GEMM.
		for _, w := range []*GpuTensor{gl.Wq, gl.Wk, gl.Wv, gl.Wo,
			gl.FFNGate, gl.FFNUp, gl.FFNDown} {
			if w == nil || (w.Type != 60 && w.Type != 61) {
				return false
			}
		}
	}
	return true
}

// cudaForwardLayerBatch runs one transformer layer over npos prompt tokens.
func cudaForwardLayerBatch(lc *LayerConf, npos, startPos int, scale float32, nextAttnNorm Buf) error {
	c := &lc.c
	dim := int(c.dim)
	headDim := int(c.head_dim)
	numHeads := int(c.num_heads)
	numKVHeads := int(c.num_kv_heads)
	kvDim := int(c.kv_dim)
	eps := float32(c.rms_eps)

	if int(c.core_type) == 0 {
		if c.bq != 0 || c.bk != 0 || c.bv != 0 || c.bo != 0 {
			return fmt.Errorf("cuda batch: attention biases unsupported")
		}
		if c.attn_sinks != 0 {
			return fmt.Errorf("cuda batch: attention sinks unsupported")
		}
		if int(c.sliding_window) > 0 || int(c.kv_cap) > 0 {
			return fmt.Errorf("cuda batch: requires a linear KV cache")
		}
		if err := cbpE(cbpQKV, func() error {
			for _, p := range []struct {
				out, w, x  Buf
				rows, cols int
				qtype      uint32
			}{
				{Buf(c.q), Buf(c.wq), Buf(c.x_norm), int(c.wq_rows), int(c.wq_cols), uint32(c.wq_type)},
				{Buf(c.k), Buf(c.wk), Buf(c.x_norm), int(c.wk_rows), int(c.wk_cols), uint32(c.wk_type)},
				{Buf(c.v), Buf(c.wv), Buf(c.x_norm), int(c.wv_rows), int(c.wv_cols), uint32(c.wv_type)},
			} {
				if err := cudaBatchMatVec(p.out, p.w, p.x, p.rows, p.cols, npos, p.qtype); err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return err
		}
		if c.q_norm_w != 0 {
			if err := cbpE(cbpQKNorm, func() error {
				if err := cudaBatchRMSNormHeads(Buf(c.q), Buf(c.q_norm_w), numHeads, headDim, npos, eps); err != nil {
					return err
				}
				return cudaBatchRMSNormHeads(Buf(c.k), Buf(c.k_norm_w), numKVHeads, headDim, npos, eps)
			}); err != nil {
				return err
			}
		}
		if err := cbpE(cbpRoPE, func() error {
			return cudaBatchRoPE(Buf(c.q), Buf(c.k), Buf(c.rope_cos_table), Buf(c.rope_sin_table),
				numHeads, numKVHeads, headDim, int(c.rope_dim), startPos,
				int(c.rope_neox) != 0, npos)
		}); err != nil {
			return err
		}
		if err := cbpE(cbpKVStore, func() error {
			return cudaBatchKVStoreF16(Buf(c.k), Buf(c.v), Buf(c.k_cache), Buf(c.v_cache),
				startPos, kvDim, npos)
		}); err != nil {
			return err
		}
		if err := cbpE(cbpAttn, func() error {
			return cudaBatchAttentionF16(Buf(c.attn_out), Buf(c.q), Buf(c.k_cache), Buf(c.v_cache),
				numHeads, numKVHeads, headDim, kvDim, startPos, npos, scale,
				float32(c.attn_logit_softcap))
		}); err != nil {
			return err
		}
		if err := cbpE(cbpOProj, func() error {
			return cudaBatchMatVec(Buf(c.attn_proj), Buf(c.wo), Buf(c.attn_out),
				int(c.wo_rows), int(c.wo_cols), npos, uint32(c.wo_type))
		}); err != nil {
			return err
		}
	}

	if int(c.residual_type) != 0 {
		return fmt.Errorf("cuda batch: parallel residual unsupported")
	}

	if c.post_attn_norm_w != 0 {
		if err := cudaBatchRMSNorm(Buf(c.attn_proj), Buf(c.attn_proj), Buf(c.post_attn_norm_w),
			dim, npos, eps); err != nil {
			return err
		}
	}
	// x += attn_proj, and ffn_norm = rmsnorm(x) for the FFN block.
	if err := cbpE(cbpAddNorm, func() error {
		return cudaBatchAddRMSNorm(Buf(c.ffn_norm), Buf(c.ffn_in), Buf(c.x), Buf(c.attn_proj),
			Buf(c.ffn_norm_w), dim, npos, eps)
	}); err != nil {
		return err
	}

	switch int(c.ffn_type) {
	case 3:
		// MoE: the Go side owns the expert FFN and the residual.
		return nil
	case 0: // SwiGLU
		gateRows := int(c.gate_rows)
		if err := cbpE(cbpGate, func() error {
			return cudaBatchMatVec(Buf(c.gate), Buf(c.ffn_gate_w), Buf(c.ffn_norm),
				gateRows, int(c.gate_cols), npos, uint32(c.gate_type))
		}); err != nil {
			return err
		}
		if err := cbpE(cbpUp, func() error {
			return cudaBatchMatVec(Buf(c.up), Buf(c.ffn_up_w), Buf(c.ffn_norm),
				int(c.up_rows), int(c.up_cols), npos, uint32(c.up_type))
		}); err != nil {
			return err
		}
		if err := cbpE(cbpSwiGLU, func() error {
			return cudaSwiGLU(Buf(c.hidden), Buf(c.gate), Buf(c.up), gateRows*npos)
		}); err != nil {
			return err
		}
		if debugBatchPrefill {
			probe := make([]float32, 64)
			if err := DownloadF32(Buf(c.hidden), probe); err == nil {
				mx, nf := 0.0, 0
				for _, v := range probe {
					a := math.Abs(float64(v))
					if math.IsNaN(a) || math.IsInf(a, 0) {
						nf++
						continue
					}
					if a > mx {
						mx = a
					}
				}
				fmt.Fprintf(os.Stderr, "[gpu] cuda batch: swiglu out max|x|=%.4g nonfinite=%d\n", mx, nf)
			}
		}
		if err := cbpE(cbpDown, func() error {
			return cudaBatchMatVec(Buf(c.ffn_out), Buf(c.ffn_down_w), Buf(c.hidden),
				int(c.down_rows), int(c.down_cols), npos, uint32(c.down_type))
		}); err != nil {
			return err
		}
	default:
		return fmt.Errorf("cuda batch: only SwiGLU FFN is implemented (ffn_type=%d)", int(c.ffn_type))
	}

	if c.post_ffn_norm_w != 0 {
		if err := cudaBatchRMSNorm(Buf(c.ffn_out), Buf(c.ffn_out), Buf(c.post_ffn_norm_w),
			dim, npos, eps); err != nil {
			return err
		}
	}
	// x += ffn_out, then norm for the next layer (or a plain add on the last).
	if nextAttnNorm != 0 {
		return cbpE(cbpAddNorm, func() error {
			return cudaBatchAddRMSNorm(Buf(c.x_norm), Buf(c.x), Buf(c.ffn_in), Buf(c.ffn_out),
				Buf(nextAttnNorm), dim, npos, eps)
		})
	}
	return cudaAdd(Buf(c.x), Buf(c.ffn_in), Buf(c.ffn_out), dim*npos)
}

// cudaPrefillBatch is the CUDA form of GpuForwardPrefillBatch: embed the whole
// chunk, run every layer batched, then project the last token's hidden state.
func cudaPrefillBatch(m *llm.Model, gm *GpuModel, tokens []int32,
	rs *GpuRunState, bs *GpuBatchState, startPos int, logitsBuf []float32,
	batchLayerConfs []*LayerConf, hiddenBuf []float32) {

	npos := len(tokens)
	if npos <= 0 || bs == nil {
		return
	}
	cfg := m.Config
	dim := cfg.EmbeddingDim
	scale := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	// Decode captures a CUDA graph on its first step, and a capture cannot
	// allocate. The batched path never calls the per-token attention, so reserve
	// its partials here or the capture fails with a stream-capture error.
	if err := cudaPrimeAttentionPartials(cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim); err != nil {
		fmt.Fprintf(os.Stderr, "[gpu] cuda batch prefill: prime attention scratch: %v\n", err)
		return
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
	if err := UploadF32(bs.X, xBatch); err != nil {
		fmt.Fprintf(os.Stderr, "[gpu] cuda batch prefill: embedding upload: %v\n", err)
		return
	}
	if m.Layers[0].Spec.Norm == llm.NormRMS {
		if err := cudaBatchRMSNorm(bs.XNorm, bs.X, gm.Layers[0].AttnNorm, dim, npos,
			cfg.RMSNormEps); err != nil {
			fmt.Fprintf(os.Stderr, "[gpu] cuda batch prefill: input norm: %v\n", err)
			return
		}
	}

	for l := 0; l < cfg.NumLayers; l++ {
		var nextAttnNorm Buf
		if l < cfg.NumLayers-1 {
			nextAttnNorm = gm.Layers[l+1].AttnNorm
		}
		if err := cudaForwardLayerBatch(batchLayerConfs[l], npos, startPos, scale,
			nextAttnNorm); err != nil {
			// Loud on purpose: this path cannot fall back mid-prompt without
			// producing garbage, so a failure must never be silent.
			fmt.Fprintf(os.Stderr, "[gpu] cuda batch prefill layer %d: %v\n", l, err)
			return
		}
	}

	Barrier()
	CopyRegion(rs.X, 0, bs.X, uint64((npos-1)*dim*4), uint64(dim*4))
	if debugBatchPrefill {
		// f16 activation staging silently degrades once |x| exceeds 65504, so
		// report the real magnitudes before anything else is blamed.
		probe := make([]float32, dim)
		if err := DownloadF32(rs.X, probe); err == nil {
			mx, nf := 0.0, 0
			for _, v := range probe {
				a := math.Abs(float64(v))
				if math.IsNaN(a) || math.IsInf(a, 0) {
					nf++
					continue
				}
				if a > mx {
					mx = a
				}
			}
			fmt.Fprintf(os.Stderr, "[gpu] cuda batch: final hidden max|x|=%.4g nonfinite=%d\n", mx, nf)
		}
	}
	Barrier()
	_ = RMSNorm(rs.X, rs.X, gm.OutputNorm, dim, cfg.RMSNormEps)
	Barrier()
	if hiddenBuf != nil {
		_ = DownloadF32(rs.X, hiddenBuf)
		return
	}
	output := gm.Output
	if output == nil {
		output = gm.TokenEmbed
	}
	_ = MatVec(rs.Logits, output.Buf, rs.X, output.Rows, output.Cols, output.Type)
	_ = DownloadF32(rs.Logits, logitsBuf)
}
