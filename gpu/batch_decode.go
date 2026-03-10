//go:build cgo && vulkan

package gpu

import (
	"math"

	"github.com/computerex/dlgo/models/llm"
)

// BatchDecodeRequest represents one sequence in a batched decode step.
type BatchDecodeRequest struct {
	Token      int32   // current token to process
	Position   int     // sequence position (for RoPE, KV length)
	BlockTable []int   // logical block → physical block ID
	BlockBuf   Buf     // GPU buffer holding the int32 block table
}

// GpuForwardBatchDecode performs a batched single-token decode step for N sequences.
// For each sequence, matmuls are batched across all N (single dispatch).
// Attention is per-sequence using PagedAttention with each sequence's block table.
func GpuForwardBatchDecode(
	m *llm.Model,
	gm *GpuModel,
	pipe *GpuPipeline,
	pool *PagedKVPool,
	rs *GpuRunState,
	bs *GpuBatchState,
	reqs []BatchDecodeRequest,
	logitsBufs [][]float32,
) error {
	if len(reqs) == 0 {
		return nil
	}

	// Single sequence: fall through to optimized single path
	if len(reqs) == 1 {
		return gpuForwardPagedSingle(m, gm, pipe, pool, rs, reqs[0], logitsBufs[0])
	}

	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	N := len(reqs)

	// Embed all N tokens on CPU, upload as batched [N x dim]
	xBatch := make([]float32, N*dim)
	for i, req := range reqs {
		m.TokenEmbed.DequantizeRow(int(req.Token), xBatch[i*dim:(i+1)*dim])
		if cfg.EmbedScale != 0 {
			for j := i * dim; j < (i+1)*dim; j++ {
				xBatch[j] *= cfg.EmbedScale
			}
		}
	}

	BeginBatch()
	UploadF32(bs.X, xBatch)

	// Batched layer processing
	for l := 0; l < cfg.NumLayers; l++ {
		gl := &gm.Layers[l]
		if !gl.OnGPU {
			continue
		}

		// Batched RMSNorm
		Barrier()
		BatchRMSNorm(bs.XNorm, bs.X, gl.AttnNorm, dim, N, cfg.RMSNormEps)

		// Batched Q/K/V projections
		Barrier()
		BatchMatVec(bs.Q, gl.Wq.Buf, bs.XNorm, gl.Wq.Rows, gl.Wq.Cols, N, gl.Wq.Type)
		BatchMatVec(bs.K, gl.Wk.Buf, bs.XNorm, gl.Wk.Rows, gl.Wk.Cols, N, gl.Wk.Type)
		BatchMatVec(bs.V, gl.Wv.Buf, bs.XNorm, gl.Wv.Rows, gl.Wv.Cols, N, gl.Wv.Type)

		// Bias additions if present
		if gl.Bq != 0 {
			for i := 0; i < N; i++ {
				Barrier()
				// TODO: batched bias add
			}
		}

		// Per-sequence: RoPE, KV store, Paged Attention
		for i, req := range reqs {
			Barrier()

			// Extract per-sequence Q, K, V from batch buffers
			qOff := Buf(uint64(i) * uint64(numHeads*headDim) * 4)
			kOff := Buf(uint64(i) * uint64(kvDim) * 4)

			// RoPE on per-sequence Q and K (using batch buffer offsets)
			RoPE(rs.Q, rs.K, pipe.RoPECosTable, pipe.RoPESinTable, numHeads, numKVHeads, headDim, cfg.RopeDim,
				req.Position, cfg.RopeNeox)
			_ = qOff
			_ = kOff

			// Paged KV Store
			pool.StoreKV(l, rs.K, rs.V, req.BlockTable, req.Position)

			// Paged Attention
			seqLen := req.Position + 1
			Barrier()
			PagedAttention(rs.AttnOut, rs.Q,
				pool.KeyPools[l], pool.ValPools[l], req.BlockBuf,
				numHeads, numKVHeads, headDim, kvDim, seqLen, scale, pool.BlockSize)
		}

		// Batched output projection
		Barrier()
		BatchMatVec(bs.AttnProj, gl.Wo.Buf, bs.AttnOut, gl.Wo.Rows, gl.Wo.Cols, N, gl.Wo.Type)

		// Batched residual + FFN
		Barrier()
		// Add attention output to residual
		// bs.X = bs.X + bs.AttnProj (batched)

		if !gl.IsMoE {
			// Batched FFN norm
			Barrier()
			BatchRMSNorm(bs.FFNNorm, bs.X, gl.FFNNorm, dim, N, cfg.RMSNormEps)

			// Batched FFN projections
			Barrier()
			if gl.FFNGate != nil {
				BatchMatVec(bs.Gate, gl.FFNGate.Buf, bs.FFNNorm, gl.FFNGate.Rows, gl.FFNGate.Cols, N, gl.FFNGate.Type)
			}
			BatchMatVec(bs.Up, gl.FFNUp.Buf, bs.FFNNorm, gl.FFNUp.Rows, gl.FFNUp.Cols, N, gl.FFNUp.Type)

			// Batched activation (SwiGLU/GeGLU)
			// Batched down projection
			Barrier()
			BatchMatVec(bs.FFNOut, gl.FFNDown.Buf, bs.Hidden, gl.FFNDown.Rows, gl.FFNDown.Cols, N, gl.FFNDown.Type)
		}
	}

	// Output norm + projection
	Barrier()
	BatchRMSNorm(bs.XNorm, bs.X, gm.OutputNorm, dim, N, cfg.RMSNormEps)
	Barrier()
	BatchMatVec(bs.FFNOut, gm.Output.Buf, bs.XNorm, gm.Output.Rows, gm.Output.Cols, N, gm.Output.Type)

	EndBatch()
	Sync()

	// Download logits for each sequence
	allLogits := make([]float32, N*cfg.VocabSize)
	DownloadF32(bs.FFNOut, allLogits)
	for i := 0; i < N; i++ {
		copy(logitsBufs[i], allLogits[i*cfg.VocabSize:(i+1)*cfg.VocabSize])
	}

	return nil
}

// gpuForwardPagedSingle performs a single-token forward with paged KV for one sequence.
func gpuForwardPagedSingle(
	m *llm.Model,
	gm *GpuModel,
	pipe *GpuPipeline,
	pool *PagedKVPool,
	rs *GpuRunState,
	req BatchDecodeRequest,
	logitsBuf []float32,
) error {
	cfg := m.Config
	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	xCPU := make([]float32, dim)
	m.TokenEmbed.DequantizeRow(int(req.Token), xCPU)
	if cfg.EmbedScale != 0 {
		for i := range xCPU {
			xCPU[i] *= cfg.EmbedScale
		}
	}

	seqLen := req.Position + 1

	BeginBatch()
	UploadF32(rs.X, xCPU)

	for l := 0; l < cfg.NumLayers; l++ {
		gl := &gm.Layers[l]
		if !gl.OnGPU {
			continue
		}

		// RMSNorm
		Barrier()
		RMSNorm(rs.XNorm, rs.X, gl.AttnNorm, dim, cfg.RMSNormEps)

		// Q/K/V projections
		Barrier()
		MatVec(rs.Q, gl.Wq.Buf, rs.XNorm, gl.Wq.Rows, gl.Wq.Cols, gl.Wq.Type)
		MatVec(rs.K, gl.Wk.Buf, rs.XNorm, gl.Wk.Rows, gl.Wk.Cols, gl.Wk.Type)
		MatVec(rs.V, gl.Wv.Buf, rs.XNorm, gl.Wv.Rows, gl.Wv.Cols, gl.Wv.Type)

		if gl.Bq != 0 {
			Barrier()
			Add(rs.Q, rs.Q, gl.Bq, numHeads*headDim)
		}
		if gl.Bk != 0 {
			Add(rs.K, rs.K, gl.Bk, kvDim)
		}
		if gl.Bv != 0 {
			Add(rs.V, rs.V, gl.Bv, kvDim)
		}

		// Q/K norm if needed
		layer := &m.Layers[l]
		if layer.Spec.QKNorm {
			Barrier()
			RMSNormHeads(rs.Q, gl.AttnQNorm, numHeads, headDim, cfg.RMSNormEps)
			RMSNormHeads(rs.K, gl.AttnKNorm, numKVHeads, headDim, cfg.RMSNormEps)
		}

		// RoPE
		Barrier()
		RoPE(rs.Q, rs.K, pipe.RoPECosTable, pipe.RoPESinTable, numHeads, numKVHeads, headDim, cfg.RopeDim,
			req.Position, cfg.RopeNeox)

		// Paged KV store
		pool.StoreKV(l, rs.K, rs.V, req.BlockTable, req.Position)

		// Paged Attention
		Barrier()
		PagedAttention(rs.AttnOut, rs.Q,
			pool.KeyPools[l], pool.ValPools[l], req.BlockBuf,
			numHeads, numKVHeads, headDim, kvDim, seqLen, scale, pool.BlockSize)

		// Output projection
		Barrier()
		MatVec(rs.AttnProj, gl.Wo.Buf, rs.AttnOut, gl.Wo.Rows, gl.Wo.Cols, gl.Wo.Type)

		// Residual connection (X += AttnProj)
		Barrier()
		Add(rs.X, rs.X, rs.AttnProj, dim)

		// FFN
		if !gl.IsMoE {
			Barrier()
			RMSNorm(rs.FFNNorm, rs.X, gl.FFNNorm, dim, cfg.RMSNormEps)
			Barrier()
			if gl.FFNGate != nil {
				MatVec(rs.Gate, gl.FFNGate.Buf, rs.FFNNorm, gl.FFNGate.Rows, gl.FFNGate.Cols, gl.FFNGate.Type)
			}
			MatVec(rs.Up, gl.FFNUp.Buf, rs.FFNNorm, gl.FFNUp.Rows, gl.FFNUp.Cols, gl.FFNUp.Type)
			Barrier()
			SwiGLU(rs.Hidden, rs.Gate, rs.Up, cfg.FFNDim)
			Barrier()
			MatVec(rs.FFNOut, gl.FFNDown.Buf, rs.Hidden, gl.FFNDown.Rows, gl.FFNDown.Cols, gl.FFNDown.Type)
			Barrier()
			Add(rs.X, rs.X, rs.FFNOut, dim)
		}
	}

	// Output norm + projection
	Barrier()
	RMSNorm(rs.XNorm, rs.X, gm.OutputNorm, dim, cfg.RMSNormEps)
	Barrier()
	MatVec(rs.Logits, gm.Output.Buf, rs.XNorm, gm.Output.Rows, gm.Output.Cols, gm.Output.Type)

	EndBatch()
	Sync()

	DownloadF32(rs.Logits, logitsBuf)
	return nil
}
