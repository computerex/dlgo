package llm

import (
	"math"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/ops"
	"github.com/computerex/dlgo/quant"
)

func sigmoidF32(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

// BatchState holds pre-allocated buffers for batch (prefill) forward passes.
type BatchState struct {
	maxPos    int // maximum batch size for prefill processing
	maxSeqLen int // maximum total context length (for score/gather buffers)
	dim    int
	qDim   int
	kvDim  int
	ffnDim int

	XBatch     []float32 // [maxPos * dim]
	XNormBatch []float32 // [maxPos * dim]
	QBatch     []float32 // [maxPos * qDim]
	KBatch     []float32 // [maxPos * kvDim]
	VBatch     []float32 // [maxPos * kvDim]
	AttnBatch  []float32 // [maxPos * qDim]
	ProjBatch  []float32 // [maxPos * dim]
	FFNInBatch []float32 // [maxPos * dim]
	NormBatch  []float32 // [maxPos * dim]
	GateBatch  []float32 // [maxPos * ffnDim]
	UpBatch    []float32 // [maxPos * ffnDim]
	HidBatch   []float32 // [maxPos * ffnDim]
	FFNBatch   []float32 // [maxPos * dim]

	Q8Buf []byte // pre-allocated Q8 quantization buffer

	// GatedQ attention: Wq outputs 2*qDim (interleaved Q + gate per head)
	QFullBatch []float32 // [maxPos * 2*qDim]
	QGateBatch []float32 // [maxPos * qDim]

	// SSM batch buffers: matmuls batched across positions, recurrence per-position
	SSMQKVBatch   []float32 // [maxPos * qkvDim]
	SSMZBatch     []float32 // [maxPos * valueDim]
	SSMAlphaBatch []float32 // [maxPos * numHeads]
	SSMBetaBatch  []float32 // [maxPos * numHeads]
	SSMYBatch     []float32 // [maxPos * valueDim]

	// MoE batch buffers
	MoERouterBatch []float32 // [maxPos * expertCount] router logits
	MoEExpDim      int       // expert FFN hidden dim
	MoEShDim       int       // shared expert FFN hidden dim

	// Score buffers sized to maxSeqLen (full context) — each worker needs one
	// score vector covering ALL positions for correct causal attention.
	ScoreBufs [][]float32 // [numWorkers][maxSeqLen]

	// KGather/VGather for SIMD batched attention.
	// Layout: [numKVHeads * maxSeqLen * headDim]
	// Nil when maxSeqLen is too large (SIMD disabled, non-SIMD path used instead).
	KGather   []float32
	VGather   []float32
}

// NewBatchState allocates batch buffers for up to maxPos positions.
// maxSeqLenHint (optional) specifies the total context length for score/gather
// buffer sizing; defaults to maxPos when not provided.
func NewBatchState(cfg ModelConfig, maxPos int, maxSeqLenHint ...int) *BatchState {
	maxSeqLen := maxPos
	if len(maxSeqLenHint) > 0 && maxSeqLenHint[0] > maxPos {
		maxSeqLen = maxSeqLenHint[0]
	}
	dim := cfg.EmbeddingDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	ffnDim := cfg.FFNDim
	maxDim := dim
	if ffnDim > maxDim {
		maxDim = ffnDim
	}
	q8Size := quant.Q8BufferSize(2, maxDim) // Q4_0 Q8 size as base
	q8SizeK := quant.Q8BufferSize(12, maxDim) // K-quant Q8 size
	if q8SizeK > q8Size {
		q8Size = q8SizeK
	}

	numWorkers := blas.DefaultPool().NumWorkers()
	// ScoreBufs must hold scores for ALL positions up to maxSeqLen, not just the
	// current batch. During chunked prefill chunk N, position p in the chunk
	// attends to positions 0..startPos+p, which can exceed the chunk batch size.
	scoreBufs := make([][]float32, numWorkers)
	for i := range scoreBufs {
		scoreBufs[i] = make([]float32, maxSeqLen)
	}

	hasGatedQ := cfg.FullAttentionInterval > 0
	qFullBatch := []float32(nil)
	qGateBatch := []float32(nil)
	if hasGatedQ {
		qFullBatch = make([]float32, maxPos*2*qDim)
		qGateBatch = make([]float32, maxPos*qDim)
	}

	// MoE batch buffers
	var moERouterBatch []float32
	moEExpDim := cfg.ExpertFFNDim
	moEShDim := cfg.SharedExpertFFNDim
	if moEShDim == 0 {
		moEShDim = moEExpDim
	}
	if cfg.ExpertCount > 0 {
		moERouterBatch = make([]float32, maxPos*cfg.ExpertCount)
	}

	// SSM batch buffers
	var ssmQKVBatch, ssmZBatch, ssmAlphaBatch, ssmBetaBatch, ssmYBatch []float32
	if cfg.FullAttentionInterval > 0 && cfg.SSMInnerSize > 0 {
		ssmNumHeads := cfg.SSMTimeStepRank
		ssmNumKVGroups := cfg.SSMGroupCount
		if ssmNumKVGroups <= 0 {
			ssmNumKVGroups = ssmNumHeads
		}
		ssmHeadKDim := cfg.SSMStateSize
		ssmHeadVDim := cfg.SSMInnerSize / ssmNumHeads
		ssmKeyDim := ssmNumKVGroups * ssmHeadKDim
		ssmValueDim := ssmNumHeads * ssmHeadVDim
		ssmQKVDim := ssmKeyDim*2 + ssmValueDim
		ssmQKVBatch = make([]float32, maxPos*ssmQKVDim)
		ssmZBatch = make([]float32, maxPos*ssmValueDim)
		ssmAlphaBatch = make([]float32, maxPos*ssmNumHeads)
		ssmBetaBatch = make([]float32, maxPos*ssmNumHeads)
		ssmYBatch = make([]float32, maxPos*ssmValueDim)
	}

	// KGather/VGather for SIMD batched attention: layout [numKVHeads*maxSeqLen*headDim].
	// Only allocate when the total size is affordable (≤ 512 MB each).
	// When nil, ForwardBatch falls back to the non-SIMD attention path.
	const simdGatherLimit = 512 * 1024 * 1024 // 512 MB per buffer
	kgatherElems := cfg.NumKVHeads * maxSeqLen * cfg.HeadDim
	var kGather, vGather []float32
	if int64(kgatherElems)*4 <= simdGatherLimit {
		kGather = make([]float32, kgatherElems)
		vGather = make([]float32, kgatherElems)
	}

	return &BatchState{
		maxPos:    maxPos,
		maxSeqLen: maxSeqLen,
		dim:        dim,
		qDim:       qDim,
		kvDim:      kvDim,
		ffnDim:     ffnDim,
		XBatch:     make([]float32, maxPos*dim),
		XNormBatch: make([]float32, maxPos*dim),
		QBatch:     make([]float32, maxPos*qDim),
		KBatch:     make([]float32, maxPos*kvDim),
		VBatch:     make([]float32, maxPos*kvDim),
		AttnBatch:  make([]float32, maxPos*qDim),
		ProjBatch:  make([]float32, maxPos*dim),
		FFNInBatch: make([]float32, maxPos*dim),
		NormBatch:  make([]float32, maxPos*dim),
		GateBatch:  make([]float32, maxPos*ffnDim),
		UpBatch:    make([]float32, maxPos*ffnDim),
		HidBatch:   make([]float32, maxPos*ffnDim),
		FFNBatch:   make([]float32, maxPos*dim),
		Q8Buf:      make([]byte, maxPos*q8Size),
		QFullBatch:    qFullBatch,
		QGateBatch:    qGateBatch,
		MoERouterBatch: moERouterBatch,
		MoEExpDim:      moEExpDim,
		MoEShDim:       moEShDim,
		SSMQKVBatch:   ssmQKVBatch,
		SSMZBatch:     ssmZBatch,
		SSMAlphaBatch: ssmAlphaBatch,
		SSMBetaBatch:  ssmBetaBatch,
		SSMYBatch:     ssmYBatch,
		ScoreBufs:     scoreBufs,
		KGather:       kGather,
		VGather:       vGather,
	}
}

// ForwardBatch processes multiple tokens in a single pass (prefill).
// Returns logits for the last position only. Fills the KV cache for all positions.
// If the number of tokens exceeds bs.maxPos, the prompt is processed in chunks
// of bs.maxPos, correctly maintaining KV cache and SSM state across chunks.
func ForwardBatch(m *Model, tokens []int32, startPos int, kv *memory.MultiLayerKVCache, rs *RunState, bs *BatchState) []float32 {
	cfg := m.Config
	nPos := len(tokens)
	if nPos == 0 {
		return rs.Logits
	}
	if nPos == 1 {
		return Forward(m, tokens[0], startPos, kv, rs)
	}

	// Chunked prefill: if the prompt is larger than the batch buffer, process in
	// chunks. Each chunk updates KV cache and SSM state correctly for the next.
	if nPos > bs.maxPos {
		chunkSize := bs.maxPos
		for start := 0; start < nPos; start += chunkSize {
			end := start + chunkSize
			if end > nPos {
				end = nPos
			}
			ForwardBatch(m, tokens[start:end], startPos+start, kv, rs, bs)
		}
		return rs.Logits
	}

	dim := cfg.EmbeddingDim
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	kvDim := numKVHeads * headDim
	kvMul := numHeads / numKVHeads
	pool := rs.Pool

	// Embed all tokens — parallelized
	pool.ParallelFor(nPos, func(p int) {
		_ = m.TokenEmbed.DequantizeRow(int(tokens[p]), bs.XBatch[p*dim:(p+1)*dim])
		if cfg.EmbedScale != 0 {
			ops.Scale(bs.XBatch[p*dim:(p+1)*dim], cfg.EmbedScale)
		}
	})

	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		spec := &layer.Spec

		if spec.Core == CoreSSM {
			ssmState := rs.SSMState.Layers[l]
			ssmNumHeads := ssmState.NumHeads
			ssmNumKVGroups := ssmState.NumKVGroups
			if ssmNumKVGroups <= 0 {
				ssmNumKVGroups = ssmNumHeads
			}
			ssmHeadKDim := ssmState.HeadKDim
			ssmHeadVDim := ssmState.HeadVDim
			ssmConvK := ssmState.ConvK
			ssmValueDim := ssmNumHeads * ssmHeadVDim
			ssmKeyDim := ssmNumKVGroups * ssmHeadKDim
			ssmQKVDim := ssmState.Channels

			// Batch norm all positions
			pool.ParallelFor(nPos, func(p int) {
				ops.RMSNorm(bs.XNormBatch[p*dim:(p+1)*dim], bs.XBatch[p*dim:(p+1)*dim], layer.AttnNorm, cfg.RMSNormEps)
			})

			// Batch the 4 SSM input projections across all positions
			blas.QBatchGEMMParallel(bs.SSMQKVBatch[:nPos*ssmQKVDim], layer.SSMInProj, bs.XNormBatch[:nPos*dim], nPos, pool)
			blas.QBatchGEMMParallel(bs.SSMZBatch[:nPos*ssmValueDim], layer.AttnGate, bs.XNormBatch[:nPos*dim], nPos, pool)
			blas.QBatchGEMMParallel(bs.SSMAlphaBatch[:nPos*ssmNumHeads], layer.SSMAlpha, bs.XNormBatch[:nPos*dim], nPos, pool)
			blas.QBatchGEMMParallel(bs.SSMBetaBatch[:nPos*ssmNumHeads], layer.SSMBeta, bs.XNormBatch[:nPos*dim], nPos, pool)

			// Per-position recurrence (conv1d + preprocess + delta rule + norm gate)
			qScale := float32(1.0 / math.Sqrt(float64(ssmHeadKDim)))
			for p := 0; p < nPos; p++ {
				qkv := bs.SSMQKVBatch[p*ssmQKVDim : (p+1)*ssmQKVDim]
				z := bs.SSMZBatch[p*ssmValueDim : (p+1)*ssmValueDim]
				alpha := bs.SSMAlphaBatch[p*ssmNumHeads : (p+1)*ssmNumHeads]
				beta := bs.SSMBetaBatch[p*ssmNumHeads : (p+1)*ssmNumHeads]
				y := bs.SSMYBatch[p*ssmValueDim : (p+1)*ssmValueDim]

				// Conv1d: shift buffer, store, depthwise conv
				buf := ssmState.ConvBuf
				copy(buf[0:(ssmConvK-1)*ssmQKVDim], buf[ssmQKVDim:ssmConvK*ssmQKVDim])
				copy(buf[(ssmConvK-1)*ssmQKVDim:ssmConvK*ssmQKVDim], qkv[:ssmQKVDim])
				w := layer.SSMConv1dW
				for c := 0; c < ssmQKVDim; c++ {
					var acc float32
					wOff := c * ssmConvK
					for ki := 0; ki < ssmConvK; ki++ {
						acc += buf[ki*ssmQKVDim+c] * w[wOff+ki]
					}
					qkv[c] = acc
				}
				ops.SiLU(qkv[:ssmQKVDim])

				qq := qkv[:ssmKeyDim]
				kk := qkv[ssmKeyDim : 2*ssmKeyDim]
				vv := qkv[2*ssmKeyDim : 2*ssmKeyDim+ssmValueDim]

				// Preprocess: decay + learning rate + normalize Q/K
				for h := 0; h < ssmNumHeads; h++ {
					a := alpha[h]
					if layer.SSMDtBias != nil {
						a += layer.SSMDtBias[h]
					}
					alpha[h] = layer.SSMA[h] * float32(math.Log(1.0+math.Exp(float64(a))))
					beta[h] = ops.Sigmoid(beta[h])
				}
				for g := 0; g < ssmNumKVGroups; g++ {
					l2Normalize(qq[g*ssmHeadKDim:(g+1)*ssmHeadKDim], cfg.RMSNormEps)
					l2Normalize(kk[g*ssmHeadKDim:(g+1)*ssmHeadKDim], cfg.RMSNormEps)
				}
				for i := 0; i < ssmKeyDim; i++ {
					qq[i] *= qScale
				}

				// Delta rule recurrence — parallelized across heads,
				// loop order changed: outer=i(key), inner=j(value) for sequential memory access
				state := ssmState.State
				pool.ParallelFor(ssmNumHeads, func(h int) {
					decay := float32(math.Exp(float64(alpha[h])))
					lr := beta[h]
					kvGroup := h % ssmNumKVGroups
					if !cfg.SSMTiledVOrder {
						kvGroup = h / (ssmNumHeads / ssmNumKVGroups)
					}
					qH := qq[kvGroup*ssmHeadKDim : (kvGroup+1)*ssmHeadKDim]
					kH := kk[kvGroup*ssmHeadKDim : (kvGroup+1)*ssmHeadKDim]
					vH := vv[h*ssmHeadVDim : (h+1)*ssmHeadVDim]
					yH := y[h*ssmHeadVDim : (h+1)*ssmHeadVDim]
					sOff := h * ssmHeadKDim * ssmHeadVDim

					// Decay state
					for idx := sOff; idx < sOff+ssmHeadKDim*ssmHeadVDim; idx++ {
						state[idx] *= decay
					}

					// Predict: vPred = S^T @ k (row-major traversal)
					var vPred [256]float32
					for i := 0; i < ssmHeadKDim; i++ {
						row := state[sOff+i*ssmHeadVDim : sOff+(i+1)*ssmHeadVDim]
						ki := kH[i]
						for j := 0; j < ssmHeadVDim; j++ {
							vPred[j] += row[j] * ki
						}
					}

					// Update: S += lr * outer(k, v - vPred)
					for i := 0; i < ssmHeadKDim; i++ {
						row := state[sOff+i*ssmHeadVDim : sOff+(i+1)*ssmHeadVDim]
						lrk := lr * kH[i]
						for j := 0; j < ssmHeadVDim; j++ {
							row[j] += lrk * (vH[j] - vPred[j])
						}
					}

					// Output: y = S^T @ q (row-major traversal)
					for j := 0; j < ssmHeadVDim; j++ {
						yH[j] = 0
					}
					for i := 0; i < ssmHeadKDim; i++ {
						row := state[sOff+i*ssmHeadVDim : sOff+(i+1)*ssmHeadVDim]
						qi := qH[i]
						for j := 0; j < ssmHeadVDim; j++ {
							yH[j] += row[j] * qi
						}
					}

					// Per-head RMSNorm + SiLU gate
					zH := z[h*ssmHeadVDim : (h+1)*ssmHeadVDim]
					ops.RMSNormInPlace(yH, layer.SSMNorm, cfg.RMSNormEps)
					for j := 0; j < ssmHeadVDim; j++ {
						yH[j] *= zH[j] * ops.Sigmoid(zH[j])
					}
				})
			}

			// Batch SSMOut matmul across all positions
			blas.QBatchGEMMParallel(bs.ProjBatch[:nPos*dim], layer.SSMOut, bs.SSMYBatch[:nPos*ssmValueDim], nPos, pool)

			batchResidualFFN(layer, bs, rs, nPos, dim, cfg, pool)
			continue
		}

		// MLA layers: process sequentially (KV cache has sequential dependencies)
		if spec.Core == CoreMLA {
			for p := 0; p < nPos; p++ {
				pos := startPos + p
				x := bs.XBatch[p*dim : (p+1)*dim]
				xn := bs.XNormBatch[p*dim : (p+1)*dim]
				ops.RMSNorm(xn, x, layer.AttnNorm, cfg.RMSNormEps)
				copy(rs.XNorm, xn)
				ForwardMLA(layer, rs, kv, l, pos, cfg, pool)
				copy(bs.ProjBatch[p*dim:(p+1)*dim], rs.AttnProj)
			}
			batchResidualFFN(layer, bs, rs, nPos, dim, cfg, pool)
			continue
		}

		// Batch norm — parallelize across positions
		pool.ParallelFor(nPos, func(p int) {
			x := bs.XBatch[p*dim : (p+1)*dim]
			xn := bs.XNormBatch[p*dim : (p+1)*dim]
			switch spec.Norm {
			case NormRMS:
				ops.RMSNorm(xn, x, layer.AttnNorm, cfg.RMSNormEps)
			case NormLayer:
				ops.LayerNorm(xn, x, layer.AttnNorm, layer.AttnNormBias, cfg.RMSNormEps)
			}
		})

		// Batch Q/K/V projections (fused: quantize input once, single dispatch)
		qDim := numHeads * headDim
		if spec.GatedQ {
			// GatedQ: Wq produces 2*qDim (Q + gate interleaved per head)
			blas.QBatchGEMMParallel(bs.QFullBatch[:nPos*2*qDim], layer.Wq, bs.XNormBatch[:nPos*dim], nPos, pool)
			blas.QBatchGEMMParallel(bs.KBatch[:nPos*kvDim], layer.Wk, bs.XNormBatch[:nPos*dim], nPos, pool)
			blas.QBatchGEMMParallel(bs.VBatch[:nPos*kvDim], layer.Wv, bs.XNormBatch[:nPos*dim], nPos, pool)
			pool.ParallelFor(nPos, func(p int) {
				qFull := bs.QFullBatch[p*2*qDim : (p+1)*2*qDim]
				qp := bs.QBatch[p*qDim : (p+1)*qDim]
				gp := bs.QGateBatch[p*qDim : (p+1)*qDim]
				for h := 0; h < numHeads; h++ {
					copy(qp[h*headDim:(h+1)*headDim], qFull[h*2*headDim:h*2*headDim+headDim])
					copy(gp[h*headDim:(h+1)*headDim], qFull[h*2*headDim+headDim:(h+1)*2*headDim])
				}
			})
		} else {
			blas.QTripleBatchGEMMParallel(
				bs.QBatch[:nPos*qDim], layer.Wq,
				bs.KBatch[:nPos*kvDim], layer.Wk,
				bs.VBatch[:nPos*kvDim], layer.Wv,
				bs.XNormBatch[:nPos*dim], nPos, pool,
			)
		}

		// Pre-compute attention constants needed for KV gather fusion
		maxSeqLen := startPos + nPos
		// SIMD batched attention requires pre-allocated KGather/VGather buffers.
		// These are nil when maxSeqLen would exceed the allocation limit.
		useSIMDAttn := quant.HasCausalAttn() && len(bs.KGather) >= maxSeqLen*kvDim

		// Per-position: bias, QK norm, RoPE, KV store — parallelized
		pool.ParallelFor(nPos, func(p int) {
			pos := startPos + p
			qp := bs.QBatch[p*qDim : (p+1)*qDim]
			kp := bs.KBatch[p*kvDim : (p+1)*kvDim]
			vp := bs.VBatch[p*kvDim : (p+1)*kvDim]

			if layer.Bq != nil {
				ops.AddBias(qp, layer.Bq)
			}
			if layer.Bk != nil {
				ops.AddBias(kp, layer.Bk)
			}
			if layer.Bv != nil {
				ops.AddBias(vp, layer.Bv)
			}

			if spec.QKNorm {
				for h := 0; h < numHeads; h++ {
					ops.RMSNormInPlace(qp[h*headDim:(h+1)*headDim], layer.AttnQNorm, cfg.RMSNormEps)
				}
				for h := 0; h < numKVHeads; h++ {
					ops.RMSNormInPlace(kp[h*headDim:(h+1)*headDim], layer.AttnKNorm, cfg.RMSNormEps)
				}
			}

			if rs.ropeCos != nil {
				for h := 0; h < numHeads; h++ {
					rs.ApplyRoPEFast(qp[h*headDim:(h+1)*headDim], pos)
				}
				for h := 0; h < numKVHeads; h++ {
					rs.ApplyRoPEFast(kp[h*headDim:(h+1)*headDim], pos)
				}
			} else {
				ops.ApplyRoPEBatch(qp, numHeads, kp, numKVHeads, pos, headDim, cfg.RopeFreqBase, cfg.RopeNeox)
			}

			kv.Layers[l].Store(pos, kp, vp)

			if useSIMDAttn {
				for kvH := 0; kvH < numKVHeads; kvH++ {
					srcOff := kvH * headDim
					dstOff := kvH*maxSeqLen*headDim + pos*headDim
					copy(bs.KGather[dstOff:dstOff+headDim], kp[srcOff:srcOff+headDim])
					copy(bs.VGather[dstOff:dstOff+headDim], vp[srcOff:srcOff+headDim])
				}
			}
		})

		if useSIMDAttn && startPos > 0 {
			// Gather historical positions not covered by the KV store above.
			kd := kv.Layers[l].KeyData
			vd := kv.Layers[l].ValData
			pool.ParallelFor(startPos, func(t int) {
				for kvH := 0; kvH < numKVHeads; kvH++ {
					srcOff := t*kvDim + kvH*headDim
					dstOff := kvH*maxSeqLen*headDim + t*headDim
					copy(bs.KGather[dstOff:dstOff+headDim], kd[srcOff:srcOff+headDim])
					copy(bs.VGather[dstOff:dstOff+headDim], vd[srcOff:srcOff+headDim])
				}
			})
		}

		// Batch causal attention — parallelise over heads × positions
		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		ops.Clear(bs.AttnBatch[:nPos*qDim])
		numTasks := numHeads * nPos
		numW := pool.NumWorkers()
		if numW > numTasks {
			numW = numTasks
		}

		pool.DispatchChunked(numTasks, numW, func(workerID, start, end int) {
			scores := bs.ScoreBufs[workerID][:maxSeqLen]
			for idx := start; idx < end; idx++ {
				h := idx / nPos
				p := idx % nPos
				kvH := h / kvMul
				pos := startPos + p
				seqLen := pos + 1

				qHead := bs.QBatch[p*qDim+h*headDim : p*qDim+(h+1)*headDim]
				headOut := bs.AttnBatch[p*qDim+h*headDim : p*qDim+(h+1)*headDim]

				if useSIMDAttn {
					kvBase := kvH * maxSeqLen * headDim
					quant.CausalAttnHead(qHead, headDim,
						bs.KGather[kvBase:], bs.VGather[kvBase:],
						0, headDim, seqLen, scale,
						scores[:seqLen], headOut)
				} else {
					sc := scores[:seqLen]
					for t := 0; t < seqLen; t++ {
						kHead := kv.Layers[l].Keys[t][kvH*headDim : (kvH+1)*headDim]
						sc[t] = ops.DotProduct(qHead, kHead, headDim) * scale
					}
					ops.Softmax(sc)
					for t := 0; t < seqLen; t++ {
						vHead := kv.Layers[l].Vals[t][kvH*headDim : (kvH+1)*headDim]
						ops.AddScaled(headOut, sc[t], vHead, headDim)
					}
				}
			}
		})

		// GatedQ: apply sigmoid gate to attention output
		if spec.GatedQ {
			pool.ParallelFor(nPos, func(p int) {
				attn := bs.AttnBatch[p*qDim : (p+1)*qDim]
				gate := bs.QGateBatch[p*qDim : (p+1)*qDim]
				for i := range attn {
					attn[i] *= sigmoidF32(gate[i])
				}
			})
		}

		// Batch output projection
		blas.QBatchGEMMParallel(bs.ProjBatch[:nPos*dim], layer.Wo, bs.AttnBatch[:nPos*qDim], nPos, pool)
		if layer.Bo != nil {
			pool.ParallelFor(nPos, func(p int) {
				ops.AddBias(bs.ProjBatch[p*dim:(p+1)*dim], layer.Bo)
			})
		}
		// Residual + FFN
		batchResidualFFN(layer, bs, rs, nPos, dim, cfg, pool)
	}

	// Final norm + logits (last position only)
	lastX := bs.XBatch[(nPos-1)*dim : nPos*dim]
	copy(rs.X, lastX)

	if m.OutputNormBias != nil {
		ops.LayerNorm(rs.X[:dim], rs.X[:dim], m.OutputNorm, m.OutputNormBias, cfg.RMSNormEps)
	} else {
		ops.RMSNormInPlace(rs.X[:dim], m.OutputNorm, cfg.RMSNormEps)
	}

	output := m.Output
	if output == nil {
		output = m.TokenEmbed
	}
	blas.QMatVecMulParallel(rs.Logits, output, rs.X, pool)


	if m.OutputBias != nil {
		ops.AddBias(rs.Logits, m.OutputBias)
	}

	return rs.Logits
}

// batchResidualFFN applies residual wiring + FFN for all positions in the batch.
func batchResidualFFN(layer *Layer, bs *BatchState, rs *RunState, nPos, dim int, cfg ModelConfig, pool *blas.Pool) {
	spec := &layer.Spec
	ffnDim := cfg.FFNDim

	switch spec.Residual {
	case ResStandard:
		pool.ParallelFor(nPos, func(p int) {
			proj := bs.ProjBatch[p*dim : (p+1)*dim]
			x := bs.XBatch[p*dim : (p+1)*dim]
			ffnIn := bs.FFNInBatch[p*dim : (p+1)*dim]
			if layer.PostAttnNorm != nil {
				ops.RMSNormInPlace(proj, layer.PostAttnNorm, cfg.RMSNormEps)
			}
			ops.Add(ffnIn, x, proj)
			ops.RMSNorm(bs.NormBatch[p*dim:(p+1)*dim], ffnIn, layer.FFNNorm, cfg.RMSNormEps)
		})
		batchFFN(layer, bs, rs, nPos, dim, ffnDim, bs.NormBatch, cfg, pool)
		pool.ParallelFor(nPos, func(p int) {
			if layer.PostFFNNorm != nil {
				ops.RMSNormInPlace(bs.FFNBatch[p*dim:(p+1)*dim], layer.PostFFNNorm, cfg.RMSNormEps)
			}
			ops.Add(bs.XBatch[p*dim:(p+1)*dim], bs.FFNInBatch[p*dim:(p+1)*dim], bs.FFNBatch[p*dim:(p+1)*dim])
		})

	case ResPostAttnFFN:
		pool.ParallelFor(nPos, func(p int) {
			x := bs.XBatch[p*dim : (p+1)*dim]
			proj := bs.ProjBatch[p*dim : (p+1)*dim]
			ffnIn := bs.FFNInBatch[p*dim : (p+1)*dim]
			ops.Add(ffnIn, x, proj)
			ops.RMSNorm(bs.NormBatch[p*dim:(p+1)*dim], ffnIn, layer.PostAttnNorm, cfg.RMSNormEps)
		})
		batchFFN(layer, bs, rs, nPos, dim, ffnDim, bs.NormBatch, cfg, pool)
		pool.ParallelFor(nPos, func(p int) {
			ops.Add(bs.XBatch[p*dim:(p+1)*dim], bs.FFNInBatch[p*dim:(p+1)*dim], bs.FFNBatch[p*dim:(p+1)*dim])
		})

	case ResParallel:
		batchFFN(layer, bs, rs, nPos, dim, ffnDim, bs.XNormBatch, cfg, pool)
		pool.ParallelFor(nPos, func(p int) {
			x := bs.XBatch[p*dim : (p+1)*dim]
			proj := bs.ProjBatch[p*dim : (p+1)*dim]
			ffn := bs.FFNBatch[p*dim : (p+1)*dim]
			for i := 0; i < dim; i++ {
				x[i] = x[i] + proj[i] + ffn[i]
			}
		})
	}
}

// batchFFN runs the FFN for all positions using batch GEMM.
func batchFFN(layer *Layer, bs *BatchState, rs *RunState, nPos, dim, ffnDim int, inputBatch []float32, cfg ModelConfig, pool *blas.Pool) {
	switch layer.Spec.FFN {
	case FFNSwiGLU:
		blas.QDualBatchGEMMParallel(
			bs.GateBatch[:nPos*ffnDim], layer.FFNGate,
			bs.UpBatch[:nPos*ffnDim], layer.FFNUp,
			inputBatch[:nPos*dim], nPos, pool,
		)
		pool.ParallelFor(nPos, func(p int) {
			quant.SIMDSwiGLU(
				bs.HidBatch[p*ffnDim:(p+1)*ffnDim],
				bs.GateBatch[p*ffnDim:(p+1)*ffnDim],
				bs.UpBatch[p*ffnDim:(p+1)*ffnDim],
				ffnDim,
			)
		})
		blas.QBatchGEMMParallel(bs.FFNBatch[:nPos*dim], layer.FFNDown, bs.HidBatch[:nPos*ffnDim], nPos, pool)

	case FFNGeGLU:
		blas.QDualBatchGEMMParallel(
			bs.GateBatch[:nPos*ffnDim], layer.FFNGate,
			bs.UpBatch[:nPos*ffnDim], layer.FFNUp,
			inputBatch[:nPos*dim], nPos, pool,
		)
		pool.ParallelFor(nPos, func(p int) {
			ops.GeGLU(
				bs.HidBatch[p*ffnDim:(p+1)*ffnDim],
				bs.GateBatch[p*ffnDim:(p+1)*ffnDim],
				bs.UpBatch[p*ffnDim:(p+1)*ffnDim],
				ffnDim,
			)
		})
		blas.QBatchGEMMParallel(bs.FFNBatch[:nPos*dim], layer.FFNDown, bs.HidBatch[:nPos*ffnDim], nPos, pool)

	case FFNPlain:
		blas.QBatchGEMMParallel(bs.UpBatch[:nPos*ffnDim], layer.FFNUp, inputBatch[:nPos*dim], nPos, pool)
		pool.ParallelFor(nPos, func(p int) {
			up := bs.UpBatch[p*ffnDim : (p+1)*ffnDim]
			if layer.FFNUpBias != nil {
				ops.AddBias(up, layer.FFNUpBias)
			}
			ops.GELU(up)
		})
		blas.QBatchGEMMParallel(bs.FFNBatch[:nPos*dim], layer.FFNDown, bs.UpBatch[:nPos*ffnDim], nPos, pool)

	case FFNMoE:
		batchMoEFFN(layer, bs, rs, nPos, dim, inputBatch, cfg, pool)
		return
	}

	if layer.FFNDownBias != nil {
		pool.ParallelFor(nPos, func(p int) {
			ops.AddBias(bs.FFNBatch[p*dim:(p+1)*dim], layer.FFNDownBias)
		})
	}
}

// batchMoEFFN runs Mixture-of-Experts FFN for all positions. Per-position routing.
func batchMoEFFN(layer *Layer, bs *BatchState, rs *RunState, nPos, dim int, inputBatch []float32, cfg ModelConfig, pool *blas.Pool) {
	expertCount := cfg.ExpertCount
	nUsed := cfg.ExpertUsedCount
	expDim := bs.MoEExpDim

	// Batch router projection
	blas.QBatchGEMMParallel(bs.MoERouterBatch[:nPos*expertCount], layer.FFNRouter, inputBatch[:nPos*dim], nPos, pool)

	// Per-position expert selection and computation
	for p := 0; p < nPos; p++ {
		input := inputBatch[p*dim : (p+1)*dim]
		output := bs.FFNBatch[p*dim : (p+1)*dim]
		ops.Clear(output)

		logits := bs.MoERouterBatch[p*expertCount : (p+1)*expertCount]
		quant.SIMDSoftmax(logits)

		indices, weights := topKIndices(logits, nUsed)

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

		// Build expert slices for batched gate+up dispatch
		downBpr := quant.BytesForN(layer.FFNDownExps.Type, layer.FFNDownExps.Cols)

		fused := layer.FFNGateUpExps != nil
		var gateTensor, upTensor *core.QuantizedTensor
		if fused {
			gateTensor = layer.FFNGateUpExps
			upTensor = layer.FFNGateUpExps
		} else {
			gateTensor = layer.FFNGateExps
			upTensor = layer.FFNUpExps
		}
		bpr := quant.BytesForN(gateTensor.Type, gateTensor.Cols)

		gateSlices := make([]blas.ExpertSlice, 0, nUsed)
		upSlices := make([]blas.ExpertSlice, 0, nUsed)
		activeExperts := make([]int, 0, nUsed)

		for e := 0; e < nUsed; e++ {
			idx := indices[e]
			if idx < 0 {
				continue
			}
			activeExperts = append(activeExperts, e)
			if fused {
				gateSlices = append(gateSlices, blas.ExpertSlice{
					Out: rs.MoEGates[e], Rows: expDim, Off: idx * 2 * expDim * bpr,
				})
				upSlices = append(upSlices, blas.ExpertSlice{
					Out: rs.MoEUps[e], Rows: expDim, Off: (idx*2 + 1) * expDim * bpr,
				})
			} else {
				gateSlices = append(gateSlices, blas.ExpertSlice{
					Out: rs.MoEGates[e], Rows: expDim, Off: idx * expDim * bpr,
				})
				upSlices = append(upSlices, blas.ExpertSlice{
					Out: rs.MoEUps[e], Rows: expDim, Off: idx * expDim * bpr,
				})
			}
		}

		blas.QDualMultiExpertMatVec(gateTensor, upTensor,
			gateSlices, upSlices, input, pool)

		for _, e := range activeExperts {
			quant.SIMDSwiGLU(rs.MoEHiddens[e], rs.MoEGates[e], rs.MoEUps[e], expDim)
		}

		nActive := len(activeExperts)
		totalDownRows := nActive * dim
		useFusedDown := quant.HasSIMDDot(layer.FFNDownExps.Type)
		downCols := layer.FFNDownExps.Cols
		pool.DispatchChunked(totalDownRows, pool.NumWorkers(), func(_, start, end int) {
			for row := start; row < end; {
				ei := row / dim
				rowInExpert := row % dim
				e := activeExperts[ei]
				idx := indices[e]
				endInExpert := end - ei*dim
				if endInExpert > dim {
					endInExpert = dim
				}
				nrows := endInExpert - rowInExpert
				downOff := idx*dim*downBpr + rowInExpert*downBpr
				out := rs.MoEExpertOuts[e]
				if useFusedDown {
					quant.SIMDDotBatch(
						layer.FFNDownExps.Data[downOff:downOff+nrows*downBpr],
						layer.FFNDownExps.Type, rs.MoEHiddens[e],
						downCols, out[rowInExpert:endInExpert], nrows, downBpr)
				} else {
					buf := make([]float32, downCols)
					for r := rowInExpert; r < endInExpert; r++ {
						rOff := idx*dim*downBpr + r*downBpr
						quant.DequantizeInto(buf, layer.FFNDownExps.Data[rOff:rOff+downBpr], layer.FFNDownExps.Type, downCols)
						out[r] = quant.SIMDDotF32(buf, rs.MoEHiddens[e], downCols)
					}
				}
				row = (ei + 1) * dim
			}
		})

		for _, e := range activeExperts {
			w := weights[e]
			out := rs.MoEExpertOuts[e]
			for i := 0; i < dim; i++ {
				output[i] += w * out[i]
			}
		}

		// Shared expert
		if layer.FFNGateShared != nil {
			shDim := layer.FFNGateShared.Rows
			blas.QDualMatVecMulParallel(rs.MoEShGate, layer.FFNGateShared, rs.MoEShUp, layer.FFNUpShared, input, pool)
			quant.SIMDSwiGLU(rs.MoEShHidden, rs.MoEShGate, rs.MoEShUp, shDim)
			blas.QMatVecMulParallel(rs.MoEShOut, layer.FFNDownShared, rs.MoEShHidden, pool)

			if layer.FFNRouterShared != nil {
				gate := sigmoidF32(ops.DotProduct(layer.FFNRouterShared, input, dim))
				for i := 0; i < dim; i++ {
					output[i] += gate * rs.MoEShOut[i]
				}
			} else {
				for i := 0; i < dim; i++ {
					output[i] += rs.MoEShOut[i]
				}
			}
		}
	}
}
