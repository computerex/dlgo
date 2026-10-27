//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"math"
	"os"
	"sync"

	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/models/llm"
)

// Position-batched MoE.
//
// The per-token path runs one matvec per (token, selected expert) pair, so a
// prompt token re-reads the experts it selected and nothing is shared between
// tokens. For 64 experts with 4 used, a chunk of B tokens issues B*4 expert
// reads where one dense pass over the expert tensor would do — roughly 16x the
// DRAM traffic — and every one of those matvecs is a separate dispatch, which is
// what actually dominates (see gpu/optime.go: moe-ffn is ~35% of a prefill
// token).
//
// Here the routing for the whole chunk is decided once, the (token, expert)
// pairs are grouped by expert, and each expert runs one GEMM over just the rows
// that selected it. Two tiny kernels (gather_rows / scatter_rows) bridge the
// token-axis indexing that the matvec kernels do not have.

// moeBatchSupported reports whether the position-batched MoE can run this layer
// exactly. Anything it cannot reproduce falls back to the per-token path, which
// stays the reference implementation.
func moeBatchSupported(layer *llm.Layer, gl *GpuLayer, cfg llm.ModelConfig) bool {
	if gl == nil || layer == nil || !gl.IsMoE {
		return false
	}
	if gl.FFNRouter == nil || gl.FFNDownExps == nil {
		return false
	}
	if gl.FFNGateExps == nil && gl.FFNGateUpExps == nil {
		return false
	}
	// Router logit bias (gpt-oss ffn_gate_inp.bias). gpu_forward_moe_ffn adds it
	// to the logits on the device before moe_topk; the batched path has the whole
	// chunk's logits on the host anyway, so it adds the same float32 values there
	// and the selection matches. That needs the host copy, not just the GPU one.
	if gl.FFNRouterBias != 0 && len(layer.FFNRouterBias) != cfg.ExpertCount {
		return false
	}
	// Per-expert projection biases (gpt-oss ffn_{gate,up,down}_exps.bias), laid
	// out [expertCount][expDim] for gate/up and [expertCount][dim] for down.
	// gpu_moe_expert_bias_add applies them after each projection.
	if gl.FFNGateExpsBias != 0 || gl.FFNUpExpsBias != 0 {
		if cfg.ExpertFFNDim <= 0 {
			return false
		}
		if gl.FFNGateExpsBias != 0 && len(layer.FFNGateExpsBias) != cfg.ExpertCount*cfg.ExpertFFNDim {
			return false
		}
		if gl.FFNUpExpsBias != 0 && len(layer.FFNUpExpsBias) != cfg.ExpertCount*cfg.ExpertFFNDim {
			return false
		}
	}
	if gl.FFNDownExpsBias != 0 && len(layer.FFNDownExpsBias) != cfg.ExpertCount*cfg.EmbeddingDim {
		return false
	}
	// Shared expert (Qwen3-Next / Qwen3.5-class MoE): a dense FFN that runs on
	// every token, whose output is added to the routed experts'. Batched by
	// gpuForwardMoESharedBatch, which needs all three projections with matching
	// inner dimensions.
	if gl.FFNGateShared != nil {
		if gl.FFNUpShared == nil || gl.FFNDownShared == nil {
			return false
		}
		if !moeBatchQTypeOK(gl.FFNGateShared.Type) || !moeBatchQTypeOK(gl.FFNUpShared.Type) ||
			!moeBatchQTypeOK(gl.FFNDownShared.Type) {
			return false
		}
		if gl.FFNGateShared.Rows <= 0 || gl.FFNGateShared.Rows != gl.FFNUpShared.Rows ||
			gl.FFNDownShared.Cols != gl.FFNGateShared.Rows {
			return false
		}
	}
	// moe_topk.comp keeps at most 512 experts in registers.
	if cfg.ExpertCount <= 0 || cfg.ExpertCount > 512 {
		return false
	}
	if cfg.ExpertUsedCount <= 0 || cfg.ExpertUsedCount > cfg.ExpertCount {
		return false
	}
	// Every projection must have a plain batched matvec pipeline.
	if !moeBatchQTypeOK(gl.FFNRouter.Type) || !moeBatchQTypeOK(gl.FFNDownExps.Type) {
		return false
	}
	if gl.FFNGateUpExps != nil {
		if !moeBatchQTypeOK(gl.FFNGateUpExps.Type) {
			return false
		}
	} else if !moeBatchQTypeOK(gl.FFNGateExps.Type) || !moeBatchQTypeOK(gl.FFNUpExps.Type) {
		return false
	}
	// The expert stride is used as a descriptor offset, so it must satisfy the
	// device's storage-buffer alignment.
	align := MinSSBOOffsetAlign()
	if align == 0 {
		align = 1
	}
	for _, n := range moeBatchExpertStrides(layer, gl, cfg) {
		if n <= 0 || uint64(n)%align != 0 {
			return false
		}
	}
	return true
}

// moeBatchScratchReady reports whether the batch state carries the scratch
// gpuForwardMoEBatch writes into. Only AllocMoEBatch allocates it, and the
// pipeline skips that allocation (or it fails) when VRAM is tight, so callers
// must check rather than assume.
func moeBatchScratchReady(bs *GpuBatchState) bool {
	return bs != nil && bs.MoELogits != 0 && bs.MoEOut != 0 && bs.MoEIdx != 0 &&
		bs.MoEW != 0 && bs.Gate != 0 && bs.Up != 0 && bs.Hidden != 0
}

// moeBatchQTypeOK mirrors the quant types gpu_batch_matvec_offset can dispatch.
// 60/61 are the canonical Affine8/Affine4 forms, which the default unified-quant
// path repacks every weight tensor into.
func moeBatchQTypeOK(t uint32) bool {
	switch t {
	case 0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 34, 35, 39, 60, 61:
		return true
	}
	return false
}

// moeBatchExpertBytes returns the byte size of one expert's slice of a GPU
// weight tensor, for the layouts whose block size is known here.
//
// This must be derived from the GPU tensor, not from the CPU tensor's payload:
// the unified-quant path repacks every weight to Affine4/Affine8 at upload time
// while the CPU tensor keeps the original GGUF bytes, so mixing the two gives a
// stride that is off by the block-size ratio (e.g. 18/20 for a Q4_0 source
// repacked to Affine4).
func moeBatchExpertBytes(gt *GpuTensor, nExperts int) int {
	if gt == nil || nExperts <= 0 || gt.Rows <= 0 || gt.Cols <= 0 {
		return 0
	}
	var blockBytes int
	switch gt.Type {
	case 60: // Affine8: 36-byte block per 32 weights
		blockBytes = 36
	case 61: // Affine4: 20-byte block per 32 weights
		blockBytes = 20
	default:
		return 0
	}
	blocksPerRow := (gt.Cols + 31) / 32
	return (gt.Rows / nExperts) * blocksPerRow * blockBytes
}

// moeBatchExpertStrides returns the byte sizes of one expert slice for the gate,
// up and down tensors.
func moeBatchExpertStrides(layer *llm.Layer, gl *GpuLayer, cfg llm.ModelConfig) [3]int {
	n := cfg.ExpertCount
	var out [3]int
	if n <= 0 {
		return out
	}
	// bytesOf prefers the canonical (repacked) GPU layout and falls back to the
	// CPU payload for native per-quant tensors, where the byte length is exact.
	bytesOf := func(gt *GpuTensor, ct *core.QuantizedTensor) int {
		if v := moeBatchExpertBytes(gt, n); v > 0 {
			return v
		}
		if ct != nil {
			return len(ct.Data) / n
		}
		return 0
	}
	if gl.FFNGateUpExps != nil && layer.FFNGateUpExps != nil {
		// Fused [expert][gate | up]: gate first, then up at +stride/2.
		out[0] = bytesOf(gl.FFNGateUpExps, layer.FFNGateUpExps)
		out[1] = out[0] / 2
	} else {
		out[0] = bytesOf(gl.FFNGateExps, layer.FFNGateExps)
		out[1] = bytesOf(gl.FFNUpExps, layer.FFNUpExps)
	}
	out[2] = bytesOf(gl.FFNDownExps, layer.FFNDownExps)
	return out
}

// moeBatchPlan holds the per-chunk routing decision plus the scratch needed to
// derive it. Reused across layers and chunks to keep the CPU side allocation-free.
type moeBatchPlan struct {
	npos, nExperts, nUsed int

	counts []int32   // [nExperts]
	starts []int32   // [nExperts+1]
	cursor []int32   // [nExperts]
	tokens []int32   // [npos*nUsed] token ids grouped by expert
	weight []float32 // [npos*nUsed] routing weights, parallel to tokens
	pair   []int32   // [npos*nUsed] slot-major -> pair index (inverse of tokens)
	selIdx []int32   // [npos*nUsed] slot-major top-k before grouping
	selW   []float32 // [npos*nUsed]
	logits []float32 // [npos*nExperts] router logits downloaded from the GPU
	vals   []float32 // [nExperts]
	used   []bool
}

var moePlanScratch moeBatchPlan

func getMoEPlan(npos, nExperts, nUsed int) *moeBatchPlan {
	p := &moePlanScratch
	if cap(p.counts) < nExperts {
		p.counts = make([]int32, nExperts)
		p.starts = make([]int32, nExperts+1)
		p.cursor = make([]int32, nExperts)
		p.vals = make([]float32, nExperts)
		p.used = make([]bool, nExperts)
	}
	p.counts = p.counts[:nExperts]
	p.starts = p.starts[:nExperts+1]
	p.cursor = p.cursor[:nExperts]
	p.vals = p.vals[:nExperts]
	p.used = p.used[:nExperts]

	total := npos * nUsed
	if cap(p.tokens) < total {
		p.tokens = make([]int32, total)
		p.weight = make([]float32, total)
		p.selIdx = make([]int32, total)
		p.selW = make([]float32, total)
		p.pair = make([]int32, total)
	}
	p.tokens = p.tokens[:total]
	p.weight = p.weight[:total]
	p.selIdx = p.selIdx[:total]
	p.selW = p.selW[:total]
	p.pair = p.pair[:total]

	nl := npos * nExperts
	if cap(p.logits) < nl {
		p.logits = make([]float32, nl)
	}
	p.logits = p.logits[:nl]

	p.npos, p.nExperts, p.nUsed = npos, nExperts, nUsed
	return p
}

// planMoEBatch applies the router gating and top-k for every position in the
// chunk, then groups the selections by expert.
//
// The gating, tie-breaking and renormalisation must reproduce moe_topk.comp
// exactly — in particular the ascending scan with a strict `>` comparison (ties
// go to the lower index) and the post-activation, selection-only bias, which is
// added before the top-k and subtracted again from the emitted weight.
func planMoEBatch(p *moeBatchPlan, selBias []float32, gatingFunc int,
	weightsNorm bool, weightsScale float32) {
	npos, nExperts, nUsed := p.npos, p.nExperts, p.nUsed
	topK := nUsed
	if topK > nExperts {
		topK = nExperts
	}

	for t := 0; t < npos; t++ {
		lg := p.logits[t*nExperts : (t+1)*nExperts]
		vals := p.vals

		switch gatingFunc {
		case 2: // sigmoid, then the optional selection-only bias
			for i := 0; i < nExperts; i++ {
				v := float32(1) / (float32(1) + float32(math.Exp(float64(-lg[i]))))
				if selBias != nil {
					v += selBias[i]
				}
				vals[i] = v
			}
		case 3: // top-k on raw logits, softmax on the selected afterwards
			copy(vals, lg)
		default: // softmax over all experts
			maxV := lg[0]
			for i := 1; i < nExperts; i++ {
				if lg[i] > maxV {
					maxV = lg[i]
				}
			}
			var sum float32
			for i := 0; i < nExperts; i++ {
				vals[i] = float32(math.Exp(float64(lg[i] - maxV)))
				sum += vals[i]
			}
			if sum != 0 {
				for i := 0; i < nExperts; i++ {
					vals[i] /= sum
				}
			}
		}

		for i := 0; i < nExperts; i++ {
			p.used[i] = false
		}
		for s := 0; s < topK; s++ {
			bestIdx := -1
			bestVal := float32(-1e30)
			for i := 0; i < nExperts; i++ {
				if !p.used[i] && vals[i] > bestVal {
					bestVal = vals[i]
					bestIdx = i
				}
			}
			if bestIdx < 0 {
				bestIdx = 0
				bestVal = 0
			}
			if selBias != nil && gatingFunc == 2 {
				bestVal -= selBias[bestIdx]
			}
			p.selIdx[t*nUsed+s] = int32(bestIdx)
			p.selW[t*nUsed+s] = bestVal
			p.used[bestIdx] = true
		}

		if gatingFunc == 3 && topK > 0 {
			maxW := p.selW[t*nUsed]
			for s := 1; s < topK; s++ {
				if p.selW[t*nUsed+s] > maxW {
					maxW = p.selW[t*nUsed+s]
				}
			}
			var sum float32
			for s := 0; s < topK; s++ {
				v := float32(math.Exp(float64(p.selW[t*nUsed+s] - maxW)))
				p.selW[t*nUsed+s] = v
				sum += v
			}
			if sum != 0 {
				for s := 0; s < topK; s++ {
					p.selW[t*nUsed+s] /= sum
				}
			}
		}

		if weightsNorm && topK > 0 {
			var wSum float32
			for s := 0; s < topK; s++ {
				wSum += p.selW[t*nUsed+s]
			}
			if wSum > 1e-12 {
				inv := float32(1) / wSum
				for s := 0; s < topK; s++ {
					p.selW[t*nUsed+s] *= inv
				}
			}
		}

		if weightsScale != 0 && weightsScale != 1 {
			for s := 0; s < topK; s++ {
				p.selW[t*nUsed+s] *= weightsScale
			}
		}
	}

	// Group by expert: counts, prefix sum, then a stable fill in token order.
	for i := range p.counts {
		p.counts[i] = 0
	}
	for i := 0; i < npos*nUsed; i++ {
		p.counts[p.selIdx[i]]++
	}
	p.starts[0] = 0
	for e := 0; e < nExperts; e++ {
		p.starts[e+1] = p.starts[e] + p.counts[e]
		p.cursor[e] = p.starts[e]
	}
	for t := 0; t < npos; t++ {
		for s := 0; s < nUsed; s++ {
			i := t*nUsed + s
			e := p.selIdx[i]
			d := p.cursor[e]
			p.tokens[d] = int32(t)
			p.weight[d] = p.selW[i]
			p.pair[i] = d
			p.cursor[e] = d + 1
		}
	}
}

// gpuForwardMoEBatch runs one MoE layer for a chunk of npos positions, consuming
// bs.FFNNorm (the normed FFN input for the chunk) and producing bs.FFNOut.
//
// Two implementations exist. The grouped one (gpuForwardMoEGrouped) issues a
// single dispatch per projection covering every expert; the per-expert one
// (gpuForwardMoEBatchPerExpert) loops over experts with one dispatch each. The
// grouped path needs canonical Affine weights (see moeGroupedSupported) and the
// pair-major scratch from AllocMoEGrouped, so native-quant models — and any build
// that skipped that allocation — keep the per-expert loop.
func gpuForwardMoEBatch(layer *llm.Layer, gl *GpuLayer, bs *GpuBatchState,
	cfg llm.ModelConfig, npos int) error {
	// Both implementations index npos-sized rows of bs.FFNNorm and npos*nUsed
	// pairs of the pair-major scratch, so a chunk longer than the batch state was
	// sized for would write past the end of it. The pipeline chunks by
	// min(npos, prefillChunkSize) and sizes the state with the same value, but
	// nothing else enforces it.
	if bs == nil || npos <= 0 || npos > bs.Npos {
		return fmt.Errorf("moe: chunk of %d positions does not fit the batch state (%d)", npos, bs.Npos)
	}
	if moeGroupedSupported(layer, gl, cfg, bs) {
		if err := gpuForwardMoEGrouped(layer, gl, bs, cfg, npos); err != nil {
			return err
		}
	} else if err := gpuForwardMoEBatchPerExpert(layer, gl, bs, cfg, npos); err != nil {
		return err
	}
	// The shared expert is common to both implementations and runs after they
	// have accumulated the routed experts into bs.FFNOut.
	return gpuForwardMoESharedBatch(layer, gl, bs, cfg, npos)
}

// gpuForwardMoESharedBatch adds the shared expert's output to bs.FFNOut.
//
// The shared expert is a plain dense FFN that runs on every token (Qwen3-Next /
// Qwen3.5-class MoE), consuming the same normed FFN input and adding into the
// same output the routed experts do: gate/up -> SwiGLU -> down.
//
// The one wrinkle is the optional FFNRouterShared: a [dim] vector whose dot
// product with the FFN input, passed through a sigmoid, scales the whole
// shared-expert output — one scalar per position. That is applied with
// ScatterRows over an identity index rather than a dedicated row-scale kernel:
// it already computes dst[i] += w[i] * src[i] for `dim`-wide rows, which is
// exactly the weighted accumulate wanted, so the shared expert needs no shader
// of its own.
func gpuForwardMoESharedBatch(layer *llm.Layer, gl *GpuLayer, bs *GpuBatchState,
	cfg llm.ModelConfig, npos int) error {
	if gl == nil || gl.FFNGateShared == nil {
		return nil
	}
	dim := cfg.EmbeddingDim
	shDim := gl.FFNGateShared.Rows
	if dim <= 0 || shDim <= 0 {
		return fmt.Errorf("moe shared: bad shape dim=%d shDim=%d", dim, shDim)
	}
	if bs.Gate == 0 || bs.Up == 0 || bs.Hidden == 0 || bs.MoEOut == 0 {
		return fmt.Errorf("moe shared: missing batch scratch")
	}

	// MoEOut is dead here: the per-expert loop already scattered its last
	// expert's down output and the grouped path never touches it.
	Barrier()
	if err := BatchMatVec(bs.Gate, gl.FFNGateShared.Buf, bs.FFNNorm, shDim, dim, npos,
		gl.FFNGateShared.Type); err != nil {
		return fmt.Errorf("moe shared gate (type=%d): %w", gl.FFNGateShared.Type, err)
	}
	if err := BatchMatVec(bs.Up, gl.FFNUpShared.Buf, bs.FFNNorm, shDim, dim, npos,
		gl.FFNUpShared.Type); err != nil {
		return fmt.Errorf("moe shared up (type=%d): %w", gl.FFNUpShared.Type, err)
	}
	Barrier()
	if layer.Spec.FFN == llm.FFNMoESwiOAI {
		if err := SwiGLU_OAI(bs.Hidden, bs.Gate, bs.Up, shDim*npos, 1.702, 7.0); err != nil {
			return fmt.Errorf("moe shared swiglu_oai: %w", err)
		}
	} else if err := SwiGLU(bs.Hidden, bs.Gate, bs.Up, shDim*npos); err != nil {
		return fmt.Errorf("moe shared swiglu: %w", err)
	}
	Barrier()
	if err := BatchMatVec(bs.MoEOut, gl.FFNDownShared.Buf, bs.Hidden, dim, shDim, npos,
		gl.FFNDownShared.Type); err != nil {
		return fmt.Errorf("moe shared down (type=%d): %w", gl.FFNDownShared.Type, err)
	}
	Barrier()

	if gl.FFNRouterShared == 0 {
		return Add(bs.FFNOut, bs.FFNOut, bs.MoEOut, dim*npos)
	}

	// Per-position gate. One row of router weights over the whole chunk gives
	// [npos] dot products; the sigmoid and the weighted accumulate stay cheap.
	if bs.MoELogits == 0 || bs.MoEW == 0 || bs.MoEIdx == 0 {
		return fmt.Errorf("moe shared: missing gate scratch")
	}
	if err := BatchMatVec(bs.MoELogits, gl.FFNRouterShared, bs.FFNNorm, 1, dim, npos, 0); err != nil {
		return fmt.Errorf("moe shared router: %w", err)
	}
	g := ensureScratch(moeSharedGate, npos)
	moeSharedGate = g
	if err := DownloadF32(bs.MoELogits, g); err != nil {
		return fmt.Errorf("moe shared router download: %w", err)
	}
	for i := 0; i < npos; i++ {
		g[i] = float32(1) / (float32(1) + float32(math.Exp(float64(-g[i]))))
	}
	// DownloadF32 submits (and waits on) the recording batch, so the recording
	// state is gone; reopen it or every remaining dispatch of this chunk becomes
	// its own submit+wait.
	BeginBatch()
	if err := UploadF32(bs.MoEW, g); err != nil {
		return fmt.Errorf("moe shared gate upload: %w", err)
	}
	moeSharedIdx = ensureI32Scratch(moeSharedIdx, npos)
	for i := range moeSharedIdx {
		moeSharedIdx[i] = int32(i)
	}
	if err := UploadI32(bs.MoEIdx, moeSharedIdx); err != nil {
		return fmt.Errorf("moe shared index upload: %w", err)
	}
	Barrier()
	if err := ScatterRows(bs.FFNOut, bs.MoEOut, bs.MoEIdx, bs.MoEW, npos, dim, true); err != nil {
		return fmt.Errorf("moe shared gate apply: %w", err)
	}
	return nil
}

// moeSharedGate and moeSharedIdx are scratch for gpuForwardMoESharedBatch, kept
// across layers (and chunks) the way moePlanScratch is. They hold npos entries,
// so they only grow.
var (
	moeSharedGate []float32
	moeSharedIdx  []int32
)

// moePairExpert is the host staging copy of GpuBatchState.MoEPairExpert: a
// pair -> expert id table for the grouped path's per-expert bias add. Only used
// by models whose MoE layers carry expert biases (gpt-oss). Holds npair entries,
// so it only grows.
var moePairExpert []int32

func ensureI32Scratch(buf []int32, n int) []int32 {
	if cap(buf) < n {
		return make([]int32, n)
	}
	return buf[:n]
}

// applyRouterBias adds the router logit bias (gpt-oss ffn_gate_inp.bias) to the
// downloaded logits for the whole chunk.
//
// gpu_forward_moe_ffn adds the same vector on the device with a plain float32
// gpu_add before moe_topk runs, so adding it here is bit-identical: both are one
// IEEE float32 addition of the same two values. Length mismatches are ignored
// because moeBatchSupported already rejects them.
func applyRouterBias(bias, logits []float32, npos, nExperts int) {
	if len(bias) != nExperts || nExperts <= 0 {
		return
	}
	for t := 0; t < npos; t++ {
		lg := logits[t*nExperts : (t+1)*nExperts]
		for i := range lg {
			lg[i] += bias[i]
		}
	}
}

// gpuForwardMoEBatchPerExpert is the position-batched MoE with one GEMM per
// expert. It is the reference implementation and the fallback for quant types
// that have no grouped kernel.
//
// Gate/Up/Hidden come from the dense-FFN scratch, which is unused on MoE layers.
func gpuForwardMoEBatchPerExpert(layer *llm.Layer, gl *GpuLayer, bs *GpuBatchState,
	cfg llm.ModelConfig, npos int) error {

	dim := cfg.EmbeddingDim
	expDim := cfg.ExpertFFNDim
	nExperts := cfg.ExpertCount
	nUsed := cfg.ExpertUsedCount

	strides := moeBatchExpertStrides(layer, gl, cfg)

	// Router for the whole chunk in one dispatch.
	if err := BatchMatVec(bs.MoELogits, gl.FFNRouter.Buf, bs.FFNNorm,
		nExperts, dim, npos, gl.FFNRouter.Type); err != nil {
		return fmt.Errorf("moe batch router: %w", err)
	}
	Barrier()

	p := getMoEPlan(npos, nExperts, nUsed)
	if err := DownloadF32(bs.MoELogits, p.logits); err != nil {
		return fmt.Errorf("moe batch router download: %w", err)
	}
	applyRouterBias(layer.FFNRouterBias, p.logits, npos, nExperts)
	planMoEBatch(p, layer.FFNExpertSelBias, cfg.ExpertGatingFunc,
		cfg.ExpertWeightsNorm, float32(cfg.ExpertWeightsScale))

	// gpu_download has to submit the batch to read the staging buffer, so the
	// recording state is gone here. Re-open it, otherwise every remaining
	// dispatch of this chunk becomes its own submit+wait. No-op if a batch is
	// already open.
	BeginBatch()

	fused := gl.FFNGateUpExps != nil
	gateW := gl.FFNGateExps
	upW := gl.FFNUpExps
	upStride, upBase := strides[1], 0
	if fused {
		// Fused [expert][gate | up]: one stride for both, up at +stride/2.
		gateW, upW = gl.FFNGateUpExps, gl.FFNGateUpExps
		upStride, upBase = strides[0], strides[1]
	}
	if gateW == nil || upW == nil || gl.FFNDownExps == nil {
		return fmt.Errorf("moe batch: missing expert tensors")
	}

	// Accumulate the experts into a zeroed output.
	if err := Scale(bs.FFNOut, 0, dim*npos); err != nil {
		return fmt.Errorf("moe batch zero: %w", err)
	}
	Barrier()

	// gpt-oss carries per-expert projection biases and the OAI SwiGLU variant with
	// clamp + alpha-scaled sigmoid. Both are per-layer constants, so hoist them.
	oai := layer.Spec.FFN == llm.FFNMoESwiOAI
	gateBias, upBias, downBias := gl.FFNGateExpsBias, gl.FFNUpExpsBias, gl.FFNDownExpsBias

	for e := 0; e < nExperts; e++ {
		cnt := int(p.counts[e])
		if cnt == 0 {
			continue
		}
		lo, hi := p.starts[e], p.starts[e+1]
		if err := UploadI32(bs.MoEIdx, p.tokens[lo:hi]); err != nil {
			return fmt.Errorf("moe batch idx upload: %w", err)
		}
		if err := UploadF32(bs.MoEW, p.weight[lo:hi]); err != nil {
			return fmt.Errorf("moe batch weight upload: %w", err)
		}
		Barrier()

		if err := GatherRows(bs.MoEXE, bs.FFNNorm, bs.MoEIdx, cnt, dim); err != nil {
			return fmt.Errorf("moe batch gather: %w", err)
		}
		Barrier()

		if err := BatchMatVecOffset(bs.Gate, gateW.Buf, bs.MoEXE, expDim, dim, cnt,
			gateW.Type, uint64(e)*uint64(strides[0])); err != nil {
			return fmt.Errorf("moe batch gate (type=%d rows=%d cols=%d stride=%d off=%d): %w",
				gateW.Type, expDim, dim, strides[0], uint64(e)*uint64(strides[0]), err)
		}
		if err := BatchMatVecOffset(bs.Up, upW.Buf, bs.MoEXE, expDim, dim, cnt,
			upW.Type, uint64(e)*uint64(upStride)+uint64(upBase)); err != nil {
			return fmt.Errorf("moe batch up: %w", err)
		}
		if gateBias != 0 {
			Barrier()
			if err := MoEExpertBiasAdd(bs.Gate, gateBias, bs.MoEIdx, expDim*cnt, expDim,
				false, e*expDim); err != nil {
				return fmt.Errorf("moe batch gate bias: %w", err)
			}
		}
		if upBias != 0 {
			Barrier()
			if err := MoEExpertBiasAdd(bs.Up, upBias, bs.MoEIdx, expDim*cnt, expDim,
				false, e*expDim); err != nil {
				return fmt.Errorf("moe batch up bias: %w", err)
			}
		}
		Barrier()

		if oai {
			if err := SwiGLU_OAI(bs.Hidden, bs.Gate, bs.Up, expDim*cnt, 1.702, 7.0); err != nil {
				return fmt.Errorf("moe batch swiglu_oai: %w", err)
			}
		} else if err := SwiGLU(bs.Hidden, bs.Gate, bs.Up, expDim*cnt); err != nil {
			return fmt.Errorf("moe batch swiglu: %w", err)
		}
		Barrier()

		if err := BatchMatVecOffset(bs.MoEOut, gl.FFNDownExps.Buf, bs.Hidden, dim, expDim,
			cnt, gl.FFNDownExps.Type, uint64(e)*uint64(strides[2])); err != nil {
			return fmt.Errorf("moe batch down: %w", err)
		}
		if downBias != 0 {
			Barrier()
			if err := MoEExpertBiasAdd(bs.MoEOut, downBias, bs.MoEIdx, dim*cnt, dim,
				false, e*dim); err != nil {
				return fmt.Errorf("moe batch down bias: %w", err)
			}
		}
		Barrier()

		if err := ScatterRows(bs.FFNOut, bs.MoEOut, bs.MoEIdx, bs.MoEW, cnt, dim, true); err != nil {
			return fmt.Errorf("moe batch scatter: %w", err)
		}
		Barrier()
	}
	return nil
}

// ---------------------------------------------------------------------------
// Grouped MoE: one dispatch per projection covering every expert.
// ---------------------------------------------------------------------------

// moeGroupedQTypeOK reports whether a grouped MoE GEMM kernel exists for a
// weight type. Only the canonical Affine formats have one: they are what the
// default unified-quant path repacks every weight tensor into, so that is where
// the win is. Native per-quant types keep the per-expert loop, which is already
// correct and only one kernel per type away if it ever matters.
func moeGroupedQTypeOK(t uint32) bool {
	return t == QuantTypeAffine4 || t == QuantTypeAffine8
}

// moeAffineBlockBytes returns the byte size of one canonical Affine block (32
// weights), or 0 for any other type.
func moeAffineBlockBytes(t uint32) int {
	switch t {
	case QuantTypeAffine8:
		return 36
	case QuantTypeAffine4:
		return 20
	}
	return 0
}

// moeGroupedForce is a test hook for the end-to-end parity tests, which need both
// implementations inside one process: 0 forces the per-expert loop, any other
// value leaves the normal decision alone.
var moeGroupedForce = -1

// moeGroupedEnvOn caches DLGO_MOE_GROUPED. moeGroupedSupported runs once per MoE
// layer per chunk, so the lookup is cached rather than repeated 38 times a chunk.
var moeGroupedEnvOn = struct {
	once sync.Once
	on   bool
}{}

func moeGroupedEnabled() bool {
	moeGroupedEnvOn.once.Do(func() {
		moeGroupedEnvOn.on = os.Getenv("DLGO_MOE_GROUPED") != "0"
	})
	return moeGroupedEnvOn.on
}

// moeGroupedSupported reports whether the grouped path can run this layer.
//
// Besides the path's own buffer requirements it reuses moeBatchSupported, so
// every condition the per-expert loop needs (no shared expert, no expert biases,
// a batched router, an aligned expert stride) holds for both — the grouped path
// is a faster way to compute the same thing, not a second semantics.
func moeGroupedSupported(layer *llm.Layer, gl *GpuLayer, cfg llm.ModelConfig, bs *GpuBatchState) bool {
	if bs == nil || bs.MoEGate == 0 || bs.MoEUp == 0 || bs.MoEPairOut == 0 ||
		bs.MoEStarts == 0 || bs.MoETokIdx == 0 || bs.MoEPairIdx == 0 ||
		bs.MoESlotPair == 0 || bs.MoESlotW == 0 {
		return false
	}
	if moeGroupedForce == 0 {
		return false
	}
	if !moeBatchSupported(layer, gl, cfg) {
		return false
	}
	if !moeGroupedEnabled() {
		return false
	}
	// BK == 32 is exactly one Affine block, so the k-tile boundary and the
	// per-row block count only line up when both dims are block-multiples.
	if cfg.EmbeddingDim%32 != 0 || cfg.ExpertFFNDim%32 != 0 {
		return false
	}
	types := []uint32{gl.FFNDownExps.Type}
	if gl.FFNGateUpExps != nil {
		types = append(types, gl.FFNGateUpExps.Type)
	} else {
		if gl.FFNGateExps == nil || gl.FFNUpExps == nil {
			return false
		}
		types = append(types, gl.FFNGateExps.Type, gl.FFNUpExps.Type)
	}
	for _, t := range types {
		if !moeGroupedQTypeOK(t) {
			return false
		}
	}
	return true
}

// gpuForwardMoEGrouped runs one MoE layer for a chunk with one GEMM dispatch per
// projection covering every expert.
//
// This removes ~3*nExperts dispatches per layer (192 on a 64-expert layer) and,
// more importantly, the reason each expert's weights were read once per
// (token, expert) pair: every expert now sweeps its slice a single time for all
// of its tokens. The gather/scatter pair disappears too — activations are read
// through the routing table by the GEMM's B-side, and the down projection writes
// pair-major, so moe_slot_reduce can do the weighted top-k sum without atomics
// (which is what keeps a chunk bit-reproducible).
//
// Expert slices are addressed as Affine *block* indices (a_blocks stride, a_base
// offset), so unlike the descriptor-offset form of the per-expert loop they need
// no storage-buffer alignment.
func gpuForwardMoEGrouped(layer *llm.Layer, gl *GpuLayer, bs *GpuBatchState,
	cfg llm.ModelConfig, npos int) error {

	dim := cfg.EmbeddingDim
	expDim := cfg.ExpertFFNDim
	nExperts := cfg.ExpertCount
	nUsed := cfg.ExpertUsedCount
	npair := npos * nUsed

	// Router for the whole chunk in one dispatch.
	if err := BatchMatVec(bs.MoELogits, gl.FFNRouter.Buf, bs.FFNNorm,
		nExperts, dim, npos, gl.FFNRouter.Type); err != nil {
		return fmt.Errorf("moe grouped router: %w", err)
	}
	Barrier()

	p := getMoEPlan(npos, nExperts, nUsed)
	if err := DownloadF32(bs.MoELogits, p.logits); err != nil {
		return fmt.Errorf("moe grouped router download: %w", err)
	}
	applyRouterBias(layer.FFNRouterBias, p.logits, npos, nExperts)
	planMoEBatch(p, layer.FFNExpertSelBias, cfg.ExpertGatingFunc,
		cfg.ExpertWeightsNorm, float32(cfg.ExpertWeightsScale))

	// gpu_download has to submit the batch to read the staging buffer, so the
	// recording state is gone at this point. Re-open it: otherwise the routing
	// tables below and every dispatch for the rest of the chunk becomes its own
	// submit+wait, which drains the GPU between dispatches. BeginBatch is a no-op
	// when a batch is already open, so this is safe from any caller.
	BeginBatch()

	// Routing tables: pair -> token (so the GEMMs can read the activations), the
	// per-expert pair range, and the slot-major inverse the reduce needs.
	if err := UploadI32(bs.MoEStarts, p.starts[:nExperts+1]); err != nil {
		return fmt.Errorf("moe grouped starts upload: %w", err)
	}
	if err := UploadI32(bs.MoETokIdx, p.tokens[:npair]); err != nil {
		return fmt.Errorf("moe grouped token idx upload: %w", err)
	}
	if err := UploadI32(bs.MoESlotPair, p.pair[:npair]); err != nil {
		return fmt.Errorf("moe grouped slot pair upload: %w", err)
	}
	if err := UploadF32(bs.MoESlotW, p.selW[:npair]); err != nil {
		return fmt.Errorf("moe grouped slot weight upload: %w", err)
	}
	// Per-expert biases need a pair -> expert table: the grouped activations are
	// pair-major, so a row's bias row is not a function of its index alone.
	oai := layer.Spec.FFN == llm.FFNMoESwiOAI
	gateBias, upBias, downBias := gl.FFNGateExpsBias, gl.FFNUpExpsBias, gl.FFNDownExpsBias
	needPairExpert := gateBias != 0 || upBias != 0 || downBias != 0
	if needPairExpert {
		pe := ensureI32Scratch(moePairExpert, npair)
		for e := 0; e < nExperts; e++ {
			for i := int(p.starts[e]); i < int(p.starts[e+1]); i++ {
				pe[i] = int32(e)
			}
		}
		moePairExpert = pe
		if err := UploadI32(bs.MoEPairExpert, pe[:npair]); err != nil {
			return fmt.Errorf("moe grouped pair-expert upload: %w", err)
		}
	}
	Barrier()

	fused := gl.FFNGateUpExps != nil
	gateW, upW := gl.FFNGateExps, gl.FFNUpExps
	if fused {
		// Fused [expert][gate | up]: one tensor, up at +expertStride/2.
		gateW, upW = gl.FFNGateUpExps, gl.FFNGateUpExps
	}
	downW := gl.FFNDownExps
	if gateW == nil || upW == nil || downW == nil {
		return fmt.Errorf("moe grouped: missing expert tensors")
	}
	gateBlk := moeAffineBlockBytes(gateW.Type)
	upBlk := moeAffineBlockBytes(upW.Type)
	downBlk := moeAffineBlockBytes(downW.Type)
	if gateBlk == 0 || upBlk == 0 || downBlk == 0 {
		return fmt.Errorf("moe grouped: unsupported quant types gate=%d up=%d down=%d",
			gateW.Type, upW.Type, downW.Type)
	}

	// Derive the block strides from the same byte strides the per-expert path
	// uses, so both paths cannot disagree about the layout.
	strides := moeBatchExpertStrides(layer, gl, cfg)
	blocksOf := func(name string, idx, blk int) (int, error) {
		if strides[idx] <= 0 || strides[idx]%blk != 0 {
			return 0, fmt.Errorf("moe grouped: %s expert stride %d is not a multiple of %d",
				name, strides[idx], blk)
		}
		return strides[idx] / blk, nil
	}
	gateBlocks, err := blocksOf("gate", 0, gateBlk)
	if err != nil {
		return err
	}
	upBlocks, upBase := 0, 0
	if fused {
		// Same tensor and therefore the same stride, with the up half offset.
		upBlocks = gateBlocks
		upBase, err = blocksOf("up", 1, upBlk)
		if err != nil {
			return err
		}
	} else {
		upBlocks, err = blocksOf("up", 1, upBlk)
		if err != nil {
			return err
		}
	}
	downBlocks, err := blocksOf("down", 2, downBlk)
	if err != nil {
		return err
	}

	gemm := func(out, weights, idx, x Buf, qtype uint32, rows, cols, blocks, base int) error {
		if qtype == QuantTypeAffine8 {
			return GemmAffine8MoEGrouped(out, weights, x, idx, bs.MoEStarts,
				rows, cols, npos, nExperts, blocks, base)
		}
		return GemmAffine4MoEGrouped(out, weights, x, idx, bs.MoEStarts,
			rows, cols, npos, nExperts, blocks, base)
	}

	// gate / up: [npair][expDim] each, over the normed activations of the whole
	// chunk. The B-side rows come from the routing table, so no gathered copy of
	// the activations is needed.
	if err := gemm(bs.MoEGate, gateW.Buf, bs.MoETokIdx, bs.FFNNorm, gateW.Type,
		expDim, dim, gateBlocks, 0); err != nil {
		return fmt.Errorf("moe grouped gate (type=%d blocks=%d): %w", gateW.Type, gateBlocks, err)
	}
	if err := gemm(bs.MoEUp, upW.Buf, bs.MoETokIdx, bs.FFNNorm, upW.Type,
		expDim, dim, upBlocks, upBase); err != nil {
		return fmt.Errorf("moe grouped up (type=%d blocks=%d base=%d): %w",
			upW.Type, upBlocks, upBase, err)
	}
	if gateBias != 0 {
		Barrier()
		if err := MoEExpertBiasAdd(bs.MoEGate, gateBias, bs.MoEPairExpert, expDim*npair,
			expDim, true, 0); err != nil {
			return fmt.Errorf("moe grouped gate bias: %w", err)
		}
	}
	if upBias != 0 {
		Barrier()
		if err := MoEExpertBiasAdd(bs.MoEUp, upBias, bs.MoEPairExpert, expDim*npair,
			expDim, true, 0); err != nil {
			return fmt.Errorf("moe grouped up bias: %w", err)
		}
	}
	Barrier()

	// SwiGLU over the pair axis, in place into MoEGate (each thread reads before
	// writing its own index). MoEGate then holds the down projection's input.
	if oai {
		if err := SwiGLU_OAI(bs.MoEGate, bs.MoEGate, bs.MoEUp, expDim*npair, 1.702, 7.0); err != nil {
			return fmt.Errorf("moe grouped swiglu_oai: %w", err)
		}
	} else if err := SwiGLU(bs.MoEGate, bs.MoEGate, bs.MoEUp, expDim*npair); err != nil {
		return fmt.Errorf("moe grouped swiglu: %w", err)
	}
	Barrier()

	// The down projection's input is already pair-major, so its index table is the
	// identity 0..npair-1 (filled once at allocation).
	if err := gemm(bs.MoEPairOut, downW.Buf, bs.MoEPairIdx, bs.MoEGate, downW.Type,
		dim, expDim, downBlocks, 0); err != nil {
		return fmt.Errorf("moe grouped down (type=%d blocks=%d): %w",
			downW.Type, downBlocks, err)
	}
	if downBias != 0 {
		Barrier()
		if err := MoEExpertBiasAdd(bs.MoEPairOut, downBias, bs.MoEPairExpert, dim*npair,
			dim, true, 0); err != nil {
			return fmt.Errorf("moe grouped down bias: %w", err)
		}
	}
	Barrier()

	// Weighted top-k sum in slot order. Deterministic, so no atomics anywhere in
	// this path and a chunk reproduces itself exactly.
	if err := MoESlotReduce(bs.FFNOut, bs.MoEPairOut, bs.MoESlotPair, bs.MoESlotW,
		dim, nUsed, npos); err != nil {
		return fmt.Errorf("moe grouped slot reduce: %w", err)
	}
	return nil
}
