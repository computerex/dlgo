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
	// Not batched yet: shared expert, expert biases, router bias. None of these
	// occur on LFM2/SmolLM-style MoE, so falling back keeps the surface small.
	if gl.FFNGateShared != nil || gl.FFNRouterShared != 0 {
		return false
	}
	if gl.FFNRouterBias != 0 {
		return false
	}
	if gl.FFNGateExpsBias != 0 || gl.FFNUpExpsBias != 0 || gl.FFNDownExpsBias != 0 {
		return false
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
	if moeGroupedSupported(layer, gl, cfg, bs) {
		return gpuForwardMoEGrouped(layer, gl, bs, cfg, npos)
	}
	return gpuForwardMoEBatchPerExpert(layer, gl, bs, cfg, npos)
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
	planMoEBatch(p, layer.FFNExpertSelBias, cfg.ExpertGatingFunc,
		cfg.ExpertWeightsNorm, float32(cfg.ExpertWeightsScale))

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
		Barrier()

		if err := SwiGLU(bs.Hidden, bs.Gate, bs.Up, expDim*cnt); err != nil {
			return fmt.Errorf("moe batch swiglu: %w", err)
		}
		Barrier()

		if err := BatchMatVecOffset(bs.MoEOut, gl.FFNDownExps.Buf, bs.Hidden, dim, expDim,
			cnt, gl.FFNDownExps.Type, uint64(e)*uint64(strides[2])); err != nil {
			return fmt.Errorf("moe batch down: %w", err)
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
	planMoEBatch(p, layer.FFNExpertSelBias, cfg.ExpertGatingFunc,
		cfg.ExpertWeightsNorm, float32(cfg.ExpertWeightsScale))

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
	Barrier()

	// SwiGLU over the pair axis, in place into MoEGate (each thread reads before
	// writing its own index). MoEGate then holds the down projection's input.
	if err := SwiGLU(bs.MoEGate, bs.MoEGate, bs.MoEUp, expDim*npair); err != nil {
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
	Barrier()

	// Weighted top-k sum in slot order. Deterministic, so no atomics anywhere in
	// this path and a chunk reproduces itself exactly.
	if err := MoESlotReduce(bs.FFNOut, bs.MoEPairOut, bs.MoESlotPair, bs.MoESlotW,
		dim, nUsed, npos); err != nil {
		return fmt.Errorf("moe grouped slot reduce: %w", err)
	}
	return nil
}
