package llm

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/format/gguf"
	"github.com/computerex/dlgo/mmap"
	"github.com/computerex/dlgo/quant"
)

// LoadModel opens a GGUF file, parses config from metadata, and loads all tensors.
func LoadModel(path string) (*Model, error) {
	gf, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("parse GGUF: %w", err)
	}

	cfg, err := parseConfig(gf.Metadata)
	if err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Fix embed scale for Gemma
	if cfg.EmbedScale > 0 {
		cfg.EmbedScale = float32(math.Sqrt(float64(cfg.EmbeddingDim)))
	}

	m := &Model{
		Config: cfg,
		Layers: make([]Layer, cfg.NumLayers),
	}

	// Memory-map the GGUF file so tensor data is backed by the OS page cache.
	// Weights are accessed directly from the mmap'd region — no heap copies.
	// Models larger than physical RAM work transparently via demand paging.
	mf, err := mmap.Open(path)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}
	m.MmapFile = mf

	// Precompute derived dimensions for fused QKV splitting
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim

	for _, ti := range gf.Tensors {
		totalElements := int64(1)
		for _, d := range ti.Dimensions {
			totalElements *= d
		}
		nbytes := int64(quant.BytesForN(uint32(ti.Type), int(totalElements)))
		offset := gf.DataOffset + int64(ti.Offset)
		data := mf.Slice(offset, nbytes)

		rows, cols := inferRowsCols(ti.Dimensions)

		if isNormOrBias(ti.Name) {
			fp32 := dequantToF32(data, uint32(ti.Type), int(totalElements))
			mapTensorF32(m, ti.Name, fp32)
		} else {
			qt := &core.QuantizedTensor{
				Data: data,
				Type: uint32(ti.Type),
				Rows: rows,
				Cols: cols,
			}
			mapTensorQT(m, ti.Name, qt, qDim, kvDim, cfg)
		}
	}

	// Validate all tensor quant types have zero-alloc dequantize paths.
	// Types that fall through to the allocating default are dangerous when
	// GC is suppressed during inference.
	{
		unsupported := map[uint32]bool{}
		for _, ti := range gf.Tensors {
			if !isNormOrBias(ti.Name) && !quant.HasZeroAllocDequantize(uint32(ti.Type)) {
				unsupported[uint32(ti.Type)] = true
			}
		}
		for t := range unsupported {
			fmt.Fprintf(os.Stderr, "[dlgo] WARNING: model uses quant type %d which lacks a zero-alloc "+
				"dequantizer — inference may use excessive memory\n", t)
		}
	}

	// Weight tying: if output projection is nil, share with token embeddings
	if m.Output == nil {
		m.Output = m.TokenEmbed
	}

	// Split fused BA tensors into separate SSMAlpha/SSMBeta for GPU compatibility.
	// The fused tensor has interleaved columns: [beta_g0..., alpha_g0..., beta_g1..., alpha_g1...]
	// per KV group. We dequantize and rearrange into two FP32 matrices.
	for i := range m.Layers {
		l := &m.Layers[i]
		if l.SSMFusedBA != nil && l.SSMAlpha == nil && l.SSMBeta == nil {
			splitSSMFusedBA(l, cfg)
		}
	}

	// Resolve per-layer specs from loaded tensor presence
	for i := range m.Layers {
		m.Layers[i].Spec = resolveLayerSpec(&m.Layers[i], cfg, i)
	}

	return m, nil
}

// resolveLayerSpec infers all architectural choices for a layer from its loaded
// weights. Called once at load time so the forward pass can dispatch via switch.
func resolveLayerSpec(l *Layer, cfg ModelConfig, layerIdx int) LayerSpec {
	var s LayerSpec

	if l.AttnNormBias != nil {
		s.Norm = NormLayer
	} else {
		s.Norm = NormRMS
	}

	if isSSMLayer(layerIdx, cfg) && l.SSMInProj != nil {
		s.Core = CoreSSM
	} else if l.WqA != nil && l.WqB != nil {
		s.Core = CoreMLA
	} else {
		s.Core = CoreAttention
	}

	if l.FFNNorm != nil {
		s.Residual = ResStandard
	} else if l.PostAttnNorm != nil {
		s.Residual = ResPostAttnFFN
	} else {
		s.Residual = ResParallel
	}

	if l.FFNRouter != nil && (l.FFNGateExps != nil || l.FFNGateUpExps != nil) {
		if cfg.Architecture == "gpt-oss" {
			s.FFN = FFNMoESwiOAI
		} else {
			s.FFN = FFNMoE
		}
	} else if l.FFNGate != nil {
		if cfg.FFNGelu {
			s.FFN = FFNGeGLU
		} else {
			s.FFN = FFNSwiGLU
		}
	} else {
		s.FFN = FFNPlain
	}

	s.GatedQ = l.Wq != nil && l.Wq.Rows > cfg.NumHeads*cfg.HeadDim
	s.QKNorm = l.AttnQNorm != nil

	// Sliding window attention: apply to alternating layers if pattern is set
	if cfg.SlidingWindow > 0 && s.Core == CoreAttention {
		if cfg.SlidingWindowPattern <= 0 {
			// All attention layers use sliding window
			s.SlidingWindow = cfg.SlidingWindow
		} else {
			// Pattern: every Nth layer is full attention, others use sliding window
			if ((layerIdx + 1) % cfg.SlidingWindowPattern) != 0 {
				s.SlidingWindow = cfg.SlidingWindow
			}
		}
	}

	return s
}

// PinLayerToRAM copies a layer's weight data from the mmap'd region
// into heap-allocated slices, ensuring the layer stays in physical RAM
// and avoids page faults during inference.
func PinLayerToRAM(l *Layer) {
	pinTensor := func(qt *core.QuantizedTensor) {
		if qt == nil || len(qt.Data) == 0 {
			return
		}
		pinned := make([]byte, len(qt.Data))
		copy(pinned, qt.Data)
		qt.Data = pinned
	}
	pinTensor(l.Wq)
	pinTensor(l.Wk)
	pinTensor(l.Wv)
	pinTensor(l.Wo)
	pinTensor(l.AttnGate)
	pinTensor(l.WqA)
	pinTensor(l.WqB)
	pinTensor(l.WkvA)
	pinTensor(l.WkB)
	pinTensor(l.WvB)
	pinTensor(l.FFNGate)
	pinTensor(l.FFNUp)
	pinTensor(l.FFNDown)
	pinTensor(l.SSMInProj)
	pinTensor(l.SSMAlpha)
	pinTensor(l.SSMBeta)
	pinTensor(l.SSMOut)
	pinTensor(l.FFNRouter)
	pinTensor(l.FFNGateExps)
	pinTensor(l.FFNUpExps)
	pinTensor(l.FFNGateUpExps)
	pinTensor(l.FFNDownExps)
	pinTensor(l.FFNGateShared)
	pinTensor(l.FFNUpShared)
	pinTensor(l.FFNDownShared)
}

// EstimateLayerBytes returns the approximate size in bytes of a layer's weight data.
func EstimateLayerBytes(l *Layer) int64 {
	var total int64
	add := func(qt *core.QuantizedTensor) {
		if qt != nil {
			total += int64(len(qt.Data))
		}
	}
	add(l.Wq)
	add(l.Wk)
	add(l.Wv)
	add(l.Wo)
	add(l.AttnGate)
	add(l.WqA)
	add(l.WqB)
	add(l.WkvA)
	add(l.WkB)
	add(l.WvB)
	add(l.FFNGate)
	add(l.FFNUp)
	add(l.FFNDown)
	add(l.SSMInProj)
	add(l.SSMAlpha)
	add(l.SSMBeta)
	add(l.SSMOut)
	add(l.FFNRouter)
	add(l.FFNGateExps)
	add(l.FFNUpExps)
	add(l.FFNGateUpExps)
	add(l.FFNDownExps)
	add(l.FFNGateShared)
	add(l.FFNUpShared)
	add(l.FFNDownShared)
	return total
}

func inferRowsCols(dims []int64) (int, int) {
	if len(dims) == 0 {
		return 1, 1
	}
	if len(dims) == 1 {
		return int(dims[0]), 1
	}
	if len(dims) == 3 {
		// 3D tensor: [cols, inner_dim, num_experts] → rows = inner_dim*num_experts, cols = cols
		return int(dims[1]) * int(dims[2]), int(dims[0])
	}
	// GGUF stores [cols, rows] (reversed from row-major convention)
	return int(dims[len(dims)-1]), int(dims[0])
}

// isNormOrBias returns true if the tensor name indicates a norm weight, bias,
// or other small 1D parameter that should be stored as dequantized float32.
func isNormOrBias(name string) bool {
	return strings.HasSuffix(name, "_norm.weight") ||
		strings.HasSuffix(name, ".bias") ||
		strings.HasSuffix(name, "_norm.bias") ||
		strings.HasSuffix(name, "ssm_a") ||
		strings.HasSuffix(name, "ssm_conv1d.weight") ||
		strings.HasSuffix(name, "ffn_gate_inp_shexp.weight") ||
		strings.HasSuffix(name, "attn_sinks.weight")
}

func dequantToF32(data []byte, ggmlType uint32, n int) []float32 {
	result, _ := quant.Dequantize(data, ggmlType, n)
	return result
}

func mapTensorF32(m *Model, name string, data []float32) {
	switch {
	case name == "output_norm.weight":
		m.OutputNorm = data
	case name == "output_norm.bias":
		m.OutputNormBias = data
	case name == "output.bias":
		m.OutputBias = data
	default:
		if layerIdx, field := parseLayerName(name); layerIdx >= 0 && layerIdx < len(m.Layers) {
			l := &m.Layers[layerIdx]
			switch field {
			case "attn_norm.weight":
				l.AttnNorm = data
			case "attn_norm.bias":
				l.AttnNormBias = data
			case "ffn_norm.weight":
				l.FFNNorm = data
			case "attn_q.bias":
				l.Bq = data
			case "attn_k.bias":
				l.Bk = data
			case "attn_v.bias":
				l.Bv = data
			case "attn_qkv.bias":
				qDim := m.Config.NumHeads * m.Config.HeadDim
				kvDim := m.Config.NumKVHeads * m.Config.HeadDim
				l.Bq = data[:qDim]
				l.Bk = data[qDim : qDim+kvDim]
				l.Bv = data[qDim+kvDim : qDim+2*kvDim]
			case "attn_output.bias":
				l.Bo = data
			case "attn_q_norm.weight":
				l.AttnQNorm = data
			case "attn_k_norm.weight":
				l.AttnKNorm = data
			case "attn_q_a_norm.weight":
				l.WqANorm = data
			case "attn_kv_a_norm.weight":
				l.WkvANorm = data
			case "post_attention_norm.weight":
				l.PostAttnNorm = data
			case "post_ffw_norm.weight":
				l.PostFFNNorm = data
			case "ffn_up.bias":
				l.FFNUpBias = data
			case "ffn_down.bias":
				l.FFNDownBias = data
			case "ssm_dt.bias":
				l.SSMDtBias = data
			case "ssm_norm.weight":
				l.SSMNorm = data
			case "ssm_a":
				l.SSMA = data
		case "ssm_conv1d.weight":
			l.SSMConv1dW = data
		case "ffn_gate_inp_shexp.weight":
			l.FFNRouterShared = data
		case "exp_probs_b.bias":
			l.FFNRouterBias = data
		case "attn_sinks.weight":
			l.AttnSinks = data
		case "ffn_gate_inp.bias":
			l.FFNRouterBias = data
		case "ffn_gate_exps.bias":
			l.FFNGateExpsBias = data
		case "ffn_up_exps.bias":
			l.FFNUpExpsBias = data
		case "ffn_down_exps.bias":
			l.FFNDownExpsBias = data
		}
		}
	}
}

func mapTensorQT(m *Model, name string, qt *core.QuantizedTensor, qDim, kvDim int, cfg ModelConfig) {
	switch {
	case name == "token_embd.weight":
		m.TokenEmbed = qt
	case name == "output.weight":
		m.Output = qt
	default:
		if layerIdx, field := parseLayerName(name); layerIdx >= 0 && layerIdx < len(m.Layers) {
			l := &m.Layers[layerIdx]
			switch field {
			case "attn_q.weight":
				l.Wq = qt
			case "attn_k.weight":
				l.Wk = qt
			case "attn_v.weight":
				l.Wv = qt
			case "attn_qkv.weight":
				splitFusedQKV(l, qt, qDim, kvDim, cfg.EmbeddingDim)
			case "attn_output.weight":
				l.Wo = qt
			case "attn_gate.weight":
				l.AttnGate = qt
			case "ffn_gate.weight":
				l.FFNGate = qt
			case "ffn_up.weight":
				splitFusedFFNUp(l, qt, cfg.FFNDim, cfg.EmbeddingDim)
			case "ffn_down.weight":
				l.FFNDown = qt
		case "attn_q_a.weight":
			l.WqA = qt
		case "attn_q_b.weight":
			l.WqB = qt
		case "attn_kv_a_mqa.weight":
			l.WkvA = qt
		case "attn_k_b.weight":
			l.WkB = qt
		case "attn_v_b.weight":
			l.WvB = qt
		case "ssm_alpha.weight":
			l.SSMAlpha = qt
		case "ssm_beta.weight":
			l.SSMBeta = qt
		case "ssm_ba.weight":
			l.SSMFusedBA = qt
		case "ssm_out.weight":
			l.SSMOut = qt
		// MoE tensors
		case "ffn_gate_inp.weight":
			l.FFNRouter = qt
		case "ffn_gate_exps.weight":
			l.FFNGateExps = qt
		case "ffn_up_exps.weight":
			l.FFNUpExps = qt
		case "ffn_gate_up_exps.weight":
			l.FFNGateUpExps = qt
		case "ffn_down_exps.weight":
			l.FFNDownExps = qt
		case "ffn_gate_shexp.weight":
			l.FFNGateShared = qt
		case "ffn_up_shexp.weight":
			l.FFNUpShared = qt
		case "ffn_down_shexp.weight":
			l.FFNDownShared = qt
		}
	}
	}
}

// splitFusedQKV splits a fused [Q|K|V] weight tensor into separate Wq, Wk, Wv.
// Handles three cases:
//   - qDim + 2*kvDim: standard attention split
//   - 2*qDim + 2*kvDim: GatedQ attention (Q+gate interleaved, doubled)
//   - other: SSM/delta-net in-projection (stored as SSMInProj)
func splitFusedQKV(l *Layer, qt *core.QuantizedTensor, qDim, kvDim, cols int) {
	expected := qDim + 2*kvDim
	gatedExpected := 2*qDim + 2*kvDim

	bytesPerRow := quant.BytesForN(qt.Type, cols)

	if qt.Rows == expected {
		qBytes := qDim * bytesPerRow
		kvBytes := kvDim * bytesPerRow
		l.Wq = &core.QuantizedTensor{Data: qt.Data[:qBytes], Type: qt.Type, Rows: qDim, Cols: cols}
		l.Wk = &core.QuantizedTensor{Data: qt.Data[qBytes : qBytes+kvBytes], Type: qt.Type, Rows: kvDim, Cols: cols}
		l.Wv = &core.QuantizedTensor{Data: qt.Data[qBytes+kvBytes : qBytes+2*kvBytes], Type: qt.Type, Rows: kvDim, Cols: cols}
	} else if qt.Rows == gatedExpected {
		gatedQDim := 2 * qDim
		qBytes := gatedQDim * bytesPerRow
		kvBytes := kvDim * bytesPerRow
		l.Wq = &core.QuantizedTensor{Data: qt.Data[:qBytes], Type: qt.Type, Rows: gatedQDim, Cols: cols}
		l.Wk = &core.QuantizedTensor{Data: qt.Data[qBytes : qBytes+kvBytes], Type: qt.Type, Rows: kvDim, Cols: cols}
		l.Wv = &core.QuantizedTensor{Data: qt.Data[qBytes+kvBytes : qBytes+2*kvBytes], Type: qt.Type, Rows: kvDim, Cols: cols}
	} else {
		l.SSMInProj = qt
	}
}

// splitFusedFFNUp splits a fused [gate|up] weight tensor if it has 2x expected rows.
func splitFusedFFNUp(l *Layer, qt *core.QuantizedTensor, ffnDim, cols int) {
	if qt.Rows == 2*ffnDim {
		bytesPerRow := quant.BytesForN(qt.Type, cols)
		halfBytes := ffnDim * bytesPerRow
		l.FFNGate = &core.QuantizedTensor{Data: qt.Data[:halfBytes], Type: qt.Type, Rows: ffnDim, Cols: cols}
		l.FFNUp = &core.QuantizedTensor{Data: qt.Data[halfBytes : 2*halfBytes], Type: qt.Type, Rows: ffnDim, Cols: cols}
	} else {
		l.FFNUp = qt
	}
}

// splitSSMFusedBA dequantizes the fused BA weight matrix and splits it into
// separate SSMAlpha and SSMBeta FP32 matrices. The fused tensor's columns are
// interleaved per KV group: [beta_g0_v0..vN, alpha_g0_v0..vN, beta_g1_..., ...]
func splitSSMFusedBA(l *Layer, cfg ModelConfig) {
	fused := l.SSMFusedBA
	dim := fused.Cols // input dimension
	numHeads := cfg.SSMTimeStepRank
	numKVGroups := cfg.SSMGroupCount
	if numKVGroups <= 0 {
		numKVGroups = numHeads
	}
	vPerGroup := numHeads / numKVGroups

	alphaData := make([]float32, numHeads*dim)
	betaData := make([]float32, numHeads*dim)
	rowBuf := make([]float32, fused.Cols)

	for row := 0; row < fused.Rows; row++ {
		_ = fused.DequantizeRow(row, rowBuf)
		// Row layout: [beta_g0_v0..vN, alpha_g0_v0..vN, beta_g1_..., ...]
		// Determine which output head/type this row maps to
		g := row / (vPerGroup * 2)
		posInGroup := row % (vPerGroup * 2)
		if posInGroup < vPerGroup {
			headIdx := g*vPerGroup + posInGroup
			copy(betaData[headIdx*dim:(headIdx+1)*dim], rowBuf)
		} else {
			headIdx := g*vPerGroup + (posInGroup - vPerGroup)
			copy(alphaData[headIdx*dim:(headIdx+1)*dim], rowBuf)
		}
	}

	l.SSMAlpha = &core.QuantizedTensor{FP32Data: alphaData, Type: 0, Rows: numHeads, Cols: dim}
	l.SSMBeta = &core.QuantizedTensor{FP32Data: betaData, Type: 0, Rows: numHeads, Cols: dim}
	l.SSMFusedBA = nil
}

// parseLayerName extracts layer index and field name from "blk.{i}.{field}" patterns.
func parseLayerName(name string) (int, string) {
	if !strings.HasPrefix(name, "blk.") {
		return -1, ""
	}
	rest := name[4:]
	dotIdx := strings.IndexByte(rest, '.')
	if dotIdx < 0 {
		return -1, ""
	}
	idx, err := strconv.Atoi(rest[:dotIdx])
	if err != nil {
		return -1, ""
	}
	return idx, rest[dotIdx+1:]
}
