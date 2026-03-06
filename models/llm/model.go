package llm

import (
	"github.com/computerex/dlgo/core"
)

// Layer holds the weights for one transformer block.
type Layer struct {
	// Attention
	AttnNorm     []float32            // [dim] norm weight
	AttnNormBias []float32            // [dim] optional LayerNorm bias (Phi-2)
	Wq           *core.QuantizedTensor // [qDim × dim]
	Wk           *core.QuantizedTensor // [kvDim × dim]
	Wv           *core.QuantizedTensor // [kvDim × dim]
	Wo           *core.QuantizedTensor // [dim × qDim]
	Bq           []float32            // [qDim] optional (Qwen)
	Bk           []float32            // [kvDim] optional
	Bv           []float32            // [kvDim] optional
	Bo           []float32            // [dim] optional attn_output bias (Phi-2)
	AttnQNorm    []float32            // [headDim] optional QK norm (Qwen3/Gemma3)
	AttnKNorm    []float32            // [headDim] optional QK norm (Qwen3/Gemma3)
	AttnGate     *core.QuantizedTensor // [dim × dim] optional gated attention (Qwen3.5)

	PostAttnNorm []float32            // [dim] optional post-attention norm (Gemma 3)
	FFNNorm      []float32            // [dim] norm weight (nil = parallel attn+FFN)
	FFNGate      *core.QuantizedTensor // [ffnDim × dim] w1 (nil = plain MLP)
	FFNUp        *core.QuantizedTensor // [ffnDim × dim] w3
	FFNDown      *core.QuantizedTensor // [dim × ffnDim] w2
	FFNUpBias    []float32            // [ffnDim] optional (Phi-2)
	FFNDownBias  []float32            // [dim] optional (Phi-2)
	PostFFNNorm  []float32            // [dim] optional post-FFN norm (Gemma 3)

	// Gated Delta Net weights — only for Qwen3.5 linear attention layers
	SSMInProj  *core.QuantizedTensor // [dim × qkvDim] fused QKV in-projection (stored via attn_qkv when not standard attention)
	SSMConv1dW []float32            // [channels × convKernel] depthwise conv weights (flat)
	SSMA       []float32            // [numHeads] log(-A) decay parameter
	SSMAlpha   *core.QuantizedTensor // [dim × numHeads] dt/alpha projection
	SSMBeta    *core.QuantizedTensor // [dim × numHeads] beta/learning-rate projection
	SSMDtBias  []float32            // [numHeads] dt bias
	SSMNorm    []float32            // [headVDim] per-head output norm weight
	SSMOut     *core.QuantizedTensor // [dim × valueDim] output projection
}

// Model holds all weights for a decoder-only transformer LLM.
type Model struct {
	Config       ModelConfig
	TokenEmbed   *core.QuantizedTensor // [vocabSize × dim]
	OutputNorm   []float32            // [dim]
	OutputNormBias []float32          // [dim] optional LayerNorm bias (Phi-2)
	Output       *core.QuantizedTensor // [vocabSize × dim] (may tie with TokenEmbed)
	OutputBias   []float32            // [vocabSize] optional
	Layers       []Layer
}
