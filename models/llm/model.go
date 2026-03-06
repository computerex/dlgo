package llm

import (
	"github.com/computerex/dlgo/core"
)

// Layer holds the weights for one transformer block.
type Layer struct {
	AttnNorm  []float32            // [dim] RMSNorm weight
	Wq        *core.QuantizedTensor // [qDim × dim]
	Wk        *core.QuantizedTensor // [kvDim × dim]
	Wv        *core.QuantizedTensor // [kvDim × dim]
	Wo        *core.QuantizedTensor // [dim × qDim]
	Bq        []float32            // [qDim] optional (Qwen)
	Bk        []float32            // [kvDim] optional
	Bv        []float32            // [kvDim] optional
	AttnQNorm []float32            // [headDim] optional QK norm (Qwen3)
	AttnKNorm []float32            // [headDim] optional QK norm (Qwen3)

	PostAttnNorm []float32         // [dim] optional post-attention norm (Gemma 3)
	FFNNorm      []float32         // [dim] RMSNorm weight
	FFNGate      *core.QuantizedTensor // [ffnDim × dim] w1
	FFNUp        *core.QuantizedTensor // [ffnDim × dim] w3
	FFNDown      *core.QuantizedTensor // [dim × ffnDim] w2
	PostFFNNorm  []float32         // [dim] optional post-FFN norm (Gemma 3)
}

// Model holds all weights for a decoder-only transformer LLM.
type Model struct {
	Config     ModelConfig
	TokenEmbed *core.QuantizedTensor // [vocabSize × dim]
	OutputNorm []float32            // [dim]
	Output     *core.QuantizedTensor // [vocabSize × dim] (may tie with TokenEmbed)
	OutputBias []float32            // [vocabSize] optional
	Layers     []Layer
}
