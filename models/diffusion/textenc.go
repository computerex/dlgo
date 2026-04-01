package diffusion

import (
	"fmt"
	"log"

	"github.com/computerex/dlgo/models/llm"
)

// TextEncoder wraps a Qwen3 LLM pipeline for text feature extraction.
type TextEncoder struct {
	Pipeline *llm.Pipeline
}

// NewTextEncoder loads a Qwen3 model from GGUF for text encoding.
func NewTextEncoder(modelPath string, maxSeqLen int) (*TextEncoder, error) {
	p, err := llm.NewPipeline(modelPath, maxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("load text encoder: %w", err)
	}
	return &TextEncoder{Pipeline: p}, nil
}

// Encode tokenizes a prompt and extracts hidden states from the LLM.
// Matches sd.cpp's LLMEmbedder for Z-Image:
//   - Wraps prompt in ChatML template (no system prompt)
//   - Extracts second-to-last layer hidden states (no final RMSNorm)
// Returns [contextLen, embeddingDim] flat float32 array.
func (te *TextEncoder) Encode(prompt string) ([]float32, int) {
	p := te.Pipeline
	m := p.Model
	cfg := m.Config
	dim := cfg.EmbeddingDim
	numLayers := cfg.NumLayers

	// Wrap prompt in ChatML template matching sd.cpp's LLMEmbedder for z-image:
	// <|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n
	templatePrompt := "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
	tokens := p.Tokenizer.Encode(templatePrompt)

	contextLen := len(tokens)
	context := make([]float32, contextLen*dim)

	// Extract hidden states from the second-to-last layer (layer numLayers-2 in 0-indexed).
	// sd.cpp uses out_layers={35} (-2) for 36-layer Qwen3, with NO final RMSNorm.
	// ForwardRange with endLayer < numLayers returns raw hidden state without final norm.
	extractLayer := numLayers - 1 // exclusive end: processes layers 0..numLayers-2

	log.Printf("TextEncoder: %d tokens (with template), dim=%d, extract layer=%d/%d",
		contextLen, dim, extractLayer-1, numLayers)

	rs := p.RunState
	kv := p.KVCache

	for i, tok := range tokens {
		llm.ForwardRange(m, tok, i, 0, extractLayer, kv, rs)
		// rs.X now has raw hidden state after layer extractLayer-1 (no final RMSNorm)
		copy(context[i*dim:(i+1)*dim], rs.X[:dim])
	}

	log.Printf("TextEncoder: encoded %d tokens → [%d, %d]", contextLen, contextLen, dim)
	return context, contextLen
}

// EmbeddingDim returns the dimension of hidden states produced by this encoder.
func (te *TextEncoder) EmbeddingDim() int {
	return te.Pipeline.Model.Config.EmbeddingDim
}
