package llm

import (
	"fmt"
	"math"
)

// ModelConfig holds all architecture parameters for a decoder-only transformer LLM.
// Auto-populated from GGUF metadata.
type ModelConfig struct {
	Architecture  string
	VocabSize     int
	ContextLength int
	EmbeddingDim  int
	NumLayers     int
	FFNDim        int
	NumHeads      int
	NumKVHeads    int
	HeadDim       int
	RMSNormEps    float32
	RopeFreqBase  float32
	RopeNeox      bool
	BOS           int32
	EOS           int32
	StopTokens    []int32
	AddBOS        bool
	FFNGelu       bool     // true = GeGLU (Gemma), false = SwiGLU (LLaMA/Qwen)
	EmbedScale    float32  // non-zero = scale embeddings (Gemma: sqrt(dim))
	ChatTemplate  string   // chat format: "chatml", "llama2", "llama3", "gemma", "phi"
}

// parseConfig extracts a ModelConfig from GGUF metadata.
func parseConfig(md map[string]interface{}) (ModelConfig, error) {
	arch := metaString(md, "general.architecture")
	if arch == "" {
		return ModelConfig{}, fmt.Errorf("missing general.architecture")
	}

	// Vocab size: prefer metadata, fall back to token list length
	vocabSize := metaInt(md, arch+".vocab_size", 0)
	if vocabSize == 0 {
		if tokArr, ok := md["tokenizer.ggml.tokens"].([]interface{}); ok {
			vocabSize = len(tokArr)
		}
	}
	if vocabSize == 0 {
		vocabSize = 32000
	}

	c := ModelConfig{
		Architecture:  arch,
		VocabSize:     vocabSize,
		ContextLength: metaInt(md, arch+".context_length", 2048),
		EmbeddingDim:  metaInt(md, arch+".embedding_length", 0),
		NumLayers:     metaInt(md, arch+".block_count", 0),
		FFNDim:        metaInt(md, arch+".feed_forward_length", 0),
		NumHeads:      metaInt(md, arch+".attention.head_count", 0),
		NumKVHeads:    metaInt(md, arch+".attention.head_count_kv", 0),
		HeadDim:       metaInt(md, arch+".attention.key_length", 0),
		RMSNormEps:    metaFloat(md, arch+".attention.layer_norm_rms_epsilon", 1e-5),
		RopeFreqBase:  metaFloat(md, arch+".rope.freq_base", 10000.0),
		BOS:           int32(metaInt(md, "tokenizer.ggml.bos_token_id", 1)),
		EOS:           int32(metaInt(md, "tokenizer.ggml.eos_token_id", 2)),
		AddBOS:        metaBool(md, "tokenizer.ggml.add_bos_token", true),
	}

	if c.EmbeddingDim == 0 {
		return c, fmt.Errorf("missing %s.embedding_length", arch)
	}
	if c.NumLayers == 0 {
		return c, fmt.Errorf("missing %s.block_count", arch)
	}
	if c.NumHeads == 0 {
		return c, fmt.Errorf("missing %s.attention.head_count", arch)
	}
	if c.NumKVHeads == 0 {
		c.NumKVHeads = c.NumHeads
	}
	if c.HeadDim == 0 {
		c.HeadDim = c.EmbeddingDim / c.NumHeads
	}

	applyArchDefaults(&c)

	// Parse stop tokens from GGUF metadata
	if eosArr, ok := md["tokenizer.ggml.eos_token_id"].([]interface{}); ok {
		for _, v := range eosArr {
			if id, ok := toInt(v); ok {
				c.StopTokens = append(c.StopTokens, int32(id))
			}
		}
	}

	return c, nil
}

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// metaString extracts a string from GGUF metadata.
func metaString(md map[string]interface{}, key string) string {
	if v, ok := md[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// metaInt extracts an integer from GGUF metadata (handles uint32, int32, uint64).
func metaInt(md map[string]interface{}, key string, def int) int {
	v, ok := md[key]
	if !ok {
		return def
	}
	if i, ok := toInt(v); ok {
		return i
	}
	return def
}

func toInt(v interface{}) (int, bool) {
	switch x := v.(type) {
	case uint32:
		return int(x), true
	case int32:
		return int(x), true
	case uint64:
		return int(x), true
	case int64:
		return int(x), true
	case uint8:
		return int(x), true
	case int8:
		return int(x), true
	case uint16:
		return int(x), true
	case int16:
		return int(x), true
	default:
		return 0, false
	}
}

func metaFloat(md map[string]interface{}, key string, def float32) float32 {
	v, ok := md[key]
	if !ok {
		return def
	}
	switch x := v.(type) {
	case float32:
		return x
	case float64:
		return float32(x)
	default:
		return def
	}
}

func metaBool(md map[string]interface{}, key string, def bool) bool {
	v, ok := md[key]
	if !ok {
		return def
	}
	if b, ok := v.(bool); ok {
		return b
	}
	return def
}
