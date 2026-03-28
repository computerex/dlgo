package llm

import (
	"fmt"
	"math"
	"strings"
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
	RopeDim       int      // partial RoPE: 0 = full headDim, else only first RopeDim dims
	RopeScaleType int      // 0=none, 1=linear, 2=yarn
	RopeScaleFactor    float32 // scaling factor for extended context
	RopeOrigMaxPos     int     // original max position embeddings (for YaRN)
	RopeYaRNBetaFast   float32 // YaRN beta_fast (default 32)
	RopeYaRNBetaSlow   float32 // YaRN beta_slow (default 1)
	RopeYaRNExtFactor  float32 // YaRN ext_factor: 0=disable ramp, 1=full ramp (default 1)
	RopeYaRNAttnFactor float32 // YaRN attn_factor: magnitude scaling base (default 1)
	SlidingWindow      int     // sliding window attention size (0 = disabled)
	SlidingWindowPattern int   // 0=all layers, N=alternating (every Nth layer is full)
	BOS           int32
	EOS           int32
	StopTokens    []int32
	AddBOS        bool
	FFNGelu       bool     // true = GeGLU (Gemma), false = SwiGLU (LLaMA/Qwen)
	EmbedScale    float32  // non-zero = scale embeddings (Gemma: sqrt(dim))
	ChatTemplate  string   // chat format: "chatml", "llama2", "llama3", "gemma", "phi"

	// Gemma 2 soft-capping
	AttnLogitSoftcap  float32 // 0=disabled; >0 = tanh(logit/cap)*cap before softmax
	FinalLogitSoftcap float32 // 0=disabled; >0 = tanh(logit/cap)*cap on final logits

	// Qwen3.5 hybrid Mamba/Attention
	FullAttentionInterval int  // 0 = all attention; N = every Nth layer is attention
	SSMConvKernel         int
	SSMInnerSize          int
	SSMStateSize          int
	SSMTimeStepRank       int
	SSMGroupCount         int
	SSMTiledVOrder        bool // true = GGUF V heads in tiled order (Qwen3.5); false = grouped (Qwen3Next)

	// MoE (Mixture of Experts)
	ExpertCount          int     // 0 = dense (no MoE); >0 = number of experts per layer
	ExpertUsedCount      int     // top-K experts selected per token
	ExpertFFNDim         int     // hidden dim per expert
	SharedExpertFFNDim   int     // hidden dim for shared expert (0 = no shared expert)
	ExpertGatingFunc     int     // 0=none, 1=softmax, 2=sigmoid, 3=softmax_weight (top-k raw then softmax)
	ExpertWeightsNorm    bool    // normalize selected expert weights by sum
	ExpertWeightsScale   float32 // scale factor for expert weights (0 = no scaling)

	// MLA (Multi-head Latent Attention) — DeepSeek-V2/GLM-4
	QLORARank         int // Q compression rank (attn_q_a output dim)
	KVLORARank        int // KV compression rank (compressed KV without rope)
	QKNopeDim         int // Non-positional K dim per head (k_b output per head)
	QKRopeDim         int // Positional (RoPE) K dim per head
	VHeadDim          int // V dim per head (v_b output per head)
	LeadingDenseCount int // Number of initial dense layers before MoE
}

// ParseConfig extracts a ModelConfig from GGUF metadata.
func ParseConfig(md map[string]interface{}) (ModelConfig, error) {
	return parseConfig(md)
}

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

	// Norm epsilon: try RMSNorm key first, then LayerNorm key (Phi-2)
	normEps := metaFloat(md, arch+".attention.layer_norm_rms_epsilon", 0)
	if normEps == 0 {
		normEps = metaFloat(md, arch+".attention.layer_norm_epsilon", 1e-5)
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
		RMSNormEps:    normEps,
		RopeFreqBase:       metaFloat(md, arch+".rope.freq_base", 10000.0),
		RopeScaleType:      parseRopeScaleType(md, arch),
		RopeScaleFactor:    metaFloat(md, arch+".rope.scaling.factor", 0),
		RopeOrigMaxPos:     metaInt(md, arch+".rope.scaling.original_context_length", 0),
		RopeYaRNBetaFast:   metaFloat(md, arch+".rope.scaling.yarn.beta_fast", 32.0),
		RopeYaRNBetaSlow:   metaFloat(md, arch+".rope.scaling.yarn.beta_slow", 1.0),
		RopeYaRNExtFactor:  metaFloat(md, arch+".rope.scaling.yarn.ext_factor", 1.0),
		RopeYaRNAttnFactor: metaFloat(md, arch+".rope.scaling.yarn.attn_factor", 1.0),
		SlidingWindow:      metaInt(md, arch+".attention.sliding_window", 0),
		SlidingWindowPattern: metaInt(md, arch+".attention.sliding_window_pattern", 0),
		AttnLogitSoftcap:  metaFloat(md, arch+".attn_logit_softcapping", 0),
		FinalLogitSoftcap: metaFloat(md, arch+".final_logit_softcapping", 0),
		BOS:                int32(metaInt(md, "tokenizer.ggml.bos_token_id", 1)),
		EOS:           int32(metaInt(md, "tokenizer.ggml.eos_token_id", 2)),
		AddBOS:        metaBool(md, "tokenizer.ggml.add_bos_token", true),

		FullAttentionInterval: metaInt(md, arch+".full_attention_interval", 0),
		SSMConvKernel:         metaInt(md, arch+".ssm.conv_kernel", 4),
		SSMInnerSize:          metaInt(md, arch+".ssm.inner_size", 0),
		SSMStateSize:          metaInt(md, arch+".ssm.state_size", 0),
		SSMTimeStepRank:       metaInt(md, arch+".ssm.time_step_rank", 0),
		SSMGroupCount:         metaInt(md, arch+".ssm.group_count", 0),
		ChatTemplate:          inferChatTemplate(md, arch),

		ExpertCount:        metaInt(md, arch+".expert_count", 0),
		ExpertUsedCount:    metaInt(md, arch+".expert_used_count", 0),
		ExpertFFNDim:       metaInt(md, arch+".expert_feed_forward_length", 0),
		SharedExpertFFNDim: metaInt(md, arch+".expert_shared_feed_forward_length", 0),
		ExpertGatingFunc:   metaInt(md, arch+".expert_gating_func", 1),
		ExpertWeightsNorm:  metaBool(md, arch+".expert_weights_norm", false),
		ExpertWeightsScale: metaFloat(md, arch+".expert_weights_scale", 0),

		QLORARank:         metaInt(md, arch+".attention.q_lora_rank", 0),
		KVLORARank:        metaInt(md, arch+".attention.kv_lora_rank", 0),
		QKNopeDim:         metaInt(md, arch+".attention.key_length_mla", 0),
		VHeadDim:          metaInt(md, arch+".attention.value_length_mla", 0),
		LeadingDenseCount: metaInt(md, arch+".leading_dense_block_count", 0),
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
	c.RopeDim = metaInt(md, arch+".rope.dimension_count", 0)
	if c.QLORARank > 0 {
		c.QKRopeDim = c.RopeDim
		// key_length_mla is the TOTAL per-head Q/K dim (nope + rope);
		// derive the non-positional portion by subtracting the rope dim.
		if c.QKNopeDim > c.QKRopeDim {
			c.QKNopeDim -= c.QKRopeDim
		}
	}
	if c.RopeDim == 0 || c.RopeDim > c.HeadDim {
		c.RopeDim = c.HeadDim
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

// parseRopeScaleType detects the RoPE scaling type from GGUF metadata.
// Returns 0=none, 1=linear, 2=yarn.
func parseRopeScaleType(md map[string]interface{}, arch string) int {
	// Try string key (GGUF standard)
	s := metaString(md, arch+".rope.scaling.type")
	switch strings.ToLower(s) {
	case "linear":
		return 1
	case "yarn":
		return 2
	}
	// Try int key (some converters)
	v := metaInt(md, arch+".rope.scaling.type", 0)
	if v > 0 {
		return v
	}
	// Auto-detect: if factor > 0 and original_context_length > 0, assume yarn
	factor := metaFloat(md, arch+".rope.scaling.factor", 0)
	origCtx := metaInt(md, arch+".rope.scaling.original_context_length", 0)
	if factor > 1.0 && origCtx > 0 {
		return 2
	}
	return 0
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

func inferChatTemplate(md map[string]interface{}, arch string) string {
	chatTemplate := metaString(md, "tokenizer.chat_template")
	if chatTemplate != "" {
		lower := strings.ToLower(chatTemplate)
		switch {
		case strings.Contains(lower, "<|start_header_id|>") || strings.Contains(lower, "<|eot_id|>"):
			return "llama3"
		case strings.Contains(lower, "<|im_start|>") && strings.Contains(lower, "<|im_sep|>"):
			return "chatml_sep"
		case strings.Contains(lower, "<|im_start|>"):
			return "chatml"
		case strings.Contains(lower, "<start_of_turn>"):
			return "gemma"
		case strings.Contains(lower, "[gmask]<sop>"):
			return "chatglm"
		case strings.Contains(lower, "<|start|") && strings.Contains(lower, "<|message|"):
			return "harmony"
		case strings.Contains(lower, "<|end|>"):
			return "phi"
		case strings.Contains(lower, "<|assistant|>"):
			return "llama2"
		}
	}

	// Llama-family models can be either Llama-2/3 chat templates.
	if arch == "llama" && hasTokenizerToken(md, "<|start_header_id|>") {
		return "llama3"
	}

	if strings.HasPrefix(arch, "gemma") {
		name := strings.ToLower(metaString(md, "general.name"))
		switch {
		case strings.Contains(name, " instruct"), strings.Contains(name, "-instruct"), strings.Contains(name, "_instruct"):
			return "gemma"
		case strings.Contains(name, " it"), strings.Contains(name, "-it"), strings.Contains(name, "_it"):
			return "gemma"
		default:
			return "plain"
		}
	}

	return ""
}

func hasTokenizerToken(md map[string]interface{}, token string) bool {
	raw, ok := md["tokenizer.ggml.tokens"]
	if !ok {
		return false
	}
	tokens, ok := raw.([]interface{})
	if !ok {
		return false
	}
	for _, t := range tokens {
		if s, ok := t.(string); ok && s == token {
			return true
		}
	}
	return false
}
