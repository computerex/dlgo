package llm

import (
	"math"
	"strings"
)

// ArchDescriptor describes architecture-specific behavior for an LLM.
type ArchDescriptor struct {
	RopeNeox      bool   // true = NeoX-style RoPE, false = interleaved
	FFNGelu       bool   // true = GeGLU (Gemma), false = SwiGLU (LLaMA/Qwen)
	EmbedScaleMode string // "none" or "sqrt_dim"
	ChatTemplate  string // "chatml", "llama2", "llama3", "gemma", "phi"
}

// archRegistry maps architecture names to their descriptors.
var archRegistry = map[string]ArchDescriptor{
	"llama":     {RopeNeox: false, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "llama2"},
	"qwen2":     {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml"},
	"qwen3":     {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml"},
	"qwen2moe":  {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml"},
	"gemma":     {RopeNeox: true, FFNGelu: true, EmbedScaleMode: "sqrt_dim", ChatTemplate: "gemma"},
	"gemma2":    {RopeNeox: true, FFNGelu: true, EmbedScaleMode: "sqrt_dim", ChatTemplate: "gemma"},
	"gemma3":    {RopeNeox: true, FFNGelu: true, EmbedScaleMode: "sqrt_dim", ChatTemplate: "gemma"},
	"phi2":      {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "phi"},
	"phi3":      {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "phi"},
	"mistral":   {RopeNeox: false, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "llama2"},
	"qwen35":    {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml"},
	"qwen35moe": {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml"},
	"deepseek2": {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml"},
	"gpt-oss":   {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "harmony"},
	"qwen3moe":  {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml"},
	"qwen3next": {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml"},
}

// GetArchDescriptor returns the descriptor for the given architecture.
// Unknown architectures receive a default descriptor (interleaved RoPE, SwiGLU, no embed scale, chatml).
func GetArchDescriptor(arch string) ArchDescriptor {
	if d, ok := archRegistry[arch]; ok {
		return d
	}
	return ArchDescriptor{
		RopeNeox:      false,
		FFNGelu:       false,
		EmbedScaleMode: "none",
		ChatTemplate:  "chatml",
	}
}

// applyArchDefaults applies architecture-specific defaults to config,
// including RopeNeox, FFNGelu, and embed scale calculation using math.Sqrt.
func applyArchDefaults(config *ModelConfig) {
	desc := GetArchDescriptor(config.Architecture)
	config.RopeNeox = desc.RopeNeox
	config.FFNGelu = desc.FFNGelu
	if config.ChatTemplate == "" {
		config.ChatTemplate = desc.ChatTemplate
	}
	if desc.EmbedScaleMode == "sqrt_dim" && config.EmbeddingDim > 0 {
		config.EmbedScale = float32(math.Sqrt(float64(config.EmbeddingDim)))
	}
	// gpt-oss (OpenAI MoE) architecture-specific defaults matching llama.cpp
	if config.Architecture == "gpt-oss" {
		if config.SlidingWindow > 0 && config.SlidingWindowPattern == 0 {
			config.SlidingWindowPattern = 2
		}
		if config.ExpertCount > 0 {
			config.ExpertGatingFunc = 3
		}
	}
	// qwen35moe: llama.cpp hardcodes norm_w=true in build_moe_ffn
	if config.Architecture == "qwen35moe" && config.ExpertCount > 0 {
		config.ExpertWeightsNorm = true
	}
	// Qwen3.5 GGUF files have V heads reordered from grouped to tiled order
	// (see _LinearAttentionVReorderBase in convert_hf_to_gguf.py).
	// GQA mapping for SSM layers: tiled → h % numKVGroups, grouped → h / headsPerGroup.
	if config.Architecture == "qwen35" || config.Architecture == "qwen35moe" {
		config.SSMTiledVOrder = true
	}
}

// RegisterArchitecture registers or overwrites an architecture descriptor.
// Use for extensibility when adding support for new model families.
func RegisterArchitecture(name string, desc ArchDescriptor) {
	archRegistry[name] = desc
}

// Message represents a single chat message with role and content.
type Message struct {
	Role    string
	Content string
}

// FormatChat formats a single-turn chat prompt (system + user) for the model.
func FormatChat(cfg ModelConfig, system, user string) string {
	var msgs []Message
	if system != "" {
		msgs = append(msgs, Message{Role: "system", Content: system})
	}
	msgs = append(msgs, Message{Role: "user", Content: user})
	return FormatMessages(cfg, msgs)
}

// FormatMessages formats a multi-turn conversation for the model.
func FormatMessages(cfg ModelConfig, messages []Message) string {
	template := cfg.ChatTemplate
	if template == "" {
		template = GetArchDescriptor(cfg.Architecture).ChatTemplate
	}
	switch template {
	case "chatml":
		return formatChatMLMessages(messages)
	case "chatml_sep":
		return formatChatMLSepMessages(messages)
	case "llama3":
		return formatLlama3Messages(messages)
	case "llama2":
		return formatLlamaMessages(messages)
	case "gemma":
		return formatGemmaMessages(messages)
	case "phi":
		return formatPhiMessages(messages)
	case "chatglm":
		return formatChatGLMMessages(messages)
	case "harmony":
		return formatHarmonyMessages(messages)
	case "plain":
		return formatPlainMessages(messages)
	default:
		return formatChatMLMessages(messages)
	}
}

func formatPlainMessages(messages []Message) string {
	var b strings.Builder
	for _, m := range messages {
		switch m.Role {
		case "system":
			if m.Content != "" {
				b.WriteString(m.Content)
				b.WriteString("\n\n")
			}
		case "user":
			b.WriteString(m.Content)
			b.WriteString("\n")
		case "assistant":
			b.WriteString(m.Content)
			b.WriteString("\n")
		default:
			b.WriteString(m.Content)
			b.WriteString("\n")
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// formatChatMLMessages formats messages using ChatML template.
// Format: <|im_start|>role\ncontent<|im_end|>\n
func formatChatMLMessages(messages []Message) string {
	var b strings.Builder
	for _, m := range messages {
		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteString("\n")
		b.WriteString(m.Content)
		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

// formatLlama3Messages formats messages using LLaMA 3 template.
// Format: <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
func formatLlama3Messages(messages []Message) string {
	var b strings.Builder
	b.WriteString("<|begin_of_text|>")
	for _, m := range messages {
		b.WriteString("<|start_header_id|>")
		b.WriteString(m.Role)
		b.WriteString("<|end_header_id|>\n\n")
		b.WriteString(m.Content)
		b.WriteString("<|eot_id|>")
	}
	b.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return b.String()
}

// formatLlamaMessages formats messages using LLaMA 2 template.
// Format: <|system|>\ncontent</s>\n<|user|>\ncontent</s>\n<|assistant|>\n
func formatLlamaMessages(messages []Message) string {
	var b strings.Builder
	for _, m := range messages {
		b.WriteString("<|")
		b.WriteString(m.Role)
		b.WriteString("|>\n")
		b.WriteString(m.Content)
		b.WriteString("</s>\n")
	}
	b.WriteString("<|assistant|>\n")
	return b.String()
}

// formatGemmaMessages formats messages using Gemma template.
// Gemma only supports "user" and "model" roles, so system messages are
// prepended to the first user message.
// Format: <start_of_turn>user\ncontent<end_of_turn>\n<start_of_turn>model\n
func formatGemmaMessages(messages []Message) string {
	var systemText string
	var filtered []Message
	for _, m := range messages {
		if m.Role == "system" {
			systemText = m.Content
		} else {
			filtered = append(filtered, m)
		}
	}

	var b strings.Builder
	for i, m := range filtered {
		role := m.Role
		if role == "assistant" {
			role = "model"
		}
		b.WriteString("<start_of_turn>")
		b.WriteString(role)
		b.WriteString("\n")
		if i == 0 && systemText != "" && role == "user" {
			b.WriteString(systemText)
			b.WriteString("\n\n")
		}
		b.WriteString(m.Content)
		b.WriteString("<end_of_turn>\n")
	}
	b.WriteString("<start_of_turn>model\n")
	return b.String()
}

// formatPhiMessages formats messages using Phi template.
// Format: <|system|>\ncontent<|end|>\n<|user|>\ncontent<|end|>\n<|assistant|>\n
func formatPhiMessages(messages []Message) string {
	var b strings.Builder
	for _, m := range messages {
		b.WriteString("<|")
		b.WriteString(m.Role)
		b.WriteString("|>\n")
		b.WriteString(m.Content)
		b.WriteString("<|end|>\n")
	}
	b.WriteString("<|assistant|>\n")
	return b.String()
}

// formatChatGLMMessages formats messages using the ChatGLM-4 / GLM-4.7 template.
// The [gMASK]<sop> prefix is part of the Jinja2 template and must be tokenized.
// Format: [gMASK]<sop><|role|>\ncontent...<|assistant|>\n
func formatChatGLMMessages(messages []Message) string {
	var b strings.Builder
	b.WriteString("[gMASK]<sop>")
	for _, m := range messages {
		b.WriteString("<|")
		b.WriteString(m.Role)
		b.WriteString("|>\n")
		b.WriteString(m.Content)
	}
	b.WriteString("<|assistant|>\n")
	return b.String()
}

// formatChatMLSepMessages formats messages using ChatML with <|im_sep|> separator.
// Matches llama.cpp LLM_CHAT_TEMPLATE_PHI_4 / Qwen3 format.
// Format: <|im_start|>role<|im_sep|>content<|im_end|>\n
func formatChatMLSepMessages(messages []Message) string {
	var b strings.Builder
	for _, m := range messages {
		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteString("<|im_sep|>")
		b.WriteString(m.Content)
		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant<|im_sep|>\n")
	return b.String()
}

// formatHarmonyMessages formats messages using OpenAI Harmony template (gpt-oss).
// Matches llama.cpp LLM_CHAT_TEMPLATE_OPENAI_MOE.
// Format: <|start|>role<|message|>content<|end|> (user/system) or <|return|> (assistant)
func formatHarmonyMessages(messages []Message) string {
	var b strings.Builder
	for _, m := range messages {
		b.WriteString("<|start|>")
		b.WriteString(m.Role)
		b.WriteString("<|message|>")
		b.WriteString(m.Content)
		if m.Role == "assistant" {
			b.WriteString("<|return|>")
		} else {
			b.WriteString("<|end|>")
		}
	}
	b.WriteString("<|start|>assistant")
	return b.String()
}
