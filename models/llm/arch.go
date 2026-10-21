package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// ArchDescriptor describes architecture-specific behavior for an LLM.
type ArchDescriptor struct {
	RopeNeox         bool   // true = NeoX-style RoPE, false = interleaved
	FFNGelu          bool   // true = GeGLU (Gemma), false = SwiGLU (LLaMA/Qwen)
	EmbedScaleMode   string // "none" or "sqrt_dim"
	ChatTemplate     string // "chatml", "llama2", "llama3", "gemma", "phi"
	SupportsThinking bool   // true = model uses <think> blocks (Qwen3/3.5)
	ToolCallFormat   string // native tool-call protocol, see ToolCallFormat* constants
}

// Native tool-call protocols.
//
// ToolCallFormatQwen35 selects the Qwen3.5 / Qwen3.6 native protocol that the
// model's embedded chat template defines:
//
//   - Function signatures are rendered into the system turn inside a
//     <tools>...</tools> XML block, one compact JSON object per line.
//   - The model emits calls as:
//          <tool_call>
//          <function=NAME>
//          <parameter=KEY>
//          VALUE
//          </parameter>
//          </function>
//          </tool_call>
//   - Tool results are returned in a <tool_response>...</tool_response> block
//     inside a *user* turn.
//
// The empty value selects the legacy behavior (prose tool description +
// bare-JSON scraping) used for architectures that have no native protocol
// wired up yet.
const (
	ToolCallFormatNone   = ""
	ToolCallFormatQwen35 = "qwen35"
)

// archRegistry maps architecture names to their descriptors.
var archRegistry = map[string]ArchDescriptor{
	"llama":     {RopeNeox: false, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "llama2"},
	"qwen2":     {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml"},
	"qwen3":     {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml", SupportsThinking: true},
	"qwen2moe":  {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml"},
	"gemma":     {RopeNeox: true, FFNGelu: true, EmbedScaleMode: "sqrt_dim", ChatTemplate: "gemma"},
	"gemma2":    {RopeNeox: true, FFNGelu: true, EmbedScaleMode: "sqrt_dim", ChatTemplate: "gemma"},
	"gemma3":    {RopeNeox: true, FFNGelu: true, EmbedScaleMode: "sqrt_dim", ChatTemplate: "gemma"},
	"phi2":      {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "phi"},
	"phi3":      {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "phi"},
	"mistral":   {RopeNeox: false, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "llama2"},
	"qwen35":    {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml", SupportsThinking: true, ToolCallFormat: ToolCallFormatQwen35},
	"qwen35moe": {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml", SupportsThinking: true, ToolCallFormat: ToolCallFormatQwen35},
	"deepseek2": {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml"},
	"gpt-oss":   {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "harmony"},
	"qwen3moe":  {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml", SupportsThinking: true},
	"qwen3next": {RopeNeox: true, FFNGelu: false, EmbedScaleMode: "none", ChatTemplate: "chatml", SupportsThinking: true},
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
	// Gemma 2: alternating sliding window (even layers = sliding, odd layers = full)
	if config.Architecture == "gemma2" &&
		config.SlidingWindow > 0 && config.SlidingWindowPattern == 0 {
		config.SlidingWindowPattern = 2
	}
	// Gemma 3: every 6th layer is full attention, rest are SWA (matches llama.cpp default)
	if config.Architecture == "gemma3" &&
		config.SlidingWindow > 0 && config.SlidingWindowPattern == 0 {
		config.SlidingWindowPattern = 6
	}
	// gpt-oss (OpenAI MoE ISWA): alternating SWA/full (period = 2, matches llama.cpp default)
	if config.Architecture == "gpt-oss" {
		if config.SlidingWindow > 0 && config.SlidingWindowPattern == 0 {
			config.SlidingWindowPattern = 2
		}
		if config.ExpertCount > 0 {
			config.ExpertGatingFunc = 3
		}
	}
	// qwen3moe / qwen35moe: llama.cpp hardcodes norm_w=true in build_moe_ffn
	if (config.Architecture == "qwen3moe" || config.Architecture == "qwen35moe") && config.ExpertCount > 0 {
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

// ToolCallData represents a single function call made by the model.
type ToolCallData struct {
	Name      string
	Arguments string // JSON string of arguments
}

// Message represents a single chat message with role, content, and optional tool calls.
type Message struct {
	Role      string
	Content   string
	ToolCalls []ToolCallData // set for assistant messages with function calls
}

// FormatOptions controls template-level formatting (e.g. reasoning effort).
type FormatOptions struct {
	ReasoningEffort string // "low", "medium", "high" (default: "medium")
	EnableThinking  *bool  // nil = auto (enabled for thinking models), false = disable
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
func FormatMessages(cfg ModelConfig, messages []Message, opts ...FormatOptions) string {
	template := cfg.ChatTemplate
	if template == "" {
		template = GetArchDescriptor(cfg.Architecture).ChatTemplate
	}
	var opt FormatOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	// Thinking model support: determine whether to inject <think> primer.
	// Matches the official Qwen3/3.5 Jinja template behavior.
	arch := GetArchDescriptor(cfg.Architecture)
	thinking := arch.SupportsThinking
	if opt.EnableThinking != nil {
		thinking = *opt.EnableThinking
	}

	// For thinking models, strip <think>...</think> blocks from assistant
	// messages in history. The Jinja template does this to prevent context
	// explosion: each thinking block can be hundreds of tokens; including them
	// verbatim in every subsequent turn exhausts MaxSeqLen after 1-2 turns.
	if thinking {
		stripped := make([]Message, len(messages))
		copy(stripped, messages)
		for i := range stripped {
			if stripped[i].Role == "assistant" {
				stripped[i].Content = stripThinkingBlock(stripped[i].Content)
			}
		}
		messages = stripped
	}

	var prompt string
	switch template {
	case "chatml":
		prompt = formatChatMLDispatch(messages, arch)
	case "chatml_sep":
		prompt = formatChatMLSepMessages(messages)
	case "llama3":
		prompt = formatLlama3Messages(messages)
	case "llama2":
		prompt = formatLlamaMessages(messages)
	case "gemma":
		prompt = formatGemmaMessages(messages)
	case "phi":
		prompt = formatPhiMessages(messages)
	case "chatglm":
		prompt = formatChatGLMMessages(messages)
	case "harmony":
		prompt = formatHarmonyMessages(messages, opt)
	case "plain":
		prompt = formatPlainMessages(messages)
	default:
		prompt = formatChatMLDispatch(messages, arch)
	}

	if thinking {
		prompt += "<think>\n"
	} else if arch.SupportsThinking {
		// Thinking-capable model with thinking explicitly disabled.
		// Append an empty closed think block so the model skips the thinking
		// phase and goes straight to the answer. This matches the official
		// Qwen3/3.5 Jinja template behavior and is what Ollama does.
		prompt += "<think></think>\n"
	}

	return prompt
}

// stripThinkingBlock removes the leading <think>...</think> block from an
// assistant message. The official Qwen3/3.5 Jinja template strips reasoning
// from history turns so only the final answer is included for context.
func stripThinkingBlock(content string) string {
	if !strings.HasPrefix(content, "<think>") {
		return content
	}
	if idx := strings.Index(content, "</think>"); idx >= 0 {
		content = strings.TrimLeft(content[idx+len("</think>"):], "\n")
	}
	return content
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
//
// For tool calling (Qwen-native format):
//   - assistant messages with ToolCalls use <|tool_call|> markers
//   - tool messages use <|tool_response|> markers
func formatChatMLMessages(messages []Message) string {
	var b strings.Builder
	for _, m := range messages {
		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteString("\n")

		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			// Qwen format: each tool call wrapped in <|tool_call|> marker
			for _, tc := range m.ToolCalls {
				b.WriteString("<|tool_call|>\n")
				fmt.Fprintf(&b, `{"name": %q, "arguments": %s}`, tc.Name, tc.Arguments)
				b.WriteString("\n")
			}
		} else if m.Role == "tool" {
			// Qwen format: tool result wrapped in <|tool_response|> marker
			b.WriteString("<|tool_response|>\n")
			b.WriteString(m.Content)
		} else {
			b.WriteString(m.Content)
		}

		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

// formatChatMLDispatch selects the chatml renderer variant for the given
// architecture. Qwen3.5/3.6 use a native tool protocol (see
// formatChatMLQwen35Messages); everything else uses the classic renderer.
func formatChatMLDispatch(messages []Message, arch ArchDescriptor) string {
	if arch.ToolCallFormat == ToolCallFormatQwen35 {
		return formatChatMLQwen35Messages(messages)
	}
	return formatChatMLMessages(messages)
}

// formatChatMLQwen35Messages renders messages using the Qwen3.5/3.6 native
// ChatML template, matching the model's embedded Jinja template exactly so the
// model uses its trained tool-call protocol.
//
// Key differences from the classic renderer:
//   - Assistant tool calls are rendered as:
//         <tool_call>
//         <function=NAME>
//         <parameter=KEY>
//         VALUE
//         </parameter>
//         </function>
//         </tool_call>
//   - Tool results are emitted inside a *user* turn wrapped in
//     <tool_response>...</tool_response> (consecutive tool results are grouped
//     into a single user turn, matching the Jinja template).
func formatChatMLQwen35Messages(messages []Message) string {
	// lastQueryIndex mirrors the Jinja template's ns.last_query_index: the index
	// of the last user message that is a genuine query (not a <tool_response>
	// block). Assistant messages after it are rendered with an (empty) <think>
	// block, matching the reference template.
	lastQueryIndex := -1
	for idx := len(messages) - 1; idx >= 0; idx-- {
		if messages[idx].Role != "user" {
			continue
		}
		c := strings.TrimSpace(messages[idx].Content)
		if strings.HasPrefix(c, "<tool_response>") && strings.HasSuffix(c, "</tool_response>") {
			continue
		}
		lastQueryIndex = idx
		break
	}

	var b strings.Builder
	for i := 0; i < len(messages); i++ {
		m := messages[i]

		// Group consecutive tool result messages into one user turn.
		if m.Role == "tool" {
			b.WriteString("<|im_start|>user")
			for i < len(messages) && messages[i].Role == "tool" {
				b.WriteString("\n<tool_response>\n")
				b.WriteString(messages[i].Content)
				b.WriteString("\n</tool_response>")
				i++
			}
			b.WriteString("<|im_end|>\n")
			i-- // outer loop will i++
			continue
		}

		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteString("\n")

		if m.Role == "assistant" {
			if i > lastQueryIndex {
				// No separate reasoning_content is carried on history turns, so
				// render the empty thinking block the template expects.
				b.WriteString("<think>\n\n</think>\n\n")
			}
			b.WriteString(m.Content)
			for j, tc := range m.ToolCalls {
				if j == 0 {
					if m.Content != "" {
						b.WriteString("\n\n")
					}
				} else {
					b.WriteString("\n")
				}
				b.WriteString("<tool_call>\n<function=")
				b.WriteString(tc.Name)
				b.WriteString(">\n")
				for _, kv := range orderedArgs(tc.Arguments) {
					b.WriteString("<parameter=")
					b.WriteString(kv.key)
					b.WriteString(">\n")
					b.WriteString(kv.value)
					b.WriteString("\n</parameter>\n")
				}
				b.WriteString("</function>\n</tool_call>")
			}
		} else {
			b.WriteString(m.Content)
		}

		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

type argKV struct{ key, value string }

// orderedArgs parses an OpenAI-style arguments JSON object into an ordered list
// of key/value pairs whose values are rendered the way the Jinja template does
// (objects/arrays as compact JSON, scalars as their string form).
func orderedArgs(argsJSON string) []argKV {
	trimmed := strings.TrimSpace(argsJSON)
	if trimmed == "" || trimmed == "null" {
		return nil
	}
	// The arguments field is a JSON object encoded as a string. Some clients
	// pass a JSON string containing JSON; unwrap one level if needed.
	if strings.HasPrefix(trimmed, "\"") {
		var s string
		if json.Unmarshal([]byte(trimmed), &s) == nil {
			trimmed = strings.TrimSpace(s)
		}
	}
	dec := json.NewDecoder(strings.NewReader(trimmed))
	dec.UseNumber()
	var obj map[string]interface{}
	if err := dec.Decode(&obj); err != nil {
		return nil
	}
	// Preserve input order for stability where possible.
	var keys []string
	keys = append(keys, orderedKeys(trimmed)...)
	seen := map[string]bool{}
	out := make([]argKV, 0, len(obj))
	add := func(k string) {
		if seen[k] {
			return
		}
		if v, ok := obj[k]; ok {
			seen[k] = true
			out = append(out, argKV{key: k, value: renderArgValue(v)})
		}
	}
	for _, k := range keys {
		add(k)
	}
	for k := range obj {
		add(k)
	}
	return out
}

// orderedKeys returns object keys in the order they appear in rawJSON.
func orderedKeys(rawJSON string) []string {
	dec := json.NewDecoder(strings.NewReader(rawJSON))
	tok, err := dec.Token()
	if err != nil {
		return nil
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil
	}
	var keys []string
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return keys
		}
		key, _ := kt.(string)
		keys = append(keys, key)
		// Skip the value.
		var skip json.RawMessage
		if err := dec.Decode(&skip); err != nil {
			return keys
		}
	}
	return keys
}

// renderArgValue renders a decoded JSON value the way the Qwen Jinja template
// does: mappings and sequences become compact JSON, everything else its string
// form.
func renderArgValue(v interface{}) string {
	switch x := v.(type) {
	case string:
		return x
	case json.Number:
		return x.String()
	case bool:
		return strconv.FormatBool(x)
	case nil:
		return "null"
	case map[string]interface{}, []interface{}:
		if b, err := json.Marshal(x); err == nil {
			return string(b)
		}
		return ""
	default:
		if b, err := json.Marshal(x); err == nil {
			return string(b)
		}
		return fmt.Sprintf("%v", x)
	}
}

// formatLlama3Messages formats messages using LLaMA 3 template.
// Format: <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
//
// For tool calling (LLaMA 3.1+ native format):
//   - assistant messages with ToolCalls use <|python_tag|> marker
//   - tool results use the ipython role
func formatLlama3Messages(messages []Message) string {
	var b strings.Builder
	b.WriteString("<|begin_of_text|>")
	for _, m := range messages {
		role := m.Role
		if role == "tool" {
			// LLaMA 3.1+ uses ipython role for tool results
			role = "ipython"
		}
		b.WriteString("<|start_header_id|>")
		b.WriteString(role)
		b.WriteString("<|end_header_id|>\n\n")
		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			for _, tc := range m.ToolCalls {
				b.WriteString("<|python_tag|>")
				fmt.Fprintf(&b, `{"name": %q, "arguments": %s}`, tc.Name, tc.Arguments)
			}
		} else {
			b.WriteString(m.Content)
		}
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
//
// For tool calling (GLM-4 native format):
//   - assistant messages with ToolCalls embed JSON function calls
//   - tool results use the observation role
func formatChatGLMMessages(messages []Message) string {
	var b strings.Builder
	b.WriteString("[gMASK]<sop>")
	for _, m := range messages {
		role := m.Role
		if role == "tool" {
			// GLM-4/4.7 uses observation role for tool results
			role = "observation"
		}
		b.WriteString("<|")
		b.WriteString(role)
		b.WriteString("|>\n")
		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			for _, tc := range m.ToolCalls {
				fmt.Fprintf(&b, `{"name": %q, "arguments": %s}`, tc.Name, tc.Arguments)
				b.WriteString("\n")
			}
		} else {
			b.WriteString(m.Content)
		}
	}
	b.WriteString("<|assistant|>\n")
	return b.String()
}

// formatChatMLSepMessages formats messages using ChatML with <|im_sep|> separator.
// Matches llama.cpp LLM_CHAT_TEMPLATE_PHI_4 / Qwen3 format.
// Format: <|im_start|>role<|im_sep|>content<|im_end|>\n
//
// For tool calling (Qwen/Phi-native format):
//   - assistant messages with ToolCalls use <|tool_call|> markers
//   - tool messages use <|tool_response|> markers
func formatChatMLSepMessages(messages []Message) string {
	var b strings.Builder
	for _, m := range messages {
		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteString("<|im_sep|>")

		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			for _, tc := range m.ToolCalls {
				b.WriteString("<|tool_call|>\n")
				fmt.Fprintf(&b, `{"name": %q, "arguments": %s}`, tc.Name, tc.Arguments)
				b.WriteString("\n")
			}
		} else if m.Role == "tool" {
			b.WriteString("<|tool_response|>\n")
			b.WriteString(m.Content)
		} else {
			b.WriteString(m.Content)
		}

		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant<|im_sep|>\n")
	return b.String()
}

// formatHarmonyMessages formats messages using the OpenAI Harmony template (gpt-oss).
// Matches the official Jinja chat_template embedded in the GGUF metadata.
//
// Structure:
//   1. Built-in system message (model identity, date, reasoning, channel info)
//   2. Optional developer message (from user's "system" role message)
//   3. User/assistant turns
//   4. Generation prompt forcing the "final" channel
func formatHarmonyMessages(messages []Message, opt FormatOptions) string {
	reasoning := opt.ReasoningEffort
	if reasoning == "" {
		reasoning = "medium"
	}

	var b strings.Builder

	// 1. Built-in system message (always present per official template)
	b.WriteString("<|start|>system<|message|>")
	b.WriteString("You are a helpful assistant.\n")
	b.WriteString("Knowledge cutoff: 2024-06\n")
	b.WriteString(fmt.Sprintf("Current date: %s\n\n", time.Now().Format("2006-01-02")))
	b.WriteString(fmt.Sprintf("Reasoning: %s\n\n", reasoning))
	b.WriteString("# Valid channels: analysis, commentary, final. Channel must be included for every message.")
	b.WriteString("<|end|>")

	// 2. Separate user-provided system/developer message if present
	start := 0
	if len(messages) > 0 && (messages[0].Role == "system" || messages[0].Role == "developer") {
		if messages[0].Content != "" {
			b.WriteString("<|start|>developer<|message|>")
			b.WriteString("# Instructions\n\n")
			b.WriteString(messages[0].Content)
			b.WriteString("\n\n<|end|>")
		}
		start = 1
	}

	// 3. Conversation turns
	for _, m := range messages[start:] {
		switch m.Role {
		case "assistant":
			b.WriteString("<|start|>assistant<|channel|>final<|message|>")
			b.WriteString(m.Content)
			b.WriteString("<|end|>")
		case "user":
			b.WriteString("<|start|>user<|message|>")
			b.WriteString(m.Content)
			b.WriteString("<|end|>")
		}
	}

	// 4. Generation prompt — let the model choose its channel (analysis/final).
	// The official template uses just "<|start|>assistant" so the model can
	// decide to reason first (analysis channel) before giving the final answer.
	b.WriteString("<|start|>assistant")
	return b.String()
}
