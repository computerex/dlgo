package server

import (
	"encoding/json"
	"strings"
	"time"
)

// OpenAI-compatible API types for /v1/chat/completions and /v1/models.

// Tool represents a function tool definition (OpenAI-compatible).
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function describes a callable function.
type Function struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"`
}

// ToolCall represents a single tool call made by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall is the name + arguments of a tool call.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolResult holds a tool call result to pass back to the model.
type ToolResult struct {
	Role       string `json:"role"`
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id"`
}

type Message struct {
	Role             string         `json:"role"`
	Content          MessageContent `json:"content"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall     `json:"tool_calls,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
	Name             string         `json:"name,omitempty"`
}

// ContentPart is one element of an OpenAI-style multimodal content array.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL holds an image reference (data URI or URL).
type ImageURL struct {
	URL string `json:"url"`
}

// MessageContent accepts either a plain string (classic) or an array of
// content parts (multimodal). It records both the flattened text and any image
// references so handlers can build image embeddings.
type MessageContent struct {
	Text   string
	Images []string // raw URLs / data URIs, in order
}

func (c *MessageContent) UnmarshalJSON(data []byte) error {
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "null" {
		return nil
	}
	if len(trimmed) > 0 && trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(data, &s); err != nil {
			return err
		}
		c.Text = s
		return nil
	}
	var parts []ContentPart
	if err := json.Unmarshal(data, &parts); err != nil {
		return err
	}
	var sb strings.Builder
	for _, p := range parts {
		switch p.Type {
		case "text", "input_text":
			sb.WriteString(p.Text)
		case "image_url", "input_image":
			if p.ImageURL != nil && p.ImageURL.URL != "" {
				c.Images = append(c.Images, p.ImageURL.URL)
				// Place a marker so the prompt builder can locate the image.
				// Qwen VL-style delimiters teach the model an image block is
				// present; other models tolerate them as text. Wrapped so the
				// embeddings replace only the inner <image> marker.
				sb.WriteString("<|vision_start|><image><|vision_end|>")
			}

		}
	}
	c.Text = sb.String()
	return nil
}

func (c MessageContent) MarshalJSON() ([]byte, error) {
	return json.Marshal(c.Text)
}

type ChatCompletionRequest struct {
	Model             string      `json:"model"`
	Messages          []Message   `json:"messages"`
	Temperature       *float64    `json:"temperature,omitempty"`
	TopP              *float64    `json:"top_p,omitempty"`
	TopK              *int        `json:"top_k,omitempty"`
	MinP              *float64    `json:"min_p,omitempty"`
	MaxTokens         int         `json:"max_tokens,omitempty"`
	Stream            bool        `json:"stream,omitempty"`
	Stop              []string    `json:"stop,omitempty"`
	Seed              *int64      `json:"seed,omitempty"`
	RepetitionPenalty *float64    `json:"repetition_penalty,omitempty"`
	FrequencyPenalty  *float64    `json:"frequency_penalty,omitempty"`
	PresencePenalty   *float64    `json:"presence_penalty,omitempty"`
	// NoRepeatNgramSize bans any token that would complete an n-gram already
	// present in the generated text. Opt-in only: it truncates/mangles text
	// that legitimately repeats an earlier span (tool-call arguments that
	// restate a value from the model's reasoning, enumerated lists, quotes).
	NoRepeatNgramSize *int        `json:"no_repeat_ngram_size,omitempty"`
	ReasoningEffort   string      `json:"reasoning_effort,omitempty"`
	EnableThinking    *bool       `json:"enable_thinking,omitempty"`
	Tools             []Tool      `json:"tools,omitempty"`
	ToolChoice        interface{} `json:"tool_choice,omitempty"`
}

type ChatCompletionChoice struct {
	Index        int      `json:"index"`
	Message      *Message `json:"message,omitempty"`
	Delta        *Message `json:"delta,omitempty"`
	FinishReason *string  `json:"finish_reason"`
}

type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   *UsageInfo             `json:"usage,omitempty"`
}

type ChatCompletionChunk struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
}

type ModelObject struct {
	ID       string `json:"id"`
	Object   string `json:"object"`
	Created  int64  `json:"created"`
	OwnedBy  string `json:"owned_by"`
	Arch     string `json:"architecture,omitempty"`
	Quant    string `json:"quantization,omitempty"`
	ParamStr string `json:"parameters,omitempty"`
	GPU      bool   `json:"gpu"`
	Path     string `json:"path,omitempty"`
}

type AvailableModelObject struct {
	ID   string `json:"id"`
	Path string `json:"path"`
}

type ModelListResponse struct {
	Object    string                 `json:"object"`
	Data      []ModelObject          `json:"data"`
	Available []AvailableModelObject `json:"available,omitempty"`
}

type LoadModelRequest struct {
	ID      string `json:"id"`
	Path    string `json:"path"`
	GPU     bool   `json:"gpu,omitempty"`
	Context int    `json:"context,omitempty"`
}

type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code,omitempty"`
	} `json:"error"`
}

func newCompletionID() string {
	return "chatcmpl-" + randomHex(12)
}

func nowUnix() int64 {
	return time.Now().Unix()
}

func strPtr(s string) *string { return &s }
