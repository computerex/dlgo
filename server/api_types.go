package server

import "time"

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
	Role             string     `json:"role"`
	Content          string     `json:"content"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
	Name             string     `json:"name,omitempty"`
}

type ChatCompletionRequest struct {
	Model             string      `json:"model"`
	Messages          []Message   `json:"messages"`
	Temperature       *float64    `json:"temperature,omitempty"`
	TopP              *float64    `json:"top_p,omitempty"`
	TopK              *int        `json:"top_k,omitempty"`
	MaxTokens         int         `json:"max_tokens,omitempty"`
	Stream            bool        `json:"stream,omitempty"`
	Stop              []string    `json:"stop,omitempty"`
	Seed              *int64      `json:"seed,omitempty"`
	RepetitionPenalty *float64    `json:"repetition_penalty,omitempty"`
	ReasoningEffort   string      `json:"reasoning_effort,omitempty"`
	EnableThinking    *bool       `json:"enable_thinking,omitempty"`
	Tools             []Tool      `json:"tools,omitempty"`
	ToolChoice        interface{} `json:"tool_choice,omitempty"`
}

type ChatCompletionChoice struct {
	Index        int       `json:"index"`
	Message      *Message  `json:"message,omitempty"`
	Delta        *Message  `json:"delta,omitempty"`
	FinishReason *string   `json:"finish_reason"`
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
