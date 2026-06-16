package server

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/computerex/dlgo/models/llm"
)

// convertMessages converts API-level messages (with optional tool definitions and
// tool call references) into llm.Message slices suitable for prompt formatting.
// If tools are provided, their definitions are injected into the system message,
// and assistant/tool messages with tool_calls/tool_call_id are formatted accordingly.
func convertMessages(apiMsgs []Message, tools []Tool) []llm.Message {
	if len(tools) == 0 {
		// Fast path: no tools, simple role/content conversion
		msgs := make([]llm.Message, len(apiMsgs))
		for i, m := range apiMsgs {
			msgs[i] = convertAPIMessage(m)
		}
		return msgs
	}

	// Inject tool definitions into the system prompt.
	toolDefBlock := formatToolDefinitions(tools)

	var msgs []llm.Message
	hasSystem := false
	for _, m := range apiMsgs {
		if m.Role == "system" {
			hasSystem = true
			combined := m.Content
			if combined != "" {
				combined += "\n\n"
			}
			combined += toolDefBlock
			msgs = append(msgs, llm.Message{Role: "system", Content: combined})
		} else {
			msgs = append(msgs, convertAPIMessage(m))
		}
	}
	if !hasSystem {
		// Prepend a system message with tool definitions
		prepended := []llm.Message{{Role: "system", Content: toolDefBlock}}
		msgs = append(prepended, msgs...)
	}
	return msgs
}

// convertAPIMessage converts a single API-level Message to an llm.Message.
// It handles tool_calls, tool results, and plain role/content messages.
// Tool calls are represented as structured ToolCallData on the message so
// the chat template layer can apply model-specific formatting (e.g. Qwen's
// <|tool_call|> / <|tool_response|> markers, Llama's <|python_tag|> format).
func convertAPIMessage(m Message) llm.Message {
	// Assistant message with tool calls
	if len(m.ToolCalls) > 0 {
		tcs := make([]llm.ToolCallData, len(m.ToolCalls))
		for i, tc := range m.ToolCalls {
			tcs[i] = llm.ToolCallData{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			}
		}
		return llm.Message{Role: "assistant", ToolCalls: tcs}
	}

	// Tool result message
	if m.Role == "tool" {
		return llm.Message{Role: "tool", Content: m.Content}
	}

	return llm.Message{Role: m.Role, Content: m.Content}
}

// formatToolDefinitions renders the OpenAI tools array into a prompt-friendly
// system message that describes the available functions.
func formatToolDefinitions(tools []Tool) string {
	var b strings.Builder
	b.WriteString("You have access to the following functions. " +
		"To call a function, respond with JSON specifying the function name and arguments.\n\n")

	for _, tool := range tools {
		b.WriteString("### ")
		b.WriteString(tool.Function.Name)
		b.WriteString("\n\n")
		if tool.Function.Description != "" {
			b.WriteString(tool.Function.Description)
			b.WriteString("\n\n")
		}
		if tool.Function.Parameters != nil {
			paramsJSON, err := json.MarshalIndent(tool.Function.Parameters, "", "  ")
			if err == nil {
				b.WriteString("Parameters:\n\n")
				b.Write(paramsJSON)
				b.WriteString("\n\n\n")
			}
		}
	}

	b.WriteString("Use the function in the format: " +
		"{\"name\": \"function_name\", \"arguments\": { ... }}\n" +
		"After receiving a function result, respond to the user in natural language.")
	return b.String()
}

// toolCallRe matches a JSON function call object, possibly on its own line.
// Supported formats:
//
//	{"name": "func_name", "arguments": {...}}
//	{"name": "func_name", "arguments": "{\"key\": \"value\"}"}
//
// The regex is lenient: it matches any top-level JSON object with "name" and "arguments" keys.
var toolCallRe = regexp.MustCompile(`\{"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}|"[^"]*")\s*\}`)

// parseToolCalls scans generated text for JSON function call objects and returns
// the parsed ToolCall slice. Returns nil if no tool calls are found.
func parseToolCalls(text string) []ToolCall {
	matches := toolCallRe.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return nil
	}

	tcs := make([]ToolCall, 0, len(matches))
	for i, match := range matches {
		name := match[1]
		argsRaw := match[2]

		// If arguments came in as a JSON object, keep it as-is.
		// If it was string-encoded JSON, extract and use as-is.
		// In both cases, validate it's parseable JSON.
		var args string
		if strings.HasPrefix(argsRaw, "\"") {
			// String-encoded JSON: unescape
			var s string
			if err := json.Unmarshal([]byte(argsRaw), &s); err == nil {
				args = s
			} else {
				args = argsRaw
			}
		} else {
			args = argsRaw
		}

		tcs = append(tcs, ToolCall{
			ID:   fmt.Sprintf("call_%d", i),
			Type: "function",
			Function: FunctionCall{
				Name:      name,
				Arguments: args,
			},
		})
	}
	return tcs
}
