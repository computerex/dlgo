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
// If tools are provided, their definitions are injected into the system message
// using the model's native tool protocol (selected by toolFormat), and
// assistant/tool messages with tool_calls/tool_call_id are formatted accordingly.
func convertMessages(apiMsgs []Message, tools []Tool, toolFormat string) []llm.Message {
	if len(tools) == 0 {
		// Fast path: no tools, simple role/content conversion
		msgs := make([]llm.Message, len(apiMsgs))
		for i, m := range apiMsgs {
			msgs[i] = convertAPIMessage(m)
		}
		return msgs
	}

	// Render tool definitions in the model's native protocol.
	toolDefBlock := formatToolDefinitions(tools, toolFormat)

	var msgs []llm.Message
	hasSystem := false
	for _, m := range apiMsgs {
		if m.Role == "system" {
			hasSystem = true
			msgs = append(msgs, llm.Message{Role: "system", Content: combineSystemWithTools(m.Content.Text, toolDefBlock, toolFormat)})
		} else {
			msgs = append(msgs, convertAPIMessage(m))
		}
	}
	if !hasSystem {
		// Prepend a system message with tool definitions
		prepended := []llm.Message{{Role: "system", Content: combineSystemWithTools("", toolDefBlock, toolFormat)}}
		msgs = append(prepended, msgs...)
	}
	return msgs
}

// combineSystemWithTools merges the caller's system content with the rendered
// tool-definition block. The Qwen3.5 native template places the tool block
// first and appends the user system content after it; the legacy protocol
// keeps the original ordering (system content then tool block).
func combineSystemWithTools(systemContent, toolDefBlock, toolFormat string) string {
	if toolFormat == llm.ToolCallFormatQwen35 {
		if systemContent == "" {
			return toolDefBlock
		}
		return toolDefBlock + "\n\n" + systemContent
	}
	if systemContent == "" {
		return toolDefBlock
	}
	return systemContent + "\n\n" + toolDefBlock
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
		return llm.Message{Role: "tool", Content: m.Content.Text}
	}

	return llm.Message{Role: m.Role, Content: m.Content.Text}
}

// formatToolDefinitions renders the OpenAI tools array into the model's native
// tool-definition block. For the Qwen3.5/3.6 protocol this reproduces the
// model's embedded chat template: the function signatures in a <tools>...</tools>
// XML block followed by the exact call-format instructions. For other models it
// falls back to a prose description that asks the model to reply with JSON.
func formatToolDefinitions(tools []Tool, toolFormat string) string {
	if toolFormat == llm.ToolCallFormatQwen35 {
		return formatToolDefinitionsQwen35(tools)
	}
	return formatToolDefinitionsLegacy(tools)
}

// formatToolDefinitionsQwen35 renders tool signatures the way the Qwen3.5/3.6
// Jinja template does: "# Tools" header, a <tools> block with one compact JSON
// object per tool, then the strict <function=...> call-format reminder.
func formatToolDefinitionsQwen35(tools []Tool) string {
	var b strings.Builder
	b.WriteString("# Tools\n\n")
	b.WriteString("You have access to the following functions:\n\n<tools>")
	for _, tool := range tools {
		entry, err := json.Marshal(tool)
		if err != nil {
			continue
		}
		b.WriteString("\n")
		b.Write(entry)
	}
	b.WriteString("\n</tools>")
	b.WriteString("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
		"<tool_call>\n" +
		"<function=example_function_name>\n" +
		"<parameter=example_parameter_1>\n" +
		"value_1\n" +
		"</parameter>\n" +
		"<parameter=example_parameter_2>\n" +
		"This is the value for the second parameter\n" +
		"that can span\n" +
		"multiple lines\n" +
		"</parameter>\n" +
		"</function>\n" +
		"</tool_call>\n\n" +
		"<IMPORTANT>\n" +
		"Reminder:\n" +
		"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
		"- Required parameters MUST be specified\n" +
		"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
		"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
		"</IMPORTANT>")
	return b.String()
}

// formatToolDefinitionsLegacy renders a prose description of the tools and asks
// the model to reply with a bare JSON object (the pre-native protocol).
func formatToolDefinitionsLegacy(tools []Tool) string {
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

// toolCallRe matches a bare JSON function call object of the form
// `{"name": "func_name", "arguments": ...}`. It is only used as a last-resort
// fallback for the legacy (non-native) protocol; native tool calls are parsed
// from <tool_call> blocks by parseNativeToolCalls.
var toolCallRe = regexp.MustCompile(`\{"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}|"[^"]*")\s*\}`)

// parseToolCalls scans generated text for tool calls and returns the parsed
// ToolCall slice. It understands:
//
//   - Qwen3.5/3.6 native calls:
//         <tool_call>
//         <function=NAME>
//         <parameter=KEY>
//         VALUE
//         </parameter>
//         </function>
//         </tool_call>
//   - <tool_call>{ "name": ..., "arguments": ... }</tool_call> (Qwen2.5/3)
//   - a bare { "name": ..., "arguments": ... } object (legacy fallback)
//
// Returns nil if no tool calls are found.
func parseToolCalls(text string) []ToolCall {
	if tcs := parseNativeToolCalls(text); len(tcs) > 0 {
		return finalizeToolCalls(tcs)
	}
	return parseLegacyToolCalls(text)
}

// parseNativeToolCalls extracts calls from <tool_call>...</tool_call> blocks.
// The body may be the native <function=...> XML or a JSON object.
func parseNativeToolCalls(text string) []ToolCall {
	var out []ToolCall
	search := 0
	for {
		open := strings.Index(text[search:], "<tool_call>")
		if open < 0 {
			break
		}
		bodyStart := search + open + len("<tool_call>")
		closeRel := strings.Index(text[bodyStart:], "</tool_call>")
		if closeRel < 0 {
			// Unterminated block (truncated generation): stop.
			break
		}
		body := text[bodyStart : bodyStart+closeRel]
		search = bodyStart + closeRel + len("</tool_call>")

		if tc, ok := parseToolCallBody(body); ok {
			out = append(out, tc)
		}
	}
	return out
}

// parseToolCallBody parses the inner content of one <tool_call> block.
func parseToolCallBody(body string) (ToolCall, bool) {
	trimmed := strings.TrimSpace(body)
	if strings.Contains(trimmed, "<function=") {
		if name, args, ok := parseXMLFunction(trimmed); ok {
			return ToolCall{Type: "function", Function: FunctionCall{Name: name, Arguments: args}}, true
		}
	}
	// JSON body: {"name": ..., "arguments": ...}
	var obj struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(trimmed), &obj); err == nil && obj.Name != "" {
		args := "{}"
		if len(obj.Arguments) > 0 && string(obj.Arguments) != "null" {
			raw := strings.TrimSpace(string(obj.Arguments))
			// arguments may be a JSON string containing JSON; unwrap it.
			if strings.HasPrefix(raw, "\"") {
				var s string
				if json.Unmarshal([]byte(raw), &s) == nil {
					raw = s
				}
			}
			args = raw
		}
		return ToolCall{Type: "function", Function: FunctionCall{Name: obj.Name, Arguments: args}}, true
	}
	return ToolCall{}, false
}

// parseXMLFunction parses a native <function=NAME> ... </function> block and
// returns the function name plus an arguments JSON string built from the
// <parameter=KEY>VALUE</parameter> pairs.
func parseXMLFunction(body string) (name string, args string, ok bool) {
	fidx := strings.Index(body, "<function=")
	if fidx < 0 {
		return "", "", false
	}
	rest := body[fidx+len("<function="):]
	gt := strings.IndexByte(rest, '>')
	if gt < 0 {
		return "", "", false
	}
	name = strings.TrimSpace(rest[:gt])
	if name == "" {
		return "", "", false
	}
	// Body of the function: up to </function>.
	inner := rest[gt+1:]
	if end := strings.Index(inner, "</function>"); end >= 0 {
		inner = inner[:end]
	}

	argsMap := map[string]interface{}{}
	search := 0
	for {
		p := strings.Index(inner[search:], "<parameter=")
		if p < 0 {
			break
		}
		p += search + len("<parameter=")
		gtv := strings.IndexByte(inner[p:], '>')
		if gtv < 0 {
			break
		}
		key := strings.TrimSpace(inner[p : p+gtv])
		valStart := p + gtv + 1
		closeRel := strings.Index(inner[valStart:], "</parameter>")
		if closeRel < 0 {
			break
		}
		raw := inner[valStart : valStart+closeRel]
		search = valStart + closeRel + len("</parameter>")

		// The template wraps each value in a leading and trailing newline.
		raw = strings.TrimPrefix(raw, "\n")
		raw = strings.TrimSuffix(raw, "\n")
		argsMap[key] = decodeArgValue(raw)
	}

	encoded, err := json.Marshal(argsMap)
	if err != nil {
		encoded = []byte("{}")
	}
	return name, string(encoded), true
}

// decodeArgValue interprets a raw parameter value: if it is valid JSON (object,
// array, number, bool, null) it is used as-is; otherwise it is treated as a
// plain string. This lets structured arguments survive the string-based
// transport while keeping free-form text (e.g. source code) as a string.
func decodeArgValue(raw string) interface{} {
	t := strings.TrimSpace(raw)
	if t == "" {
		return ""
	}
	var v interface{}
	if err := json.Unmarshal([]byte(t), &v); err == nil {
		switch v.(type) {
		case map[string]interface{}, []interface{}, float64, bool, nil:
			return v
		}
	}
	return raw
}

// parseLegacyToolCalls is the pre-native fallback: scrape a bare JSON object.
func parseLegacyToolCalls(text string) []ToolCall {
	matches := toolCallRe.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return nil
	}

	tcs := make([]ToolCall, 0, len(matches))
	for i, match := range matches {
		name := match[1]
		argsRaw := match[2]

		var args string
		if strings.HasPrefix(argsRaw, "\"") {
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

// stripToolCalls removes <tool_call>...</tool_call> blocks (and any trailing
// legacy bare-JSON tool object) from generated text so the remaining natural
// language can be returned as the message content.
func stripToolCalls(text string) string {
	for {
		open := strings.Index(text, "<tool_call>")
		if open < 0 {
			break
		}
		closeRel := strings.Index(text[open:], "</tool_call>")
		if closeRel < 0 {
			// Unterminated tool block: drop the rest.
			text = text[:open]
			break
		}
		text = text[:open] + text[open+closeRel+len("</tool_call>"):]
	}
	// Fallback: strip any legacy bare-JSON object.
	if strings.Contains(text, `"arguments"`) {
		text = toolCallRe.ReplaceAllString(text, "")
	}
	cleaned := strings.TrimSpace(text)
	cleaned = strings.TrimPrefix(cleaned, "<|tool_call|>")
	return strings.TrimSpace(cleaned)
}
