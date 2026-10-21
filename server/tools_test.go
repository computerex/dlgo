package server

import (
	"encoding/json"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

func TestFormatToolDefinitions(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: Function{
				Name:        "get_weather",
				Description: "Get the current weather for a location",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "The city and state, e.g. San Francisco, CA",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	result := formatToolDefinitions(tools, "")

	if result == "" {
		t.Fatal("formatToolDefinitions returned empty string")
	}

	if !containsString(result, "get_weather") {
		t.Error("result should contain tool name 'get_weather'")
	}

	if !containsString(result, "Get the current weather") {
		t.Error("result should contain tool description")
	}

	t.Logf("Formatted tool definitions:\n%s", result)
}

func TestFormatToolDefinitions_Multiple(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: Function{
				Name:        "get_weather",
				Description: "Get weather info",
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "search",
				Description: "Search the web",
			},
		},
	}

	result := formatToolDefinitions(tools, "")

	if !containsString(result, "get_weather") {
		t.Error("result should contain 'get_weather'")
	}

	if !containsString(result, "search") {
		t.Error("result should contain 'search'")
	}
}

func TestConvertMessages(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: Function{
				Name:        "get_weather",
				Description: "Get weather info",
			},
		},
	}

	messages := []Message{
		{Role: "user", Content: MessageContent{Text: "What's the weather in NYC?"}},
	}

	result := convertMessages(messages, tools, "")

	// Should have a system message prepended
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}

	if result[0].Role != "system" {
		t.Errorf("first message role should be 'system', got '%s'", result[0].Role)
	}

	if !containsString(result[0].Content, "get_weather") {
		t.Error("system message should contain tool definitions")
	}

	if result[1].Role != "user" {
		t.Errorf("second message role should be 'user', got '%s'", result[1].Role)
	}
}

func TestConvertMessages_ExistingSystem(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: Function{
				Name:        "get_weather",
				Description: "Get weather info",
			},
		},
	}

	messages := []Message{
		{Role: "system", Content: MessageContent{Text: "You are a helpful assistant."}},
		{Role: "user", Content: MessageContent{Text: "What's the weather in NYC?"}},
	}

	result := convertMessages(messages, tools, "")

	// Should still have 2 messages (system message should be updated, not duplicated)
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}

	if result[0].Role != "system" {
		t.Errorf("first message role should be 'system', got '%s'", result[0].Role)
	}

	// Should contain both tool definitions AND original system content
	if !containsString(result[0].Content, "get_weather") {
		t.Error("system message should contain tool definitions")
	}

	if !containsString(result[0].Content, "You are a helpful assistant.") {
		t.Error("system message should contain original system content")
	}
}

func TestConvertMessages_NoTools(t *testing.T) {
	messages := []Message{
		{Role: "user", Content: MessageContent{Text: "Hello"}},
	}

	result := convertMessages(messages, nil, "")

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}

	if result[0].Content != "Hello" {
		t.Errorf("message content should be unchanged, got '%s'", result[0].Content)
	}
}

func TestConvertMessages_ToolCalls(t *testing.T) {
	messages := []Message{
		{Role: "user", Content: MessageContent{Text: "What's the weather?"}},
		{
			Role: "assistant",
			ToolCalls: []ToolCall{{
				ID:   "call_0",
				Type: "function",
				Function: FunctionCall{
					Name:      "get_weather",
					Arguments: `{"city": "Tokyo"}`,
				},
			}},
		},
		{Role: "tool", ToolCallID: "call_0", Content: MessageContent{Text: "22°C, sunny"}},
	}

	result := convertMessages(messages, nil, "")

	if len(result) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(result))
	}

	// Assistant message should have ToolCalls
	if result[1].Role != "assistant" {
		t.Errorf("second message role should be 'assistant', got '%s'", result[1].Role)
	}
	if len(result[1].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result[1].ToolCalls))
	}
	if result[1].ToolCalls[0].Name != "get_weather" {
		t.Errorf("tool call name should be 'get_weather', got '%s'", result[1].ToolCalls[0].Name)
	}

	// Tool result message
	if result[2].Role != "tool" {
		t.Errorf("third message role should be 'tool', got '%s'", result[2].Role)
	}
	if result[2].Content != "22°C, sunny" {
		t.Errorf("third message content mismatch: %q", result[2].Content)
	}
}

func TestParseToolCalls(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantOk   bool
		wantName string
	}{
		{
			name:     "valid simple JSON tool call",
			input:    `{"name": "get_weather", "arguments": {"location": "NYC"}}`,
			wantOk:   true,
			wantName: "get_weather",
		},
		{
			name:     "tool call with <|tool_call|> prefix",
			input:    "<|tool_call|>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}",
			wantOk:   true,
			wantName: "get_weather",
		},
		{
			name:     "plain text",
			input:    "The weather in NYC is sunny.",
			wantOk:   false,
			wantName: "",
		},
		{
			name:     "empty",
			input:    "",
			wantOk:   false,
			wantName: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := parseToolCalls(tt.input)
			gotOk := len(calls) > 0
			if gotOk != tt.wantOk {
				t.Errorf("parseToolCalls() found = %v, want %v", gotOk, tt.wantOk)
			}
			if gotOk && len(calls) > 0 {
				if calls[0].Function.Name != tt.wantName {
					t.Errorf("expected function name '%s', got '%s'", tt.wantName, calls[0].Function.Name)
				}
				if calls[0].ID == "" {
					t.Error("tool call should have an ID")
				}
				if calls[0].Type != "function" {
					t.Errorf("tool call type should be 'function', got '%s'", calls[0].Type)
				}
			}
		})
	}
}

func TestFormatToolDefinitions_EmptyParams(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: Function{
				Name:        "no_params_tool",
				Description: "A tool with no parameters",
			},
		},
	}

	result := formatToolDefinitions(tools, "")

	if !containsString(result, "no_params_tool") {
		t.Error("result should contain tool name")
	}

	if !containsString(result, "A tool with no parameters") {
		t.Error("result should contain description")
	}
}

// Helper function to check if a string contains a substring
func containsString(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || findSubstring(s, substr))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func BenchmarkParseToolCalls(b *testing.B) {
	input := `<|tool_call|>
{"name": "get_weather", "arguments": {"location": "NYC"}}`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parseToolCalls(input)
	}
}

func BenchmarkFormatToolDefinitions(b *testing.B) {
	tools := []Tool{
		{
			Type: "function",
			Function: Function{
				Name:        "get_weather",
				Description: "Get the current weather for a location",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "The city and state",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		formatToolDefinitions(tools, "")
	}
}

func TestToolCallJSONRoundTrip(t *testing.T) {
	// Test that tool calls survive JSON marshaling/unmarshaling
	original := []ToolCall{
		{
			ID:   "call_abc123",
			Type: "function",
			Function: FunctionCall{
				Name:      "get_weather",
				Arguments: `{"location": "NYC"}`,
			},
		},
	}

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var decoded []ToolCall
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if len(decoded) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(decoded))
	}

	if decoded[0].ID != original[0].ID {
		t.Errorf("ID mismatch: %s != %s", decoded[0].ID, original[0].ID)
	}

	if decoded[0].Function.Name != original[0].Function.Name {
		t.Errorf("Name mismatch: %s != %s", decoded[0].Function.Name, original[0].Function.Name)
	}
}

func TestFormatToolDefinitionsQwen35(t *testing.T) {
	tools := []Tool{{
		Type: "function",
		Function: Function{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"city": map[string]interface{}{"type": "string"},
				},
			},
		},
	}}
	out := formatToolDefinitions(tools, "qwen35")
	for _, want := range []string{"# Tools", "<tools>", "</tools>", `"name":"get_weather"`, "<function=example_function_name>", "<IMPORTANT>", "<tool_call>"} {
		if !containsString(out, want) {
			t.Errorf("qwen35 tool definitions missing %q\n%s", want, out)
		}
	}
	// It must not emit the legacy prose "respond with JSON" instruction.
	if containsString(out, "respond with JSON") {
		t.Error("qwen35 tool definitions should not use legacy prose instruction")
	}
}

func TestParseNativeToolCallsXML(t *testing.T) {
	input := "Sure, let me check.\n" +
		"<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>"
	calls := parseToolCalls(input)
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Errorf("name = %q, want get_weather", calls[0].Function.Name)
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &args); err != nil {
		t.Fatalf("arguments not valid JSON: %v (%q)", err, calls[0].Function.Arguments)
	}
	if args["city"] != "Tokyo" {
		t.Errorf("city = %q, want Tokyo", args["city"])
	}
	if args["unit"] != "celsius" {
		t.Errorf("unit = %q, want celsius", args["unit"])
	}
}

func TestParseNativeToolCallsMultilineValue(t *testing.T) {
	input := "<tool_call>\n<function=write_file>\n<parameter=content>\ndef fibonacci():\n    return 1\n</parameter>\n<parameter=path>\n/tmp/fib.py\n</parameter>\n</function>\n</tool_call>"
	calls := parseToolCalls(input)
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &args); err != nil {
		t.Fatalf("arguments not valid JSON: %v", err)
	}
	if args["path"] != "/tmp/fib.py" {
		t.Errorf("path = %q", args["path"])
	}
	if args["content"] != "def fibonacci():\n    return 1" {
		t.Errorf("content = %q", args["content"])
	}
}

func TestParseNativeToolCallsMultiple(t *testing.T) {
	input := "<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"
	calls := parseToolCalls(input)
	if len(calls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(calls))
	}
	if calls[0].Function.Name != "a" || calls[1].Function.Name != "b" {
		t.Errorf("names = %q, %q; want a, b", calls[0].Function.Name, calls[1].Function.Name)
	}
	if calls[1].Function.Arguments != `{"x":1}` {
		t.Errorf("b args = %q, want {\"x\":1}", calls[1].Function.Arguments)
	}
}

func TestParseNativeToolCallsJSONBody(t *testing.T) {
	input := `<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>`
	calls := parseToolCalls(input)
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Errorf("name = %q", calls[0].Function.Name)
	}
}

func TestStripToolCalls(t *testing.T) {
	input := "Here you go.\n<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"
	if got := stripToolCalls(input); got != "Here you go." {
		t.Errorf("stripToolCalls = %q, want %q", got, "Here you go.")
	}
}

func TestConvertMessagesQwen35ToolsFirst(t *testing.T) {
	tools := []Tool{{Type: "function", Function: Function{Name: "f", Description: "d"}}}
	msgs := []Message{
		{Role: "system", Content: MessageContent{Text: "You are helpful."}},
		{Role: "user", Content: MessageContent{Text: "hi"}},
	}
	out := convertMessages(msgs, tools, "qwen35")
	if len(out) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(out))
	}
	sys := out[0].Content
	if !containsString(sys, "<tools>") || !containsString(sys, "You are helpful.") {
		t.Fatalf("system missing expected content: %q", sys)
	}
	// Tool block must precede the user system content (matches the template).
	if !containsStringPrefixBefore(sys, "# Tools", "You are helpful.") {
		t.Errorf("tool block should precede user system content; got %q", sys[:min2(len(sys), 80)])
	}
}

func containsStringPrefixBefore(s, first, second string) bool {
	i := indexOfStr(s, first)
	j := indexOfStr(s, second)
	return i >= 0 && j >= 0 && i < j
}

func indexOfStr(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

func min2(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// TestLegacyToolConventionUnchanged locks in that architectures without a native
// tool protocol (every arch except qwen35/qwen35moe) keep the original
// OpenAI-style behavior: prose tool definitions that ask for bare JSON, and the
// bare-JSON parser/stripper.
func TestLegacyToolConventionUnchanged(t *testing.T) {
	tools := []Tool{{
		Type:     "function",
		Function: Function{Name: "get_weather", Description: "Get weather"},
	}}

	defs := formatToolDefinitions(tools, llm.ToolCallFormatNone)
	if !containsString(defs, "respond with JSON") {
		t.Errorf("legacy tool definitions must ask the model to respond with JSON; got: %s", defs)
	}
	if containsString(defs, "<tools>") || containsString(defs, "<IMPORTANT>") {
		t.Errorf("legacy tool definitions must not use the native <tools> block; got: %s", defs)
	}

	// Bare-JSON tool call (the legacy convention) still parses.
	calls := parseToolCalls(`{"name": "get_weather", "arguments": {"city": "NYC"}}`)
	if len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("legacy bare-JSON parse failed: %+v", calls)
	}

	// And it still strips from content.
	if got := stripToolCalls(`ok {"name": "get_weather", "arguments": {"city": "NYC"}}`); got != "ok" {
		t.Errorf("legacy strip = %q, want ok", got)
	}
}
