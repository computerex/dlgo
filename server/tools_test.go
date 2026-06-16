package server

import (
	"encoding/json"
	"testing"
)

func TestFormatToolDefinitions(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: ToolDefinition{
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

	result := formatToolDefinitions(tools)

	if result == "" {
		t.Fatal("formatToolDefinitions returned empty string")
	}

	if !containsString(result, "get_weather") {
		t.Error("result should contain tool name 'get_weather'")
	}

	if !containsString(result, "Get the current weather") {
		t.Error("result should contain tool description")
	}

	if !containsString(result, "tool_calls") {
		t.Error("result should contain tool_calls format instructions")
	}

	t.Logf("Formatted tool definitions:\n%s", result)
}

func TestFormatToolDefinitions_Multiple(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: ToolDefinition{
				Name:        "get_weather",
				Description: "Get weather info",
			},
		},
		{
			Type: "function",
			Function: ToolDefinition{
				Name:        "search",
				Description: "Search the web",
			},
		},
	}

	result := formatToolDefinitions(tools)

	if !containsString(result, "get_weather") {
		t.Error("result should contain 'get_weather'")
	}

	if !containsString(result, "search") {
		t.Error("result should contain 'search'")
	}
}

func TestBuildToolPrompt(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: ToolDefinition{
				Name:        "get_weather",
				Description: "Get weather info",
			},
		},
	}

	messages := []Message{
		{Role: "user", Content: "What's the weather in NYC?"},
	}

	result := buildToolPrompt(messages, tools)

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
}

func TestBuildToolPrompt_ExistingSystem(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: ToolDefinition{
				Name:        "get_weather",
				Description: "Get weather info",
			},
		},
	}

	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What's the weather in NYC?"},
	}

	result := buildToolPrompt(messages, tools)

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

func TestBuildToolPrompt_NoTools(t *testing.T) {
	messages := []Message{
		{Role: "user", Content: "Hello"},
	}

	result := buildToolPrompt(messages, nil)

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}

	if result[0].Content != "Hello" {
		t.Errorf("message content should be unchanged, got '%s'", result[0].Content)
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
			name: "valid tool call",
			input: `{"tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"NYC\"}"}}]}`,
			wantOk:   true,
			wantName: "get_weather",
		},
		{
			name: "markdown fenced tool call",
			input: "I'll check the weather for you.\n\n" + "```" + "json\n{\"tool_calls\": [{\"id\": \"call_123\", \"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"arguments\": \"{\\\"location\\\": \\\"NYC\\\"}\"}}]}\n" + "```",
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
		{
			name:     "incomplete JSON",
			input:    `{"tool_calls": [`,
			wantOk:   false,
			wantName: "",
		},
		{
			name:     "empty tool_calls array",
			input:    `{"tool_calls": []}`,
			wantOk:   false,
			wantName: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls, ok := parseToolCalls(tt.input)
			if ok != tt.wantOk {
				t.Errorf("parseToolCalls() ok = %v, want %v", ok, tt.wantOk)
			}
			if ok && len(calls) > 0 {
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

func TestIsToolCallResponse(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  bool
	}{
		{
			name:  "raw JSON tool call",
			input: `{"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]}`,
			want:  true,
		},
		{
			name:  "fenced JSON tool call",
			input: "\n" + "```" + "\n{\"tool_calls\": [{\"id\": \"call_1\", \"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"arguments\": \"{}\"}}]}\n" + "```" + "\n",
			want:  true,
		},
		{
			name:  "plain text",
			input: "The weather is sunny today.",
			want:  false,
		},
		{
			name:  "regular JSON",
			input: `{"name": "John", "age": 30}`,
			want:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isToolCallResponse(tt.input); got != tt.want {
				t.Errorf("isToolCallResponse() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestExtractJSON(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "raw JSON",
			input: `{"tool_calls": []}`,
			want:  `{"tool_calls": []}`,
		},
		{
			name:  "markdown fenced",
			input: "\n" + "```" + "json\n{\"key\": \"value\"}\n" + "```",
			want:  `{"key": "value"}`,
		},
		{
			name:  "fenced with surrounding text",
			input: "Here is the result:\n\n" + "```" + "json\n{\"key\": \"value\"}\n" + "```" + "\n\nDone.",
			want:  `{"key": "value"}`,
		},
		{
			name:  "no JSON",
			input: "No JSON here",
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractJSON(tt.input)
			if got != tt.want {
				t.Errorf("extractJSON() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestFormatToolResult(t *testing.T) {
	msg := formatToolResult("call_123", "Sunny, 72°F")

	if msg.Role != "tool" {
		t.Errorf("expected role 'tool', got '%s'", msg.Role)
	}

	if msg.ToolCallID != "call_123" {
		t.Errorf("expected tool_call_id 'call_123', got '%s'", msg.ToolCallID)
	}

	if msg.Content != "Sunny, 72°F" {
		t.Errorf("expected content 'Sunny, 72°F', got '%s'", msg.Content)
	}
}

func TestParseToolCalls_MissingArguments(t *testing.T) {
	input := `{"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}`

	calls, ok := parseToolCalls(input)
	if !ok {
		t.Fatal("parseToolCalls should succeed")
	}

	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}

	if calls[0].Function.Name != "get_weather" {
		t.Errorf("expected function name 'get_weather', got '%s'", calls[0].Function.Name)
	}
}

func TestToolCallJSONRoundTrip(t *testing.T) {
	// Test that tool calls survive JSON marshaling/unmarshaling
	original := []ToolCall{
		{
			ID:   "call_abc123",
			Type: "function",
			Function: FuncCall{
				Name:      "get_weather",
				Arguments: json.RawMessage(`{"location": "NYC"}`),
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

func TestFormatToolDefinitions_EmptyParams(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: ToolDefinition{
				Name:        "no_params_tool",
				Description: "A tool with no parameters",
			},
		},
	}

	result := formatToolDefinitions(tools)

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
	input := `{"tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"NYC\"}"}}]}`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parseToolCalls(input)
	}
}

func BenchmarkFormatToolDefinitions(b *testing.B) {
	tools := []Tool{
		{
			Type: "function",
			Function: ToolDefinition{
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
		formatToolDefinitions(tools)
	}
}
