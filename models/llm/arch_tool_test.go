package llm

import (
	"strings"
	"testing"
)

// TestFormatChatMLQwen35ToolHistory verifies that assistant tool calls and tool
// results are rendered in the model's native <tool_call>/<tool_response> format.
func TestFormatChatMLQwen35ToolHistory(t *testing.T) {
	cfg := ModelConfig{Architecture: "qwen35", ChatTemplate: "chatml"}

	msgs := []Message{
		{Role: "user", Content: "Weather in Tokyo?"},
		{Role: "assistant", ToolCalls: []ToolCallData{{
			Name:      "get_weather",
			Arguments: `{"city":"Tokyo","unit":"celsius"}`,
		}}},
		{Role: "tool", Content: "22C, sunny"},
	}

	prompt := FormatMessages(cfg, msgs, FormatOptions{EnableThinking: boolPtr(false)})

	wantSubstrings := []string{
		"<|im_start|>user\nWeather in Tokyo?<|im_end|>\n",
		"<tool_call>\n<function=get_weather>\n",
		"<parameter=city>\nTokyo\n</parameter>\n",
		"<parameter=unit>\ncelsius\n</parameter>\n",
		"</function>\n</tool_call>",
		"<|im_start|>user\n<tool_response>\n22C, sunny\n</tool_response><|im_end|>\n",
		"<|im_start|>assistant\n",
	}
	for _, w := range wantSubstrings {
		if !strings.Contains(prompt, w) {
			t.Errorf("prompt missing %q\n--- prompt ---\n%s", w, prompt)
		}
	}
	// The wrong (non-native) markers must not appear.
	for _, bad := range []string{"<|tool_call|>", "<|tool_response|>"} {
		if strings.Contains(prompt, bad) {
			t.Errorf("prompt should not contain legacy marker %q", bad)
		}
	}
}

// TestFormatChatMLQwen35StructuredArg verifies nested JSON arguments render as
// compact JSON (matching the Jinja template's tojson behavior).
func TestFormatChatMLQwen35StructuredArg(t *testing.T) {
	cfg := ModelConfig{Architecture: "qwen35", ChatTemplate: "chatml"}
	msgs := []Message{
		{Role: "user", Content: "go"},
		{Role: "assistant", ToolCalls: []ToolCallData{{
			Name:      "f",
			Arguments: `{"obj":{"a":1},"n":5}`,
		}}},
	}
	prompt := FormatMessages(cfg, msgs, FormatOptions{EnableThinking: boolPtr(false)})
	if !strings.Contains(prompt, "<parameter=obj>\n{\"a\":1}\n</parameter>\n") {
		t.Errorf("nested object not rendered as compact JSON:\n%s", prompt)
	}
	if !strings.Contains(prompt, "<parameter=n>\n5\n</parameter>\n") {
		t.Errorf("numeric value not rendered correctly:\n%s", prompt)
	}
}

// TestFormatChatMLQwen35NoTools ensures normal (non-tool) rendering is unchanged.
func TestFormatChatMLQwen35NoTools(t *testing.T) {
	cfg := ModelConfig{Architecture: "qwen35", ChatTemplate: "chatml"}
	msgs := []Message{
		{Role: "system", Content: "Be nice."},
		{Role: "user", Content: "Hi"},
	}
	prompt := FormatMessages(cfg, msgs, FormatOptions{EnableThinking: boolPtr(false)})
	if !strings.HasPrefix(prompt, "<|im_start|>system\nBe nice.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n") {
		t.Errorf("unexpected non-tool rendering:\n%s", prompt)
	}
}
