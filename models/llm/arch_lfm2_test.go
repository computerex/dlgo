package llm

import "testing"

// cfgLFM2 mirrors what ParseConfig produces for lfm2moe: the dedicated "lfm2"
// chat template with the LFM2 native tool-call protocol.
func cfgLFM2() ModelConfig {
	return ModelConfig{
		Architecture: "lfm2moe",
		ChatTemplate: "lfm2",
	}
}

// lfmGenPrompt is the generation prompt the LFM2 template emits: the model
// always opens a reasoning block, which it closes with </think>.
const lfmGenPrompt = "<|im_start|>assistant\n thinking"

func TestFormatLFM2MessagesSimpleTurn(t *testing.T) {
	got := FormatChat(cfgLFM2(), "", "What is 2+2?")
	want := "<|im_start|>user\nWhat is 2+2?<|im_end|>\n" + lfmGenPrompt
	if got != want {
		t.Fatalf("prompt mismatch\n got: %q\nwant: %q", got, want)
	}
}

func TestFormatLFM2MessagesSystemTurn(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "You are terse."},
		{Role: "user", Content: "Hi"},
	}
	got := FormatMessages(cfgLFM2(), msgs)
	want := "<|im_start|>system\nYou are terse.<|im_end|>\n" +
		"<|im_start|>user\nHi<|im_end|>\n" + lfmGenPrompt
	if got != want {
		t.Fatalf("prompt mismatch\n got: %q\nwant: %q", got, want)
	}
}

func TestFormatLFM2MessagesToolCall(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "weather in Paris?"},
		{
			Role:    "assistant",
			Content: "",
			ToolCalls: []ToolCallData{{
				Name:      "get_weather",
				Arguments: `{"city":"Paris","days":2,"units":"celsius"}`,
			}},
		},
		{Role: "tool", Content: `{"temp":18}`},
		{Role: "user", Content: "thanks"},
	}
	got := FormatMessages(cfgLFM2(), msgs)
	want := "<|im_start|>user\nweather in Paris?<|im_end|>\n" +
		"<|im_start|>assistant\n<|tool_call_start|>[get_weather(city='Paris', days=2, units='celsius')]<|tool_call_end|><|im_end|>\n" +
		"<|im_start|>tool\n{\"temp\":18}<|im_end|>\n" +
		"<|im_start|>user\nthanks<|im_end|>\n" + lfmGenPrompt
	if got != want {
		t.Fatalf("prompt mismatch\n got: %q\nwant: %q", got, want)
	}
}

func TestFormatLFM2MessagesStripsHistoryThinking(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: " thinkinginternal reasoning</think>First answer."},
		{Role: "user", Content: "second"},
	}
	got := FormatMessages(cfgLFM2(), msgs)
	want := "<|im_start|>user\nfirst<|im_end|>\n" +
		"<|im_start|>assistant\nFirst answer.<|im_end|>\n" +
		"<|im_start|>user\nsecond<|im_end|>\n" + lfmGenPrompt
	if got != want {
		t.Fatalf("prompt mismatch\n got: %q\nwant: %q", got, want)
	}
}

func TestLFM2ArgValueEscaping(t *testing.T) {
	// The template escapes backslash, single quote, newline and CR.
	cases := map[string]string{
		`{"path":"C:\\tmp\\a.txt"}`: `path='C:\\tmp\\a.txt'`,
		`{"q":"it's"}`:              `q='it\'s'`,
		`{"m":"a\nb"}`:              `m='a\nb'`,
		`{"n":3}`:                   `n=3`,
		`{"ok":true}`:               `ok=true`,
		`{"o":{"k":"v"}}`:           `o={"k":"v"}`,
		`{}`:                        ``,
	}
	for in, want := range cases {
		if got := lfmToolArgs(in); got != want {
			t.Errorf("lfmToolArgs(%s) = %q, want %q", in, got, want)
		}
	}
}
