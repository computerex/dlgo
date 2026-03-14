package llm

import (
	"strings"
	"testing"
)

// boolPtr is a helper for *bool literals.
func boolPtr(b bool) *bool { return &b }

// TestThinkingTogglePrompt verifies that FormatMessages produces the correct
// prompt suffix for thinking-capable models when thinking is enabled/disabled.
// This does NOT require a model file on disk.
func TestThinkingTogglePrompt(t *testing.T) {
	cfg := ModelConfig{
		Architecture: "qwen35",
		ChatTemplate: "chatml",
	}

	msgs := []Message{
		{Role: "user", Content: "What is 2+2?"},
	}

	// Default (nil EnableThinking) — should enable thinking for qwen35.
	promptDefault := FormatMessages(cfg, msgs)
	if !strings.HasSuffix(promptDefault, "<think>\n") {
		t.Errorf("default prompt: expected suffix <think>\\n, got: %q", promptDefault[max(0, len(promptDefault)-30):])
	}

	// Explicitly enabled.
	promptOn := FormatMessages(cfg, msgs, FormatOptions{EnableThinking: boolPtr(true)})
	if !strings.HasSuffix(promptOn, "<think>\n") {
		t.Errorf("enabled prompt: expected suffix <think>\\n, got: %q", promptOn[max(0, len(promptOn)-30):])
	}

	// Explicitly disabled — must use empty think block so the model skips thinking.
	promptOff := FormatMessages(cfg, msgs, FormatOptions{EnableThinking: boolPtr(false)})
	if !strings.HasSuffix(promptOff, "<think></think>\n") {
		t.Errorf("disabled prompt: expected suffix <think></think>\\n, got: %q", promptOff[max(0, len(promptOff)-40):])
	}

	// A non-thinking model should produce neither suffix.
	cfgNoThink := ModelConfig{Architecture: "qwen2", ChatTemplate: "chatml"}
	promptNoThink := FormatMessages(cfgNoThink, msgs)
	if strings.Contains(promptNoThink, "<think>") {
		t.Errorf("non-thinking model prompt should not contain <think>, got: %q", promptNoThink)
	}
	// Explicitly disabling on a non-thinking model should also produce nothing.
	promptNoThinkOff := FormatMessages(cfgNoThink, msgs, FormatOptions{EnableThinking: boolPtr(false)})
	if strings.Contains(promptNoThinkOff, "<think>") {
		t.Errorf("non-thinking model + disable prompt should not contain <think>, got: %q", promptNoThinkOff)
	}

	t.Logf("default  prompt tail: %q", last40(promptDefault))
	t.Logf("enabled  prompt tail: %q", last40(promptOn))
	t.Logf("disabled prompt tail: %q", last40(promptOff))
}

func last40(s string) string {
	if len(s) <= 40 {
		return s
	}
	return "..." + s[len(s)-40:]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// TestThinkingTokenizerDiag checks how the tokenizer encodes <think></think>
// so we can confirm whether </think> is a single token in the prompt.
func TestThinkingTokenizerDiag(t *testing.T) {
	const modelFile = "Qwen3.5-0.8B-Q8_0.gguf"
	path := findModel(modelFile)
	if path == "" {
		t.Skipf("model not found: %s", modelFile)
	}
	pipe, err := NewPipeline(path, 256)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	tok := pipe.Tokenizer

	cases := []string{
		"<think>",
		"</think>",
		"<think></think>",
		"<think></think>\n",
	}
	for _, s := range cases {
		ids := tok.Encode(s)
		texts := make([]string, len(ids))
		for i, id := range ids {
			texts[i] = tok.DecodeToken(id)
		}
		t.Logf("Encode(%q) => %v  decoded=%v", s, ids, texts)
	}
	// Also check SpecialTokens
	t.Logf("SpecialTokens with 'think': ")
	for k, v := range tok.SpecialTokens {
		if strings.Contains(k, "think") {
			t.Logf("  %q => %d", k, v)
		}
	}
}

// TestThinkingToggleGeneration runs an end-to-end generation test on the
// Qwen3.5-0.8B model and verifies that:
//   - thinking ON  → generated text contains reasoning content (</think> separator present)
//   - thinking OFF → generated text does NOT contain prolonged thinking; the
//     spurious leading </think> token (model closing its empty think block) is
//     stripped so the consumer receives a direct answer.
func TestThinkingToggleGeneration(t *testing.T) {
	const modelFile = "Qwen3.5-0.8B-Q8_0.gguf"
	path := findModel(modelFile)
	if path == "" {
		t.Skipf("model not found: %s", modelFile)
	}

	pipe, err := NewPipeline(path, 2048)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}

	cfg := DefaultGenerateConfig()
	cfg.MaxTokens = 256
	cfg.Sampler.Temperature = 0 // greedy – deterministic, fast

	msgs := []Message{
		{Role: "user", Content: "What is the capital of France? Reply in one sentence."},
	}

	// ---- thinking ON (default for qwen35) -----------------------------------
	promptOn := FormatMessages(pipe.Model.Config, msgs, FormatOptions{EnableThinking: boolPtr(true)})
	t.Logf("Prompt (thinking ON) tail: %q", last40(promptOn))

	outOn, tpsOn, err := pipe.GenerateText(promptOn, cfg)
	if err != nil {
		t.Fatalf("GenerateText (thinking ON): %v", err)
	}
	t.Logf("Output (thinking ON, %.1f tok/s):\n%s", tpsOn, outOn)

	// When thinking is ON the prompt ends with <think>\n so the model generates
	// thinking content, then </think>, then the answer.
	if !strings.Contains(outOn, "</think>") {
		t.Errorf("thinking ON: expected output to contain </think> separator, got: %q", outOn[:min(200, len(outOn))])
	}

	// ---- thinking OFF -------------------------------------------------------
	pipe2, err := NewPipeline(path, 2048)
	if err != nil {
		t.Fatalf("NewPipeline (2nd): %v", err)
	}

	promptOff := FormatMessages(pipe2.Model.Config, msgs, FormatOptions{EnableThinking: boolPtr(false)})
	t.Logf("Prompt (thinking OFF) tail: %q", last40(promptOff))
	if !strings.HasSuffix(promptOff, "<think></think>\n") {
		t.Errorf("thinking OFF: prompt should end with <think></think>\\n, got tail: %q", last40(promptOff))
	}

	outOff, tpsOff, err := pipe2.GenerateText(promptOff, cfg)
	if err != nil {
		t.Fatalf("GenerateText (thinking OFF): %v", err)
	}
	// Strip the leading </think> the model emits when closing its empty block.
	// (The scheduler does this automatically; for the raw Pipeline path we do it here.)
	outOff = strings.TrimPrefix(outOff, "</think>")
	outOff = strings.TrimLeft(outOff, "\n ")
	t.Logf("Output (thinking OFF cleaned, %.1f tok/s):\n%s", tpsOff, outOff)

	// The final output should not contain any </think> blocks now.
	if strings.Contains(outOff, "</think>") {
		t.Errorf("thinking OFF: cleaned output must not contain </think>, got: %q", outOff[:min(200, len(outOff))])
	}
	// It should contain a direct answer.
	if !strings.Contains(strings.ToLower(outOff), "paris") {
		t.Errorf("thinking OFF: expected 'Paris' in answer, got: %q", outOff[:min(200, len(outOff))])
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
