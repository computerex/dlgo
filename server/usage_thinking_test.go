package server

import (
	"encoding/json"
	"net/http/httptest"
	"testing"
)

// Thinking-mode tokens are buffered by the scheduler and never emitted as
// EventToken, so usage accounting must take the total from EventDone. This
// guards the regression where completion_tokens reported only the final answer.
func TestNonStreamUsageCountsThinkingTokens(t *testing.T) {
	s := &Server{}
	out := make(chan StreamEvent, 4)
	out <- StreamEvent{Type: EventToken, Token: "The Attribute is LIGHT."}
	out <- StreamEvent{Type: EventDone, PromptTokens: 33, ReasoningContent: "long thinking...",
		GeneratedTokens: 3157, FinishReason: "stop"}
	close(out)
	req := &InferenceRequest{ID: "test", Output: out}
	rec := httptest.NewRecorder()
	s.handleNonStreamResponse(rec, req, "qwen35")

	var resp struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Usage.CompletionTokens != 3157 {
		t.Fatalf("completion_tokens = %d, want 3157 (thinking tokens must count)",
			resp.Usage.CompletionTokens)
	}
	if resp.Usage.TotalTokens != 33+3157 {
		t.Fatalf("total_tokens = %d, want %d", resp.Usage.TotalTokens, 33+3157)
	}
}