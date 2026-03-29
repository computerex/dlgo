package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "invalid_request_error")
		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages array is required", "invalid_request_error")
		return
	}

	model := s.manager.GetModel(req.Model)
	if model == nil {
		// If only one model is loaded and no model specified, use it
		models := s.manager.ListModels()
		if req.Model == "" && len(models) == 1 {
			model = s.manager.GetModel(models[0].ID)
		}
		if model == nil {
			writeError(w, http.StatusNotFound, fmt.Sprintf("model %q not found", req.Model), "not_found_error")
			return
		}
	}

	// Convert messages
	msgs := make([]llm.Message, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = llm.Message{Role: m.Role, Content: m.Content}
	}

	// Build sampler config
	sampler := ops.DefaultSamplerConfig()
	if req.Temperature != nil {
		sampler.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		sampler.TopP = float32(*req.TopP)
	}
	if req.TopK != nil {
		sampler.TopK = *req.TopK
	}
	if req.RepetitionPenalty != nil {
		sampler.RepetitionPenalty = float32(*req.RepetitionPenalty)
	}

	maxTokens := 8192
	if req.MaxTokens > 0 {
		maxTokens = req.MaxTokens
	}

	seed := int64(-1)
	if req.Seed != nil {
		seed = *req.Seed
	}

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	// Submit to the scheduler
	infReq := &InferenceRequest{
		ID:              newCompletionID(),
		Messages:        msgs,
		StopSequences:   req.Stop,
		ReasoningEffort: req.ReasoningEffort,
		EnableThinking:  req.EnableThinking,
		Config: llm.GenerateConfig{
			MaxTokens: maxTokens,
			Sampler:   sampler,
			Seed:      seed,
		},
		Output: make(chan StreamEvent, 64),
		Ctx:    ctx,
		Cancel: cancel,
	}

	if err := model.Scheduler.Submit(infReq); err != nil {
		writeError(w, http.StatusServiceUnavailable, "scheduler busy: "+err.Error(), "server_error")
		return
	}

	if req.Stream {
		s.handleStreamResponse(w, r, infReq, req.Model)
	} else {
		s.handleNonStreamResponse(w, infReq, req.Model)
	}
}

func (s *Server) handleStreamResponse(w http.ResponseWriter, r *http.Request, infReq *InferenceRequest, modelID string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported", "server_error")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	for ev := range infReq.Output {
		switch ev.Type {
		case EventToken:
			chunk := ChatCompletionChunk{
				ID:      infReq.ID,
				Object:  "chat.completion.chunk",
				Created: nowUnix(),
				Model:   modelID,
				Choices: []ChatCompletionChoice{{
					Index: 0,
					Delta: &Message{Role: "assistant", Content: ev.Token},
				}},
			}
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()

		case EventDone:
			chunk := ChatCompletionChunk{
				ID:      infReq.ID,
				Object:  "chat.completion.chunk",
				Created: nowUnix(),
				Model:   modelID,
				Choices: []ChatCompletionChoice{{
					Index:        0,
					Delta:        &Message{},
					FinishReason: strPtr(ev.FinishReason),
				}},
			}
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()

		case EventError:
			log.Printf("stream error for %s: %s", infReq.ID, ev.Error)
			return
		}
	}
}

func (s *Server) handleNonStreamResponse(w http.ResponseWriter, infReq *InferenceRequest, modelID string) {
	var fullText string
	var reasoningContent string
	var promptTokens, completionTokens int
	finishReason := "stop"

	for ev := range infReq.Output {
		switch ev.Type {
		case EventToken:
			fullText += ev.Token
			completionTokens++
		case EventDone:
			promptTokens = ev.PromptTokens
			reasoningContent = ev.ReasoningContent
			if ev.FinishReason != "" {
				finishReason = ev.FinishReason
			}
		case EventError:
			writeError(w, http.StatusInternalServerError, ev.Error, "server_error")
			return
		}
	}

	fullText = trimTrailingStopTokens(fullText)

	// Post-generation cleanup: strip any <think>...</think> blocks or stray
	// </think> tags that leaked through the scheduler's inline parsing.
	fullText, extractedReasoning := stripThinkTags(fullText)
	if extractedReasoning != "" && reasoningContent == "" {
		reasoningContent = extractedReasoning
	}

	resp := ChatCompletionResponse{
		ID:      infReq.ID,
		Object:  "chat.completion",
		Created: nowUnix(),
		Model:   modelID,
		Choices: []ChatCompletionChoice{{
			Index:        0,
			Message:      &Message{Role: "assistant", Content: fullText, ReasoningContent: reasoningContent},
			FinishReason: &finishReason,
		}},
		Usage: &UsageInfo{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func trimTrailingStopTokens(text string) string {
	stops := []string{
		"<|end|>", "<|return|>", "<|im_end|>", "<|endoftext|>",
		"<end_of_turn><eos>", "<end_of_turn>", "<eos>", "</s>",
		"<|eot_id|>", "<|assistant|>", "<|user|>",
		"<|channel|>", "<|start|>", "<|message|>", "<|constrain|>", "<|call|>",
	}
	for {
		trimmed := strings.TrimRight(text, " \t\r\n")
		changed := false
		for _, s := range stops {
			if strings.HasSuffix(trimmed, s) {
				trimmed = strings.TrimSuffix(trimmed, s)
				changed = true
			}
		}
		if !changed {
			return trimmed
		}
		text = trimmed
	}
}

// stripThinkTags removes <think>...</think> blocks and stray </think> tags
// from generated text. Returns cleaned content and any extracted reasoning.
func stripThinkTags(text string) (content string, reasoning string) {
	if idx := strings.Index(text, "</think>"); idx >= 0 {
		reasoning = strings.TrimSpace(text[:idx])
		content = strings.TrimLeft(text[idx+len("</think>"):], "\n")
		reasoning = strings.TrimPrefix(reasoning, "<think>")
		reasoning = strings.TrimSpace(reasoning)
		return content, reasoning
	}
	return text, ""
}

