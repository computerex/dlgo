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

	// Convert messages, injecting tool definitions and formatting tool results
	msgs := convertMessages(req.Messages, req.Tools)

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

	// When tools are provided, apply grammar-constrained decoding for
	// non-thinking models to prevent malformed JSON at generation time.
	if len(req.Tools) > 0 {
		supportsThinking := llm.GetArchDescriptor(model.CPUPipeline.Model.Config.Architecture).SupportsThinking
		if !supportsThinking {
			g, err := ToolCallGrammar(false)
			if err != nil {
				writeError(w, http.StatusInternalServerError, "failed to build tool call grammar: "+err.Error(), "server_error")
				return
			}
			infReq.Config.Grammar = g
		}
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

	// Buffer full text for tool call detection at the end
	var fullText strings.Builder
	var reasoningContent string

	for ev := range infReq.Output {
		switch ev.Type {
		case EventToken:
			fullText.WriteString(ev.Token)
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
			reasoningContent = ev.ReasoningContent
			collected := fullText.String()

			// Try to detect tool calls in the final output
			tcs := parseToolCalls(collected)
			if len(tcs) > 0 {
				// For streaming, we send a final chunk with the tool_calls
				// and finish_reason "tool_calls". The previously streamed
				// content tokens are still valid (they may contain the
				// function call JSON, which the client should ignore when
				// tool_calls is present).
				chunk := ChatCompletionChunk{
					ID:      infReq.ID,
					Object:  "chat.completion.chunk",
					Created: nowUnix(),
					Model:   modelID,
					Choices: []ChatCompletionChoice{{
						Index: 0,
						Delta: &Message{
							Role:             "assistant",
							ReasoningContent: reasoningContent,
							ToolCalls:        tcs,
						},
						FinishReason: strPtr("tool_calls"),
					}},
				}
				data, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
			} else {
				chunk := ChatCompletionChunk{
					ID:      infReq.ID,
					Object:  "chat.completion.chunk",
					Created: nowUnix(),
					Model:   modelID,
					Choices: []ChatCompletionChoice{{
						Index:        0,
						Delta:        &Message{ReasoningContent: reasoningContent},
						FinishReason: strPtr(ev.FinishReason),
					}},
				}
				data, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
			}
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

	// If thinking didn't complete (no </think> found) and content is empty,
	// use the reasoning content as the response content. This handles cases
	// where aggressive quantization (IQ3_XXS etc.) makes the model verbose
	// in thinking and unable to close </think> within the token budget.
	if fullText == "" && reasoningContent != "" {
		fullText = reasoningContent
		completionTokens = len(strings.Fields(reasoningContent)) // approximate
	}

	// Try to parse tool calls from the generated text.
	// When found, strip the JSON from the content and keep any remaining
	// natural language (the model often generates both).
	tcs := parseToolCalls(fullText)
	if len(tcs) > 0 {
		// Strip tool call JSON from the text to recover natural language
		cleaned := toolCallRe.ReplaceAllString(fullText, "")
		cleaned = strings.TrimSpace(cleaned)
		// The remaining text after stripping tool calls is the content
		cleaned = strings.TrimPrefix(cleaned, "<|tool_call|>")
		cleaned = strings.TrimSpace(cleaned)
		finishReason = "tool_calls"

		resp := ChatCompletionResponse{
			ID:      infReq.ID,
			Object:  "chat.completion",
			Created: nowUnix(),
			Model:   modelID,
			Choices: []ChatCompletionChoice{{
				Index: 0,
				Message: &Message{
					Role:             "assistant",
					Content:          cleaned,
					ReasoningContent: reasoningContent,
					ToolCalls:        tcs,
				},
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
		return
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

