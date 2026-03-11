package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"

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

	maxTokens := 512
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
		ID:       newCompletionID(),
		Messages: msgs,
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
	var promptTokens, completionTokens int
	finishReason := "stop"

	for ev := range infReq.Output {
		switch ev.Type {
		case EventToken:
			fullText += ev.Token
			completionTokens++
		case EventDone:
			promptTokens = ev.PromptTokens
			if ev.FinishReason != "" {
				finishReason = ev.FinishReason
			}
		case EventError:
			writeError(w, http.StatusInternalServerError, ev.Error, "server_error")
			return
		}
	}

	resp := ChatCompletionResponse{
		ID:      infReq.ID,
		Object:  "chat.completion",
		Created: nowUnix(),
		Model:   modelID,
		Choices: []ChatCompletionChoice{{
			Index:        0,
			Message:      &Message{Role: "assistant", Content: fullText},
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
