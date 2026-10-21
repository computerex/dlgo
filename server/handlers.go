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

	// Determine the model's native tool-call protocol so tool definitions and
	// parsing match what the model was trained on.
	toolFormat := llm.GetArchDescriptor(model.CPUPipeline.Model.Config.Architecture).ToolCallFormat

	// Convert messages, injecting tool definitions and formatting tool results
	msgs := convertMessages(req.Messages, req.Tools, toolFormat)

	// Collect image references (multimodal content arrays) in message order.
	var imageRefs []string
	for _, m := range req.Messages {
		imageRefs = append(imageRefs, m.Content.Images...)
	}

	var imageBlocks [][][]float32
	var imageGrids [][2]int
	if len(imageRefs) > 0 {
		if s.vision == nil {
			writeError(w, http.StatusBadRequest, "image input provided but no vision projector is loaded (start the server with -mmproj)", "invalid_request_error")
			return
		}
		for i, ref := range imageRefs {
			img, err := decodeImageRef(ref)
			if err != nil {
				writeError(w, http.StatusBadRequest, fmt.Sprintf("image %d: %v", i, err), "invalid_request_error")
				return
			}
			block, gh, gw, err := s.vision.EncodeWithGrid(img)
			if err != nil {
				writeError(w, http.StatusInternalServerError, fmt.Sprintf("encode image %d: %v", i, err), "server_error")
				return
			}
			imageGrids = append(imageGrids, [2]int{gh, gw})
			if len(block) > 0 && len(block[0]) != model.CPUPipeline.Model.Config.EmbeddingDim {
				writeError(w, http.StatusBadRequest,
					fmt.Sprintf("image embedding dim %d does not match model embedding dim %d", len(block[0]), model.CPUPipeline.Model.Config.EmbeddingDim),
					"invalid_request_error")
				return
			}
			imageBlocks = append(imageBlocks, block)
		}
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
	if req.MinP != nil {
		sampler.MinP = float32(*req.MinP)
	}
	if req.RepetitionPenalty != nil {
		sampler.RepetitionPenalty = float32(*req.RepetitionPenalty)
	}
	if req.FrequencyPenalty != nil {
		sampler.FrequencyPenalty = float32(*req.FrequencyPenalty)
	}
	if req.PresencePenalty != nil {
		sampler.PresencePenalty = float32(*req.PresencePenalty)
	}
	if req.NoRepeatNgramSize != nil {
		sampler.NoRepeatNgramSize = *req.NoRepeatNgramSize
	}

	maxTokens := 8192
	if req.MaxTokens > 0 {
		maxTokens = req.MaxTokens
	}

	seed := int64(-1)
	if req.Seed != nil {
		seed = *req.Seed
	}

	// Agentic tool-calling defaults to non-thinking mode (faster, and in practice
	// more reliable tool arguments) ONLY when the caller gave no reasoning signal.
	// We always honour an explicit enable_thinking, and a positive reasoning_effort
	// keeps thinking enabled so the model reasons first and then emits a tool call.
	if len(req.Tools) > 0 && req.EnableThinking == nil {
		switch strings.ToLower(strings.TrimSpace(req.ReasoningEffort)) {
		case "", "none", "off", "disabled":
			off := false
			req.EnableThinking = &off
		}
	}

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	// Submit to the scheduler
	infReq := &InferenceRequest{
		ID:              newCompletionID(),
		Messages:        msgs,
		Images:          imageBlocks,
		ImageGrids:      imageGrids,
		StopSequences:   req.Stop,
		ReasoningEffort: req.ReasoningEffort,
		EnableThinking:  req.EnableThinking,
		Config: llm.GenerateConfig{
			MaxTokens:  maxTokens,
			Sampler:    sampler,
			Seed:       seed,
			ImageGrids: imageGrids,
		},
		ToolParamTypes: buildToolParamTypes(req.Tools),
		Output: make(chan StreamEvent, 64),
		Ctx:    ctx,
		Cancel: cancel,
	}

	// When tools are provided, apply grammar-constrained decoding only for the
	// legacy (bare-JSON) protocol on non-thinking models. Native protocols
	// (e.g. Qwen3.5) emit structured <tool_call> markers rather than bare JSON,
	// so a JSON grammar would wrongly forbid the model's trained output.
	if len(req.Tools) > 0 && toolFormat == llm.ToolCallFormatNone {
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

	// Content-streaming guard for native tool calls: hold back text that could
	// be, or is, a <tool_call> block so the raw markers are not emitted as
	// assistant content. The parsed tool_calls arrive in the final chunk. Any
	// natural-language prefix before the call is still streamed.
	const tcMarker = "<tool_call>"
	var emitBuf strings.Builder
	suppressed := false
	reasoningStreamed := false

	writeContent := func(txt string) {
		if txt == "" {
			return
		}
		chunk := ChatCompletionChunk{
			ID:      infReq.ID,
			Object:  "chat.completion.chunk",
			Created: nowUnix(),
			Model:   modelID,
			Choices: []ChatCompletionChoice{{
				Index: 0,
				Delta: &Message{Role: "assistant", Content: MessageContent{Text: txt}},
			}},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
	writeReasoning := func(txt string) {
		if txt == "" {
			return
		}
		chunk := ChatCompletionChunk{
			ID:      infReq.ID,
			Object:  "chat.completion.chunk",
			Created: nowUnix(),
			Model:   modelID,
			Choices: []ChatCompletionChoice{{
				Index: 0,
				Delta: &Message{Role: "assistant", ReasoningContent: txt},
			}},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
	flushSafe := func(final bool) {
		if suppressed {
			emitBuf.Reset()
			return
		}
		s := emitBuf.String()
		if final {
			writeContent(s)
			emitBuf.Reset()
			return
		}
		if idx := strings.Index(s, tcMarker); idx >= 0 {
			writeContent(s[:idx])
			suppressed = true
			emitBuf.Reset()
			return
		}
		keep := longestSuffixPrefixLen(s, tcMarker)
		if len(s) > keep {
			writeContent(s[:len(s)-keep])
		}
		emitBuf.Reset()
		emitBuf.WriteString(s[len(s)-keep:])
	}

	for ev := range infReq.Output {
		switch ev.Type {
		case EventToken:
			fullText.WriteString(ev.Token)
			if !suppressed {
				emitBuf.WriteString(ev.Token)
				flushSafe(false)
			}

		case EventReasoningToken:
			// Reasoning deltas go straight through so the client is not blind
			// while the model is still thinking.
			reasoningStreamed = true
			writeReasoning(ev.Token)

		case EventDone:
			reasoningContent = ev.ReasoningContent
			reasoningStreamed = reasoningStreamed || ev.ReasoningStreamed
			flushSafe(true)
			collected := fullText.String()

			// Tool calls can land in the content stream or, if the model emitted
			// them inside its reasoning, in the reasoning content. Extract from
			// content first; only fall back to reasoning when content has none.
			tcs := parseToolCalls(collected, infReq.ToolParamTypes)
			if len(tcs) == 0 && reasoningContent != "" {
				if rtcs := parseToolCalls(reasoningContent, infReq.ToolParamTypes); len(rtcs) > 0 {
					tcs = rtcs
				}
			}
			// Never leak tool-call syntax into the visible reasoning block.
			if reasoningContent != "" && strings.Contains(reasoningContent, "<tool_call>") {
				reasoningContent = stripToolCalls(reasoningContent)
			}
			if len(tcs) > 0 {
				// Send a final chunk carrying the tool_calls with finish_reason
				// "tool_calls".
				finalDelta := &Message{
					Role:             "assistant",
					ReasoningContent: reasoningContent,
					ToolCalls:        tcs,
				}
				if reasoningStreamed {
					finalDelta.ReasoningContent = ""
				}
				chunk := ChatCompletionChunk{
					ID:      infReq.ID,
					Object:  "chat.completion.chunk",
					Created: nowUnix(),
					Model:   modelID,
					Choices: []ChatCompletionChoice{{
						Index:        0,
						Delta:        finalDelta,
						FinishReason: strPtr("tool_calls"),
					}},
				}
				data, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
			} else {
				finalDelta := &Message{ReasoningContent: reasoningContent}
				if reasoningStreamed {
					finalDelta.ReasoningContent = ""
				}
				chunk := ChatCompletionChunk{
					ID:      infReq.ID,
					Object:  "chat.completion.chunk",
					Created: nowUnix(),
					Model:   modelID,
					Choices: []ChatCompletionChoice{{
						Index:        0,
						Delta:        finalDelta,
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

// longestSuffixPrefixLen returns the length of the longest suffix of s that is a
// proper prefix of marker (i.e. could be the beginning of a marker split across
// stream chunks).
func longestSuffixPrefixLen(s, marker string) int {
	max := len(marker) - 1
	if max > len(s) {
		max = len(s)
	}
	for k := max; k > 0; k-- {
		if strings.HasSuffix(s, marker[:k]) {
			return k
		}
	}
	return 0
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
			if ev.GeneratedTokens > 0 {
				completionTokens = ev.GeneratedTokens
			}
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
	// the scheduler may have left everything as reasoning. We handle that fallback
	// AFTER tool-call extraction below, so a tool call hidden in the reasoning is
	// still surfaced as a call instead of being dumped into content.

	// Tool calls can land in the content stream, or — if the model emitted them
	// inside its reasoning — in the reasoning content. Extract from content first;
	// only fall back to reasoning when content has none (avoids duplicates).
	tcs := parseToolCalls(fullText, infReq.ToolParamTypes)
	if len(tcs) == 0 && reasoningContent != "" {
		if rtcs := parseToolCalls(reasoningContent, infReq.ToolParamTypes); len(rtcs) > 0 {
			tcs = rtcs
		}
	}
	// Never leak tool-call syntax into the visible reasoning block.
	if reasoningContent != "" && strings.Contains(reasoningContent, "<tool_call>") {
		reasoningContent = stripToolCalls(reasoningContent)
	}

	if len(tcs) > 0 {
		// Strip tool call blocks from the text to recover any natural language
		// the model emitted alongside the call.
		cleaned := stripToolCalls(fullText)
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
					Content:          MessageContent{Text: cleaned},
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

	// No tool call. If thinking didn't complete and content is empty, fall back
	// to the reasoning content as the response content (aggressive quantizations
	// sometimes can't close the think block within the budget).
	if fullText == "" && reasoningContent != "" {
		fullText = reasoningContent
		if completionTokens == 0 {
			completionTokens = len(strings.Fields(reasoningContent))
		}
	}

	resp := ChatCompletionResponse{
		ID:      infReq.ID,
		Object:  "chat.completion",
		Created: nowUnix(),
		Model:   modelID,
		Choices: []ChatCompletionChoice{{
			Index:        0,
			Message:      &Message{Role: "assistant", Content: MessageContent{Text: fullText}, ReasoningContent: reasoningContent},
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
