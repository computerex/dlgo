package server

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

// EventType identifies the kind of streaming event.
type EventType int

const (
	EventToken EventType = iota
	EventDone
	EventError
)

// StreamEvent is a single event sent from the scheduler to the HTTP handler.
type StreamEvent struct {
	Type             EventType
	Token            string
	ReasoningContent string // thinking block content (set on EventDone)
	FinishReason     string
	PromptTokens     int
	Error            string
}

// RequestStatus tracks where a request is in the pipeline.
type RequestStatus int

const (
	StatusWaiting    RequestStatus = iota
	StatusPrefilling
	StatusDecoding
	StatusDone
	StatusError
)

// InferenceRequest represents a single chat completion request in flight.
type InferenceRequest struct {
	ID              string
	Messages        []llm.Message
	StopSequences   []string      // user-provided stop sequences from API
	ReasoningEffort string        // "low", "medium", "high" (default: "medium")
	EnableThinking  *bool         // nil = auto (enabled for thinking models), false = disable
	Config          llm.GenerateConfig
	Tokens          []int32       // prompt tokens after formatting
	Generated       []int32       // output tokens so far
	Position        int           // current sequence position
	Status          RequestStatus
	Output          chan StreamEvent
	Ctx             context.Context
	Cancel          context.CancelFunc
}

// Scheduler manages the inference loop for a single loaded model.
// It processes requests sequentially for now, with the framework
// ready for continuous batching in the future.
type Scheduler struct {
	mu       sync.Mutex
	model    *LoadedModel
	submit   chan *InferenceRequest
	stop     chan struct{}
	wg       sync.WaitGroup
}

// NewScheduler creates and starts a scheduler for the given model.
func NewScheduler(model *LoadedModel) *Scheduler {
	s := &Scheduler{
		model:  model,
		submit: make(chan *InferenceRequest, 64),
		stop:   make(chan struct{}),
	}
	s.wg.Add(1)
	go s.loop()
	return s
}

// Submit enqueues a request for processing.
func (s *Scheduler) Submit(req *InferenceRequest) error {
	select {
	case s.submit <- req:
		return nil
	default:
		return fmt.Errorf("request queue full")
	}
}

// Stop shuts down the scheduler.
func (s *Scheduler) Stop() {
	close(s.stop)
	s.wg.Wait()
}

func (s *Scheduler) loop() {
	defer s.wg.Done()
	for {
		select {
		case <-s.stop:
			return
		case req := <-s.submit:
			s.processRequest(req)
		}
	}
}

func (s *Scheduler) processRequest(req *InferenceRequest) {
	defer close(req.Output)

	s.mu.Lock()
	defer s.mu.Unlock()

	m := s.model

	// Format messages into a prompt
	fmtOpts := llm.FormatOptions{ReasoningEffort: req.ReasoningEffort, EnableThinking: req.EnableThinking}
	prompt := llm.FormatMessages(m.CPUPipeline.Model.Config, req.Messages, fmtOpts)

	// Tokenize
	tokens := m.CPUPipeline.Tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		req.Output <- StreamEvent{Type: EventError, Error: "tokenizer produced no tokens"}
		return
	}
	req.Tokens = tokens
	promptTokens := len(tokens)

	if promptTokens >= m.CPUPipeline.MaxSeqLen {
		req.Output <- StreamEvent{Type: EventError, Error: fmt.Sprintf("prompt too long: %d tokens (max %d)", promptTokens, m.CPUPipeline.MaxSeqLen)}
		return
	}

	rng := rand.New(rand.NewSource(req.Config.Seed))
	if req.Config.Seed < 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	// Use GPU pipeline if available, otherwise CPU
	if m.GpuPipeline != nil {
		s.processGPU(req, rng, promptTokens)
	} else {
		s.processCPU(req, rng, promptTokens)
	}
}

func (s *Scheduler) processCPU(req *InferenceRequest, rng *rand.Rand, promptTokens int) {
	pipe := s.model.CPUPipeline

	// Reset KV cache
	pipe.KVCache.Reset()
	if pipe.RunState.SSMState != nil {
		pipe.RunState.SSMState.Reset()
	}

	// Prefill
	llm.ForwardBatch(pipe.Model, req.Tokens, 0, pipe.KVCache, pipe.RunState, pipe.BatchState)

	pos := len(req.Tokens)
	var recentTokens []int32

	nextToken := int32(ops.SampleToken(pipe.RunState.Logits, req.Config.Sampler, recentTokens, rng))

	if isStopToken(nextToken, pipe.Model.Config) {
		req.Output <- StreamEvent{Type: EventDone, FinishReason: "stop", PromptTokens: promptTokens}
		return
	}

	stopStrings := collectStopStrings(pipe.Model.Config)
	stopStrings = append(stopStrings, req.StopSequences...)

	supportsThinking := llm.GetArchDescriptor(pipe.Model.Config.Architecture).SupportsThinking

	// Buffer output to separate reasoning from content when thinking is
	// active. When thinking is explicitly disabled, stream tokens normally
	// and let the post-generation stripThinkTags() safety net in handlers.go
	// catch any stray think tags the model may emit.
	thinkingEnabled := supportsThinking && (req.EnableThinking == nil || *req.EnableThinking)
	inThinkBlock := thinkingEnabled
	var thinkingContent string

	tokenText := pipe.Tokenizer.DecodeToken(nextToken)
	var genText strings.Builder
	genText.WriteString(tokenText)

	if checkTextStop(genText.String(), stopStrings) {
		req.Output <- StreamEvent{Type: EventDone, FinishReason: "stop", PromptTokens: promptTokens}
		return
	}

	if inThinkBlock {
		if idx := strings.Index(genText.String(), "</think>"); idx >= 0 {
			inThinkBlock = false
			thinkingContent = strings.TrimSpace(genText.String()[:idx])
			after := genText.String()[idx+len("</think>"):]
			after = strings.TrimLeft(after, "\n")
			if after != "" {
				req.Output <- StreamEvent{Type: EventToken, Token: after}
			}
		}
	} else if tokenText != "" {
		req.Output <- StreamEvent{Type: EventToken, Token: tokenText}
	}
	req.Generated = append(req.Generated, nextToken)
	recentTokens = append(recentTokens, nextToken)

	for step := 1; step < req.Config.MaxTokens; step++ {
		if req.Ctx.Err() != nil {
			req.Output <- StreamEvent{Type: EventDone, FinishReason: "cancelled", PromptTokens: promptTokens}
			return
		}
		if pos >= pipe.MaxSeqLen-1 {
			break
		}

		if isStopToken(nextToken, pipe.Model.Config) {
			break
		}

		llm.Forward(pipe.Model, nextToken, pos, pipe.KVCache, pipe.RunState)
		pos++

		nextToken = int32(ops.SampleToken(pipe.RunState.Logits, req.Config.Sampler, recentTokens, rng))

		if isStopToken(nextToken, pipe.Model.Config) {
			break
		}

		tokenText = pipe.Tokenizer.DecodeToken(nextToken)
		req.Generated = append(req.Generated, nextToken)
		recentTokens = append(recentTokens, nextToken)
		if len(recentTokens) > 256 {
			recentTokens = recentTokens[1:]
		}

		genText.WriteString(tokenText)
		if checkTextStop(genText.String(), stopStrings) {
			break
		}

		if inThinkBlock {
			if idx := strings.Index(genText.String(), "</think>"); idx >= 0 {
				inThinkBlock = false
				thinkingContent = strings.TrimSpace(genText.String()[:idx])
				after := genText.String()[idx+len("</think>"):]
				after = strings.TrimLeft(after, "\n")
				if after != "" {
					req.Output <- StreamEvent{Type: EventToken, Token: after}
				}
			}
		} else {
			req.Output <- StreamEvent{Type: EventToken, Token: tokenText}
		}
	}

	// If still buffering (no </think> found), treat as truncated reasoning.
	if inThinkBlock {
		thinkingContent = strings.TrimSpace(genText.String())
	}

	req.Output <- StreamEvent{Type: EventDone, FinishReason: "stop", PromptTokens: promptTokens, ReasoningContent: thinkingContent}
}

func (s *Scheduler) processGPU(req *InferenceRequest, rng *rand.Rand, promptTokens int) {
	pipe := s.model.GpuPipeline

	// Use GenerateDetailed with streaming callback
	fmtOpts := llm.FormatOptions{ReasoningEffort: req.ReasoningEffort, EnableThinking: req.EnableThinking}
	prompt := llm.FormatMessages(s.model.CPUPipeline.Model.Config, req.Messages, fmtOpts)

	supportsThinking := llm.GetArchDescriptor(s.model.CPUPipeline.Model.Config.Architecture).SupportsThinking

	// Buffer output to separate reasoning from content when thinking is
	// active. When explicitly disabled, stream normally and let handlers.go
	// stripThinkTags() catch any stray think tags.
	thinkingEnabled := supportsThinking && (req.EnableThinking == nil || *req.EnableThinking)
	inThinkBlock := thinkingEnabled
	var thinkBuf strings.Builder
	var thinkingContent string

	cfg := req.Config
	cfg.Stream = func(token string) {
		if req.Ctx.Err() != nil {
			return
		}
		if inThinkBlock {
			thinkBuf.WriteString(token)
			if idx := strings.Index(thinkBuf.String(), "</think>"); idx >= 0 {
				inThinkBlock = false
				thinkingContent = strings.TrimSpace(thinkBuf.String()[:idx])
				after := thinkBuf.String()[idx+len("</think>"):]
				after = strings.TrimLeft(after, "\n")
				if after != "" {
					req.Output <- StreamEvent{Type: EventToken, Token: after}
				}
			}
			return
		}
		req.Output <- StreamEvent{Type: EventToken, Token: token}
	}

	result, err := pipe.GenerateDetailed(prompt, cfg)
	if err != nil {
		req.Output <- StreamEvent{Type: EventError, Error: err.Error()}
		return
	}

	// If still buffering (no </think> found), treat as truncated reasoning.
	if inThinkBlock {
		thinkingContent = strings.TrimSpace(thinkBuf.String())
	}

	finishReason := "stop"
	if result.TotalTokens >= cfg.MaxTokens {
		finishReason = "length"
	}
	req.Output <- StreamEvent{
		Type:             EventDone,
		FinishReason:     finishReason,
		PromptTokens:     result.PromptTokens,
		ReasoningContent: thinkingContent,
	}
}

func isStopToken(token int32, cfg llm.ModelConfig) bool {
	if token == cfg.EOS {
		return true
	}
	for _, st := range cfg.StopTokens {
		if token == st {
			return true
		}
	}
	return false
}

func collectStopStrings(cfg llm.ModelConfig) []string {
	return []string{
		"<end_of_turn><eos>",
		"<eos>",
		"<|im_end|>",
		"<|endoftext|>",
		"<|end|>",
		"<|return|>",
		"</s>",
		"<|assistant|>",
		"<|user|>",
		"<|observation|>",
		"<end_of_turn>",
		"<|eot_id|>",
		"<|channel|>",
		"<|start|>",
		"<|message|>",
		"<|constrain|>",
		"<|call|>",
	}
}

func checkTextStop(text string, stops []string) bool {
	for _, ss := range stops {
		if strings.HasSuffix(text, ss) {
			return true
		}
	}
	return false
}

// Suppress unused import warning
var _ = log.Printf
