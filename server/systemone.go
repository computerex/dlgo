package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/computerex/dlgo/models/llm"
)

// This file implements the POST /v1/systemone endpoint with two wire shapes:
//
//  1. Jev-native shape (TypeSafe System One API contract, verified against
//     docs.typesafe.ai and its SDK mirrors): request {model, state, questions}
//     where questions is a MAP keyed by caller-chosen ids and each question is
//     {type, instructions, criteria}; response {model, answers (map keyed by
//     the same ids), usage}. Choice answers are {type, choice, probabilities,
//     confidence}; Score answers are {type, score, legend, probabilities,
//     confidence}; Noul answers are {type, noul} with no confidence.
//
//  2. dlgo legacy shape (kept for backward compatibility): top-level
//     type/instructions/criteria or a questions ARRAY with id/prompt fields,
//     flat response with value/confidence/choices and the dlgo extensions
//     (readout, min_confidence, degraded, engine, abstained).
//
// Both shapes may be mixed: Jev-native fields are accepted alongside the dlgo
// extension fields, and every response carries both the Jev answers map and
// the legacy fields. See docs/systemone-jev-compat.md for the full contract.

// SystemOneRequest is the JSON body for a native System One decision.
// Type selects the primitive: "choice" (default), "noul" (yes/no), or "score"
// (discretized numeric range using Min/Max). Questions allows several typed
// questions against the same state in one request — either as the Jev-native
// map (id -> question) or the legacy array (with id fields).
type SystemOneRequest struct {
	Model         string            `json:"model"`
	Readout       string            `json:"readout"`
	Type          string            `json:"type,omitempty"`
	State         json.RawMessage   `json:"state"`
	Instructions  string            `json:"instructions,omitempty"`
	Prompt        string            `json:"prompt,omitempty"`
	Criteria      map[string]string `json:"criteria,omitempty"`
	Min           int               `json:"min,omitempty"`
	Max           int               `json:"max,omitempty"`
	MinConfidence float64           `json:"min_confidence,omitempty"`
	Questions     json.RawMessage   `json:"questions,omitempty"`
}

// SystemOneQuestion is one typed question in a multi-question request
// (legacy array shape). JevShape marks a question parsed from the Jev-native
// map form.
type SystemOneQuestion struct {
	ID       string            `json:"id"`
	Type     string            `json:"type,omitempty"`
	Prompt   string            `json:"prompt"`
	Criteria map[string]string `json:"criteria,omitempty"`
	Min      int               `json:"min,omitempty"`
	Max      int               `json:"max,omitempty"`
}

// SystemOneResponse is the typed decision result.
type SystemOneResponse struct {
	Model        string            `json:"model"`
	Value        string            `json:"value"`
	Confidence   float64           `json:"confidence"`
	Abstained    bool              `json:"abstained"`
	Choices      []llm.ChoiceScore `json:"choices"`
	PromptTokens int               `json:"prompt_tokens"`
	// Degraded reports that the request exceeded the trained readout's option
	// capacity and was answered by generic continuation scoring instead.
	Degraded bool `json:"degraded,omitempty"`
	// Engine reports which engine served the decision: "gpu" or "cpu".
	Engine string `json:"engine,omitempty"`
}

// readoutCache caches loaded readout artifacts per path.
var readoutCache = struct {
	sync.Mutex
	m map[string]*llm.SystemOneReadout
}{m: make(map[string]*llm.SystemOneReadout)}

func cachedReadout(path string, hiddenSize int) (*llm.SystemOneReadout, error) {
	readoutCache.Lock()
	defer readoutCache.Unlock()
	if r, ok := readoutCache.m[path]; ok {
		return r, nil
	}
	r, err := llm.LoadSystemOneReadout(path, hiddenSize)
	if err != nil {
		return nil, err
	}
	readoutCache.m[path] = r
	return r, nil
}

func (s *Server) handleSystemOne(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}
	var req SystemOneRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error(), "invalid_request_error")
		return
	}
	if req.Model == "" || req.Readout == "" || len(req.State) == 0 {
		writeError(w, http.StatusBadRequest, "model, readout, and state are required", "invalid_request_error")
		return
	}
	if req.Type != "multi" && req.Instructions == "" && req.Prompt == "" && len(req.Questions) == 0 {
		writeError(w, http.StatusBadRequest, "instructions or questions are required", "invalid_request_error")
		return
	}
	// The natural single-question form is {"prompt": "...", "criteria": {...}}
	// (mirroring the questions-map shape); "instructions" is the legacy alias.
	if req.Instructions == "" && req.Prompt != "" {
		req.Instructions = req.Prompt
	}
	loaded := s.manager.GetModel(req.Model)
	if loaded == nil || loaded.CPUPipeline == nil {
		writeError(w, http.StatusNotFound, "model not loaded: "+req.Model, "model_not_found")
		return
	}
	// System One evaluation drives the same pipelines as generation (KV cache,
	// run state, GPU command buffers), so it must be mutually exclusive with
	// the generation scheduler loop for this model.
	if loaded.Scheduler == nil {
		writeError(w, http.StatusServiceUnavailable, "model scheduler is not running", "server_error")
		return
	}
	serveSystemOneResponse(r.Context(), w, loaded.Scheduler, func(buffer http.ResponseWriter) error {
		return s.evaluateSystemOneContext(r.Context(), buffer, req, loaded)
	})
}

// serveSystemOneResponse buffers the wire representation while pipeline state
// is exclusively owned. No client I/O takes place under the inference gate,
// and a canceled request never publishes a partial or completed response.
func serveSystemOneResponse(ctx context.Context, w http.ResponseWriter, scheduler *Scheduler, evaluate func(http.ResponseWriter) error) {
	buffer := &systemOneResponseBuffer{header: make(http.Header)}
	waitStart := time.Now()
	var queueWait, evaluation time.Duration
	err := scheduler.WithInferenceLockContext(ctx, func() error {
		evaluationStart := time.Now()
		queueWait = evaluationStart.Sub(waitStart)
		err := evaluate(buffer)
		evaluation = time.Since(evaluationStart)
		return err
	})
	if ctx.Err() != nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return
	}
	// Durations are milliseconds per Server-Timing. Evaluation includes local
	// preparation and buffered serialization, but excludes client network I/O.
	timing := fmt.Sprintf("queue;dur=%.6f, decision;dur=%.6f", float64(queueWait)/float64(time.Millisecond), float64(evaluation)/float64(time.Millisecond))
	w.Header().Set("Server-Timing", timing)
	if err != nil {
		var he *systemOneHTTPError
		if errors.As(err, &he) {
			writeError(w, he.status, he.message, he.code)
			return
		}
		writeError(w, http.StatusInternalServerError, "evaluate: "+err.Error(), "server_error")
		return
	}
	for key, values := range buffer.header {
		w.Header()[key] = append([]string(nil), values...)
	}
	w.Header().Set("Server-Timing", timing)
	if buffer.status != 0 {
		w.WriteHeader(buffer.status)
	}
	_, _ = w.Write(buffer.Bytes())
}

type systemOneResponseBuffer struct {
	bytes.Buffer
	header http.Header
	status int
}

func (b *systemOneResponseBuffer) Header() http.Header { return b.header }
func (b *systemOneResponseBuffer) WriteHeader(status int) {
	if b.status == 0 {
		b.status = status
	}
}
func (b *systemOneResponseBuffer) Write(data []byte) (int, error) {
	b.WriteHeader(http.StatusOK)
	return b.Buffer.Write(data)
}

// systemOneHTTPError carries an HTTP-shaped failure out of the locked
// evaluation section so writeError is called without holding the lock.
type systemOneHTTPError struct {
	status  int
	message string
	code    string
}

func (e *systemOneHTTPError) Error() string { return e.message }

// systemOneStateText renders a wire state (string, JSON object, or array) into
// the plain text the decision prompt is compiled from. Objects render as
// "key: value" lines (nested values as compact JSON) and arrays as one line
// per element, mirroring Jev's string|object|array state inputs with a
// deterministic local rendering.
func systemOneStateText(raw json.RawMessage) (string, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return "", fmt.Errorf("state is empty")
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", fmt.Errorf("invalid state string: %w", err)
		}
		return s, nil
	}
	var value interface{}
	dec := json.NewDecoder(strings.NewReader(trimmed))
	dec.UseNumber()
	if err := dec.Decode(&value); err != nil {
		return "", fmt.Errorf("invalid state JSON: %w", err)
	}
	var b strings.Builder
	renderStateValue(&b, value, "")
	out := strings.TrimRight(b.String(), "\n")
	if out == "" {
		return "", fmt.Errorf("state renders to empty text")
	}
	return out, nil
}

func renderStateValue(b *strings.Builder, value interface{}, key string) {
	prefix := ""
	if key != "" {
		prefix = key + ": "
	}
	switch v := value.(type) {
	case map[string]interface{}:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			renderStateValue(b, v[k], k)
		}
	case []interface{}:
		for i, item := range v {
			itemKey := key
			if itemKey != "" {
				itemKey = fmt.Sprintf("%s[%d]", key, i)
			}
			renderStateValue(b, item, itemKey)
		}
	case string:
		b.WriteString(prefix + v + "\n")
	case nil:
		b.WriteString(prefix + "\n")
	case bool:
		b.WriteString(fmt.Sprintf("%s%v\n", prefix, v))
	default: // json.Number
		b.WriteString(prefix + v.(json.Number).String() + "\n")
	}
}

// systemOneInstructionText renders wire instructions (string, object, or
// array) into prompt text using the same deterministic rendering as state.
func systemOneInstructionText(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return "", nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", fmt.Errorf("invalid instructions string: %w", err)
		}
		return s, nil
	}
	// Object/array instructions: render like state text.
	return systemOneStateText(raw)
}

// jevWireQuestion is the Jev-native question form: type + instructions +
// criteria, where criteria is type-specific (choice: map option->description;
// score: ordered array of 2-10 level descriptions; noul: optional map with
// true/false descriptions).
type jevWireQuestion struct {
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria"`
}

// parsedQuestion is the engine-ready form of one wire question.
type parsedQuestion struct {
	ID       string
	Type     string // "choice" | "noul" | "score"
	Prompt   string
	Choices  []llm.Choice
	Min, Max int // legacy score bucket range
	// ScoreLevels holds the ordered level descriptions for Jev-native score
	// questions (empty for legacy min/max bucket scores).
	ScoreLevels []string
	// NoulCriteria holds optional true/false descriptions.
	NoulTrue, NoulFalse string
	HasNoulCriteria     bool
}

// parseSystemOneQuestions accepts both wire shapes:
//   - Jev-native: {"id": {type, instructions, criteria}} map
//   - legacy: [{"id", type, prompt, criteria, min, max}] array
//
// It returns engine-ready questions in a stable order (sorted ids for the map
// form, array order for the legacy form).
func parseSystemOneQuestions(raw json.RawMessage) ([]parsedQuestion, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return nil, nil
	}
	if trimmed[0] == '{' {
		var wire map[string]jevWireQuestion
		if err := json.Unmarshal(raw, &wire); err != nil {
			return nil, fmt.Errorf("invalid questions map: %w", err)
		}
		ids := make([]string, 0, len(wire))
		for id := range wire {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		out := make([]parsedQuestion, 0, len(wire))
		for _, id := range ids {
			pq, err := parseJevQuestion(id, wire[id])
			if err != nil {
				return nil, err
			}
			out = append(out, pq)
		}
		return out, nil
	}
	// Legacy array form.
	var wire []SystemOneQuestion
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("invalid questions array: %w", err)
	}
	out := make([]parsedQuestion, 0, len(wire))
	for i, q := range wire {
		pq, err := convertWireQuestion(q, i)
		if err != nil {
			return nil, err
		}
		qType := q.Type
		if qType == "" {
			qType = questionTypeOf(pq)
		}
		out = append(out, parsedQuestion{
			ID: pq.ID, Type: qType, Prompt: pq.Prompt, Choices: pq.Choices,
			Min: q.Min, Max: q.Max,
		})
	}
	return out, nil
}

// questionTypeOf infers the primitive type of a legacy-converted question
// whose type field was omitted. Only integer-bucket scores (all-numeric
// values) are classified as score; everything else is a choice.
func questionTypeOf(q llm.SystemOneQuestion) string {
	if len(q.Choices) == 2 && q.Choices[0].Value == "yes" && q.Choices[1].Value == "no" {
		return "noul"
	}
	numeric := len(q.Choices) > 0
	for _, c := range q.Choices {
		if _, err := strconv.Atoi(c.Value); err != nil {
			numeric = false
			break
		}
	}
	if numeric {
		return "score"
	}
	return "choice"
}

// parseJevQuestion converts one Jev-native question into engine form.
func parseJevQuestion(id string, wire jevWireQuestion) (parsedQuestion, error) {
	pq := parsedQuestion{ID: id, Type: wire.Type}
	if pq.Type == "" {
		pq.Type = "choice"
	}
	instructions, err := systemOneInstructionText(wire.Instructions)
	if err != nil {
		return pq, fmt.Errorf("question %q: %v", id, err)
	}
	if instructions == "" {
		return pq, fmt.Errorf("question %q: instructions are required", id)
	}
	pq.Prompt = instructions

	switch pq.Type {
	case "choice":
		var criteria map[string]string
		if len(wire.Criteria) > 0 {
			if err := json.Unmarshal(wire.Criteria, &criteria); err != nil {
				return pq, fmt.Errorf("question %q: choice criteria must map options to descriptions", id)
			}
		}
		if len(criteria) < 2 {
			return pq, fmt.Errorf("question %q: choice requires at least two criteria", id)
		}
		if len(criteria) > 255 {
			return pq, fmt.Errorf("question %q: choice supports at most 255 options", id)
		}
		keys := make([]string, 0, len(criteria))
		for key := range criteria {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		pq.Choices = make([]llm.Choice, len(keys))
		for i, key := range keys {
			pq.Choices[i] = llm.Choice{Value: key, Label: criteria[key]}
		}
	case "score":
		var levels []interface{}
		if len(wire.Criteria) > 0 {
			if err := json.Unmarshal(wire.Criteria, &levels); err != nil {
				return pq, fmt.Errorf("question %q: score criteria must be an ordered array of level descriptions", id)
			}
		}
		if len(levels) < 2 || len(levels) > 10 {
			return pq, fmt.Errorf("question %q: score requires 2 to 10 ordered levels, got %d", id, len(levels))
		}
		pq.ScoreLevels = make([]string, len(levels))
		for i, level := range levels {
			text, err := systemOneInstructionText(mustRaw(level))
			if err != nil || strings.TrimSpace(text) == "" {
				return pq, fmt.Errorf("question %q: level %d description is empty", id, i)
			}
			pq.ScoreLevels[i] = text
		}
		pq.Choices = make([]llm.Choice, len(levels))
		for i, level := range pq.ScoreLevels {
			pq.Choices[i] = llm.Choice{Value: fmt.Sprintf("%d", i), Label: level}
		}
	case "noul":
		if len(wire.Criteria) > 0 {
			var criteria map[string]string
			if err := json.Unmarshal(wire.Criteria, &criteria); err != nil {
				return pq, fmt.Errorf("question %q: noul criteria must map true/false to descriptions", id)
			}
			pq.NoulTrue = criteria["true"]
			pq.NoulFalse = criteria["false"]
			pq.HasNoulCriteria = pq.NoulTrue != "" || pq.NoulFalse != ""
		}
		trueLabel, falseLabel := "yes", "no"
		if pq.HasNoulCriteria {
			if pq.NoulTrue != "" {
				trueLabel = pq.NoulTrue
			}
			if pq.NoulFalse != "" {
				falseLabel = pq.NoulFalse
			}
		}
		// Emit choices in sorted-value order (no before yes) so the compiled
		// prompt lists answer codes ascending (" A: no", " B: yes"). Training
		// prompts always list options in code-ascending order; a yes-first
		// order produces " B: yes" before " A: no", which is off-distribution
		// for the readout and measurably flips noul answers.
		pq.Choices = []llm.Choice{{Value: "no", Label: falseLabel}, {Value: "yes", Label: trueLabel}}
	default:
		return pq, fmt.Errorf("question %q: unknown type %q (expected choice, noul, or score)", id, pq.Type)
	}
	return pq, nil
}

func mustRaw(v interface{}) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		return json.RawMessage("null")
	}
	return b
}

// evaluateSystemOne performs the actual decision work while the model's
// inference lock is held. All pipeline state (CPU and GPU) is exclusively
// owned by this call for its duration.
func (s *Server) evaluateSystemOne(w http.ResponseWriter, req SystemOneRequest, loaded *LoadedModel) error {
	return s.evaluateSystemOneContext(context.Background(), w, req, loaded)
}

func (s *Server) evaluateSystemOneContext(ctx context.Context, w http.ResponseWriter, req SystemOneRequest, loaded *LoadedModel) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	pipe := loaded.CPUPipeline
	readout, err := cachedReadout(req.Readout, pipe.Model.Config.EmbeddingDim)
	if err != nil {
		return &systemOneHTTPError{http.StatusBadRequest, "load readout: " + err.Error(), "invalid_request_error"}
	}
	stateText, err := systemOneStateText(req.State)
	if err != nil {
		return &systemOneHTTPError{http.StatusBadRequest, err.Error(), "invalid_request_error"}
	}
	// Prefer the GPU decision engine when the model is GPU-resident; every GPU
	// call falls back to the CPU engine on error so decision availability never
	// depends on GPU health. Device-aware artifacts keep engines honest: a
	// readout trained on CPU features is refused by the GPU engine (and vice
	// versa), so a mismatch falls back instead of serving off-distribution
	// decisions.
	var gpuPipe GpuPipelineInterface
	if loaded.UseGPU && loaded.GpuPipeline != nil && (readout.Device == "gpu" || readout.Device == "any") {
		gpuPipe = loaded.GpuPipeline
	}
	if gpuPipe == nil && loaded.GpuPipeline != nil {
		// The model is GPU-resident: its CPU-side KV/run buffers were freed
		// when the GPU pipeline was created, so the CPU engine cannot serve.
		// Refuse cpu-device readouts clearly instead of failing deep inside
		// the engine with "pipeline is not initialized".
		return &systemOneHTTPError{http.StatusServiceUnavailable, "this model is GPU-resident; a cpu-device readout cannot be served — use a gpu or any-device readout", "invalid_request_error"}
	}
	cfg := llm.SystemOneConfig{MinConfidence: req.MinConfidence, Context: ctx}

	// Build the question list: Jev-native questions map, legacy questions
	// array, or the top-level single-question form.
	var questions []parsedQuestion
	if len(req.Questions) > 0 {
		questions, err = parseSystemOneQuestions(req.Questions)
		if err != nil {
			return &systemOneHTTPError{http.StatusBadRequest, err.Error(), "invalid_request_error"}
		}
		if len(questions) == 0 {
			return &systemOneHTTPError{http.StatusBadRequest, "questions must not be empty", "invalid_request_error"}
		}
	} else if req.Type == "multi" {
		return &systemOneHTTPError{http.StatusBadRequest, "multi requires at least one question", "invalid_request_error"}
	} else {
		// Top-level single question (legacy shape). Jev-native callers use the
		// questions map with one entry instead.
		pq := parsedQuestion{ID: "decision", Type: "choice"}
		switch req.Type {
		case "", "choice":
			if len(req.Criteria) < 2 {
				return &systemOneHTTPError{http.StatusBadRequest, "choice requires at least two criteria", "invalid_request_error"}
			}
			if len(req.Criteria) > 255 {
				return &systemOneHTTPError{http.StatusBadRequest, "choice supports at most 255 options", "invalid_request_error"}
			}
			keys := make([]string, 0, len(req.Criteria))
			for key := range req.Criteria {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			pq.Choices = make([]llm.Choice, len(keys))
			for i, key := range keys {
				pq.Choices[i] = llm.Choice{Value: key, Label: req.Criteria[key]}
			}
		case "noul":
			pq.Type = "noul"
			pq.Choices = []llm.Choice{{Value: "yes", Label: "yes"}, {Value: "no", Label: "no"}}
		case "score":
			pq.Type = "score"
			pq.Min, pq.Max = req.Min, req.Max
			if pq.Max < pq.Min {
				return &systemOneHTTPError{http.StatusBadRequest, fmt.Sprintf("score max %d < min %d", pq.Max, pq.Min), "invalid_request_error"}
			}
			if pq.Max-pq.Min+1 > 64 {
				return &systemOneHTTPError{http.StatusBadRequest, fmt.Sprintf("score range %d..%d has more than 64 buckets", pq.Min, pq.Max), "invalid_request_error"}
			}
			pq.Choices = make([]llm.Choice, 0, pq.Max-pq.Min+1)
			for v := pq.Min; v <= pq.Max; v++ {
				pq.Choices = append(pq.Choices, llm.Choice{Value: fmt.Sprintf("%d", v), Label: fmt.Sprintf("%d", v)})
			}
		default:
			return &systemOneHTTPError{http.StatusBadRequest, "unknown type: " + req.Type + " (expected choice, noul, score, or multi)", "invalid_request_error"}
		}
		if req.Instructions == "" {
			return &systemOneHTTPError{http.StatusBadRequest, "instructions are required", "invalid_request_error"}
		}
		pq.Prompt = req.Instructions
		questions = []parsedQuestion{pq}
	}

	// Evaluate every question through the shared-prefill multi engine (the
	// single-question case is the N=1 special case with identical semantics).
	engineQuestions := make([]llm.SystemOneQuestion, len(questions))
	for i, pq := range questions {
		engineQuestions[i] = llm.SystemOneQuestion{ID: pq.ID, Prompt: pq.Prompt, Choices: pq.Choices}
	}
	var multi []*llm.SystemOneResult
	if gpuPipe != nil {
		// GPU-resident models cannot run CPU continuation scoring (the CPU
		// KV/run buffers were freed when the GPU pipeline was created), so
		// questions wider than the readout are answered with uniform
		// degenerate+abstained placeholders instead of failing the request.
		servable := make([]llm.SystemOneQuestion, 0, len(engineQuestions))
		overflow := make(map[int]bool, len(engineQuestions))
		for i, q := range engineQuestions {
			if readout.MaxOptions > 0 && len(q.Choices) > readout.MaxOptions {
				overflow[i] = true
				continue
			}
			servable = append(servable, q)
		}
		var served []*llm.SystemOneResult
		if len(servable) > 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
			served, err = gpuPipe.SystemOneMulti(stateText, servable, cfg, readout)
			if err != nil {
				// The CPU pipeline on a GPU-resident model has no buffers, so
				// there is no engine left to fall back to for these questions.
				return &systemOneHTTPError{http.StatusServiceUnavailable, "gpu decision engine failed: " + err.Error(), "server_error"}
			}
		}
		multi = make([]*llm.SystemOneResult, len(engineQuestions))
		si := 0
		for i, q := range engineQuestions {
			if overflow[i] {
				multi[i] = uniformDegradedResult(q)
				continue
			}
			if si < len(served) {
				multi[i] = served[si]
				si++
			}
		}
		err = nil
	} else {
		if err := ctx.Err(); err != nil {
			return err
		}
		multi, err = pipe.SystemOneMulti(stateText, engineQuestions, cfg, readout)
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if err != nil {
		return fmt.Errorf("evaluate: %w", err)
	}

	// Shared prefix token count for usage reporting (Jev bills the state once
	// plus the longest question; our shared prefill matches that shape).
	prefixTokens := 0
	if len(multi) > 0 {
		prefixTokens = multi[0].PromptTokens - suffixShare(multi[0])
	}
	longest := 0
	for _, r := range multi {
		if r.PromptTokens > longest {
			longest = r.PromptTokens
		}
	}
	usageInput := longest
	if prefixTokens > 0 {
		usageInput = prefixTokens + longest - prefixTokens
	}

	w.Header().Set("Content-Type", "application/json")
	writeSystemOneResponse(w, req.Model, questions, multi, usageInput)
	return nil
}

// suffixShare estimates the suffix token count of a multi result. The engine
// reports the full per-question prompt length; the shared prefix is common to
// all questions, so the longest question's prompt length is the Jev-style
// usage figure (state ingested once + longest question).
func suffixShare(r *llm.SystemOneResult) int {
	return 0
}

// jevConfidence implements TypeSafe's documented confidence statistic for
// Choice and Score answers: a measure of how concentrated the probability
// distribution is, clamp((n*max(p)-1)/(n-1), 0, 1), where n is the option or
// level count. It is deliberately NOT the selected answer's probability.
func jevConfidence(probabilities map[string]float64) float64 {
	n := len(probabilities)
	if n < 2 {
		return 0
	}
	maxP := 0.0
	for _, p := range probabilities {
		if p > maxP {
			maxP = p
		}
	}
	c := (float64(n)*maxP - 1) / float64(n-1)
	if c < 0 {
		return 0
	}
	if c > 1 {
		return 1
	}
	return c
}

// uniformDegradedResult builds the placeholder answer for a question a
// GPU-resident server cannot serve (more choices than the readout supports,
// and the CPU continuation scorer's buffers were freed when the GPU pipeline
// was created). Probabilities are uniform (no signal), and the answer is
// flagged degraded and abstained so clients know not to rely on it.
func uniformDegradedResult(q llm.SystemOneQuestion) *llm.SystemOneResult {
	result := &llm.SystemOneResult{QuestionID: q.ID, Degraded: true, Abstained: true}
	n := float64(len(q.Choices))
	if n == 0 {
		return result
	}
	result.Choices = make([]llm.ChoiceScore, len(q.Choices))
	for i, c := range q.Choices {
		result.Choices[i] = llm.ChoiceScore{Value: c.Value, Label: c.Label, Probability: 1.0 / n}
	}
	result.Value = q.Choices[0].Value
	return result
}

// writeSystemOneResponse emits the Jev-native answers map plus the legacy
// fields. Choice answers are {type, choice, probabilities, confidence}; Score
// answers are {type, score, legend, probabilities, confidence} where score is
// the probability-weighted mean level index; Noul answers are {type, noul}
// with no confidence (the probability itself is the uncertainty signal).
// Legacy fields (results array with value/confidence/choices) are preserved
// for existing dlgo callers, alongside the dlgo extensions (degraded, engine,
// abstained, min_confidence abstention).
func writeSystemOneResponse(w http.ResponseWriter, model string, questions []parsedQuestion, results []*llm.SystemOneResult, usageInput int) {
	answers := make(map[string]interface{}, len(results))
	legacy := make([]SystemOneResponse, len(results))
	for i, r := range results {
		var pq parsedQuestion
		if i < len(questions) {
			pq = questions[i]
		}
		probabilities := make(map[string]float64, len(r.Choices))
		for _, c := range r.Choices {
			probabilities[c.Value] = c.Probability
		}
		switch pq.Type {
		case "noul":
			// Jev noul: single probability of yes, no confidence field.
			yesP := 0.0
			if len(r.Choices) > 0 && r.Choices[0].Value == "yes" {
				yesP = r.Choices[0].Probability
			} else {
				for _, c := range r.Choices {
					if c.Value == "yes" {
						yesP = c.Probability
					}
				}
			}
			answer := map[string]interface{}{
				"type": "noul",
				"noul": yesP,
			}
			if r.Abstained {
				answer["abstained"] = true
			}
			if r.Degraded {
				answer["degraded"] = true
			}
			answers[r.QuestionID] = answer
		case "score":
			// Probability-weighted mean level index (can land between levels).
			var score float64
			for _, c := range r.Choices {
				idx := 0.0
				fmt.Sscanf(c.Value, "%f", &idx)
				score += idx * c.Probability
			}
			legend := make(map[string]string, len(pq.ScoreLevels))
			if len(pq.ScoreLevels) > 0 {
				for li, level := range pq.ScoreLevels {
					legend[fmt.Sprintf("%d", li)] = level
				}
			} else {
				// Legacy bucket scores: legend echoes bucket -> label.
				for _, c := range r.Choices {
					legend[c.Value] = c.Label
				}
			}
			answer := map[string]interface{}{
				"type":          "score",
				"score":         score,
				"legend":        legend,
				"probabilities": probabilities,
				"confidence":    jevConfidence(probabilities),
			}
			if r.Abstained {
				answer["abstained"] = true
			}
			if r.Degraded {
				answer["degraded"] = true
			}
			answers[r.QuestionID] = answer
		default:
			answer := map[string]interface{}{
				"type":          "choice",
				"choice":        r.Value,
				"probabilities": probabilities,
				"confidence":    jevConfidence(probabilities),
			}
			if r.Abstained {
				answer["abstained"] = true
			}
			if r.Degraded {
				answer["degraded"] = true
			}
			answers[r.QuestionID] = answer
		}

		engine := r.Engine
		if engine == "" {
			engine = "cpu"
		}
		legacy[i] = SystemOneResponse{
			Model: model, Value: r.Value, Confidence: r.Confidence,
			Abstained: r.Abstained, Choices: r.Choices, PromptTokens: r.PromptTokens,
			Degraded: r.Degraded, Engine: engine,
		}
	}
	resp := map[string]interface{}{
		"model":   model,
		"answers": answers,
		"usage": map[string]interface{}{
			"input_tokens":  usageInput,
			"output_tokens": 0,
		},
	}
	if len(results) == 1 {
		resp["value"] = legacy[0].Value
		resp["confidence"] = legacy[0].Confidence
		resp["abstained"] = legacy[0].Abstained
		resp["choices"] = legacy[0].Choices
		resp["prompt_tokens"] = legacy[0].PromptTokens
		if legacy[0].Degraded {
			resp["degraded"] = true
		}
		if legacy[0].Engine != "" {
			resp["engine"] = legacy[0].Engine
		}
	} else {
		resp["results"] = legacy
	}
	json.NewEncoder(w).Encode(resp)
}

// convertWireQuestion maps a legacy wire question to the engine question type,
// validating per-type requirements.
func convertWireQuestion(q SystemOneQuestion, index int) (llm.SystemOneQuestion, error) {
	out := llm.SystemOneQuestion{ID: q.ID, Prompt: q.Prompt}
	if out.ID == "" {
		out.ID = fmt.Sprintf("question-%d", index)
	}
	if out.Prompt == "" {
		return out, fmt.Errorf("question %d: prompt is required", index)
	}
	switch q.Type {
	case "", "choice":
		if len(q.Criteria) < 2 {
			return out, fmt.Errorf("question %d: choice requires at least two criteria", index)
		}
		keys := make([]string, 0, len(q.Criteria))
		for key := range q.Criteria {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		out.Choices = make([]llm.Choice, len(keys))
		for i, key := range keys {
			out.Choices[i] = llm.Choice{Value: key, Label: q.Criteria[key]}
		}
	case "noul":
		// Keep code/prompt order stable with all trained choice prompts:
		// "no" receives A and is emitted before "yes" (B).
		out.Choices = []llm.Choice{{Value: "no", Label: "no"}, {Value: "yes", Label: "yes"}}
	case "score":
		if q.Max < q.Min {
			return out, fmt.Errorf("question %d: max %d < min %d", index, q.Max, q.Min)
		}
		if q.Max-q.Min+1 > 64 {
			return out, fmt.Errorf("question %d: range %d..%d has more than 64 buckets", index, q.Min, q.Max)
		}
		out.Choices = make([]llm.Choice, 0, q.Max-q.Min+1)
		for v := q.Min; v <= q.Max; v++ {
			out.Choices = append(out.Choices, llm.Choice{Value: fmt.Sprintf("%d", v), Label: fmt.Sprintf("%d", v)})
		}
	default:
		return out, fmt.Errorf("question %d: unknown type %q", index, q.Type)
	}
	return out, nil
}

var _ = math.Abs // keep math import if unused by future edits
