package llm

import (
	"context"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/computerex/dlgo/memory"
)

// Choice is one allowed typed answer to a System One question. Value is the
// stable application value returned to callers; Label is the semantic text
// included in the prompt. Code is the short continuation scored by the model.
type Choice struct {
	Value string
	Label string
	Code  string
}

// SystemOneQuestion describes one closed-set decision. Prompt should explain
// what to decide; state is supplied separately so several questions may reuse
// the same contextual state in future optimized implementations.
type SystemOneQuestion struct {
	ID      string
	Prompt  string
	Choices []Choice
}

// SystemOneConfig controls typed decision prompting and confidence handling.
type SystemOneConfig struct {
	System        string
	MinConfidence float64
	// Context permits cancellation at safe inference boundaries. Nil means
	// no cancellation; in-flight GPU submissions must finish before returning.
	Context context.Context
}

// Err reports cancellation without requiring callers to supply a context.
func (cfg SystemOneConfig) Err() error {
	if cfg.Context != nil {
		return cfg.Context.Err()
	}
	return nil
}

// ChoiceScore is the likelihood and normalized probability of one choice.
type ChoiceScore struct {
	Value       string
	Label       string
	Code        string
	LogProb     float64
	Probability float64
}

// SystemOneResult is a typed closed-set decision result. Confidence is the
// chosen choice's normalized probability, not a calibrated correctness claim.
type SystemOneResult struct {
	QuestionID    string
	Value         string
	Confidence    float64
	Abstained     bool
	Choices       []ChoiceScore
	PromptTokens  int
	ScoredTokens  int
	PrefillTimeMs float64
	ScoreTimeMs   float64
	// SharedPrefillTimeMs reports the one-time shared-prefix prefill cost in
	// multi-question requests (zero for single-question calls).
	SharedPrefillTimeMs float64
	// Degraded reports that the trained readout could not serve this question
	// (more choices than it supports) and generic continuation scoring was
	// used instead. Only set by multi-question evaluation; single-question
	// callers observe degradation through their own response handling.
	Degraded bool
	// Engine reports which engine produced this result ("gpu" or "cpu").
	// Empty means the CPU engine (set only by the GPU pipeline today).
	Engine string
}

// SystemOneHidden returns the final normalized hidden state for a compiled
// System One prompt. It is the frozen-Q4 feature extractor used by the native
// decision-readout trainer. The returned choices contain resolved answer codes.
// Callers must not mutate the pipeline concurrently.
func (p *Pipeline) SystemOneHidden(state string, q SystemOneQuestion, cfg SystemOneConfig) ([]float32, []Choice, int, error) {
	if err := cfg.Err(); err != nil {
		return nil, nil, 0, err
	}
	if p == nil || p.Model == nil || p.Tokenizer == nil || p.KVCache == nil || p.RunState == nil {
		return nil, nil, 0, fmt.Errorf("System One pipeline is not initialized")
	}
	if len(q.Choices) < 2 {
		return nil, nil, 0, fmt.Errorf("System One requires at least two choices")
	}
	if q.Prompt == "" {
		return nil, nil, 0, fmt.Errorf("System One question prompt is empty")
	}
	prompt, choices, err := CompileSystemOnePrompt(p.Model.Config, state, q, cfg)
	if err != nil {
		return nil, nil, 0, err
	}
	tokens := p.Tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return nil, nil, 0, fmt.Errorf("System One tokenizer produced no prompt tokens")
	}
	if len(tokens) >= p.MaxSeqLen {
		return nil, nil, 0, fmt.Errorf("System One prompt too long: %d tokens (max %d)", len(tokens), p.MaxSeqLen)
	}
	p.KVCache.Reset()
	if p.RunState.SSMState != nil {
		p.RunState.SSMState.Reset()
	}
	// ForwardBatch is faster for generation prefill but finishes with the full
	// vocabulary projection. For a native readout, consume the final hidden
	// state directly and skip that projection. The remaining prompt positions
	// still use the batch path to populate the shared KV/SSM state.
	if len(tokens) == 1 {
		return ForwardHidden(p.Model, tokens[0], 0, p.KVCache, p.RunState), choices, len(tokens), nil
	}
	ForwardBatch(p.Model, tokens[:len(tokens)-1], 0, p.KVCache, p.RunState, p.BatchState)
	hidden := ForwardHidden(p.Model, tokens[len(tokens)-1], len(tokens)-1, p.KVCache, p.RunState)
	return hidden, choices, len(tokens), nil
}

// SystemOne evaluates a closed set of choices using continuation likelihoods.
// It is deterministic and does not use token sampling. The prompt is prefilled
// once, then every short answer code is scored from an isolated clone of that
// prefill state. This is deliberately separate from Generate so existing text
// generation behavior remains unchanged.
func (p *Pipeline) SystemOne(state string, q SystemOneQuestion, cfg SystemOneConfig) (*SystemOneResult, error) {
	if err := cfg.Err(); err != nil {
		return nil, err
	}
	if p == nil || p.Model == nil || p.Tokenizer == nil || p.KVCache == nil || p.RunState == nil {
		return nil, fmt.Errorf("System One pipeline is not initialized")
	}
	if len(q.Choices) < 2 {
		return nil, fmt.Errorf("System One requires at least two choices")
	}
	if q.Prompt == "" {
		return nil, fmt.Errorf("System One question prompt is empty")
	}

	prompt, choices, err := CompileSystemOnePrompt(p.Model.Config, state, q, cfg)
	if err != nil {
		return nil, err
	}
	promptTokens := p.Tokenizer.Encode(prompt)
	if len(promptTokens) == 0 {
		return nil, fmt.Errorf("System One tokenizer produced no prompt tokens")
	}
	if len(promptTokens) >= p.MaxSeqLen {
		return nil, fmt.Errorf("System One prompt too long: %d tokens (max %d)", len(promptTokens), p.MaxSeqLen)
	}

	p.KVCache.Reset()
	if p.RunState.SSMState != nil {
		p.RunState.SSMState.Reset()
	}
	prefillStart := time.Now()
	ForwardBatch(p.Model, promptTokens, 0, p.KVCache, p.RunState, p.BatchState)
	prefillMs := float64(time.Since(prefillStart).Microseconds()) / 1000.0

	baseKV := p.KVCache.Clone(len(promptTokens))
	baseSSM := p.RunState.SSMState.Clone()
	baseLogits := append([]float32(nil), p.RunState.Logits...)

	scoreStart := time.Now()
	scores := make([]ChoiceScore, len(choices))
	totalScoredTokens := 0
	for i, choice := range choices {
		candidateTokens := p.Tokenizer.Encode(choice.Code)
		if len(candidateTokens) == 0 {
			return nil, fmt.Errorf("System One choice %q has no tokens", choice.Value)
		}
		if len(promptTokens)+len(candidateTokens) > p.MaxSeqLen {
			return nil, fmt.Errorf("System One choice %q exceeds context length", choice.Value)
		}
		logProb, err := p.scoreContinuation(baseKV, baseSSM, baseLogits, len(promptTokens), candidateTokens)
		if err != nil {
			return nil, fmt.Errorf("score choice %q: %w", choice.Value, err)
		}
		totalScoredTokens += len(candidateTokens)
		scores[i] = ChoiceScore{Value: choice.Value, Label: choice.Label, Code: choice.Code, LogProb: logProb}
	}
	scoreMs := float64(time.Since(scoreStart).Microseconds()) / 1000.0

	normalizeChoiceScores(scores)
	best := 0
	for i := 1; i < len(scores); i++ {
		if scores[i].Probability > scores[best].Probability {
			best = i
		}
	}
	result := &SystemOneResult{
		QuestionID:    q.ID,
		Value:         scores[best].Value,
		Confidence:    scores[best].Probability,
		Choices:       scores,
		PromptTokens:  len(promptTokens),
		ScoredTokens:  totalScoredTokens,
		PrefillTimeMs: prefillMs,
		ScoreTimeMs:   scoreMs,
	}
	if cfg.MinConfidence > 0 && result.Confidence < cfg.MinConfidence {
		result.Abstained = true
		result.Value = ""
	}
	return result, nil
}

// systemOnePromptParts holds the two tokenization-relevant halves of a System
// One prompt. The prefix (system + state) is shared across all questions in a
// multi-question request; only the suffix (question + allowed answers) varies.
type systemOnePromptParts struct {
	Prefix  string
	Suffix  string
	Choices []Choice
}

func compileSystemOneParts(state string, q SystemOneQuestion, cfg SystemOneConfig) (systemOnePromptParts, error) {
	seenValues := make(map[string]bool, len(q.Choices))
	seenCodes := make(map[string]bool, len(q.Choices))
	choices := append([]Choice(nil), q.Choices...)

	// Automatically assigned answer codes must not depend on caller choice
	// ordering. Stable codes prevent a harmless permutation of choices from
	// changing the model's prior over A/B/C. Explicit caller codes are retained.
	autoCodeByValue := make(map[string]string, len(choices))
	var autoValues []string
	for _, choice := range choices {
		if choice.Code == "" {
			autoValues = append(autoValues, choice.Value)
		}
	}
	sort.Strings(autoValues)
	for i, value := range autoValues {
		autoCodeByValue[value] = fmt.Sprintf(" %c", 'A'+rune(i))
	}

	for i := range choices {
		if choices[i].Value == "" {
			return systemOnePromptParts{}, fmt.Errorf("System One choice %d has empty value", i)
		}
		if seenValues[choices[i].Value] {
			return systemOnePromptParts{}, fmt.Errorf("duplicate System One choice value %q", choices[i].Value)
		}
		seenValues[choices[i].Value] = true
		if choices[i].Label == "" {
			choices[i].Label = choices[i].Value
		}
		if choices[i].Code == "" {
			// Prefix the code because GPT-2/Qwen BPE tokenizers encode the
			// natural continuation after "Answer code:" as " A", " B", ... .
			choices[i].Code = autoCodeByValue[choices[i].Value]
		}
		if seenCodes[choices[i].Code] {
			return systemOnePromptParts{}, fmt.Errorf("duplicate System One choice code %q", choices[i].Code)
		}
		seenCodes[choices[i].Code] = true
	}

	prefix := ""
	if cfg.System != "" {
		prefix += cfg.System + "\n\n"
	}
	prefix += "You are a precise decision engine. Return exactly one answer code and no explanation.\n\n"
	if state != "" {
		prefix += "State:\n" + state + "\n\n"
	}
	suffix := "Question:\n" + q.Prompt + "\n\nAllowed answers:\n"
	for _, choice := range choices {
		suffix += choice.Code + ": " + choice.Label + "\n"
	}
	suffix += "\nAnswer code:"
	return systemOnePromptParts{Prefix: prefix, Suffix: suffix, Choices: choices}, nil
}

// CompileSystemOnePrompt builds the raw completion prompt (no chat template)
// for a System One question: system instructions, state, question, allowed
// answer codes, and the "Answer code:" trigger. It also resolves and validates
// answer codes (stable value-sorted codes for auto-assigned slots). Exported so
// non-CPU engines (e.g. the Vulkan GPU pipeline) compile byte-identical
// prompts.
func CompileSystemOnePrompt(modelCfg ModelConfig, state string, q SystemOneQuestion, cfg SystemOneConfig) (string, []Choice, error) {
	parts, err := compileSystemOneParts(state, q, cfg)
	if err != nil {
		return "", nil, err
	}
	// Qwen3/3.5 chat formatting normally appends a generation primer. System
	// One decisions deliberately use a raw completion prompt so no reasoning
	// phase is requested and the scored answer remains a short code
	// continuation, so the chat template is intentionally not applied.
	_ = modelCfg
	return parts.Prefix + parts.Suffix, parts.Choices, nil
}

// CompileSystemOneParts splits a System One prompt into its shared prefix
// (system instructions + state) and per-question suffix (question + allowed
// answers + trigger), with resolved answer codes. Non-CPU engines use it to
// prefill one shared prefix and continue each question with only its suffix.
func CompileSystemOneParts(state string, q SystemOneQuestion, cfg SystemOneConfig) (prefix, suffix string, choices []Choice, err error) {
	parts, err := compileSystemOneParts(state, q, cfg)
	if err != nil {
		return "", "", nil, err
	}
	return parts.Prefix, parts.Suffix, parts.Choices, nil
}

// SystemOneMulti evaluates several independent questions against the same
// state. The shared prompt prefix (system + state) is tokenized and prefilled
// exactly once; each question then runs only its short suffix (question +
// allowed answers) from a cloned branch of that state, so N questions cost
// roughly one full prefill plus N small suffix passes instead of N full
// prefills. Questions whose suffix does not concatenate at the token boundary
// (a BPE merge would span the prefix/suffix seam) automatically fall back to a
// full per-question prefill, so results are always identical to evaluating the
// question alone. Results are returned in caller question order.
func (p *Pipeline) SystemOneMulti(state string, questions []SystemOneQuestion, cfg SystemOneConfig, readout *SystemOneReadout) ([]*SystemOneResult, error) {
	if err := cfg.Err(); err != nil {
		return nil, err
	}
	if p == nil || p.Model == nil || p.Tokenizer == nil || p.KVCache == nil || p.RunState == nil {
		return nil, fmt.Errorf("System One pipeline is not initialized")
	}
	if len(questions) == 0 {
		return nil, fmt.Errorf("SystemOneMulti requires at least one question")
	}

	// Compile every question up front so validation errors surface before any
	// expensive work, and resolve per-question suffix tokenizations.
	type compiledQuestion struct {
		suffixTokens []int32
		choices      []Choice
		questionID   string
	}
	compiled := make([]compiledQuestion, len(questions))
	prefix := ""
	var prefixTokens []int32
	for i, q := range questions {
		if len(q.Choices) < 2 {
			return nil, fmt.Errorf("question %d (%q): System One requires at least two choices", i, q.ID)
		}
		if q.Prompt == "" {
			return nil, fmt.Errorf("question %d (%q): prompt is empty", i, q.ID)
		}
		parts, err := compileSystemOneParts(state, q, cfg)
		if err != nil {
			return nil, fmt.Errorf("question %d (%q): %w", i, q.ID, err)
		}
		if i == 0 {
			prefix = parts.Prefix
			prefixTokens = p.Tokenizer.Encode(prefix)
			if len(prefixTokens) == 0 {
				return nil, fmt.Errorf("System One tokenizer produced no prefix tokens")
			}
			if len(prefixTokens) >= p.MaxSeqLen {
				return nil, fmt.Errorf("System One prefix too long: %d tokens (max %d)", len(prefixTokens), p.MaxSeqLen)
			}
		} else if parts.Prefix != prefix {
			// All questions must share the same system+state prefix.
			return nil, fmt.Errorf("question %d (%q): inconsistent prompt prefix", i, q.ID)
		}
		suffixTokens := p.Tokenizer.Encode(parts.Suffix)
		if len(suffixTokens) == 0 {
			return nil, fmt.Errorf("question %d (%q): tokenizer produced no suffix tokens", i, q.ID)
		}
		if len(prefixTokens)+len(suffixTokens) >= p.MaxSeqLen {
			return nil, fmt.Errorf("question %d (%q): prompt too long: %d tokens (max %d)", i, q.ID, len(prefixTokens)+len(suffixTokens), p.MaxSeqLen)
		}
		compiled[i] = compiledQuestion{suffixTokens: suffixTokens, choices: parts.Choices, questionID: q.ID}
	}

	// Token-boundary safety: tokenizing prefix and suffix separately must
	// produce exactly the token stream of the concatenated prompt. A BPE merge
	// spanning the seam would make the shared KV state invalid for the suffix.
	for i := range compiled {
		full := p.Tokenizer.Encode(prefix + compileSuffixForCheck(state, questions[i], cfg))
		joined := append(append([]int32(nil), prefixTokens...), compiled[i].suffixTokens...)
		if len(full) != len(joined) {
			compiled[i].suffixTokens = nil // mark token-unsafe
		} else {
			for t := range full {
				if full[t] != joined[t] {
					compiled[i].suffixTokens = nil
					break
				}
			}
		}
	}

	// Prefill the shared prefix once.
	p.KVCache.Reset()
	if p.RunState.SSMState != nil {
		p.RunState.SSMState.Reset()
	}
	prefillStart := time.Now()
	ForwardBatch(p.Model, prefixTokens, 0, p.KVCache, p.RunState, p.BatchState)
	sharedPrefillMs := float64(time.Since(prefillStart).Microseconds()) / 1000.0

	baseKV := p.KVCache.Clone(len(prefixTokens))
	baseSSM := p.RunState.SSMState.Clone()

	results := make([]*SystemOneResult, len(questions))
	for i, cq := range compiled {
		if err := cfg.Err(); err != nil {
			return nil, err
		}
		// Per-question graceful degradation: the trained readout covers a
		// fixed number of answer slots; questions wider than that fall back
		// to generic continuation scoring instead of failing the request.
		effectiveReadout := readout
		degraded := false
		if readout != nil && len(cq.choices) > readout.MaxOptions {
			effectiveReadout = nil
			degraded = true
		}
		if cq.suffixTokens == nil {
			// Token-boundary mismatch: evaluate this question standalone.
			var result *SystemOneResult
			var err error
			if effectiveReadout != nil {
				result, err = p.SystemOneWithReadout(state, questions[i], cfg, effectiveReadout)
			} else {
				result, err = p.SystemOne(state, questions[i], cfg)
			}
			if err != nil {
				return nil, fmt.Errorf("question %d (%q): %w", i, cq.questionID, err)
			}
			result.SharedPrefillTimeMs = sharedPrefillMs
			result.Degraded = degraded
			results[i] = result
			continue
		}

		result, err := p.systemOneSuffixQuestion(prefixTokens, cq.suffixTokens, cq.choices, questions[i], cfg, effectiveReadout, baseKV, baseSSM, sharedPrefillMs)
		if err != nil {
			return nil, fmt.Errorf("question %d (%q): %w", i, cq.questionID, err)
		}
		result.Degraded = degraded
		results[i] = result
	}
	return results, nil
}

// compileSuffixForCheck recompiles only the suffix of a question for the
// token-boundary verification pass.
func compileSuffixForCheck(state string, q SystemOneQuestion, cfg SystemOneConfig) string {
	parts, err := compileSystemOneParts(state, q, cfg)
	if err != nil {
		return ""
	}
	return parts.Suffix
}

// systemOneSuffixQuestion evaluates one question by continuing a cloned shared
// prefix state with the question suffix, then applying either the trained
// readout (one pass, no vocabulary projection) or continuation scoring.
func (p *Pipeline) systemOneSuffixQuestion(prefixTokens, suffixTokens []int32, choices []Choice, q SystemOneQuestion, cfg SystemOneConfig, readout *SystemOneReadout, baseKV *memory.MultiLayerKVCache, baseSSM *memory.SSMStateCache, sharedPrefillMs float64) (*SystemOneResult, error) {
	startPos := len(prefixTokens)
	kv := baseKV.Clone(startPos)
	rs := NewRunState(p.Model.Config, p.MaxSeqLen)
	rs.SSMState = baseSSM.Clone()

	suffixStart := time.Now()
	if readout != nil {
		// All but the last suffix token through the batch path to populate
		// KV/SSM state; the final token skips the vocabulary projection and
		// yields the hidden state for the readout.
		if len(suffixTokens) > 1 {
			ForwardBatch(p.Model, suffixTokens[:len(suffixTokens)-1], startPos, kv, rs, p.BatchState)
		}
		hidden := ForwardHidden(p.Model, suffixTokens[len(suffixTokens)-1], startPos+len(suffixTokens)-1, kv, rs)
		suffixMs := float64(time.Since(suffixStart).Microseconds()) / 1000.0
		return p.buildReadoutResult(q, choices, hidden, readout, cfg, len(prefixTokens)+len(suffixTokens), suffixMs, sharedPrefillMs)
	}

	// Fallback path: forward the whole suffix, then score answer-code
	// continuations from the resulting boundary state.
	ForwardBatch(p.Model, suffixTokens, startPos, kv, rs, p.BatchState)
	boundaryPos := startPos + len(suffixTokens)
	baseLogits := append([]float32(nil), rs.Logits...)
	scoreStart := time.Now()
	scores := make([]ChoiceScore, len(choices))
	totalScoredTokens := 0
	for i, choice := range choices {
		candidateTokens := p.Tokenizer.Encode(choice.Code)
		if len(candidateTokens) == 0 {
			return nil, fmt.Errorf("System One choice %q has no tokens", choice.Value)
		}
		if boundaryPos+len(candidateTokens) > p.MaxSeqLen {
			return nil, fmt.Errorf("System One choice %q exceeds context length", choice.Value)
		}
		logProb, err := p.scoreContinuation(kv, rs.SSMState, baseLogits, boundaryPos, candidateTokens)
		if err != nil {
			return nil, fmt.Errorf("score choice %q: %w", choice.Value, err)
		}
		totalScoredTokens += len(candidateTokens)
		scores[i] = ChoiceScore{Value: choice.Value, Label: choice.Label, Code: choice.Code, LogProb: logProb}
	}
	suffixMs := float64(time.Since(suffixStart).Microseconds()) / 1000.0
	scoreMs := float64(time.Since(scoreStart).Microseconds()) / 1000.0
	return buildScoredResult(q, scores, cfg, len(prefixTokens)+len(suffixTokens), totalScoredTokens, suffixMs, scoreMs, sharedPrefillMs), nil
}

// buildReadoutResult applies the trained readout to a hidden state and assembles
// the typed result with abstention handling.
func (p *Pipeline) buildReadoutResult(q SystemOneQuestion, choices []Choice, hidden []float32, readout *SystemOneReadout, cfg SystemOneConfig, promptTokens int, prefillMs, sharedPrefillMs float64) (*SystemOneResult, error) {
	normalizeSystemOneFeature(hidden)
	if len(choices) > readout.MaxOptions {
		return nil, fmt.Errorf("question has %d choices but readout supports %d", len(choices), readout.MaxOptions)
	}
	scoreStart := time.Now()
	scores := make([]ChoiceScore, len(choices))
	for i, choice := range choices {
		class, err := systemOneCodeIndex(choice.Code)
		if err != nil || class >= readout.MaxOptions {
			return nil, fmt.Errorf("choice %q: %w", choice.Value, err)
		}
		var value float64
		for j, feature := range hidden {
			value += float64(readout.Weights[class][j]) * float64(feature)
		}
		scores[i] = ChoiceScore{Value: choice.Value, Label: choice.Label, Code: choice.Code, LogProb: value / readout.Temperature}
	}
	scoreMs := float64(time.Since(scoreStart).Microseconds()) / 1000.0
	result := buildScoredResult(q, scores, cfg, promptTokens, 0, prefillMs, scoreMs, sharedPrefillMs)
	return result, nil
}

// buildScoredResult normalizes choice scores, selects the best, and applies
// minimum-confidence abstention.
func buildScoredResult(q SystemOneQuestion, scores []ChoiceScore, cfg SystemOneConfig, promptTokens, scoredTokens int, prefillMs, scoreMs, sharedPrefillMs float64) *SystemOneResult {
	normalizeChoiceScores(scores)
	best := 0
	for i := 1; i < len(scores); i++ {
		if scores[i].Probability > scores[best].Probability {
			best = i
		}
	}
	result := &SystemOneResult{
		QuestionID:          q.ID,
		Value:               scores[best].Value,
		Confidence:          scores[best].Probability,
		Choices:             scores,
		PromptTokens:        promptTokens,
		ScoredTokens:        scoredTokens,
		PrefillTimeMs:       prefillMs,
		ScoreTimeMs:         scoreMs,
		SharedPrefillTimeMs: sharedPrefillMs,
	}
	if cfg.MinConfidence > 0 && result.Confidence < cfg.MinConfidence {
		result.Abstained = true
		result.Value = ""
	}
	return result
}

// SystemOneNoul is the Jev-style yes/no primitive. It is a thin wrapper over
// SystemOne with exactly two choices ("yes"/"no"). The returned Value is
// "yes" or "no"; Confidence is the chosen side's normalized probability.
// When readout is non-nil the trained one-pass decision head is used instead
// of continuation scoring.
func (p *Pipeline) SystemOneNoul(state string, prompt string, cfg SystemOneConfig, readout *SystemOneReadout) (*SystemOneResult, error) {
	q := BuildSystemOneNoulQuestion(prompt)
	if readout != nil {
		return p.SystemOneWithReadout(state, q, cfg, readout)
	}
	return p.SystemOne(state, q, cfg)
}

// SystemOneScore is the Jev-style numeric-score primitive. Because the native
// decision engine is finite-choice, the requested [min,max] range is
// discretized into integer buckets and each bucket is scored as a choice. The
// returned Value is the chosen integer formatted as a string; callers should
// parse it with strconv.Atoi. A range wider than 64 buckets is rejected to keep
// the prompt and scoring tractable. When readout is non-nil the trained
// one-pass decision head is used instead of continuation scoring.
func (p *Pipeline) SystemOneScore(state string, prompt string, min, max int, cfg SystemOneConfig, readout *SystemOneReadout) (*SystemOneResult, error) {
	choices, err := BuildSystemOneScoreChoices(min, max)
	if err != nil {
		return nil, err
	}
	q := SystemOneQuestion{ID: "score", Prompt: prompt, Choices: choices}
	if readout != nil {
		return p.SystemOneWithReadout(state, q, cfg, readout)
	}
	return p.SystemOne(state, q, cfg)
}

// BuildSystemOneNoulQuestion returns the canonical yes/no question used by
// SystemOneNoul so callers can evaluate the same decision through any engine
// (CPU or GPU) with identical prompts.
func BuildSystemOneNoulQuestion(prompt string) SystemOneQuestion {
	return SystemOneQuestion{
		ID:     "noul",
		Prompt: prompt,
		Choices: []Choice{
			{Value: "yes", Label: "yes"},
			{Value: "no", Label: "no"},
		},
	}
}

// BuildSystemOneScoreChoices discretizes the inclusive integer range [min,max]
// into one choice per bucket, validating the range the same way SystemOneScore
// does, so callers can construct score questions for any engine.
func BuildSystemOneScoreChoices(min, max int) ([]Choice, error) {
	if max < min {
		return nil, fmt.Errorf("SystemOneScore: max %d < min %d", max, min)
	}
	if max-min+1 > 64 {
		return nil, fmt.Errorf("SystemOneScore: range %d..%d has more than 64 buckets", min, max)
	}
	choices := make([]Choice, 0, max-min+1)
	for v := min; v <= max; v++ {
		choices = append(choices, Choice{Value: fmt.Sprintf("%d", v), Label: fmt.Sprintf("%d", v)})
	}
	return choices, nil
}

func (p *Pipeline) scoreContinuation(baseKV *memory.MultiLayerKVCache, baseSSM *memory.SSMStateCache, baseLogits []float32, pos int, tokens []int32) (float64, error) {
	kv := baseKV.Clone(pos)
	rs := NewRunState(p.Model.Config, p.MaxSeqLen)
	rs.SSMState = baseSSM.Clone()
	copy(rs.Logits, baseLogits)

	var score float64
	logits := rs.Logits
	for i, token := range tokens {
		if int(token) < 0 || int(token) >= len(logits) {
			return 0, fmt.Errorf("token id %d is outside logits vocabulary", token)
		}
		score += tokenLogProbability(logits, int(token))
		if i+1 < len(tokens) {
			Forward(p.Model, token, pos+i, kv, rs)
			logits = rs.Logits
		}
	}
	return score, nil
}

func tokenLogProbability(logits []float32, token int) float64 {
	maxLogit := float32(math.Inf(-1))
	for _, value := range logits {
		if value > maxLogit {
			maxLogit = value
		}
	}
	if math.IsInf(float64(maxLogit), -1) || math.IsInf(float64(logits[token]), -1) {
		return math.Inf(-1)
	}
	var sum float64
	for _, value := range logits {
		if !math.IsInf(float64(value), -1) {
			sum += math.Exp(float64(value - maxLogit))
		}
	}
	return float64(logits[token]-maxLogit) - math.Log(sum)
}

func normalizeChoiceScores(scores []ChoiceScore) {
	maxScore := math.Inf(-1)
	for _, score := range scores {
		if score.LogProb > maxScore {
			maxScore = score.LogProb
		}
	}
	if math.IsInf(maxScore, -1) {
		return
	}
	var sum float64
	for _, score := range scores {
		sum += math.Exp(score.LogProb - maxScore)
	}
	for i := range scores {
		scores[i].Probability = math.Exp(scores[i].LogProb-maxScore) / sum
	}
}

// SortedChoices returns a probability-descending copy of choice scores. The
// original model order remains in SystemOneResult.Choices for stable mapping.
func (r *SystemOneResult) SortedChoices() []ChoiceScore {
	choices := append([]ChoiceScore(nil), r.Choices...)
	sort.SliceStable(choices, func(i, j int) bool { return choices[i].Probability > choices[j].Probability })
	return choices
}
