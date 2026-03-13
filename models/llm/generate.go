package llm

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"time"

	"github.com/computerex/dlgo/format/gguf"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/mmap"
	"github.com/computerex/dlgo/ops"
)


// GenerateConfig controls text generation behavior.
type GenerateConfig struct {
	MaxTokens int
	Sampler   ops.SamplerConfig
	Seed      int64
	Stream    func(token string) // called for each generated token (nil = no streaming)
}

// DefaultGenerateConfig returns sensible defaults.
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		MaxTokens: 256,
		Sampler:   ops.DefaultSamplerConfig(),
		Seed:      -1,
	}
}

// Pipeline bundles a loaded model, tokenizer, KV cache, and run state for inference.
type Pipeline struct {
	Model      *Model
	Tokenizer  *Tokenizer
	KVCache    *memory.MultiLayerKVCache
	RunState   *RunState
	BatchState *BatchState
	MaxSeqLen  int
}

const (
	minContextLen = 64 // smallest context we'll auto-shrink to
)

// EstimateRuntimeBytes estimates heap bytes needed for KV cache, RunState,
// and BatchState at a given context length, WITHOUT counting the model weights
// (which are memory-mapped and paged in by the OS on demand).
func EstimateRuntimeBytes(cfg ModelConfig, seqLen int) int64 {
	dim := int64(cfg.EmbeddingDim)
	qDim := int64(cfg.NumHeads) * int64(cfg.HeadDim)
	kvDim := int64(cfg.NumKVHeads) * int64(cfg.HeadDim)
	ffnDim := int64(cfg.FFNDim)
	nLayers := int64(cfg.NumLayers)
	seq := int64(seqLen)
	vocabSize := int64(cfg.VocabSize)
	nUsed := int64(cfg.ExpertUsedCount)
	expDim := int64(cfg.ExpertFFNDim)

	// KV cache: 2 * nLayers * seqLen * kvDim * 4 bytes (float32)
	kvBytes := 2 * nLayers * seq * kvDim * 4

	// RunState: ~10 buffers of dim, plus qDim, kvDim, ffnDim, vocabSize
	rsBytes := (3*dim + 2*qDim + 2*kvDim + 3*ffnDim + vocabSize) * 4
	rsBytes += int64(cfg.NumHeads) * seq * 4 // HeadScores

	if nUsed > 0 && expDim > 0 {
		rsBytes += nUsed * expDim * 4 * 4 // gates, ups, hiddens, outs per expert
		rsBytes += nUsed * dim * 4         // expert output buffers
	}

	// BatchState: ~14 buffers of seqLen*dim, plus seqLen*qDim, etc.
	bsBytes := seq * (3*dim + 2*qDim + 2*kvDim + 3*ffnDim + 4*dim) * 4

	// SSM state (if hybrid model)
	if cfg.SSMInnerSize > 0 {
		ssmHeads := int64(cfg.SSMTimeStepRank)
		ssmHK := int64(cfg.SSMStateSize)
		ssmHV := int64(cfg.SSMInnerSize) / max64(ssmHeads, 1)
		statePerLayer := ssmHeads * ssmHK * ssmHV * 4
		convPerLayer := int64(4) * (ssmHK*2 + ssmHeads*ssmHV) * 4
		rsBytes += nLayers * (statePerLayer + convPerLayer)
	}

	return kvBytes + rsBytes + bsBytes
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// CheckMemoryBudget checks whether loading the given model at the requested
// context length will fit in available RAM. Returns an adjusted (possibly
// reduced) maxSeqLen and an error only if even the minimum context won't fit.
//
// Model weights are memory-mapped and demand-paged by the OS — they do NOT
// consume heap RAM. Only runtime buffers (KV cache, RunState, BatchState)
// need actual RAM. The budget is: 85% of total physical RAM minus current
// usage. This ensures any model can load regardless of size; throughput
// degrades gracefully via mmap paging but the system never crashes.
func CheckMemoryBudget(modelPath string, cfg ModelConfig, requestedSeqLen int) (int, error) {
	sysInfo, err := mmap.GetSystemMemInfo()
	if err != nil {
		return requestedSeqLen, nil // can't query RAM, skip check
	}

	totalRAM := int64(sysInfo.TotalPhysical)
	availRAM := int64(sysInfo.AvailablePhysical)

	// Budget = available RAM right now. We don't use a ceiling-based formula
	// because other processes may legitimately consume RAM; we only care
	// whether OUR runtime buffers (KV cache, RunState, BatchState) can fit.
	// Model weights are mmap'd and demand-paged — they don't need heap RAM.
	// Reserve 2 GB for OS/other processes as a safety margin.
	const reserveBytes = 2 * (1 << 30) // 2 GB
	budget := availRAM - reserveBytes
	if budget < 0 {
		budget = 0
	}

	seqLen := requestedSeqLen
	if seqLen <= 0 || seqLen > cfg.ContextLength {
		seqLen = cfg.ContextLength
	}

	runtimeBytes := EstimateRuntimeBytes(cfg, seqLen)

	if runtimeBytes <= budget {
		return seqLen, nil
	}

	// Auto-shrink context to fit runtime buffers in available RAM.
	origSeqLen := seqLen
	for seqLen > minContextLen {
		seqLen = seqLen / 2
		if seqLen < minContextLen {
			seqLen = minContextLen
		}
		runtimeBytes = EstimateRuntimeBytes(cfg, seqLen)
		if runtimeBytes <= budget {
			fmt.Fprintf(os.Stderr, "[dlgo] memory budget: reducing context from %d to %d tokens "+
				"(%.1f GB available, runtime needs %.1f GB)\n",
				origSeqLen, seqLen,
				float64(budget)/(1<<30), float64(runtimeBytes)/(1<<30))
			return seqLen, nil
		}
		if seqLen == minContextLen {
			break
		}
	}

	return 0, fmt.Errorf(
		"insufficient memory: runtime buffers need ~%.1f GB even at minimum context (%d tokens) "+
			"but only %.1f GB available (%.1f GB total, %.1f GB free). "+
			"Close other applications to free RAM",
		float64(runtimeBytes)/(1<<30), minContextLen,
		float64(budget)/(1<<30),
		float64(totalRAM)/(1<<30),
		float64(availRAM)/(1<<30),
	)
}

// NewPipeline loads a GGUF model and creates a ready-to-use inference pipeline
// with automatic tokenizer extraction from GGUF metadata.
func NewPipeline(modelPath string, maxSeqLen int) (*Pipeline, error) {
	gf, err := gguf.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("parse GGUF: %w", err)
	}

	// Parse config first for the memory budget check (lightweight, no mmap).
	cfg, parseErr := parseConfig(gf.Metadata)
	if parseErr == nil {
		safeSeqLen, memErr := CheckMemoryBudget(modelPath, cfg, maxSeqLen)
		if memErr != nil {
			return nil, memErr
		}
		if safeSeqLen != maxSeqLen && maxSeqLen > 0 {
			fmt.Fprintf(os.Stderr, "[dlgo] memory budget: reducing context from %d to %d tokens to fit in RAM\n",
				maxSeqLen, safeSeqLen)
		}
		maxSeqLen = safeSeqLen
	}

	m, err := LoadModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}

	// After mmap-based model loading, trim the working set to release
	// pages the OS speculatively read-ahead. They'll fault back in on demand.
	mmap.TrimWorkingSet()

	if maxSeqLen <= 0 || maxSeqLen > m.Config.ContextLength {
		maxSeqLen = m.Config.ContextLength
	}

	// Second check after LoadModel resolved the actual config
	safeSeqLen, memErr := CheckMemoryBudget(modelPath, m.Config, maxSeqLen)
	if memErr != nil {
		return nil, memErr
	}
	maxSeqLen = safeSeqLen

	tok, err := NewTokenizerFromGGUF(gf.Metadata, m.Config)
	if err != nil {
		tok = &Tokenizer{
			BOS:    m.Config.BOS,
			EOS:    m.Config.EOS,
			AddBOS: m.Config.AddBOS,
			PreBOS: -1,
		}
	}
	m.Config.AddBOS = tok.AddBOS

	// For architectures with structural tokens that should never appear in output,
	// register them as stop tokens for fast token-level detection.
	for _, special := range []string{"<|channel|>", "<|start|>", "<|message|>", "<|constrain|>", "<|call|>"} {
		if id, ok := tok.TokenToID[special]; ok {
			m.Config.StopTokens = append(m.Config.StopTokens, id)
		}
	}

	kvDim := m.Config.NumKVHeads * m.Config.HeadDim
	kv := memory.NewMultiLayerKVCache(m.Config.NumLayers, maxSeqLen, kvDim)
	rs := NewRunState(m.Config, maxSeqLen)

	return &Pipeline{
		Model:      m,
		Tokenizer:  tok,
		KVCache:    kv,
		RunState:   rs,
		BatchState: NewBatchState(m.Config, maxSeqLen),
		MaxSeqLen:  maxSeqLen,
	}, nil
}

// Generate produces text from a prompt using the loaded model.
func (p *Pipeline) Generate(prompt []int32, cfg GenerateConfig) ([]int32, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("empty prompt")
	}
	if len(prompt) >= p.MaxSeqLen {
		return nil, fmt.Errorf("prompt too long: %d tokens (max %d)", len(prompt), p.MaxSeqLen)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	if cfg.Seed < 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	p.KVCache.Reset()
	if p.RunState.SSMState != nil {
		p.RunState.SSMState.Reset()
	}

	runtime.GC()
	prev := debug.SetGCPercent(2000)
	defer debug.SetGCPercent(prev)

	var generated []int32
	var recentTokens []int32

	// Prefill (batch)
	ForwardBatch(p.Model, prompt, 0, p.KVCache, p.RunState, p.BatchState)

	pos := len(prompt)
	nextToken := ops.SampleToken(p.RunState.Logits, cfg.Sampler, recentTokens, rng)
	generated = append(generated, int32(nextToken))
	recentTokens = append(recentTokens, int32(nextToken))

	if cfg.Stream != nil {
		cfg.Stream(p.Tokenizer.DecodeToken(int32(nextToken)))
	}

	for step := 1; step < cfg.MaxTokens; step++ {
		if pos >= p.MaxSeqLen-1 {
			break
		}

		lastTok := int32(nextToken)
		if lastTok == p.Model.Config.EOS {
			break
		}
		for _, stop := range p.Model.Config.StopTokens {
			if lastTok == stop {
				return generated, nil
			}
		}

		Forward(p.Model, lastTok, pos, p.KVCache, p.RunState)
		pos++

		// Periodically trim the working set to evict mmap pages that were
		// read during Forward. Prevents page cache from filling all RAM.
		if step%32 == 0 {
			mmap.TrimWorkingSet()
		}

		nextToken = ops.SampleToken(p.RunState.Logits, cfg.Sampler, recentTokens, rng)
		generated = append(generated, int32(nextToken))

		recentTokens = append(recentTokens, int32(nextToken))
		if len(recentTokens) > 256 {
			recentTokens = recentTokens[1:]
		}

		if cfg.Stream != nil {
			cfg.Stream(p.Tokenizer.DecodeToken(int32(nextToken)))
		}
	}

	return generated, nil
}

// GenerateText is a convenience method that takes a text prompt, encodes it,
// generates tokens, and decodes the result. Returns the generated text and
// token/second throughput.
func (p *Pipeline) GenerateText(prompt string, cfg GenerateConfig) (string, float64, error) {
	tokens := p.Tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return "", 0, fmt.Errorf("tokenizer produced no tokens for prompt")
	}

	start := time.Now()
	generated, err := p.Generate(tokens, cfg)
	elapsed := time.Since(start)

	if err != nil {
		return "", 0, err
	}

	text := trimStopText(p.Tokenizer.Decode(generated), p.Model.Config)
	tokPerSec := float64(len(generated)) / elapsed.Seconds()
	return text, tokPerSec, nil
}

// Chat formats a user message (with optional system prompt) using the model's
// chat template, then generates a response. Returns generated text and tok/s.
func (p *Pipeline) Chat(system, user string, cfg GenerateConfig) (string, float64, error) {
	prompt := FormatChat(p.Model.Config, system, user)
	return p.GenerateText(prompt, cfg)
}

// ChatMessages formats a multi-turn conversation and generates the assistant's
// next response. Returns generated text and tok/s.
func (p *Pipeline) ChatMessages(messages []Message, cfg GenerateConfig) (string, float64, error) {
	prompt := FormatMessages(p.Model.Config, messages)
	return p.GenerateText(prompt, cfg)
}

// GenerateResult holds detailed output from a generation run.
type GenerateResult struct {
	Text          string
	Tokens        []int32
	TokensPerSec  float64
	PrefillTimeMs float64
	GenerateTimeMs float64
	TotalTokens   int
	PromptTokens  int
}

// GenerateDetailed is like GenerateText but returns detailed timing information.
func (p *Pipeline) GenerateDetailed(prompt string, cfg GenerateConfig) (*GenerateResult, error) {
	tokens := p.Tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenizer produced no tokens for prompt")
	}
	if len(tokens) >= p.MaxSeqLen {
		return nil, fmt.Errorf("prompt too long: %d tokens (max %d)", len(tokens), p.MaxSeqLen)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	if cfg.Seed < 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	p.KVCache.Reset()
	if p.RunState.SSMState != nil {
		p.RunState.SSMState.Reset()
	}

	// Minimize GC interference during inference (2000% = rare but not disabled,
	// preventing unbounded heap growth from any allocating fallback paths)
	runtime.GC()
	prev := debug.SetGCPercent(2000)

	// Prefill (batch)
	prefillStart := time.Now()
	ForwardBatch(p.Model, tokens, 0, p.KVCache, p.RunState, p.BatchState)
	prefillMs := float64(time.Since(prefillStart).Microseconds()) / 1000.0

	// Generate
	genStart := time.Now()
	var generated []int32
	var recentTokens []int32

	pos := len(tokens)
	nextToken := ops.SampleToken(p.RunState.Logits, cfg.Sampler, recentTokens, rng)
	generated = append(generated, int32(nextToken))
	recentTokens = append(recentTokens, int32(nextToken))

	if cfg.Stream != nil {
		cfg.Stream(p.Tokenizer.DecodeToken(int32(nextToken)))
	}

	for step := 1; step < cfg.MaxTokens; step++ {
		if pos >= p.MaxSeqLen-1 {
			break
		}
		lastTok := int32(nextToken)
		if lastTok == p.Model.Config.EOS {
			break
		}
		for _, stop := range p.Model.Config.StopTokens {
			if lastTok == stop {
				goto done
			}
		}

		Forward(p.Model, lastTok, pos, p.KVCache, p.RunState)
		pos++

		if step%32 == 0 {
			mmap.TrimWorkingSet()
		}

		nextToken = ops.SampleToken(p.RunState.Logits, cfg.Sampler, recentTokens, rng)
		generated = append(generated, int32(nextToken))
		recentTokens = append(recentTokens, int32(nextToken))
		if len(recentTokens) > 256 {
			recentTokens = recentTokens[1:]
		}

		if cfg.Stream != nil {
			cfg.Stream(p.Tokenizer.DecodeToken(int32(nextToken)))
		}
	}

done:
	genMs := float64(time.Since(genStart).Microseconds()) / 1000.0
	debug.SetGCPercent(prev)
	text := trimStopText(p.Tokenizer.Decode(generated), p.Model.Config)

	var tokPerSec float64
	if genMs > 0 {
		tokPerSec = float64(len(generated)) / (genMs / 1000.0)
	}

	return &GenerateResult{
		Text:           text,
		Tokens:         generated,
		TokensPerSec:   tokPerSec,
		PrefillTimeMs:  prefillMs,
		GenerateTimeMs: genMs,
		TotalTokens:    len(generated),
		PromptTokens:   len(tokens),
	}, nil
}

// collectStopStrings returns text-level stop sequences for the model's arch.
func collectStopStrings(cfg ModelConfig) []string {
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

// TrimStopText removes trailing stop strings and whitespace from generated text.
func TrimStopText(text string, cfg ModelConfig) string {
	return trimStopText(text, cfg)
}

func trimStopText(text string, cfg ModelConfig) string {
	for {
		trimmed := strings.TrimRight(text, " \t\r\n")
		for _, ss := range collectStopStrings(cfg) {
			trimmed = strings.TrimSuffix(trimmed, ss)
			trimmed = strings.TrimRight(trimmed, " \t\r\n")
		}
		if trimmed == text {
			return trimmed
		}
		text = trimmed
	}
}

// GenerateTextWithStopStrings is like GenerateText but also handles text-level
// stop string detection for multi-token stop sequences.
func (p *Pipeline) GenerateTextWithStopStrings(prompt string, cfg GenerateConfig) (string, float64, error) {
	tokens := p.Tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return "", 0, fmt.Errorf("tokenizer produced no tokens")
	}
	if len(tokens) >= p.MaxSeqLen {
		return "", 0, fmt.Errorf("prompt too long: %d tokens (max %d)", len(tokens), p.MaxSeqLen)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	if cfg.Seed < 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	p.KVCache.Reset()
	if p.RunState.SSMState != nil {
		p.RunState.SSMState.Reset()
	}
	stopStrings := collectStopStrings(p.Model.Config)

	for i, tok := range tokens {
		Forward(p.Model, tok, i, p.KVCache, p.RunState)
	}

	start := time.Now()
	var generated []int32
	var recentTokens []int32
	var genText strings.Builder

	pos := len(tokens)
	for step := 0; step < cfg.MaxTokens; step++ {
		if pos >= p.MaxSeqLen-1 {
			break
		}

		nextToken := int32(ops.SampleToken(p.RunState.Logits, cfg.Sampler, recentTokens, rng))

		if nextToken == p.Model.Config.EOS {
			break
		}
		stopped := false
		for _, stop := range p.Model.Config.StopTokens {
			if nextToken == stop {
				stopped = true
				break
			}
		}
		if stopped {
			break
		}

		generated = append(generated, nextToken)
		recentTokens = append(recentTokens, nextToken)
		if len(recentTokens) > 256 {
			recentTokens = recentTokens[1:]
		}

		tokenText := p.Tokenizer.DecodeToken(nextToken)
		genText.WriteString(tokenText)

		if cfg.Stream != nil {
			cfg.Stream(tokenText)
		}

		// Text-level stop detection
		fullText := genText.String()
		for _, ss := range stopStrings {
			if strings.HasSuffix(fullText, ss) {
				trimmed := strings.TrimSuffix(fullText, ss)
				elapsed := time.Since(start)
				tokPerSec := float64(len(generated)) / elapsed.Seconds()
				return trimmed, tokPerSec, nil
			}
		}

		Forward(p.Model, nextToken, pos, p.KVCache, p.RunState)
		pos++
	}

	elapsed := time.Since(start)
	tokPerSec := float64(len(generated)) / elapsed.Seconds()
	return trimStopText(genText.String(), p.Model.Config), tokPerSec, nil
}
