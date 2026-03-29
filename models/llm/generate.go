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
	"github.com/computerex/dlgo/grammar"
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
	Grammar   *grammar.Grammar   // optional grammar constraint (nil = unconstrained)
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

// FreeForGPU releases CPU-side KV cache, RunState, and BatchState that are
// unused when a GPU pipeline handles all inference. Only Model, Tokenizer,
// and MaxSeqLen are retained (needed by the scheduler for tokenization and
// config lookups). Call this after the GPU pipeline is successfully created.
func (p *Pipeline) FreeForGPU() {
	p.KVCache = nil
	p.RunState = nil
	p.BatchState = nil
}

// RebuildBuffers re-creates KV cache, RunState, and BatchState using the
// current MaxSeqLen. Used to restore CPU-side buffers after FreeForGPU when
// GPU pipeline creation fails and we fall back to CPU inference.
func (p *Pipeline) RebuildBuffers() {
	cfg := p.Model.Config
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	p.KVCache = newSparseKVCache(cfg, p.MaxSeqLen, kvDim)
	p.RunState = NewRunState(cfg, p.MaxSeqLen)
	batchCap := p.MaxSeqLen
	if batchCap > prefillChunkSize {
		batchCap = prefillChunkSize
	}
	p.BatchState = NewBatchState(cfg, batchCap, p.MaxSeqLen)
}

const (
	minContextLen = 64 // smallest context we'll auto-shrink to
)

// prefillChunkSize is the maximum batch size for prefill processing.
// BatchState is allocated at this size (not full context) to save RAM.
// ForwardBatch chunks large prompts automatically.
const prefillChunkSize = 8192

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

	// For hybrid SSM+attention models, only attention layers need KV cache.
	// SSM layers maintain their own recurrent state (SSMState) outside KV cache.
	attnLayers := nLayers
	isHybrid := cfg.FullAttentionInterval > 0 && cfg.SSMInnerSize > 0
	if isHybrid {
		// Every FullAttentionInterval-th layer is a full-attention layer.
		attnLayers = (nLayers + int64(cfg.FullAttentionInterval) - 1) / int64(cfg.FullAttentionInterval)
	}

	// KV cache: only attention layers need it (float32 K and V per position)
	kvBytes := 2 * attnLayers * seq * kvDim * 4

	// RunState: fixed per-token buffers (independent of sequence length except
	// for HeadScores, Scores, RoPE tables which are small).
	rsBytes := (3*dim + 2*qDim + 2*kvDim + 3*ffnDim + vocabSize) * 4
	rsBytes += int64(cfg.NumHeads) * seq * 4 // HeadScores[numHeads][maxSeqLen]
	rsBytes += seq * 4                         // Scores[maxSeqLen]
	ropeDim := int64(cfg.RopeDim)
	if ropeDim <= 0 || ropeDim > int64(cfg.HeadDim) {
		ropeDim = int64(cfg.HeadDim)
	}
	rsBytes += 2 * seq * (ropeDim / 2) * 4 // RoPE cos/sin tables

	if nUsed > 0 && expDim > 0 {
		rsBytes += nUsed * expDim * 4 * 4 // gates, ups, hiddens, outs per expert
		rsBytes += nUsed * dim * 4         // expert output buffers
	}

	// SSM state (recurrent state maintained in RunState, not BatchState)
	if cfg.SSMInnerSize > 0 {
		ssmHeads := int64(cfg.SSMTimeStepRank)
		ssmHK := int64(cfg.SSMStateSize)
		ssmHV := int64(cfg.SSMInnerSize) / max64(ssmHeads, 1)
		statePerLayer := ssmHeads * ssmHK * ssmHV * 4
		convPerLayer := int64(4) * (ssmHK*2 + ssmHeads*ssmHV) * 4
		rsBytes += nLayers * (statePerLayer + convPerLayer)
	}

	// BatchState: batch processing buffers are capped at prefillChunkSize.
	// Large prompts are chunked, so we don't need to allocate at full context.
	batchCap := seq
	if batchCap > prefillChunkSize {
		batchCap = prefillChunkSize
	}

	// Standard attention+FFN batch buffers (capped at batch size)
	bsBytes := batchCap * (3*dim + 2*qDim + 2*kvDim + 3*ffnDim + 4*dim) * 4

	// GatedQ batch buffers (for hybrid models with FullAttentionInterval > 0)
	if cfg.FullAttentionInterval > 0 {
		bsBytes += batchCap * (2*qDim + qDim) * 4 // qFullBatch + qGateBatch
	}

	// SSM batch buffers (scale with batch size, not context)
	if cfg.SSMInnerSize > 0 {
		ssmHeads := int64(cfg.SSMTimeStepRank)
		ssmHK := int64(cfg.SSMStateSize)
		ssmHV := int64(cfg.SSMInnerSize) / max64(ssmHeads, 1)
		numKVGroups := int64(cfg.SSMGroupCount)
		if numKVGroups <= 0 {
			numKVGroups = ssmHeads
		}
		ssmKeyDim := numKVGroups * ssmHK
		ssmValueDim := ssmHeads * ssmHV
		ssmQKVDim := ssmKeyDim*2 + ssmValueDim
		bsBytes += batchCap * (ssmQKVDim + ssmValueDim*2 + ssmHeads*2) * 4
	}

	// Score buffers: numWorkers * seqLen * 4, scaled to full context length
	// (each attention head needs scores for all positions). Use est. 8 workers.
	const estWorkers = 8
	bsBytes += estWorkers * seq * 4

	// KGather/VGather for SIMD batched attention: numKVHeads * seqLen * headDim * 4 each.
	// Only allocated when affordable (≤ 512 MB each direction = 1 GB total).
	const simdGatherLimit = 512 * 1024 * 1024
	kgatherBytes := int64(cfg.NumKVHeads) * seq * int64(cfg.HeadDim) * 4
	if kgatherBytes <= simdGatherLimit {
		bsBytes += kgatherBytes * 2 // K and V gather
	}

	return kvBytes + rsBytes + bsBytes
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// newSparseKVCache creates a KV cache that only allocates full-size entries
// for attention layers. SSM layers get a minimal 1-slot cache (they use their
// own recurrent SSMState and never access the KV cache during inference).
// This reduces RAM usage significantly for hybrid SSM+attention models.
func newSparseKVCache(cfg ModelConfig, maxSeqLen, kvDim int) *memory.MultiLayerKVCache {
	kv := &memory.MultiLayerKVCache{
		Layers: make([]*memory.KVCache, cfg.NumLayers),
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if ssmLayerIndex(l, cfg) {
			// SSM layer: allocate a minimal 1-slot cache (never accessed during
			// forward pass, but keeps slice indexing safe).
			kv.Layers[l] = memory.NewKVCache(1, 1)
		} else {
			kv.Layers[l] = memory.NewKVCache(maxSeqLen, kvDim)
		}
	}
	return kv
}

// ssmLayerIndex returns true if layer l is an SSM layer (not a full-attention layer).
func ssmLayerIndex(l int, cfg ModelConfig) bool {
	if cfg.FullAttentionInterval <= 0 || cfg.SSMInnerSize == 0 {
		return false
	}
	return ((l + 1) % cfg.FullAttentionInterval) != 0
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

	// Build sparse KV cache: only attention layers need full-context KV storage.
	// SSM layers maintain their own recurrent state (SSMState) — allocating a
	// full-size KV cache for them wastes significant RAM (4x for Qwen3.5 0.8B).
	kv := newSparseKVCache(m.Config, maxSeqLen, kvDim)
	rs := NewRunState(m.Config, maxSeqLen)

	// BatchState is capped at prefillChunkSize to keep RAM usage predictable.
	// ForwardBatch chunks large prompts automatically.
	batchCap := maxSeqLen
	if batchCap > prefillChunkSize {
		batchCap = prefillChunkSize
	}

	return &Pipeline{
		Model:      m,
		Tokenizer:  tok,
		KVCache:    kv,
		RunState:   rs,
		BatchState: NewBatchState(m.Config, batchCap, maxSeqLen),
		MaxSeqLen:  maxSeqLen,
	}, nil
}

// buildTokenPieces builds the token-id-to-string mapping needed for grammar masking.
func (p *Pipeline) buildTokenPieces() []string {
	vocabSize := p.Tokenizer.VocabSize()
	pieces := make([]string, vocabSize)
	for i := 0; i < vocabSize; i++ {
		pieces[i] = p.Tokenizer.DecodeToken(int32(i))
	}
	return pieces
}

// buildEOSSet returns a set of EOS/stop token IDs for grammar masking.
func (p *Pipeline) buildEOSSet() map[int32]bool {
	eos := map[int32]bool{p.Model.Config.EOS: true}
	for _, stop := range p.Model.Config.StopTokens {
		eos[stop] = true
	}
	return eos
}

// grammarSample applies grammar constraints (if any) then samples a token.
// Uses the speculative approach: sample first, check grammar, resample if needed.
func grammarSample(logits []float32, cfg GenerateConfig, recentTokens []int32, rng *rand.Rand,
	gram *grammar.Grammar, tokenPieces []string, eosTokens map[int32]bool) int {

	if gram == nil {
		return ops.SampleToken(logits, cfg.Sampler, recentTokens, rng)
	}

	// Apply grammar constraint before sampling
	gram.ApplyToLogits(logits, tokenPieces, eosTokens)
	return ops.SampleToken(logits, cfg.Sampler, recentTokens, rng)
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

	// Build token pieces and EOS set for grammar masking
	var tokenPieces []string
	var eosTokens map[int32]bool
	if cfg.Grammar != nil {
		tokenPieces = p.buildTokenPieces()
		eosTokens = p.buildEOSSet()
	}

	var generated []int32
	var recentTokens []int32

	// Prefill (batch)
	ForwardBatch(p.Model, prompt, 0, p.KVCache, p.RunState, p.BatchState)

	pos := len(prompt)
	nextToken := grammarSample(p.RunState.Logits, cfg, recentTokens, rng, cfg.Grammar, tokenPieces, eosTokens)
	generated = append(generated, int32(nextToken))
	recentTokens = append(recentTokens, int32(nextToken))

	// Advance grammar state
	if cfg.Grammar != nil {
		cfg.Grammar.AcceptToken(p.Tokenizer.DecodeToken(int32(nextToken)))
	}

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

		nextToken = grammarSample(p.RunState.Logits, cfg, recentTokens, rng, cfg.Grammar, tokenPieces, eosTokens)
		generated = append(generated, int32(nextToken))

		// Advance grammar state
		if cfg.Grammar != nil && !eosTokens[int32(nextToken)] {
			cfg.Grammar.AcceptToken(p.Tokenizer.DecodeToken(int32(nextToken)))
		}

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
