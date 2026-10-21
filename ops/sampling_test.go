package ops

import (
	"math"
	"math/rand"
	"testing"
)

func TestSampleTokenGreedy(t *testing.T) {
	logits := []float32{1, 5, 2, 3}
	cfg := SamplerConfig{Temperature: 0}
	idx := SampleToken(logits, cfg, nil, nil)
	if idx != 1 {
		t.Errorf("greedy: got %d, want 1", idx)
	}
}

func TestSampleTokenWithRng(t *testing.T) {
	logits := []float32{100, -100, -100, -100}
	cfg := SamplerConfig{Temperature: 0.1, TopK: 0, TopP: 1.0}
	rng := rand.New(rand.NewSource(42))

	// With logits[0] overwhelmingly dominant, sampling should almost always pick 0
	for trial := 0; trial < 100; trial++ {
		idx := SampleToken(logits, cfg, nil, rng)
		if idx != 0 {
			t.Errorf("trial %d: got %d, want 0", trial, idx)
		}
	}
}

func TestApplyRepetitionPenalty(t *testing.T) {
	logits := []float32{2, -1, 3, 0}
	ApplyRepetitionPenalty(logits, []int32{0, 1}, 2.0)
	// token 0: positive, divide by 2 => 1
	// token 1: negative, multiply by 2 => -2
	if !approxEq(logits[0], 1, 1e-5) {
		t.Errorf("repPen logit[0] = %f, want 1", logits[0])
	}
	if !approxEq(logits[1], -2, 1e-5) {
		t.Errorf("repPen logit[1] = %f, want -2", logits[1])
	}
	if logits[2] != 3 || logits[3] != 0 {
		t.Errorf("repPen unchanged tokens modified: %v", logits)
	}
}

func TestApplyTemperature(t *testing.T) {
	logits := []float32{2, 4, 6}
	ApplyTemperature(logits, 2.0)
	want := []float32{1, 2, 3}
	for i := range logits {
		if !approxEq(logits[i], want[i], 1e-5) {
			t.Errorf("temp[%d] = %f, want %f", i, logits[i], want[i])
		}
	}
}

func TestApplyTopK(t *testing.T) {
	logits := []float32{1, 4, 2, 5, 3}
	ApplyTopK(logits, 2)
	// Top 2 are indices 3 (5) and 1 (4)
	negInf := float32(math.Inf(-1))
	if logits[3] != 5 || logits[1] != 4 {
		t.Errorf("top-k didn't preserve top values: %v", logits)
	}
	if logits[0] != negInf || logits[2] != negInf {
		t.Errorf("top-k didn't zero out low values: %v", logits)
	}
}

func TestApplyTopP(t *testing.T) {
	logits := []float32{10, 1, 1, 1}
	ApplyTopP(logits, 0.5)
	// After softmax, token 0 dominates (~0.999), so only token 0 should remain
	negInf := float32(math.Inf(-1))
	if logits[0] == negInf {
		t.Error("top-p removed dominant token")
	}
}

func TestApplyMinP(t *testing.T) {
	logits := []float32{10, 5, 1, -5}
	ApplyMinP(logits, 0.1)
	// Token 0 is dominant; tokens with prob < 0.1 * max_prob should be filtered
	if logits[0] == float32(math.Inf(-1)) {
		t.Error("min-p removed top token")
	}
}

func TestDefaultSamplerConfig(t *testing.T) {
	cfg := DefaultSamplerConfig()
	if cfg.Temperature != 0.7 {
		t.Errorf("default temp = %f, want 0.7", cfg.Temperature)
	}
	if cfg.TopK != 40 {
		t.Errorf("default topK = %d, want 40", cfg.TopK)
	}
	if cfg.TopP != 0.9 {
		t.Errorf("default topP = %f, want 0.9", cfg.TopP)
	}
	// Anti-repetition penalties must stay off by default: n-gram blocking
	// truncates text that legitimately repeats an earlier span (see the
	// tool-call argument case in DefaultSamplerConfig).
	if cfg.NoRepeatNgramSize != 0 {
		t.Errorf("default NoRepeatNgramSize = %d, want 0 (off)", cfg.NoRepeatNgramSize)
	}
	// A mild repetition penalty (logit scaling) is fine and stops runaway loops;
	// it never makes a token impossible, unlike n-gram blocking.
	if cfg.RepetitionPenalty != 1.1 {
		t.Errorf("default RepetitionPenalty = %f, want 1.1", cfg.RepetitionPenalty)
	}
}

// TestDefaultSamplerAllowsRepeatingEarlierNgram is a regression test for the
// bug where a tool-call argument that restated a value already produced in the
// reasoning block was truncated: with the shipped defaults, the token that
// completes an already-seen n-gram must NOT be banned.
func TestDefaultSamplerAllowsRepeatingEarlierNgram(t *testing.T) {
	cfg := DefaultSamplerConfig()
	cfg.Temperature = 0 // greedy, deterministic

	// Pretend the model already emitted "F(13) = 233" while thinking, and is
	// now about to restate it inside a tool-call argument.
	recent := []int32{9, 8, 7, 6, 5, 4, 3, 2, 1}
	continuation := int32(1234) // stands in for the repeated value token

	// Make the repeated value the greedy choice, but put a decoy close behind
	// so a banned token would still be replaced by something else.
	logits := make([]float32, 4096)
	for i := range logits {
		logits[i] = -10
	}
	logits[continuation] = 20
	logits[999] = 19

	for i := 0; i < 8; i++ {
		got := SampleToken(logits, cfg, recent, nil)
		if int32(got) != continuation {
			t.Fatalf("sample %d = %d, want %d (repeated n-gram must not be blocked)", i, got, continuation)
		}
		logits = make([]float32, 4096)
		for j := range logits {
			logits[j] = -10
		}
		logits[continuation] = 20
		logits[999] = 19
	}

	// And confirm the hazard is real when the knob is explicitly turned on:
	// the token completing the seen 8-gram becomes unavailable.
	on := cfg
	on.NoRepeatNgramSize = 8
	// Build a history whose last 7 tokens repeat early, followed by the value.
	hist := make([]int32, 0, 16)
	hist = append(hist, 1, 2, 3, 4, 5, 6, 7, 8)
	hist = append(hist, 1, 2, 3, 4, 5, 6, 7)
	blocked := make([]float32, 32)
	for i := range blocked {
		blocked[i] = -10
	}
	blocked[8] = 20
	blocked[9] = 19
	if got := SampleToken(blocked, on, hist, nil); got == 8 {
		t.Fatalf("with NoRepeatNgramSize=8 the repeated token 8 should have been blocked")
	}
}
