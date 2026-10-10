package llm

import (
	"math"
	"strings"
	"testing"
)

func TestCompileSystemOnePromptUsesStableCodes(t *testing.T) {
	prompt, choices, err := CompileSystemOnePrompt(ModelConfig{}, "duplicate charge", SystemOneQuestion{
		ID:     "route",
		Prompt: "Route this request.",
		Choices: []Choice{
			{Value: "billing", Label: "Billing issue"},
			{Value: "technical", Label: "Technical issue"},
		},
	}, SystemOneConfig{System: "Classify precisely."})
	if err != nil {
		t.Fatal(err)
	}
	if choices[0].Code != " A" || choices[1].Code != " B" {
		t.Fatalf("codes = %#v, want stable space-prefixed A/B", choices)
	}

	_, reversed, err := CompileSystemOnePrompt(ModelConfig{}, "state", SystemOneQuestion{
		Prompt: "Route.", Choices: []Choice{{Value: "technical"}, {Value: "billing"}},
	}, SystemOneConfig{})
	if err != nil || reversed[0].Code != " B" || reversed[1].Code != " A" {
		t.Fatalf("automatic codes are not value-stable: %#v, err=%v", reversed, err)
	}
	for _, required := range []string{
		"Classify precisely.", "State:\nduplicate charge", "A: Billing issue", "B: Technical issue", "Answer code:",
	} {
		if !strings.Contains(prompt, required) {
			t.Errorf("prompt does not contain %q:\n%s", required, prompt)
		}
	}
}

func TestCompileSystemOnePromptRejectsDuplicateValuesAndCodes(t *testing.T) {
	_, _, err := CompileSystemOnePrompt(ModelConfig{}, "", SystemOneQuestion{
		Prompt:  "test",
		Choices: []Choice{{Value: "same"}, {Value: "same"}},
	}, SystemOneConfig{})
	if err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("error = %v, want duplicate value error", err)
	}

	_, _, err = CompileSystemOnePrompt(ModelConfig{}, "", SystemOneQuestion{
		Prompt:  "test",
		Choices: []Choice{{Value: "one", Code: "X"}, {Value: "two", Code: "X"}},
	}, SystemOneConfig{})
	if err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("error = %v, want duplicate code error", err)
	}
}

func TestNormalizeChoiceScores(t *testing.T) {
	scores := []ChoiceScore{{Value: "a", LogProb: -0.1}, {Value: "b", LogProb: -1.1}, {Value: "c", LogProb: -2.1}}
	normalizeChoiceScores(scores)
	var sum float64
	for _, score := range scores {
		sum += score.Probability
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("probability sum = %f, want 1", sum)
	}
	if !(scores[0].Probability > scores[1].Probability && scores[1].Probability > scores[2].Probability) {
		t.Fatalf("probabilities not rank-preserving: %#v", scores)
	}
}

func TestTokenLogProbability(t *testing.T) {
	logits := []float32{0, float32(math.Log(2)), float32(math.Log(4))}
	got := tokenLogProbability(logits, 2)
	want := math.Log(4.0 / 7.0)
	if math.Abs(got-want) > 1e-6 {
		t.Fatalf("log probability = %.8f, want %.8f", got, want)
	}
}

func TestSortedChoicesPreservesResultOrder(t *testing.T) {
	result := SystemOneResult{Choices: []ChoiceScore{{Value: "a", Probability: 0.2}, {Value: "b", Probability: 0.8}}}
	sorted := result.SortedChoices()
	if sorted[0].Value != "b" {
		t.Fatalf("sorted first = %q, want b", sorted[0].Value)
	}
	if result.Choices[0].Value != "a" {
		t.Fatal("SortedChoices mutated the original result")
	}
}

func TestSystemOneNoulBuildsYesNoQuestion(t *testing.T) {
	p := &Pipeline{}
	// SystemOneNoul requires a real pipeline to run, but we can verify the
	// question shape by checking the error path is a pipeline-init error, not a
	// choice-count error (which would indicate the wrapper built the wrong set).
	_, err := p.SystemOneNoul("state", "Is this urgent?", SystemOneConfig{}, nil)
	if err == nil {
		t.Fatal("expected error from uninitialized pipeline")
	}
	if !strings.Contains(err.Error(), "not initialized") {
		t.Fatalf("error = %v, want pipeline-not-initialized (wrapper built wrong choices?)", err)
	}
}

func TestSystemOneScoreValidatesRange(t *testing.T) {
	p := &Pipeline{}
	// A valid narrow range should reach the pipeline-init check, not a range
	// validation error.
	_, err := p.SystemOneScore("state", "Rate 0-10", 0, 10, SystemOneConfig{}, nil)
	if err == nil {
		t.Fatal("expected error from uninitialized pipeline")
	}
	if !strings.Contains(err.Error(), "not initialized") {
		t.Fatalf("error = %v, want pipeline-not-initialized", err)
	}

	// An inverted range must be rejected before touching the pipeline.
	_, err = p.SystemOneScore("state", "Rate", 10, 0, SystemOneConfig{}, nil)
	if err == nil || !strings.Contains(err.Error(), "max") {
		t.Fatalf("error = %v, want inverted-range error", err)
	}

	// A range wider than 64 buckets must be rejected.
	_, err = p.SystemOneScore("state", "Rate", 0, 100, SystemOneConfig{}, nil)
	if err == nil || !strings.Contains(err.Error(), "64") {
		t.Fatalf("error = %v, want too-wide-range error", err)
	}
}

func TestSystemOneMultiValidatesQuestions(t *testing.T) {
	p := &Pipeline{}
	// Empty question list must be rejected before touching the pipeline.
	if _, err := p.SystemOneMulti("state", nil, SystemOneConfig{}, nil); err == nil {
		t.Fatal("expected error for empty question list")
	}
	// A valid list should reach the pipeline-init check, not a question-count error.
	_, err := p.SystemOneMulti("state", []SystemOneQuestion{{ID: "a", Prompt: "p", Choices: []Choice{{Value: "x"}, {Value: "y"}}}}, SystemOneConfig{}, nil)
	if err == nil {
		t.Fatal("expected error from uninitialized pipeline")
	}
	if !strings.Contains(err.Error(), "not initialized") {
		t.Fatalf("error = %v, want pipeline-not-initialized", err)
	}
}
