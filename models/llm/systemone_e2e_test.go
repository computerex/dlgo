package llm

import (
	"math"
	"os"
	"testing"
)

const systemOneE2EModelPath = `C:\Users\mohd\Downloads\Qwen3.8-2B-Q4_K_M.gguf`

// TestSystemOneE2EWithQwen38 exercises the complete local decision path:
// GGUF load, tokenizer, shared prefill, hybrid Qwen3.5 SSM/KV branch cloning,
// continuation scoring, and probability normalization. It is intentionally
// opt-in because loading and running the development model is expensive.
func TestSystemOneE2EWithQwen38(t *testing.T) {
	if os.Getenv("DLGO_SYSTEMONE_E2E") != "1" {
		t.Skip("set DLGO_SYSTEMONE_E2E=1 to run Qwen3.8 System One integration test")
	}
	if _, err := os.Stat(systemOneE2EModelPath); err != nil {
		t.Skipf("System One development model not found: %s", systemOneE2EModelPath)
	}

	pipeline, err := NewPipeline(systemOneE2EModelPath, 512)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	if pipeline.Model.Config.Architecture != "qwen35" {
		t.Fatalf("architecture = %q, want qwen35", pipeline.Model.Config.Architecture)
	}

	question := SystemOneQuestion{
		ID:     "route",
		Prompt: "Which queue should handle this customer message?",
		Choices: []Choice{
			{Value: "billing", Label: "A billing, charge, refund, or payment issue."},
			{Value: "technical", Label: "A technical product or software issue."},
			{Value: "account", Label: "An account access or password issue."},
		},
	}
	result, err := pipeline.SystemOne("I was charged twice for my subscription and need a refund.", question, SystemOneConfig{})
	if err != nil {
		t.Fatalf("SystemOne: %v", err)
	}
	if len(result.Choices) != len(question.Choices) {
		t.Fatalf("choices = %d, want %d", len(result.Choices), len(question.Choices))
	}
	if result.Value == "" {
		t.Fatal("SystemOne returned an empty selected value")
	}
	if result.PromptTokens == 0 || result.ScoredTokens == 0 {
		t.Fatalf("unexpected token accounting: prompt=%d scored=%d", result.PromptTokens, result.ScoredTokens)
	}
	var probabilitySum float64
	for _, choice := range result.Choices {
		if math.IsNaN(choice.LogProb) || math.IsNaN(choice.Probability) || math.IsInf(choice.LogProb, 0) {
			t.Fatalf("invalid score for %q: %+v", choice.Value, choice)
		}
		probabilitySum += choice.Probability
	}
	if math.Abs(probabilitySum-1) > 1e-5 {
		t.Fatalf("probability sum = %.8f, want 1", probabilitySum)
	}
	t.Logf("System One result: value=%q confidence=%.4f prefill=%.1fms score=%.1fms choices=%+v",
		result.Value, result.Confidence, result.PrefillTimeMs, result.ScoreTimeMs, result.Choices)
}

func loadSystemOneE2EPipeline(t *testing.T) *Pipeline {
	t.Helper()
	if os.Getenv("DLGO_SYSTEMONE_E2E") != "1" {
		t.Skip("set DLGO_SYSTEMONE_E2E=1 to run Qwen3.8 System One integration tests")
	}
	if _, err := os.Stat(systemOneE2EModelPath); err != nil {
		t.Skipf("System One development model not found: %s", systemOneE2EModelPath)
	}
	pipeline, err := NewPipeline(systemOneE2EModelPath, 512)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	return pipeline
}

func runSystemOneCase(t *testing.T, pipeline *Pipeline, name, state string, question SystemOneQuestion) *SystemOneResult {
	t.Helper()
	result, err := pipeline.SystemOne(state, question, SystemOneConfig{})
	if err != nil {
		t.Fatalf("%s: SystemOne: %v", name, err)
	}
	if len(result.Choices) != len(question.Choices) {
		t.Fatalf("%s: got %d choices, want %d", name, len(result.Choices), len(question.Choices))
	}
	var sum float64
	for _, choice := range result.Choices {
		if math.IsNaN(choice.Probability) || math.IsInf(choice.Probability, 0) || math.IsNaN(choice.LogProb) || math.IsInf(choice.LogProb, 0) {
			t.Fatalf("%s: invalid score: %+v", name, choice)
		}
		sum += choice.Probability
	}
	if math.Abs(sum-1) > 1e-5 {
		t.Fatalf("%s: probability sum %.8f", name, sum)
	}
	t.Logf("%s: value=%q confidence=%.3f prefill=%.1fms score=%.1fms choices=%+v", name, result.Value, result.Confidence, result.PrefillTimeMs, result.ScoreTimeMs, result.Choices)
	return result
}

func TestSystemOneMultivariateCasesWithQwen38(t *testing.T) {
	pipeline := loadSystemOneE2EPipeline(t)
	cases := []struct {
		name, state, prompt, want string
		choices                   []Choice
	}{
		{name: "moderation", state: "The user says: You are an idiot and I hate you.", prompt: "Classify the message.", want: "abusive"},
		{name: "sentiment", state: "The product arrived early and works perfectly.", prompt: "What is the sentiment?", want: "positive"},
		{name: "binary", state: "The user cannot log in and asks to reset a forgotten password.", prompt: "Does this require account recovery?", want: "yes"},
		{name: "technical", state: "The application crashes with an out-of-memory error on startup.", prompt: "Which team should handle this?", want: "technical", choices: []Choice{{Value: "billing", Label: "Billing or payment issue.", Code: " A"}, {Value: "technical", Label: "Technical software issue.", Code: " B"}, {Value: "account", Label: "Account access issue.", Code: " C"}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			choices := tc.choices
			if choices == nil {
				switch tc.name {
				case "moderation":
					choices = []Choice{{Value: "safe", Label: "A harmless message."}, {Value: "abusive", Label: "An insulting or abusive message."}, {Value: "threat", Label: "A credible threat of harm."}}
				case "sentiment":
					choices = []Choice{{Value: "positive", Label: "Positive sentiment."}, {Value: "negative", Label: "Negative sentiment."}, {Value: "neutral", Label: "Neutral sentiment."}}
				case "binary":
					choices = []Choice{{Value: "yes", Label: "Yes, account recovery is required."}, {Value: "no", Label: "No, account recovery is not required."}}
				case "technical":
					choices = []Choice{{Value: "billing", Label: "Billing or payment issue."}, {Value: "technical", Label: "Technical software issue."}, {Value: "account", Label: "Account access issue."}}
				}
			}
			result := runSystemOneCase(t, pipeline, tc.name, tc.state, SystemOneQuestion{ID: tc.name, Prompt: tc.prompt, Choices: choices})
			if result.Value != tc.want {
				// This is an evaluation matrix, not a correctness oracle: a
				// generic causal LM can disagree with the expected class. Keep
				// the result visible without making the integration suite flaky.
				t.Logf("evaluation mismatch: selected %q, expected semantic class %q", result.Value, tc.want)
			}
		})
	}
}

func TestSystemOneChoiceOrderAndDeterminismWithQwen38(t *testing.T) {
	pipeline := loadSystemOneE2EPipeline(t)
	state := "I was charged twice and need a refund."
	question := SystemOneQuestion{ID: "route", Prompt: "Choose the best queue.", Choices: []Choice{
		{Value: "billing", Label: "Billing, refund, or payment issue."},
		{Value: "technical", Label: "Technical software issue."},
		{Value: "account", Label: "Account access issue."},
	}}
	first := runSystemOneCase(t, pipeline, "order-a", state, question)
	second := runSystemOneCase(t, pipeline, "order-a-repeat", state, question)
	if first.Value != second.Value || math.Abs(first.Confidence-second.Confidence) > 1e-6 {
		t.Fatalf("non-deterministic result: first=%+v second=%+v", first, second)
	}
	reversed := question
	reversed.Choices = []Choice{question.Choices[2], question.Choices[0], question.Choices[1]}
	third := runSystemOneCase(t, pipeline, "order-b", state, reversed)
	if third.Value != "billing" {
		t.Logf("choice-order permutation selected %q instead of billing; this is an evaluation signal for code/prompt bias", third.Value)
	}
}

func TestSystemOneConfidenceAbstentionWithQwen38(t *testing.T) {
	pipeline := loadSystemOneE2EPipeline(t)
	question := SystemOneQuestion{ID: "ambiguous", Prompt: "Choose the best category.", Choices: []Choice{
		{Value: "a", Label: "Category A."}, {Value: "b", Label: "Category B."}, {Value: "c", Label: "Category C."},
	}}
	result, err := pipeline.SystemOne("This is an ambiguous request with no clear category.", question, SystemOneConfig{MinConfidence: 1.01})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Abstained || result.Value != "" {
		t.Fatalf("expected abstention, got %+v", result)
	}
}
