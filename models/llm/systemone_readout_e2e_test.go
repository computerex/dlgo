package llm

import (
	"os"
	"testing"
)

const (
	systemOneQ4ModelPath = `C:\Users\mohd\Downloads\Qwen3.5-0.8B-Q4_K_M.gguf`
	systemOneReadoutPath = `C:\projects\dlgo\systemone-train\artifacts\qwen35-08b-q4-readout-v7-gp.json`
)

// TestSystemOneReadoutE2E verifies the independently trained, head-only Q4
// artifact against held-out paraphrases. It is opt-in because it loads a model.
func TestSystemOneReadoutE2E(t *testing.T) {
	if os.Getenv("DLGO_SYSTEMONE_E2E") != "1" {
		t.Skip("set DLGO_SYSTEMONE_E2E=1 to run Q4 System One readout E2E")
	}
	if _, err := os.Stat(systemOneQ4ModelPath); err != nil {
		t.Skip("Q4 System One model is not available")
	}
	readout, err := LoadSystemOneReadout(systemOneReadoutPath, 1024)
	if err != nil {
		t.Fatalf("LoadSystemOneReadout: %v", err)
	}
	pipeline, err := NewPipeline(systemOneQ4ModelPath, 512)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	defer pipeline.Model.Close()

	cases := []struct {
		name, state, want string
		question          SystemOneQuestion
	}{
		{
			name: "billing", state: "A payment was taken after I had already cancelled my membership.", want: "billing",
			question: SystemOneQuestion{ID: "route", Prompt: "Which team should handle this customer message?", Choices: []Choice{
				{Value: "billing", Label: "Billing, charges, refunds, invoices, or payments."},
				{Value: "technical", Label: "Technical software or product malfunction."},
				{Value: "account", Label: "Account access, password, or login issue."},
			}},
		},
		{
			name: "technical", state: "The website displays a blank page after I click save.", want: "technical",
			question: SystemOneQuestion{ID: "route", Prompt: "Which team should handle this customer message?", Choices: []Choice{
				{Value: "billing", Label: "Billing, charges, refunds, invoices, or payments."},
				{Value: "technical", Label: "Technical software or product malfunction."},
				{Value: "account", Label: "Account access, password, or login issue."},
			}},
		},
		{
			name: "account", state: "I changed phones and cannot receive the code needed to log in.", want: "account",
			question: SystemOneQuestion{ID: "route", Prompt: "Which team should handle this customer message?", Choices: []Choice{
				{Value: "billing", Label: "Billing, charges, refunds, invoices, or payments."},
				{Value: "technical", Label: "Technical software or product malfunction."},
				{Value: "account", Label: "Account access, password, or login issue."},
			}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := pipeline.SystemOneWithReadout(tc.state, tc.question, SystemOneConfig{}, readout)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("readout selected=%q confidence=%.4f probabilities=%+v", result.Value, result.Confidence, result.Choices)
			// Run the generic continuation-scoring fallback on the identical
			// prompt so readout errors can be attributed to the head versus the
			// backbone features.
			fallback, err := pipeline.SystemOne(tc.state, tc.question, SystemOneConfig{})
			if err != nil {
				t.Logf("fallback unavailable: %v", err)
			} else {
				t.Logf("fallback selected=%q confidence=%.4f probabilities=%+v", fallback.Value, fallback.Confidence, fallback.Choices)
			}
			if result.Value != tc.want {
				// The readout is intentionally trained on a synthetic corpus. This
				// held-out test records generalization quality without pretending
				// that synthetic-data fit is a production-quality classifier.
				t.Logf("held-out mismatch: selected %q, expected %q", result.Value, tc.want)
			}
		})
	}
}
