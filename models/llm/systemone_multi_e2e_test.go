package llm

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// TestSystemOneMultiParityE2E verifies that shared-prefill multi-question
// evaluation returns the same decisions as evaluating each question
// standalone, for both the trained readout path and the continuation-scoring
// fallback. Shared prefill is an optimization, not a behavior change: any
// divergence means the cloned-branch state is wrong. It is opt-in because it
// loads a model.
func TestSystemOneMultiParityE2E(t *testing.T) {
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

	state := "A payment was taken after I had already cancelled my membership, and I still cannot log in because the account uses an old email address."
	questions := []SystemOneQuestion{
		{ID: "route", Prompt: "Which team should handle this customer message?", Choices: []Choice{
			{Value: "billing", Label: "Billing, charges, refunds, invoices, or payments."},
			{Value: "technical", Label: "Technical software or product malfunction."},
			{Value: "account", Label: "Account access, password, or login issue."},
		}},
		{ID: "noul", Prompt: "Is this customer at risk of churning over the billing dispute?", Choices: []Choice{
			{Value: "yes", Label: "yes"},
			{Value: "no", Label: "no"},
		}},
		{ID: "severity", Prompt: "How severe is this issue for the customer?", Choices: []Choice{
			{Value: "minor", Label: "Minor inconvenience."},
			{Value: "major", Label: "Major blocking problem."},
		}},
		// Wider than the readout's option count: must degrade to continuation
		// scoring instead of failing the multi request.
		{ID: "rating", Prompt: "Rate the overall frustration level from 0 to 9.", Choices: func() []Choice {
			c := make([]Choice, 0, 10)
			for v := 0; v <= 9; v++ {
				c = append(c, Choice{Value: fmt.Sprintf("%d", v)})
			}
			return c
		}()},
	}

	for _, tc := range []struct {
		name    string
		readout *SystemOneReadout
	}{
		{name: "readout", readout: readout},
		{name: "fallback", readout: nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			multi, err := pipeline.SystemOneMulti(state, questions, SystemOneConfig{}, tc.readout)
			if err != nil {
				t.Fatalf("SystemOneMulti: %v", err)
			}
			if len(multi) != len(questions) {
				t.Fatalf("got %d results, want %d", len(multi), len(questions))
			}
			for i, q := range questions {
				// The wide rating question cannot be served by the readout at
				// all, so its standalone baseline is the fallback engine.
				wantDegraded := q.ID == "rating" && tc.readout != nil
				var single *SystemOneResult
				if tc.readout != nil && !wantDegraded {
					single, err = pipeline.SystemOneWithReadout(state, q, SystemOneConfig{}, tc.readout)
				} else {
					single, err = pipeline.SystemOne(state, q, SystemOneConfig{})
				}
				if err != nil {
					t.Fatalf("standalone question %d: %v", i, err)
				}
				if multi[i].QuestionID != q.ID {
					t.Errorf("question %d: result ID %q, want %q (order must match caller order)", i, multi[i].QuestionID, q.ID)
				}
				if multi[i].Value != single.Value {
					t.Errorf("question %d (%s): multi value %q, standalone %q", i, q.ID, multi[i].Value, single.Value)
				}
				if math.Abs(multi[i].Confidence-single.Confidence) > 1e-4 {
					t.Errorf("question %d (%s): multi confidence %.6f, standalone %.6f", i, q.ID, multi[i].Confidence, single.Confidence)
				}
				if len(multi[i].Choices) != len(single.Choices) {
					t.Errorf("question %d (%s): choice count %d, standalone %d", i, q.ID, len(multi[i].Choices), len(single.Choices))
				}
				if multi[i].Degraded != wantDegraded {
					t.Errorf("question %d (%s): degraded=%v, want %v (wide question must degrade)", i, q.ID, multi[i].Degraded, wantDegraded)
				}
				t.Logf("%s multi=%q conf=%.4f standalone=%q conf=%.4f sharedPrefill=%.1fms suffixPass=%.1fms",
					q.ID, multi[i].Value, multi[i].Confidence, single.Value, single.Confidence, multi[i].SharedPrefillTimeMs, multi[i].PrefillTimeMs)
			}
			if multi[0].SharedPrefillTimeMs <= 0 {
				t.Errorf("SharedPrefillTimeMs = %v, want > 0 for shared-prefill path", multi[0].SharedPrefillTimeMs)
			}
		})
	}
}

// BenchmarkSystemOneMultiVsStandalone measures wall-clock cost of answering the
// same three questions either as three standalone evaluations or as one
// shared-prefill SystemOneMulti call, for the readout and fallback engines.
// Run with DLGO_SYSTEMONE_E2E=1 and: go test -run xxx -bench BenchmarkSystemOneMultiVsStandalone -benchtime 5x
func BenchmarkSystemOneMultiVsStandalone(b *testing.B) {
	if os.Getenv("DLGO_SYSTEMONE_E2E") != "1" {
		b.Skip("set DLGO_SYSTEMONE_E2E=1 to run Q4 System One readout E2E")
	}
	if _, err := os.Stat(systemOneQ4ModelPath); err != nil {
		b.Skip("Q4 System One model is not available")
	}
	readout, err := LoadSystemOneReadout(systemOneReadoutPath, 1024)
	if err != nil {
		b.Fatalf("LoadSystemOneReadout: %v", err)
	}
	pipeline, err := NewPipeline(systemOneQ4ModelPath, 512)
	if err != nil {
		b.Fatalf("NewPipeline: %v", err)
	}
	defer pipeline.Model.Close()

	state := "A payment was taken after I had already cancelled my membership, and I still cannot log in because the account uses an old email address."
	questions := []SystemOneQuestion{
		{ID: "route", Prompt: "Which team should handle this customer message?", Choices: []Choice{
			{Value: "billing", Label: "Billing, charges, refunds, invoices, or payments."},
			{Value: "technical", Label: "Technical software or product malfunction."},
			{Value: "account", Label: "Account access, password, or login issue."},
		}},
		{ID: "noul", Prompt: "Is this customer at risk of churning over the billing dispute?", Choices: []Choice{
			{Value: "yes", Label: "yes"},
			{Value: "no", Label: "no"},
		}},
		{ID: "severity", Prompt: "How severe is this issue for the customer?", Choices: []Choice{
			{Value: "minor", Label: "Minor inconvenience."},
			{Value: "major", Label: "Major blocking problem."},
		}},
	}

	// Warm up both engines once so lazy allocations do not skew the first bench.
	if _, err := pipeline.SystemOneMulti(state, questions, SystemOneConfig{}, readout); err != nil {
		b.Fatalf("warmup multi: %v", err)
	}
	if _, err := pipeline.SystemOneWithReadout(state, questions[0], SystemOneConfig{}, readout); err != nil {
		b.Fatalf("warmup standalone: %v", err)
	}

	b.Run("standalone-readout", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, q := range questions {
				if _, err := pipeline.SystemOneWithReadout(state, q, SystemOneConfig{}, readout); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
	b.Run("multi-readout", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if _, err := pipeline.SystemOneMulti(state, questions, SystemOneConfig{}, readout); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("standalone-fallback", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, q := range questions {
				if _, err := pipeline.SystemOne(state, q, SystemOneConfig{}); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
	b.Run("multi-fallback", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if _, err := pipeline.SystemOneMulti(state, questions, SystemOneConfig{}, nil); err != nil {
				b.Fatal(err)
			}
		}
	})
}