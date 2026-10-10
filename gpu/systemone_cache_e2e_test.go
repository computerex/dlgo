//go:build cgo && vulkan

package gpu

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// Exercise request-local recurrent-only branching, cache interleaving, and
// eviction with a real model. Synthetic snapshots alone cannot validate that
// attention suffix passes leave the prefix region immutable.
func TestSystemOneCacheBranchesE2E(t *testing.T) {
	_, p, _, head := newSystemOneTestPipelines(t)
	questions := []llm.SystemOneQuestion{
		{ID: "sentiment", Prompt: "Classify sentiment.", Choices: []llm.Choice{{Value: "positive", Label: "positive"}, {Value: "negative", Label: "negative"}}},
		{ID: "intent", Prompt: "What does the customer need?", Choices: []llm.Choice{{Value: "refund", Label: "request a refund"}, {Value: "delivery", Label: "track a delivery"}, {Value: "account", Label: "fix account access"}}},
	}
	state := "I am disappointed. My order arrived damaged and I want my money back."
	cfg := llm.SystemOneConfig{}
	p.freePrefixCache()
	fresh, err := p.SystemOneMulti(state, questions, cfg, head)
	if err != nil {
		t.Fatal(err)
	}
	compare := func(label string) {
		t.Helper()
		got, err := p.SystemOneMulti(state, questions, cfg, head)
		if err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if got[i].Value != fresh[i].Value {
				t.Fatalf("%s question %d pick changed", label, i)
			}
			for j := range got[i].Choices {
				drift := math.Abs(got[i].Choices[j].Probability - fresh[i].Choices[j].Probability)
				if drift > 1e-6 {
					t.Fatalf("%s question %d choice %d probability drift %g", label, i, j, drift)
				}
			}
		}
	}
	compare("cache hit")
	// Compare shared branches against independent full prompt extraction;
	// cache-vs-cache alone could conceal a consistently incorrect branch.
	for i, q := range questions {
		standalone, err := p.SystemOneWithReadout(state, q, cfg, head)
		if err != nil {
			t.Fatal(err)
		}
		if standalone.Value != fresh[i].Value {
			t.Fatalf("standalone question %d pick changed", i)
		}
		for j := range standalone.Choices {
			if drift := math.Abs(standalone.Choices[j].Probability - fresh[i].Choices[j].Probability); drift > 0.02 {
				t.Fatalf("standalone question %d choice %d drift %g", i, j, drift)
			}
		}
	}
	if _, err := p.SystemOneWithReadout("The shipment has not arrived. Please track it.", questions[1], cfg, head); err != nil {
		t.Fatal(err)
	}
	compare("after standalone")
	for _, other := range []string{"A different document.", "Another different document.", "Third unrelated document.", "Fourth unrelated document.", "Fifth unrelated document."} {
		if _, err := p.SystemOneMulti(other, questions, cfg, head); err != nil {
			t.Fatal(err)
		}
	}
	compare("after eviction")
}

func TestSystemOneGPUCanceledBeforeInference(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	p := &GpuPipeline{}
	if _, err := p.SystemOneMulti("", nil, llm.SystemOneConfig{Context: ctx}, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
	if _, _, _, err := p.SystemOneHidden("", llm.SystemOneQuestion{}, llm.SystemOneConfig{Context: ctx}); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
}
