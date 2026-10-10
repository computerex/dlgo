//go:build cgo && vulkan

package gpu

import (
	"math"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// maxAbsDiff returns the largest absolute element-wise difference.
func maxAbsDiff(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var m float64
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

// TestSystemOneGPUChunkDiagE2E isolates where the shared-prefill multi suffix
// pass diverges from a single-call prefill:
//
//	h1: whole prompt in ONE hybrid batch call (startPos=0, lastChunk=true)
//	h2: prefix call (lastChunk=false) + suffix call (startPos=P, lastChunk=true)
//	h3: like h2 but with snapshot/restore around an interleaved other-question
//	    suffix pass (the production multi flow)
//
// h1 vs h2 isolates the startPos>0 batch path; h2 vs h3 isolates
// snapshot/restore. Readout decisions are printed for each path.
func TestSystemOneGPUChunkDiagE2E(t *testing.T) {
	_, gpuPipe, _, gpuReadout := newSystemOneTestPipelines(t)

	state := "I was charged twice for my subscription this month, the app crashes when I open the invoice page, and I cannot log in from my new phone."
	target := llm.SystemOneQuestion{ID: "route", Prompt: "Which team should handle this customer message?", Choices: []llm.Choice{
		{Value: "billing", Label: "Billing, charges, refunds, invoices, subscriptions, or payments."},
		{Value: "technical", Label: "Technical software, device, service, or product malfunction."},
		{Value: "account", Label: "Account access, identity, password, login, or authentication issue."},
	}}
	other := llm.BuildSystemOneNoulQuestion("Does this message describe a billing problem?")
	cfg := llm.SystemOneConfig{}

	prompt, targetChoices, err := llm.CompileSystemOnePrompt(gpuPipe.CPUModel.Config, state, target, cfg)
	if err != nil {
		t.Fatal(err)
	}
	tokens := gpuPipe.Tokenizer.Encode(prompt)
	npos := len(tokens)
	dim := gpuPipe.CPUModel.Config.EmbeddingDim
	split := npos / 2
	t.Logf("prompt tokens=%d split=%d", npos, split)

	if err := gpuPipe.ensurePrefillBatchState(npos); err != nil {
		t.Fatal(err)
	}

	runSuffix := func(suffix []int32, startPos int, hidden []float32) {
		GpuForwardPrefillBatchHybrid(gpuPipe.CPUModel, gpuPipe.GpuModel, suffix, gpuPipe.KVCache, gpuPipe.RunState,
			gpuPipe.BatchState, nil, gpuPipe.BatchLayerConfs, gpuPipe, startPos, true, hidden)
		Sync()
	}

	// h1: single call over the whole prompt.
	gpuPipe.ResetState()
	h1 := make([]float32, dim)
	GpuForwardPrefillBatchHybrid(gpuPipe.CPUModel, gpuPipe.GpuModel, tokens, gpuPipe.KVCache, gpuPipe.RunState,
		gpuPipe.BatchState, nil, gpuPipe.BatchLayerConfs, gpuPipe, 0, true, h1)
	Sync()

	// h2: prefix + suffix, no snapshot/restore.
	gpuPipe.ResetState()
	h2 := make([]float32, dim)
	GpuForwardPrefillBatchHybrid(gpuPipe.CPUModel, gpuPipe.GpuModel, tokens[:split], gpuPipe.KVCache, gpuPipe.RunState,
		gpuPipe.BatchState, nil, gpuPipe.BatchLayerConfs, gpuPipe, 0, false, nil)
	Sync()
	runSuffix(tokens[split:], split, h2)

	// h3: prefix + snapshot + interleaved other suffix + restore + suffix.
	gpuPipe.ResetState()
	h3 := make([]float32, dim)
	GpuForwardPrefillBatchHybrid(gpuPipe.CPUModel, gpuPipe.GpuModel, tokens[:split], gpuPipe.KVCache, gpuPipe.RunState,
		gpuPipe.BatchState, nil, gpuPipe.BatchLayerConfs, gpuPipe, 0, false, nil)
	Sync()
	snap, err := gpuPipe.SnapshotState(split)
	if err != nil {
		t.Fatal(err)
	}
	// Interleave another question's suffix pass (mutates live state), exactly
	// like the production multi loop does between questions.
	otherPrompt, _, err := llm.CompileSystemOnePrompt(gpuPipe.CPUModel.Config, state, other, cfg)
	if err != nil {
		t.Fatal(err)
	}
	otherTokens := gpuPipe.Tokenizer.Encode(otherPrompt)
	if len(otherTokens) > split {
		otherTokens = otherTokens[:split]
	}
	runSuffix(otherTokens[len(otherTokens)-1:], split+len(otherTokens)-1, make([]float32, dim))
	if err := gpuPipe.RestoreState(snap); err != nil {
		t.Fatal(err)
	}
	snap.Free()
	runSuffix(tokens[split:], split, h3)

	t.Logf("cos(h1,h2)=%.6f maxdiff=%.3e  (1=single call, 2=plain chunked)", cosineSim(h1, h2), maxAbsDiff(h1, h2))
	t.Logf("cos(h1,h3)=%.6f maxdiff=%.3e  (3=snapshot/restore flow)", cosineSim(h1, h3), maxAbsDiff(h1, h3))
	t.Logf("cos(h2,h3)=%.6f maxdiff=%.3e", cosineSim(h2, h3), maxAbsDiff(h2, h3))

	for name, h := range map[string][]float32{"h1": h1, "h2": h2, "h3": h3} {
		r, err := llm.ApplySystemOneReadout(target, targetChoices, append([]float32(nil), h...), gpuReadout, cfg, npos, 0)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s decision: %q conf=%.4f", name, r.Value, r.Confidence)
	}
}