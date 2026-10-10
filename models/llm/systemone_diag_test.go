package llm

import (
	"os"
	"testing"
	"time"
)

// TestSystemOneSuffixCostBreakdownE2E measures where shared-prefill suffix time
// goes: KV/SSM cloning, the batched suffix pass, and the final single-token
// hidden extraction, compared against a full standalone prefill and tiny-batch
// probes that expose fixed per-call overhead. Opt-in; diagnostics only.
func TestSystemOneSuffixCostBreakdownE2E(t *testing.T) {
	if os.Getenv("DLGO_SYSTEMONE_E2E") != "1" {
		t.Skip("set DLGO_SYSTEMONE_E2E=1 to run Q4 System One readout E2E")
	}
	if _, err := os.Stat(systemOneQ4ModelPath); err != nil {
		t.Skip("Q4 System One model is not available")
	}
	pipeline, err := NewPipeline(systemOneQ4ModelPath, 512)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	defer pipeline.Model.Close()

	state := "A payment was taken after I had already cancelled my membership, and I still cannot log in because the account uses an old email address."
	q := SystemOneQuestion{ID: "route", Prompt: "Which team should handle this customer message?", Choices: []Choice{
		{Value: "billing", Label: "Billing, charges, refunds, invoices, or payments."},
		{Value: "technical", Label: "Technical software or product malfunction."},
		{Value: "account", Label: "Account access, password, or login issue."},
	}}
	parts, err := compileSystemOneParts(state, q, SystemOneConfig{})
	if err != nil {
		t.Fatal(err)
	}
	prefixTokens := pipeline.Tokenizer.Encode(parts.Prefix)
	suffixTokens := pipeline.Tokenizer.Encode(parts.Suffix)
	t.Logf("prefix tokens=%d suffix tokens=%d", len(prefixTokens), len(suffixTokens))

	timeit := func(name string, fn func()) {
		start := time.Now()
		fn()
		t.Logf("%-42s %8.1fms", name, float64(time.Since(start).Microseconds())/1000.0)
	}

	// Warm every path once.
	timeit("warmup full prefill", func() {
		pipeline.KVCache.Reset()
		if pipeline.RunState.SSMState != nil {
			pipeline.RunState.SSMState.Reset()
		}
		ForwardBatch(pipeline.Model, append(append([]int32(nil), prefixTokens...), suffixTokens...), 0, pipeline.KVCache, pipeline.RunState, pipeline.BatchState)
	})

	const reps = 3
	avg := func(name string, fn func()) {
		var total time.Duration
		for i := 0; i < reps; i++ {
			start := time.Now()
			fn()
			total += time.Since(start)
		}
		t.Logf("%-42s %8.1fms (avg of %d)", name, float64(total.Microseconds())/float64(reps)/1000.0, reps)
	}

	avg("full standalone prefill (prefix+suffix)", func() {
		pipeline.KVCache.Reset()
		if pipeline.RunState.SSMState != nil {
			pipeline.RunState.SSMState.Reset()
		}
		ForwardBatch(pipeline.Model, append(append([]int32(nil), prefixTokens...), suffixTokens...), 0, pipeline.KVCache, pipeline.RunState, pipeline.BatchState)
	})

	avg("shared prefix prefill only", func() {
		pipeline.KVCache.Reset()
		if pipeline.RunState.SSMState != nil {
			pipeline.RunState.SSMState.Reset()
		}
		ForwardBatch(pipeline.Model, prefixTokens, 0, pipeline.KVCache, pipeline.RunState, pipeline.BatchState)
	})

	var baseKV = pipeline.KVCache.Clone(len(prefixTokens))
	var baseSSM = pipeline.RunState.SSMState.Clone()
	avg("KV+SSM clone at prefix boundary", func() {
		baseKV = pipeline.KVCache.Clone(len(prefixTokens))
		baseSSM = pipeline.RunState.SSMState.Clone()
	})

	avg("suffix batch pass (startPos>0)", func() {
		kv := baseKV.Clone(len(prefixTokens))
		rs := NewRunState(pipeline.Model.Config, pipeline.MaxSeqLen)
		rs.SSMState = baseSSM.Clone()
		ForwardBatch(pipeline.Model, suffixTokens[:len(suffixTokens)-1], len(prefixTokens), kv, rs, pipeline.BatchState)
	})

	avg("suffix batch pass incl clones (per-question)", func() {
		kv := baseKV.Clone(len(prefixTokens))
		rs := NewRunState(pipeline.Model.Config, pipeline.MaxSeqLen)
		rs.SSMState = baseSSM.Clone()
		ForwardBatch(pipeline.Model, suffixTokens[:len(suffixTokens)-1], len(prefixTokens), kv, rs, pipeline.BatchState)
		_ = ForwardHidden(pipeline.Model, suffixTokens[len(suffixTokens)-1], len(prefixTokens)+len(suffixTokens)-1, kv, rs)
	})

	avg("tiny batch probe: ForwardBatch(2 tokens)", func() {
		pipeline.KVCache.Reset()
		if pipeline.RunState.SSMState != nil {
			pipeline.RunState.SSMState.Reset()
		}
		ForwardBatch(pipeline.Model, prefixTokens[:2], 0, pipeline.KVCache, pipeline.RunState, pipeline.BatchState)
	})

	avg("tiny batch probe: ForwardBatch(16 tokens)", func() {
		pipeline.KVCache.Reset()
		if pipeline.RunState.SSMState != nil {
			pipeline.RunState.SSMState.Reset()
		}
		ForwardBatch(pipeline.Model, prefixTokens[:16], 0, pipeline.KVCache, pipeline.RunState, pipeline.BatchState)
	})

	avg("single-token Forward at pos=prefixLen", func() {
		kv := baseKV.Clone(len(prefixTokens))
		rs := NewRunState(pipeline.Model.Config, pipeline.MaxSeqLen)
		rs.SSMState = baseSSM.Clone()
		_ = Forward(pipeline.Model, suffixTokens[0], len(prefixTokens), kv, rs)
	})
}