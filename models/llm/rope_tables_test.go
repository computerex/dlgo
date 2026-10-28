package llm

import (
	"reflect"
	"testing"

	"github.com/computerex/dlgo/ops"
)

// TestRoPETablesForMatchesRunState locks the invariant that RoPETablesFor
// reproduces, exactly, the tables a full RunState would build.
//
// gpu.NewGpuPipeline carries RoPE tables over from the CPU run state. A caller
// that used llm.WithoutCPUState() (the HTTP server does, to avoid a multi-GiB
// transient allocation) has no run state, so it constructs them with
// RoPETablesFor instead. If that ever diverges from NewRunState, a
// WithoutCPUState pipeline would silently run on unscaled frequencies: gpt-oss-20b
// (rope scaling type 2, factor 32, native context 4096) then mis-rotates every
// position beyond 4096 and its sliding-window layers lose their separate SWA
// base — producing fluent-looking garbage on long prompts while short prompts
// still look correct.
func TestRoPETablesForMatchesRunState(t *testing.T) {
	const seq = 512

	base := ModelConfig{
		EmbeddingDim: 64,
		NumHeads:     4,
		NumKVHeads:   2,
		HeadDim:      16,
		FFNDim:       128,
		VocabSize:    256,
		RopeDim:      16,
		RopeFreqBase: 10000,
	}

	yarn := base
	yarn.RopeFreqBase = 150000
	yarn.RopeScaleType = 2
	yarn.RopeScaleFactor = 32
	yarn.RopeOrigMaxPos = 4096
	yarn.RopeYaRNBetaFast = 32
	yarn.RopeYaRNBetaSlow = 1
	yarn.RopeYaRNExtFactor = 1
	yarn.RopeYaRNAttnFactor = 1

	yarnSWA := yarn
	yarnSWA.RopeFreqBaseSWA = 10000

	partial := base
	partial.RopeDim = 8

	cases := []struct {
		name        string
		cfg         ModelConfig
		wantScaling bool // tables must differ from unscaled frequencies
		wantSWA     bool // a separate SWA table must exist
	}{
		{name: "plain", cfg: base},
		{name: "partial_rope", cfg: partial},
		{name: "yarn", cfg: yarn, wantScaling: true},
		{name: "yarn_swa", cfg: yarnSWA, wantScaling: true, wantSWA: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rs := NewRunState(tc.cfg, seq)
			wantCos, wantSin := rs.RoPETables()
			wantSWACos, wantSWASin := rs.RoPETablesSWA()

			gotCos, gotSin, gotSWACos, gotSWASin := RoPETablesFor(tc.cfg, seq)

			if !reflect.DeepEqual(gotCos, wantCos) || !reflect.DeepEqual(gotSin, wantSin) {
				t.Fatalf("main RoPE table differs from NewRunState (%d/%d vs %d/%d samples)",
					len(gotCos), len(gotSin), len(wantCos), len(wantSin))
			}
			if !reflect.DeepEqual(gotSWACos, wantSWACos) || !reflect.DeepEqual(gotSWASin, wantSWASin) {
				t.Fatalf("SWA RoPE table differs from NewRunState (nil-ness %v/%v vs %v/%v)",
					gotSWACos == nil, gotSWASin == nil, wantSWACos == nil, wantSWASin == nil)
			}

			if tc.wantSWA {
				if len(gotSWACos) == 0 || len(gotSWASin) == 0 {
					t.Fatal("model declares a separate SWA frequency base but no SWA table was built")
				}
				if reflect.DeepEqual(gotSWACos, gotCos) {
					t.Fatal("SWA table is identical to the main table; the separate frequency base was ignored")
				}
			} else if gotSWACos != nil || gotSWASin != nil {
				t.Fatal("unexpected SWA RoPE table for a model without a separate SWA frequency base")
			}

			if tc.wantScaling {
				plainCos, plainSin := ops.RoPEFrequencyTable(seq, tc.cfg.RopeDim, tc.cfg.RopeFreqBase)
				if reflect.DeepEqual(gotCos, plainCos) || reflect.DeepEqual(gotSin, plainSin) {
					t.Fatal("scaled model produced unscaled RoPE tables")
				}
			}
		})
	}
}
