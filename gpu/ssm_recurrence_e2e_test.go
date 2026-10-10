//go:build cgo && vulkan

package gpu

import (
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/computerex/dlgo/models/llm"
)

// Compare cold (reset state, no prefix cache) model prefill in the same
// process. Opt-in; does not load a readout or compare CPU/GPU numerics.
func TestSSMRecurrenceModelE2E(t *testing.T) {
	if os.Getenv("DLGO_SYSTEMONE_E2E") != "1" {
		t.Skip("set DLGO_SYSTEMONE_E2E=1")
	}
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(Shutdown)
	cpu, err := llm.NewPipeline(gpuTestModelPath, 16384)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { cpu.Model.Close() })
	pipe, err := NewGpuPipeline(cpu)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pipe.FreeAll)
	old := ssmReference
	defer func() { ssmReference = old }()
	q := llm.SystemOneQuestion{ID: "area", Prompt: "Which legal area does this question involve?", Choices: []llm.Choice{{Value: "housing", Label: "Leases and tenancy."}, {Value: "privacy", Label: "Personal data and privacy."}, {Value: "employment", Label: "Employment and workplace."}}}
	short := "Our landlord raised the rent in the middle of a fixed-term lease and says we must accept it or leave within two weeks."
	long := strings.Repeat("The controller processes personal data under the GDPR. Data subjects may request access, correction, erasure and restriction. Processing requires a lawful basis and appropriate security safeguards. ", 340)
	run := func(state string, reference bool) ([]float32, int, time.Duration) {
		ssmReference = reference
		start := time.Now()
		hidden, _, tokens, err := pipe.SystemOneHidden(state, q, llm.SystemOneConfig{})
		if err != nil {
			t.Fatal(err)
		}
		return hidden, tokens, time.Since(start)
	}
	// Warm pipelines and batch buffers, not recurrent state: each call resets it.
	run(short, true)
	run(short, false)
	for _, c := range []struct{ name, state string }{{"short", short}, {"long", long}} {
		t.Run(c.name, func(t *testing.T) {
			var refTime, fusedTime time.Duration
			for repeat := 0; repeat < 3; repeat++ {
				var x, y []float32
				var nx, ny int
				var rx, ry time.Duration
				if repeat%2 == 0 {
					x, nx, rx = run(c.state, true)
					y, ny, ry = run(c.state, false)
				} else {
					y, ny, ry = run(c.state, false)
					x, nx, rx = run(c.state, true)
				}
				if nx != ny {
					t.Fatalf("tokens differ %d/%d", nx, ny)
				}
				if len(x) != len(y) {
					t.Fatal("hidden dimensions differ")
				}
				for i := range x {
					if math.IsNaN(float64(x[i])) || math.IsInf(float64(x[i]), 0) || math.Float32bits(x[i]) != math.Float32bits(y[i]) {
						t.Fatalf("hidden mismatch at %d: reference=%g fused=%g", i, x[i], y[i])
					}
				}
				refTime += rx
				fusedTime += ry
				t.Logf("repeat=%d tokens=%d reference=%.3fms fused=%.3fms speedup=%.2fx hidden=bit-identical", repeat+1, nx, float64(rx)/1e6, float64(ry)/1e6, float64(rx)/float64(ry))
			}
			t.Logf("GPU=%s mean reference=%.3fms fused=%.3fms speedup=%.2fx (cold state, warm pipelines; single-question hidden extraction, not HTTP)", DeviceName(), float64(refTime)/3e6, float64(fusedTime)/3e6, float64(refTime)/float64(fusedTime))
		})
	}
}
