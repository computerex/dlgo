//go:build cgo && vulkan

package gpu

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

const (
	gpuTestModelPath = `C:\Users\mohd\Downloads\Qwen3.5-0.8B-Q4_K_M.gguf`
	// Device-native artifacts: each engine must be served a readout trained on
	// its own feature distribution. GPU and CPU extract structurally different
	// hidden states (GPU matvec kernels quantize activations to Q8_1), so a
	// CPU-trained readout is refused by the GPU engine and vice versa.
	gpuTestCPUReadoutPath = `C:\projects\dlgo\systemone-train\artifacts\qwen35-08b-q4-readout-v7-gp.json`
	gpuTestGPUReadoutPath = `C:\projects\dlgo\systemone-train\artifacts\qwen35-08b-q4-readout-v8-gpu.json`
	gpuTestCorpusPath     = `..\systemone-train\data\generated\splits-gp\test.jsonl`
)

// newSystemOneTestPipelines loads the CPU and GPU pipelines plus both
// device-native readouts. Opt-in via DLGO_SYSTEMONE_E2E=1 because both load a
// model.
func newSystemOneTestPipelines(t *testing.T) (*llm.Pipeline, *GpuPipeline, *llm.SystemOneReadout, *llm.SystemOneReadout) {
	t.Helper()
	if os.Getenv("DLGO_SYSTEMONE_E2E") != "1" {
		t.Skip("set DLGO_SYSTEMONE_E2E=1 to run Q4 System One GPU E2E")
	}
	if _, err := os.Stat(gpuTestModelPath); err != nil {
		t.Skip("Q4 System One model is not available")
	}
	if err := Init(); err != nil {
		t.Skipf("Vulkan init failed: %v", err)
	}
	cpuPipe, err := llm.NewPipeline(gpuTestModelPath, 512)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	gpuPipe, err := NewGpuPipeline(cpuPipe)
	if err != nil {
		t.Fatalf("NewGpuPipeline: %v", err)
	}
	cpuReadout, err := llm.LoadSystemOneReadout(gpuTestCPUReadoutPath, 1024)
	if err != nil {
		t.Fatalf("LoadSystemOneReadout(cpu): %v", err)
	}
	gpuReadout, err := llm.LoadSystemOneReadout(gpuTestGPUReadoutPath, 1024)
	if err != nil {
		t.Fatalf("LoadSystemOneReadout(gpu): %v", err)
	}
	t.Cleanup(func() {
		gpuPipe.FreeAll()
		cpuPipe.Model.Close()
	})
	return cpuPipe, gpuPipe, cpuReadout, gpuReadout
}

func cosineSim(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// TestSystemOneGPUHiddenParityE2E compares GPU hidden-state extraction against
// the CPU reference on the same compiled prompt. GPU and CPU are not
// bit-identical by design — F16 KV cache, Q8_1 activation quantization inside
// the GPU matvec kernels, F16 coopmat inputs, and different SSM accumulation
// order. Probes (DLGO_F32_KV=1, DLGO_NO_DP4A=1) showed neither switch closes
// the gap, so ~0.97 cosine is the GPU's intrinsic numerical profile. The
// production gate is device-native decision accuracy
// (TestSystemOneGPUCorpusAccuracyE2E), not feature bit-parity.
func TestSystemOneGPUHiddenParityE2E(t *testing.T) {
	cpuPipe, gpuPipe, _, _ := newSystemOneTestPipelines(t)

	state := "Our landlord raised the rent in the middle of the fixed-term lease and says we have to accept it or leave within two weeks."
	q := llm.SystemOneQuestion{ID: "area", Prompt: "Which legal area does this question involve?", Choices: []llm.Choice{
		{Value: "employment", Label: "Workplace, hiring, or termination matters."},
		{Value: "housing", Label: "Leases, tenancy, or landlord disputes."},
		{Value: "family", Label: "Marriage, divorce, custody, or guardianship."},
		{Value: "consumer", Label: "Purchases, warranties, or consumer protection."},
		{Value: "immigration", Label: "Visas, residency, or citizenship."},
	}}

	cpuHidden, cpuChoices, cpuTokens, err := cpuPipe.SystemOneHidden(state, q, llm.SystemOneConfig{})
	if err != nil {
		t.Fatalf("cpu SystemOneHidden: %v", err)
	}
	gpuHidden, gpuChoices, gpuTokens, err := gpuPipe.SystemOneHidden(state, q, llm.SystemOneConfig{})
	if err != nil {
		t.Fatalf("gpu SystemOneHidden: %v", err)
	}
	if cpuTokens != gpuTokens {
		t.Errorf("token counts differ: cpu %d, gpu %d", cpuTokens, gpuTokens)
	}
	if len(cpuChoices) != len(gpuChoices) {
		t.Fatalf("choice counts differ: cpu %d, gpu %d", len(cpuChoices), len(gpuChoices))
	}
	for i := range cpuChoices {
		if cpuChoices[i] != gpuChoices[i] {
			t.Errorf("choice %d differs: cpu %+v, gpu %+v", i, cpuChoices[i], gpuChoices[i])
		}
	}
	sim := cosineSim(cpuHidden, gpuHidden)
	t.Logf("hidden cosine similarity cpu-vs-gpu = %.6f (dim=%d)", sim, len(gpuHidden))
	if sim < 0.95 {
		t.Errorf("GPU hidden state diverged from CPU: cosine %.6f < 0.95", sim)
	}
}

// TestSystemOneReadoutDeviceGatesE2E verifies device-aware artifacts are
// refused cross-device: a CPU-trained readout cannot serve GPU decisions and a
// GPU-trained readout cannot serve CPU decisions. This is what makes a device
// mismatch fail loudly instead of serving off-distribution decisions.
func TestSystemOneReadoutDeviceGatesE2E(t *testing.T) {
	cpuPipe, gpuPipe, cpuReadout, gpuReadout := newSystemOneTestPipelines(t)

	state := "A payment was taken after I had already cancelled my membership."
	q := llm.BuildSystemOneNoulQuestion("Is this a billing problem?")

	if _, err := gpuPipe.SystemOneWithReadout(state, q, llm.SystemOneConfig{}, cpuReadout); err == nil {
		t.Error("GPU engine accepted a cpu-device readout; device gate is missing")
	} else {
		t.Logf("gpu engine refused cpu readout: %v", err)
	}
	if _, err := cpuPipe.SystemOneWithReadout(state, q, llm.SystemOneConfig{}, gpuReadout); err == nil {
		t.Error("CPU engine accepted a gpu-device readout; device gate is missing")
	} else {
		t.Logf("cpu engine refused gpu readout: %v", err)
	}
	if _, err := gpuPipe.SystemOneMulti(state, []llm.SystemOneQuestion{q}, llm.SystemOneConfig{}, cpuReadout); err == nil {
		t.Error("GPU multi accepted a cpu-device readout; device gate is missing")
	}
}

// TestSystemOneReadoutDeviceMetadata verifies the device field survives the
// artifact formats: the v8-gpu JSON and GGUF carry device=gpu, the legacy v7
// JSON normalizes to cpu. Pure file parsing; no model load.
func TestSystemOneReadoutDeviceMetadata(t *testing.T) {
	gpuReadout, err := llm.LoadSystemOneReadout(gpuTestGPUReadoutPath, 1024)
	if err != nil {
		if _, statErr := os.Stat(gpuTestGPUReadoutPath); statErr != nil {
			t.Skip("v8-gpu artifact not present")
		}
		t.Fatalf("LoadSystemOneReadout(v8-gpu): %v", err)
	}
	if gpuReadout.Device != "gpu" {
		t.Errorf("v8-gpu JSON device = %q, want gpu", gpuReadout.Device)
	}
	ggufReadout, err := llm.LoadSystemOneReadoutGGUF(strings.Replace(gpuTestGPUReadoutPath, ".json", ".gguf", 1), 1024)
	if err != nil {
		t.Fatalf("LoadSystemOneReadoutGGUF(v8-gpu): %v", err)
	}
	if ggufReadout.Device != "gpu" {
		t.Errorf("v8-gpu GGUF device = %q, want gpu", ggufReadout.Device)
	}
	cpuReadout, err := llm.LoadSystemOneReadout(gpuTestCPUReadoutPath, 1024)
	if err != nil {
		if _, statErr := os.Stat(gpuTestCPUReadoutPath); statErr != nil {
			t.Skip("v7-gp artifact not present")
		}
		t.Fatalf("LoadSystemOneReadout(v7-gp): %v", err)
	}
	if cpuReadout.Device != "cpu" {
		t.Errorf("v7-gp device = %q, want cpu (legacy artifact normalizes to cpu)", cpuReadout.Device)
	}
}

// TestSystemOneGPUReadoutDecisionsE2E runs a hand-crafted decision battery
// through both engines with device-native artifacts. Hand-picked one-liner
// states are not from the corpus distribution and both engines wobble on them
// (measured independently), so decisions are logged for observability; only
// structural engine attribution and the strong legal zero-shot case are
// asserted. Decision quality is gated by the corpus test below.
func TestSystemOneGPUReadoutDecisionsE2E(t *testing.T) {
	cpuPipe, gpuPipe, cpuReadout, gpuReadout := newSystemOneTestPipelines(t)

	routeQuestion := func() llm.SystemOneQuestion {
		return llm.SystemOneQuestion{ID: "route", Prompt: "Which team should handle this customer message?", Choices: []llm.Choice{
			{Value: "billing", Label: "Billing, charges, refunds, invoices, or payments."},
			{Value: "technical", Label: "Technical software or product malfunction."},
			{Value: "account", Label: "Account access, password, or login issue."},
		}}
	}
	cases := []struct {
		name, state string
		question    llm.SystemOneQuestion
		want        string
		assert      bool // only cases with corpus-level evidence are asserted
	}{
		{name: "billing", state: "A payment was taken after I had already cancelled my membership.", question: routeQuestion(), want: "billing"},
		{name: "technical", state: "The website displays a blank page after I click save.", question: routeQuestion(), want: "technical"},
		{name: "account", state: "I changed phones and cannot receive the code needed to log in.", question: routeQuestion(), want: "account"},
		{name: "noul-churn", state: "I have been a loyal customer for six years but this is the third double charge in two months.", question: llm.BuildSystemOneNoulQuestion("Is this customer at risk of churning?"), want: "yes"},
		{name: "legal-zeroshot", state: "Our landlord raised the rent in the middle of the fixed-term lease and says we have to accept it or leave within two weeks.",
			question: llm.SystemOneQuestion{ID: "area", Prompt: "Which legal area does this question involve?", Choices: []llm.Choice{
				{Value: "employment", Label: "Workplace, hiring, or termination matters."},
				{Value: "housing", Label: "Leases, tenancy, or landlord disputes."},
				{Value: "family", Label: "Marriage, divorce, custody, or guardianship."},
				{Value: "consumer", Label: "Purchases, warranties, or consumer protection."},
				{Value: "immigration", Label: "Visas, residency, or citizenship."},
			}}, want: "housing", assert: true},
		{name: "medical-zeroshot", state: "I have a slight sore throat and runny nose but no fever, and I can go to work as usual.",
			question: llm.SystemOneQuestion{ID: "care", Prompt: "What level of care does this symptom report need?", Choices: []llm.Choice{
				{Value: "emergency", Label: "Life-threatening; call emergency services now."},
				{Value: "urgent_care", Label: "Needs same-day professional attention."},
				{Value: "routine", Label: "Can wait for a scheduled appointment."},
				{Value: "self_care", Label: "Safe home care is appropriate."},
			}}, want: "self_care"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cpuResult, err := cpuPipe.SystemOneWithReadout(tc.state, tc.question, llm.SystemOneConfig{}, cpuReadout)
			if err != nil {
				t.Fatalf("cpu: %v", err)
			}
			gpuResult, err := gpuPipe.SystemOneWithReadout(tc.state, tc.question, llm.SystemOneConfig{}, gpuReadout)
			if err != nil {
				t.Fatalf("gpu: %v", err)
			}
			t.Logf("cpu(%s)=%q conf=%.4f (%.0fms) gpu(%s)=%q conf=%.4f (%.0fms) want=%q",
				cpuResult.Engine, cpuResult.Value, cpuResult.Confidence, cpuResult.PrefillTimeMs,
				gpuResult.Engine, gpuResult.Value, gpuResult.Confidence, gpuResult.PrefillTimeMs, tc.want)
			if gpuResult.Engine != "gpu" {
				t.Errorf("gpu result engine = %q, want gpu", gpuResult.Engine)
			}
			if !tc.assert {
				return
			}
			if gpuResult.Value != tc.want {
				t.Errorf("gpu decision: got %q, want %q", gpuResult.Value, tc.want)
			}
			if cpuResult.Value != tc.want {
				t.Errorf("cpu decision: got %q, want %q", cpuResult.Value, tc.want)
			}
		})
	}
}

// gpuEvalExample mirrors the corpus JSONL rows.
type gpuEvalExample struct {
	ID           string            `json:"id"`
	Family       string            `json:"family"`
	State        string            `json:"state"`
	Instructions string            `json:"instructions"`
	Criteria     map[string]string `json:"criteria"`
	Target       string            `json:"target"`
	Codes        map[string]string `json:"codes,omitempty"`
	SplitRole    string            `json:"split_role,omitempty"`
}

func gpuEvalQuestion(ex gpuEvalExample) (llm.SystemOneQuestion, error) {
	keys := make([]string, 0, len(ex.Criteria))
	for key := range ex.Criteria {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	choices := make([]llm.Choice, len(keys))
	for i, key := range keys {
		choice := llm.Choice{Value: key, Label: ex.Criteria[key]}
		if letter, ok := ex.Codes[key]; ok {
			if len(letter) != 1 || letter[0] < 'A' || letter[0] > 'Z' {
				return llm.SystemOneQuestion{}, fmt.Errorf("row %s: invalid code letter %q", ex.ID, letter)
			}
			choice.Code = fmt.Sprintf(" %c", letter[0])
		}
		choices[i] = choice
	}
	return llm.SystemOneQuestion{ID: ex.ID, Prompt: ex.Instructions, Choices: choices}, nil
}

// TestSystemOneGPUCorpusAccuracyE2E is the production accuracy gate: evaluate
// the full held-out GP test corpus on the GPU engine with the GPU-native
// readout and require accuracy consistent with the offline eval (measured
// 78.2% overall, 81.8% zero-shot). Gates sit well below those numbers to
// absorb GPU nondeterminism on borderline rows while still catching real
// regressions — a device-mismatched readout scores ~35-40%.
func TestSystemOneGPUCorpusAccuracyE2E(t *testing.T) {
	_, gpuPipe, _, gpuReadout := newSystemOneTestPipelines(t)

	file, err := os.Open(gpuTestCorpusPath)
	if err != nil {
		t.Skipf("GP test corpus not available: %v", err)
	}
	defer file.Close()
	var examples []gpuEvalExample
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024), 4*1024*1024)
	for scanner.Scan() {
		var ex gpuEvalExample
		if err := json.Unmarshal(scanner.Bytes(), &ex); err != nil {
			t.Fatalf("corpus line: %v", err)
		}
		examples = append(examples, ex)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if len(examples) == 0 {
		t.Skip("empty corpus")
	}

	// A family is "seen" when any of its rows trained; zero-shot families are
	// entirely held out (every row carries SplitRole == "zeroshot").
	seenFamilies := map[string]bool{}
	for _, ex := range examples {
		if ex.SplitRole != "zeroshot" {
			seenFamilies[ex.Family] = true
		}
	}
	byFamily := map[string]*struct{ rows, correct int }{}
	total, correct := 0, 0
	var totalMs float64
	for _, ex := range examples {
		q, err := gpuEvalQuestion(ex)
		if err != nil {
			t.Fatalf("row %s: %v", ex.ID, err)
		}
		r, err := gpuPipe.SystemOneWithReadout(ex.State, q, llm.SystemOneConfig{}, gpuReadout)
		if err != nil {
			t.Fatalf("row %s: %v", ex.ID, err)
		}
		totalMs += r.PrefillTimeMs
		ok := r.Value == ex.Target
		total++
		if ok {
			correct++
		}
		fm := byFamily[ex.Family]
		if fm == nil {
			fm = &struct{ rows, correct int }{}
			byFamily[ex.Family] = fm
		}
		fm.rows++
		if ok {
			fm.correct++
		}
	}
	names := make([]string, 0, len(byFamily))
	for name := range byFamily {
		names = append(names, name)
	}
	sort.Strings(names)
	zsTotal, zsCorrect := 0, 0
	for _, name := range names {
		fm := byFamily[name]
		if seenFamilies[name] {
			continue
		}
		zsTotal += fm.rows
		zsCorrect += fm.correct
		t.Logf("ZERO-SHOT family %-18s accuracy=%.1f%% rows=%d", name, 100*float64(fm.correct)/float64(fm.rows), fm.rows)
	}
	overall := float64(correct) / float64(total)
	zsAcc := 0.0
	if zsTotal > 0 {
		zsAcc = float64(zsCorrect) / float64(zsTotal)
	}
	t.Logf("GPU corpus accuracy=%.2f%% (%d/%d) zero-shot=%.2f%% (%d/%d) avg=%.0fms/question",
		100*overall, correct, total, 100*zsAcc, zsCorrect, zsTotal, totalMs/float64(total))
	if overall < 0.72 {
		t.Errorf("GPU corpus accuracy %.2f%% below 72%% gate", 100*overall)
	}
	if zsTotal > 0 && zsAcc < 0.76 {
		t.Errorf("GPU zero-shot accuracy %.2f%% below 76%% gate", 100*zsAcc)
	}
}

// TestSystemOneGPUMultiE2E verifies the GPU multi-question path returns one
// result per question in caller order with per-result engine attribution and
// graceful degradation for wide questions.
func TestSystemOneGPUMultiE2E(t *testing.T) {
	_, gpuPipe, _, gpuReadout := newSystemOneTestPipelines(t)

	state := "I was charged twice for my subscription this month, the app crashes when I open the invoice page, and I cannot log in from my new phone."
	questions := []llm.SystemOneQuestion{
		{ID: "route", Prompt: "Which team should handle the billing aspect of this message?", Choices: []llm.Choice{
			{Value: "billing", Label: "Billing, charges, refunds, invoices, subscriptions, or payments."},
			{Value: "technical", Label: "Technical software, device, service, or product malfunction."},
			{Value: "account", Label: "Account access, identity, password, login, or authentication issue."},
		}},
		llm.BuildSystemOneNoulQuestion("Does this message describe a billing problem?"),
		{ID: "severity", Prompt: "How severe is the technical problem described?", Choices: []llm.Choice{
			{Value: "critical", Label: "Complete outage, data loss, or security breach."},
			{Value: "major", Label: "Major functionality broken with a workaround possible."},
			{Value: "minor", Label: "Small defect with limited impact."},
			{Value: "cosmetic", Label: "Visual or cosmetic issue only."},
		}},
		// Wider than the readout: must degrade to CPU continuation scoring.
		{ID: "rating", Prompt: "Rate the overall frustration level from 0 to 9.", Choices: func() []llm.Choice {
			c := make([]llm.Choice, 0, 10)
			for v := 0; v <= 9; v++ {
				c = append(c, llm.Choice{Value: fmt.Sprintf("%d", v)})
			}
			return c
		}()},
	}

	gpuMulti, err := gpuPipe.SystemOneMulti(state, questions, llm.SystemOneConfig{}, gpuReadout)
	if err != nil {
		t.Fatalf("gpu SystemOneMulti: %v", err)
	}
	if len(gpuMulti) != len(questions) {
		t.Fatalf("gpu results = %d, want %d", len(gpuMulti), len(questions))
	}
	var gpuTotalMs float64
	var sharedMs float64
	for i, q := range questions {
		if gpuMulti[i].QuestionID != q.ID {
			t.Errorf("question %d order mismatch: gpu %q want %q", i, gpuMulti[i].QuestionID, q.ID)
		}
		wantEngine := "gpu"
		wantDegraded := false
		if q.ID == "rating" {
			wantEngine = "cpu"
			wantDegraded = true
		}
		if gpuMulti[i].Engine != wantEngine {
			t.Errorf("question %d (%s): engine %q, want %q", i, q.ID, gpuMulti[i].Engine, wantEngine)
		}
		if gpuMulti[i].Degraded != wantDegraded {
			t.Errorf("question %d (%s): degraded %v, want %v", i, q.ID, gpuMulti[i].Degraded, wantDegraded)
		}
		if gpuMulti[i].SharedPrefillTimeMs <= 0 {
			t.Errorf("question %d (%s): SharedPrefillTimeMs = %v, want > 0", i, q.ID, gpuMulti[i].SharedPrefillTimeMs)
		}
		sharedMs = gpuMulti[i].SharedPrefillTimeMs

		// Shared-prefill parity: a GPU-served question must produce the same
		// decision and probabilities as a standalone full-prefill evaluation.
		// The suffix pass continues from a restored copy of the shared prefix
		// state, which must be numerically identical to processing prefix +
		// suffix in one call (per-token batch kernels and the causal attention
		// bound seq_len+pos_id make the two paths mathematically equivalent).
		if !wantDegraded {
			standalone, err := gpuPipe.SystemOneWithReadout(state, q, llm.SystemOneConfig{}, gpuReadout)
			if err != nil {
				t.Fatalf("standalone SystemOneWithReadout(%s): %v", q.ID, err)
			}
			if standalone.Value != gpuMulti[i].Value {
				t.Errorf("question %s parity: multi value %q != standalone %q", q.ID, gpuMulti[i].Value, standalone.Value)
			}
			if diff := math.Abs(standalone.Confidence - gpuMulti[i].Confidence); diff > 1e-6 {
				t.Errorf("question %s parity: multi confidence %.8f != standalone %.8f (diff %.2e)", q.ID, gpuMulti[i].Confidence, standalone.Confidence, diff)
			}
			t.Logf("%s parity ok: value=%q conf=%.4f (multi %.0fms suffix vs standalone %.0fms full prefill)",
				q.ID, gpuMulti[i].Value, gpuMulti[i].Confidence, gpuMulti[i].PrefillTimeMs, standalone.PrefillTimeMs)
		}
		t.Logf("%s engine=%s value=%q conf=%.4f (%.0fms) degraded=%v",
			q.ID, gpuMulti[i].Engine, gpuMulti[i].Value, gpuMulti[i].Confidence, gpuMulti[i].PrefillTimeMs, gpuMulti[i].Degraded)
		gpuTotalMs += gpuMulti[i].PrefillTimeMs
	}
	t.Logf("multi total: %.0fms for %d questions (1 degraded to CPU), shared prefix %.0fms", gpuTotalMs, len(questions), sharedMs)
}