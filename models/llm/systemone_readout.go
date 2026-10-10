package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/computerex/dlgo/format/gguf"
)

// SystemOneReadout is an independently trained linear classifier over final
// hidden states from a frozen GGUF backbone. It intentionally contains no
// language-model weights and can be trained without modifying Q4 blocks.
type SystemOneReadout struct {
	FormatVersion int         `json:"format_version"`
	Architecture  string      `json:"architecture"`
	ModelPath     string      `json:"model_path"`
	HiddenSize    int         `json:"hidden_size"`
	MaxOptions    int         `json:"max_options"`
	PromptVersion string      `json:"prompt_version"`
	Temperature   float64     `json:"temperature"`
	// Device records which engine's feature distribution the readout was
	// trained on: "cpu", "gpu", or "any". GPU and CPU extract structurally
	// different hidden states (GPU matvec kernels quantize activations), so a
	// single-device readout must only be served by the engine it was trained
	// for. "any" artifacts are trained on the union of both devices' features
	// and are accepted by either engine. Empty means "cpu" for backward
	// compatibility with early artifacts.
	Device      string      `json:"device,omitempty"`
	Weights     [][]float32 `json:"weights"`
	CreatedAt   string      `json:"created_at"`
}

// NormalizeDevice resolves an artifact's training device, treating the empty
// value as "cpu" for artifacts written before the field existed.
func (r *SystemOneReadout) NormalizeDevice() string {
	if r.Device == "" {
		return "cpu"
	}
	return r.Device
}

// validReadoutDevices reports whether the device string is one the loaders
// accept.
func validReadoutDevices(device string) bool {
	return device == "cpu" || device == "gpu" || device == "any"
}

// LoadSystemOneReadout reads and validates a head-only System One artifact.
func LoadSystemOneReadout(path string, expectedHiddenSize int) (*SystemOneReadout, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var readout SystemOneReadout
	if err := json.Unmarshal(data, &readout); err != nil {
		return nil, fmt.Errorf("parse System One readout: %w", err)
	}
	if readout.FormatVersion != 1 || readout.Architecture != "dlgo.systemone.linear-readout" {
		return nil, fmt.Errorf("unsupported System One readout format %d / %q", readout.FormatVersion, readout.Architecture)
	}
	if readout.Device == "" {
		readout.Device = "cpu"
	}
	if !validReadoutDevices(readout.Device) {
		return nil, fmt.Errorf("System One readout device %q is not cpu, gpu, or any", readout.Device)
	}
	if readout.HiddenSize != expectedHiddenSize || readout.HiddenSize <= 0 {
		return nil, fmt.Errorf("System One readout hidden size %d does not match model dimension %d", readout.HiddenSize, expectedHiddenSize)
	}
	if readout.MaxOptions <= 0 || len(readout.Weights) != readout.MaxOptions {
		return nil, fmt.Errorf("System One readout has invalid option count")
	}
	if math.IsNaN(readout.Temperature) || math.IsInf(readout.Temperature, 0) || readout.Temperature <= 0 {
		return nil, fmt.Errorf("System One readout temperature must be finite and positive")
	}
	for i, row := range readout.Weights {
		if len(row) != readout.HiddenSize {
			return nil, fmt.Errorf("System One readout row %d has %d values, want %d", i, len(row), readout.HiddenSize)
		}
	}
	return &readout, nil
}

// LoadSystemOneReadoutGGUF loads a System One readout from a standalone GGUF
// artifact produced by WriteGGUF (see format/gguf/writer.go). The artifact must
// contain the "systemone.readout.weight" F32 tensor [max_options, hidden_size]
// plus systemone.* metadata keys.
func LoadSystemOneReadoutGGUF(path string, expectedHiddenSize int) (*SystemOneReadout, error) {
	gf, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open readout GGUF: %w", err)
	}
	arch, _ := gf.Metadata["general.architecture"].(string)
	if arch != "dlgo.systemone.linear-readout" {
		return nil, fmt.Errorf("readout GGUF architecture %q is not dlgo.systemone.linear-readout", arch)
	}
	device, _ := gf.Metadata["systemone.device"].(string)
	if device == "" {
		device = "cpu"
	}
	if !validReadoutDevices(device) {
		return nil, fmt.Errorf("readout GGUF device %q is not cpu, gpu, or any", device)
	}
	hiddenSize, _ := gf.Metadata["systemone.hidden_size"].(uint32)
	maxOptions, _ := gf.Metadata["systemone.max_options"].(uint32)
	temperature, _ := gf.Metadata["systemone.temperature"].(float64)
	promptVersion, _ := gf.Metadata["systemone.prompt_version"].(string)
	if int(hiddenSize) != expectedHiddenSize || hiddenSize == 0 {
		return nil, fmt.Errorf("readout hidden size %d does not match model dimension %d", hiddenSize, expectedHiddenSize)
	}
	if maxOptions == 0 {
		return nil, fmt.Errorf("readout has invalid option count %d", maxOptions)
	}
	if math.IsNaN(temperature) || math.IsInf(temperature, 0) || temperature <= 0 {
		return nil, fmt.Errorf("readout temperature must be finite and positive, got %v", temperature)
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := gguf.ReadTensorData(f, gf, "systemone.readout.weight")
	if err != nil {
		return nil, err
	}
	flat, err := gguf.Float32Tensor(data)
	if err != nil {
		return nil, err
	}
	if len(flat) != int(maxOptions)*int(hiddenSize) {
		return nil, fmt.Errorf("readout tensor has %d values, want %d", len(flat), int(maxOptions)*int(hiddenSize))
	}
	weights := make([][]float32, maxOptions)
	for c := range weights {
		weights[c] = flat[c*int(hiddenSize) : (c+1)*int(hiddenSize)]
	}
	return &SystemOneReadout{
		FormatVersion: 1,
		Architecture:  "dlgo.systemone.linear-readout",
		ModelPath:     path,
		HiddenSize:    int(hiddenSize),
		MaxOptions:    int(maxOptions),
		PromptVersion: promptVersion,
		Temperature:   temperature,
		Device:        device,
		Weights:       weights,
	}, nil
}

// SystemOneWithReadout evaluates choices through the native trained readout.
// It requires answer codes generated by the System One prompt compiler. Choice
// output remains in caller order, while readout rows are addressed by code.
func (p *Pipeline) SystemOneWithReadout(state string, q SystemOneQuestion, cfg SystemOneConfig, readout *SystemOneReadout) (*SystemOneResult, error) {
	if readout == nil {
		return nil, fmt.Errorf("System One readout is nil")
	}
	if readout.NormalizeDevice() == "gpu" {
		return nil, fmt.Errorf("System One readout was trained on %s features; use the %s engine or a cpu/any-device readout", readout.Device, readout.Device)
	}
	start := time.Now()
	hidden, choices, promptTokens, err := p.SystemOneHidden(state, q, cfg)
	if err != nil {
		return nil, err
	}
	prefillMs := float64(time.Since(start).Microseconds()) / 1000.0
	return ApplySystemOneReadout(q, choices, hidden, readout, cfg, promptTokens, prefillMs)
}

// ApplySystemOneReadout scores choices against an already-extracted final
// hidden state (for example produced by a GPU pipeline) using the trained
// linear readout. It applies exactly the same normalization, code addressing,
// temperature scaling, softmax, and minimum-confidence abstention rules as
// SystemOneWithReadout, so results are comparable across engines. The hidden
// slice is L2-normalized in place and must have exactly readout.HiddenSize
// entries.
func ApplySystemOneReadout(q SystemOneQuestion, choices []Choice, hidden []float32, readout *SystemOneReadout, cfg SystemOneConfig, promptTokens int, prefillMs float64) (*SystemOneResult, error) {
	if readout == nil {
		return nil, fmt.Errorf("System One readout is nil")
	}
	if len(hidden) != readout.HiddenSize {
		return nil, fmt.Errorf("hidden state has %d entries, readout expects %d", len(hidden), readout.HiddenSize)
	}
	normalizeSystemOneFeature(hidden)
	if len(choices) > readout.MaxOptions {
		return nil, fmt.Errorf("question has %d choices but readout supports %d", len(choices), readout.MaxOptions)
	}

	scoreStart := time.Now()
	scores := make([]ChoiceScore, len(choices))
	for i, choice := range choices {
		class, err := systemOneCodeIndex(choice.Code)
		if err != nil || class >= readout.MaxOptions {
			return nil, fmt.Errorf("choice %q: %w", choice.Value, err)
		}
		var value float64
		for j, feature := range hidden {
			value += float64(readout.Weights[class][j]) * float64(feature)
		}
		scores[i] = ChoiceScore{Value: choice.Value, Label: choice.Label, Code: choice.Code, LogProb: value / readout.Temperature}
	}
	normalizeChoiceScores(scores)
	best := 0
	for i := 1; i < len(scores); i++ {
		if scores[i].Probability > scores[best].Probability {
			best = i
		}
	}
	scoreMs := float64(time.Since(scoreStart).Microseconds()) / 1000.0
	result := &SystemOneResult{QuestionID: q.ID, Value: scores[best].Value, Confidence: scores[best].Probability, Choices: scores, PromptTokens: promptTokens, PrefillTimeMs: prefillMs, ScoreTimeMs: scoreMs}
	if cfg.MinConfidence > 0 && result.Confidence < cfg.MinConfidence {
		result.Abstained = true
		result.Value = ""
	}
	return result, nil
}

func normalizeSystemOneFeature(feature []float32) {
	var squared float64
	for _, value := range feature {
		squared += float64(value) * float64(value)
	}
	norm := math.Sqrt(squared)
	if norm > 0 {
		for i := range feature {
			feature[i] /= float32(norm)
		}
	}
}

func systemOneCodeIndex(code string) (int, error) {
	if len(code) == 2 && code[0] == ' ' && code[1] >= 'A' && code[1] <= 'Z' {
		return int(code[1] - 'A'), nil
	}
	return 0, fmt.Errorf("unsupported System One answer code %q", code)
}
