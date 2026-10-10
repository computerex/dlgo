package llm

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/computerex/dlgo/format/gguf"
)

// TestSystemOneReadoutGGUFRoundTrip verifies that a readout written as GGUF
// loads back with identical weights and metadata through
// LoadSystemOneReadoutGGUF.
func TestSystemOneReadoutGGUFRoundTrip(t *testing.T) {
	const hidden = 8
	const options = 4
	weights := make([][]float32, options)
	for c := range weights {
		weights[c] = make([]float32, hidden)
		for j := range weights[c] {
			weights[c][j] = float32(c)*0.1 + float32(j)*0.01 - 0.05
		}
	}

	path := filepath.Join(t.TempDir(), "readout.gguf")
	flat := make([]float32, 0, options*hidden)
	for _, row := range weights {
		flat = append(flat, row...)
	}
	metadata := map[string]interface{}{
		"general.architecture":     "dlgo.systemone.linear-readout",
		"general.name":             "test-readout",
		"systemone.format_version": uint32(1),
		"systemone.hidden_size":    uint32(hidden),
		"systemone.max_options":    uint32(options),
		"systemone.temperature":    0.35,
		"systemone.prompt_version": "dlgo-systemone-v1",
	}
	tensors := []gguf.WriteTensor{
		{
			Name:       "systemone.readout.weight",
			Dimensions: []int64{options, hidden},
			Type:       gguf.GGMLTypeF32,
			Data:       gguf.Float32Bytes(flat),
		},
	}
	if err := gguf.WriteGGUF(path, metadata, tensors, 32); err != nil {
		t.Fatalf("WriteGGUF: %v", err)
	}

	readout, err := LoadSystemOneReadoutGGUF(path, hidden)
	if err != nil {
		t.Fatalf("LoadSystemOneReadoutGGUF: %v", err)
	}
	if readout.HiddenSize != hidden || readout.MaxOptions != options {
		t.Fatalf("dims = %d x %d, want %d x %d", readout.MaxOptions, readout.HiddenSize, options, hidden)
	}
	if readout.Temperature != 0.35 {
		t.Fatalf("temperature = %v, want 0.35", readout.Temperature)
	}
	if readout.PromptVersion != "dlgo-systemone-v1" {
		t.Fatalf("prompt version = %q", readout.PromptVersion)
	}
	for c := range weights {
		if len(readout.Weights[c]) != hidden {
			t.Fatalf("row %d length = %d, want %d", c, len(readout.Weights[c]), hidden)
		}
		for j := range weights[c] {
			if math.Abs(float64(readout.Weights[c][j]-weights[c][j])) > 1e-9 {
				t.Fatalf("weight[%d][%d] = %v, want %v", c, j, readout.Weights[c][j], weights[c][j])
			}
		}
	}
}

// TestLoadSystemOneReadoutGGUFRejectsWrongArch verifies the architecture guard.
func TestLoadSystemOneReadoutGGUFRejectsWrongArch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.gguf")
	metadata := map[string]interface{}{
		"general.architecture": "llama",
		"systemone.hidden_size": uint32(8),
		"systemone.max_options": uint32(2),
		"systemone.temperature": 1.0,
	}
	if err := gguf.WriteGGUF(path, metadata, nil, 32); err != nil {
		t.Fatalf("WriteGGUF: %v", err)
	}
	if _, err := LoadSystemOneReadoutGGUF(path, 8); err == nil {
		t.Fatal("expected architecture rejection error")
	}
}

// TestLoadSystemOneReadoutGGUFRejectsWrongHiddenSize verifies the dimension guard.
func TestLoadSystemOneReadoutGGUFRejectsWrongHiddenSize(t *testing.T) {
	path := filepath.Join(t.TempDir(), "wrongdim.gguf")
	metadata := map[string]interface{}{
		"general.architecture":     "dlgo.systemone.linear-readout",
		"systemone.hidden_size":    uint32(8),
		"systemone.max_options":    uint32(2),
		"systemone.temperature":    1.0,
		"systemone.prompt_version": "dlgo-systemone-v1",
	}
	tensors := []gguf.WriteTensor{
		{
			Name:       "systemone.readout.weight",
			Dimensions: []int64{2, 8},
			Type:       gguf.GGMLTypeF32,
			Data:       gguf.Float32Bytes(make([]float32, 16)),
		},
	}
	if err := gguf.WriteGGUF(path, metadata, tensors, 32); err != nil {
		t.Fatalf("WriteGGUF: %v", err)
	}
	if _, err := LoadSystemOneReadoutGGUF(path, 16); err == nil {
		t.Fatal("expected hidden-size mismatch error")
	}
}