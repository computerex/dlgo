package llm

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSystemOneReadoutLoadValidation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "readout.json")
	readout := SystemOneReadout{
		FormatVersion: 1, Architecture: "dlgo.systemone.linear-readout",
		HiddenSize: 4, MaxOptions: 2, PromptVersion: "dlgo-systemone-v1", Temperature: 1,
		Weights: [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}},
	}
	data, err := json.Marshal(readout)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadSystemOneReadout(path, 4)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.MaxOptions != 2 || loaded.HiddenSize != 4 {
		t.Fatalf("bad load: %+v", loaded)
	}

	// Wrong hidden size must fail.
	if _, err := LoadSystemOneReadout(path, 8); err == nil {
		t.Fatal("expected hidden-size mismatch error")
	}
}

func TestSystemOneCodeIndex(t *testing.T) {
	if idx, err := systemOneCodeIndex(" A"); err != nil || idx != 0 {
		t.Fatalf("A -> %d, %v", idx, err)
	}
	if idx, err := systemOneCodeIndex(" Z"); err != nil || idx != 25 {
		t.Fatalf("Z -> %d, %v", idx, err)
	}
	if _, err := systemOneCodeIndex("AA"); err == nil {
		t.Fatal("expected error for unsupported code")
	}
	if _, err := systemOneCodeIndex("A"); err == nil {
		t.Fatal("expected error for missing space prefix")
	}
}

func TestNormalizeSystemOneFeature(t *testing.T) {
	feature := []float32{3, 4}
	normalizeSystemOneFeature(feature)
	if math.Abs(float64(feature[0])-0.6) > 1e-6 || math.Abs(float64(feature[1])-0.8) > 1e-6 {
		t.Fatalf("normalized = %v, want [0.6 0.8]", feature)
	}
}
