package gguf

import (
	"math"
	"path/filepath"
	"testing"
)

func TestWriteGGUFRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "readout.gguf")

	// A small readout: 3 options x 4 hidden dims, F32.
	weights := []float32{0.1, 0.2, 0.3, 0.4, -0.1, -0.2, 0.5, 0.6, 1.0, 2.0, 3.0, 4.0}
	metadata := map[string]interface{}{
		"general.architecture": "dlgo.systemone.linear-readout",
		"general.name":         "qwen35-08b-q4-readout-test",
		"systemone.hidden_size": uint32(4),
		"systemone.max_options": uint32(3),
		"systemone.temperature": float64(0.5),
		"systemone.prompt_version": "dlgo-systemone-v1",
	}
	tensors := []WriteTensor{
		{
			Name:       "systemone.readout.weight",
			Dimensions: []int64{3, 4},
			Type:       GGMLTypeF32,
			Data:       Float32Bytes(weights),
		},
	}

	if err := WriteGGUF(path, metadata, tensors, 32); err != nil {
		t.Fatalf("WriteGGUF: %v", err)
	}

	gf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if gf.Version != 3 {
		t.Errorf("Version = %d, want 3", gf.Version)
	}
	if gf.TensorCount != 1 {
		t.Errorf("TensorCount = %d, want 1", gf.TensorCount)
	}
	if gf.MetadataCount != int64(len(metadata)) {
		t.Errorf("MetadataCount = %d, want %d", gf.MetadataCount, len(metadata))
	}
	if gf.Metadata["general.architecture"] != "dlgo.systemone.linear-readout" {
		t.Errorf("architecture = %v", gf.Metadata["general.architecture"])
	}
	if gf.Metadata["systemone.hidden_size"] != uint32(4) {
		t.Errorf("hidden_size = %v, want 4", gf.Metadata["systemone.hidden_size"])
	}
	if gf.Metadata["systemone.temperature"] != float64(0.5) {
		t.Errorf("temperature = %v, want 0.5", gf.Metadata["systemone.temperature"])
	}
	if len(gf.Tensors) != 1 {
		t.Fatalf("len(Tensors) = %d, want 1", len(gf.Tensors))
	}
	tensor := gf.Tensors[0]
	if tensor.Name != "systemone.readout.weight" {
		t.Errorf("tensor name = %q", tensor.Name)
	}
	if tensor.Type != GGMLTypeF32 {
		t.Errorf("tensor type = %d, want F32", tensor.Type)
	}
	if tensor.NDims != 2 || tensor.Dimensions[0] != 3 || tensor.Dimensions[1] != 4 {
		t.Errorf("tensor dims = %v, want [3 4]", tensor.Dimensions)
	}
	if gf.DataOffset <= 0 {
		t.Errorf("DataOffset = %d, want > 0", gf.DataOffset)
	}
}

func TestFloat32Bytes(t *testing.T) {
	values := []float32{1.5, -2.25, 0.0}
	bytes := Float32Bytes(values)
	if len(bytes) != 12 {
		t.Fatalf("len = %d, want 12", len(bytes))
	}
	// Verify little-endian round trip.
	if math.Float32frombits(uint32(bytes[0])|uint32(bytes[1])<<8|uint32(bytes[2])<<16|uint32(bytes[3])<<24) != 1.5 {
		t.Errorf("first value did not round trip")
	}
}
