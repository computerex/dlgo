package llm

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/computerex/dlgo/format/gguf"
)

func mtpMetadata(arch string, blocks uint32) map[string]interface{} {
	return map[string]interface{}{
		"general.architecture":         arch,
		arch + ".block_count":          blocks,
		arch + ".embedding_length":     uint32(4),
		arch + ".attention.head_count": uint32(1),
	}
}

func TestParseConfigMTPLayers(t *testing.T) {
	for _, arch := range []string{"qwen35", "qwen35moe", "qwen3next", "deepseek2"} {
		for _, tc := range []struct {
			name  string
			value interface{}
			want  int
			bad   bool
		}{
			{"absent", nil, 25, false},
			{"zero", uint32(0), 25, false},
			{"one", uint32(1), 24, false},
			{"multiple", uint64(2), 23, false},
			{"negative", int32(-1), 0, true},
			{"all_blocks", uint32(25), 0, true},
			{"too_many", uint32(26), 0, true},
			{"wrong_type", "1", 0, true},
		} {
			t.Run(arch+"/"+tc.name, func(t *testing.T) {
				md := mtpMetadata(arch, 25)
				// Other architectures' metadata must not change this model.
				md["other.nextn_predict_layers"] = uint32(3)
				if tc.value != nil {
					md[arch+".nextn_predict_layers"] = tc.value
				}
				cfg, err := ParseConfig(md)
				if tc.bad {
					if err == nil {
						t.Fatal("expected invalid MTP count error")
					}
					return
				}
				if err != nil || cfg.NumLayers != tc.want {
					t.Fatalf("NumLayers = %d, error = %v; want %d", cfg.NumLayers, err, tc.want)
				}
				if md[arch+".block_count"] != uint32(25) {
					t.Fatal("ParseConfig mutated GGUF metadata")
				}
			})
		}
	}
}

func TestLoadModelExcludesMTP(t *testing.T) {
	for _, nextN := range []uint32{0, 1, 2} {
		t.Run(fmt.Sprint(nextN), func(t *testing.T) {
			md := mtpMetadata("qwen35", 3)
			md["qwen35.nextn_predict_layers"] = nextN
			tensors := []gguf.WriteTensor{
				{Name: "token_embd.weight", Dimensions: []int64{4, 1}, Type: gguf.GGMLTypeF32, Data: gguf.Float32Bytes([]float32{1, 2, 3, 4})},
				{Name: "output_norm.weight", Dimensions: []int64{4}, Type: gguf.GGMLTypeF32, Data: gguf.Float32Bytes([]float32{1, 1, 1, 1})},
			}
			for i := 0; i < 3; i++ {
				for _, field := range []string{"attn_q.weight", "attn_norm.weight", "nextn.hnorm.weight"} {
					tensors = append(tensors, gguf.WriteTensor{
						Name: fmt.Sprintf("blk.%d.%s", i, field), Dimensions: []int64{4},
						Type: gguf.GGMLTypeF32, Data: gguf.Float32Bytes([]float32{float32(i + 1), 1, 1, 1}),
					})
				}
			}
			path := filepath.Join(t.TempDir(), "mtp.gguf")
			if err := gguf.WriteGGUF(path, md, tensors, 32); err != nil {
				t.Fatal(err)
			}
			m, err := LoadModel(path)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { m.MmapFile.Close() })
			want := 3 - int(nextN)
			if m.Config.NumLayers != want || len(m.Layers) != want {
				t.Fatalf("config/layers = %d/%d, want %d", m.Config.NumLayers, len(m.Layers), want)
			}
			for i, l := range m.Layers {
				if l.Wq == nil || len(l.AttnNorm) != 4 || l.AttnNorm[0] != float32(i+1) {
					t.Fatalf("ordinary layer %d weights not preserved", i)
				}
			}
			if m.Output != m.TokenEmbed || len(m.OutputNorm) != 4 {
				t.Fatal("global output weights/weight tying not preserved")
			}
		})
	}
}

// Opt-in real-file check avoids requiring a large local model in CI.
func TestQwen2BMTPFile(t *testing.T) {
	path := os.Getenv("DLGO_TEST_MTP_GGUF")
	if path == "" {
		t.Skip("set DLGO_TEST_MTP_GGUF to the Qwen 2B GGUF")
	}
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	arch := metaString(gf.Metadata, "general.architecture")
	blocks := metaInt(gf.Metadata, arch+".block_count", 0)
	nextN := metaInt(gf.Metadata, arch+".nextn_predict_layers", 0)
	t.Logf("architecture=%s block_count=%d nextn_predict_layers=%d", arch, blocks, nextN)
	if blocks != 25 || nextN != 1 {
		t.Fatalf("expected 25 total blocks and 1 MTP block")
	}
	m, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.MmapFile.Close()
	if m.Config.NumLayers != 24 || len(m.Layers) != 24 {
		t.Fatalf("loaded config/layers = %d/%d, want 24", m.Config.NumLayers, len(m.Layers))
	}
	if m.Layers[23].Wq == nil || m.Layers[23].FFNDown == nil || m.Output == nil {
		t.Fatal("last ordinary layer or output weights missing")
	}
}
