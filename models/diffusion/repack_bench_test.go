package diffusion

import (
	"os"
	"testing"
	"time"

	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/quant"
)

// TestDiTRepackTiming isolates the CPU cost of the Affine8 repack (dequant to
// float32 then requant) that dominates DiT load time. It runs only when
// DLGO_DIT_BENCH points at the DiT GGUF, so it is inert in normal test runs.
// No GPU is required.
func TestDiTRepackTiming(t *testing.T) {
	path := os.Getenv("DLGO_DIT_BENCH")
	if path == "" {
		t.Skip("set DLGO_DIT_BENCH=<dit.gguf> to run")
	}
	m, err := LoadDiTModel(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	var qts []*core.QuantizedTensor
	add := func(q *core.QuantizedTensor) {
		if q != nil && q.FP32Data == nil {
			qts = append(qts, q)
		}
	}
	for i := range m.ContextRefiner {
		add(m.ContextRefiner[i].AttnQKV)
		add(m.ContextRefiner[i].AttnOut)
		add(m.ContextRefiner[i].FFNGate)
		add(m.ContextRefiner[i].FFNDown)
		add(m.ContextRefiner[i].FFNUp)
	}
	for i := range m.NoiseRefiner {
		add(m.NoiseRefiner[i].AttnQKV)
		add(m.NoiseRefiner[i].AttnOut)
		add(m.NoiseRefiner[i].FFNGate)
		add(m.NoiseRefiner[i].FFNDown)
		add(m.NoiseRefiner[i].FFNUp)
	}
	for i := range m.MainLayers {
		add(m.MainLayers[i].AttnQKV)
		add(m.MainLayers[i].AttnOut)
		add(m.MainLayers[i].FFNGate)
		add(m.MainLayers[i].FFNDown)
		add(m.MainLayers[i].FFNUp)
	}
	add(m.XEmbedWeight)
	add(m.FinalLinWeight)

	var totalElems int64
	var totalSrcBytes int64
	for _, q := range qts {
		totalElems += int64(q.Rows) * int64(q.Cols)
		totalSrcBytes += int64(len(q.Data))
	}
	t.Logf("tensors=%d elems=%.2fG srcBytes=%.2fGB",
		len(qts), float64(totalElems)/1e9, float64(totalSrcBytes)/(1<<30))

	// 1. Full repack (what upload does).
	t0 := time.Now()
	var outBytes int64
	for _, q := range qts {
		enc, err := quant.RepackAffine8(q.Data, q.Type, q.Rows*q.Cols)
		if err != nil {
			t.Fatalf("repack: %v", err)
		}
		outBytes += int64(len(enc))
	}
	repackDur := time.Since(t0)
	t.Logf("REPACK (dequant+requant): %v  (%.1f MB/s src, out=%.2fGB)",
		repackDur,
		float64(totalSrcBytes)/(1<<20)/repackDur.Seconds(),
		float64(outBytes)/(1<<30))

	// 2. Dequant only (f32 expansion), single pass per tensor so peak memory
	// stays at one tensor's worth of float32 (holding all 172 at once is ~24 GB
	// and thrashes a 32 GB box).
	t0 = time.Now()
	for _, q := range qts {
		if _, err := quant.Dequantize(q.Data, q.Type, q.Rows*q.Cols); err != nil {
			t.Fatalf("dequant: %v", err)
		}
	}
	t.Logf("DEQUANT only (serial, per-tensor): %v", time.Since(t0))
}
