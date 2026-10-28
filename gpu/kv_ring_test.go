//go:build cgo && vulkan

package gpu

import (
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// ringTestModel builds a model shaped like gpt-oss-20b: a MoE dense-attention
// model whose layers alternate between a 128-wide band and full attention.
func ringTestModel(swaLayers, fullLayers int) *llm.Model {
	m := &llm.Model{}
	m.Config.ExpertCount = 32 // makes isHybridSSMModel true -> chunked prefill
	for i := 0; i < swaLayers; i++ {
		l := llm.Layer{}
		l.Spec.Core = llm.CoreAttention
		l.Spec.SlidingWindow = 128
		m.Layers = append(m.Layers, l)
	}
	for i := 0; i < fullLayers; i++ {
		l := llm.Layer{}
		l.Spec.Core = llm.CoreAttention
		m.Layers = append(m.Layers, l)
	}
	return m
}

func TestKVRingCapFor(t *testing.T) {
	t.Setenv("DLGO_KV_RING", "")
	t.Setenv("DLGO_F32_KV", "")
	t.Setenv("DLGO_PREFILL_CHUNK", "")

	// 12 banded + 12 full layers, 128 window: 1024 chunk + 128 window -> 2048.
	if got := kvRingCapFor(ringTestModel(12, 12), true); got != 2048 {
		t.Fatalf("kvRingCapFor = %d, want 2048", got)
	}

	// A larger prefill chunk must grow the ring, or a chunk could wrap onto the
	// window it still has to read.
	t.Setenv("DLGO_PREFILL_CHUNK", "4096")
	if got := kvRingCapFor(ringTestModel(12, 12), true); got != 8192 {
		t.Fatalf("with chunk 4096: kvRingCapFor = %d, want 8192", got)
	}
	t.Setenv("DLGO_PREFILL_CHUNK", "")

	// Full-attention-only model: nothing to trim.
	if got := kvRingCapFor(ringTestModel(0, 24), true); got != 0 {
		t.Fatalf("no window: kvRingCapFor = %d, want 0", got)
	}

	// Not fully resident on the GPU: the partial path addresses KV linearly.
	if got := kvRingCapFor(ringTestModel(12, 12), false); got != 0 {
		t.Fatalf("partial GPU: kvRingCapFor = %d, want 0", got)
	}

	// F32 KV debug cache is not ring-aware.
	t.Setenv("DLGO_F32_KV", "1")
	if got := kvRingCapFor(ringTestModel(12, 12), true); got != 0 {
		t.Fatalf("F32 KV: kvRingCapFor = %d, want 0", got)
	}
	t.Setenv("DLGO_F32_KV", "")

	// Off switch.
	t.Setenv("DLGO_KV_RING", "0")
	if got := kvRingCapFor(ringTestModel(12, 12), true); got != 0 {
		t.Fatalf("DLGO_KV_RING=0: kvRingCapFor = %d, want 0", got)
	}
}

func TestKVRingCapRefusesHybridCores(t *testing.T) {
	t.Setenv("DLGO_KV_RING", "")
	t.Setenv("DLGO_F32_KV", "")
	t.Setenv("DLGO_PREFILL_CHUNK", "")

	// A gated-Q layer keeps its own attention path in Go, which is linear.
	m := ringTestModel(12, 12)
	m.Layers[3].Spec.GatedQ = true
	if got := kvRingCapFor(m, true); got != 0 {
		t.Fatalf("gated-Q model: kvRingCapFor = %d, want 0", got)
	}

	// An MLA or SSM layer would need the ring on a kernel that has no ring.
	m = ringTestModel(12, 12)
	m.Layers[5].Spec.Core = llm.CoreMLA
	if got := kvRingCapFor(m, true); got != 0 {
		t.Fatalf("MLA model: kvRingCapFor = %d, want 0", got)
	}
}

func TestGpuKVCacheRingCap(t *testing.T) {
	c := &GpuKVCache{MaxSeq: 1024, Cap: []int{1024, 2048, 0, 512}}
	if got := c.RingCap(0); got != 0 {
		t.Fatalf("Cap=MaxSeq: RingCap = %d, want 0", got)
	}
	if got := c.RingCap(1); got != 0 {
		t.Fatalf("Cap>MaxSeq: RingCap = %d, want 0", got)
	}
	if got := c.RingCap(2); got != 0 {
		t.Fatalf("Cap=0: RingCap = %d, want 0", got)
	}
	if got := c.RingCap(3); got != 512 {
		t.Fatalf("RingCap(3) = %d, want 512", got)
	}
	if got := c.RingLayers(); got != 1 {
		t.Fatalf("RingLayers = %d, want 1", got)
	}
	if got := c.RingCap(99); got != 0 {
		t.Fatalf("out of range: RingCap = %d, want 0", got)
	}
}

// TestGpuKVCacheCapsAllocation checks the allocator honours per-layer row counts
// and that a capped layer really reserves fewer bytes than a full one.
func TestGpuKVCacheCapsAllocation(t *testing.T) {
	const (
		layers = 2
		kvDim  = 64
		maxSeq = 1024
		ring   = 256
	)
	var sizes []uint64
	withFakeAlloc(t, func(size uint64) (Buf, error) {
		sizes = append(sizes, size)
		return Buf(len(sizes) * 0x1000), nil
	})

	kv, err := NewGpuKVCacheCaps(layers, layers, maxSeq, kvDim, nil, []int{0, ring})
	if err != nil {
		t.Fatalf("NewGpuKVCacheCaps: %v", err)
	}

	if kv.Cap[0] != maxSeq {
		t.Fatalf("layer 0 Cap = %d, want %d", kv.Cap[0], maxSeq)
	}
	if kv.Cap[1] != ring {
		t.Fatalf("layer 1 Cap = %d, want %d", kv.Cap[1], ring)
	}
	if got := kv.RingCap(0); got != 0 {
		t.Fatalf("RingCap(0) = %d, want 0", got)
	}
	if got := kv.RingCap(1); got != ring {
		t.Fatalf("RingCap(1) = %d, want %d", got, ring)
	}

	// Key then value per layer: full layer first, then the ring layer.
	wantSizes := []uint64{
		uint64(maxSeq * kvDim * kv.BytesPerElem),
		uint64(maxSeq * kvDim * kv.BytesPerElem),
		uint64(ring * kvDim * kv.BytesPerElem),
		uint64(ring * kvDim * kv.BytesPerElem),
	}
	if len(sizes) != len(wantSizes) {
		t.Fatalf("got %d allocations, want %d (%v)", len(sizes), len(wantSizes), sizes)
	}
	for i, want := range wantSizes {
		if sizes[i] != want {
			t.Fatalf("allocation %d = %d bytes, want %d (all: %v)", i, sizes[i], want, sizes)
		}
	}

	wantTotal := 2*wantSizes[0] + 2*wantSizes[2]
	if got := kv.KVBytes(); got != wantTotal {
		t.Fatalf("KVBytes = %d, want %d", got, wantTotal)
	}
	if kv.RingLayers() != 1 {
		t.Fatalf("RingLayers = %d, want 1", kv.RingLayers())
	}
}

// TestGpuKVCacheCapsClamp checks that a cap at or above maxSeqLen falls back to
// the linear layout, so no caller can accidentally ask for a bigger cache than
// the context.
func TestGpuKVCacheCapsClamp(t *testing.T) {
	const (
		layers = 1
		kvDim  = 32
		maxSeq = 4096
	)
	var sizes []uint64
	withFakeAlloc(t, func(size uint64) (Buf, error) {
		sizes = append(sizes, size)
		return Buf(len(sizes) * 0x1000), nil
	})

	kv, err := NewGpuKVCacheCaps(layers, layers, maxSeq, kvDim, nil, []int{maxSeq * 2})
	if err != nil {
		t.Fatalf("NewGpuKVCacheCaps: %v", err)
	}
	if kv.Cap[0] != maxSeq {
		t.Fatalf("Cap = %d, want %d", kv.Cap[0], maxSeq)
	}
	if got := kv.RingCap(0); got != 0 {
		t.Fatalf("RingCap = %d, want 0", got)
	}
}
