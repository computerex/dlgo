//go:build cgo && vulkan

package gpu

import (
	"bytes"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// These pin the primitives behind the state snapshot/restore used for
// cross-request prefix reuse. The snapshot tests that exercise the full
// pipeline used to fail intermittently with a destination that still held its
// old contents, which is indistinguishable from "the copy silently did
// nothing" -- so each layer is checked on its own here, in addition to the
// end-to-end tests in systemone_cache_test.go.

// TestCopyRegionRoundTrip covers the only primitive the snapshot and the
// restore are built from, in the recorded path they use and across a later
// upload that would otherwise mask a lost copy.
func TestCopyRegionRoundTrip(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	const size = 32
	pattern := systemOneTestBytes(size, 140)

	alloc := func(data []byte) Buf {
		b := Alloc(uint64(len(data)))
		if b == 0 {
			t.Fatal("alloc failed")
		}
		t.Cleanup(func() { freeBuf(b) })
		if err := Upload(b, data); err != nil {
			t.Fatal(err)
		}
		return b
	}

	src := alloc(pattern)
	dst := alloc(make([]byte, size))

	BeginBatch()
	if err := CopyRegion(dst, 0, src, 0, size); err != nil {
		t.Fatal(err)
	}
	EndBatch()
	Sync()
	if got := systemOneTestRead(t, dst, size); !bytes.Equal(got, pattern) {
		t.Fatalf("recorded copy: got %v, want %v", got, pattern)
	}

	// A second copy, this time over a fresh upload of something else: a lost
	// copy would leave the zeros behind rather than raise an error.
	zeros := make([]byte, size)
	if err := Upload(dst, zeros); err != nil {
		t.Fatal(err)
	}
	if got := systemOneTestRead(t, dst, size); !bytes.Equal(got, zeros) {
		t.Fatalf("upload after copy: got %v, want zeros", got)
	}
	BeginBatch()
	if err := CopyRegion(dst, 0, src, 0, size); err != nil {
		t.Fatal(err)
	}
	EndBatch()
	Sync()
	if got := systemOneTestRead(t, dst, size); !bytes.Equal(got, pattern) {
		t.Fatalf("copy over a fresh upload: got %v, want %v", got, pattern)
	}
}

// testSnapshotPipeline builds the smallest pipeline that has one attention-only
// layer and one SSM layer, with recognisable contents in every buffer.
func testSnapshotPipeline(t *testing.T, keySeed, valSeed, stateSeed, convSeed byte) *GpuPipeline {
	t.Helper()
	p := &GpuPipeline{
		KVCache: &GpuKVCache{
			KeyBufs: []Buf{systemOneTestBuffer(t, systemOneTestBytes(64, keySeed)), 0},
			ValBufs: []Buf{systemOneTestBuffer(t, systemOneTestBytes(64, valSeed)), 0},
			KVDim:   4, MaxSeq: 8, Len: 2, BytesPerElem: 2,
		},
		CPUModel: &llm.Model{Config: llm.ModelConfig{
			NumLayers: 2, SSMTimeStepRank: 2, SSMGroupCount: 1,
			SSMInnerSize: 4, SSMStateSize: 2, SSMConvKernel: 2,
		}},
		GpuModel: &GpuModel{Layers: make([]GpuLayer, 2)},
		HasSSM:   true,
	}
	p.GpuModel.Layers[1].SSMState = systemOneTestBuffer(t, systemOneTestBytes(32, stateSeed))
	p.GpuModel.Layers[1].SSMConvBuf = systemOneTestBuffer(t, systemOneTestBytes(64, convSeed))
	return p
}

// TestSnapshotCapturesState checks the capture half: the recurrent buffers must
// land in the snapshot verbatim and at the sizes the model config implies.
func TestSnapshotCapturesState(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	p := testSnapshotPipeline(t, 10, 80, 140, 180)
	s, err := p.SnapshotState(2)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Free)

	if s.kvLen != 2 || s.kvBytes != 16 {
		t.Fatalf("snapshot KV: len=%d bytes=%d, want 2/16", s.kvLen, s.kvBytes)
	}
	if s.ssmStateBytes != 32 || s.ssmConvBytes != 64 {
		t.Fatalf("snapshot SSM sizes: state=%d conv=%d, want 32/64", s.ssmStateBytes, s.ssmConvBytes)
	}
	if len(s.ssmState) != 2 {
		t.Fatalf("snapshot ssmState layers = %d, want 2", len(s.ssmState))
	}
	if s.ssmState[1] == 0 {
		t.Fatal("layer 1 SSM state was not captured at all")
	}
	if got := systemOneTestRead(t, s.ssmState[1], 32); !bytes.Equal(got, systemOneTestBytes(32, 140)) {
		t.Fatalf("captured SSM state: got %v, want seed 140", got)
	}
	if got := systemOneTestRead(t, s.ssmConv[1], 64); !bytes.Equal(got, systemOneTestBytes(64, 180)) {
		t.Fatalf("captured SSM conv: got %v, want seed 180", got)
	}
	if got := systemOneTestRead(t, s.kvKeys[0], 16); !bytes.Equal(got, systemOneTestBytes(16, 10)) {
		t.Fatalf("captured KV prefix: got %v, want seed 10 prefix", got)
	}
}

// TestRestoreStateRoundTrip checks the restore half: clobber everything the
// snapshot covers, restore, and require the original contents back.
func TestRestoreStateRoundTrip(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	keyA := systemOneTestBytes(64, 10)
	stateA := systemOneTestBytes(32, 140)
	convA := systemOneTestBytes(64, 180)

	p := testSnapshotPipeline(t, 10, 80, 140, 180)
	s, err := p.SnapshotState(2)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Free)

	if err := Upload(p.KVCache.KeyBufs[0], systemOneTestBytes(64, 40)); err != nil {
		t.Fatal(err)
	}
	if err := Upload(p.GpuModel.Layers[1].SSMState, make([]byte, 32)); err != nil {
		t.Fatal(err)
	}
	if err := Upload(p.GpuModel.Layers[1].SSMConvBuf, make([]byte, 64)); err != nil {
		t.Fatal(err)
	}

	if err := p.RestoreState(s); err != nil {
		t.Fatal(err)
	}

	if got := systemOneTestRead(t, p.KVCache.KeyBufs[0], 16); !bytes.Equal(got, keyA[:16]) {
		t.Fatalf("restored KV prefix: got %v, want %v", got, keyA[:16])
	}
	if got := systemOneTestRead(t, p.GpuModel.Layers[1].SSMState, 32); !bytes.Equal(got, stateA) {
		t.Fatalf("restored SSM state: got %v, want %v", got, stateA)
	}
	if got := systemOneTestRead(t, p.GpuModel.Layers[1].SSMConvBuf, 64); !bytes.Equal(got, convA) {
		t.Fatalf("restored SSM conv: got %v, want %v", got, convA)
	}
	if p.KVCache.Len != 2 {
		t.Fatalf("restored KVCache.Len = %d, want 2", p.KVCache.Len)
	}
}

// TestRestoreRecurrentStateSkipsKV pins the documented difference between the
// two restore entry points: the recurrent-only one must not touch KV bytes.
func TestRestoreRecurrentStateSkipsKV(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	p := testSnapshotPipeline(t, 10, 80, 140, 180)
	s, err := p.SnapshotState(2)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Free)

	suffix := systemOneTestBytes(64, 40)
	if err := Upload(p.KVCache.KeyBufs[0], suffix); err != nil {
		t.Fatal(err)
	}
	if err := Upload(p.GpuModel.Layers[1].SSMState, make([]byte, 32)); err != nil {
		t.Fatal(err)
	}

	if err := p.RestoreRecurrentState(s); err != nil {
		t.Fatal(err)
	}

	if got := systemOneTestRead(t, p.KVCache.KeyBufs[0], 64); !bytes.Equal(got, suffix) {
		t.Fatalf("recurrent-only restore modified KV: got %v, want %v", got, suffix)
	}
	if got := systemOneTestRead(t, p.GpuModel.Layers[1].SSMState, 32); !bytes.Equal(got, systemOneTestBytes(32, 140)) {
		t.Fatalf("recurrent-only restore did not restore SSM state: got %v, want seed 140", got)
	}
}
