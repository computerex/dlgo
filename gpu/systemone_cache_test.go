//go:build cgo && vulkan

package gpu

import (
	"bytes"
	"crypto/sha256"
	"testing"
	"time"

	"github.com/computerex/dlgo/models/llm"
)

// These tests use only tiny synthetic buffers, never model weights or kernels.
// Do not run in parallel: Vulkan batching and device lifetime are global.
func initSystemOneSnapshotTest(t *testing.T) {
	t.Helper()
	if err := Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	t.Cleanup(Shutdown)
}

func systemOneTestBytes(n int, seed byte) []byte {
	out := make([]byte, n)
	for i := range out {
		out[i] = seed + byte(i)
	}
	return out
}

func systemOneTestUpload(t *testing.T, b Buf, data []byte) {
	t.Helper()
	if err := Upload(b, data); err != nil {
		t.Fatal(err)
	}
}

func systemOneTestBuffer(t *testing.T, data []byte) Buf {
	t.Helper()
	b := Alloc(uint64(len(data)))
	if b == 0 {
		t.Fatal("synthetic buffer allocation failed")
	}
	t.Cleanup(func() { freeBuf(b) })
	systemOneTestUpload(t, b, data)
	return b
}

func systemOneTestRead(t *testing.T, b Buf, n int) []byte {
	t.Helper()
	out := make([]byte, n)
	if err := Download(b, out); err != nil {
		t.Fatal(err)
	}
	return out
}

func systemOneTestEqual(t *testing.T, b Buf, want []byte) {
	t.Helper()
	if got := systemOneTestRead(t, b, len(want)); !bytes.Equal(got, want) {
		t.Fatalf("buffer mismatch: got %v, want %v", got, want)
	}
}

func systemOneTestPipeline(t *testing.T, elemBytes int, recurrent bool) *GpuPipeline {
	t.Helper()
	p := &GpuPipeline{
		KVCache: &GpuKVCache{
			KeyBufs: []Buf{systemOneTestBuffer(t, systemOneTestBytes(8*4*elemBytes, 10)), 0},
			ValBufs: []Buf{systemOneTestBuffer(t, systemOneTestBytes(8*4*elemBytes, 80)), 0},
			KVDim:   4, MaxSeq: 8, Len: 2, BytesPerElem: elemBytes,
		},
		CPUModel: &llm.Model{Config: llm.ModelConfig{
			NumLayers: 2, SSMTimeStepRank: 2, SSMGroupCount: 1,
			SSMInnerSize: 4, SSMStateSize: 2, SSMConvKernel: 2,
		}},
		GpuModel: &GpuModel{Layers: make([]GpuLayer, 2)},
		HasSSM:   recurrent,
	}
	if recurrent {
		// Layer 0 is attention-only; layer 1 has SSM and no KV.
		p.GpuModel.Layers[1].SSMState = systemOneTestBuffer(t, systemOneTestBytes(32, 140))
		p.GpuModel.Layers[1].SSMConvBuf = systemOneTestBuffer(t, systemOneTestBytes(64, 180))
	}
	return p
}

func systemOneTestSnapshot(t *testing.T, p *GpuPipeline) *GpuStateSnapshot {
	t.Helper()
	s, err := p.SnapshotState(2)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Free)
	return s
}

func TestSystemOneSnapshotRecurrentRestore(t *testing.T) {
	initSystemOneSnapshotTest(t)
	for _, recurrent := range []bool{false, true} {
		for _, elemBytes := range []int{2, 4} {
			p := systemOneTestPipeline(t, elemBytes, recurrent)
			s := systemOneTestSnapshot(t, p)
			prefixBytes := int(s.kvBytes)
			key := systemOneTestRead(t, p.KVCache.KeyBufs[0], 8*4*elemBytes)
			val := systemOneTestRead(t, p.KVCache.ValBufs[0], len(key))
			for branch := 0; branch < 3; branch++ {
				// Simulate suffix-only writes; the prefix stays byte-identical.
				copy(key[prefixBytes:], systemOneTestBytes(len(key)-prefixBytes, byte(30+branch)))
				copy(val[prefixBytes:], systemOneTestBytes(len(val)-prefixBytes, byte(60+branch)))
				systemOneTestUpload(t, p.KVCache.KeyBufs[0], key)
				systemOneTestUpload(t, p.KVCache.ValBufs[0], val)
				p.KVCache.Len = 5 + branch
				if recurrent {
					systemOneTestUpload(t, p.GpuModel.Layers[1].SSMState, make([]byte, 32))
					systemOneTestUpload(t, p.GpuModel.Layers[1].SSMConvBuf, make([]byte, 64))
				}
				if err := p.RestoreRecurrentState(s); err != nil {
					t.Fatal(err)
				}
				if p.KVCache.Len != 2 {
					t.Fatalf("branch %d: Len=%d, want 2", branch, p.KVCache.Len)
				}
				systemOneTestEqual(t, p.KVCache.KeyBufs[0], key)
				systemOneTestEqual(t, p.KVCache.ValBufs[0], val)
				if recurrent {
					systemOneTestEqual(t, p.GpuModel.Layers[1].SSMState, systemOneTestBytes(32, 140))
					systemOneTestEqual(t, p.GpuModel.Layers[1].SSMConvBuf, systemOneTestBytes(64, 180))
				}
			}
			// No KV snapshot buffer may be read by the recurrent-only path.
			// A recurrent-only view deliberately omits the captured KV buffers.
			view := *s
			view.kvKeys, view.kvVals = nil, nil
			if err := p.RestoreRecurrentState(&view); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestSystemOneSnapshotCachedPrefixRestore(t *testing.T) {
	initSystemOneSnapshotTest(t)
	for _, elemBytes := range []int{2, 4} {
		p := systemOneTestPipeline(t, elemBytes, true)
		t.Cleanup(p.freePrefixCache)
		a := systemOneTestSnapshot(t, p)
		keyA := sha256.Sum256([]byte("prefix A"))
		keyB := sha256.Sum256([]byte("prefix B"))
		p.storePrefixCache(keyA, 2, a)
		prefixBytes := int(a.kvBytes)
		// Install a different prefix and capture it, overwriting live A.
		keyBData := systemOneTestBytes(8*4*elemBytes, 40)
		valBData := systemOneTestBytes(len(keyBData), 100)
		systemOneTestUpload(t, p.KVCache.KeyBufs[0], keyBData)
		systemOneTestUpload(t, p.KVCache.ValBufs[0], valBData)
		systemOneTestUpload(t, p.GpuModel.Layers[1].SSMState, systemOneTestBytes(32, 20))
		systemOneTestUpload(t, p.GpuModel.Layers[1].SSMConvBuf, systemOneTestBytes(64, 50))
		b := systemOneTestSnapshot(t, p)
		p.storePrefixCache(keyB, 2, b)
		if p.lookupPrefixCache(keyA, 3) != nil || p.lookupPrefixCache(sha256.Sum256([]byte("missing")), 2) != nil {
			t.Fatal("cache matched a different prefix or length")
		}
		for _, tc := range []struct {
			key                                   [32]byte
			keySeed, valSeed, stateSeed, convSeed byte
		}{{keyA, 10, 80, 140, 180}, {keyB, 40, 100, 20, 50}, {keyA, 10, 80, 140, 180}} {
			cached := p.lookupPrefixCache(tc.key, 2)
			if cached == nil {
				t.Fatal("cached prefix missing")
			}
			p.KVCache.Len = 7
			if err := p.RestoreState(cached); err != nil {
				t.Fatal(err)
			}
			if p.KVCache.Len != 2 {
				t.Fatalf("Len=%d, want 2", p.KVCache.Len)
			}
			wantKey := append(systemOneTestBytes(prefixBytes, tc.keySeed), keyBData[prefixBytes:]...)
			wantVal := append(systemOneTestBytes(prefixBytes, tc.valSeed), valBData[prefixBytes:]...)
			systemOneTestEqual(t, p.KVCache.KeyBufs[0], wantKey)
			systemOneTestEqual(t, p.KVCache.ValBufs[0], wantVal)
			systemOneTestEqual(t, p.GpuModel.Layers[1].SSMState, systemOneTestBytes(32, tc.stateSeed))
			systemOneTestEqual(t, p.GpuModel.Layers[1].SSMConvBuf, systemOneTestBytes(64, tc.convSeed))
		}
	}
}

func TestSystemOneSnapshotCacheEviction(t *testing.T) {
	initSystemOneSnapshotTest(t)
	p := systemOneTestPipeline(t, 2, true)
	t.Cleanup(p.freePrefixCache)
	keys := make([][32]byte, prefixCacheMaxEntries+1)
	snaps := make([]*GpuStateSnapshot, len(keys))
	for i := 0; i < prefixCacheMaxEntries; i++ {
		keys[i] = sha256.Sum256([]byte{byte(i)})
		snaps[i] = systemOneTestSnapshot(t, p)
		p.storePrefixCache(keys[i], 2, snaps[i])
		// Deterministic LRU order without sleeping or depending on clock resolution.
		p.prefixCache[i].lastUsed = time.Unix(int64(i), 0)
	}
	if p.lookupPrefixCache(keys[0], 2) != snaps[0] {
		t.Fatal("cache lookup did not return original snapshot")
	}
	keys[prefixCacheMaxEntries] = sha256.Sum256([]byte("new prefix"))
	p.storePrefixCache(keys[prefixCacheMaxEntries], 2, systemOneTestSnapshot(t, p))
	if len(p.prefixCache) != prefixCacheMaxEntries || p.lookupPrefixCache(keys[1], 2) != nil {
		t.Fatal("cache did not evict least-recently-used entry")
	}
	if snaps[1].kvKeys != nil || snaps[1].ssmState != nil || snaps[1].ssmConv != nil {
		t.Fatal("evicted snapshot buffers were not released")
	}
	if p.lookupPrefixCache(keys[0], 2) != snaps[0] {
		t.Fatal("recently touched snapshot was evicted")
	}
	replacement := systemOneTestSnapshot(t, p)
	p.storePrefixCache(keys[0], 2, replacement)
	if snaps[0].kvKeys != nil || snaps[0].ssmState != nil || p.lookupPrefixCache(keys[0], 2) != replacement {
		t.Fatal("same-key replacement did not release the previous snapshot")
	}
	p.freePrefixCache()
	if p.prefixCache != nil {
		t.Fatal("cache cleanup left entries")
	}
	p.freePrefixCache() // Repeated cleanup, like snapshot Free, is harmless.
}

func TestSystemOneSnapshotRestoreInvalid(t *testing.T) {
	// Validation errors must be detected before touching Vulkan or live state.
	p := &GpuPipeline{KVCache: &GpuKVCache{MaxSeq: 8, Len: 5}}
	for _, restore := range []func(*GpuStateSnapshot) error{p.RestoreState, p.RestoreRecurrentState} {
		for _, s := range []*GpuStateSnapshot{nil, {}, {kvLen: -1}, {kvLen: 9}} {
			if err := restore(s); err == nil {
				t.Fatal("invalid snapshot accepted")
			}
			if p.KVCache.Len != 5 {
				t.Fatal("invalid restore changed Len")
			}
		}
	}
}
