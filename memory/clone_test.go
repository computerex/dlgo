package memory

import "testing"

func TestMultiLayerKVCacheCloneIsIndependent(t *testing.T) {
	cache := NewMultiLayerKVCache(2, 4, 2)
	cache.Layers[0].Store(0, []float32{1, 2}, []float32{3, 4})
	cache.Layers[1].Store(0, []float32{5, 6}, []float32{7, 8})

	clone := cache.Clone(1)
	clone.Layers[0].KeyData[0] = 99
	clone.Layers[1].ValData[1] = 88

	if cache.Layers[0].KeyData[0] != 1 {
		t.Fatalf("source key mutated: %v", cache.Layers[0].KeyData[0])
	}
	if cache.Layers[1].ValData[1] != 8 {
		t.Fatalf("source value mutated: %v", cache.Layers[1].ValData[1])
	}
}

func TestSSMStateCloneIsIndependent(t *testing.T) {
	state := NewSSMStateCache(2, 1, 1, 1, 1, 1, 2, func(index int) bool { return index == 1 })
	state.Layers[1].State[0] = 1
	state.Layers[1].State64 = []float64{2}
	state.Layers[1].ConvBuf[0] = 3

	clone := state.Clone()
	clone.Layers[1].State[0] = 10
	clone.Layers[1].State64[0] = 20
	clone.Layers[1].ConvBuf[0] = 30

	if state.Layers[1].State[0] != 1 || state.Layers[1].State64[0] != 2 || state.Layers[1].ConvBuf[0] != 3 {
		t.Fatalf("source state mutated: %#v", state.Layers[1])
	}
}
