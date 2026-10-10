package memory

// Clone creates a deep copy of all layer KV caches. Only positions before
// currentPos are copied because later cache entries cannot affect decoding.
func (m *MultiLayerKVCache) Clone(currentPos int) *MultiLayerKVCache {
	if m == nil {
		return nil
	}
	clone := &MultiLayerKVCache{Layers: make([]*KVCache, len(m.Layers))}
	for i, layer := range m.Layers {
		if layer != nil {
			clone.Layers[i] = layer.Clone(currentPos)
		}
	}
	return clone
}

// Clone creates a deep copy of a recurrent SSM layer state.
func (s *SSMLayerState) Clone() *SSMLayerState {
	if s == nil {
		return nil
	}
	clone := &SSMLayerState{
		State:       make([]float32, len(s.State)),
		State64:     make([]float64, len(s.State64)),
		ConvBuf:     make([]float32, len(s.ConvBuf)),
		NumHeads:    s.NumHeads,
		NumKVGroups: s.NumKVGroups,
		HeadKDim:    s.HeadKDim,
		HeadVDim:    s.HeadVDim,
		Channels:    s.Channels,
		ConvK:       s.ConvK,
	}
	copy(clone.State, s.State)
	copy(clone.State64, s.State64)
	copy(clone.ConvBuf, s.ConvBuf)
	return clone
}

// Clone creates a deep copy of all recurrent SSM layer states.
func (c *SSMStateCache) Clone() *SSMStateCache {
	if c == nil {
		return nil
	}
	clone := &SSMStateCache{Layers: make([]*SSMLayerState, len(c.Layers))}
	for i, layer := range c.Layers {
		clone.Layers[i] = layer.Clone()
	}
	return clone
}
