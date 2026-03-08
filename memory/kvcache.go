package memory

// KVCache stores key and value tensors for one attention layer.
// Data is stored in contiguous flat buffers (KeyData/ValData) for efficient
// SIMD access. The Keys/Vals slice-of-slice view is maintained for API compat.
type KVCache struct {
	Keys    [][]float32 // [maxPos][dim] — slices into KeyData
	Vals    [][]float32 // [maxPos][dim] — slices into ValData
	KeyData []float32   // flat contiguous [maxPos * dim]
	ValData []float32   // flat contiguous [maxPos * dim]
	Len     int
	Dim     int
}

// NewKVCache creates a KV cache that can hold up to maxPos positions of dimension dim.
func NewKVCache(maxPos, dim int) *KVCache {
	c := &KVCache{
		Keys:    make([][]float32, maxPos),
		Vals:    make([][]float32, maxPos),
		KeyData: make([]float32, maxPos*dim),
		ValData: make([]float32, maxPos*dim),
		Dim:     dim,
	}
	for p := 0; p < maxPos; p++ {
		c.Keys[p] = c.KeyData[p*dim : (p+1)*dim]
		c.Vals[p] = c.ValData[p*dim : (p+1)*dim]
	}
	return c
}

// Reset clears the cache without deallocating.
func (c *KVCache) Reset() {
	c.Len = 0
}

// Store writes key and value vectors at the given position.
func (c *KVCache) Store(pos int, key, val []float32) {
	copy(c.Keys[pos], key)
	copy(c.Vals[pos], val)
	if pos+1 > c.Len {
		c.Len = pos + 1
	}
}

// Clone creates a deep copy of the cache up to currentPos.
func (c *KVCache) Clone(currentPos int) *KVCache {
	maxPos := len(c.Keys)
	d := &KVCache{
		Keys:    make([][]float32, maxPos),
		Vals:    make([][]float32, maxPos),
		KeyData: make([]float32, maxPos*c.Dim),
		ValData: make([]float32, maxPos*c.Dim),
		Len:     c.Len,
		Dim:     c.Dim,
	}
	for p := 0; p < maxPos; p++ {
		d.Keys[p] = d.KeyData[p*c.Dim : (p+1)*c.Dim]
		d.Vals[p] = d.ValData[p*c.Dim : (p+1)*c.Dim]
	}
	if currentPos > 0 {
		copy(d.KeyData[:currentPos*c.Dim], c.KeyData[:currentPos*c.Dim])
		copy(d.ValData[:currentPos*c.Dim], c.ValData[:currentPos*c.Dim])
	}
	return d
}

// MultiLayerKVCache holds KV caches for all layers of a transformer model.
type MultiLayerKVCache struct {
	Layers []*KVCache
}

// NewMultiLayerKVCache creates caches for nLayers transformer layers.
func NewMultiLayerKVCache(nLayers, maxPos, dim int) *MultiLayerKVCache {
	m := &MultiLayerKVCache{
		Layers: make([]*KVCache, nLayers),
	}
	for l := 0; l < nLayers; l++ {
		m.Layers[l] = NewKVCache(maxPos, dim)
	}
	return m
}

// Reset clears all layer caches.
func (m *MultiLayerKVCache) Reset() {
	for _, l := range m.Layers {
		l.Reset()
	}
}
