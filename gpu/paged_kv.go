//go:build cgo && vulkan

package gpu

import "fmt"

// PagedKVPool manages block-pool GPU buffers for PagedAttention.
// Each layer has one large K pool and one large V pool.
// Layout: [totalBlocks][blockSize][kvDim] for both K and V.
type PagedKVPool struct {
	KeyPools    []Buf // [nLayers]
	ValPools    []Buf // [nLayers]
	NLayers     int
	KVDim       int
	BlockSize   int
	TotalBlocks int
}

// NewPagedKVPool allocates per-layer block pool buffers on GPU.
func NewPagedKVPool(nLayers, kvDim, blockSize, totalBlocks int) (*PagedKVPool, error) {
	pool := &PagedKVPool{
		KeyPools:    make([]Buf, nLayers),
		ValPools:    make([]Buf, nLayers),
		NLayers:     nLayers,
		KVDim:       kvDim,
		BlockSize:   blockSize,
		TotalBlocks: totalBlocks,
	}
	bufSize := uint64(totalBlocks) * uint64(blockSize) * uint64(kvDim) * 4
	a := allocChecker{}
	for l := 0; l < nLayers; l++ {
		pool.KeyPools[l] = a.alloc(bufSize)
		pool.ValPools[l] = a.alloc(bufSize)
		if a.err != nil {
			pool.FreeAll()
			return nil, fmt.Errorf("gpu: NewPagedKVPool(layer %d/%d): %w", l, nLayers, a.err)
		}
	}
	return pool, nil
}

// FreeAll releases all GPU block pool buffers.
func (p *PagedKVPool) FreeAll() {
	if p == nil {
		return
	}
	for _, b := range p.KeyPools {
		freeBuf(b)
	}
	for _, b := range p.ValPools {
		freeBuf(b)
	}
}

// EstimateVRAM returns the total VRAM needed for this pool in bytes.
func EstimatePagedKVPoolVRAM(nLayers, kvDim, blockSize, totalBlocks int) uint64 {
	perLayer := uint64(totalBlocks) * uint64(blockSize) * uint64(kvDim) * 4 * 2
	return perLayer * uint64(nLayers)
}

// StoreKV writes K and V for a single token into the block pool.
// blockTable is the sequence's mapping from logical to physical blocks.
// pos is the logical token position in the sequence.
func (p *PagedKVPool) StoreKV(layer int, k, v Buf, blockTable []int, pos int) error {
	logicalBlock := pos / p.BlockSize
	slot := pos % p.BlockSize
	physicalBlock := blockTable[logicalBlock]
	effectivePos := physicalBlock*p.BlockSize + slot
	return PagedKVStore(p.KeyPools[layer], p.ValPools[layer], k, v, effectivePos, p.KVDim)
}

// UploadBlockTable uploads a sequence's block table to a GPU buffer.
func UploadBlockTable(blockIDs []int32) (Buf, error) {
	if len(blockIDs) == 0 {
		return 0, nil
	}
	buf := Alloc(uint64(len(blockIDs) * 4))
	if buf == 0 {
		return 0, fmt.Errorf("gpu: failed to alloc block table buffer")
	}
	if err := UploadI32(buf, blockIDs); err != nil {
		Free(buf)
		return 0, err
	}
	return buf, nil
}
