package server

import (
	"fmt"
	"sync"
)

// BlockSize is the number of tokens per KV cache block (page).
const BlockSize = 16

// BlockManager manages a pool of fixed-size KV cache blocks for PagedAttention.
// Each physical block stores BlockSize tokens worth of K and V vectors.
// Sequences map logical block indices to physical block IDs via block tables.
type BlockManager struct {
	mu          sync.Mutex
	nLayers     int
	kvDim       int
	blockSize   int
	totalBlocks int

	// Block allocator
	freeBlocks []int // stack of free physical block IDs
	refCounts  []int // reference count per physical block (for CoW sharing)
}

// SeqBlockTable is the per-sequence mapping from logical blocks to physical blocks.
type SeqBlockTable struct {
	BlockIDs  []int // logical block index → physical block ID
	NumTokens int   // total tokens stored
	blockSize int
}

// NewBlockManager creates a block manager with the given capacity.
func NewBlockManager(nLayers, kvDim, blockSize, totalBlocks int) *BlockManager {
	bm := &BlockManager{
		nLayers:     nLayers,
		kvDim:       kvDim,
		blockSize:   blockSize,
		totalBlocks: totalBlocks,
		freeBlocks:  make([]int, totalBlocks),
		refCounts:   make([]int, totalBlocks),
	}
	for i := 0; i < totalBlocks; i++ {
		bm.freeBlocks[i] = totalBlocks - 1 - i // stack: top = 0
	}
	return bm
}

// FreeCount returns the number of available blocks.
func (bm *BlockManager) FreeCount() int {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	return len(bm.freeBlocks)
}

// TotalBlocks returns the total block capacity.
func (bm *BlockManager) TotalBlocks() int {
	return bm.totalBlocks
}

// BlockSize returns tokens per block.
func (bm *BlockManager) BlockSizeTokens() int {
	return bm.blockSize
}

// AllocBlock pops a free block from the pool.
func (bm *BlockManager) AllocBlock() (int, bool) {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	if len(bm.freeBlocks) == 0 {
		return -1, false
	}
	n := len(bm.freeBlocks)
	blockID := bm.freeBlocks[n-1]
	bm.freeBlocks = bm.freeBlocks[:n-1]
	bm.refCounts[blockID] = 1
	return blockID, true
}

// FreeBlock decrements the ref count and returns the block to the pool if zero.
func (bm *BlockManager) FreeBlock(blockID int) {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	if blockID < 0 || blockID >= bm.totalBlocks {
		return
	}
	bm.refCounts[blockID]--
	if bm.refCounts[blockID] <= 0 {
		bm.refCounts[blockID] = 0
		bm.freeBlocks = append(bm.freeBlocks, blockID)
	}
}

// IncrRef increments the reference count for a block (used for sharing).
func (bm *BlockManager) IncrRef(blockID int) {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	if blockID >= 0 && blockID < bm.totalBlocks {
		bm.refCounts[blockID]++
	}
}

// RefCount returns the current reference count for a block.
func (bm *BlockManager) RefCount(blockID int) int {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	if blockID < 0 || blockID >= bm.totalBlocks {
		return 0
	}
	return bm.refCounts[blockID]
}

// NewSeqBlockTable creates an empty block table for a new sequence.
func NewSeqBlockTable(blockSize int) *SeqBlockTable {
	return &SeqBlockTable{
		blockSize: blockSize,
	}
}

// AppendToken records that one more token has been added to the sequence.
// If the current last block is full, a new block must be allocated first.
func (bt *SeqBlockTable) AppendToken(bm *BlockManager) error {
	slotInBlock := bt.NumTokens % bt.blockSize
	if slotInBlock == 0 {
		blockID, ok := bm.AllocBlock()
		if !ok {
			return fmt.Errorf("out of KV cache blocks")
		}
		bt.BlockIDs = append(bt.BlockIDs, blockID)
	}
	bt.NumTokens++
	return nil
}

// AppendTokens reserves space for n tokens, allocating blocks as needed.
func (bt *SeqBlockTable) AppendTokens(bm *BlockManager, n int) error {
	for i := 0; i < n; i++ {
		if err := bt.AppendToken(bm); err != nil {
			return err
		}
	}
	return nil
}

// FreeAll releases all blocks owned by this sequence back to the pool.
func (bt *SeqBlockTable) FreeAll(bm *BlockManager) {
	for _, blockID := range bt.BlockIDs {
		bm.FreeBlock(blockID)
	}
	bt.BlockIDs = nil
	bt.NumTokens = 0
}

// NumBlocks returns the number of blocks currently allocated.
func (bt *SeqBlockTable) NumBlocks() int {
	return len(bt.BlockIDs)
}

// BlocksNeeded returns how many blocks are needed for n tokens.
func BlocksNeeded(nTokens, blockSize int) int {
	if nTokens <= 0 {
		return 0
	}
	return (nTokens + blockSize - 1) / blockSize
}

// BlockTableAsInt32 returns the block table as int32 slice for GPU upload.
func (bt *SeqBlockTable) BlockTableAsInt32() []int32 {
	result := make([]int32, len(bt.BlockIDs))
	for i, id := range bt.BlockIDs {
		result[i] = int32(id)
	}
	return result
}

// ForkSeq creates a copy of a block table with shared blocks (CoW).
func (bt *SeqBlockTable) ForkSeq(bm *BlockManager) *SeqBlockTable {
	newBT := &SeqBlockTable{
		BlockIDs:  make([]int, len(bt.BlockIDs)),
		NumTokens: bt.NumTokens,
		blockSize: bt.blockSize,
	}
	copy(newBT.BlockIDs, bt.BlockIDs)
	for _, blockID := range bt.BlockIDs {
		bm.IncrRef(blockID)
	}
	return newBT
}
