// Package mmap provides read-only memory-mapped file access.
// On supported platforms, the OS virtual memory system pages data in
// from disk on demand, allowing models larger than physical RAM.
package mmap

import "os"

// MappedFile represents a read-only memory-mapped file.
type MappedFile struct {
	Data []byte // Read-only view of the entire file
	f    *os.File
	size int64
}

// Slice returns a sub-slice of the mapped region.
// The returned slice shares the underlying mmap'd memory.
func (mf *MappedFile) Slice(offset, length int64) []byte {
	return mf.Data[offset : offset+length]
}
