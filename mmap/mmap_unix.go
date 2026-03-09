//go:build !windows

package mmap

import (
	"fmt"
	"os"
	"syscall"
)

// Open memory-maps a file read-only using POSIX mmap.
func Open(path string) (*MappedFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	size := fi.Size()
	if size == 0 {
		f.Close()
		return nil, fmt.Errorf("mmap: file is empty")
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("mmap: %w", err)
	}

	return &MappedFile{
		Data: data,
		f:    f,
		size: size,
	}, nil
}

// Close unmaps the file and releases all resources.
func (mf *MappedFile) Close() error {
	if mf.Data != nil {
		syscall.Munmap(mf.Data)
		mf.Data = nil
	}
	if mf.f != nil {
		mf.f.Close()
		mf.f = nil
	}
	return nil
}
