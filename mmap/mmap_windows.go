package mmap

import (
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

// Open memory-maps a file read-only using Windows file mapping APIs.
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

	handle := windows.Handle(f.Fd())
	mapping, err := windows.CreateFileMapping(handle, nil, windows.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("mmap: CreateFileMapping: %w", err)
	}

	ptr, err := windows.MapViewOfFile(mapping, windows.FILE_MAP_READ, 0, 0, 0)
	if err != nil {
		windows.CloseHandle(mapping)
		f.Close()
		return nil, fmt.Errorf("mmap: MapViewOfFile: %w", err)
	}
	windows.CloseHandle(mapping)

	data := unsafe.Slice((*byte)(unsafe.Pointer(ptr)), size)

	return &MappedFile{
		Data: data,
		f:    f,
		size: size,
	}, nil
}

// Close unmaps the file and releases all resources.
func (mf *MappedFile) Close() error {
	if mf.Data != nil {
		ptr := uintptr(unsafe.Pointer(&mf.Data[0]))
		windows.UnmapViewOfFile(ptr)
		mf.Data = nil
	}
	if mf.f != nil {
		mf.f.Close()
		mf.f = nil
	}
	return nil
}
