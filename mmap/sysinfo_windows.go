package mmap

import (
	"unsafe"

	"golang.org/x/sys/windows"
)

type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

var (
	kernel32                     = windows.NewLazySystemDLL("kernel32.dll")
	procGlobalMemoryStatusEx     = kernel32.NewProc("GlobalMemoryStatusEx")
	procSetProcessWorkingSetSize = kernel32.NewProc("SetProcessWorkingSetSize")
	procPrefetchVirtualMemory    = kernel32.NewProc("PrefetchVirtualMemory")
)

// GetSystemMemInfo queries Windows for physical memory statistics.
func GetSystemMemInfo() (SystemMemInfo, error) {
	var ms memoryStatusEx
	ms.Length = uint32(unsafe.Sizeof(ms))
	r, _, err := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&ms)))
	if r == 0 {
		return SystemMemInfo{}, err
	}
	return SystemMemInfo{
		TotalPhysical:     ms.TotalPhys,
		AvailablePhysical: ms.AvailPhys,
		CommitLimit:       ms.TotalPageFile,
		CommitAvailable:   ms.AvailPageFile,
	}, nil
}

// TrimWorkingSet asks the OS to minimize the process working set, releasing
// resident mmap pages back to standby. File-backed pages can be faulted back
// in from disk on demand. Call after bulk GPU uploads or RAM pinning to
// reclaim physical memory consumed by the mmap page cache.
func TrimWorkingSet() {
	proc, _ := windows.GetCurrentProcess()
	procSetProcessWorkingSetSize.Call(
		uintptr(proc),
		^uintptr(0), // SIZE_T(-1)
		^uintptr(0), // SIZE_T(-1)
	)
}

// SetWorkingSetLimit caps the process working set so mmap page cache cannot
// grow beyond maxBytes of physical RAM. The OS will aggressively evict
// file-backed pages to stay under this limit, preventing system-wide memory
// pressure from large mmap'd models. Pages are faulted back from disk on demand.
func SetWorkingSetLimit(maxBytes uint64) {
	proc, _ := windows.GetCurrentProcess()
	minWS := uintptr(1024 * 1024) // 1 MB minimum
	maxWS := uintptr(maxBytes)
	procSetProcessWorkingSetSize.Call(uintptr(proc), minWS, maxWS)
}

// PrefetchRegion asks Windows to pre-fault pages for the given memory range
// into the page cache. Unlike heap pinning, this does NOT increase commit
// charge — the pages are file-backed and can be evicted by the OS at any time.
// This is how llama.cpp warms CPU layer pages on Windows.
func PrefetchRegion(data []byte) {
	if len(data) == 0 {
		return
	}
	if procPrefetchVirtualMemory.Find() != nil {
		return
	}
	type memoryRangeEntry struct {
		VirtualAddress uintptr
		NumberOfBytes  uintptr
	}
	entry := memoryRangeEntry{
		VirtualAddress: uintptr(unsafe.Pointer(&data[0])),
		NumberOfBytes:  uintptr(len(data)),
	}
	proc, _ := windows.GetCurrentProcess()
	procPrefetchVirtualMemory.Call(
		uintptr(proc),
		1,
		uintptr(unsafe.Pointer(&entry)),
		0,
	)
}
