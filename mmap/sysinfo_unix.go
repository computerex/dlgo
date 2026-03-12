//go:build !windows

package mmap

import "syscall"

// GetSystemMemInfo queries the system for physical memory statistics.
func GetSystemMemInfo() (SystemMemInfo, error) {
	var si syscall.Sysinfo_t
	if err := syscall.Sysinfo(&si); err != nil {
		return SystemMemInfo{}, err
	}
	unit := uint64(si.Unit)
	return SystemMemInfo{
		TotalPhysical:     uint64(si.Totalram) * unit,
		AvailablePhysical: uint64(si.Freeram) * unit,
	}, nil
}

// TrimWorkingSet is a no-op on Linux; the kernel aggressively reclaims
// unmapped file pages. On Linux you could use madvise(MADV_DONTNEED) on
// specific ranges, but the kernel's page cache management is already
// effective for mmap'd files.
func TrimWorkingSet() {}

// SetWorkingSetLimit is a no-op on Linux; the kernel manages page cache
// eviction effectively via its own heuristics.
func SetWorkingSetLimit(maxBytes uint64) {}
