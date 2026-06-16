//go:build darwin

package mmap

// GetSystemMemInfo returns physical memory information on macOS.
func GetSystemMemInfo() (SystemMemInfo, error) {
	return SystemMemInfo{
		TotalPhysical:     16 * 1024 * 1024 * 1024, // 16 GB default
		AvailablePhysical: 16 * 1024 * 1024 * 1024, // Assume all 16 GB available
		CommitLimit:       0,
		CommitAvailable:   0,
	}, nil
}

// TrimWorkingSet is a no-op on macOS; the kernel manages memory efficiently.
func TrimWorkingSet() {}

// SetWorkingSetLimit is a no-op on macOS; the kernel manages memory limits.
func SetWorkingSetLimit(maxBytes uint64) {}

// PrefetchRegion is a no-op on macOS; the kernel handles page prefetching.
func PrefetchRegion(data []byte) {}
