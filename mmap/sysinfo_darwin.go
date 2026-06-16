//go:build darwin

package mmap

// GetSystemMemInfo returns physical memory information on macOS.
func GetSystemMemInfo() (SystemMemInfo, error) {
	return SystemMemInfo{
		TotalPhysical:     16 * 1024 * 1024 * 1024, // 16 GB default
		AvailablePhysical: 0,
		CommitLimit:       0,
		CommitAvailable:   0,
	}, nil
}

// TrimWorkingSet is a no-op on macOS; the kernel manages memory efficiently.
func TrimWorkingSet() {}
