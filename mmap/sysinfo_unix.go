//go:build !windows && !darwin

package mmap

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"syscall"
)

// GetSystemMemInfo queries the system for physical memory statistics.
func GetSystemMemInfo() (SystemMemInfo, error) {
	var si syscall.Sysinfo_t
	if err := syscall.Sysinfo(&si); err != nil {
		return SystemMemInfo{}, err
	}
	unit := uint64(si.Unit)
	total := uint64(si.Totalram) * unit
	avail := uint64(si.Freeram) * unit

	// Prefer MemAvailable from /proc/meminfo — Freeram excludes
	// reclaimable page cache, causing false OOM on Linux/WSL2.
	if f, err := os.Open("/proc/meminfo"); err == nil {
		defer f.Close()
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			if strings.HasPrefix(sc.Text(), "MemAvailable:") {
				fields := strings.Fields(sc.Text())
				if len(fields) >= 2 {
					if kb, err := strconv.ParseUint(fields[1], 10, 64); err == nil {
						avail = kb * 1024
					}
				}
				break
			}
		}
	}

	return SystemMemInfo{
		TotalPhysical:     total,
		AvailablePhysical: avail,
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

// PrefetchRegion uses madvise(MADV_WILLNEED) on Linux to pre-fault pages.
func PrefetchRegion(data []byte) {
	if len(data) == 0 {
		return
	}
	_ = syscall.Madvise(data, syscall.MADV_WILLNEED)
}
