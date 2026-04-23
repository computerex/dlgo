//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"

	"github.com/computerex/dlgo/mmap"
)

const (
	// Minimum free physical RAM before we consider the system under pressure.
	// Below this, we trigger aggressive GC and working set trim.
	ramPressureThresholdBytes = 2 * (1 << 30) // 2 GB

	// Below this, we log a critical warning. The system is close to freezing.
	ramCriticalThresholdBytes = 1 * (1 << 30) // 1 GB

	// How often the background monitor checks RAM.
	ramMonitorInterval = 500 * time.Millisecond

	// Safety margin for Go memory limit: leave this much for OS + other processes.
	goMemLimitReserveBytes = 3 * (1 << 30) // 3 GB
)

var (
	memMonitorOnce   sync.Once
	memMonitorCancel atomic.Int32
)

// InitMemorySafety sets Go runtime memory limits and starts a background
// RAM pressure monitor. Call once after GPU initialization, before generation.
//
// This prevents the two most common crash vectors:
//   1. Go heap growing unbounded → system OOM freeze
//   2. Mmap working set inflating uncontrolled → system thrashing
func InitMemorySafety() {
	sysInfo, err := mmap.GetSystemMemInfo()
	if err != nil {
		fmt.Fprintf(os.Stderr, "[dlgo/memsafety] cannot query RAM: %v\n", err)
		return
	}

	totalRAM := int64(sysInfo.TotalPhysical)
	availRAM := int64(sysInfo.AvailablePhysical)

	// Set Go soft memory limit based on available RAM.
	// This caps the Go GC target so the runtime aggressively collects
	// before the heap grows too large. Does NOT affect C/mmap allocations.
	goLimit := availRAM - goMemLimitReserveBytes
	if goLimit < 1<<30 { // minimum 1 GB
		goLimit = 1 << 30
	}
	if goLimit > 8*(1<<30) { // cap at 8 GB — Go heap shouldn't need more
		goLimit = 8 * (1 << 30)
	}
	prev := debug.SetMemoryLimit(goLimit)
	fmt.Fprintf(os.Stderr, "[dlgo/memsafety] Go memory limit: %.1f GB (was %.1f GB), system RAM: %.1f/%.1f GB available\n",
		float64(goLimit)/(1<<30), float64(prev)/(1<<30),
		float64(availRAM)/(1<<30), float64(totalRAM)/(1<<30))

	// Start background monitor (once per process)
	memMonitorOnce.Do(func() {
		go ramPressureMonitor()
	})
}

// EnforceWorkingSetLimit caps the process working set after GPU model upload.
// Model weights have been copied to VRAM; the mmap pages are no longer needed
// in physical RAM. This forces the OS to evict them, preventing the process
// from holding GB of dead mmap pages in the working set.
func EnforceWorkingSetLimit(modelSizeBytes uint64) {
	sysInfo, err := mmap.GetSystemMemInfo()
	if err != nil {
		return
	}

	// Allow working set = total RAM minus a generous reserve for OS + VRAM backing.
	// The model data is on the GPU now; we only need RAM for:
	//   - KV cache (CPU layers), logits, sampling buffers (~100-500 MB)
	//   - Go runtime + stack (~200 MB)
	//   - OS + other processes (~3 GB)
	totalRAM := sysInfo.TotalPhysical
	reserve := uint64(4 * (1 << 30)) // 4 GB for OS + other apps
	maxWS := totalRAM - reserve
	if maxWS < 2*(1<<30) { // minimum 2 GB
		maxWS = 2 * (1 << 30)
	}

	mmap.SetWorkingSetLimit(maxWS)
	fmt.Fprintf(os.Stderr, "[dlgo/memsafety] Working set limit: %.1f GB (total RAM: %.1f GB)\n",
		float64(maxWS)/(1<<30), float64(totalRAM)/(1<<30))
}

// StopMemoryMonitor signals the background monitor to stop.
func StopMemoryMonitor() {
	memMonitorCancel.Store(1)
}

func ramPressureMonitor() {
	ticker := time.NewTicker(ramMonitorInterval)
	defer ticker.Stop()

	pressureCount := 0
	criticalCount := 0
	var lastGC time.Time

	for range ticker.C {
		if memMonitorCancel.Load() != 0 {
			return
		}

		sysInfo, err := mmap.GetSystemMemInfo()
		if err != nil {
			continue
		}

		availRAM := int64(sysInfo.AvailablePhysical)

		if availRAM < ramCriticalThresholdBytes {
			criticalCount++
			if criticalCount == 1 {
				fmt.Fprintf(os.Stderr, "\n[dlgo/memsafety] CRITICAL: only %.0f MB RAM free! Forcing GC + working set trim.\n",
					float64(availRAM)/(1<<20))
			}

			// Emergency measures: force GC and trim working set
			if time.Since(lastGC) > 2*time.Second {
				runtime.GC()
				debug.FreeOSMemory()
				mmap.TrimWorkingSet()
				lastGC = time.Now()
			}

			// If RAM is critically low for 5+ consecutive checks (2.5s),
			// something is badly wrong. Log loudly so logs can be inspected.
			if criticalCount >= 5 {
				fmt.Fprintf(os.Stderr, "[dlgo/memsafety] DANGER: sustained critical RAM pressure "+
					"(%.0f MB free for %d checks). System may freeze.\n",
					float64(availRAM)/(1<<20), criticalCount)
			}
		} else if availRAM < ramPressureThresholdBytes {
			pressureCount++
			if pressureCount == 1 {
				fmt.Fprintf(os.Stderr, "[dlgo/memsafety] WARNING: RAM pressure detected (%.0f MB free). Trimming working set.\n",
					float64(availRAM)/(1<<20))
			}
			// Moderate pressure: trim working set to release mmap pages
			if time.Since(lastGC) > 5*time.Second {
				mmap.TrimWorkingSet()
				lastGC = time.Now()
			}
		} else {
			if pressureCount > 0 || criticalCount > 0 {
				fmt.Fprintf(os.Stderr, "[dlgo/memsafety] RAM pressure resolved (%.0f MB free)\n",
					float64(availRAM)/(1<<20))
			}
			pressureCount = 0
			criticalCount = 0
		}
	}
}
