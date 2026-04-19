// gpu-watchdog: launches a child process and kills it if GPU or system becomes unstable.
//
// Monitors:
//   - System RAM (GlobalMemoryStatusEx)
//   - GPU VRAM dedicated usage (PDH: GPU Adapter Memory)
//   - GPU 3D engine utilization (PDH: GPU Engine)
//   - GPU hang pre-detection (100% util for >1.5s → kill before TDR)
//   - Child output heartbeat (no stdout/stderr for >timeout → kill)
//
// Uses a Win32 Job Object so the child is killed even if the watchdog crashes.
//
// Usage: gpu-watchdog [options] -- <command> [args...]
package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

// ---------------------------------------------------------------------------
// Win32 APIs
// ---------------------------------------------------------------------------

var (
	kernel32                 = syscall.NewLazyDLL("kernel32.dll")
	procGlobalMemoryStatusEx = kernel32.NewProc("GlobalMemoryStatusEx")
	procCreateJobObjectW     = kernel32.NewProc("CreateJobObjectW")
	procSetInformationJobObj = kernel32.NewProc("SetInformationJobObject")
	procAssignProcessToJob   = kernel32.NewProc("AssignProcessToJobObject")

	pdh              = syscall.NewLazyDLL("pdh.dll")
	pdhOpenQuery     = pdh.NewProc("PdhOpenQueryW")
	pdhAddCounterW   = pdh.NewProc("PdhAddEnglishCounterW")
	pdhCollectData   = pdh.NewProc("PdhCollectQueryData")
	pdhGetDouble     = pdh.NewProc("PdhGetFormattedCounterValue")
	pdhGetLarge      = pdh.NewProc("PdhGetFormattedCounterValue")
	pdhGetArrayD     = pdh.NewProc("PdhGetFormattedCounterArrayW")
	pdhGetArrayL     = pdh.NewProc("PdhGetFormattedCounterArrayW")
	pdhCloseQuery    = pdh.NewProc("PdhCloseQuery")
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

type memStatus struct {
	AvailPhysMB   uint64
	LoadPct       uint32
	CommitUsedMB  uint64
	CommitLimitMB uint64
}

func getMemoryStatus() (memStatus, error) {
	var ms memoryStatusEx
	ms.Length = uint32(unsafe.Sizeof(ms))
	r, _, e := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&ms)))
	if r == 0 {
		return memStatus{}, e
	}
	return memStatus{
		AvailPhysMB:   ms.AvailPhys / (1024 * 1024),
		LoadPct:       ms.MemoryLoad,
		CommitUsedMB:  (ms.TotalPageFile - ms.AvailPageFile) / (1024 * 1024),
		CommitLimitMB: ms.TotalPageFile / (1024 * 1024),
	}, nil
}

// ---------------------------------------------------------------------------
// PDH GPU monitoring
// ---------------------------------------------------------------------------

const (
	PDH_FMT_DOUBLE uint32 = 0x00000200
	PDH_FMT_LARGE  uint32 = 0x00000400
)

type pdhCounterValueDouble struct {
	CStatus uint32
	_       [4]byte
	Value   float64
}

type pdhCounterValueLarge struct {
	CStatus uint32
	_       [4]byte
	Value   int64
}

// PDH_FMT_COUNTERVALUE_ITEM structures for array queries
type pdhItemDouble struct {
	Name  *uint16
	Value pdhCounterValueDouble
}

type pdhItemLarge struct {
	Name  *uint16
	Value pdhCounterValueLarge
}

type gpuMonitor struct {
	query      uintptr
	engineCtr  uintptr // GPU Engine 3D utilization (wildcard)
	memCtr     uintptr // GPU Adapter Memory dedicated usage (wildcard)
	ready      bool
}

func newGPUMonitor() *gpuMonitor {
	m := &gpuMonitor{}

	r, _, _ := pdhOpenQuery.Call(0, 0, uintptr(unsafe.Pointer(&m.query)))
	if r != 0 {
		log.Printf("[watchdog] PDH open failed (0x%x), GPU monitoring disabled", r)
		return m
	}

	// 3D engine utilization — wildcard matches all GPU engines of type 3D
	enginePath, _ := syscall.UTF16PtrFromString(`\GPU Engine(*engtype_3D)\Utilization Percentage`)
	r, _, _ = pdhAddCounterW.Call(m.query, uintptr(unsafe.Pointer(enginePath)), 0, uintptr(unsafe.Pointer(&m.engineCtr)))
	if r != 0 {
		log.Printf("[watchdog] PDH add engine counter failed (0x%x)", r)
	}

	// Dedicated VRAM usage — wildcard matches all adapters
	memPath, _ := syscall.UTF16PtrFromString(`\GPU Adapter Memory(*phys_0)\Dedicated Usage`)
	r, _, _ = pdhAddCounterW.Call(m.query, uintptr(unsafe.Pointer(memPath)), 0, uintptr(unsafe.Pointer(&m.memCtr)))
	if r != 0 {
		log.Printf("[watchdog] PDH add mem counter failed (0x%x)", r)
	}

	// Prime the counters (PDH needs 2 samples for rate counters)
	pdhCollectData.Call(m.query)
	time.Sleep(100 * time.Millisecond)
	pdhCollectData.Call(m.query)

	m.ready = true
	log.Printf("[watchdog] GPU monitoring enabled (PDH)")
	return m
}

func (m *gpuMonitor) sample() (gpuUtil float64, vramUsedMB int64) {
	if !m.ready {
		return 0, 0
	}

	r, _, _ := pdhCollectData.Call(m.query)
	if r != 0 {
		return 0, 0
	}

	// GPU utilization: read all wildcard instances, take max
	if m.engineCtr != 0 {
		var bufSize uint32
		var itemCount uint32
		// First call: get required buffer size
		pdhGetArrayD.Call(m.engineCtr, uintptr(PDH_FMT_DOUBLE), uintptr(unsafe.Pointer(&bufSize)), uintptr(unsafe.Pointer(&itemCount)), 0)
		if bufSize > 0 && itemCount > 0 {
			buf := make([]byte, bufSize)
			r, _, _ = pdhGetArrayD.Call(m.engineCtr, uintptr(PDH_FMT_DOUBLE),
				uintptr(unsafe.Pointer(&bufSize)), uintptr(unsafe.Pointer(&itemCount)),
				uintptr(unsafe.Pointer(&buf[0])))
			if r == 0 {
				itemSize := unsafe.Sizeof(pdhItemDouble{})
				for i := uint32(0); i < itemCount; i++ {
					item := (*pdhItemDouble)(unsafe.Pointer(uintptr(unsafe.Pointer(&buf[0])) + uintptr(i)*itemSize))
					if item.Value.Value > gpuUtil {
						gpuUtil = item.Value.Value
					}
				}
			}
		}
	}

	// VRAM: read all wildcard instances, take max
	if m.memCtr != 0 {
		var bufSize uint32
		var itemCount uint32
		pdhGetArrayL.Call(m.memCtr, uintptr(PDH_FMT_LARGE), uintptr(unsafe.Pointer(&bufSize)), uintptr(unsafe.Pointer(&itemCount)), 0)
		if bufSize > 0 && itemCount > 0 {
			buf := make([]byte, bufSize)
			r, _, _ = pdhGetArrayL.Call(m.memCtr, uintptr(PDH_FMT_LARGE),
				uintptr(unsafe.Pointer(&bufSize)), uintptr(unsafe.Pointer(&itemCount)),
				uintptr(unsafe.Pointer(&buf[0])))
			if r == 0 {
				itemSize := unsafe.Sizeof(pdhItemLarge{})
				for i := uint32(0); i < itemCount; i++ {
					item := (*pdhItemLarge)(unsafe.Pointer(uintptr(unsafe.Pointer(&buf[0])) + uintptr(i)*itemSize))
					mb := item.Value.Value / (1024 * 1024)
					if mb > vramUsedMB {
						vramUsedMB = mb
					}
				}
			}
		}
	}

	return gpuUtil, vramUsedMB
}

func (m *gpuMonitor) close() {
	if m.ready {
		pdhCloseQuery.Call(m.query)
	}
}

// ---------------------------------------------------------------------------
// Job Object — ensures child dies if watchdog crashes
// ---------------------------------------------------------------------------

// JOBOBJECT_EXTENDED_LIMIT_INFORMATION for JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
type jobObjectBasicLimitInfo struct {
	PerProcessUserTimeLimit int64
	PerJobUserTimeLimit     int64
	LimitFlags              uint32
	MinimumWorkingSetSize   uintptr
	MaximumWorkingSetSize   uintptr
	ActiveProcessLimit      uint32
	Affinity                uintptr
	PriorityClass           uint32
	SchedulingClass         uint32
}

type ioCounters struct {
	ReadOperationCount  uint64
	WriteOperationCount uint64
	OtherOperationCount uint64
	ReadTransferCount   uint64
	WriteTransferCount  uint64
	OtherTransferCount  uint64
}

type jobObjectExtendedLimitInfo struct {
	BasicLimitInformation jobObjectBasicLimitInfo
	IoInfo                ioCounters
	ProcessMemoryLimit    uintptr
	JobMemoryLimit        uintptr
	PeakProcessMemoryUsed uintptr
	PeakJobMemoryUsed     uintptr
}

const (
	JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
	JobObjectExtendedLimitInformation   = 9
)

func createJobObject() (syscall.Handle, error) {
	r, _, err := procCreateJobObjectW.Call(0, 0)
	if r == 0 {
		return 0, fmt.Errorf("CreateJobObject: %v", err)
	}
	job := syscall.Handle(r)

	var info jobObjectExtendedLimitInfo
	info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	r, _, err = procSetInformationJobObj.Call(
		uintptr(job),
		uintptr(JobObjectExtendedLimitInformation),
		uintptr(unsafe.Pointer(&info)),
		uintptr(unsafe.Sizeof(info)),
	)
	if r == 0 {
		return 0, fmt.Errorf("SetInformationJobObject: %v", err)
	}

	return job, nil
}

func assignProcessToJob(job syscall.Handle, process syscall.Handle) error {
	r, _, err := procAssignProcessToJob.Call(uintptr(job), uintptr(process))
	if r == 0 {
		return fmt.Errorf("AssignProcessToJobObject: %v", err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Heartbeat writer — tracks last output time from child
// ---------------------------------------------------------------------------

type heartbeatWriter struct {
	inner     io.Writer
	lastWrite atomic.Int64 // unix millis of last write
}

func newHeartbeatWriter(w io.Writer) *heartbeatWriter {
	hw := &heartbeatWriter{inner: w}
	hw.lastWrite.Store(time.Now().UnixMilli())
	return hw
}

func (hw *heartbeatWriter) Write(p []byte) (int, error) {
	hw.lastWrite.Store(time.Now().UnixMilli())
	return hw.inner.Write(p)
}

func (hw *heartbeatWriter) silenceMs() int64 {
	return time.Now().UnixMilli() - hw.lastWrite.Load()
}

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------

const (
	minAvailRAM_MB       = 1500  // Kill if available RAM drops below this
	maxMemoryLoad        = 92    // Kill if memory load exceeds this %
	minCommitFreeMB      = 2000  // Kill if commit charge headroom drops below 2 GB
	maxVRAM_MB           = 15800 // Kill if GPU VRAM usage exceeds this (16 GB GPU, ~200 MB headroom for DWM)
	pollIntervalMS       = 100   // Check every 100ms
	graceChecks          = 3     // Must fail N consecutive checks before kill
	gpuHangThresholdMS   = 1500  // Kill if GPU at 100% for this long (before TDR at 2s)
	heartbeatTimeoutMS   = 300000 // Kill if no output for 5min
)

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	// Find "--" separator
	sep := -1
	for i, arg := range os.Args[1:] {
		if arg == "--" {
			sep = i + 1
			break
		}
	}
	if sep == -1 || sep+1 >= len(os.Args) {
		fmt.Fprintf(os.Stderr, "Usage: gpu-watchdog -- <command> [args...]\n")
		fmt.Fprintf(os.Stderr, "\nLaunches <command> and kills it if GPU/system becomes unstable.\n")
		fmt.Fprintf(os.Stderr, "Monitors: RAM, GPU VRAM (PDH), GPU utilization, output heartbeat.\n")
		fmt.Fprintf(os.Stderr, "Thresholds: RAM < %d MB or load > %d%%, GPU hang > %dms, silence > %ds\n",
			minAvailRAM_MB, maxMemoryLoad, gpuHangThresholdMS, heartbeatTimeoutMS/1000)
		os.Exit(2)
	}

	cmdArgs := os.Args[sep+1:]
	log.SetFlags(log.Ltime | log.Lmicroseconds)

	// Set up Job Object — child will be killed if watchdog exits/crashes
	job, jobErr := createJobObject()
	if jobErr != nil {
		log.Printf("[watchdog] WARNING: Job object failed: %v (child may survive crashes)", jobErr)
	}

	// Initialize GPU monitoring
	gpuMon := newGPUMonitor()
	defer gpuMon.close()

	// Check baseline
	mem, err := getMemoryStatus()
	if err != nil {
		log.Fatalf("Cannot read memory status: %v", err)
	}
	gpuUtil, vramMB := gpuMon.sample()
	log.Printf("[watchdog] Baseline: %d MB RAM free, %d%% load, commit %d/%d MB, GPU %.0f%%, VRAM %d MB",
		mem.AvailPhysMB, mem.LoadPct, mem.CommitUsedMB, mem.CommitLimitMB, gpuUtil, vramMB)
	log.Printf("[watchdog] Thresholds: RAM < %d MB, load > %d%%, commit free < %d MB, VRAM > %d MB, GPU hang > %dms",
		minAvailRAM_MB, maxMemoryLoad, minCommitFreeMB, maxVRAM_MB, gpuHangThresholdMS)
	log.Printf("[watchdog] Starting: %s", strings.Join(cmdArgs, " "))

	// Heartbeat writers to track output activity
	stdoutHB := newHeartbeatWriter(os.Stdout)
	stderrHB := newHeartbeatWriter(os.Stderr)

	// Start child process — pass through stdin so interactive prompts work
	cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = stdoutHB
	cmd.Stderr = stderrHB
	cmd.Env = os.Environ()

	if err := cmd.Start(); err != nil {
		log.Fatalf("[watchdog] Failed to start: %v", err)
	}

	pid := cmd.Process.Pid
	log.Printf("[watchdog] Child PID: %d", pid)

	// Assign to job object
	if jobErr == nil {
		const PROCESS_ALL_ACCESS = 0x1F0FFF
		h, e := syscall.OpenProcess(PROCESS_ALL_ACCESS, false, uint32(pid))
		if e == nil {
			if err := assignProcessToJob(job, h); err != nil {
				log.Printf("[watchdog] WARNING: assign to job failed: %v", err)
			} else {
				log.Printf("[watchdog] Child assigned to job object (kill-on-close)")
			}
			syscall.CloseHandle(h)
		}
	}

	// Channel for child exit
	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()

	// Kill helper
	var killOnce sync.Once
	killChild := func(reason string) {
		killOnce.Do(func() {
			log.Printf("[watchdog] KILLING child (PID %d): %s", pid, reason)
			cmd.Process.Kill()
			select {
			case <-done:
			case <-time.After(3 * time.Second):
				log.Printf("[watchdog] Child did not exit after kill")
			}
		})
	}

	// Monitor loop
	ticker := time.NewTicker(pollIntervalMS * time.Millisecond)
	defer ticker.Stop()

	ramFailCount := 0
	commitFailCount := 0
	vramFailCount := 0
	gpuFullSinceMS := int64(0)
	sampleCount := 0

	for {
		select {
		case err := <-done:
			if err != nil {
				if exitErr, ok := err.(*exec.ExitError); ok {
					code := exitErr.ExitCode()
					log.Printf("[watchdog] Child exited with code %d", code)
					os.Exit(code)
				}
				log.Printf("[watchdog] Child error: %v", err)
				os.Exit(1)
			}
			log.Printf("[watchdog] Child exited successfully")
			os.Exit(0)

		case <-ticker.C:
			now := time.Now().UnixMilli()
			sampleCount++

			// --- RAM check ---
			mem, err := getMemoryStatus()
			if err == nil {
				ramCritical := mem.AvailPhysMB < minAvailRAM_MB || mem.LoadPct > maxMemoryLoad
				if ramCritical {
					ramFailCount++
					if ramFailCount == 1 {
						log.Printf("[watchdog] RAM WARNING: %d MB free, %d%% load (%d/%d)",
							mem.AvailPhysMB, mem.LoadPct, ramFailCount, graceChecks)
					}
					if ramFailCount >= graceChecks {
						killChild(fmt.Sprintf("RAM critical: %d MB free, %d%% load", mem.AvailPhysMB, mem.LoadPct))
						os.Exit(99)
					}
				} else {
					if ramFailCount > 0 {
						log.Printf("[watchdog] RAM recovered: %d MB free, %d%% load", mem.AvailPhysMB, mem.LoadPct)
					}
					ramFailCount = 0
				}

				// --- Commit charge check ---
				commitFreeMB := int64(mem.CommitLimitMB) - int64(mem.CommitUsedMB)
				if commitFreeMB < minCommitFreeMB {
					commitFailCount++
					if commitFailCount == 1 {
						log.Printf("[watchdog] COMMIT WARNING: %d MB free of %d MB (%d/%d)",
							commitFreeMB, mem.CommitLimitMB, commitFailCount, graceChecks)
					}
					if commitFailCount >= graceChecks {
						killChild(fmt.Sprintf("commit charge critical: %d/%d MB (only %d MB free)",
							mem.CommitUsedMB, mem.CommitLimitMB, commitFreeMB))
						os.Exit(96)
					}
				} else {
					if commitFailCount > 0 {
						log.Printf("[watchdog] Commit recovered: %d MB free", commitFreeMB)
					}
					commitFailCount = 0
				}
			}

			// --- GPU check ---
			gpuUtil, vramMB := gpuMon.sample()

			// VRAM absolute ceiling — kill before WDDM starts evicting DWM pages
			if vramMB > maxVRAM_MB {
				vramFailCount++
				if vramFailCount == 1 {
					log.Printf("[watchdog] VRAM WARNING: %d MB > %d MB threshold (%d/%d)",
						vramMB, maxVRAM_MB, vramFailCount, graceChecks)
				}
				if vramFailCount >= graceChecks {
					killChild(fmt.Sprintf("VRAM critical: %d MB > %d MB threshold", vramMB, maxVRAM_MB))
					os.Exit(95)
				}
			} else {
				if vramFailCount > 0 {
					log.Printf("[watchdog] VRAM recovered: %d MB", vramMB)
				}
				vramFailCount = 0
			}

			// GPU hang detection: 100% utilization for too long → TDR imminent
			if gpuUtil >= 99.0 {
				if gpuFullSinceMS == 0 {
					gpuFullSinceMS = now
				} else if now-gpuFullSinceMS > gpuHangThresholdMS {
					killChild(fmt.Sprintf("GPU hang detected: 100%% util for %dms, VRAM %d MB",
						now-gpuFullSinceMS, vramMB))
					os.Exit(98)
				}
			} else {
				gpuFullSinceMS = 0
			}

			// Periodic status (every ~5s)
			if sampleCount%(5000/pollIntervalMS) == 0 {
				log.Printf("[watchdog] Status: RAM %d MB free (%d%%), commit %d/%d MB, VRAM %d MB, GPU %.0f%%",
					mem.AvailPhysMB, mem.LoadPct, mem.CommitUsedMB, mem.CommitLimitMB, vramMB, gpuUtil)
			}

			// --- Heartbeat check ---
			silenceStdout := stdoutHB.silenceMs()
			silenceStderr := stderrHB.silenceMs()
			silence := silenceStdout
			if silenceStderr < silence {
				silence = silenceStderr
			}

			if silence > heartbeatTimeoutMS {
				killChild(fmt.Sprintf("no output for %ds (GPU %.0f%%, VRAM %d MB)",
					silence/1000, gpuUtil, vramMB))
				os.Exit(97)
			}
		}
	}
}
