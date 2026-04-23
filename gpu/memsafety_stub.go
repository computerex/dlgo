//go:build !vulkan || !cgo

package gpu

func InitMemorySafety()                            {}
func EnforceWorkingSetLimit(_ uint64)               {}
func StopMemoryMonitor()                            {}
