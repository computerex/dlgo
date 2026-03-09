package mmap

// SystemMemInfo holds system memory statistics.
type SystemMemInfo struct {
	TotalPhysical     uint64 // Total physical RAM in bytes
	AvailablePhysical uint64 // Currently available physical RAM in bytes
}
