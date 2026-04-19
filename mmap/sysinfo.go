package mmap

// SystemMemInfo holds system memory statistics.
type SystemMemInfo struct {
	TotalPhysical     uint64 // Total physical RAM in bytes
	AvailablePhysical uint64 // Currently available physical RAM in bytes
	CommitLimit       uint64 // Max committed memory (RAM + pagefile), 0 if unknown
	CommitAvailable   uint64 // Remaining commit charge available, 0 if unknown
}
