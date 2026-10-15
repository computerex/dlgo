//go:build !windows

// gpu-watchdog uses Windows Job Objects and PDH GPU counters. This stub keeps
// `go install github.com/computerex/dlgo/cmd/...@latest` cross-platform while
// making the platform limitation explicit when the command is invoked.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "gpu-watchdog is currently supported on Windows only (it requires Win32 Job Objects and PDH GPU counters).")
	os.Exit(1)
}
