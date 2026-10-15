//go:build !vulkan || !cgo

// test_glm is a Vulkan diagnostic command and is intentionally excluded from
// normal portable installs.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "test_glm requires a cgo-enabled Vulkan build: go run -tags vulkan ./cmd/test_glm")
	os.Exit(1)
}
