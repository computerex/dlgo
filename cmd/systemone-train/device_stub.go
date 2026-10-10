//go:build !cgo || !vulkan

package main

import (
	"fmt"

	"github.com/computerex/dlgo/models/llm"
)

// newGPUExtractor is unavailable without a Vulkan build.
func newGPUExtractor(pipeline *llm.Pipeline) (hiddenExtractor, func(), error) {
	return nil, nil, fmt.Errorf("-device gpu requires a build with -tags \"cgo vulkan\"")
}

func gpuCacheSuffix() string {
	return "device=gpu"
}

// ensureGPUCacheSuffix is unavailable without a Vulkan build; union mode
// lookups against gpu-keyed caches will report missing entries.
func ensureGPUCacheSuffix() (func(), error) {
	return func() {}, nil
}