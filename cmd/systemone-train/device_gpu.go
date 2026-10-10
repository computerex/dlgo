//go:build cgo && vulkan

package main

import (
	"fmt"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

// newGPUExtractor wraps the CPU pipeline's model in a Vulkan GPU pipeline for
// GPU-native feature extraction. The returned cleanup releases all GPU
// resources; the caller must invoke it before closing the CPU model.
func newGPUExtractor(pipeline *llm.Pipeline) (hiddenExtractor, func(), error) {
	gp, err := gpu.NewGpuPipeline(pipeline)
	if err != nil {
		return nil, nil, fmt.Errorf("gpu pipeline: %w", err)
	}
	return gp, func() { gp.FreeAll() }, nil
}

// gpuCacheSuffix identifies the extraction device (and hardware) in feature
// cache keys so GPU features never collide with CPU features.
func gpuCacheSuffix() string {
	return "device=gpu:" + gpu.DeviceName()
}

// ensureGPUCacheSuffix initializes the Vulkan backend (if needed) so
// gpuCacheSuffix reports the real device name — required for cache-key
// lookups in union mode, which never creates a GPU pipeline. The returned
// cleanup shuts the backend down only if this call initialized it.
func ensureGPUCacheSuffix() (func(), error) {
	wasInitialized := gpu.IsInitialized()
	if !wasInitialized {
		if err := gpu.Init(); err != nil {
			return nil, fmt.Errorf("gpu init for cache keys: %w", err)
		}
	}
	return func() {
		if !wasInitialized {
			gpu.Shutdown()
		}
	}, nil
}