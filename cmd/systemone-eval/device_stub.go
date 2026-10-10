//go:build !cgo || !vulkan

package main

import (
	"fmt"

	"github.com/computerex/dlgo/models/llm"
)

// hiddenExtractor is the feature-extraction interface shared by the CPU
// pipeline and the Vulkan GPU pipeline.
type hiddenExtractor interface {
	SystemOneHidden(state string, q llm.SystemOneQuestion, cfg llm.SystemOneConfig) ([]float32, []llm.Choice, int, error)
}

// newGPUExtractor is unavailable without a Vulkan build.
func newGPUExtractor(pipeline *llm.Pipeline) (hiddenExtractor, func(), error) {
	return nil, nil, fmt.Errorf("-device gpu requires a build with -tags \"cgo vulkan\"")
}