//go:build cgo && vulkan

package main

import (
	"fmt"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
)

// hiddenExtractor is the feature-extraction interface shared by the CPU
// pipeline and the Vulkan GPU pipeline.
type hiddenExtractor interface {
	SystemOneHidden(state string, q llm.SystemOneQuestion, cfg llm.SystemOneConfig) ([]float32, []llm.Choice, int, error)
}

// newGPUExtractor wraps the CPU pipeline's model in a Vulkan GPU pipeline for
// GPU-native feature extraction and decision evaluation.
func newGPUExtractor(pipeline *llm.Pipeline) (hiddenExtractor, func(), error) {
	gp, err := gpu.NewGpuPipeline(pipeline)
	if err != nil {
		return nil, nil, fmt.Errorf("gpu pipeline: %w", err)
	}
	return gp, func() { gp.FreeAll() }, nil
}