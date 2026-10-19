//go:build !vulkan || !cgo

package gpu

import "github.com/computerex/dlgo/models/llm"

type GpuPipeline struct{}

// GenerateResult is an alias of llm.GenerateResult so the CPU and GPU backends
// share one result type (see gpu/pipeline.go).
type GenerateResult = llm.GenerateResult

func NewGpuPipeline(_ *llm.Pipeline) (*GpuPipeline, error) {
	return nil, errNoGPU
}

// NewRunner in a non-Vulkan build has no GPU to use, so it always returns the
// CPU pipeline as the Runner. This keeps callers backend-agnostic: the same
// one-call API works whether or not the binary was built with Vulkan support.
func NewRunner(modelPath string, ctx int) (llm.Runner, string, error) {
	pipe, err := llm.NewPipeline(modelPath, ctx)
	if err != nil {
		return nil, "", err
	}
	return pipe, "CPU", nil
}

func (p *GpuPipeline) FreeAll()    {}
func (p *GpuPipeline) ResetState() {}

func (p *GpuPipeline) GenerateDetailed(_ string, _ llm.GenerateConfig) (*GenerateResult, error) {
	return nil, errNoGPU
}
