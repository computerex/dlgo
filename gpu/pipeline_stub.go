//go:build !vulkan || !cgo

package gpu

import "github.com/computerex/dlgo/models/llm"

type GpuPipeline struct{}

type GenerateResult struct {
	Text           string
	Tokens         []int32
	TokensPerSec   float64
	PrefillTimeMs  float64
	GenerateTimeMs float64
	TotalTokens    int
	PromptTokens   int
}

func NewGpuPipeline(_ *llm.Pipeline) (*GpuPipeline, error) {
	return nil, errNoGPU
}

func (p *GpuPipeline) FreeAll()    {}
func (p *GpuPipeline) ResetState() {}

func (p *GpuPipeline) GenerateDetailed(_ string, _ llm.GenerateConfig) (*GenerateResult, error) {
	return nil, errNoGPU
}
