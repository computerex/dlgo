//go:build cgo && vulkan

package main

import (
	"sync"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/server"
)

var gpuInitOnce sync.Once

type gpuPipelineAdapter struct {
	pipe *gpu.GpuPipeline
}

func (a *gpuPipelineAdapter) GenerateDetailed(prompt string, cfg llm.GenerateConfig) (*llm.GenerateResult, error) {
	result, err := a.pipe.GenerateDetailed(prompt, cfg)
	if err != nil {
		return nil, err
	}
	return &llm.GenerateResult{
		Text:           result.Text,
		Tokens:         result.Tokens,
		TokensPerSec:   result.TokensPerSec,
		PrefillTimeMs:  result.PrefillTimeMs,
		GenerateTimeMs: result.GenerateTimeMs,
		TotalTokens:    result.TotalTokens,
		PromptTokens:   result.PromptTokens,
	}, nil
}

func (a *gpuPipelineAdapter) Free() {
	if a.pipe != nil {
		a.pipe.FreeAll()
	}
}

func registerGPU(manager *server.ModelManager) {
	manager.SetGPUFunctions(
		func() error {
			var err error
			gpuInitOnce.Do(func() {
				err = gpu.Init()
			})
			return err
		},
		func(pipe *llm.Pipeline) (server.GpuPipelineInterface, error) {
			gp, err := gpu.NewGpuPipeline(pipe)
			if err != nil {
				return nil, err
			}
			return &gpuPipelineAdapter{pipe: gp}, nil
		},
	)
	manager.SetVRAMStatusFunc(func() *server.VRAMStatus {
		if !gpu.IsInitialized() {
			return nil
		}
		total := float64(gpu.VRAMBytes()) / (1024 * 1024)
		free := float64(gpu.VRAMFreeBytes()) / (1024 * 1024)
		return &server.VRAMStatus{
			TotalMB: total,
			FreeMB:  free,
			UsedMB:  total - free,
		}
	})
}
