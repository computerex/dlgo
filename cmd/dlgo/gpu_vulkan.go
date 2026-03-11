//go:build cgo && vulkan

package main

import (
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/server"
)

var gpuInitOnce sync.Once

type gpuChatRunner struct {
	cpuPipe *llm.Pipeline
	gpuPipe *gpu.GpuPipeline
}

func (r *gpuChatRunner) generate(prompt string, cfg llm.GenerateConfig) (*turnResult, error) {
	result, err := r.gpuPipe.GenerateDetailed(prompt, cfg)
	if err != nil {
		return nil, err
	}
	text := strings.TrimSpace(trimStopText(result.Text))
	return &turnResult{
		Text:         text,
		TokensPerSec: result.TokensPerSec,
		PrefillMs:    result.PrefillTimeMs,
		PrefillDelta: result.PromptTokens,
		GenerateMs:   result.GenerateTimeMs,
		PromptTokens: result.PromptTokens,
		OutputTokens: result.TotalTokens,
	}, nil
}

func setupRunner(pipe *llm.Pipeline, useGPU bool) (generateRunner, string) {
	if !useGPU {
		return &cpuRunner{pipe: pipe}, ""
	}

	if err := gpu.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: GPU init failed, falling back to CPU: %v\n", err)
		return &cpuRunner{pipe: pipe}, ""
	}

	gp, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: GPU pipeline failed, falling back to CPU: %v\n", err)
		return &cpuRunner{pipe: pipe}, ""
	}

	return &gpuChatRunner{cpuPipe: pipe, gpuPipe: gp}, gpu.DeviceName()
}

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
}
