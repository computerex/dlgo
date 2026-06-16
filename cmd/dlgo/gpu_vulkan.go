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

	fmt.Fprintln(os.Stdout, "[GPU] Attempting to initialize Vulkan backend...")
	if err := gpu.Init(); err != nil {
		fmt.Fprintf(os.Stdout, "[GPU] Init failed: %v\n", err)
		fmt.Fprintln(os.Stdout, "[GPU] Falling back to CPU backend")
		return &cpuRunner{pipe: pipe}, ""
	}
	fmt.Fprintln(os.Stdout, "[GPU] Vulkan initialized successfully")

	fmt.Fprintln(os.Stdout, "[GPU] Creating GPU pipeline...")
	gp, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Fprintf(os.Stdout, "[GPU] Pipeline creation failed: %v\n", err)
		fmt.Fprintln(os.Stdout, "[GPU] Falling back to CPU backend")
		return &cpuRunner{pipe: pipe}, ""
	}
	fmt.Fprintln(os.Stdout, "[GPU] GPU pipeline created successfully")

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
