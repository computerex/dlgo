//go:build !vulkan

package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/server"
)

func setupRunner(pipe *llm.Pipeline, useGPU bool) (generateRunner, string) {
	if useGPU {
		fmt.Fprintln(os.Stdout, "[GPU-STUB] GPU requested but build tag not detected. Using CPU.")
	}
	return &cpuRunner{pipe: pipe}, ""
}

func registerGPU(manager *server.ModelManager) {
	// GPU support not compiled in
}
