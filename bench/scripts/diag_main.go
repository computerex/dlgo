//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: diag <gguf_path>")
		os.Exit(1)
	}
	ggufPath := os.Args[1]

	if err := gpu.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "gpu init: %v\n", err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	pipe, err := llm.NewPipeline(ggufPath, 512)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	gpuPipe, err := gpu.NewGpuPipeline(pipe)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gpu pipeline: %v\n", err)
		os.Exit(1)
	}
	defer gpuPipe.FreeAll()

	cfg := pipe.Model.Config
	dim := cfg.EmbeddingDim
	m := pipe.Model

	token := int32(1) // BOS or common token
	xCPU := make([]float32, dim)
	_ = m.TokenEmbed.DequantizeRow(int(token), xCPU)
	if cfg.EmbedScale != 0 {
		for i := range xCPU {
			xCPU[i] *= cfg.EmbedScale
		}
	}
	fmt.Printf("Token %d embedding L2: %.6f\n", token, l2norm(xCPU))

	// CPU: RMSNorm
	cpuNorm := make([]float32, dim)
	ops.RMSNorm(cpuNorm, xCPU, m.Layers[0].AttnNorm, cfg.RMSNormEps)
	fmt.Printf("CPU attn_norm L2: %.6f\n", l2norm(cpuNorm))

	// CPU: Q = Wq * norm
	cpuQ := make([]float32, m.Layers[0].Wq.Rows)
	blas.QMatVecMul(cpuQ, m.Layers[0].Wq, cpuNorm)
	fmt.Printf("CPU Q L2: %.6f first5: %v\n", l2norm(cpuQ), cpuQ[:5])

	// GPU: upload x, rmsnorm, matvec Wq
	rs := gpuPipe.RunState
	gpu.BeginBatch()
	gpu.UploadF32(rs.X, xCPU)
	gpu.Barrier()
	gpu.RMSNorm(rs.XNorm, rs.X, gpuPipe.GpuModel.Layers[0].AttnNorm, dim, cfg.RMSNormEps)
	gpu.Barrier()

	// Download XNorm to check
	gpuNorm := make([]float32, dim)
	gpu.EndBatch()
	gpu.Sync()
	gpu.DownloadF32(rs.XNorm, gpuNorm)
	fmt.Printf("GPU attn_norm L2: %.6f\n", l2norm(gpuNorm))

	// Check RMSNorm match
	normDiff := maxAbsDiff(cpuNorm, gpuNorm)
	fmt.Printf("RMSNorm max diff: %.8f\n", normDiff)

	// GPU: Q = Wq * XNorm
	gl := &gpuPipe.GpuModel.Layers[0]
	gpu.BeginBatch()
	gpu.MatVec(rs.Q, gl.Wq.Buf, rs.XNorm, gl.Wq.Rows, gl.Wq.Cols, gl.Wq.Type)
	gpu.EndBatch()
	gpu.Sync()

	gpuQ := make([]float32, m.Layers[0].Wq.Rows)
	gpu.DownloadF32(rs.Q, gpuQ)
	fmt.Printf("GPU Q L2: %.6f first5: %v\n", l2norm(gpuQ), gpuQ[:5])

	// Compare
	qDiff := maxAbsDiff(cpuQ, gpuQ)
	qRelDiff := maxRelDiff(cpuQ, gpuQ)
	fmt.Printf("Q max abs diff: %.8f, max rel diff: %.8f\n", qDiff, qRelDiff)

	// Check if GPU Q has any NaN/Inf
	nanCount, infCount := 0, 0
	for _, v := range gpuQ {
		if math.IsNaN(float64(v)) {
			nanCount++
		}
		if math.IsInf(float64(v), 0) {
			infCount++
		}
	}
	fmt.Printf("GPU Q: %d NaN, %d Inf out of %d\n", nanCount, infCount, len(gpuQ))

	// Also test CPU MoE router matvec
	if m.Layers[0].FFNRouter != nil {
		fmt.Println("\n--- MoE Router test ---")
		cpuFFNNorm := make([]float32, dim)
		ops.RMSNorm(cpuFFNNorm, xCPU, m.Layers[0].FFNNorm, cfg.RMSNormEps)
		fmt.Printf("CPU ffn_norm L2: %.6f\n", l2norm(cpuFFNNorm))

		cpuRouterOut := make([]float32, m.Layers[0].FFNRouter.Rows)
		blas.QMatVecMul(cpuRouterOut, m.Layers[0].FFNRouter, cpuFFNNorm)
		fmt.Printf("CPU router out L2: %.6f first5: %v\n", l2norm(cpuRouterOut), cpuRouterOut[:5])

		// Allocate temp buffer for router output
		routerBuf := gpu.Alloc(uint64(m.Layers[0].FFNRouter.Rows * 4))
		defer gpu.Free(routerBuf)

		gpu.BeginBatch()
		gpu.RMSNorm(rs.FFNNorm, rs.X, gpuPipe.GpuModel.Layers[0].FFNNorm, dim, cfg.RMSNormEps)
		gpu.Barrier()
		gpu.MatVec(routerBuf, gl.FFNRouter.Buf, rs.FFNNorm, gl.FFNRouter.Rows, gl.FFNRouter.Cols, gl.FFNRouter.Type)
		gpu.EndBatch()
		gpu.Sync()

		gpuRouterOut := make([]float32, m.Layers[0].FFNRouter.Rows)
		gpu.DownloadF32(routerBuf, gpuRouterOut)
		fmt.Printf("GPU router out L2: %.6f first5: %v\n", l2norm(gpuRouterOut), gpuRouterOut[:5])

		routerDiff := maxAbsDiff(cpuRouterOut, gpuRouterOut)
		fmt.Printf("Router max abs diff: %.8f\n", routerDiff)

		// Test expert gate matvec too
		fmt.Println("\n--- Expert gate matvec test ---")
		expIdx := 0
		routerType := gl.FFNRouter.Type
		fmt.Printf("Router weight type: %d (IQ3_XXS=18, IQ4_XS=23)\n", routerType)
		if gl.FFNGateExps != nil {
			fmt.Printf("Expert gate weight type: %d, rows=%d, cols=%d\n",
				gl.FFNGateExps.Type, gl.FFNGateExps.Rows, gl.FFNGateExps.Cols)
		}
		_ = expIdx
	}
}

func l2norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum / float64(len(v)))
}

func maxAbsDiff(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var m float64
	for i := 0; i < n; i++ {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

func maxRelDiff(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var m float64
	for i := 0; i < n; i++ {
		denom := math.Max(math.Abs(float64(a[i])), math.Abs(float64(b[i])))
		if denom < 1e-10 {
			continue
		}
		d := math.Abs(float64(a[i])-float64(b[i])) / denom
		if d > m {
			m = d
		}
	}
	return m
}
