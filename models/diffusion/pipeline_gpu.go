//go:build cgo && vulkan

package diffusion

import (
	"fmt"
	"log"
	"time"

	"github.com/computerex/dlgo/gpu"
)

// setupDiffusionGPU initializes GPU resources for diffusion if cfg.UseGPU is true.
// Returns (cleanup func, model callback, error).
// If GPU is not requested, returns (nil, nil, nil) and the caller uses CPU.
func setupDiffusionGPU(
	dit *DiTModel,
	rs *DiTRunState,
	cfg ImageGenConfig,
	context []float32,
	contextLen int,
	latentH, latentW int,
	maxSeqLen int,
) (cleanup func(), modelFn func([]float32, float32) []float32, err error) {
	if !cfg.UseGPU {
		return nil, nil, nil
	}

	log.Println("[diffusion/gpu] Initializing GPU...")
	gpuStart := time.Now()
	if err := gpu.Init(); err != nil {
		return nil, nil, fmt.Errorf("GPU init: %w", err)
	}
	log.Printf("[diffusion/gpu] GPU: %s (%.0f MB VRAM)", gpu.DeviceName(),
		float64(gpu.VRAMBytes())/(1024*1024))

	// Upload model weights to GPU
	log.Println("[diffusion/gpu] Uploading DiT weights to GPU...")
	gm, err := UploadDiTModel(dit)
	if err != nil {
		gpu.Shutdown()
		return nil, nil, fmt.Errorf("upload DiT: %w", err)
	}

	// Allocate GPU run state
	grs, err := NewGpuDiTRunState(dit.Config, maxSeqLen)
	if err != nil {
		gpu.Shutdown()
		return nil, nil, fmt.Errorf("GPU run state: %w", err)
	}

	log.Printf("[diffusion/gpu] GPU setup complete in %v", time.Since(gpuStart))

	cleanup = func() {
		log.Println("[diffusion/gpu] Freeing GPU resources...")
		// Run state buffers
		gpu.Free(grs.X)
		gpu.Free(grs.XNorm)
		gpu.Free(grs.QKV)
		gpu.Free(grs.Q)
		gpu.Free(grs.K)
		gpu.Free(grs.V)
		gpu.Free(grs.AttnOut)
		gpu.Free(grs.Proj)
		gpu.Free(grs.Gate)
		gpu.Free(grs.Up)
		gpu.Free(grs.Hidden)
		gpu.Free(grs.FFNOut)
		gpu.Free(grs.Residual)
		gpu.Free(grs.Mod)
		gpu.Free(grs.ScaleBuf)
		gpu.Free(grs.GateBuf)
		if grs.PE != 0 {
			gpu.Free(grs.PE)
		}

		// Layer buffers
		for i := range gm.Layers {
			l := &gm.Layers[i]
			if l.AttnQKV != nil {
				gpu.Free(l.AttnQKV.Buf)
			}
			if l.AttnOut != nil {
				gpu.Free(l.AttnOut.Buf)
			}
			if l.FFNGate != nil {
				gpu.Free(l.FFNGate.Buf)
			}
			if l.FFNDown != nil {
				gpu.Free(l.FFNDown.Buf)
			}
			if l.FFNUp != nil {
				gpu.Free(l.FFNUp.Buf)
			}
			if l.QNorm != 0 {
				gpu.Free(l.QNorm)
			}
			if l.KNorm != 0 {
				gpu.Free(l.KNorm)
			}
			if l.AttnNorm1 != 0 {
				gpu.Free(l.AttnNorm1)
			}
			if l.AttnNorm2 != 0 {
				gpu.Free(l.AttnNorm2)
			}
			if l.FFNNorm1 != 0 {
				gpu.Free(l.FFNNorm1)
			}
			if l.FFNNorm2 != 0 {
				gpu.Free(l.FFNNorm2)
			}
			if l.AdaLNWeight != nil {
				gpu.Free(l.AdaLNWeight.Buf)
			}
			if l.AdaLNBias != 0 {
				gpu.Free(l.AdaLNBias)
			}
		}

		gpu.Shutdown()
		log.Println("[diffusion/gpu] GPU resources freed")
	}

	modelFn = func(x []float32, timestep float32) []float32 {
		return GpuDiTForward(dit, gm, rs, grs, x, timestep, context, contextLen, latentH, latentW)
	}

	return cleanup, modelFn, nil
}
