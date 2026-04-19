//go:build cgo && vulkan

package diffusion

import (
	"fmt"
	"log"
	"time"

	"github.com/computerex/dlgo/gpu"
)

// setupDiffusionGPU initializes GPU resources for diffusion if cfg.UseGPU is true.
// Returns (cleanup func, model callback, prepareVAE func, vae decode func, error).
// VAE scratch buffers are allocated lazily via prepareVAE, which also frees DiT
// resources first. This allows 1024x1024 generation on 16 GB cards.
// If GPU is not requested, returns (nil, nil, nil, nil, nil) and the caller uses CPU.
func setupDiffusionGPU(
	dit *DiTModel,
	rs *DiTRunState,
	vae *VAEDecoder,
	cfg ImageGenConfig,
	context []float32,
	contextLen int,
	latentH, latentW int,
	maxSeqLen int,
) (cleanup func(), modelFn func([]float32, float32) []float32,
	prepareVAE func() error,
	vaeFn func([]float32, int, int) []float32, err error) {
	if !cfg.UseGPU {
		return nil, nil, nil, nil, nil
	}

	log.Println("[diffusion/gpu] Initializing GPU...")
	gpuStart := time.Now()
	if err := gpu.Init(); err != nil {
		return nil, nil, nil, nil, fmt.Errorf("GPU init: %w", err)
	}
	log.Printf("[diffusion/gpu] GPU: %s (%.0f MB VRAM)", gpu.DeviceName(),
		float64(gpu.VRAMBytes())/(1024*1024))

	// Upload DiT weights to GPU
	log.Println("[diffusion/gpu] Uploading DiT weights to GPU...")
	gm, err := UploadDiTModel(dit)
	if err != nil {
		gpu.Shutdown()
		return nil, nil, nil, nil, fmt.Errorf("upload DiT: %w", err)
	}

	// Allocate DiT GPU run state
	grs, err := NewGpuDiTRunState(dit.Config, maxSeqLen)
	if err != nil {
		gpu.Shutdown()
		return nil, nil, nil, nil, fmt.Errorf("GPU run state: %w", err)
	}

	// Upload VAE weights to GPU (small, ~189 MB — keep resident)
	log.Println("[diffusion/gpu] Uploading VAE weights to GPU...")
	gvm, err := UploadVAEModel(vae)
	if err != nil {
		gpu.Shutdown()
		return nil, nil, nil, nil, fmt.Errorf("upload VAE: %w", err)
	}

	log.Printf("[diffusion/gpu] GPU setup complete in %v (VRAM used: %.1f MB)",
		time.Since(gpuStart), float64(gpu.AllocatedBytes())/(1024*1024))

	// VAE scratch buffers — allocated lazily via prepareVAE
	var gvrs *GpuVAERunState
	ditFreed := false

	freeDiTResources := func() {
		if ditFreed {
			return
		}
		ditFreed = true
		log.Println("[diffusion/gpu] Freeing DiT resources for VAE...")
		beforeMB := float64(gpu.AllocatedBytes()) / (1024 * 1024)

		// DiT run state buffers (pool-backed: Free just destroys VkBuffer handles)
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
		if grs.OnesBuf != 0 {
			gpu.Free(grs.OnesBuf)
		}
		if grs.PE != 0 {
			gpu.Free(grs.PE)
		}
		// Release the pool memory (single vkFreeMemory for all RunState buffers)
		gpu.PoolDestroy()

		// DiT layer weight buffers
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

		afterMB := float64(gpu.AllocatedBytes()) / (1024 * 1024)
		log.Printf("[diffusion/gpu] Freed %.1f MB DiT resources (%.1f MB -> %.1f MB)",
			beforeMB-afterMB, beforeMB, afterMB)
	}

	prepareVAE = func() error {
		freeDiTResources()
		log.Println("[diffusion/gpu] Allocating VAE scratch buffers...")
		var allocErr error
		gvrs, allocErr = NewGpuVAERunState(vae.Config, latentH, latentW)
		if allocErr != nil {
			return fmt.Errorf("VAE GPU run state: %w", allocErr)
		}
		log.Printf("[diffusion/gpu] VAE ready (VRAM used: %.1f MB)",
			float64(gpu.AllocatedBytes())/(1024*1024))
		return nil
	}

	cleanup = func() {
		log.Println("[diffusion/gpu] Freeing GPU resources...")
		freeDiTResources()

		// VAE scratch buffers (may be nil if prepareVAE was never called)
		if gvrs != nil {
			gpu.Free(gvrs.Act)
			gpu.Free(gvrs.Tmp1)
			gpu.Free(gvrs.Tmp2)
			gpu.Free(gvrs.Ups)
			gpu.Free(gvrs.AttnQ)
			gpu.Free(gvrs.AttnK)
			gpu.Free(gvrs.AttnV)
			gpu.Free(gvrs.AttnOut)
		}

		gpu.Shutdown()
		log.Println("[diffusion/gpu] GPU resources freed")
	}

	modelFn = func(x []float32, timestep float32) []float32 {
		return GpuDiTForward(dit, gm, rs, grs, x, timestep, context, contextLen, latentH, latentW, cfg.Debug)
	}

	vaeFn = func(latent []float32, h, w int) []float32 {
		return GpuVAEDecode(vae, gvm, gvrs, latent, h, w)
	}

	return cleanup, modelFn, prepareVAE, vaeFn, nil
}
