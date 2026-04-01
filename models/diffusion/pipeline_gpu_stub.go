//go:build !cgo || !vulkan

package diffusion

import "fmt"

// setupDiffusionGPU is a stub when GPU support is not compiled in.
func setupDiffusionGPU(
	dit *DiTModel,
	rs *DiTRunState,
	vae *VAEDecoder,
	cfg ImageGenConfig,
	context []float32,
	contextLen int,
	latentH, latentW int,
	maxSeqLen int,
) (func(), func([]float32, float32) []float32, func([]float32, int, int) []float32, error) {
	if cfg.UseGPU {
		return nil, nil, nil, fmt.Errorf("GPU support not compiled in (build with -tags 'cgo vulkan')")
	}
	return nil, nil, nil, nil
}
