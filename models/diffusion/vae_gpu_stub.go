//go:build !cgo || !vulkan

package diffusion

import "fmt"

var errNoVAEGPU = fmt.Errorf("GPU VAE support not compiled in (build with -tags 'cgo vulkan')")

type GpuVAEModel struct{}
type GpuVAERunState struct{}

func UploadVAEModel(d *VAEDecoder) (*GpuVAEModel, error) {
	return nil, errNoVAEGPU
}

func NewGpuVAERunState(cfg VAEConfig, latentH, latentW int) (*GpuVAERunState, error) {
	return nil, errNoVAEGPU
}

func GpuVAEDecode(d *VAEDecoder, gm *GpuVAEModel, grs *GpuVAERunState,
	latent []float32, H, W int) []float32 {
	return nil
}
