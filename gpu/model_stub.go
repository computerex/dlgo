//go:build !vulkan || !cgo

package gpu

import "github.com/computerex/dlgo/core"

type GpuTensor struct {
	Buf  Buf
	Type uint32
	Rows int
	Cols int
}

type GpuLayer struct{}
type GpuModel struct {
	TokenEmbed   *GpuTensor
	OutputNorm   Buf
	OutputNormBias Buf
	Output       *GpuTensor
	OutputBias   Buf
	Layers       []GpuLayer
	NumGPULayers int
}
type GpuRunState struct{}
type GpuKVCache struct{}

func UploadTensor(*core.QuantizedTensor) (*GpuTensor, error) { return nil, errNoGPU }
func UploadF32Slice([]float32) (Buf, error)                  { return 0, errNoGPU }
func NewGpuRunState(_, _, _, _, _ int) (*GpuRunState, error)  { return nil, errNoGPU }
func NewGpuKVCache(_, _, _, _ int, _ []bool) (*GpuKVCache, error) { return nil, errNoGPU }
func (c *GpuKVCache) Reset()                                  {}
func (gm *GpuModel) FreeAll()                                 {}
func (rs *GpuRunState) FreeAll()                               {}
func (c *GpuKVCache) FreeAll()                                 {}
