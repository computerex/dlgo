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
func UploadTensorAffine8(*core.QuantizedTensor) (*GpuTensor, error) { return nil, errNoGPU }
func UploadTensorAffine4(*core.QuantizedTensor) (*GpuTensor, error) { return nil, errNoGPU }
func Affine8BytesFor(rows, cols int) uint64                  { return uint64(rows) * uint64(cols) / 32 * 36 }
func Affine4BytesFor(rows, cols int) uint64                  { return uint64(rows) * uint64(cols) / 32 * 20 }
func UploadF32Slice([]float32) (Buf, error)                  { return 0, errNoGPU }
func NewGpuRunState(_, _, _, _, _ int) (*GpuRunState, error)  { return nil, errNoGPU }
func NewGpuKVCache(_, _, _, _ int, _ []bool) (*GpuKVCache, error) { return nil, errNoGPU }
func NewGpuKVCacheCaps(_, _, _, _ int, _ []bool, _ []int) (*GpuKVCache, error) {
	return nil, errNoGPU
}
func (c *GpuKVCache) RingCap(_ int) int { return 0 }
func (c *GpuKVCache) KVBytes() uint64   { return 0 }
func (c *GpuKVCache) RingLayers() int   { return 0 }
func (c *GpuKVCache) Reset()                                  {}
func (gm *GpuModel) FreeAll()                                 {}
func (rs *GpuRunState) FreeAll()                               {}
func (c *GpuKVCache) FreeAll()                                 {}
