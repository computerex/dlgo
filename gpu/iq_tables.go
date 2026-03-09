//go:build cgo && vulkan

package gpu

/*
#include "csrc/vulkan_gpu.h"
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/computerex/dlgo/quant"
)

var iqTablesOnce sync.Once

func InitIQTables() error {
	var initErr error
	iqTablesOnce.Do(func() {
		iq1sBuf, err := uploadGrid64(quant.IQ1sGrid())
		if err != nil {
			initErr = err
			return
		}

		iq2xxsBuf, err := uploadIQ2XXSTables()
		if err != nil {
			initErr = err
			return
		}

		iq2sBuf, err := uploadGrid64(quant.IQ2sGrid())
		if err != nil {
			initErr = err
			return
		}

		iq3xxsBuf, err := uploadIQ3XXSTables()
		if err != nil {
			initErr = err
			return
		}

		iq3sBuf, err := uploadGrid32(quant.IQ3sGrid())
		if err != nil {
			initErr = err
			return
		}

		C.gpu_set_iq_tables(C.GpuBuf(iq1sBuf), C.GpuBuf(iq2xxsBuf), C.GpuBuf(iq2sBuf),
			C.GpuBuf(iq3xxsBuf), C.GpuBuf(iq3sBuf))
	})
	return initErr
}

func uploadGrid64(grid []uint64) (Buf, error) {
	data := make([]uint32, len(grid)*2)
	for i, v := range grid {
		data[i*2] = uint32(v)
		data[i*2+1] = uint32(v >> 32)
	}
	return uploadU32Slice(data)
}

func uploadIQ2XXSTables() (Buf, error) {
	grid := quant.IQ2xxsGrid()
	ksigns := quant.KSignsIQ2xs()

	nGrid := len(grid) * 2
	nKsigns := (len(ksigns) + 3) / 4
	data := make([]uint32, nGrid+nKsigns)

	for i, v := range grid {
		data[i*2] = uint32(v)
		data[i*2+1] = uint32(v >> 32)
	}

	for i := 0; i < len(ksigns); i += 4 {
		var w uint32
		for k := 0; k < 4 && i+k < len(ksigns); k++ {
			w |= uint32(ksigns[i+k]) << (k * 8)
		}
		data[nGrid+i/4] = w
	}

	return uploadU32Slice(data)
}

func uploadGrid32(grid []uint32) (Buf, error) {
	return uploadU32Slice(grid)
}

func uploadIQ3XXSTables() (Buf, error) {
	grid := quant.IQ3xxsGrid()
	ksigns := quant.KSignsIQ2xs()

	nGrid := len(grid)
	nKsigns := (len(ksigns) + 3) / 4
	data := make([]uint32, nGrid+nKsigns)

	copy(data, grid)

	for i := 0; i < len(ksigns); i += 4 {
		var w uint32
		for k := 0; k < 4 && i+k < len(ksigns); k++ {
			w |= uint32(ksigns[i+k]) << (k * 8)
		}
		data[nGrid+i/4] = w
	}

	return uploadU32Slice(data)
}

func uploadU32Slice(data []uint32) (Buf, error) {
	sizeBytes := uint64(len(data)) * 4
	buf := Alloc(sizeBytes)
	if buf == 0 {
		return 0, fmt.Errorf("gpu: failed to alloc %d bytes for IQ table", sizeBytes)
	}
	src := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), sizeBytes)
	if err := Upload(buf, src); err != nil {
		Free(buf)
		return 0, err
	}
	return buf, nil
}
