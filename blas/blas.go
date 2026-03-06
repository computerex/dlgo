// Package blas provides quantized linear algebra operations for neural network inference.
//
// It bridges the core.QuantizedTensor type with the quant package's dequantization and
// fused dot product routines, providing high-level matrix-vector and batch GEMM operations.
package blas

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/ops"
	"github.com/computerex/dlgo/quant"
)

// Pool manages persistent goroutines for parallel matrix operations.
type Pool struct {
	numWorkers int
	taskChs    []chan func()
	wg         sync.WaitGroup
	alive      atomic.Bool
}

var defaultPool *Pool
var defaultPoolOnce sync.Once

// DefaultPool returns a shared worker pool sized to the number of CPUs.
func DefaultPool() *Pool {
	defaultPoolOnce.Do(func() {
		n := runtime.NumCPU()
		if n < 1 {
			n = 1
		}
		defaultPool = NewPool(n)
	})
	return defaultPool
}

// NewPool creates a worker pool with n persistent goroutines.
func NewPool(n int) *Pool {
	if n < 1 {
		n = 1
	}
	p := &Pool{
		numWorkers: n,
		taskChs:    make([]chan func(), n),
	}
	p.alive.Store(true)
	for i := 0; i < n; i++ {
		ch := make(chan func(), 1)
		p.taskChs[i] = ch
		p.wg.Add(1)
		go func(c chan func()) {
			defer p.wg.Done()
			for fn := range c {
				fn()
			}
		}(ch)
	}
	return p
}

func (p *Pool) dispatch(total, numActive int, work func(start, end int)) {
	if total <= 0 || numActive <= 0 || !p.alive.Load() {
		return
	}
	if numActive > p.numWorkers {
		numActive = p.numWorkers
	}
	if numActive > total {
		numActive = total
	}
	chunk := (total + numActive - 1) / numActive
	var done sync.WaitGroup

	for w := 0; w < numActive; w++ {
		s := w * chunk
		e := s + chunk
		if s >= total {
			break
		}
		if e > total {
			e = total
		}
		done.Add(1)
		start, end := s, e
		p.taskChs[w%p.numWorkers] <- func() {
			defer done.Done()
			work(start, end)
		}
	}
	done.Wait()
}

// Shutdown stops all workers.
func (p *Pool) Shutdown() {
	if p.alive.CompareAndSwap(true, false) {
		for _, ch := range p.taskChs {
			close(ch)
		}
		p.wg.Wait()
	}
}

// QMatVecMul performs quantized matrix-vector multiply (single-threaded).
//
//	out[r] = dot(qt[r, :], x)   for r in [0, qt.Rows)
func QMatVecMul(out []float32, qt *core.QuantizedTensor, x []float32) {
	if qt.FP32Data != nil {
		ops.MatVecMul(out[:qt.Rows], qt.FP32Data, x, qt.Rows, qt.Cols)
		return
	}

	bytesPerRow := quant.BytesForN(qt.Type, qt.Cols)
	useFused := quant.HasSIMDDot(qt.Type)

	if useFused {
		quant.SIMDDotBatch(qt.Data, qt.Type, x, qt.Cols, out[:qt.Rows], qt.Rows, bytesPerRow)
	} else {
		buf := make([]float32, qt.Cols)
		for r := 0; r < qt.Rows; r++ {
			rowData := qt.Data[r*bytesPerRow : (r+1)*bytesPerRow]
			quant.DequantizeInto(buf, rowData, qt.Type, qt.Cols)
			out[r] = quant.SIMDDotF32(buf, x, qt.Cols)
		}
	}
}

// QMatVecMulParallel performs parallel quantized matrix-vector multiply.
// Splits rows across pool workers for large matrices.
func QMatVecMulParallel(out []float32, qt *core.QuantizedTensor, x []float32, pool *Pool) {
	if pool == nil || qt.Rows < 256 {
		QMatVecMul(out, qt, x)
		return
	}

	if qt.FP32Data != nil {
		pool.dispatch(qt.Rows, pool.numWorkers, func(start, end int) {
			quant.SIMDDotF32Batch(qt.FP32Data[start*qt.Cols:end*qt.Cols], x, qt.Cols, out[start:end], end-start)
		})
		return
	}

	bytesPerRow := quant.BytesForN(qt.Type, qt.Cols)
	useFused := quant.HasSIMDDot(qt.Type)

	if useFused {
		pool.dispatch(qt.Rows, pool.numWorkers, func(start, end int) {
			nrows := end - start
			startByte := start * bytesPerRow
			endByte := end * bytesPerRow
			quant.SIMDDotBatch(qt.Data[startByte:endByte], qt.Type, x, qt.Cols, out[start:end], nrows, bytesPerRow)
		})
	} else {
		pool.dispatch(qt.Rows, pool.numWorkers, func(start, end int) {
			buf := make([]float32, qt.Cols)
			for r := start; r < end; r++ {
				rowData := qt.Data[r*bytesPerRow : (r+1)*bytesPerRow]
				quant.DequantizeInto(buf, rowData, qt.Type, qt.Cols)
				out[r] = quant.SIMDDotF32(buf, x, qt.Cols)
			}
		})
	}
}

// QDualMatVecMulParallel runs two matrix-vector multiplies concurrently.
// Used for FFN gate+up which share the same input.
func QDualMatVecMulParallel(out1 []float32, qt1 *core.QuantizedTensor,
	out2 []float32, qt2 *core.QuantizedTensor, x []float32, pool *Pool) {
	if pool == nil {
		QMatVecMul(out1, qt1, x)
		QMatVecMul(out2, qt2, x)
		return
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		QMatVecMulParallel(out1, qt1, x, pool)
	}()
	go func() {
		defer wg.Done()
		QMatVecMulParallel(out2, qt2, x, pool)
	}()
	wg.Wait()
}

// QBatchGEMM performs batch quantized matrix multiply.
//   outFlat[p*qt.Rows + r] = dot(qt[r, :], xFlat[p*qt.Cols : (p+1)*qt.Cols])
//
// For each of the nPos input positions, computes the full matrix-vector product.
// outFlat: [nPos * qt.Rows], xFlat: [nPos * qt.Cols].
func QBatchGEMM(outFlat []float32, qt *core.QuantizedTensor, xFlat []float32, nPos int) {
	if nPos <= 0 {
		return
	}

	if qt.FP32Data != nil && nPos >= 4 {
		for p := 0; p < nPos; p++ {
			x := xFlat[p*qt.Cols : (p+1)*qt.Cols]
			out := outFlat[p*qt.Rows : (p+1)*qt.Rows]
			ops.MatVecMul(out, qt.FP32Data, x, qt.Rows, qt.Cols)
		}
		return
	}

	bytesPerRow := quant.BytesForN(qt.Type, qt.Cols)
	useFused := quant.HasSIMDDot(qt.Type)

	if useFused && nPos == 1 {
		quant.SIMDDotBatch(qt.Data, qt.Type, xFlat[:qt.Cols], qt.Cols, outFlat[:qt.Rows], qt.Rows, bytesPerRow)
		return
	}

	for p := 0; p < nPos; p++ {
		x := xFlat[p*qt.Cols : (p+1)*qt.Cols]
		out := outFlat[p*qt.Rows : (p+1)*qt.Rows]
		if useFused {
			quant.SIMDDotBatch(qt.Data, qt.Type, x, qt.Cols, out, qt.Rows, bytesPerRow)
		} else {
			buf := make([]float32, qt.Cols)
			for r := 0; r < qt.Rows; r++ {
				rowData := qt.Data[r*bytesPerRow : (r+1)*bytesPerRow]
				quant.DequantizeInto(buf, rowData, qt.Type, qt.Cols)
				out[r] = quant.SIMDDotF32(buf, x, qt.Cols)
			}
		}
	}
}

// DequantizeAll returns the entire quantized tensor dequantized to float32.
func DequantizeAll(qt *core.QuantizedTensor) []float32 {
	if qt.FP32Data != nil {
		out := make([]float32, len(qt.FP32Data))
		copy(out, qt.FP32Data)
		return out
	}
	n := qt.Rows * qt.Cols
	result, _ := quant.Dequantize(qt.Data, qt.Type, n)
	return result
}

// PreDequantize converts a quantized tensor to FP32 and caches the result.
// Subsequent QMatVecMul / QBatchGEMM calls will use the cached FP32 data.
// The raw quantized bytes are released to save memory.
func PreDequantize(qt *core.QuantizedTensor) {
	if qt == nil || qt.FP32Data != nil {
		return
	}
	qt.FP32Data = DequantizeAll(qt)
	qt.Data = nil
}
