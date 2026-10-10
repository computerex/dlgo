package llm

import (
	"math"
	"sync"
	"sync/atomic"

	"github.com/computerex/dlgo/ops"
)

// WorkerPool manages persistent goroutines for parallel inference.
// Each worker has its own task channel; work is distributed round-robin.
type WorkerPool struct {
	numWorkers int
	taskChs    []chan func()
	wg         sync.WaitGroup
	asyncWg    sync.WaitGroup
	alive      atomic.Bool
}

// NewWorkerPool creates a pool with n persistent workers.
// Callers must call Shutdown when done.
func NewWorkerPool(n int) *WorkerPool {
	if n < 1 {
		n = 1
	}
	wp := &WorkerPool{
		numWorkers: n,
		taskChs:    make([]chan func(), n),
	}
	wp.alive.Store(true)
	for i := 0; i < n; i++ {
		ch := make(chan func(), 1)
		wp.taskChs[i] = ch
		wp.wg.Add(1)
		go wp.worker(i, ch)
	}
	return wp
}

func (wp *WorkerPool) worker(id int, ch chan func()) {
	defer wp.wg.Done()
	for fn := range ch {
		if fn != nil {
			fn()
		}
	}
}

// Dispatch distributes work across numActive workers and blocks until complete.
// total is the total number of items; work(workerID, start, end) processes [start, end).
func (wp *WorkerPool) Dispatch(total, numActive int, work func(workerID, start, end int)) {
	if total <= 0 || numActive <= 0 || !wp.alive.Load() {
		return
	}
	if numActive > wp.numWorkers {
		numActive = wp.numWorkers
	}
	if numActive > total {
		numActive = total
	}

	chunk := (total + numActive - 1) / numActive
	var done sync.WaitGroup
	done.Add(numActive)

	for w := 0; w < numActive; w++ {
		start := w * chunk
		end := start + chunk
		if start >= total {
			done.Done()
			continue
		}
		if end > total {
			end = total
		}
		workerID := w
		s, e := start, end
		task := func() {
			defer done.Done()
			work(workerID, s, e)
		}
		ch := wp.taskChs[w%wp.numWorkers]
		ch <- task
	}

	done.Wait()
}

// DispatchAsync distributes work without waiting.
// Call Wait() to block until all async work completes.
func (wp *WorkerPool) DispatchAsync(total, numActive int, work func(workerID, start, end int)) {
	if total <= 0 || numActive <= 0 || !wp.alive.Load() {
		return
	}
	if numActive > wp.numWorkers {
		numActive = wp.numWorkers
	}
	if numActive > total {
		numActive = total
	}

	chunk := (total + numActive - 1) / numActive
	wp.asyncWg.Add(numActive)

	for w := 0; w < numActive; w++ {
		start := w * chunk
		end := start + chunk
		if start >= total {
			wp.asyncWg.Done()
			continue
		}
		if end > total {
			end = total
		}
		workerID := w
		s, e := start, end
		task := func() {
			defer wp.asyncWg.Done()
			work(workerID, s, e)
		}
		ch := wp.taskChs[w%wp.numWorkers]
		select {
		case ch <- task:
		default:
			// Channel full; run synchronously to avoid deadlock
			task()
		}
	}
}

// Wait blocks until all work dispatched via DispatchAsync has completed.
func (wp *WorkerPool) Wait() {
	wp.asyncWg.Wait()
}

// Shutdown stops all workers and waits for them to exit.
func (wp *WorkerPool) Shutdown() {
	wp.alive.Store(false)
	for _, ch := range wp.taskChs {
		close(ch)
	}
	wp.wg.Wait()
}

// PrecomputeRoPE fills RunState with precomputed cos/sin tables for RoPE.
// maxSeqLen: maximum sequence length; ropeDim, headDim: dimensions; freqBase: RoPE frequency base.
// ropeDim may be less than headDim for partial RoPE (e.g. Phi-2: 32 of 80).
func (rs *RunState) PrecomputeRoPE(maxSeqLen, ropeDim, headDim int, freqBase float32) {
	if maxSeqLen <= 0 || headDim <= 0 {
		return
	}
	if ropeDim <= 0 || ropeDim > headDim {
		ropeDim = headDim
	}
	half := ropeDim / 2
	if half <= 0 {
		return
	}
	rs.ropeHeadDim = headDim
	rs.ropeDim = ropeDim
	rs.ropeNeox = false
	rs.ropeCos = make([]float32, maxSeqLen*half)
	rs.ropeSin = make([]float32, maxSeqLen*half)
	for pos := 0; pos < maxSeqLen; pos++ {
		for i := 0; i < half; i++ {
			theta := 1.0 / math.Pow(float64(freqBase), float64(2*i)/float64(ropeDim))
			angle := float64(pos) * theta
			rs.ropeCos[pos*half+i] = float32(math.Cos(angle))
			rs.ropeSin[pos*half+i] = float32(math.Sin(angle))
		}
	}
}

// PrecomputeYaRNRoPE precomputes RoPE tables with YaRN extended-context scaling.
func (rs *RunState) PrecomputeYaRNRoPE(maxSeqLen, ropeDim, headDim int,
	freqBase, factor float32, origMaxPos int, betaFast, betaSlow, extFactor, attnFactor float32) {
	if maxSeqLen <= 0 || headDim <= 0 {
		return
	}
	if ropeDim <= 0 || ropeDim > headDim {
		ropeDim = headDim
	}
	half := ropeDim / 2
	if half <= 0 {
		return
	}
	rs.ropeHeadDim = headDim
	rs.ropeDim = ropeDim
	rs.ropeNeox = false
	rs.ropeCos, rs.ropeSin = ops.YaRNRoPEFrequencyTable(maxSeqLen, ropeDim, freqBase,
		factor, origMaxPos, betaFast, betaSlow, extFactor, attnFactor)
}

// SetRopeNeox sets whether to use NeoX-style (split-half) RoPE pairing.
// Call after PrecomputeRoPE if needed.
func (rs *RunState) SetRopeNeox(neox bool) {
	rs.ropeNeox = neox
}

// RoPETables returns the precomputed cos/sin tables for GPU upload.
func (rs *RunState) RoPETables() (cosTable, sinTable []float32) {
	return rs.ropeCos, rs.ropeSin
}

// RoPETablesSWA returns the SWA-specific RoPE tables (nil if not set).
func (rs *RunState) RoPETablesSWA() (cosTable, sinTable []float32) {
	return rs.ropeCosSWA, rs.ropeSinSWA
}

// PrecomputeSWARoPE builds separate cos/sin tables for SWA layers with a
// different frequency base (e.g. gpt-oss ISWA uses theta=10000 for SWA,
// theta=150000 for full attention).
func (rs *RunState) PrecomputeSWARoPE(maxSeqLen, ropeDim, headDim int, freqBase float32) {
	if maxSeqLen <= 0 || headDim <= 0 {
		return
	}
	if ropeDim <= 0 || ropeDim > headDim {
		ropeDim = headDim
	}
	half := ropeDim / 2
	if half <= 0 {
		return
	}
	rs.ropeCosSWA = make([]float32, maxSeqLen*half)
	rs.ropeSinSWA = make([]float32, maxSeqLen*half)
	for pos := 0; pos < maxSeqLen; pos++ {
		for i := 0; i < half; i++ {
			theta := 1.0 / math.Pow(float64(freqBase), float64(2*i)/float64(ropeDim))
			angle := float64(pos) * theta
			rs.ropeCosSWA[pos*half+i] = float32(math.Cos(angle))
			rs.ropeSinSWA[pos*half+i] = float32(math.Sin(angle))
		}
	}
}

// applyRoPEWithTables applies RoPE using the given cos/sin tables.
func (rs *RunState) applyRoPEWithTables(vec []float32, pos int, cos, sin []float32) {
	if cos == nil || sin == nil || len(vec) < rs.ropeHeadDim {
		return
	}
	rd := rs.ropeDim
	if rd <= 0 {
		rd = rs.ropeHeadDim
	}
	half := rd / 2
	base := pos * half
	if base+half > len(cos) {
		return
	}
	if rs.ropeNeox {
		for i := 0; i < half; i++ {
			c := cos[base+i]
			s := sin[base+i]
			v0 := vec[i]
			v1 := vec[i+half]
			vec[i] = v0*c - v1*s
			vec[i+half] = v0*s + v1*c
		}
	} else {
		for i := 0; i < half; i++ {
			c := cos[base+i]
			s := sin[base+i]
			v0 := vec[2*i]
			v1 := vec[2*i+1]
			vec[2*i] = v0*c - v1*s
			vec[2*i+1] = v0*s + v1*c
		}
	}
}

// EnableIMRoPE turns on interleaved multi-axis RoPE (Qwen3VL/Qwen3.5 IMROPE).
// sections is the {T,H,W,E} pair-count vector; freqBase is the RoPE theta.
func (rs *RunState) EnableIMRoPE(sections [4]int, freqBase float32) {
	rs.useMRoPE = true
	rs.mropeSections = sections
	rs.mropeFreq = freqBase
}

// SetMRoPEPos sets the per-axis positions for the next forward token.
func (rs *RunState) SetMRoPEPos(t, h, w, e int) {
	rs.mropePos = [4]int{t, h, w, e}
}

// ClearMRoPEPos restores scalar RoPE behaviour (text tokens using the shared
// sequence counter) until the next SetMRoPEPos.
func (rs *RunState) ClearMRoPEPos() {
	rs.mropePos = [4]int{-1, -1, -1, -1}
}

// SetLogicalPos sets the RoPE position for the next forward token when it
// differs from its KV slot (Qwen M-RoPE: image blocks store n slots but the
// logical counter advances by max(nx,ny)). -1 restores slot==position.
func (rs *RunState) SetLogicalPos(p int) {
	rs.logicalPos = p
}

// ropePos returns the position to use for rotary embedding: the logical
// position when set, otherwise the KV slot.
func (rs *RunState) ropePos(slot int) int {
	if rs.logicalPos >= 0 {
		return rs.logicalPos
	}
	return slot
}

// applyIMRoPE applies Qwen interleaved multi-axis RoPE to a single head using
// rs.mropePos. It mirrors ggml compute_imrope_cache (rope_f32.c) exactly:
// pairs use NEOX half-split pairing (i, i+half) where half = ropeDim/2; each
// pair index i selects its axis via the mod-3 sector pattern; all axis angles
// advance by theta_scale = freqBase^(-2/ropeDim) per pair.
func (rs *RunState) applyIMRoPE(vec []float32) {
	rd := rs.ropeDim
	if rd <= 0 {
		rd = rs.ropeHeadDim
	}
	if rd <= 0 || len(vec) < rd {
		return
	}
	half := rd / 2
	if rs.mropePos[0] < 0 {
		return
	}
	thetaScale := math.Pow(float64(rs.mropeFreq), -2.0/float64(rd))
	thetaT := float64(rs.mropePos[0])
	thetaH := float64(rs.mropePos[1])
	thetaW := float64(rs.mropePos[2])
	thetaE := float64(rs.mropePos[3])
	sectDims := rs.mropeSections[0] + rs.mropeSections[1] + rs.mropeSections[2] + rs.mropeSections[3]
	if sectDims <= 0 {
		return
	}
	for i := 0; i < half; i++ {
		sector := i % sectDims
		var theta float64
		switch {
		case sector%3 == 1 && sector < 3*rs.mropeSections[1]:
			theta = thetaH
		case sector%3 == 2 && sector < 3*rs.mropeSections[2]:
			theta = thetaW
		case sector%3 == 0 && sector < 3*rs.mropeSections[0]:
			theta = thetaT
		default:
			theta = thetaE
		}
		c := float32(math.Cos(theta))
		s := float32(math.Sin(theta))
		v0 := vec[i]
		v1 := vec[i+half]
		vec[i] = v0*c - v1*s
		vec[i+half] = v0*s + v1*c
		thetaT *= thetaScale
		thetaH *= thetaScale
		thetaW *= thetaScale
		thetaE *= thetaScale
	}
}

// ApplyRoPEFast applies precomputed RoPE to vec in-place.
// vec must be one head's worth [headDim]; pos is the token position.
// Only the first ropeDim dimensions are rotated; the rest pass through unchanged.
func (rs *RunState) ApplyRoPEFast(vec []float32, pos int) {
	if rs.ropeCos == nil || rs.ropeSin == nil || len(vec) < rs.ropeHeadDim {
		return
	}
	rd := rs.ropeDim
	if rd <= 0 {
		rd = rs.ropeHeadDim
	}
	half := rd / 2
	base := pos * half
	if base+half > len(rs.ropeCos) {
		return
	}

	if rs.ropeNeox {
		for i := 0; i < half; i++ {
			cos := rs.ropeCos[base+i]
			sin := rs.ropeSin[base+i]
			x0 := vec[i]
			x1 := vec[i+half]
			vec[i] = x0*cos - x1*sin
			vec[i+half] = x0*sin + x1*cos
		}
	} else {
		for i := 0; i < half; i++ {
			cos := rs.ropeCos[base+i]
			sin := rs.ropeSin[base+i]
			x0 := vec[2*i]
			x1 := vec[2*i+1]
			vec[2*i] = x0*cos - x1*sin
			vec[2*i+1] = x0*sin + x1*cos
		}
	}
}

// ApplyRoPEFastDim applies RoPE to a vector of exactly ropeDim elements
// (e.g. the rope portion of MLA Q/K heads).
func (rs *RunState) ApplyRoPEFastDim(vec []float32, pos, ropeDim int) {
	if rs.ropeCos == nil || rs.ropeSin == nil || len(vec) < ropeDim {
		return
	}
	half := ropeDim / 2
	base := pos * half
	if base+half > len(rs.ropeCos) {
		return
	}

	if rs.ropeNeox {
		for i := 0; i < half; i++ {
			cos := rs.ropeCos[base+i]
			sin := rs.ropeSin[base+i]
			x0 := vec[i]
			x1 := vec[i+half]
			vec[i] = x0*cos - x1*sin
			vec[i+half] = x0*sin + x1*cos
		}
	} else {
		for i := 0; i < half; i++ {
			cos := rs.ropeCos[base+i]
			sin := rs.ropeSin[base+i]
			x0 := vec[2*i]
			x1 := vec[2*i+1]
			vec[2*i] = x0*cos - x1*sin
			vec[2*i+1] = x0*sin + x1*cos
		}
	}
}
