//go:build cgo && vulkan

package gpu

import (
	"math"
	"math/rand"
	"testing"
)

func TestBroadcastMul(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer Shutdown()

	const seqLen = 8
	const dim = 16
	totalN := seqLen * dim

	rng := rand.New(rand.NewSource(42))
	data := make([]float32, totalN)
	scale := make([]float32, dim)
	for i := range data {
		data[i] = rng.Float32()*2 - 1
	}
	for i := range scale {
		scale[i] = rng.Float32()*2 - 1
	}

	// CPU reference: data[i] *= scale[i % dim]
	cpuOut := make([]float32, totalN)
	copy(cpuOut, data)
	for i := range cpuOut {
		cpuOut[i] *= scale[i%dim]
	}

	// GPU
	dataBuf := Alloc(uint64(totalN) * 4)
	defer Free(dataBuf)
	scaleBuf := Alloc(uint64(dim) * 4)
	defer Free(scaleBuf)

	UploadF32(dataBuf, data)
	UploadF32(scaleBuf, scale)

	BeginBatch()
	if err := BroadcastMul(dataBuf, scaleBuf, totalN, dim); err != nil {
		EndBatch()
		t.Fatal(err)
	}
	EndBatch()

	gpuOut := make([]float32, totalN)
	DownloadF32(dataBuf, gpuOut)

	maxErr := compareF32(cpuOut, gpuOut)
	t.Logf("BroadcastMul: MaxErr=%.8f", maxErr)
	if maxErr > 1e-5 {
		t.Errorf("MaxErr too high: %f", maxErr)
	}
}

func TestTanhGateResidual(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer Shutdown()

	const seqLen = 8
	const dim = 16
	totalN := seqLen * dim

	rng := rand.New(rand.NewSource(42))
	residual := make([]float32, totalN)
	data := make([]float32, totalN)
	gate := make([]float32, dim)
	for i := range residual {
		residual[i] = rng.Float32()*2 - 1
	}
	for i := range data {
		data[i] = rng.Float32()*2 - 1
	}
	for i := range gate {
		gate[i] = rng.Float32()*2 - 1
	}

	// CPU reference: out[i] = residual[i] + data[i] * tanh(gate[i % dim])
	cpuOut := make([]float32, totalN)
	for i := range cpuOut {
		cpuOut[i] = residual[i] + data[i]*float32(math.Tanh(float64(gate[i%dim])))
	}

	resBuf := Alloc(uint64(totalN) * 4)
	defer Free(resBuf)
	dataBuf := Alloc(uint64(totalN) * 4)
	defer Free(dataBuf)
	gateBuf := Alloc(uint64(dim) * 4)
	defer Free(gateBuf)
	outBuf := Alloc(uint64(totalN) * 4)
	defer Free(outBuf)

	UploadF32(resBuf, residual)
	UploadF32(dataBuf, data)
	UploadF32(gateBuf, gate)

	BeginBatch()
	if err := TanhGateResidual(outBuf, resBuf, dataBuf, gateBuf, totalN, dim); err != nil {
		EndBatch()
		t.Fatal(err)
	}
	EndBatch()

	gpuOut := make([]float32, totalN)
	DownloadF32(outBuf, gpuOut)

	maxErr := compareF32(cpuOut, gpuOut)
	t.Logf("TanhGateResidual: MaxErr=%.8f", maxErr)
	if maxErr > 1e-5 {
		t.Errorf("MaxErr too high: %f", maxErr)
	}
}

func TestRoPE3D(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer Shutdown()

	const nPos = 4
	const nHeads = 2
	const headDim = 8
	const peOffset = 0
	peStride := headDim * 2
	halfDim := headDim / 2

	rng := rand.New(rand.NewSource(42))

	dim := nHeads * headDim
	vec := make([]float32, nPos*dim)
	for i := range vec {
		vec[i] = rng.Float32()*2 - 1
	}

	// PE table: [nPos, peStride] = [nPos, headDim*2]
	// pe[p*peStride + d*4 + 0] = cos, pe[p*peStride + d*4 + 2] = sin
	pe := make([]float32, nPos*peStride)
	for p := 0; p < nPos; p++ {
		for d := 0; d < halfDim; d++ {
			theta := rng.Float64() * math.Pi * 2
			pe[p*peStride+d*4+0] = float32(math.Cos(theta))
			pe[p*peStride+d*4+2] = float32(math.Sin(theta))
		}
	}

	// CPU reference
	cpuOut := make([]float32, len(vec))
	copy(cpuOut, vec)
	for p := 0; p < nPos; p++ {
		peBase := (peOffset + p) * peStride
		for h := 0; h < nHeads; h++ {
			for d := 0; d < halfDim; d++ {
				cosVal := pe[peBase+d*4]
				sinVal := pe[peBase+d*4+2]
				idx0 := p*dim + h*headDim + 2*d
				idx1 := idx0 + 1
				re := cpuOut[idx0]
				im := cpuOut[idx1]
				cpuOut[idx0] = re*cosVal - im*sinVal
				cpuOut[idx1] = re*sinVal + im*cosVal
			}
		}
	}

	// GPU
	vecBuf := Alloc(uint64(len(vec)) * 4)
	defer Free(vecBuf)
	peBuf := Alloc(uint64(len(pe)) * 4)
	defer Free(peBuf)

	UploadF32(vecBuf, vec)
	UploadF32(peBuf, pe)

	BeginBatch()
	if err := RoPE3D(vecBuf, peBuf, nPos, nHeads, headDim, peOffset, peStride); err != nil {
		EndBatch()
		t.Fatal(err)
	}
	EndBatch()

	gpuOut := make([]float32, len(vec))
	DownloadF32(vecBuf, gpuOut)

	maxErr := compareF32(cpuOut, gpuOut)
	t.Logf("RoPE3D: MaxErr=%.8f", maxErr)
	if maxErr > 1e-5 {
		t.Errorf("MaxErr too high: %f", maxErr)
	}
}

func TestAttentionFullF32(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer Shutdown()

	const seqLen = 8
	const numHeads = 2
	const numKVHeads = 2
	const headDim = 16
	qDim := numHeads * headDim
	kvDim := numKVHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	rng := rand.New(rand.NewSource(42))
	q := make([]float32, seqLen*qDim)
	k := make([]float32, seqLen*kvDim)
	v := make([]float32, seqLen*kvDim)
	for i := range q {
		q[i] = rng.Float32()*2 - 1
	}
	for i := range k {
		k[i] = rng.Float32()*2 - 1
	}
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}

	// CPU reference: full bidirectional attention
	cpuOut := cpuFullAttention(q, k, v, seqLen, numHeads, numKVHeads, headDim, scale)

	// GPU
	qBuf := Alloc(uint64(len(q)) * 4)
	defer Free(qBuf)
	kBuf := Alloc(uint64(len(k)) * 4)
	defer Free(kBuf)
	vBuf := Alloc(uint64(len(v)) * 4)
	defer Free(vBuf)
	outBuf := Alloc(uint64(seqLen*qDim) * 4)
	defer Free(outBuf)

	UploadF32(qBuf, q)
	UploadF32(kBuf, k)
	UploadF32(vBuf, v)

	BeginBatch()
	if err := AttentionFullF32(outBuf, qBuf, kBuf, vBuf, numHeads, numKVHeads, headDim, kvDim, seqLen, scale); err != nil {
		EndBatch()
		t.Fatal(err)
	}
	EndBatch()

	gpuOut := make([]float32, seqLen*qDim)
	DownloadF32(outBuf, gpuOut)

	maxErr := compareF32(cpuOut, gpuOut)
	t.Logf("AttentionFullF32: MaxErr=%.6f", maxErr)
	if maxErr > 1e-3 {
		t.Errorf("MaxErr too high: %f", maxErr)
	}
}

func TestRMSNormHeadsBatch(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer Shutdown()

	const npos = 4
	const numHeads = 2
	const headDim = 16
	const eps = 1e-6
	totalDim := numHeads * headDim

	rng := rand.New(rand.NewSource(42))
	data := make([]float32, npos*totalDim)
	weight := make([]float32, headDim)
	for i := range data {
		data[i] = rng.Float32()*2 - 1
	}
	for i := range weight {
		weight[i] = rng.Float32()*0.5 + 0.75
	}

	// CPU reference
	cpuOut := make([]float32, len(data))
	copy(cpuOut, data)
	for p := 0; p < npos; p++ {
		for h := 0; h < numHeads; h++ {
			base := p*totalDim + h*headDim
			head := cpuOut[base : base+headDim]
			var ss float64
			for _, v := range head {
				ss += float64(v) * float64(v)
			}
			invRMS := 1.0 / math.Sqrt(ss/float64(headDim)+eps)
			for d := 0; d < headDim; d++ {
				head[d] = float32(float64(head[d]) * invRMS * float64(weight[d]))
			}
		}
	}

	dataBuf := Alloc(uint64(len(data)) * 4)
	defer Free(dataBuf)
	weightBuf := Alloc(uint64(headDim) * 4)
	defer Free(weightBuf)

	UploadF32(dataBuf, data)
	UploadF32(weightBuf, weight)

	BeginBatch()
	if err := RMSNormHeadsBatch(dataBuf, weightBuf, numHeads, headDim, npos, eps); err != nil {
		EndBatch()
		t.Fatal(err)
	}
	EndBatch()

	gpuOut := make([]float32, len(data))
	DownloadF32(dataBuf, gpuOut)

	maxErr := compareF32(cpuOut, gpuOut)
	t.Logf("RMSNormHeadsBatch: MaxErr=%.8f", maxErr)
	if maxErr > 1e-4 {
		t.Errorf("MaxErr too high: %f", maxErr)
	}
}

// --- Helpers ---

func compareF32(a, b []float32) float64 {
	var maxErr float64
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > maxErr {
			maxErr = d
		}
	}
	return maxErr
}

func cpuFullAttention(q, k, v []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) []float32 {
	qDim := numHeads * headDim
	kvDim := numKVHeads * headDim
	headsPerKV := numHeads / numKVHeads
	out := make([]float32, seqLen*qDim)

	scores := make([]float64, seqLen)

	for h := 0; h < numHeads; h++ {
		kvH := h / headsPerKV
		for qi := 0; qi < seqLen; qi++ {
			// Compute scores
			maxScore := math.Inf(-1)
			for ki := 0; ki < seqLen; ki++ {
				dot := float64(0)
				for d := 0; d < headDim; d++ {
					dot += float64(q[qi*qDim+h*headDim+d]) * float64(k[ki*kvDim+kvH*headDim+d])
				}
				scores[ki] = dot * float64(scale)
				if scores[ki] > maxScore {
					maxScore = scores[ki]
				}
			}
			// Softmax
			var sum float64
			for ki := 0; ki < seqLen; ki++ {
				scores[ki] = math.Exp(scores[ki] - maxScore)
				sum += scores[ki]
			}
			invSum := 1.0 / sum
			// Weighted sum
			for d := 0; d < headDim; d++ {
				var acc float64
				for ki := 0; ki < seqLen; ki++ {
					acc += scores[ki] * float64(v[ki*kvDim+kvH*headDim+d])
				}
				out[qi*qDim+h*headDim+d] = float32(acc * invSum)
			}
		}
	}
	return out
}
