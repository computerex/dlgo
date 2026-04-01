//go:build cgo && vulkan

package diffusion

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/gpu"
)

// makeF32Tensor creates a QuantizedTensor with F32 data from a float32 slice.
func makeF32Tensor(rows, cols int, data []float32) *core.QuantizedTensor {
	if len(data) != rows*cols {
		panic("data length mismatch")
	}
	raw := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	return &core.QuantizedTensor{
		Data: raw,
		Type: 0, // F32
		Rows: rows,
		Cols: cols,
	}
}

// randFloats returns n random floats in [-scale, scale].
func randFloats(rng *rand.Rand, n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = (rng.Float32()*2 - 1) * scale
	}
	return out
}

// testConfig returns a tiny ZImageConfig for testing.
func testConfig() ZImageConfig {
	return ZImageConfig{
		PatchSize:       2,
		HiddenSize:      16,
		InChannels:      4,
		OutChannels:     4,
		NumLayers:       1,
		NumRefinerLayers: 0,
		HeadDim:         4,
		NumHeads:        4,
		NumKVHeads:      4,
		MultipleOf:      8,
		FFNDimMult:      2.0,
		NormEps:         1e-5,
		QKNorm:          true,
		CapFeatDim:      8,
		Theta:           256,
		AxesDim:         [3]int{4, 4, 4},
		AdaLNEmbedDim:   8,
		SeqMultiOf:      1,
	}
}

// makeSyntheticPE creates simple PE data for testing.
// layout: pe[(peOffset+p)*peStride + d*4 + 0] = cos, pe[...+2] = sin
func makeSyntheticPE(nPos, headDim, peOffset int) []float32 {
	peStride := headDim * 2
	total := (peOffset + nPos) * peStride
	pe := make([]float32, total)
	halfDim := headDim / 2
	for p := 0; p < nPos; p++ {
		base := (peOffset + p) * peStride
		for d := 0; d < halfDim; d++ {
			theta := float64(p) * 0.1 * float64(d+1)
			pe[base+d*4+0] = float32(math.Cos(theta))
			pe[base+d*4+1] = -float32(math.Sin(theta)) // unused by our impl but fill for safety
			pe[base+d*4+2] = float32(math.Sin(theta))
			pe[base+d*4+3] = float32(math.Cos(theta)) // unused
		}
	}
	return pe
}

// TestGpuForwardBlockNoAdaLN compares GPU vs CPU for one transformer block
// without adaLN (context refiner style).
func TestGpuForwardBlockNoAdaLN(t *testing.T) {
	if err := gpu.Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer gpu.Shutdown()

	rng := rand.New(rand.NewSource(42))
	cfg := testConfig()
	hidden := cfg.HiddenSize
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	qDim := numHeads * headDim
	kvDim := numKVHeads * headDim
	qkvDim := qDim + 2*kvDim
	ffnDim := cfg.FFNHiddenDim()
	seqLen := 4

	t.Logf("Config: hidden=%d headDim=%d numHeads=%d ffnDim=%d seqLen=%d qkvDim=%d",
		hidden, headDim, numHeads, ffnDim, seqLen, qkvDim)

	// Create random weights (F32)
	attnQKVData := randFloats(rng, qkvDim*hidden, 0.1)
	attnOutData := randFloats(rng, hidden*qDim, 0.1)
	ffnGateData := randFloats(rng, ffnDim*hidden, 0.1)
	ffnDownData := randFloats(rng, hidden*ffnDim, 0.1)
	ffnUpData := randFloats(rng, ffnDim*hidden, 0.1)
	qNorm := randFloats(rng, headDim, 0.5)
	kNorm := randFloats(rng, headDim, 0.5)
	attnNorm1 := randFloats(rng, hidden, 0.5)
	attnNorm2 := randFloats(rng, hidden, 0.5)
	ffnNorm1 := randFloats(rng, hidden, 0.5)
	ffnNorm2 := randFloats(rng, hidden, 0.5)

	// Make norm weights positive to avoid numerical issues
	for i := range qNorm {
		qNorm[i] = float32(math.Abs(float64(qNorm[i]))) + 0.1
	}
	for i := range kNorm {
		kNorm[i] = float32(math.Abs(float64(kNorm[i]))) + 0.1
	}
	for i := range attnNorm1 {
		attnNorm1[i] = float32(math.Abs(float64(attnNorm1[i]))) + 0.1
	}
	for i := range attnNorm2 {
		attnNorm2[i] = float32(math.Abs(float64(attnNorm2[i]))) + 0.1
	}
	for i := range ffnNorm1 {
		ffnNorm1[i] = float32(math.Abs(float64(ffnNorm1[i]))) + 0.1
	}
	for i := range ffnNorm2 {
		ffnNorm2[i] = float32(math.Abs(float64(ffnNorm2[i]))) + 0.1
	}

	// CPU layer
	cpuLayer := &DiTLayer{
		AttnQKV:   makeF32Tensor(qkvDim, hidden, attnQKVData),
		AttnOut:   makeF32Tensor(hidden, qDim, attnOutData),
		FFNGate:   makeF32Tensor(ffnDim, hidden, ffnGateData),
		FFNDown:   makeF32Tensor(hidden, ffnDim, ffnDownData),
		FFNUp:     makeF32Tensor(ffnDim, hidden, ffnUpData),
		QNorm:     qNorm,
		KNorm:     kNorm,
		AttnNorm1: attnNorm1,
		AttnNorm2: attnNorm2,
		FFNNorm1:  ffnNorm1,
		FFNNorm2:  ffnNorm2,
		// No AdaLN (context refiner style)
		AdaLNWeight: nil,
		AdaLNBias:   nil,
	}

	cpuModel := &DiTModel{Config: cfg}

	// Create PE
	peOffset := 0
	pe := makeSyntheticPE(seqLen, headDim, peOffset)
	peStride := headDim * 2

	// Random input
	input := randFloats(rng, seqLen*hidden, 0.5)

	// --- CPU forward ---
	cpuRS := NewDiTRunState(cfg, seqLen)
	cpuX := make([]float32, seqLen*hidden)
	copy(cpuX, input)
	forwardBlock(cpuModel, cpuRS, cpuLayer, cpuX, seqLen, pe, peOffset, nil)

	// --- GPU forward ---
	// Upload layer weights
	gpuAttnQKV, err := gpu.UploadTensor(cpuLayer.AttnQKV)
	if err != nil {
		t.Fatalf("upload AttnQKV: %v", err)
	}
	defer gpu.Free(gpuAttnQKV.Buf)

	gpuAttnOut, err := gpu.UploadTensor(cpuLayer.AttnOut)
	if err != nil {
		t.Fatalf("upload AttnOut: %v", err)
	}
	defer gpu.Free(gpuAttnOut.Buf)

	gpuFFNGate, err := gpu.UploadTensor(cpuLayer.FFNGate)
	if err != nil {
		t.Fatalf("upload FFNGate: %v", err)
	}
	defer gpu.Free(gpuFFNGate.Buf)

	gpuFFNDown, err := gpu.UploadTensor(cpuLayer.FFNDown)
	if err != nil {
		t.Fatalf("upload FFNDown: %v", err)
	}
	defer gpu.Free(gpuFFNDown.Buf)

	gpuFFNUp, err := gpu.UploadTensor(cpuLayer.FFNUp)
	if err != nil {
		t.Fatalf("upload FFNUp: %v", err)
	}
	defer gpu.Free(gpuFFNUp.Buf)

	gpuQNorm, err := gpu.UploadF32Slice(qNorm)
	if err != nil {
		t.Fatalf("upload QNorm: %v", err)
	}
	defer gpu.Free(gpuQNorm)

	gpuKNorm, err := gpu.UploadF32Slice(kNorm)
	if err != nil {
		t.Fatalf("upload KNorm: %v", err)
	}
	defer gpu.Free(gpuKNorm)

	gpuAttnNorm1, err := gpu.UploadF32Slice(attnNorm1)
	if err != nil {
		t.Fatalf("upload AttnNorm1: %v", err)
	}
	defer gpu.Free(gpuAttnNorm1)

	gpuAttnNorm2, err := gpu.UploadF32Slice(attnNorm2)
	if err != nil {
		t.Fatalf("upload AttnNorm2: %v", err)
	}
	defer gpu.Free(gpuAttnNorm2)

	gpuFFNNorm1, err := gpu.UploadF32Slice(ffnNorm1)
	if err != nil {
		t.Fatalf("upload FFNNorm1: %v", err)
	}
	defer gpu.Free(gpuFFNNorm1)

	gpuFFNNorm2, err := gpu.UploadF32Slice(ffnNorm2)
	if err != nil {
		t.Fatalf("upload FFNNorm2: %v", err)
	}
	defer gpu.Free(gpuFFNNorm2)

	gpuLayer := &GpuDiTLayer{
		AttnQKV:   gpuAttnQKV,
		AttnOut:   gpuAttnOut,
		FFNGate:   gpuFFNGate,
		FFNDown:   gpuFFNDown,
		FFNUp:     gpuFFNUp,
		QNorm:     gpuQNorm,
		KNorm:     gpuKNorm,
		AttnNorm1: gpuAttnNorm1,
		AttnNorm2: gpuAttnNorm2,
		FFNNorm1:  gpuFFNNorm1,
		FFNNorm2:  gpuFFNNorm2,
	}

	gpuModel := &GpuDiTModel{Config: cfg}
	gpuRS, err := NewGpuDiTRunState(cfg, seqLen)
	if err != nil {
		t.Fatalf("GPU run state: %v", err)
	}
	defer func() {
		gpu.Free(gpuRS.X)
		gpu.Free(gpuRS.XNorm)
		gpu.Free(gpuRS.QKV)
		gpu.Free(gpuRS.Q)
		gpu.Free(gpuRS.K)
		gpu.Free(gpuRS.V)
		gpu.Free(gpuRS.AttnOut)
		gpu.Free(gpuRS.Proj)
		gpu.Free(gpuRS.Gate)
		gpu.Free(gpuRS.Up)
		gpu.Free(gpuRS.Hidden)
		gpu.Free(gpuRS.FFNOut)
		gpu.Free(gpuRS.Residual)
		gpu.Free(gpuRS.Mod)
		gpu.Free(gpuRS.ScaleBuf)
		gpu.Free(gpuRS.GateBuf)
	}()

	// Upload PE
	peBuf, err := gpu.AllocE(uint64(len(pe)) * 4)
	if err != nil {
		t.Fatalf("alloc PE: %v", err)
	}
	defer gpu.Free(peBuf)
	gpu.UploadF32(peBuf, pe)
	gpuRS.PE = peBuf

	// Upload input to GPU
	gpu.UploadF32(gpuRS.X, input)

	// Run GPU forward block
	gpuForwardBlock(gpuModel, gpuRS, gpuLayer, seqLen, hidden, pe, peOffset, nil, peStride)
	gpu.Sync()

	// Download result
	gpuOut := make([]float32, seqLen*hidden)
	gpu.DownloadF32(gpuRS.X, gpuOut)

	// Compare
	maxErr := float32(0)
	sumSqErr := float64(0)
	sumSq := float64(0)
	for i := 0; i < seqLen*hidden; i++ {
		diff := float32(math.Abs(float64(cpuX[i] - gpuOut[i])))
		if diff > maxErr {
			maxErr = diff
		}
		sumSqErr += float64(diff) * float64(diff)
		sumSq += float64(cpuX[i]) * float64(cpuX[i])
	}
	rmse := float32(math.Sqrt(sumSqErr / float64(seqLen*hidden)))
	relErr := float32(math.Sqrt(sumSqErr / (sumSq + 1e-12)))

	t.Logf("GPU vs CPU (no adaLN): MaxErr=%.6f RMSE=%.6f RelErr=%.6f", maxErr, rmse, relErr)

	// With F32 weights, we expect near-exact match.
	// Tolerance accounts for different reduction order in GPU vs CPU.
	if maxErr > 0.01 {
		t.Errorf("MaxErr too large: %.6f (want < 0.01)", maxErr)
		// Print first few mismatches
		for i := 0; i < seqLen*hidden && i < 10; i++ {
			if math.Abs(float64(cpuX[i]-gpuOut[i])) > 0.001 {
				t.Logf("  [%d] CPU=%.6f GPU=%.6f diff=%.6f", i, cpuX[i], gpuOut[i], cpuX[i]-gpuOut[i])
			}
		}
	}
}

// TestGpuForwardBlockWithAdaLN compares GPU vs CPU for a block with adaLN
// (noise refiner / main layer style).
func TestGpuForwardBlockWithAdaLN(t *testing.T) {
	if err := gpu.Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer gpu.Shutdown()

	rng := rand.New(rand.NewSource(99))
	cfg := testConfig()
	hidden := cfg.HiddenSize
	headDim := cfg.HeadDim
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	qDim := numHeads * headDim
	kvDim := numKVHeads * headDim
	qkvDim := qDim + 2*kvDim
	ffnDim := cfg.FFNHiddenDim()
	adaLNDim := cfg.AdaLNEmbedDim
	seqLen := 4

	t.Logf("Config: hidden=%d headDim=%d numHeads=%d ffnDim=%d adaLNDim=%d seqLen=%d",
		hidden, headDim, numHeads, ffnDim, adaLNDim, seqLen)

	// Create weights
	attnQKVData := randFloats(rng, qkvDim*hidden, 0.1)
	attnOutData := randFloats(rng, hidden*qDim, 0.1)
	ffnGateData := randFloats(rng, ffnDim*hidden, 0.1)
	ffnDownData := randFloats(rng, hidden*ffnDim, 0.1)
	ffnUpData := randFloats(rng, ffnDim*hidden, 0.1)
	adaLNData := randFloats(rng, 4*hidden*adaLNDim, 0.1)
	adaLNBias := randFloats(rng, 4*hidden, 0.02)
	qNorm := randFloats(rng, headDim, 0.5)
	kNorm := randFloats(rng, headDim, 0.5)
	attnNorm1 := randFloats(rng, hidden, 0.5)
	attnNorm2 := randFloats(rng, hidden, 0.5)
	ffnNorm1 := randFloats(rng, hidden, 0.5)
	ffnNorm2 := randFloats(rng, hidden, 0.5)

	// Positive norms
	for _, nw := range [][]float32{qNorm, kNorm, attnNorm1, attnNorm2, ffnNorm1, ffnNorm2} {
		for i := range nw {
			nw[i] = float32(math.Abs(float64(nw[i]))) + 0.1
		}
	}

	cpuLayer := &DiTLayer{
		AttnQKV:     makeF32Tensor(qkvDim, hidden, attnQKVData),
		AttnOut:     makeF32Tensor(hidden, qDim, attnOutData),
		FFNGate:     makeF32Tensor(ffnDim, hidden, ffnGateData),
		FFNDown:     makeF32Tensor(hidden, ffnDim, ffnDownData),
		FFNUp:       makeF32Tensor(ffnDim, hidden, ffnUpData),
		QNorm:       qNorm,
		KNorm:       kNorm,
		AttnNorm1:   attnNorm1,
		AttnNorm2:   attnNorm2,
		FFNNorm1:    ffnNorm1,
		FFNNorm2:    ffnNorm2,
		AdaLNWeight: makeF32Tensor(4*hidden, adaLNDim, adaLNData),
		AdaLNBias:   adaLNBias,
	}
	cpuModel := &DiTModel{Config: cfg}

	peOffset := 0
	pe := makeSyntheticPE(seqLen, headDim, peOffset)
	peStride := headDim * 2

	input := randFloats(rng, seqLen*hidden, 0.5)
	adaLNInput := randFloats(rng, adaLNDim, 0.3)

	// CPU forward
	cpuRS := NewDiTRunState(cfg, seqLen)
	cpuX := make([]float32, seqLen*hidden)
	copy(cpuX, input)
	forwardBlock(cpuModel, cpuRS, cpuLayer, cpuX, seqLen, pe, peOffset, adaLNInput)

	// GPU forward — upload everything
	gpuAttnQKV, _ := gpu.UploadTensor(cpuLayer.AttnQKV)
	defer gpu.Free(gpuAttnQKV.Buf)
	gpuAttnOut, _ := gpu.UploadTensor(cpuLayer.AttnOut)
	defer gpu.Free(gpuAttnOut.Buf)
	gpuFFNGate, _ := gpu.UploadTensor(cpuLayer.FFNGate)
	defer gpu.Free(gpuFFNGate.Buf)
	gpuFFNDown, _ := gpu.UploadTensor(cpuLayer.FFNDown)
	defer gpu.Free(gpuFFNDown.Buf)
	gpuFFNUp, _ := gpu.UploadTensor(cpuLayer.FFNUp)
	defer gpu.Free(gpuFFNUp.Buf)
	gpuAdaLN, _ := gpu.UploadTensor(cpuLayer.AdaLNWeight)
	defer gpu.Free(gpuAdaLN.Buf)
	gpuQNorm, _ := gpu.UploadF32Slice(qNorm)
	defer gpu.Free(gpuQNorm)
	gpuKNorm, _ := gpu.UploadF32Slice(kNorm)
	defer gpu.Free(gpuKNorm)
	gpuAttnNorm1, _ := gpu.UploadF32Slice(attnNorm1)
	defer gpu.Free(gpuAttnNorm1)
	gpuAttnNorm2, _ := gpu.UploadF32Slice(attnNorm2)
	defer gpu.Free(gpuAttnNorm2)
	gpuFFNNorm1, _ := gpu.UploadF32Slice(ffnNorm1)
	defer gpu.Free(gpuFFNNorm1)
	gpuFFNNorm2, _ := gpu.UploadF32Slice(ffnNorm2)
	defer gpu.Free(gpuFFNNorm2)
	gpuAdaLNBias, _ := gpu.UploadF32Slice(adaLNBias)
	defer gpu.Free(gpuAdaLNBias)

	gpuLayer := &GpuDiTLayer{
		AttnQKV:     gpuAttnQKV,
		AttnOut:     gpuAttnOut,
		FFNGate:     gpuFFNGate,
		FFNDown:     gpuFFNDown,
		FFNUp:       gpuFFNUp,
		QNorm:       gpuQNorm,
		KNorm:       gpuKNorm,
		AttnNorm1:   gpuAttnNorm1,
		AttnNorm2:   gpuAttnNorm2,
		FFNNorm1:    gpuFFNNorm1,
		FFNNorm2:    gpuFFNNorm2,
		AdaLNWeight: gpuAdaLN,
		AdaLNBias:   gpuAdaLNBias,
	}

	gpuModel := &GpuDiTModel{Config: cfg}
	gpuRS, err := NewGpuDiTRunState(cfg, seqLen)
	if err != nil {
		t.Fatalf("GPU run state: %v", err)
	}
	defer func() {
		gpu.Free(gpuRS.X)
		gpu.Free(gpuRS.XNorm)
		gpu.Free(gpuRS.QKV)
		gpu.Free(gpuRS.Q)
		gpu.Free(gpuRS.K)
		gpu.Free(gpuRS.V)
		gpu.Free(gpuRS.AttnOut)
		gpu.Free(gpuRS.Proj)
		gpu.Free(gpuRS.Gate)
		gpu.Free(gpuRS.Up)
		gpu.Free(gpuRS.Hidden)
		gpu.Free(gpuRS.FFNOut)
		gpu.Free(gpuRS.Residual)
		gpu.Free(gpuRS.Mod)
		gpu.Free(gpuRS.ScaleBuf)
		gpu.Free(gpuRS.GateBuf)
	}()

	peBuf, _ := gpu.AllocE(uint64(len(pe)) * 4)
	defer gpu.Free(peBuf)
	gpu.UploadF32(peBuf, pe)
	gpuRS.PE = peBuf

	gpu.UploadF32(gpuRS.X, input)
	gpuForwardBlock(gpuModel, gpuRS, gpuLayer, seqLen, hidden, pe, peOffset, adaLNInput, peStride)
	gpu.Sync()

	gpuOut := make([]float32, seqLen*hidden)
	gpu.DownloadF32(gpuRS.X, gpuOut)

	maxErr := float32(0)
	sumSqErr := float64(0)
	sumSq := float64(0)
	for i := 0; i < seqLen*hidden; i++ {
		diff := float32(math.Abs(float64(cpuX[i] - gpuOut[i])))
		if diff > maxErr {
			maxErr = diff
		}
		sumSqErr += float64(diff) * float64(diff)
		sumSq += float64(cpuX[i]) * float64(cpuX[i])
	}
	rmse := float32(math.Sqrt(sumSqErr / float64(seqLen*hidden)))
	relErr := float32(math.Sqrt(sumSqErr / (sumSq + 1e-12)))

	t.Logf("GPU vs CPU (with adaLN): MaxErr=%.6f RMSE=%.6f RelErr=%.6f", maxErr, rmse, relErr)

	if maxErr > 0.01 {
		t.Errorf("MaxErr too large: %.6f (want < 0.01)", maxErr)
		for i := 0; i < seqLen*hidden && i < 20; i++ {
			if math.Abs(float64(cpuX[i]-gpuOut[i])) > 0.001 {
				t.Logf("  [%d] CPU=%.6f GPU=%.6f diff=%.6f", i, cpuX[i], gpuOut[i], cpuX[i]-gpuOut[i])
			}
		}
	}
}

// TestGpuForwardRealModel loads the real Q4_K_M DiT model, runs one main layer
// on both CPU and GPU with identical random input, and compares output.
// This validates GPU correctness with actual quantized weights.
func TestGpuForwardRealModel(t *testing.T) {
	const ditPath = `C:\Users\mohd\Downloads\z-image-turbo-Q4_K_M.gguf`
	if _, err := os.Stat(ditPath); err != nil {
		t.Skipf("DiT model not found at %s", ditPath)
	}

	if err := gpu.Init(); err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer gpu.Shutdown()

	t.Log("Loading DiT model...")
	dit, err := LoadDiTModel(ditPath)
	if err != nil {
		t.Fatalf("LoadDiTModel: %v", err)
	}
	defer dit.MmapFile.Close()

	cfg := dit.Config
	hidden := cfg.HiddenSize
	headDim := cfg.HeadDim
	seqLen := 36 // small: 32 text + 4 image tokens (tiny resolution)

	t.Logf("Model: hidden=%d headDim=%d numHeads=%d ffnDim=%d",
		hidden, headDim, cfg.NumHeads, cfg.FFNHiddenDim())

	// Use MainLayers[0] which has adaLN
	cpuLayer := &dit.MainLayers[0]

	// Create PE and random input
	rng := rand.New(rand.NewSource(123))
	pe := makeSyntheticPE(seqLen, headDim, 0)
	peStride := headDim * 2
	input := randFloats(rng, seqLen*hidden, 0.1)
	adaLNInput := randFloats(rng, cfg.AdaLNEmbedDim, 0.3)

	// --- CPU forward ---
	cpuModel := &DiTModel{Config: cfg}
	cpuRS := NewDiTRunState(cfg, seqLen)
	cpuX := make([]float32, seqLen*hidden)
	copy(cpuX, input)
	forwardBlock(cpuModel, cpuRS, cpuLayer, cpuX, seqLen, pe, 0, adaLNInput)

	// --- GPU forward ---
	t.Log("Uploading layer 0 weights to GPU...")
	gpuAttnQKV, err := gpu.UploadTensor(cpuLayer.AttnQKV)
	if err != nil {
		t.Fatalf("upload AttnQKV: %v", err)
	}
	defer gpu.Free(gpuAttnQKV.Buf)
	gpuAttnOut, err := gpu.UploadTensor(cpuLayer.AttnOut)
	if err != nil {
		t.Fatalf("upload AttnOut: %v", err)
	}
	defer gpu.Free(gpuAttnOut.Buf)
	gpuFFNGate, err := gpu.UploadTensor(cpuLayer.FFNGate)
	if err != nil {
		t.Fatalf("upload FFNGate: %v", err)
	}
	defer gpu.Free(gpuFFNGate.Buf)
	gpuFFNDown, err := gpu.UploadTensor(cpuLayer.FFNDown)
	if err != nil {
		t.Fatalf("upload FFNDown: %v", err)
	}
	defer gpu.Free(gpuFFNDown.Buf)
	gpuFFNUp, err := gpu.UploadTensor(cpuLayer.FFNUp)
	if err != nil {
		t.Fatalf("upload FFNUp: %v", err)
	}
	defer gpu.Free(gpuFFNUp.Buf)
	gpuAdaLN, err := gpu.UploadTensor(cpuLayer.AdaLNWeight)
	if err != nil {
		t.Fatalf("upload AdaLN: %v", err)
	}
	defer gpu.Free(gpuAdaLN.Buf)

	gpuQNorm, _ := gpu.UploadF32Slice(cpuLayer.QNorm)
	defer gpu.Free(gpuQNorm)
	gpuKNorm, _ := gpu.UploadF32Slice(cpuLayer.KNorm)
	defer gpu.Free(gpuKNorm)
	gpuAttnNorm1, _ := gpu.UploadF32Slice(cpuLayer.AttnNorm1)
	defer gpu.Free(gpuAttnNorm1)
	gpuAttnNorm2, _ := gpu.UploadF32Slice(cpuLayer.AttnNorm2)
	defer gpu.Free(gpuAttnNorm2)
	gpuFFNNorm1, _ := gpu.UploadF32Slice(cpuLayer.FFNNorm1)
	defer gpu.Free(gpuFFNNorm1)
	gpuFFNNorm2, _ := gpu.UploadF32Slice(cpuLayer.FFNNorm2)
	defer gpu.Free(gpuFFNNorm2)
	gpuAdaLNBias, _ := gpu.UploadF32Slice(cpuLayer.AdaLNBias)
	defer gpu.Free(gpuAdaLNBias)

	gpuLayer := &GpuDiTLayer{
		AttnQKV:     gpuAttnQKV,
		AttnOut:     gpuAttnOut,
		FFNGate:     gpuFFNGate,
		FFNDown:     gpuFFNDown,
		FFNUp:       gpuFFNUp,
		QNorm:       gpuQNorm,
		KNorm:       gpuKNorm,
		AttnNorm1:   gpuAttnNorm1,
		AttnNorm2:   gpuAttnNorm2,
		FFNNorm1:    gpuFFNNorm1,
		FFNNorm2:    gpuFFNNorm2,
		AdaLNWeight: gpuAdaLN,
		AdaLNBias:   gpuAdaLNBias,
	}

	gpuModel := &GpuDiTModel{Config: cfg}
	gpuRS, err := NewGpuDiTRunState(cfg, seqLen)
	if err != nil {
		t.Fatalf("GPU run state: %v", err)
	}
	defer func() {
		gpu.Free(gpuRS.X)
		gpu.Free(gpuRS.XNorm)
		gpu.Free(gpuRS.QKV)
		gpu.Free(gpuRS.Q)
		gpu.Free(gpuRS.K)
		gpu.Free(gpuRS.V)
		gpu.Free(gpuRS.AttnOut)
		gpu.Free(gpuRS.Proj)
		gpu.Free(gpuRS.Gate)
		gpu.Free(gpuRS.Up)
		gpu.Free(gpuRS.Hidden)
		gpu.Free(gpuRS.FFNOut)
		gpu.Free(gpuRS.Residual)
		gpu.Free(gpuRS.Mod)
		gpu.Free(gpuRS.ScaleBuf)
		gpu.Free(gpuRS.GateBuf)
	}()

	peBuf, _ := gpu.AllocE(uint64(len(pe)) * 4)
	defer gpu.Free(peBuf)
	gpu.UploadF32(peBuf, pe)
	gpuRS.PE = peBuf

	gpu.UploadF32(gpuRS.X, input)

	t.Log("Running GPU forward block...")
	gpuForwardBlock(gpuModel, gpuRS, gpuLayer, seqLen, hidden, pe, 0, adaLNInput, peStride)
	gpu.Sync()

	gpuOut := make([]float32, seqLen*hidden)
	gpu.DownloadF32(gpuRS.X, gpuOut)

	// Compare
	maxErr := float32(0)
	sumSqErr := float64(0)
	sumSq := float64(0)
	for i := 0; i < seqLen*hidden; i++ {
		diff := float32(math.Abs(float64(cpuX[i] - gpuOut[i])))
		if diff > maxErr {
			maxErr = diff
		}
		sumSqErr += float64(diff) * float64(diff)
		sumSq += float64(cpuX[i]) * float64(cpuX[i])
	}
	rmse := float32(math.Sqrt(sumSqErr / float64(seqLen*hidden)))
	relErr := float32(math.Sqrt(sumSqErr / (sumSq + 1e-12)))

	t.Logf("Real Q4_K_M weights — GPU vs CPU: MaxErr=%.6f RMSE=%.6f RelErr=%.6f",
		maxErr, rmse, relErr)

	// Q4_K_M dequantization differences compound through a full transformer block
	// (10+ matvec ops, attention, norms). RelErr < 5% is excellent for Q4_K_M.
	// MaxErr can spike in attention heads that amplify small differences.
	if relErr > 0.05 {
		t.Errorf("RelErr too large: %.6f (want < 0.05)", relErr)
	}
	if maxErr > 2.0 {
		t.Errorf("MaxErr too large: %.6f (want < 2.0)", maxErr)
		mismatches := 0
		for i := 0; i < seqLen*hidden; i++ {
			if math.Abs(float64(cpuX[i]-gpuOut[i])) > 0.01 {
				if mismatches < 20 {
					t.Logf("  [%d] CPU=%.6f GPU=%.6f diff=%.6f", i, cpuX[i], gpuOut[i], cpuX[i]-gpuOut[i])
				}
				mismatches++
			}
		}
		if mismatches > 20 {
			t.Logf("  ... and %d more mismatches", mismatches-20)
		}
	}
}
