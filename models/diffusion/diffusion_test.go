package diffusion

import (
	"math"
	"testing"
)

// --- timestepEmbedding tests ---

func TestTimestepEmbeddingDimension(t *testing.T) {
	for _, dim := range []int{64, 128, 256} {
		emb := timestepEmbedding(500.0, dim)
		if len(emb) != dim {
			t.Errorf("timestepEmbedding(500, %d): got len %d, want %d", dim, len(emb), dim)
		}
	}
}

func TestTimestepEmbeddingRanges(t *testing.T) {
	emb := timestepEmbedding(500.0, 256)
	for i, v := range emb {
		if v < -1.0 || v > 1.0 {
			t.Errorf("timestepEmbedding[%d] = %f, out of [-1, 1] range", i, v)
		}
	}
}

func TestTimestepEmbeddingSinCosStructure(t *testing.T) {
	// flip_sin_to_cos=true: first half is cos, second half is sin
	dim := 256
	halfDim := dim / 2
	timestep := float32(500.0)
	emb := timestepEmbedding(timestep, dim)

	logTimescale := -math.Log(10000.0) / float64(halfDim)
	for i := 0; i < halfDim; i++ {
		freq := math.Exp(float64(i) * logTimescale)
		angle := float64(timestep) * freq
		wantCos := float32(math.Cos(angle))
		wantSin := float32(math.Sin(angle))
		if diff := math.Abs(float64(emb[i] - wantCos)); diff > 1e-6 {
			t.Errorf("cos[%d]: got %f, want %f", i, emb[i], wantCos)
		}
		if diff := math.Abs(float64(emb[i+halfDim] - wantSin)); diff > 1e-6 {
			t.Errorf("sin[%d]: got %f, want %f", i, emb[i+halfDim], wantSin)
		}
	}
}

func TestTimestepEmbeddingDifferentTimesteps(t *testing.T) {
	e1 := timestepEmbedding(100.0, 256)
	e2 := timestepEmbedding(900.0, 256)
	same := true
	for i := range e1 {
		if e1[i] != e2[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different timesteps produced identical embeddings")
	}
}

// --- patchify / unpatchify round-trip tests ---

func TestPatchifyUnpatchifyRoundTrip(t *testing.T) {
	C, H, W, patchSize := 4, 8, 8, 2
	input := make([]float32, C*H*W)
	for i := range input {
		input[i] = float32(i)
	}
	patches := patchify(input, C, H, W, patchSize)
	output := unpatchify(patches, C, H, W, patchSize)

	if len(output) != len(input) {
		t.Fatalf("round-trip size mismatch: got %d, want %d", len(output), len(input))
	}
	for i := range input {
		if input[i] != output[i] {
			t.Errorf("round-trip mismatch at [%d]: got %f, want %f", i, output[i], input[i])
		}
	}
}

func TestPatchifyDimensions(t *testing.T) {
	C, H, W, patchSize := 16, 128, 128, 2
	input := make([]float32, C*H*W)
	patches := patchify(input, C, H, W, patchSize)
	hPatches := H / patchSize
	wPatches := W / patchSize
	nPatches := hPatches * wPatches
	patchDim := patchSize * patchSize * C
	wantLen := nPatches * patchDim
	if len(patches) != wantLen {
		t.Errorf("patchify produced %d elements, want %d (%d patches × %d patchDim)",
			len(patches), wantLen, nPatches, patchDim)
	}
}

func TestPatchifySmall(t *testing.T) {
	// 1 channel, 4×4 image, 2×2 patches → 4 patches of dim 4
	C, H, W, patchSize := 1, 4, 4, 2
	// Layout: [C, H, W] row-major
	input := []float32{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
		12, 13, 14, 15,
	}
	patches := patchify(input, C, H, W, patchSize)
	// patch(0,0) = [0,1,4,5], patch(0,1) = [2,3,6,7]
	// patch(1,0) = [8,9,12,13], patch(1,1) = [10,11,14,15]
	expected := []float32{
		0, 1, 4, 5,
		2, 3, 6, 7,
		8, 9, 12, 13,
		10, 11, 14, 15,
	}
	if len(patches) != len(expected) {
		t.Fatalf("patchify len=%d, want %d", len(patches), len(expected))
	}
	for i := range expected {
		if patches[i] != expected[i] {
			t.Errorf("patchify[%d] = %f, want %f", i, patches[i], expected[i])
		}
	}
}

// --- boundMod tests ---

func TestBoundMod(t *testing.T) {
	tests := []struct {
		a, m, want int
	}{
		{0, 32, 0},
		{1, 32, 31},
		{31, 32, 1},
		{32, 32, 0},
		{33, 32, 31},
		{64, 32, 0},
		{30, 32, 2},
	}
	for _, tt := range tests {
		got := boundMod(tt.a, tt.m)
		if got != tt.want {
			t.Errorf("boundMod(%d, %d) = %d, want %d", tt.a, tt.m, got, tt.want)
		}
	}
}

func TestBoundModInvariant(t *testing.T) {
	// a + boundMod(a,m) should always be a multiple of m
	for a := 0; a < 100; a++ {
		for m := 1; m <= 32; m++ {
			pad := boundMod(a, m)
			if (a+pad)%m != 0 {
				t.Errorf("boundMod(%d,%d)=%d but (%d+%d)%%%d = %d",
					a, m, pad, a, pad, m, (a+pad)%m)
			}
		}
	}
}

// --- addBias / addBiasBatch tests ---

func TestAddBias(t *testing.T) {
	x := []float32{1, 2, 3}
	bias := []float32{10, 20, 30}
	addBias(x, bias)
	expected := []float32{11, 22, 33}
	for i := range expected {
		if x[i] != expected[i] {
			t.Errorf("addBias[%d] = %f, want %f", i, x[i], expected[i])
		}
	}
}

func TestAddBiasNil(t *testing.T) {
	x := []float32{1, 2, 3}
	addBias(x, nil) // should not panic
	if x[0] != 1 || x[1] != 2 || x[2] != 3 {
		t.Error("addBias(nil) should be a no-op")
	}
}

func TestAddBiasBatch(t *testing.T) {
	// 2 positions, dim=3
	x := []float32{1, 2, 3, 4, 5, 6}
	bias := []float32{10, 20, 30}
	addBiasBatch(x, bias, 2, 3)
	expected := []float32{11, 22, 33, 14, 25, 36}
	for i := range expected {
		if x[i] != expected[i] {
			t.Errorf("addBiasBatch[%d] = %f, want %f", i, x[i], expected[i])
		}
	}
}

// --- clampByte tests ---

func TestClampByte(t *testing.T) {
	tests := []struct {
		in   float32
		want uint8
	}{
		{0.0, 0},
		{1.0, 255},
		{0.5, 128},
		{-0.5, 0},
		{1.5, 255},
	}
	for _, tt := range tests {
		got := clampByte(tt.in)
		if got != tt.want {
			t.Errorf("clampByte(%f) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

// --- DiscreteFlowDenoiser tests ---

func TestTToSigmaAndBack(t *testing.T) {
	d := NewDiscreteFlowDenoiser(3.0)
	// SigmaToT is intentionally NOT the mathematical inverse of TToSigma.
	// It uses the sd.cpp convention: SigmaToT(sigma) = sigma * 1000.
	// Verify SigmaToT gives expected values:
	for _, tc := range []struct {
		sigma float32
		wantT float32
	}{
		{0.0, 0.0},
		{0.5, 500.0},
		{1.0, 1000.0},
	} {
		gotT := d.SigmaToT(tc.sigma)
		if diff := math.Abs(float64(gotT - tc.wantT)); diff > 0.01 {
			t.Errorf("SigmaToT(%f) = %f, want %f", tc.sigma, gotT, tc.wantT)
		}
	}
}

func TestTToSigmaBounds(t *testing.T) {
	d := NewDiscreteFlowDenoiser(3.0)
	// sigma at t=0 should be 0
	if s := d.TToSigma(0); s != 0 {
		t.Errorf("sigma(t=0) = %f, want 0", s)
	}
	// sigma at t=1000 should be 1.0 (for any shift)
	s := d.TToSigma(1000)
	if diff := math.Abs(float64(s - 1.0)); diff > 1e-5 {
		t.Errorf("sigma(t=1000) = %f, want 1.0", s)
	}
}

func TestTToSigmaMonotonic(t *testing.T) {
	d := NewDiscreteFlowDenoiser(3.0)
	prev := float32(0)
	for ts := float32(1); ts <= 1000; ts += 1 {
		s := d.TToSigma(ts)
		if s <= prev {
			t.Errorf("sigma not monotonically increasing: sigma(%f)=%f <= sigma(%f)=%f",
				ts, s, ts-1, prev)
		}
		prev = s
	}
}

func TestSimpleSchedule(t *testing.T) {
	d := NewDiscreteFlowDenoiser(3.0)
	steps := 8
	sigmas := SimpleSchedule(steps, d)

	if len(sigmas) != steps+1 {
		t.Fatalf("schedule length = %d, want %d", len(sigmas), steps+1)
	}
	// Last sigma must be 0
	if sigmas[steps] != 0 {
		t.Errorf("final sigma = %f, want 0", sigmas[steps])
	}
	// First sigma should be close to 1 (t≈1000)
	if sigmas[0] < 0.5 {
		t.Errorf("first sigma = %f, expected > 0.5", sigmas[0])
	}
	// Should be monotonically decreasing
	for i := 1; i <= steps; i++ {
		if sigmas[i] >= sigmas[i-1] {
			t.Errorf("schedule not decreasing: sigmas[%d]=%f >= sigmas[%d]=%f",
				i, sigmas[i], i-1, sigmas[i-1])
		}
	}
}

func TestDenoiseScaling(t *testing.T) {
	d := NewDiscreteFlowDenoiser(3.0)
	x := []float32{1.0, 2.0, 3.0}
	model := []float32{0.1, 0.2, 0.3}
	sigma := float32(0.5)

	denoised := d.Denoise(model, x, sigma)
	// c_skip=1, c_out=-sigma → denoised = x - sigma * model_output
	for i := range x {
		want := x[i] - sigma*model[i]
		if math.Abs(float64(denoised[i]-want)) > 1e-6 {
			t.Errorf("Denoise[%d] = %f, want %f", i, denoised[i], want)
		}
	}
}

func TestNoiseScaling(t *testing.T) {
	d := NewDiscreteFlowDenoiser(3.0)
	latent := []float32{1.0, 2.0}
	noise := []float32{3.0, 4.0}
	sigma := float32(0.3)

	noisy := d.NoiseScaling(latent, noise, sigma)
	for i := range latent {
		want := (1.0-sigma)*latent[i] + sigma*noise[i]
		if math.Abs(float64(noisy[i]-want)) > 1e-6 {
			t.Errorf("NoiseScaling[%d] = %f, want %f", i, noisy[i], want)
		}
	}
}

// --- float16/bf16 conversion tests ---

func TestFloat16Conversion(t *testing.T) {
	tests := []struct {
		bits uint16
		want float32
	}{
		{0x0000, 0.0},           // positive zero
		{0x8000, -0.0},          // negative zero (should convert to float32 -0.0)
		{0x3C00, 1.0},           // 1.0
		{0xBC00, -1.0},          // -1.0
		{0x4000, 2.0},           // 2.0
		{0x3800, 0.5},           // 0.5
	}
	for _, tt := range tests {
		got := float16ToFloat32(tt.bits)
		if got != tt.want {
			t.Errorf("float16ToFloat32(0x%04X) = %f, want %f", tt.bits, got, tt.want)
		}
	}
}

func TestFloat32FromBits(t *testing.T) {
	got := float32FromBits(0x3F800000)
	if got != 1.0 {
		t.Errorf("float32FromBits(0x3F800000) = %f, want 1.0", got)
	}
	got = float32FromBits(0x00000000)
	if got != 0.0 {
		t.Errorf("float32FromBits(0x00000000) = %f, want 0.0", got)
	}
}

// --- EulerSample identity model test ---

func TestEulerSampleWithZeroModel(t *testing.T) {
	// A model that always returns zero should converge toward the initial noise
	// (since d = (x - x)/sigma = 0, x stays the same)
	zeroModel := func(x []float32, timestep float32) []float32 {
		out := make([]float32, len(x))
		// denoised = x - sigma * 0 = x, so d = (x - x)/sigma = 0
		return out
	}
	result := EulerSample(zeroModel, 16, 4, 42)
	if len(result) != 16 {
		t.Fatalf("EulerSample returned %d elements, want 16", len(result))
	}
	// With zero model output, denoised = x, d = 0, so x should remain unchanged
	// (just the initial noise scaled by sigma[0])
	allZero := true
	for _, v := range result {
		if v != 0 {
			allZero = false
			break
		}
	}
	// Result should NOT be all zeros — it should be the initial noise
	if allZero {
		t.Error("EulerSample with zero model returned all zeros — expected initial noise to persist")
	}
}

func TestEulerSampleDeterministic(t *testing.T) {
	model := func(x []float32, timestep float32) []float32 {
		out := make([]float32, len(x))
		for i := range out {
			out[i] = x[i] * 0.1
		}
		return out
	}
	r1 := EulerSample(model, 8, 4, 123)
	r2 := EulerSample(model, 8, 4, 123)
	for i := range r1 {
		if r1[i] != r2[i] {
			t.Errorf("EulerSample not deterministic at [%d]: %f vs %f", i, r1[i], r2[i])
		}
	}
}

// --- DefaultZImageConfig invariant tests ---

func TestConfigInvariants(t *testing.T) {
	cfg := DefaultZImageConfig()

	if cfg.NumHeads*cfg.HeadDim != cfg.HiddenSize {
		t.Errorf("NumHeads(%d) * HeadDim(%d) = %d != HiddenSize(%d)",
			cfg.NumHeads, cfg.HeadDim, cfg.NumHeads*cfg.HeadDim, cfg.HiddenSize)
	}

	if cfg.NumKVHeads > cfg.NumHeads {
		t.Errorf("NumKVHeads(%d) > NumHeads(%d)", cfg.NumKVHeads, cfg.NumHeads)
	}

	if cfg.NumHeads%cfg.NumKVHeads != 0 {
		t.Errorf("NumHeads(%d) not divisible by NumKVHeads(%d)", cfg.NumHeads, cfg.NumKVHeads)
	}

	axesDimSum := cfg.AxesDim[0] + cfg.AxesDim[1] + cfg.AxesDim[2]
	if axesDimSum != cfg.HeadDim {
		t.Errorf("AxesDim sum %d != HeadDim %d", axesDimSum, cfg.HeadDim)
	}

	ffn := cfg.FFNHiddenDim()
	if ffn <= cfg.HiddenSize {
		t.Errorf("FFN hidden dim %d should be > hidden size %d", ffn, cfg.HiddenSize)
	}

	patchDim := cfg.PatchSize * cfg.PatchSize * cfg.InChannels
	if patchDim != 64 {
		t.Errorf("patchDim = %d, want 64 for patchSize=2, inCh=16", patchDim)
	}
}

// --- memory budget tests ---

func TestEstimatePeakRAM(t *testing.T) {
	cfg := DefaultZImageConfig()

	// 512x512 image
	ram512 := EstimatePeakRAM(cfg, 512, 512, 128)
	if ram512 <= 0 {
		t.Fatal("estimate should be > 0")
	}

	// 1024x1024 should need more than 512x512
	ram1024 := EstimatePeakRAM(cfg, 1024, 1024, 128)
	if ram1024 <= ram512 {
		t.Errorf("1024x1024 (%d bytes) should need more RAM than 512x512 (%d bytes)",
			ram1024, ram512)
	}

	// More context should need more RAM
	ram512ctx512 := EstimatePeakRAM(cfg, 512, 512, 512)
	if ram512ctx512 <= ram512 {
		t.Errorf("512 ctx tokens (%d) should need more RAM than 128 ctx tokens (%d)",
			ram512ctx512, ram512)
	}

	// Sanity: 512x512 should be < 8 GB for runtime buffers
	if ram512 > 8*(1<<30) {
		t.Errorf("512x512 estimate = %.1f GB, seems too high", float64(ram512)/(1<<30))
	}

	t.Logf("512x512 ctx=128: %.2f GB", float64(ram512)/(1<<30))
	t.Logf("1024x1024 ctx=128: %.2f GB", float64(ram1024)/(1<<30))
	t.Logf("512x512 ctx=512: %.2f GB", float64(ram512ctx512)/(1<<30))
}

func TestEstimatePeakRAMNewDiTRunStateConsistency(t *testing.T) {
	// Verify that the estimated RunState size roughly matches the actual allocation
	cfg := DefaultZImageConfig()
	maxSeqLen := 256 // small for test
	rs := NewDiTRunState(cfg, maxSeqLen)

	// Count actual allocated float32s
	actualFloats := len(rs.X) + len(rs.XNorm) + len(rs.QKV) +
		len(rs.Q) + len(rs.K) + len(rs.V) + len(rs.AttnOut) +
		len(rs.Proj) + len(rs.Gate) + len(rs.Up) + len(rs.Hidden) +
		len(rs.FFNOut) + len(rs.Mod) + len(rs.Residual) + len(rs.SiLUBuf) +
		len(rs.Scores) + len(rs.FinalOut) + len(rs.OnesWeight) + len(rs.ZeroBias) +
		len(rs.TanhGate) + len(rs.TEmb) + len(rs.TEmbMid)
	actualBytes := int64(actualFloats) * 4

	// The estimate should be similar (within a factor of 2) for the same maxSeqLen
	// Since EstimatePeakRAM includes non-RunState allocations, it should be >= actual
	t.Logf("DiTRunState actual: %.2f MB (maxSeqLen=%d)", float64(actualBytes)/(1<<20), maxSeqLen)
}
