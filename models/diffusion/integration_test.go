package diffusion

import (
	"os"
	"runtime"
	"testing"
)

const (
	ditModelPath = `C:\users\mohd\downloads\z-image-turbo-Q4_K_M.gguf`
	vaePath      = `C:\projects\sd-cpp\models\ae.safetensors`
	llmPath      = `C:\projects\sd-cpp\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf`
)

func skipIfNoModel(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("model not found: %s", path)
	}
}

// TestLoadDiTModel loads the real GGUF and verifies tensor structure.
// This is safe: only mmap (read-only), no inference, no large allocations.
func TestLoadDiTModel(t *testing.T) {
	skipIfNoModel(t, ditModelPath)

	m, err := LoadDiTModel(ditModelPath)
	if err != nil {
		t.Fatalf("LoadDiTModel: %v", err)
	}
	defer m.MmapFile.Close()

	cfg := m.Config

	// Verify layer counts
	if len(m.ContextRefiner) != cfg.NumRefinerLayers {
		t.Errorf("ContextRefiner: got %d layers, want %d", len(m.ContextRefiner), cfg.NumRefinerLayers)
	}
	if len(m.NoiseRefiner) != cfg.NumRefinerLayers {
		t.Errorf("NoiseRefiner: got %d layers, want %d", len(m.NoiseRefiner), cfg.NumRefinerLayers)
	}
	if len(m.MainLayers) != cfg.NumLayers {
		t.Errorf("MainLayers: got %d layers, want %d", len(m.MainLayers), cfg.NumLayers)
	}

	// Verify embedder shapes
	if m.XEmbedWeight == nil {
		t.Fatal("XEmbedWeight is nil")
	}
	if m.XEmbedBias == nil {
		t.Fatal("XEmbedBias is nil")
	}
	expectedPatchDim := cfg.PatchSize * cfg.PatchSize * cfg.InChannels
	if len(m.XEmbedBias) != cfg.HiddenSize {
		t.Errorf("XEmbedBias len = %d, want %d", len(m.XEmbedBias), cfg.HiddenSize)
	}

	// Verify timestep embedder
	if m.TEmbedMLP0Weight == nil {
		t.Fatal("TEmbedMLP0Weight is nil")
	}
	if len(m.TEmbedMLP0Bias) != 1024 {
		t.Errorf("TEmbedMLP0Bias len = %d, want 1024", len(m.TEmbedMLP0Bias))
	}
	if len(m.TEmbedMLP2Bias) != cfg.AdaLNEmbedDim {
		t.Errorf("TEmbedMLP2Bias len = %d, want %d", len(m.TEmbedMLP2Bias), cfg.AdaLNEmbedDim)
	}

	// Verify caption embedder
	if len(m.CapEmbedNormWeight) != cfg.CapFeatDim {
		t.Errorf("CapEmbedNormWeight len = %d, want %d", len(m.CapEmbedNormWeight), cfg.CapFeatDim)
	}
	if len(m.CapEmbedLinBias) != cfg.HiddenSize {
		t.Errorf("CapEmbedLinBias len = %d, want %d", len(m.CapEmbedLinBias), cfg.HiddenSize)
	}

	// Verify pad tokens
	if len(m.XPadToken) != cfg.HiddenSize {
		t.Errorf("XPadToken len = %d, want %d", len(m.XPadToken), cfg.HiddenSize)
	}
	if len(m.CapPadToken) != cfg.HiddenSize {
		t.Errorf("CapPadToken len = %d, want %d", len(m.CapPadToken), cfg.HiddenSize)
	}

	// Verify each main layer has all expected tensors
	for i, layer := range m.MainLayers {
		if layer.AttnQKV == nil {
			t.Errorf("MainLayers[%d].AttnQKV is nil", i)
		}
		if layer.AttnOut == nil {
			t.Errorf("MainLayers[%d].AttnOut is nil", i)
		}
		if layer.FFNGate == nil {
			t.Errorf("MainLayers[%d].FFNGate is nil", i)
		}
		if layer.FFNDown == nil {
			t.Errorf("MainLayers[%d].FFNDown is nil", i)
		}
		if layer.FFNUp == nil {
			t.Errorf("MainLayers[%d].FFNUp is nil", i)
		}
		if layer.AdaLNWeight == nil {
			t.Errorf("MainLayers[%d].AdaLNWeight is nil (should have adaLN)", i)
		}
		if len(layer.QNorm) != cfg.HeadDim {
			t.Errorf("MainLayers[%d].QNorm len = %d, want %d", i, len(layer.QNorm), cfg.HeadDim)
		}
		if len(layer.KNorm) != cfg.HeadDim {
			t.Errorf("MainLayers[%d].KNorm len = %d, want %d", i, len(layer.KNorm), cfg.HeadDim)
		}
		if len(layer.AttnNorm1) != cfg.HiddenSize {
			t.Errorf("MainLayers[%d].AttnNorm1 len = %d, want %d", i, len(layer.AttnNorm1), cfg.HiddenSize)
		}
	}

	// Verify context refiner has NO adaLN
	for i, layer := range m.ContextRefiner {
		if layer.AdaLNWeight != nil {
			t.Errorf("ContextRefiner[%d].AdaLNWeight should be nil", i)
		}
	}

	// Verify noise refiner HAS adaLN
	for i, layer := range m.NoiseRefiner {
		if layer.AdaLNWeight == nil {
			t.Errorf("NoiseRefiner[%d].AdaLNWeight should not be nil", i)
		}
	}

	// Verify final layer
	if m.FinalAdaLNWeight == nil {
		t.Fatal("FinalAdaLNWeight is nil")
	}
	if m.FinalLinWeight == nil {
		t.Fatal("FinalLinWeight is nil")
	}
	if len(m.FinalLinBias) != expectedPatchDim {
		t.Errorf("FinalLinBias len = %d, want %d (patchDim)", len(m.FinalLinBias), expectedPatchDim)
	}

	t.Logf("DiT model loaded: %d+%d+%d layers, hidden=%d, patchDim=%d",
		len(m.ContextRefiner), len(m.NoiseRefiner), len(m.MainLayers),
		cfg.HiddenSize, expectedPatchDim)
}

// TestLoadVAEDecoder loads the real VAE safetensors and verifies structure.
func TestLoadVAEDecoder(t *testing.T) {
	skipIfNoModel(t, vaePath)

	vae, err := LoadVAEDecoder(vaePath)
	if err != nil {
		t.Fatalf("LoadVAEDecoder: %v", err)
	}

	// Verify it has some weights loaded
	if vae == nil {
		t.Fatal("VAE decoder is nil")
	}

	t.Logf("VAE decoder loaded successfully")
}

// TestMemoryBudgetWithRealConfig tests the budget check with the real system.
func TestMemoryBudgetWithRealConfig(t *testing.T) {
	cfg := DefaultZImageConfig()

	// 512x512 should be safe on a 32GB system
	err := CheckDiffusionMemoryBudget(cfg, 512, 512, 128)
	if err != nil {
		t.Logf("512x512 budget check: %v (may be legitimate on low-RAM system)", err)
	} else {
		t.Log("512x512 budget check: PASS")
	}

	// Get the estimate for information
	ram := EstimatePeakRAM(cfg, 512, 512, 128)
	t.Logf("512x512 estimated peak RAM: %.2f GB", float64(ram)/(1<<30))
}

// TestDiTForwardTiny runs a single forward pass at tiny resolution (64x64 image)
// with the real model weights. This verifies the entire forward pass logic
// without excessive memory usage (~36 MB runtime buffers).
func TestDiTForwardTiny(t *testing.T) {
	skipIfNoModel(t, ditModelPath)

	m, err := LoadDiTModel(ditModelPath)
	if err != nil {
		t.Fatalf("LoadDiTModel: %v", err)
	}
	defer m.MmapFile.Close()

	cfg := m.Config

	// 64x64 image → 8x8 latent
	imgW, imgH := 64, 64
	latentH := imgH / 8
	latentW := imgW / 8
	latentCh := cfg.InChannels // 16

	// Fake latent: small random-ish values
	latentSize := latentCh * latentH * latentW
	x := make([]float32, latentSize)
	for i := range x {
		x[i] = float32(i%17) * 0.01 // deterministic pseudo-random
	}

	// Fake context: 4 tokens of dim=capFeatDim
	contextLen := 4
	context := make([]float32, contextLen*cfg.CapFeatDim)
	for i := range context {
		context[i] = float32(i%31) * 0.001
	}

	// Allocate run state for this tiny resolution
	patchSize := cfg.PatchSize
	hPatches := latentH / patchSize
	wPatches := latentW / patchSize
	nImgTokens := hPatches * wPatches
	nTxtPadded := contextLen + boundMod(contextLen, cfg.SeqMultiOf)
	nImgPadded := nImgTokens + boundMod(nImgTokens, cfg.SeqMultiOf)
	maxSeqLen := nTxtPadded + nImgPadded

	t.Logf("Tiny forward: %dx%d latent, %d img tokens, %d txt tokens (padded), maxSeqLen=%d",
		latentH, latentW, nImgTokens, nTxtPadded, maxSeqLen)

	rs := NewDiTRunState(cfg, maxSeqLen)

	// Single forward pass
	timestep := float32(0.5)
	out := DiTForward(m, rs, x, timestep, context, contextLen, latentH, latentW)

	// Verify output shape: should be [outCh, H, W]
	expectedLen := cfg.OutChannels * latentH * latentW
	if len(out) != expectedLen {
		t.Fatalf("DiTForward output len = %d, want %d (outCh=%d, H=%d, W=%d)",
			len(out), expectedLen, cfg.OutChannels, latentH, latentW)
	}

	// Verify output is not all zeros (model actually did something)
	allZero := true
	for _, v := range out {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("forward output is all zeros")
	}

	// Verify no NaN or Inf
	for i, v := range out {
		if v != v { // NaN check
			t.Fatalf("output[%d] is NaN", i)
		}
		if v > 1e30 || v < -1e30 {
			t.Fatalf("output[%d] = %e (overflow?)", i, v)
		}
	}

	t.Logf("Forward pass OK: output[0:5] = %v", out[:5])
}

// TestEndToEnd256x256 runs the full pipeline up to DiT sampling at 256x256.
// VAE decode is skipped because naive conv2d is too slow at large spatial dims.
// Run manually:
//   go test ./models/diffusion/ -v -run TestEndToEnd256x256 -timeout 600s
func TestEndToEnd256x256(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping end-to-end in short mode")
	}
	skipIfNoModel(t, ditModelPath)
	skipIfNoModel(t, llmPath)

	ditCfg := DefaultZImageConfig()
	if err := CheckDiffusionMemoryBudget(ditCfg, 256, 256, 512); err != nil {
		t.Fatalf("memory budget: %v", err)
	}

	// 1. Text encoding
	t.Log("Loading text encoder...")
	te, err := NewTextEncoder(llmPath, 512)
	if err != nil {
		t.Fatalf("NewTextEncoder: %v", err)
	}
	context, contextLen := te.Encode("a red cat")
	te.Pipeline.Model.Close()
	te.Pipeline = nil
	te = nil
	runtime.GC()
	t.Logf("Encoded %d tokens", contextLen)

	// 2. Load DiT
	dit, err := LoadDiTModel(ditModelPath)
	if err != nil {
		t.Fatalf("LoadDiTModel: %v", err)
	}
	defer dit.MmapFile.Close()

	// 3. Run 1-step Euler sampling at 256x256
	latentH, latentW := 256/8, 256/8
	latentCh := dit.Config.InChannels
	latentSize := latentCh * latentH * latentW

	patchSize := dit.Config.PatchSize
	hPatches := latentH / patchSize
	wPatches := latentW / patchSize
	nImgTokens := hPatches * wPatches
	nTxtPadded := contextLen + boundMod(contextLen, dit.Config.SeqMultiOf)
	nImgPadded := nImgTokens + boundMod(nImgTokens, dit.Config.SeqMultiOf)
	maxSeqLen := nTxtPadded + nImgPadded
	rs := NewDiTRunState(dit.Config, maxSeqLen)

	modelFn := func(x []float32, timestep float32) []float32 {
		return DiTForward(dit, rs, x, timestep, context, contextLen, latentH, latentW)
	}

	t.Log("Sampling 1 step at 256x256...")
	latent := EulerSample(modelFn, latentSize, 1, 42)

	// Verify latent output
	if len(latent) != latentSize {
		t.Fatalf("latent size = %d, want %d", len(latent), latentSize)
	}

	// Check for NaN/Inf
	for i, v := range latent {
		if v != v {
			t.Fatalf("latent[%d] is NaN", i)
		}
		if v > 1e30 || v < -1e30 {
			t.Fatalf("latent[%d] = %e (overflow?)", i, v)
		}
	}

	// Verify latent has non-trivial values
	var sum float64
	for _, v := range latent {
		sum += float64(v) * float64(v)
	}
	if sum < 1e-10 {
		t.Error("latent is near-zero (model did nothing?)")
	}

	t.Logf("Sampling OK: latent L2=%.4f, latent[0:5]=%v", sum, latent[:5])
	t.Log("NOTE: VAE decode skipped (naive conv2d too slow at 256x256). Optimize conv2d for end-to-end.")
}

// TestVAEDecode256x256 loads the real VAE and decodes a random latent to 256×256 RGB.
// Run manually:
//
//	go test ./models/diffusion/ -v -run TestVAEDecode256x256 -timeout 600s
func TestVAEDecode256x256(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping VAE decode in short mode")
	}
	skipIfNoModel(t, vaePath)

	vae, err := LoadVAEDecoder(vaePath)
	if err != nil {
		t.Fatalf("LoadVAEDecoder: %v", err)
	}

	// 256×256 image → 32×32 latent
	latentH, latentW := 32, 32
	latentCh := vae.Config.ZChannels // 16

	// Deterministic pseudo-random latent
	latent := make([]float32, latentCh*latentH*latentW)
	for i := range latent {
		latent[i] = float32(i%37)*0.01 - 0.18
	}

	t.Logf("VAE decode: [%d, %d, %d] → [3, 256, 256]", latentCh, latentH, latentW)

	rgb := VAEDecode(vae, latent, latentH, latentW)

	// Verify output shape: [3, 256, 256]
	expectedLen := 3 * 256 * 256
	if len(rgb) != expectedLen {
		t.Fatalf("VAE output len = %d, want %d", len(rgb), expectedLen)
	}

	// Check for NaN/Inf
	for i, v := range rgb {
		if v != v {
			t.Fatalf("rgb[%d] is NaN", i)
		}
		if v > 1e30 || v < -1e30 {
			t.Fatalf("rgb[%d] = %e (overflow?)", i, v)
		}
	}

	// Values should be in [0, 1] range (post-processed)
	for i, v := range rgb {
		if v < 0 || v > 1 {
			t.Fatalf("rgb[%d] = %f, want in [0,1]", i, v)
		}
	}

	// Check it's not all zeros or all ones
	var sum float64
	for _, v := range rgb {
		sum += float64(v)
	}
	mean := sum / float64(len(rgb))
	if mean < 0.01 || mean > 0.99 {
		t.Errorf("rgb mean = %.4f (looks degenerate)", mean)
	}

	t.Logf("VAE decode OK: %d pixels, mean=%.4f, rgb[0:5]=%v", len(rgb), mean, rgb[:5])
}

// TestFullPipeline256x256 runs the complete text→DiT→VAE→PNG pipeline at 256×256.
// Run manually:
//
//	go test ./models/diffusion/ -v -run TestFullPipeline256x256 -timeout 600s
func TestFullPipeline256x256(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full pipeline in short mode")
	}
	skipIfNoModel(t, ditModelPath)
	skipIfNoModel(t, vaePath)
	skipIfNoModel(t, llmPath)

	imgW, imgH := 256, 256
	prompt := "a red cat"

	ditCfg := DefaultZImageConfig()
	if err := CheckDiffusionMemoryBudget(ditCfg, imgW, imgH, 512); err != nil {
		t.Fatalf("memory budget: %v", err)
	}

	// 1. Text encoding
	t.Log("Step 1: text encoding...")
	te, err := NewTextEncoder(llmPath, 512)
	if err != nil {
		t.Fatalf("NewTextEncoder: %v", err)
	}
	context, contextLen := te.Encode(prompt)
	te.Pipeline.Model.Close()
	te.Pipeline = nil
	te = nil
	runtime.GC()
	t.Logf("  Encoded %d tokens for %q", contextLen, prompt)

	// 2. Load DiT
	t.Log("Step 2: loading DiT...")
	dit, err := LoadDiTModel(ditModelPath)
	if err != nil {
		t.Fatalf("LoadDiTModel: %v", err)
	}
	defer dit.MmapFile.Close()

	// 3. DiT sampling (1 step, turbo model)
	t.Log("Step 3: DiT sampling (1 step)...")
	latentH, latentW := imgH/8, imgW/8
	latentCh := dit.Config.InChannels
	latentSize := latentCh * latentH * latentW

	patchSize := dit.Config.PatchSize
	hPatches := latentH / patchSize
	wPatches := latentW / patchSize
	nImgTokens := hPatches * wPatches
	nTxtPadded := contextLen + boundMod(contextLen, dit.Config.SeqMultiOf)
	nImgPadded := nImgTokens + boundMod(nImgTokens, dit.Config.SeqMultiOf)
	maxSeqLen := nTxtPadded + nImgPadded
	rs := NewDiTRunState(dit.Config, maxSeqLen)

	modelFn := func(x []float32, timestep float32) []float32 {
		return DiTForward(dit, rs, x, timestep, context, contextLen, latentH, latentW)
	}

	latent := EulerSample(modelFn, latentSize, 1, 42)
	t.Logf("  Latent: [%d, %d, %d], L2=%.4f", latentCh, latentH, latentW, l2norm(latent))

	// 4. VAE decode
	t.Log("Step 4: VAE decode...")
	vae, err := LoadVAEDecoder(vaePath)
	if err != nil {
		t.Fatalf("LoadVAEDecoder: %v", err)
	}

	rgb := VAEDecode(vae, latent, latentH, latentW)
	if len(rgb) != 3*imgH*imgW {
		t.Fatalf("VAE output len = %d, want %d", len(rgb), 3*imgH*imgW)
	}

	// Verify pixels
	for i, v := range rgb {
		if v != v {
			t.Fatalf("rgb[%d] is NaN", i)
		}
	}
	var sum float64
	for _, v := range rgb {
		sum += float64(v)
	}
	mean := sum / float64(len(rgb))
	t.Logf("  RGB: mean=%.4f", mean)

	// 5. Save PNG
	outPath := "test_output_256x256.png"
	t.Logf("Step 5: saving %s...", outPath)
	if err := savePNG(rgb, imgW, imgH, outPath); err != nil {
		t.Fatalf("savePNG: %v", err)
	}
	t.Logf("Full pipeline complete: %s", outPath)
}

// TestFullPipeline256x256_4step tests 4-step generation for better quality.
func TestFullPipeline256x256_4step(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full pipeline in short mode")
	}
	skipIfNoModel(t, ditModelPath)
	skipIfNoModel(t, vaePath)
	skipIfNoModel(t, llmPath)

	imgW, imgH := 256, 256
	prompt := "a red cat"

	ditCfg := DefaultZImageConfig()
	if err := CheckDiffusionMemoryBudget(ditCfg, imgW, imgH, 512); err != nil {
		t.Fatalf("memory budget: %v", err)
	}

	t.Log("Step 1: text encoding...")
	te, err := NewTextEncoder(llmPath, 512)
	if err != nil {
		t.Fatalf("NewTextEncoder: %v", err)
	}
	context, contextLen := te.Encode(prompt)
	te.Pipeline.Model.Close()
	te.Pipeline = nil
	te = nil
	runtime.GC()
	t.Logf("  Encoded %d tokens for %q", contextLen, prompt)

	t.Log("Step 2: loading DiT...")
	dit, err := LoadDiTModel(ditModelPath)
	if err != nil {
		t.Fatalf("LoadDiTModel: %v", err)
	}
	defer dit.MmapFile.Close()

	t.Log("Step 3: DiT sampling (4 steps)...")
	latentH, latentW := imgH/8, imgW/8
	latentCh := dit.Config.InChannels
	latentSize := latentCh * latentH * latentW

	patchSize := dit.Config.PatchSize
	hPatches := latentH / patchSize
	wPatches := latentW / patchSize
	nImgTokens := hPatches * wPatches
	nTxtPadded := contextLen + boundMod(contextLen, dit.Config.SeqMultiOf)
	nImgPadded := nImgTokens + boundMod(nImgTokens, dit.Config.SeqMultiOf)
	maxSeqLen := nTxtPadded + nImgPadded
	rs := NewDiTRunState(dit.Config, maxSeqLen)

	modelFn := func(x []float32, timestep float32) []float32 {
		return DiTForward(dit, rs, x, timestep, context, contextLen, latentH, latentW)
	}

	latent := EulerSample(modelFn, latentSize, 4, 42)
	t.Logf("  Latent: [%d, %d, %d], L2=%.4f", latentCh, latentH, latentW, l2norm(latent))

	t.Log("Step 4: VAE decode...")
	vae, err := LoadVAEDecoder(vaePath)
	if err != nil {
		t.Fatalf("LoadVAEDecoder: %v", err)
	}

	rgb := VAEDecode(vae, latent, latentH, latentW)
	if len(rgb) != 3*imgH*imgW {
		t.Fatalf("VAE output len = %d, want %d", len(rgb), 3*imgH*imgW)
	}

	for i, v := range rgb {
		if v != v {
			t.Fatalf("rgb[%d] is NaN", i)
		}
	}
	var sum float64
	for _, v := range rgb {
		sum += float64(v)
	}
	mean := sum / float64(len(rgb))
	t.Logf("  RGB: mean=%.4f", mean)

	outPath := "test_output_256x256_4step.png"
	t.Logf("Step 5: saving %s...", outPath)
	if err := savePNG(rgb, imgW, imgH, outPath); err != nil {
		t.Fatalf("savePNG: %v", err)
	}
	t.Logf("Full pipeline (4 steps) complete: %s", outPath)
}

func l2norm(x []float32) float64 {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	return sum
}
