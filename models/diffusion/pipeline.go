package diffusion

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
	"runtime"
	"time"

	"github.com/computerex/dlgo/mmap"
)

// ImageGenConfig holds parameters for image generation.
type ImageGenConfig struct {
	Width   int
	Height  int
	Steps   int
	CFGScale float32
	Seed    int64
	UseGPU  bool
}

// DefaultImageGenConfig returns default generation parameters matching sd-cli.
func DefaultImageGenConfig() ImageGenConfig {
	return ImageGenConfig{
		Width:    1024,
		Height:   1024,
		Steps:    8,
		CFGScale: 1.0,
		Seed:     42,
	}
}

// EstimatePeakRAM estimates the peak RAM needed for diffusion inference.
// DiT weights are mmap'd (not counted), but DiTRunState, VAE weights, and
// transient per-step allocations all consume real RAM.
func EstimatePeakRAM(ditCfg ZImageConfig, width, height, contextLen int) int64 {
	latentH := height / 8
	latentW := width / 8
	patchSize := ditCfg.PatchSize
	hPatches := latentH / patchSize
	wPatches := latentW / patchSize
	nImgTokens := hPatches * wPatches
	nTxtPadded := contextLen + ditCfg.SeqMultiOf
	nImgPadded := nImgTokens + ditCfg.SeqMultiOf
	maxSeqLen := nTxtPadded + nImgPadded

	hidden := ditCfg.HiddenSize
	ffnDim := ditCfg.FFNHiddenDim()
	qDim := ditCfg.NumHeads * ditCfg.HeadDim
	kvDim := ditCfg.NumKVHeads * ditCfg.HeadDim
	qkvDim := qDim + 2*kvDim
	patchDim := patchSize * patchSize * ditCfg.OutChannels

	// DiTRunState buffers (all float32 = 4 bytes each)
	runState := int64(0)
	runState += int64(maxSeqLen) * int64(hidden)       // X
	runState += int64(maxSeqLen) * int64(hidden)       // XNorm
	runState += int64(maxSeqLen) * int64(qkvDim)       // QKV
	runState += int64(maxSeqLen) * int64(qDim)         // Q
	runState += int64(maxSeqLen) * int64(kvDim)        // K
	runState += int64(maxSeqLen) * int64(kvDim)        // V
	runState += int64(maxSeqLen) * int64(qDim)         // AttnOut
	runState += int64(maxSeqLen) * int64(hidden)       // Proj
	runState += int64(maxSeqLen) * int64(ffnDim)       // Gate
	runState += int64(maxSeqLen) * int64(ffnDim)       // Up
	runState += int64(maxSeqLen) * int64(ffnDim)       // Hidden
	runState += int64(maxSeqLen) * int64(hidden)       // FFNOut
	runState += int64(4 * hidden)                       // Mod
	runState += int64(maxSeqLen) * int64(hidden)       // Residual
	runState += int64(ditCfg.AdaLNEmbedDim)            // SiLUBuf
	runState += int64(maxSeqLen)                        // Scores
	runState += int64(maxSeqLen) * int64(patchDim)     // FinalOut
	runState += int64(hidden)                           // OnesWeight
	runState += int64(hidden)                           // TanhGate
	runState += int64(ditCfg.AdaLNEmbedDim)            // TEmb
	runState += int64(1024)                             // TEmbMid
	runState *= 4 // float32

	// Per-step transient: txtNormed, txt, img (still heap-allocated)
	transient := int64(0)
	transient += int64(contextLen) * int64(ditCfg.CapFeatDim) // txtNormed
	transient += int64(contextLen) * int64(hidden)             // txt
	transient += int64(nImgTokens) * int64(hidden)             // img
	transient *= 4

	// VAE decoder weights (~320MB for FLUX ae.safetensors)
	const vaeEstimate = 320 * (1 << 20) // 320 MB

	// VAE decode transient: conv2d intermediates, roughly 3 * H * W * 512 * 4
	vaeTransient := int64(3) * int64(height) * int64(width) * 512 * 4

	return runState + transient + vaeEstimate + vaeTransient
}

// CheckDiffusionMemoryBudget checks if the system has enough RAM for diffusion.
// Returns nil if OK, error if insufficient.
func CheckDiffusionMemoryBudget(ditCfg ZImageConfig, width, height, contextLen int) error {
	sysInfo, err := mmap.GetSystemMemInfo()
	if err != nil {
		return nil // can't query, skip check
	}

	availRAM := int64(sysInfo.AvailablePhysical)
	const reserveBytes = 2 * (1 << 30) // 2 GB safety margin
	budget := availRAM - reserveBytes
	if budget < 0 {
		budget = 0
	}

	needed := EstimatePeakRAM(ditCfg, width, height, contextLen)

	log.Printf("Memory budget: need ~%.1f GB, available ~%.1f GB (%.1f GB total, %.1f GB free)",
		float64(needed)/(1<<30), float64(budget)/(1<<30),
		float64(sysInfo.TotalPhysical)/(1<<30), float64(availRAM)/(1<<30))

	if needed > budget {
		return fmt.Errorf(
			"insufficient memory: diffusion needs ~%.1f GB but only ~%.1f GB available "+
				"(%.1f GB free, 2 GB reserved). Try a smaller resolution (current: %dx%d)",
			float64(needed)/(1<<30), float64(budget)/(1<<30),
			float64(availRAM)/(1<<30), width, height)
	}
	return nil
}

// GenerateImage runs the full Z-Image pipeline:
// text encoder → DiT denoising → VAE decode → PNG
func GenerateImage(
	ditPath string,
	vaePath string,
	llmPath string,
	prompt string,
	cfg ImageGenConfig,
	outputPath string,
) error {
	totalStart := time.Now()

	// 0. Memory budget check (estimate with max context length)
	ditCfg := DefaultZImageConfig()
	if err := CheckDiffusionMemoryBudget(ditCfg, cfg.Width, cfg.Height, 512); err != nil {
		return err
	}

	// 1. Load text encoder, encode, then free it before loading DiT
	log.Println("Loading text encoder...")
	te, err := NewTextEncoder(llmPath, 512)
	if err != nil {
		return fmt.Errorf("load text encoder: %w", err)
	}
	log.Printf("Text encoder loaded in %v", time.Since(totalStart))

	// 2. Encode text
	log.Println("Encoding prompt...")
	encStart := time.Now()
	context, contextLen := te.Encode(prompt)
	log.Printf("Text encoded: %d tokens in %v", contextLen, time.Since(encStart))

	// Free text encoder before loading DiT to save RAM
	te.Pipeline.Model.Close()
	te.Pipeline = nil
	te = nil
	runtime.GC()
	log.Println("Text encoder freed")

	// 3. Load DiT model (stays quantized — no pre-dequantization)
	log.Println("Loading DiT model...")
	loadStart := time.Now()
	dit, err := LoadDiTModel(ditPath)
	if err != nil {
		return fmt.Errorf("load DiT: %w", err)
	}
	log.Printf("DiT loaded in %v", time.Since(loadStart))

	// 4. Load VAE
	log.Println("Loading VAE decoder...")
	loadStart = time.Now()
	vae, err := LoadVAEDecoder(vaePath)
	if err != nil {
		return fmt.Errorf("load VAE: %w", err)
	}
	log.Printf("VAE loaded in %v", time.Since(loadStart))

	// 5. Compute latent dimensions
	// VAE uses 8× spatial downscale
	latentH := cfg.Height / 8
	latentW := cfg.Width / 8
	latentCh := dit.Config.InChannels // 16
	latentSize := latentCh * latentH * latentW

	log.Printf("Latent: [%d, %d, %d] = %d elements", latentCh, latentH, latentW, latentSize)

	// 6. Allocate DiT run state
	patchSize := dit.Config.PatchSize
	hPatches := latentH / patchSize
	wPatches := latentW / patchSize
	nImgTokens := hPatches * wPatches
	nTxtPaddedMax := contextLen + dit.Config.SeqMultiOf
	nImgPaddedMax := nImgTokens + dit.Config.SeqMultiOf
	maxSeqLen := nTxtPaddedMax + nImgPaddedMax
	rs := NewDiTRunState(dit.Config, maxSeqLen)

	// 7. GPU setup (if requested)
	var modelFn func(x []float32, timestep float32) []float32
	var gpuVaeFn func([]float32, int, int) []float32
	gpuCleanup, gpuModelFn, gpuVaeDecodeFn, gpuErr := setupDiffusionGPU(dit, rs, vae, cfg, context, contextLen, latentH, latentW, maxSeqLen)
	if gpuErr != nil {
		return fmt.Errorf("GPU setup: %w", gpuErr)
	}
	if gpuCleanup != nil {
		defer gpuCleanup()
	}

	if gpuModelFn != nil {
		modelFn = gpuModelFn
	} else {
		modelFn = func(x []float32, timestep float32) []float32 {
			return DiTForward(dit, rs, x, timestep, context, contextLen, latentH, latentW)
		}
	}
	gpuVaeFn = gpuVaeDecodeFn

	// 8. Run Euler sampler
	// 8. Run Euler sampler
	log.Printf("Sampling: %d steps, seed=%d, cfg_scale=%.1f", cfg.Steps, cfg.Seed, cfg.CFGScale)
	sampleStart := time.Now()

	latent := EulerSample(modelFn, latentSize, cfg.Steps, cfg.Seed)
	log.Printf("Sampling done in %v", time.Since(sampleStart))

	// 9. VAE decode
	log.Println("Decoding with VAE...")
	decStart := time.Now()
	var pixels []float32
	if gpuVaeFn != nil {
		pixels = gpuVaeFn(latent, latentH, latentW)
	} else {
		pixels = VAEDecode(vae, latent, latentH, latentW)
	}
	log.Printf("VAE decode done in %v", time.Since(decStart))

	// 10. Save as PNG
	log.Printf("Saving to %s...", outputPath)
	err = savePNG(pixels, cfg.Width, cfg.Height, outputPath)
	if err != nil {
		return fmt.Errorf("save PNG: %w", err)
	}

	log.Printf("Total generation time: %v", time.Since(totalStart))
	return nil
}

// savePNG saves float32 RGB pixels [3, H, W] in [0,1] range to PNG.
func savePNG(pixels []float32, W, H int, path string) error {
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			r := pixels[0*H*W+y*W+x]
			g := pixels[1*H*W+y*W+x]
			b := pixels[2*H*W+y*W+x]
			img.Set(x, y, color.RGBA{
				R: clampByte(r),
				G: clampByte(g),
				B: clampByte(b),
				A: 255,
			})
		}
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return png.Encode(f, img)
}

func clampByte(v float32) uint8 {
	x := int(v*255.0 + 0.5)
	if x < 0 {
		return 0
	}
	if x > 255 {
		return 255
	}
	return uint8(x)
}
