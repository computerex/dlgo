//go:build cgo && vulkan

package diffusion

import (
	"fmt"
	"log"
	"runtime"
	"time"
)

// GenerateImageBatch renders several prompts in a single process, loading the
// text encoder, DiT and VAE exactly once and keeping the DiT resident across
// prompts. It exists to amortise the one-off model-load cost — dominated by the
// DiT weight upload, tens of seconds — over many small images such as sprites:
// N images cost load + N*render instead of N*(load + render).
//
// Returns one float32 RGB [3][H][W] buffer per prompt (same layout as
// GenerateImagePixels). The output for a given prompt is byte-identical to
// calling GenerateImagePixels for that prompt alone with the same config (same
// seed/noise), so batch mode changes only the wall-clock cost, not the result.
//
// Requires DiT + VAE + run-state to fit in VRAM together (small resolutions,
// e.g. <=512x512). For larger sizes, or CPU inference, call GenerateImagePixels
// per prompt instead.
func GenerateImageBatch(ditPath, vaePath, llmPath string, prompts []string, cfg ImageGenConfig) ([][]float32, error) {
	totalStart := time.Now()
	if len(prompts) == 0 {
		return nil, fmt.Errorf("batch: no prompts")
	}

	ditCfg := DefaultZImageConfig()
	if err := CheckDiffusionMemoryBudget(ditCfg, cfg.Width, cfg.Height, 512); err != nil {
		return nil, err
	}

	// 1. Load the text encoder once and encode every prompt up front, then free
	// it before the DiT is loaded (same RAM strategy as the single-image path).
	log.Println("Batch: loading text encoder...")
	te, err := NewTextEncoder(llmPath, 512)
	if err != nil {
		return nil, fmt.Errorf("load text encoder: %w", err)
	}
	contexts := make([][]float32, len(prompts))
	contextLens := make([]int, len(prompts))
	maxCtxLen := 0
	for i, p := range prompts {
		log.Printf("Batch: encoding prompt %d/%d", i+1, len(prompts))
		c, cl := te.Encode(p)
		contexts[i], contextLens[i] = c, cl
		if cl > maxCtxLen {
			maxCtxLen = cl
		}
	}
	te.Pipeline.Model.Close()
	te.Pipeline = nil
	te = nil
	runtime.GC()
	log.Println("Batch: text encoder freed")

	// 2. Load the DiT and VAE once.
	log.Println("Batch: loading DiT model...")
	dit, err := LoadDiTModel(ditPath)
	if err != nil {
		return nil, fmt.Errorf("load DiT: %w", err)
	}
	defer dit.MmapFile.Close()

	log.Println("Batch: loading VAE decoder...")
	vae, err := LoadVAEDecoder(vaePath)
	if err != nil {
		return nil, fmt.Errorf("load VAE: %w", err)
	}

	// 3. Latent / sequence geometry, sized for the longest prompt.
	latentH := cfg.Height / 8
	latentW := cfg.Width / 8
	latentSize := dit.Config.InChannels * latentH * latentW
	patchSize := dit.Config.PatchSize
	nImgTokens := (latentH / patchSize) * (latentW / patchSize)
	maxSeqLen := (maxCtxLen + dit.Config.SeqMultiOf) + (nImgTokens + dit.Config.SeqMultiOf)

	rs := NewDiTRunState(dit.Config, 1) // GPU path only uses CPU RunState for preprocessing

	// 4. GPU setup that keeps the DiT resident for VAE decoding.
	log.Printf("Batch: setting up GPU (maxSeq=%d)...", maxSeqLen)
	cleanup, modelFn, setContext, prepareVAE, vaeFn, gpuErr := setupDiffusionGPU(
		dit, rs, vae, cfg, contexts[0], contextLens[0], latentH, latentW, maxSeqLen, true)
	if gpuErr != nil {
		return nil, fmt.Errorf("GPU setup: %w", gpuErr)
	}
	defer cleanup()

	// Allocate VAE scratch once up front (DiT stays resident thanks to keepDiTForVAE).
	if err := prepareVAE(); err != nil {
		return nil, fmt.Errorf("prepare VAE: %w", err)
	}

	// 5. Denoise + decode each prompt, reusing the uploaded weights.
	pixels := make([][]float32, len(prompts))
	for i := range prompts {
		pStart := time.Now()
		setContext(contexts[i], contextLens[i])
		latent := EulerSample(modelFn, latentSize, cfg.Steps, cfg.Seed)
		if err := validateLatent(latent); err != nil {
			return nil, fmt.Errorf("prompt %d (%q): %w", i, shortPrompt(prompts[i]), err)
		}
		pixels[i] = vaeFn(latent, latentH, latentW)
		log.Printf("Batch: prompt %d/%d rendered in %v", i+1, len(prompts), time.Since(pStart))
	}

	log.Printf("Batch: %d image(s) in %v", len(prompts), time.Since(totalStart))
	return pixels, nil
}

// shortPrompt truncates a prompt for log and error lines.
func shortPrompt(p string) string {
	if len(p) > 40 {
		return p[:40] + "..."
	}
	return p
}
