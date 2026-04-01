package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/diffusion"
)

func main() {
	ditPath := flag.String("diffusion-model", "", "Path to Z-Image DiT GGUF model")
	vaePath := flag.String("vae", "", "Path to VAE safetensors file (ae.safetensors)")
	llmPath := flag.String("llm", "", "Path to text encoder GGUF (Qwen3-4B)")
	prompt := flag.String("p", "a beautiful sunset over the mountains", "Text prompt")
	output := flag.String("o", "output.png", "Output PNG path")
	width := flag.Int("W", 1024, "Image width")
	height := flag.Int("H", 1024, "Image height")
	steps := flag.Int("steps", 8, "Number of sampling steps")
	seed := flag.Int64("seed", 42, "Random seed")
	cfgScale := flag.Float64("cfg-scale", 1.0, "CFG scale (1.0 = no guidance)")
	useGPU := flag.Bool("gpu", false, "Use GPU acceleration for DiT inference")

	flag.Parse()

	if *ditPath == "" || *vaePath == "" || *llmPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: dlgo-image --diffusion-model <dit.gguf> --vae <ae.safetensors> --llm <qwen3.gguf> -p \"prompt\"")
		flag.PrintDefaults()
		os.Exit(1)
	}

	cfg := diffusion.ImageGenConfig{
		Width:    *width,
		Height:   *height,
		Steps:    *steps,
		CFGScale: float32(*cfgScale),
		Seed:     *seed,
		UseGPU:   *useGPU,
	}

	err := diffusion.GenerateImage(*ditPath, *vaePath, *llmPath, *prompt, cfg, *output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Image saved to %s\n", *output)
}
