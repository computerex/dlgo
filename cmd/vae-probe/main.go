// vae-probe decodes a fixed pseudo-random latent through the VAE to test
// the decoder path in isolation (no DiT involved).
package main

import (
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/computerex/dlgo/models/diffusion"
)

func main() {
	vaePath := os.Args[1]
	outPath := os.Args[2]
	latentH, latentW := 32, 32

	vae, err := diffusion.LoadVAEDecoder(vaePath)
	if err != nil {
		log.Fatalf("load vae: %v", err)
	}

	rng := rand.New(rand.NewSource(7))
	latent := make([]float32, 16*latentH*latentW)
	for i := range latent {
		latent[i] = float32(rng.NormFloat64()) * 0.6
	}
	log.Printf("latent: min=%.4f max=%.4f mean=%.4f", minf(latent), maxf(latent), meanf(latent))

	pixels := diffusion.VAEDecode(vae, latent, latentH, latentW)
	log.Printf("pixels: min=%.4f max=%.4f mean=%.4f", minf(pixels), maxf(pixels), meanf(pixels))

	W, H := latentW*8, latentH*8
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			r := clamp(pixels[0*H*W+y*W+x])
			g := clamp(pixels[1*H*W+y*W+x])
			b := clamp(pixels[2*H*W+y*W+x])
			img.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}
	f, err := os.Create(outPath)
	if err != nil {
		log.Fatalf("create: %v", err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		log.Fatalf("encode: %v", err)
	}
	log.Printf("saved %s", outPath)
}

func clamp(v float32) uint8 {
	x := int(v*255.0 + 0.5)
	if x < 0 {
		return 0
	}
	if x > 255 {
		return 255
	}
	return uint8(x)
}

func minf(s []float32) float32 {
	m := float32(math.Inf(1))
	for _, v := range s {
		if v < m {
			m = v
		}
	}
	return m
}
func maxf(s []float32) float32 {
	m := float32(math.Inf(-1))
	for _, v := range s {
		if v > m {
			m = v
		}
	}
	return m
}
func meanf(s []float32) float64 {
	var t float64
	for _, v := range s {
		t += float64(v)
	}
	return t / float64(len(s))
}
