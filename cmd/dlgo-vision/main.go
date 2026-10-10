package main

// dlgo-vision is an end-to-end multimodal demo: it loads a vision projector
// (mmproj) and a text model, encodes an image, splices the resulting embedding
// block into the prompt at the <image> marker, and generates a continuation.
//
// Usage:
//
//	dlgo-vision -mmproj <mmproj.gguf> -model <model.gguf> \
//	    -image <png|white|black|gray|cb> [-n <px>] \
//	    -prompt "Describe: <image>" [-max-tokens 64]
//
// The projector output dimension must equal the text model embedding length
// (e.g. llava-1.5-7b mmproj = 4096 -> fits Qwen3.5-9B).

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"os"
	"strings"

	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/vision"
)

func main() {
	mmproj := flag.String("mmproj", "", "path to mmproj GGUF (vision projector)")
	model := flag.String("model", "", "path to text model GGUF")
	imagePath := flag.String("image", "cb", "image file, or synthetic: white|black|gray|cb|red|green|blue")
	n := flag.Int("n", 336, "synthetic image edge size in pixels")
	prompt := flag.String("prompt", "Describe this image in detail:\n<image>", "prompt with <image> marker")
	maxTokens := flag.Int("max-tokens", 64, "max tokens to generate")
	temp := flag.Float64("temp", 0, "sampling temperature (0 = greedy)")
	seed := flag.Int64("seed", 0, "sampling seed")
	flag.Parse()

	if *mmproj == "" || *model == "" {
		fmt.Fprintln(os.Stderr, "usage: dlgo-vision -mmproj <file> -model <file> [-image png|white|black|gray|cb] [-n px] [-prompt ...]")
		os.Exit(2)
	}

	var img image.Image
	switch strings.ToLower(*imagePath) {
	case "white", "black", "gray", "cb", "red", "green", "blue":
		img = synthetic(*imagePath, *n)
	default:
		f, err := os.Open(*imagePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "open image: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()
		im, _, err := image.Decode(f)
		if err != nil {
			fmt.Fprintf(os.Stderr, "decode image: %v\n", err)
			os.Exit(1)
		}
		img = im
	}

	enc, err := vision.LoadMmproj(*mmproj)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load mmproj: %v\n", err)
		os.Exit(1)
	}
	defer enc.Close()

	if !strings.Contains(*prompt, "<image>") {
		fmt.Fprintln(os.Stderr, "warning: prompt has no <image> marker; the image will not be used")
	}

	fmt.Fprintf(os.Stderr, "mmproj: proj=%s image=%d patch=%d embd=%d layers=%d ffn=%d\n",
		enc.Config.ProjectorType, enc.Config.ImageSize, enc.Config.PatchSize, enc.Config.EmbeddingDim,
		enc.Config.NumLayers, enc.Config.FFNDim)

	rows, gh, gw, err := enc.EncodeWithGrid(img)
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "image tokens=%d dim=%d grid=%dx%d\n", len(rows), len(rows[0]), gh, gw)

	p, err := llm.NewPipeline(*model, 4096)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}
	mdim := p.Model.Config.EmbeddingDim
	if mdim != len(rows[0]) {
		fmt.Fprintf(os.Stderr, "dimension mismatch: mmproj outputs %d, model embedding dim %d\n", len(rows[0]), mdim)
		os.Exit(1)
	}

	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = *maxTokens
	cfg.ImageMarker = "<image>"
	cfg.ImageGrids = [][2]int{{gh, gw}}
	cfg.Sampler.Temperature = float32(*temp)
	cfg.Seed = *seed

	res, err := p.GenerateDetailedWithEmbeddings(*prompt, [][][]float32{rows}, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("%s\n", res.Text)
	fmt.Fprintf(os.Stderr, "\n[prompt_tokens=%d generated=%d %.1f tok/s]\n",
		res.PromptTokens, res.TotalTokens, res.TokensPerSec)
}

func synthetic(kind string, n int) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, n, n))
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			switch kind {
			case "white":
				img.Set(x, y, color.NRGBA{255, 255, 255, 255})
			case "black":
				img.Set(x, y, color.NRGBA{0, 0, 0, 255})
			case "gray":
				img.Set(x, y, color.NRGBA{128, 128, 128, 255})
			case "cb":
				v := uint8(255)
				if (x+y)%2 == 1 {
					v = 0
				}
				img.Set(x, y, color.NRGBA{v, v, v, 255})
			case "red":
				img.Set(x, y, color.NRGBA{255, 0, 0, 255})
			case "green":
				img.Set(x, y, color.NRGBA{0, 255, 0, 255})
			case "blue":
				img.Set(x, y, color.NRGBA{0, 0, 255, 255})
			}
		}
	}
	return img
}
