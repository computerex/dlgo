package main

// mmproj_encode is a debug CLI: load an mmproj vision encoder, encode an
// image (or a synthetic pattern), and dump per-token embeddings so the Go
// implementation can be diffed against llama.cpp's llama-mtmd-debug.
//
// Usage: go run ./cmd/mmproj-encode -mmproj <file> [-image <png|white|black|gray|cb>] [-n <px>] [-dump <txt>] [-stats]

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"os"
	"strings"

	"github.com/computerex/dlgo/vision"
)

func main() {
	mmproj := flag.String("mmproj", "", "path to mmproj GGUF")
	imagePath := flag.String("image", "cb", "image file, or synthetic: white|black|gray|cb")
	n := flag.Int("n", 336, "synthetic image edge size in pixels")
	dump := flag.String("dump", "", "optional path to write embeddings (one row per line)")
	compact := flag.Bool("compact", false, "print a single-line summary only")
	flag.Parse()

	if *mmproj == "" {
		fmt.Fprintln(os.Stderr, "usage: mmproj-encode -mmproj <file> [-image <png|white|black|gray|cb>] [-n px] [-dump file]")
		os.Exit(2)
	}

	var img image.Image
	switch strings.ToLower(*imagePath) {
	case "white", "black", "gray", "cb":
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

	fmt.Fprintf(os.Stderr, "config: proj=%s image=%d patch=%d embd=%d heads=%d layers=%d ffn=%d merge=%s npos=%d\n",
		enc.Config.ProjectorType, enc.Config.ImageSize, enc.Config.PatchSize, enc.Config.EmbeddingDim,
		enc.Config.NumHeads, enc.Config.NumLayers, enc.Config.FFNDim, enc.Config.MergeType, enc.NumImageTokens())

	rows, err := enc.Encode(img)
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}

	var sum, sq float64
	for _, r := range rows {
		for _, v := range r {
			sum += float64(v)
			sq += float64(v) * float64(v)
		}
	}
	nEl := float64(len(rows) * len(rows[0]))
	fmt.Fprintf(os.Stderr, "tokens=%d dim=%d mean=%.6f l2=%.6f\n",
		len(rows), len(rows[0]), sum/nEl, sq)

	if !*compact && *dump == "" {
		// print first few rows compactly for eyeballing
		for i := 0; i < len(rows) && i < 3; i++ {
			fmt.Printf("token[%d] first8: ", i)
			for j := 0; j < 8 && j < len(rows[i]); j++ {
				fmt.Printf("%.5f ", rows[i][j])
			}
			fmt.Println()
		}
	}

	if *dump != "" {
		f, err := os.Create(*dump)
		if err != nil {
			fmt.Fprintf(os.Stderr, "create dump: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()
		for _, r := range rows {
			for j, v := range r {
				if j > 0 {
					fmt.Fprint(f, " ")
				}
				fmt.Fprintf(f, "%.8g", v)
			}
			fmt.Fprintln(f)
		}
		fmt.Fprintf(os.Stderr, "wrote %s (%d rows)\n", *dump, len(rows))
	}
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
				// checkerboard of 1x1 pixels, matching mtmd-debug's "cb"
				v := uint8(255)
				if (x+y)%2 == 1 {
					v = 0
				}
				img.Set(x, y, color.NRGBA{v, v, v, 255})
			}
		}
	}
	return img
}
