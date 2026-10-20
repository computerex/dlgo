// dlgo-sprite generates retro game sprites with the diffusion pipeline.
//
// The pipeline itself renders a smooth RGB photograph-style image; a sprite is a
// small pixel grid drawn from a tiny indexed palette. This tool runs the Z-Image
// pipeline, then box-downsamples to the sprite grid, keys out the background and
// quantises to the palette limits of the era:
//
//	16-bit (default): 32x32 grid, 15 colours + transparent (SNES palette limit)
//	8-bit:            -size 16 -colors 3 (NES sprite limit)
//
// Each run writes three files into -outdir:
//
//	<name>.png        native size, e.g. 32x32, RGBA
//	<name>@8x.png     nearest-neighbour preview at -upscale
//	sheet.png         contact sheet of every sprite in -outdir on a checkerboard
//
// To render many sprites, pass several prompts with -prompts "a;b;c". With -gpu
// every prompt is rendered inside ONE process from a single model load (see
// diffusion.GenerateImageBatch): the DiT upload, which dominates startup, is
// paid once instead of per sprite.
//
// Run:
//
//	go build -tags "cgo vulkan" -ldflags "-linkmode internal" -o dlgo-sprite ./cmd/dlgo-sprite
//	dlgo-sprite --gpu -p "pixel art sprite of a knight" -name knight
//	dlgo-sprite --gpu -prompts "a knight;a slime;a treasure chest" -nes-size 16
//
// Use -compose <dir> to rebuild only the contact sheet from existing sprites
// (no model load, no GPU).
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/computerex/dlgo/models/diffusion"
)

func main() {
	ditPath := flag.String("diffusion-model", "", "Path to Z-Image DiT GGUF model")
	vaePath := flag.String("vae", "", "Path to VAE safetensors (ae.safetensors)")
	llmPath := flag.String("llm", "", "Path to the Qwen3-4B text encoder GGUF")
	prompt := flag.String("p", "", "Text prompt (single)")
	promptsFlag := flag.String("prompts", "", "Semicolon-separated list of prompts to render in one model load")
	name := flag.String("name", "", "Output base name (default: slug of the prompt)")
	outDir := flag.String("outdir", "sprites", "Output directory")
	compose := flag.String("compose", "", "Only rebuild sheet.png from the sprites in this directory")

	width := flag.Int("W", 512, "Render width (must be a multiple of -size)")
	height := flag.Int("H", 512, "Render height (must be a multiple of -size)")
	steps := flag.Int("steps", 8, "Sampling steps")
	seed := flag.Int64("seed", 42, "Random seed")
	cfgScale := flag.Float64("cfg-scale", 1.0, "CFG scale (1.0 = no guidance)")
	useGPU := flag.Bool("gpu", false, "Use GPU acceleration for DiT inference")
	debug := flag.Bool("debug", false, "Enable detailed GPU profiling and timing")

	size := flag.Int("size", 32, "Sprite grid edge in pixels (16-bit uses 16..64)")
	colors := flag.Int("colors", 15, "Opaque palette entries (SNES 15, NES 3)")
	upscale := flag.Int("upscale", 8, "Nearest-neighbour preview factor")
	keepOpaque := flag.Bool("keep-opaque", false, "Keep the background instead of keying it transparent")
	bgTol := flag.Float64("bg-tolerance", 0.12, "Background colour distance treated as background (0..1)")
	asciiOut := flag.Bool("ascii", false, "Print the quantised grid as ASCII to stdout")
	nesSize := flag.Int("nes-size", 0, "Also emit an 8-bit variant at this grid edge into <outdir>/nes (0 = off)")
	nesColors := flag.Int("nes-colors", 3, "8-bit variant palette size (NES allowed 3 + transparent)")
	flag.Parse()

	if *compose != "" {
		if err := composeSheet(*compose, *upscale); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	prompts := gatherPrompts(*prompt, *promptsFlag)
	if *ditPath == "" || *vaePath == "" || *llmPath == "" || len(prompts) == 0 {
		fmt.Fprintln(os.Stderr, "Usage: dlgo-sprite --diffusion-model <dit.gguf> --vae <ae.safetensors> --llm <qwen3-4b.gguf> -p \"prompt\"")
		fmt.Fprintln(os.Stderr, "       dlgo-sprite ... --gpu -prompts \"prompt one;prompt two;prompt three\"")
		flag.PrintDefaults()
		os.Exit(1)
	}
	if *width%*size != 0 || *height%*size != 0 {
		fmt.Fprintf(os.Stderr, "Error: -W %d / -H %d must be whole multiples of -size %d\n",
			*width, *height, *size)
		os.Exit(1)
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "Error: create output dir: %v\n", err)
		os.Exit(1)
	}

	start := time.Now()
	genCfg := diffusion.ImageGenConfig{
		Width:    *width,
		Height:   *height,
		Steps:    *steps,
		CFGScale: float32(*cfgScale),
		Seed:     *seed,
		UseGPU:   *useGPU,
		Debug:    *debug,
	}

	pixSets, err := renderAll(*ditPath, *vaePath, *llmPath, prompts, genCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: generate: %v\n", err)
		os.Exit(1)
	}

	spCfg := diffusion.SpriteConfig{
		Size:        *size,
		Colors:      *colors,
		KeepOpaque:  *keepOpaque,
		BgTolerance: float32(*bgTol),
		Upscale:     *upscale,
	}
	var nesDir string
	if *nesSize > 0 {
		nesDir = filepath.Join(*outDir, "nes")
		if err := os.MkdirAll(nesDir, 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "Error: create %s: %v\n", nesDir, err)
			os.Exit(1)
		}
	}

	used := map[string]bool{}
	for i := range pixSets {
		base := *name
		if base == "" || len(pixSets) > 1 {
			base = slug(prompts[i])
		}
		base = uniqueName(base, used)

		sp, err := emitSprite(pixSets[i], *width, *height, base, *outDir, spCfg, *upscale)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		if *asciiOut {
			printASCII(sp)
		}

		if *nesSize > 0 {
			nesCfg := spCfg
			nesCfg.Size = *nesSize
			nesCfg.Colors = *nesColors
			nes, err := emitSprite(pixSets[i], *width, *height, base, nesDir, nesCfg, *upscale)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: 8-bit sprite: %v\n", err)
				os.Exit(1)
			}
			if *asciiOut {
				fmt.Printf("8-bit (%dx%d, %d colours):\n", nes.Size, nes.Size, len(nes.Palette))
				printASCII(nes)
			}
			fmt.Printf("  8-bit palette: %s\n", strings.Join(nes.PaletteHex(), " "))
		}
	}

	// Contact sheets once, after every sprite exists.
	if err := composeSheet(*outDir, *upscale); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: sheet: %v\n", err)
	}
	fmt.Printf("  %s\n", filepath.Join(*outDir, "sheet.png"))
	if nesDir != "" {
		if err := composeSheet(nesDir, *upscale); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: sheet: %v\n", err)
		}
		fmt.Printf("  %s\n", filepath.Join(nesDir, "sheet.png"))
	}
	fmt.Printf("Done in %v\n", time.Since(start))
}

// gatherPrompts builds the prompt list from -p and/or -prompts.
func gatherPrompts(single, list string) []string {
	var out []string
	if strings.TrimSpace(list) != "" {
		for _, p := range strings.Split(list, ";") {
			if p = strings.TrimSpace(p); p != "" {
				out = append(out, p)
			}
		}
		return out
	}
	if single = strings.TrimSpace(single); single != "" {
		out = append(out, single)
	}
	return out
}

// renderAll renders every prompt. With a single prompt (or no GPU) it calls the
// one-shot pipeline per prompt; with several prompts on the GPU it renders them
// all from a single model load.
func renderAll(ditPath, vaePath, llmPath string, prompts []string, cfg diffusion.ImageGenConfig) ([][]float32, error) {
	if cfg.UseGPU && len(prompts) > 1 {
		return diffusion.GenerateImageBatch(ditPath, vaePath, llmPath, prompts, cfg)
	}
	sets := make([][]float32, len(prompts))
	for i, p := range prompts {
		px, err := diffusion.GenerateImagePixels(ditPath, vaePath, llmPath, p, cfg)
		if err != nil {
			return nil, err
		}
		sets[i] = px
	}
	return sets, nil
}

// emitSprite quantises one render to a sprite, writes <name>.png and
// <name>@Nx.png into dir, prints a summary line, and returns the sprite.
func emitSprite(pixels []float32, w, h int, base, dir string, spCfg diffusion.SpriteConfig, upscale int) (*diffusion.Sprite, error) {
	sp, err := diffusion.MakeSprite(pixels, w, h, spCfg)
	if err != nil {
		return nil, fmt.Errorf("sprite %s: %w", base, err)
	}
	nativePath := filepath.Join(dir, base+".png")
	previewPath := filepath.Join(dir, fmt.Sprintf("%s@%dx.png", base, upscale))
	if err := sp.SavePNG(nativePath); err != nil {
		return nil, fmt.Errorf("write %s: %w", nativePath, err)
	}
	if err := sp.SavePreview(previewPath, upscale); err != nil {
		return nil, fmt.Errorf("write %s: %w", previewPath, err)
	}
	opaquePct := 100.0 * float64(sp.Opaque) / float64(sp.Size*sp.Size)
	fmt.Printf("%s (%dx%d, %d colours, %.0f%% opaque, source had %d distinct colours)\n",
		base, sp.Size, sp.Size, len(sp.Palette), opaquePct, sp.SourceColors)
	fmt.Printf("  palette: %s\n", strings.Join(sp.PaletteHex(), " "))
	fmt.Printf("  %s\n  %s\n", nativePath, previewPath)
	return sp, nil
}

// uniqueName appends -2, -3, ... until the base name is unused in used.
func uniqueName(base string, used map[string]bool) string {
	candidate := base
	for i := 2; used[candidate]; i++ {
		candidate = fmt.Sprintf("%s-%d", base, i)
	}
	used[candidate] = true
	return candidate
}

// composeSheet rebuilds sheet.png from the native (non-preview, non-underscore)
// sprites already in dir, and reports each palette so a set can be verified
// without opening the images.
func composeSheet(dir string, upscale int) error {
	sprites, err := diffusion.LoadSprites(dir)
	if err != nil {
		return err
	}
	if len(sprites) == 0 {
		return fmt.Errorf("no sprites found in %s", dir)
	}
	entries, err := os.ReadDir(dir)
	if err == nil {
		i := 0
		for _, e := range entries {
			n := e.Name()
			if e.IsDir() || !strings.HasSuffix(n, ".png") ||
				strings.HasPrefix(n, "_") || n == "sheet.png" || strings.Contains(n, "@") {
				continue
			}
			if i < len(sprites) {
				fmt.Printf("  %-12s %d colours  %s\n", strings.TrimSuffix(n, ".png"),
					len(sprites[i].Palette), strings.Join(sprites[i].PaletteHex(), " "))
			}
			i++
		}
	}
	return diffusion.SaveSheet(sprites, filepath.Join(dir, "sheet.png"), upscale, 6)
}

// printASCII dumps the quantised grid so the sprite can be inspected without an
// image viewer: '.' is transparent, otherwise a luminance ramp.
func printASCII(sp *diffusion.Sprite) {
	const ramp = " .:-=+*#%@"
	fmt.Println("  grid:")
	for y := 0; y < sp.Size; y++ {
		row := make([]byte, sp.Size)
		for x := 0; x < sp.Size; x++ {
			i := (y*sp.Size + x) * 4
			if sp.RGBA[i+3] == 0 {
				row[x] = '.'
				continue
			}
			lum := (0.299*float64(sp.RGBA[i]) + 0.587*float64(sp.RGBA[i+1]) + 0.114*float64(sp.RGBA[i+2])) / 255
			k := int(lum * float64(len(ramp)-1))
			row[x] = ramp[k]
		}
		fmt.Printf("  %s\n", row)
	}
}

var slugRe = regexp.MustCompile(`[^a-z0-9]+`)

// slugStop is the set of filler words stripped before slugging, so that prompts
// like "pixel art sprite of a knight" and "pixel art sprite of a slime" get the
// distinct names "knight" and "slime" instead of colliding on the shared prefix.
var slugStop = map[string]bool{
	"a": true, "an": true, "the": true, "of": true,
	"pixel": true, "art": true, "sprite": true, "sprites": true, "game": true,
	"8bit": true, "16bit": true, "8-bit": true, "16-bit": true,
}

// slug turns a prompt into a short file name: filler words are dropped, then the
// first four remaining words are joined, lowercased, with runs of non-alphanumerics
// collapsed to single dashes.
func slug(prompt string) string {
	words := strings.Fields(strings.ToLower(prompt))
	keep := make([]string, 0, len(words))
	for _, w := range words {
		if slugStop[w] {
			continue
		}
		keep = append(keep, w)
	}
	if len(keep) == 0 {
		keep = words // prompt was all filler; keep the original words
	}
	if len(keep) > 4 {
		keep = keep[:4]
	}
	s := slugRe.ReplaceAllString(strings.Join(keep, "-"), "-")
	s = strings.Trim(s, "-")
	if s == "" {
		s = "sprite"
	}
	return s
}
