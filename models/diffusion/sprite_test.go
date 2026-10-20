package diffusion

import (
	"bytes"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// synthSpriteImage builds a w x h RGB image (layout [3][h][w]) containing an
// opaque rounded blob of coloured noise on a uniform background, which is the
// shape a prompted "sprite on a plain background" render is supposed to have.
func synthSpriteImage(w, h int, bg float32, blob func(x, y int) [3]float32) []float32 {
	px := make([]float32, 3*w*h)
	cx, cy, r := float64(w)/2, float64(h)/2, float64(w)*0.3
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			c := [3]float32{bg, bg, bg}
			dx, dy := float64(x)-cx, float64(y)-cy
			if math.Hypot(dx, dy) < r {
				c = blob(x, y)
			}
			px[i] = c[0]
			px[w*h+i] = c[1]
			px[2*w*h+i] = c[2]
		}
	}
	return px
}

func TestMakeSpritePaletteAndTransparency(t *testing.T) {
	const w, h, size = 512, 512, 32
	// Interior: a two-axis gradient so median cut has real work to do and the
	// source has far more distinct colours than the palette can hold.
	px := synthSpriteImage(w, h, 1.0, func(x, y int) [3]float32 {
		return [3]float32{
			float32(x) / float32(w),
			0.2 + 0.6*float32(y)/float32(h),
			float32(w-x) / float32(w),
		}
	})

	cfg := DefaultSpriteConfig()
	cfg.Size = size
	sp, err := MakeSprite(px, w, h, cfg)
	if err != nil {
		t.Fatalf("MakeSprite: %v", err)
	}

	if sp.Size != size {
		t.Fatalf("Size = %d, want %d", sp.Size, size)
	}
	if len(sp.Palette) == 0 || len(sp.Palette) > cfg.Colors {
		t.Fatalf("palette has %d entries, want 1..%d", len(sp.Palette), cfg.Colors)
	}
	if sp.SourceColors <= len(sp.Palette) {
		t.Errorf("expected the source to carry more colours than the palette: source=%d palette=%d",
			sp.SourceColors, len(sp.Palette))
	}
	if sp.Opaque == 0 {
		t.Fatal("every pixel was keyed out")
	}
	if sp.Opaque >= size*size {
		t.Errorf("background was not keyed out: %d/%d pixels opaque", sp.Opaque, size*size)
	}

	// Every opaque pixel must be exactly one of the palette entries, and every
	// transparent pixel must be fully transparent.
	inPalette := func(r, g, b uint8) bool {
		for _, p := range sp.Palette {
			if p[0] == r && p[1] == g && p[2] == b {
				return true
			}
		}
		return false
	}
	for i := 0; i < size*size; i++ {
		r, g, b, a := sp.RGBA[i*4], sp.RGBA[i*4+1], sp.RGBA[i*4+2], sp.RGBA[i*4+3]
		if a == 0 {
			if sp.Index[i] != TransparentIndex {
				t.Fatalf("pixel %d is transparent but index = %d", i, sp.Index[i])
			}
			continue
		}
		if a != 255 {
			t.Fatalf("pixel %d has alpha %d, want 255 or 0", i, a)
		}
		if !inPalette(r, g, b) {
			t.Fatalf("pixel %d colour %02x%02x%02x is not in the palette", i, r, g, b)
		}
		if sp.Index[i] >= uint8(len(sp.Palette)) {
			t.Fatalf("pixel %d index %d out of range", i, sp.Index[i])
		}
	}

	// The subject is centred, so the middle of the grid must be opaque.
	mid := (size/2)*size + size/2
	if sp.RGBA[mid*4+3] == 0 {
		t.Error("grid centre is transparent; the blob should be there")
	}
}

func TestMakeSpriteColourLimits(t *testing.T) {
	const w, h = 256, 256
	px := synthSpriteImage(w, h, 0.0, func(x, y int) [3]float32 {
		return [3]float32{float32(x) / float32(w), float32(y) / float32(h), 0.5}
	})
	for _, tc := range []struct {
		name   string
		colors int
		size   int
	}{
		{"snes16", 15, 32},
		{"nes8", 3, 16},
		{"mono", 1, 16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := SpriteConfig{Size: tc.size, Colors: tc.colors, BgTolerance: 0.12, Upscale: 4}
			sp, err := MakeSprite(px, w, h, cfg)
			if err != nil {
				t.Fatalf("MakeSprite: %v", err)
			}
			if len(sp.Palette) > tc.colors {
				t.Fatalf("palette has %d entries, limit is %d", len(sp.Palette), tc.colors)
			}
			// A single-colour palette must collapse the whole subject to one shade.
			if tc.colors == 1 && len(sp.Palette) != 1 {
				t.Fatalf("expected exactly 1 palette entry, got %d", len(sp.Palette))
			}
		})
	}
}

func TestMakeSpriteKeepOpaque(t *testing.T) {
	const w, h, size = 128, 128, 32
	px := synthSpriteImage(w, h, 1.0, func(x, y int) [3]float32 {
		return [3]float32{0.8, 0.1, 0.1}
	})
	cfg := SpriteConfig{Size: size, Colors: 4, KeepOpaque: true}
	sp, err := MakeSprite(px, w, h, cfg)
	if err != nil {
		t.Fatalf("MakeSprite: %v", err)
	}
	if sp.Opaque != size*size {
		t.Fatalf("KeepOpaque left %d/%d pixels opaque", sp.Opaque, size*size)
	}
	for i := 0; i < size*size; i++ {
		if sp.RGBA[i*4+3] != 255 {
			t.Fatalf("pixel %d alpha = %d, want 255", i, sp.RGBA[i*4+3])
		}
	}
}

func TestMakeSpriteRejectsBadGeometry(t *testing.T) {
	px := make([]float32, 3*100*100)
	cfg := SpriteConfig{Size: 32, Colors: 8} // 100 is not a multiple of 32
	if _, err := MakeSprite(px, 100, 100, cfg); err == nil {
		t.Fatal("expected an error for a non-multiple image size")
	}
	cfg.Size = 2 // below the minimum
	if _, err := MakeSprite(make([]float32, 3*16*16), 16, 16, cfg); err == nil {
		t.Fatal("expected an error for a too-small sprite size")
	}
	// Buffer shorter than w*h*3.
	if _, err := MakeSprite(make([]float32, 10), 32, 32, SpriteConfig{Size: 32}); err == nil {
		t.Fatal("expected an error for a short pixel buffer")
	}
}

// A fully uniform image has no subject to key, which must be reported rather
// than silently written as an empty sprite.
func TestMakeSpriteEmptySubject(t *testing.T) {
	const w, h, size = 64, 64, 32
	px := synthSpriteImage(w, h, 0.5, func(x, y int) [3]float32 {
		return [3]float32{0.5, 0.5, 0.5}
	})
	if _, err := MakeSprite(px, w, h, SpriteConfig{Size: size, Colors: 8, BgTolerance: 0.12}); err == nil {
		t.Fatal("expected an error when the whole image is background")
	}
}

func TestMakeSpriteDeterministic(t *testing.T) {
	const w, h, size = 256, 256, 32
	px := synthSpriteImage(w, h, 1.0, func(x, y int) [3]float32 {
		return [3]float32{float32((x*7+y*13)%50) / 50, 0.4, 0.9}
	})
	cfg := SpriteConfig{Size: size, Colors: 12, BgTolerance: 0.12}

	a, err := MakeSprite(px, w, h, cfg)
	if err != nil {
		t.Fatalf("first MakeSprite: %v", err)
	}
	b, err := MakeSprite(px, w, h, cfg)
	if err != nil {
		t.Fatalf("second MakeSprite: %v", err)
	}
	if !bytes.Equal(a.RGBA, b.RGBA) {
		t.Fatal("two runs on identical pixels produced different sprites")
	}
	if fmt.Sprint(a.PaletteHex()) != fmt.Sprint(b.PaletteHex()) {
		t.Fatalf("palettes differ: %v vs %v", a.PaletteHex(), b.PaletteHex())
	}
}

func TestSpritePNGRoundTrip(t *testing.T) {
	const w, h, size = 256, 256, 32
	px := synthSpriteImage(w, h, 1.0, func(x, y int) [3]float32 {
		return [3]float32{0.1, 0.9, 0.3}
	})
	sp, err := MakeSprite(px, w, h, SpriteConfig{Size: size, Colors: 6, BgTolerance: 0.12})
	if err != nil {
		t.Fatalf("MakeSprite: %v", err)
	}

	dir := t.TempDir()
	native := filepath.Join(dir, "a.png")
	if err := sp.SavePNG(native); err != nil {
		t.Fatalf("SavePNG: %v", err)
	}
	preview := filepath.Join(dir, "a@8x.png")
	if err := sp.SavePreview(preview, 8); err != nil {
		t.Fatalf("SavePreview: %v", err)
	}

	// The native file is exactly one image pixel per sprite pixel.
	fi, err := os.Stat(native)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Size() == 0 {
		t.Fatal("native PNG is empty")
	}
	f, err := os.Open(native)
	if err != nil {
		t.Fatal(err)
	}
	cfgImg, _, err := image.DecodeConfig(f)
	f.Close()
	if err != nil {
		t.Fatalf("decode native PNG: %v", err)
	}
	if cfgImg.Width != size || cfgImg.Height != size {
		t.Fatalf("native PNG is %dx%d, want %dx%d", cfgImg.Width, cfgImg.Height, size, size)
	}

	back, err := LoadSpritePNG(native)
	if err != nil {
		t.Fatalf("LoadSpritePNG: %v", err)
	}
	if back.Size != sp.Size || !bytes.Equal(back.RGBA, sp.RGBA) {
		t.Fatal("sprite did not survive the PNG round trip")
	}

	// LoadSprites must pick up the native file and skip the preview.
	list, err := LoadSprites(dir)
	if err != nil {
		t.Fatalf("LoadSprites: %v", err)
	}
	if len(list) != 1 {
		t.Fatalf("LoadSprites returned %d sprites, want 1 (previews must be skipped)", len(list))
	}

	sheet := filepath.Join(dir, "sheet.png")
	if err := SaveSheet(list, sheet, 8, 4); err != nil {
		t.Fatalf("SaveSheet: %v", err)
	}
	if fi, err := os.Stat(sheet); err != nil || fi.Size() == 0 {
		t.Fatalf("sheet not written: %v", err)
	}
}
