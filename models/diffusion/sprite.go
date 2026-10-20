package diffusion

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// SpriteConfig describes a retro sprite post-process.
//
// The diffusion pipeline renders a smooth RGB image, which is the opposite of a
// sprite: a sprite is a small pixel grid drawn from a tiny indexed palette.
// Enforcing the hardware limits of the era is what makes the result read as a
// real sprite instead of a photo with a pixel-art filter. The per-sprite limits
// were:
//
//	8-bit (NES):   8x8 or 8x16 tiles, 3 colours + transparent
//	16-bit (SNES): 16x16 .. 64x64, 15 colours + transparent (16-entry palette)
//
// Background keying makes the surrounding area transparent rather than burning
// palette entries on it, so the sprite can be dropped straight onto a tile map.
type SpriteConfig struct {
	Size        int     // output grid edge in pixels (square), e.g. 16, 32, 64
	Colors      int     // opaque palette entries; SNES allows 15, NES allows 3
	KeepOpaque  bool    // keep the background instead of keying it to alpha 0
	BgTolerance float32 // 0..1 Euclidean RGB distance treated as background
	Upscale     int     // nearest-neighbour preview factor (1 = none)
}

// DefaultSpriteConfig targets 16-bit console limits: a 32x32 grid drawn from a
// 15-colour palette plus transparency, previewed at 8x.
func DefaultSpriteConfig() SpriteConfig {
	return SpriteConfig{Size: 32, Colors: 15, BgTolerance: 0.12, Upscale: 8}
}

// Sprite is a quantised pixel grid ready to write as PNG.
type Sprite struct {
	Size    int
	RGBA    []uint8    // Size*Size*4, row-major
	Index   []uint8    // Size*Size palette indices, TransparentIndex = transparent
	Palette [][3]uint8 // opaque entries only
	SourceColors int   // distinct colours in the downsampled image before quantisation
	Opaque  int        // number of non-transparent pixels
}

// TransparentIndex marks a transparent cell in Sprite.Index.
const TransparentIndex = 0xFF

// MakeSprite box-downsamples an RGB image to a Size x Size pixel grid, keys out
// the background, crops to the subject and quantises what is left to at most
// cfg.Colors entries.
//
// pixels must be float32 RGB in [0,1] laid out [3][h][w] (the layout the
// diffusion pipeline returns); w and h must be exact multiples of cfg.Size so
// every output pixel averages the same number of source pixels.
func MakeSprite(pixels []float32, w, h int, cfg SpriteConfig) (*Sprite, error) {
	if cfg.Size < 4 {
		return nil, fmt.Errorf("sprite size %d is too small (min 4)", cfg.Size)
	}
	if w%cfg.Size != 0 || h%cfg.Size != 0 {
		return nil, fmt.Errorf("image %dx%d is not a whole multiple of sprite size %d", w, h, cfg.Size)
	}
	if len(pixels) < 3*w*h {
		return nil, fmt.Errorf("pixel buffer holds %d values, need %d", len(pixels), 3*w*h)
	}
	if cfg.Colors < 1 {
		cfg.Colors = 1
	}

	// 1. Box-downsample to the sprite grid. Averaging is integer-ratio, so every
	// output pixel is an unweighted mean of its own block — no resampling
	// kernel, no phase artefacts.
	s := cfg.Size
	small := make([]float32, 3*s*s)
	bw, bh := w/s, h/s
	inv := 1.0 / float64(bw*bh)
	for y := 0; y < s; y++ {
		for x := 0; x < s; x++ {
			var r, g, b float64
			for sy := y * bh; sy < (y+1)*bh; sy++ {
				base := sy * w
				for sx := x * bw; sx < (x+1)*bw; sx++ {
					i := base + sx
					r += float64(pixels[i])
					g += float64(pixels[w*h+i])
					b += float64(pixels[2*w*h+i])
				}
			}
			dst := y*s + x
			small[dst] = float32(r * inv)
			small[s*s+dst] = float32(g * inv)
			small[2*s*s+dst] = float32(b * inv)
		}
	}

	// 2. Background key + crop to the subject, so the grid is spent on the
	// sprite rather than on empty margin.
	opaque := make([]bool, s*s)
	for i := range opaque {
		opaque[i] = true
	}
	if !cfg.KeepOpaque {
		opaque = keyBackground(small, s, cfg.BgTolerance)
	}
	cropped, cropOpaque := cropCentred(small, opaque, s)

	// 3. Quantise the surviving pixels with median cut, then snap each pixel to
	// its nearest palette entry.
	var cols []rgbTriple
	sp := &Sprite{Size: s}
	sp.RGBA = make([]uint8, s*s*4)
	sp.Index = make([]uint8, s*s)
	for i := range sp.Index {
		sp.Index[i] = TransparentIndex
	}
	for i := 0; i < s*s; i++ {
		if !cropOpaque[i] {
			continue
		}
		sp.Opaque++
		cols = append(cols, rgbTriple{
			to8(cropped[i]),
			to8(cropped[s*s+i]),
			to8(cropped[2*s*s+i]),
		})
	}
	sp.SourceColors = countDistinct(cols)
	if sp.Opaque == 0 {
		return nil, fmt.Errorf("sprite is empty: background keying removed every pixel (try --keep-opaque or a smaller --bg-tolerance)")
	}

	sp.Palette = medianCut(cols, cfg.Colors)

	for i := 0; i < s*s; i++ {
		if !cropOpaque[i] {
			continue
		}
		c := rgbTriple{to8(cropped[i]), to8(cropped[s*s+i]), to8(cropped[2*s*s+i])}
		idx := nearestEntry(sp.Palette, c)
		sp.Index[i] = uint8(idx)
		p := sp.Palette[idx]
		sp.RGBA[i*4+0] = p[0]
		sp.RGBA[i*4+1] = p[1]
		sp.RGBA[i*4+2] = p[2]
		sp.RGBA[i*4+3] = 255
	}
	return sp, nil
}

type rgbTriple [3]uint8

func to8(v float32) uint8 {
	x := int(v*255.0 + 0.5)
	if x < 0 {
		return 0
	}
	if x > 255 {
		return 255
	}
	return uint8(x)
}

// keyBackground returns an opaque mask that is false for background pixels.
//
// The reference colour is the median of the border ring (robust to a single
// bright corner). Only background-coloured pixels *connected to the border* are
// removed, so a white highlight inside the subject is preserved — a plain
// colour-distance test would punch holes in it.
func keyBackground(img []float32, s int, tol float32) []bool {
	mask := make([]bool, s*s)
	for i := range mask {
		mask[i] = true
	}
	if tol <= 0 {
		return mask
	}

	// Median border colour, per channel. Collecting and sorting a few hundred
	// values is cheaper and steadier than averaging, which a stray highlight
	// would drag off.
	var ring []int
	for i := 0; i < s; i++ {
		ring = append(ring, i, (s-1)*s+i, i*s, i*s+s-1)
	}
	rc := make([]int, 0, len(ring))
	gc := make([]int, 0, len(ring))
	bc := make([]int, 0, len(ring))
	for _, p := range ring {
		rc = append(rc, int(to8(img[p])))
		gc = append(gc, int(to8(img[s*s+p])))
		bc = append(bc, int(to8(img[2*s*s+p])))
	}
	br, bg, bb := medianInt(rc), medianInt(gc), medianInt(bc)

	// Tolerance is expressed in [0,1] against the maximum channel distance, so
	// it reads the same at any intensity.
	limit := float32(tol) * 255.0
	near := func(p int) bool {
		dr := float32(int(to8(img[p])) - br)
		dg := float32(int(to8(img[s*s+p])) - bg)
		db := float32(int(to8(img[2*s*s+p])) - bb)
		return float32(math.Sqrt(float64(dr*dr+dg*dg+db*db))) <= limit
	}

	// Flood fill inward from every near-background border pixel.
	stack := make([]int, 0, s*s)
	push := func(p int) {
		if p >= 0 && p < s*s && mask[p] && near(p) {
			mask[p] = false
			stack = append(stack, p)
		}
	}
	for i := 0; i < s; i++ {
		push(i)
		push((s-1)*s + i)
		push(i * s)
		push(i*s + s - 1)
	}
	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		x, y := p%s, p/s
		if x > 0 {
			push(p - 1)
		}
		if x < s-1 {
			push(p + 1)
		}
		if y > 0 {
			push(p - s)
		}
		if y < s-1 {
			push(p + s)
		}
	}
	return mask
}

func medianInt(v []int) int {
	if len(v) == 0 {
		return 0
	}
	sort.Ints(v)
	return v[len(v)/2]
}

// cropCentred shifts the opaque bounding box to the centre of the grid, keeping
// the pixel grid intact (a crop never rescales, so no pixel becomes a blend).
func cropCentred(img []float32, opaque []bool, s int) ([]float32, []bool) {
	minX, minY, maxX, maxY := s, s, -1, -1
	for y := 0; y < s; y++ {
		for x := 0; x < s; x++ {
			if !opaque[y*s+x] {
				continue
			}
			if x < minX {
				minX = x
			}
			if x > maxX {
				maxX = x
			}
			if y < minY {
				minY = y
			}
			if y > maxY {
				maxY = y
			}
		}
	}
	if maxX < minX || maxY < minY {
		return img, opaque // fully transparent; nothing to place
	}
	w, h := maxX-minX+1, maxY-minY+1
	offX, offY := (s-w)/2, (s-h)/2

	out := make([]float32, len(img))
	outMask := make([]bool, s*s)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src := (minY+y)*s + minX + x
			dst := (offY+y)*s + offX + x
			out[dst] = img[src]
			out[s*s+dst] = img[s*s+src]
			out[2*s*s+dst] = img[2*s*s+src]
			outMask[dst] = opaque[src]
		}
	}
	return out, outMask
}

func countDistinct(cols []rgbTriple) int {
	seen := make(map[rgbTriple]struct{}, len(cols))
	for _, c := range cols {
		seen[c] = struct{}{}
	}
	return len(seen)
}

// medianCut reduces cols to at most maxColors entries by repeatedly splitting a
// box along its widest channel at the median of that channel.
//
// Box selection is variance-weighted — the next box to split is the one with the
// largest population x range^2, a cheap proxy for its total squared error. Plain
// "largest range" selection spends palette entries repeatedly on big flat
// regions with a little noise while starving small high-contrast details (a
// sprite's trim, eyes, blade highlight), which is exactly backwards for art with
// a dominant background colour.
//
// Box colours share one backing array and each split sorts a sub-slice in place,
// so no per-box allocation is needed.
func medianCut(cols []rgbTriple, maxColors int) [][3]uint8 {
	if len(cols) == 0 {
		return nil
	}
	if maxColors < 1 {
		maxColors = 1
	}
	type box struct{ lo, hi int }
	boxes := []box{{0, len(cols)}}

	for len(boxes) < maxColors {
		best, bestScore := -1, 0
		for i, b := range boxes {
			if b.hi-b.lo < 2 {
				continue
			}
			r := boxRange(cols, b.lo, b.hi)
			if r == 0 {
				continue // a single colour: splitting it achieves nothing
			}
			if s := (b.hi - b.lo) * r * r; s > bestScore {
				best, bestScore = i, s
			}
		}
		if best < 0 {
			break // every box holds a single colour
		}
		b := boxes[best]
		ch := widestChannel(cols, b.lo, b.hi)
		sub := cols[b.lo:b.hi]
		sort.Slice(sub, func(i, j int) bool { return sub[i][ch] < sub[j][ch] })
		mid := b.lo + (b.hi-b.lo)/2
		boxes = append(boxes, box{mid, b.hi})
		boxes[best].hi = mid
	}

	pal := make([][3]uint8, 0, len(boxes))
	var sumR, sumG, sumB, n int
	for _, b := range boxes {
		if b.hi <= b.lo {
			continue
		}
		sumR, sumG, sumB, n = 0, 0, 0, 0
		for i := b.lo; i < b.hi; i++ {
			sumR += int(cols[i][0])
			sumG += int(cols[i][1])
			sumB += int(cols[i][2])
			n++
		}
		if n == 0 {
			continue
		}
		pal = append(pal, [3]uint8{uint8(sumR / n), uint8(sumG / n), uint8(sumB / n)})
	}
	return dedupePalette(pal)
}

func boxRange(cols []rgbTriple, lo, hi int) int {
	var mn [3]int
	var mx [3]int
	for c := 0; c < 3; c++ {
		mn[c], mx[c] = 255, 0
	}
	for i := lo; i < hi; i++ {
		for c := 0; c < 3; c++ {
			v := int(cols[i][c])
			if v < mn[c] {
				mn[c] = v
			}
			if v > mx[c] {
				mx[c] = v
			}
		}
	}
	best := 0
	for c := 0; c < 3; c++ {
		if d := mx[c] - mn[c]; d > best {
			best = d
		}
	}
	return best
}

func widestChannel(cols []rgbTriple, lo, hi int) int {
	var mn [3]int
	var mx [3]int
	for c := 0; c < 3; c++ {
		mn[c], mx[c] = 255, 0
	}
	for i := lo; i < hi; i++ {
		for c := 0; c < 3; c++ {
			v := int(cols[i][c])
			if v < mn[c] {
				mn[c] = v
			}
			if v > mx[c] {
				mx[c] = v
			}
		}
	}
	bestC, bestD := 0, -1
	for c := 0; c < 3; c++ {
		if d := mx[c] - mn[c]; d > bestD {
			bestC, bestD = c, d
		}
	}
	return bestC
}

// dedupePalette drops duplicates that median cut can produce when several boxes
// average to the same colour, so the reported palette size is the real one.
func dedupePalette(pal [][3]uint8) [][3]uint8 {
	out := make([][3]uint8, 0, len(pal))
	for _, p := range pal {
		dup := false
		for _, q := range out {
			if p == q {
				dup = true
				break
			}
		}
		if !dup {
			out = append(out, p)
		}
	}
	return out
}

func nearestEntry(pal [][3]uint8, c rgbTriple) int {
	best, bestD := 0, math.MaxInt32
	for i, p := range pal {
		dr := int(c[0]) - int(p[0])
		dg := int(c[1]) - int(p[1])
		db := int(c[2]) - int(p[2])
		if d := dr*dr + dg*dg + db*db; d < bestD {
			best, bestD = i, d
		}
	}
	return best
}

// SavePNG writes the sprite at its native pixel size (one image pixel per
// sprite pixel).
func (sp *Sprite) SavePNG(path string) error { return sp.writePNG(path, 1) }

// SavePreview writes the sprite scaled up by scale with nearest-neighbour
// sampling: every sprite pixel becomes a solid scale x scale block, so the grid
// stays crisp. scale <= 1 writes the native size.
func (sp *Sprite) SavePreview(path string, scale int) error {
	if scale < 1 {
		scale = 1
	}
	return sp.writePNG(path, scale)
}

func (sp *Sprite) writePNG(path string, scale int) error {
	w := sp.Size * scale
	img := image.NewRGBA(image.Rect(0, 0, w, w))
	for y := 0; y < w; y++ {
		sy := y / scale
		for x := 0; x < w; x++ {
			sx := x / scale
			i := (sy*sp.Size + sx) * 4
			a := sp.RGBA[i+3]
			img.Set(x, y, color.RGBA{R: sp.RGBA[i], G: sp.RGBA[i+1], B: sp.RGBA[i+2], A: a})
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

// SaveSheet writes every sprite into one contact sheet on a checkerboard, which
// is the only way to see at a glance that the transparency and the palette
// limits came out right.
func SaveSheet(sprites []*Sprite, path string, scale, gutter int) error {
	if len(sprites) == 0 {
		return fmt.Errorf("no sprites to write")
	}
	if scale < 1 {
		scale = 1
	}
	cell := sprites[0].Size*scale + 2*gutter
	const cols = 4
	rows := (len(sprites) + cols - 1) / cols
	w, h := cols*cell, rows*cell

	img := image.NewRGBA(image.Rect(0, 0, w, h))
	// Checkerboard backdrop: 8 px squares of two mid greys.
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			shade := uint8(0x30)
			if ((x/8)+(y/8))%2 == 0 {
				shade = uint8(0x40)
			}
			img.Set(x, y, color.RGBA{R: shade, G: shade, B: shade, A: 255})
		}
	}
	for n, sp := range sprites {
		ox := (n%cols)*cell + gutter
		oy := (n/cols)*cell + gutter
		for y := 0; y < sp.Size*scale; y++ {
			sy := y / scale
			for x := 0; x < sp.Size*scale; x++ {
				sx := x / scale
				i := (sy*sp.Size + sx) * 4
				if sp.RGBA[i+3] == 0 {
					continue // leave the checkerboard visible
				}
				img.Set(ox+x, oy+y, color.RGBA{
					R: sp.RGBA[i], G: sp.RGBA[i+1], B: sp.RGBA[i+2], A: 255,
				})
			}
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

// PaletteHex renders the palette as CSS-style hex, for reuse in an editor or a
// tile map definition.
func (sp *Sprite) PaletteHex() []string {
	out := make([]string, len(sp.Palette))
	for i, p := range sp.Palette {
		out[i] = fmt.Sprintf("#%02x%02x%02x", p[0], p[1], p[2])
	}
	return out
}

// LoadSprites reads the native sprites in dir for sheet composition, skipping
// previews (name@Nx.png), the sheet itself and underscore-prefixed scratch
// files. Results are ordered by file name so the sheet is reproducible.
func LoadSprites(dir string) ([]*Sprite, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range entries {
		n := e.Name()
		if e.IsDir() || !strings.HasSuffix(n, ".png") {
			continue
		}
		if strings.HasPrefix(n, "_") || n == "sheet.png" || strings.Contains(n, "@") {
			continue
		}
		names = append(names, n)
	}
	sort.Strings(names)

	out := make([]*Sprite, 0, len(names))
	for _, n := range names {
		sp, err := LoadSpritePNG(filepath.Join(dir, n))
		if err != nil {
			return nil, fmt.Errorf("%s: %w", n, err)
		}
		out = append(out, sp)
	}
	return out, nil
}

// LoadSpritePNG reads a native sprite PNG back into a Sprite.
func LoadSpritePNG(path string) (*Sprite, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	if b.Dx() != b.Dy() {
		return nil, fmt.Errorf("not a square sprite: %dx%d", b.Dx(), b.Dy())
	}
	s := b.Dx()
	sp := &Sprite{Size: s, RGBA: make([]uint8, s*s*4), Index: make([]uint8, s*s)}
	for i := range sp.Index {
		sp.Index[i] = TransparentIndex
	}
	for y := 0; y < s; y++ {
		for x := 0; x < s; x++ {
			r, g, bl, a := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			i := (y*s + x) * 4
			// RGBA() returns alpha-premultiplied 16-bit values.
			if a == 0 {
				continue
			}
			sp.RGBA[i+0] = uint8(r * 0xffff / a >> 8)
			sp.RGBA[i+1] = uint8(g * 0xffff / a >> 8)
			sp.RGBA[i+2] = uint8(bl * 0xffff / a >> 8)
			sp.RGBA[i+3] = uint8(a >> 8)
			sp.Opaque++
		}
	}
	// Recover the palette so PaletteHex keeps working on loaded sprites.
	pal := map[[3]uint8]struct{}{}
	for i := 0; i < s*s; i++ {
		if sp.RGBA[i*4+3] == 0 {
			continue
		}
		pal[[3]uint8{sp.RGBA[i*4], sp.RGBA[i*4+1], sp.RGBA[i*4+2]}] = struct{}{}
	}
	keys := make([][3]uint8, 0, len(pal))
	for c := range pal {
		keys = append(keys, c)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i][0] != keys[j][0] {
			return keys[i][0] < keys[j][0]
		}
		if keys[i][1] != keys[j][1] {
			return keys[i][1] < keys[j][1]
		}
		return keys[i][2] < keys[j][2]
	})
	sp.Palette = keys
	sp.SourceColors = len(keys)
	return sp, nil
}
