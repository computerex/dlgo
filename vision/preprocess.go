package vision

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"math"
)

// f32Image is an RGB float image, channels interleaved (HxWx3, row-major).
type f32Image struct {
	W, H int
	Pix  []float32 // len = W*H*3, pixel (x,y) channel c at ((y*W+x)*3+c)
}

// decodeToRGBA converts any stdlib-supported image to NRGBA (alpha filled
// with white so transparent PNGs behave like composited-on-white).
func decodeToRGBA(img image.Image) *image.NRGBA {
	// decodeToRGBA fills alpha with white so transparent PNGs behave like
	// composited-on-white.
	b := img.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	white := color.NRGBA{R: 255, G: 255, B: 255, A: 255}
	draw.Draw(out, out.Bounds(), image.NewUniform(white), image.Point{}, draw.Src)
	draw.Draw(out, out.Bounds(), img, b.Min, draw.Over)
	return out
}

// resizeBilinear resizes src to (dw, dh) with bilinear interpolation. When
// downscaling, it averages over the source pixel footprint (antialiasing),
// matching common image-processor behavior closely enough for embedding work.
func resizeBilinear(src *f32Image, dw, dh int) *f32Image {
	out := &f32Image{W: dw, H: dh, Pix: make([]float32, dw*dh*3)}
	sx := float64(src.W) / float64(dw)
	sy := float64(src.H) / float64(dh)
	for y := 0; y < dh; y++ {
		// source span covered by this dest pixel
		fy0 := float64(y) * sy
		fy1 := float64(y+1) * sy
		for x := 0; x < dw; x++ {
			fx0 := float64(x) * sx
			fx1 := float64(x+1) * sx
			var acc [3]float64
			var wsum float64
			// iterate the integer source pixels in the footprint
			x0, x1 := int(math.Floor(fx0)), int(math.Ceil(fx1))
			y0, y1 := int(math.Floor(fy0)), int(math.Ceil(fy1))
			if x1 < x0+1 {
				x1 = x0 + 1
			}
			if y1 < y0+1 {
				y1 = y0 + 1
			}
			for yy := y0; yy < y1; yy++ {
				wy := overlapWeight(fy0, fy1, yy)
				if wy <= 0 {
					continue
				}
				if yy < 0 || yy >= src.H {
					continue
				}
				for xx := x0; xx < x1; xx++ {
					if xx < 0 || xx >= src.W {
						continue
					}
					wx := overlapWeight(fx0, fx1, xx)
					if wx <= 0 {
						continue
					}
					w := wx * wy
					off := (yy*src.W + xx) * 3
					acc[0] += float64(src.Pix[off+0]) * w
					acc[1] += float64(src.Pix[off+1]) * w
					acc[2] += float64(src.Pix[off+2]) * w
					wsum += w
				}
			}
			if wsum == 0 {
				wsum = 1
			}
			o := (y*dw + x) * 3
			out.Pix[o+0] = float32(acc[0] / wsum)
			out.Pix[o+1] = float32(acc[1] / wsum)
			out.Pix[o+2] = float32(acc[2] / wsum)
		}
	}
	return out
}

// overlapWeight returns the fraction of source pixel index `idx` (spanning
// [idx, idx+1]) covered by the destination footprint [f0, f1).
func overlapWeight(f0, f1 float64, idx int) float64 {
	lo := math.Max(f0, float64(idx))
	hi := math.Min(f1, float64(idx+1))
	if hi <= lo {
		return 0
	}
	return hi - lo
}

// normalizeToPatches converts an already-resized f32Image to normalized
// patch rows: for each patch (row-major over the grid) a feature vector of
// 3*patchH*patchW values ordered (C, Y, X) with C slowest, matching the HF
// flatten convention (channel-major).
func normalizeToPatches(img *f32Image, patch int, mean, std [3]float32) (rows int, cols int, feats [][]float32) {
	ph := img.H / patch
	pw := img.W / patch
	feat := 3 * patch * patch
	out := make([][]float32, ph*pw)
	for py := 0; py < ph; py++ {
		for px := 0; px < pw; px++ {
			f := make([]float32, feat)
			for c := 0; c < 3; c++ {
				for y := 0; y < patch; y++ {
					for x := 0; x < patch; x++ {
						sx := px*patch + x
						sy := py*patch + y
						v := (img.Pix[(sy*img.W+sx)*3+c] - mean[c]) / std[c]
						// (C, Y, X) order: c * patch*patch + y * patch + x
						f[c*patch*patch+y*patch+x] = v
					}
				}
			}
			out[py*pw+px] = f
		}
	}
	return ph, pw, out
}

// convPatches applies the 4D conv-style patch embedding. feats are in (C, Y, X)
// HF order; the conv weight is in GGUF order [KH, KW, IC, OC] i.e. memory
// order (Y, X, C, OC) after the convert-time permute(2,3,1,0). We compute
// out[oc] = sum over (y, x, c) W[(y*KW + x)*IC + c][oc] * feat[c*ps*ps + y*ps + x].
func (e *Encoder) convPatches(feats [][]float32) [][]float32 {
	ps := e.Config.PatchSize
	ic := e.PatchIC
	oc := e.Config.EmbeddingDim
	kh, kw := e.PatchKH, e.PatchKW
	kernel := kh * kw * ic
	out := make([][]float32, len(feats))
	e.parallelFor(len(feats), func(idx int) {
		f := feats[idx]
		acc := make([]float64, oc)
		for y := 0; y < kh; y++ {
			for x := 0; x < kw; x++ {
				base := (y*kw + x) * ic
				for c := 0; c < ic; c++ {
					fv := float64(f[c*ps*ps+y*ps+x])
					if fv == 0 {
						continue
					}
					wRow := base + c
					for o := 0; o < oc; o++ {
						acc[o] += float64(e.Patch4DW[wRow+kernel*o]) * fv
					}
				}
			}
		}
		v := make([]float32, oc)
		for o := 0; o < oc; o++ {
			if e.PatchBias != nil {
				acc[o] += float64(e.PatchBias[o])
			}
			v[o] = float32(acc[o])
		}
		out[idx] = v
	})
	return out
}

// PrepareImage decodes, resizes (with optional aspect-preserving unpad),
// normalizes, and extracts patch features.
// Returns the patch grid dims and per-patch feature vectors.
func (e *Encoder) PrepareImage(src image.Image) (gridH, gridW int, feats [][]float32, err error) {
	if src == nil {
		return 0, 0, nil, fmt.Errorf("vision: nil image")
	}
	nrgba := decodeToRGBA(src)
	f := &f32Image{W: nrgba.Bounds().Dx(), H: nrgba.Bounds().Dy(), Pix: make([]float32, nrgba.Bounds().Dx()*nrgba.Bounds().Dy()*3)}
	i := 0
	for y := 0; y < f.H; y++ {
		for x := 0; x < f.W; x++ {
			r, g, b, _ := nrgba.At(x, y).RGBA()
			f.Pix[i+0] = float32(r>>8) / 255
			f.Pix[i+1] = float32(g>>8) / 255
			f.Pix[i+2] = float32(b>>8) / 255
			i += 3
		}
	}

	c := &e.Config
	if c.MergeType == "spatial_unpad" {
		// Resize the longest side to ImageSize, keep aspect, pad bottom/right
		// with the mean color (llava-1.6 convention).
		rw, rh := fitInside(f.W, f.H, c.ImageSize)
		resized := resizeBilinear(f, rw, rh)
		sq := &f32Image{W: c.ImageSize, H: c.ImageSize, Pix: make([]float32, c.ImageSize*c.ImageSize*3)}
		for j := range sq.Pix {
			sq.Pix[j] = c.ImageMean[j%3]
		}
		for y := 0; y < rh; y++ {
			copy(sq.Pix[(y*c.ImageSize)*3:(y*c.ImageSize+rw)*3], resized.Pix[(y*rw)*3:(y*rw+rw)*3])
		}
		ph, pw, feats := normalizeToPatches(sq, c.PatchSize, c.ImageMean, c.ImageStd)
		e.lastContentW = rw
		e.lastContentH = rh
		return ph, pw, feats, nil
	}

	resized := resizeBilinear(f, c.ImageSize, c.ImageSize)
	ph, pw, feats := normalizeToPatches(resized, c.PatchSize, c.ImageMean, c.ImageStd)
	return ph, pw, feats, nil
}

func fitInside(w, h, max int) (int, int) {
	if w >= h {
		rh := int(float64(h)*float64(max)/float64(w) + 0.5)
		return max, rh
	}
	rw := int(float64(w)*float64(max)/float64(h) + 0.5)
	return rw, max
}
