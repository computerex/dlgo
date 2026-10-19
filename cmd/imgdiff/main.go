// Command imgdiff reports the pixel-level difference between two PNGs.
// Used to verify that the unified Affine8 GEMM path reproduces the reference
// image within quantization noise (no byte-identity expected once the weight
// format changes).
package main

import (
	"fmt"
	"image"
	_ "image/png"
	"math"
	"os"
)

func load(p string) image.Image {
	f, err := os.Open(p)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		panic(err)
	}
	return img
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: imgdiff a.png b.png")
		os.Exit(2)
	}
	a := load(os.Args[1])
	b := load(os.Args[2])
	if a.Bounds() != b.Bounds() {
		fmt.Fprintf(os.Stderr, "size mismatch: %v vs %v\n", a.Bounds(), b.Bounds())
		os.Exit(1)
	}
	bd := a.Bounds()
	var mse, sumAbs, maxd float64
	n := 0
	for y := bd.Min.Y; y < bd.Max.Y; y++ {
		for x := bd.Min.X; x < bd.Max.X; x++ {
			r1, g1, b1, _ := a.At(x, y).RGBA()
			r2, g2, b2, _ := b.At(x, y).RGBA()
			for _, dd := range []float64{
				float64(int(r1>>8) - int(r2>>8)),
				float64(int(g1>>8) - int(g2>>8)),
				float64(int(b1>>8) - int(b2>>8)),
			} {
				mse += dd * dd
				sumAbs += math.Abs(dd)
				if math.Abs(dd) > maxd {
					maxd = math.Abs(dd)
				}
				n++
			}
		}
	}
	mse /= float64(n)
	psnr := 10 * math.Log10(255*255/mse)
	fmt.Printf("%s vs %s\n  PSNR=%.2f dB  meanAbs=%.3f  maxAbs=%.0f  samples=%d\n",
		os.Args[1], os.Args[2], psnr, sumAbs/float64(n), maxd, n)
}
