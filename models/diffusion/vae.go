package diffusion

import (
	"fmt"
	"log"
	"math"
	"sync"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/ops"
)

// im2colPool reuses column buffers across conv2d calls.
var im2colPool sync.Pool

// VAEConfig holds configuration for the FLUX AutoEncoderKL decoder.
type VAEConfig struct {
	ZChannels    int
	BaseCh       int
	ChMult       [4]int
	NumResBlocks int
	OutCh        int
	NumGroups    int
	Eps          float32
	ScaleFactor  float32
	ShiftFactor  float32
}

// DefaultVAEConfig returns the FLUX VAE configuration.
func DefaultVAEConfig() VAEConfig {
	return VAEConfig{
		ZChannels:    16,
		BaseCh:       128,
		ChMult:       [4]int{1, 2, 4, 4},
		NumResBlocks: 2,
		OutCh:        3,
		NumGroups:    32,
		Eps:          1e-6,
		ScaleFactor:  0.3611,
		ShiftFactor:  0.1159,
	}
}

// Conv2DWeight holds weights for a 2D convolution.
type Conv2DWeight struct {
	Weight []float32 // [outCh, inCh, kH, kW]
	Bias   []float32 // [outCh]
	InCh   int
	OutCh  int
	KH, KW int
}

// ResnetBlockWeights holds weights for a ResNet block.
type ResnetBlockWeights struct {
	Norm1  []float32 // GroupNorm weight [ch]
	Bias1  []float32 // GroupNorm bias [ch]
	Conv1  Conv2DWeight
	Norm2  []float32
	Bias2  []float32
	Conv2  Conv2DWeight
	NinSC  *Conv2DWeight // 1x1 shortcut, nil if in==out channels
}

// AttnBlockWeights holds weights for the mid-block attention.
type AttnBlockWeights struct {
	Norm   []float32 // GroupNorm weight [ch]
	Bias   []float32 // GroupNorm bias [ch]
	Q      Conv2DWeight
	K      Conv2DWeight
	V      Conv2DWeight
	ProjOut Conv2DWeight
}

// VAEDecoder holds all weights for the FLUX VAE decoder.
type VAEDecoder struct {
	Config VAEConfig

	ConvIn Conv2DWeight

	MidBlock1 ResnetBlockWeights
	MidAttn1  AttnBlockWeights
	MidBlock2 ResnetBlockWeights

	// Up blocks: index 0 is lowest resolution, 3 is highest
	// Each has NumResBlocks+1 = 3 ResNet blocks
	UpBlocks [4][3]ResnetBlockWeights
	UpSample [4]*Conv2DWeight // nil for level 0

	NormOut []float32
	BiasOut []float32
	ConvOut Conv2DWeight
}

// LoadVAEDecoder loads the FLUX VAE decoder from a safetensors file.
func LoadVAEDecoder(path string) (*VAEDecoder, error) {
	sf, err := OpenSafetensors(path)
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}

	cfg := DefaultVAEConfig()
	d := &VAEDecoder{Config: cfg}

	get := func(name string) []float32 {
		data, _, err := sf.GetFloat32(name)
		if err != nil {
			log.Printf("Warning: VAE tensor %q not found: %v", name, err)
			return nil
		}
		return data
	}

	getConv := func(prefix string, inCh, outCh, k int) Conv2DWeight {
		return Conv2DWeight{
			Weight: get(prefix + ".weight"),
			Bias:   get(prefix + ".bias"),
			InCh:   inCh,
			OutCh:  outCh,
			KH:     k,
			KW:     k,
		}
	}

	getResBlock := func(prefix string, inCh, outCh int) ResnetBlockWeights {
		rb := ResnetBlockWeights{
			Norm1: get(prefix + ".norm1.weight"),
			Bias1: get(prefix + ".norm1.bias"),
			Conv1: getConv(prefix+".conv1", inCh, outCh, 3),
			Norm2: get(prefix + ".norm2.weight"),
			Bias2: get(prefix + ".norm2.bias"),
			Conv2: getConv(prefix+".conv2", outCh, outCh, 3),
		}
		if inCh != outCh {
			sc := getConv(prefix+".nin_shortcut", inCh, outCh, 1)
			rb.NinSC = &sc
		}
		return rb
	}

	topCh := cfg.BaseCh * cfg.ChMult[len(cfg.ChMult)-1] // 512

	// conv_in: z_channels → topCh
	d.ConvIn = getConv("decoder.conv_in", cfg.ZChannels, topCh, 3)

	// Mid blocks
	d.MidBlock1 = getResBlock("decoder.mid.block_1", topCh, topCh)
	d.MidAttn1 = AttnBlockWeights{
		Norm: get("decoder.mid.attn_1.norm.weight"),
		Bias: get("decoder.mid.attn_1.norm.bias"),
		Q:    getConv("decoder.mid.attn_1.q", topCh, topCh, 1),
		K:    getConv("decoder.mid.attn_1.k", topCh, topCh, 1),
		V:    getConv("decoder.mid.attn_1.v", topCh, topCh, 1),
		ProjOut: getConv("decoder.mid.attn_1.proj_out", topCh, topCh, 1),
	}
	d.MidBlock2 = getResBlock("decoder.mid.block_2", topCh, topCh)

	// Up blocks: i=3,2,1,0 (reverse order in forward pass)
	blockInCh := topCh
	for i := len(cfg.ChMult) - 1; i >= 0; i-- {
		outCh := cfg.BaseCh * cfg.ChMult[i]
		for j := 0; j < cfg.NumResBlocks+1; j++ {
			inCh := blockInCh
			if j > 0 {
				inCh = outCh
			}
			prefix := fmt.Sprintf("decoder.up.%d.block.%d", i, j)
			d.UpBlocks[i][j] = getResBlock(prefix, inCh, outCh)
		}
		if i > 0 {
			prefix := fmt.Sprintf("decoder.up.%d.upsample.conv", i)
			sc := getConv(prefix, outCh, outCh, 3)
			d.UpSample[i] = &sc
		}
		blockInCh = outCh
	}

	// Output
	d.NormOut = get("decoder.norm_out.weight")
	d.BiasOut = get("decoder.norm_out.bias")
	d.ConvOut = getConv("decoder.conv_out", cfg.BaseCh, cfg.OutCh, 3)

	log.Printf("Loaded VAE decoder: %d→%d channels, %dx spatial upscale",
		cfg.ZChannels, cfg.OutCh, 1<<(len(cfg.ChMult)-1))

	return d, nil
}

// VAEDecode decodes a latent tensor to an RGB image.
// latent: [z_channels, H, W] flat, where H,W are latent dimensions
// Returns: [3, H*8, W*8] flat as float32 in [0, 1] range
func VAEDecode(d *VAEDecoder, latent []float32, H, W int) []float32 {
	cfg := d.Config

	// Un-scale latent: z = latent / scale_factor + shift_factor
	z := make([]float32, len(latent))
	for i := range latent {
		z[i] = latent[i]/cfg.ScaleFactor + cfg.ShiftFactor
	}

	log.Printf("VAE decode: input [%d, %d, %d]", cfg.ZChannels, H, W)

	// conv_in
	h := conv2d(z, d.ConvIn, H, W, 1)
	curH, curW := H, W

	log.Printf("  conv_in → [%d, %d, %d]", d.ConvIn.OutCh, curH, curW)

	// Mid blocks
	h = resnetBlockForward(h, &d.MidBlock1, curH, curW, cfg.NumGroups, cfg.Eps)
	h = attnBlockForward(h, &d.MidAttn1, d.MidBlock1.Conv2.OutCh, curH, curW, cfg.NumGroups, cfg.Eps)
	h = resnetBlockForward(h, &d.MidBlock2, curH, curW, cfg.NumGroups, cfg.Eps)

	log.Printf("  mid → [%d, %d, %d]", d.MidBlock2.Conv2.OutCh, curH, curW)

	// Up blocks: i = 3, 2, 1, 0
	for i := len(cfg.ChMult) - 1; i >= 0; i-- {
		for j := 0; j < cfg.NumResBlocks+1; j++ {
			h = resnetBlockForward(h, &d.UpBlocks[i][j], curH, curW, cfg.NumGroups, cfg.Eps)
		}
		if i > 0 && d.UpSample[i] != nil {
			h = upsample2x(h, d.UpBlocks[i][cfg.NumResBlocks].Conv2.OutCh, curH, curW)
			curH *= 2
			curW *= 2
			h = conv2d(h, *d.UpSample[i], curH, curW, 1)
		}
		outCh := d.UpBlocks[i][cfg.NumResBlocks].Conv2.OutCh
		log.Printf("  up.%d → [%d, %d, %d]", i, outCh, curH, curW)
	}

	// norm_out + silu + conv_out
	outCh := d.UpBlocks[0][cfg.NumResBlocks].Conv2.OutCh
	groupNormBatch(h, h, d.NormOut, d.BiasOut, outCh, curH*curW, cfg.NumGroups, cfg.Eps)
	ops.SiLU(h)
	h = conv2d(h, d.ConvOut, curH, curW, 1)

	log.Printf("  conv_out → [%d, %d, %d]", cfg.OutCh, curH, curW)

	// Post-process: (x + 1) / 2, clamp to [0, 1]
	for i := range h {
		v := (h[i] + 1.0) * 0.5
		if v < 0 {
			v = 0
		}
		if v > 1 {
			v = 1
		}
		h[i] = v
	}

	return h
}

// conv2d applies a 2D convolution with padding=kernel/2 and stride.
// input: [inCh, H, W] flat, output: [outCh, outH, outW] flat
// Uses im2col + parallel matmul for performance.
func conv2d(input []float32, w Conv2DWeight, H, W, stride int) []float32 {
	inCh := w.InCh
	outCh := w.OutCh
	kH := w.KH
	kW := w.KW
	padH := kH / 2
	padW := kW / 2
	outH := (H + 2*padH - kH)/stride + 1
	outW := (W + 2*padW - kW)/stride + 1

	spatialOut := outH * outW
	colSize := inCh * kH * kW // length of one column

	// Get or allocate column buffer [spatialOut, colSize]
	colLen := spatialOut * colSize
	var col []float32
	if v := im2colPool.Get(); v != nil {
		buf := v.([]float32)
		if cap(buf) >= colLen {
			col = buf[:colLen]
		}
	}
	if col == nil {
		col = make([]float32, colLen)
	}

	// im2col: fill column matrix
	for oh := 0; oh < outH; oh++ {
		for ow := 0; ow < outW; ow++ {
			colOff := (oh*outW + ow) * colSize
			for ic := 0; ic < inCh; ic++ {
				for kh := 0; kh < kH; kh++ {
					ih := oh*stride + kh - padH
					for kw := 0; kw < kW; kw++ {
						iw := ow*stride + kw - padW
						idx := colOff + ic*kH*kW + kh*kW + kw
						if ih >= 0 && ih < H && iw >= 0 && iw < W {
							col[idx] = input[ic*H*W+ih*W+iw]
						} else {
							col[idx] = 0
						}
					}
				}
			}
		}
	}

	// output = weight[outCh, colSize] × col^T → [outCh, spatialOut]
	out := make([]float32, outCh*spatialOut)
	pool := blas.DefaultPool()

	pool.ParallelFor(outCh, func(oc int) {
		wRow := w.Weight[oc*colSize : (oc+1)*colSize]
		biasVal := float32(0)
		if w.Bias != nil {
			biasVal = w.Bias[oc]
		}
		outRow := out[oc*spatialOut : (oc+1)*spatialOut]
		for s := 0; s < spatialOut; s++ {
			outRow[s] = ops.DotProduct(wRow, col[s*colSize:s*colSize+colSize], colSize) + biasVal
		}
	})

	im2colPool.Put(col)
	return out
}

// conv2dNaive is the reference implementation for testing.
func conv2dNaive(input []float32, w Conv2DWeight, H, W, stride int) []float32 {
	inCh := w.InCh
	outCh := w.OutCh
	kH := w.KH
	kW := w.KW
	padH := kH / 2
	padW := kW / 2
	outH := (H + 2*padH - kH)/stride + 1
	outW := (W + 2*padW - kW)/stride + 1

	out := make([]float32, outCh*outH*outW)

	for oc := 0; oc < outCh; oc++ {
		biasVal := float32(0)
		if w.Bias != nil {
			biasVal = w.Bias[oc]
		}
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				sum := biasVal
				for ic := 0; ic < inCh; ic++ {
					for kh := 0; kh < kH; kh++ {
						for kw := 0; kw < kW; kw++ {
							ih := oh*stride + kh - padH
							iw := ow*stride + kw - padW
							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								wIdx := oc*inCh*kH*kW + ic*kH*kW + kh*kW + kw
								iIdx := ic*H*W + ih*W + iw
								sum += w.Weight[wIdx] * input[iIdx]
							}
						}
					}
				}
				out[oc*outH*outW+oh*outW+ow] = sum
			}
		}
	}
	return out
}

// upsample2x performs nearest-neighbor 2× upsampling on [C, H, W] data.
func upsample2x(input []float32, C, H, W int) []float32 {
	outH := H * 2
	outW := W * 2
	out := make([]float32, C*outH*outW)
	pool := blas.DefaultPool()
	pool.ParallelFor(C, func(c int) {
		for h := 0; h < outH; h++ {
			srcH := h / 2
			for w := 0; w < outW; w++ {
				srcW := w / 2
				out[c*outH*outW+h*outW+w] = input[c*H*W+srcH*W+srcW]
			}
		}
	})
	return out
}

// groupNormBatch applies GroupNorm to a [C, spatialSize] tensor.
func groupNormBatch(out, x, w, b []float32, C, spatialSize, numGroups int, eps float32) {
	groupSize := C / numGroups
	pool := blas.DefaultPool()
	pool.ParallelFor(numGroups, func(g int) {
		// Compute mean and variance over the group
		startCh := g * groupSize
		n := groupSize * spatialSize
		var sum, sumSq float64
		for c := 0; c < groupSize; c++ {
			ch := startCh + c
			for s := 0; s < spatialSize; s++ {
				v := float64(x[ch*spatialSize+s])
				sum += v
				sumSq += v * v
			}
		}
		mean := sum / float64(n)
		variance := sumSq/float64(n) - mean*mean
		invStd := float32(1.0 / math.Sqrt(float64(variance)+float64(eps)))

		for c := 0; c < groupSize; c++ {
			ch := startCh + c
			gamma := w[ch]
			beta := b[ch]
			for s := 0; s < spatialSize; s++ {
				idx := ch*spatialSize + s
				out[idx] = (x[idx]-float32(mean))*invStd*gamma + beta
			}
		}
	})
}

// resnetBlockForward applies a ResNet block.
func resnetBlockForward(input []float32, rb *ResnetBlockWeights, H, W, numGroups int, eps float32) []float32 {
	inCh := rb.Conv1.InCh
	outCh := rb.Conv2.OutCh
	spatial := H * W

	// h = GroupNorm + SiLU + Conv1
	h := make([]float32, len(input))
	groupNormBatch(h, input, rb.Norm1, rb.Bias1, inCh, spatial, numGroups, eps)
	ops.SiLU(h)
	h = conv2d(h, rb.Conv1, H, W, 1)

	// h = GroupNorm + SiLU + Conv2
	h2 := make([]float32, len(h))
	groupNormBatch(h2, h, rb.Norm2, rb.Bias2, outCh, spatial, numGroups, eps)
	ops.SiLU(h2)
	h2 = conv2d(h2, rb.Conv2, H, W, 1)

	// Shortcut
	skip := input
	if rb.NinSC != nil {
		skip = conv2d(input, *rb.NinSC, H, W, 1)
	}

	// Residual
	for i := range h2 {
		h2[i] += skip[i]
	}
	return h2
}

// attnBlockForward applies mid-block self-attention.
func attnBlockForward(input []float32, attn *AttnBlockWeights, C, H, W, numGroups int, eps float32) []float32 {
	spatial := H * W

	// GroupNorm
	normed := make([]float32, len(input))
	groupNormBatch(normed, input, attn.Norm, attn.Bias, C, spatial, numGroups, eps)

	// Q, K, V projections (1x1 conv = per-spatial-position linear)
	q := conv2d(normed, attn.Q, H, W, 1)
	k := conv2d(normed, attn.K, H, W, 1)
	v := conv2d(normed, attn.V, H, W, 1)

	// Single-head attention over spatial positions
	// q, k, v: [C, H*W] → treat as [spatial, C] for attention
	scale := float32(1.0 / math.Sqrt(float64(C)))
	out := make([]float32, C*spatial)

	pool := blas.DefaultPool()
	pool.ParallelFor(spatial, func(qi int) {
		scores := make([]float32, spatial)
		for ki := 0; ki < spatial; ki++ {
			dot := float32(0)
			for c := 0; c < C; c++ {
				dot += q[c*spatial+qi] * k[c*spatial+ki]
			}
			scores[ki] = dot * scale
		}

		ops.Softmax(scores)

		// Weighted sum of values
		for c := 0; c < C; c++ {
			sum := float32(0)
			for ki := 0; ki < spatial; ki++ {
				sum += scores[ki] * v[c*spatial+ki]
			}
			out[c*spatial+qi] = sum
		}
	})

	// Output projection
	out = conv2d(out, attn.ProjOut, H, W, 1)

	// Residual
	for i := range out {
		out[i] += input[i]
	}
	return out
}
