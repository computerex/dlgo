//go:build cgo && vulkan

package diffusion

import (
	"fmt"
	"log"
	"math"

	"github.com/computerex/dlgo/gpu"
)

// GpuConv2DWeight holds a conv2d weight+bias pair on GPU.
type GpuConv2DWeight struct {
	Weight gpu.Buf
	Bias   gpu.Buf
	InCh   int
	OutCh  int
	KH, KW int
}

// GpuResnetBlock holds GPU buffers for one ResNet block.
type GpuResnetBlock struct {
	Norm1W, Norm1B gpu.Buf
	Conv1          GpuConv2DWeight
	Norm2W, Norm2B gpu.Buf
	Conv2          GpuConv2DWeight
	NinSC          *GpuConv2DWeight // nil if in==out channels
}

// GpuAttnBlock holds GPU buffers for mid-block attention.
type GpuAttnBlock struct {
	NormW, NormB gpu.Buf
	Q, K, V      GpuConv2DWeight
	ProjOut      GpuConv2DWeight
}

// GpuVAEModel holds all VAE decoder weights on GPU.
type GpuVAEModel struct {
	Config VAEConfig

	ConvIn GpuConv2DWeight

	MidBlock1 GpuResnetBlock
	MidAttn1  GpuAttnBlock
	MidBlock2 GpuResnetBlock

	UpBlocks [4][3]GpuResnetBlock
	UpSample [4]*GpuConv2DWeight // nil for level 0

	NormOutW, NormOutB gpu.Buf
	ConvOut            GpuConv2DWeight
}

// GpuVAERunState holds scratch buffers for GPU VAE decode.
type GpuVAERunState struct {
	Act  gpu.Buf // main activation buffer
	Tmp1 gpu.Buf // temp for GroupNorm+SiLU intermediate
	Tmp2 gpu.Buf // temp for conv output / shortcut
	Ups  gpu.Buf // upsample output

	// Mid-block attention scratch (small: 512 × 1024 = 2MB each)
	AttnQ   gpu.Buf
	AttnK   gpu.Buf
	AttnV   gpu.Buf
	AttnOut gpu.Buf

	MaxFloats int // max buffer size in floats
}

// uploadConv uploads a Conv2DWeight to GPU.
func uploadConv(w *Conv2DWeight) (GpuConv2DWeight, error) {
	gw := GpuConv2DWeight{InCh: w.InCh, OutCh: w.OutCh, KH: w.KH, KW: w.KW}
	var err error

	gw.Weight, err = gpu.AllocE(uint64(len(w.Weight)) * 4)
	if err != nil {
		return gw, fmt.Errorf("conv weight: %w", err)
	}
	if err = gpu.UploadF32(gw.Weight, w.Weight); err != nil {
		return gw, err
	}

	gw.Bias, err = gpu.AllocE(uint64(len(w.Bias)) * 4)
	if err != nil {
		return gw, fmt.Errorf("conv bias: %w", err)
	}
	if err = gpu.UploadF32(gw.Bias, w.Bias); err != nil {
		return gw, err
	}

	return gw, nil
}

// uploadF32 uploads a float32 slice to a new GPU buffer.
func uploadF32VAE(data []float32) (gpu.Buf, error) {
	buf, err := gpu.AllocE(uint64(len(data)) * 4)
	if err != nil {
		return 0, err
	}
	if err = gpu.UploadF32(buf, data); err != nil {
		return 0, err
	}
	return buf, nil
}

// uploadResnetBlock uploads a ResnetBlockWeights to GPU.
func uploadResnetBlock(rb *ResnetBlockWeights) (GpuResnetBlock, error) {
	var grb GpuResnetBlock
	var err error

	grb.Norm1W, err = uploadF32VAE(rb.Norm1)
	if err != nil {
		return grb, fmt.Errorf("norm1w: %w", err)
	}
	grb.Norm1B, err = uploadF32VAE(rb.Bias1)
	if err != nil {
		return grb, fmt.Errorf("norm1b: %w", err)
	}
	grb.Conv1, err = uploadConv(&rb.Conv1)
	if err != nil {
		return grb, fmt.Errorf("conv1: %w", err)
	}
	grb.Norm2W, err = uploadF32VAE(rb.Norm2)
	if err != nil {
		return grb, fmt.Errorf("norm2w: %w", err)
	}
	grb.Norm2B, err = uploadF32VAE(rb.Bias2)
	if err != nil {
		return grb, fmt.Errorf("norm2b: %w", err)
	}
	grb.Conv2, err = uploadConv(&rb.Conv2)
	if err != nil {
		return grb, fmt.Errorf("conv2: %w", err)
	}
	if rb.NinSC != nil {
		sc, err := uploadConv(rb.NinSC)
		if err != nil {
			return grb, fmt.Errorf("ninsc: %w", err)
		}
		grb.NinSC = &sc
	}
	return grb, nil
}

// UploadVAEModel uploads all VAE decoder weights to GPU.
func UploadVAEModel(d *VAEDecoder) (*GpuVAEModel, error) {
	gm := &GpuVAEModel{Config: d.Config}
	var err error

	gm.ConvIn, err = uploadConv(&d.ConvIn)
	if err != nil {
		return nil, fmt.Errorf("conv_in: %w", err)
	}

	gm.MidBlock1, err = uploadResnetBlock(&d.MidBlock1)
	if err != nil {
		return nil, fmt.Errorf("mid_block1: %w", err)
	}

	// Mid attention
	gm.MidAttn1.NormW, err = uploadF32VAE(d.MidAttn1.Norm)
	if err != nil {
		return nil, fmt.Errorf("mid_attn norm: %w", err)
	}
	gm.MidAttn1.NormB, err = uploadF32VAE(d.MidAttn1.Bias)
	if err != nil {
		return nil, fmt.Errorf("mid_attn bias: %w", err)
	}
	gm.MidAttn1.Q, err = uploadConv(&d.MidAttn1.Q)
	if err != nil {
		return nil, fmt.Errorf("mid_attn Q: %w", err)
	}
	gm.MidAttn1.K, err = uploadConv(&d.MidAttn1.K)
	if err != nil {
		return nil, fmt.Errorf("mid_attn K: %w", err)
	}
	gm.MidAttn1.V, err = uploadConv(&d.MidAttn1.V)
	if err != nil {
		return nil, fmt.Errorf("mid_attn V: %w", err)
	}
	gm.MidAttn1.ProjOut, err = uploadConv(&d.MidAttn1.ProjOut)
	if err != nil {
		return nil, fmt.Errorf("mid_attn proj: %w", err)
	}

	gm.MidBlock2, err = uploadResnetBlock(&d.MidBlock2)
	if err != nil {
		return nil, fmt.Errorf("mid_block2: %w", err)
	}

	// Up blocks
	cfg := d.Config
	for i := len(cfg.ChMult) - 1; i >= 0; i-- {
		for j := 0; j < cfg.NumResBlocks+1; j++ {
			gm.UpBlocks[i][j], err = uploadResnetBlock(&d.UpBlocks[i][j])
			if err != nil {
				return nil, fmt.Errorf("up.%d.block.%d: %w", i, j, err)
			}
		}
		if i > 0 && d.UpSample[i] != nil {
			sc, err := uploadConv(d.UpSample[i])
			if err != nil {
				return nil, fmt.Errorf("up.%d.upsample: %w", i, err)
			}
			gm.UpSample[i] = &sc
		}
	}

	// Output norm + conv
	gm.NormOutW, err = uploadF32VAE(d.NormOut)
	if err != nil {
		return nil, fmt.Errorf("norm_out: %w", err)
	}
	gm.NormOutB, err = uploadF32VAE(d.BiasOut)
	if err != nil {
		return nil, fmt.Errorf("bias_out: %w", err)
	}
	gm.ConvOut, err = uploadConv(&d.ConvOut)
	if err != nil {
		return nil, fmt.Errorf("conv_out: %w", err)
	}

	log.Printf("[vae/gpu] VAE weights uploaded, allocated: %.1f MB",
		float64(gpu.AllocatedBytes())/(1024*1024))
	return gm, nil
}

// NewGpuVAERunState allocates scratch buffers for GPU VAE decode.
func NewGpuVAERunState(cfg VAEConfig, latentH, latentW int) (*GpuVAERunState, error) {
	// Compute max activation size across all stages
	// The maximum is at up.1 output after upsample: [chMult[1]*baseCh, finalH, finalW]
	// For FLUX: [256, 256, 256] = 16.7M floats
	finalH := latentH * 8
	finalW := latentW * 8

	maxFloats := 0
	curH, curW := latentH, latentW
	topCh := cfg.BaseCh * cfg.ChMult[len(cfg.ChMult)-1]

	// conv_in output
	n := topCh * curH * curW
	if n > maxFloats {
		maxFloats = n
	}

	// Walk through up blocks to find max
	ch := topCh
	for i := len(cfg.ChMult) - 1; i >= 0; i-- {
		outCh := cfg.BaseCh * cfg.ChMult[i]
		// ResNet blocks may change channels (first block)
		inCh := ch
		for j := 0; j < cfg.NumResBlocks+1; j++ {
			if j > 0 {
				inCh = outCh
			}
			n = inCh * curH * curW
			if n > maxFloats {
				maxFloats = n
			}
			n = outCh * curH * curW
			if n > maxFloats {
				maxFloats = n
			}
		}
		if i > 0 {
			// After upsample: spatial doubles
			n = outCh * curH * 2 * curW * 2
			if n > maxFloats {
				maxFloats = n
			}
			curH *= 2
			curW *= 2
		}
		ch = outCh
	}

	// Final output: [3, finalH, finalW]
	n = cfg.OutCh * finalH * finalW
	if n > maxFloats {
		maxFloats = n
	}

	grs := &GpuVAERunState{MaxFloats: maxFloats}
	var err error

	bufSize := uint64(maxFloats) * 4

	grs.Act, err = gpu.AllocE(bufSize)
	if err != nil {
		return nil, fmt.Errorf("act buffer: %w", err)
	}
	grs.Tmp1, err = gpu.AllocE(bufSize)
	if err != nil {
		return nil, fmt.Errorf("tmp1 buffer: %w", err)
	}
	grs.Tmp2, err = gpu.AllocE(bufSize)
	if err != nil {
		return nil, fmt.Errorf("tmp2 buffer: %w", err)
	}
	grs.Ups, err = gpu.AllocE(bufSize)
	if err != nil {
		return nil, fmt.Errorf("ups buffer: %w", err)
	}

	// Mid-block attention scratch
	midSpatial := latentH * latentW
	attnSize := uint64(topCh*midSpatial) * 4
	grs.AttnQ, err = gpu.AllocE(attnSize)
	if err != nil {
		return nil, fmt.Errorf("attn Q: %w", err)
	}
	grs.AttnK, err = gpu.AllocE(attnSize)
	if err != nil {
		return nil, fmt.Errorf("attn K: %w", err)
	}
	grs.AttnV, err = gpu.AllocE(attnSize)
	if err != nil {
		return nil, fmt.Errorf("attn V: %w", err)
	}
	grs.AttnOut, err = gpu.AllocE(attnSize)
	if err != nil {
		return nil, fmt.Errorf("attn out: %w", err)
	}

	log.Printf("[vae/gpu] Scratch buffers: %d max floats (%.1f MB × 4 bufs + %.1f MB × 4 attn)",
		maxFloats, float64(bufSize)/(1024*1024), float64(attnSize)/(1024*1024))
	return grs, nil
}

// gpuConv2d dispatches a Conv2D on GPU.
func gpuConv2d(out, in gpu.Buf, gw *GpuConv2DWeight, H, W int) error {
	padH := gw.KH / 2
	padW := gw.KW / 2
	stride := 1
	outH := (H + 2*padH - gw.KH)/stride + 1
	outW := (W + 2*padW - gw.KW)/stride + 1
	return gpu.Conv2dF32(out, in, gw.Weight, gw.Bias,
		gw.InCh, H, W, gw.KH, gw.KW, padH, padW, stride, outH, outW, gw.OutCh)
}

// gpuResnetBlock runs a ResNet block on GPU.
// Uses act as input (preserved for skip), tmp1 as GroupNorm+SiLU scratch, tmp2 as conv output.
// Result is written back to act.
func gpuResnetBlock(act, tmp1, tmp2 gpu.Buf, grb *GpuResnetBlock, H, W, numGroups int, eps float32) error {
	inCh := grb.Conv1.InCh
	outCh := grb.Conv2.OutCh
	spatial := H * W

	// h = GroupNorm(input) + SiLU
	if err := gpu.GroupNorm(tmp1, act, grb.Norm1W, grb.Norm1B, inCh, spatial, numGroups, eps); err != nil {
		return fmt.Errorf("resnet norm1: %w", err)
	}
	gpu.Barrier()
	if err := gpu.SiLU(tmp1, inCh*spatial); err != nil {
		return fmt.Errorf("resnet silu1: %w", err)
	}
	gpu.Barrier()

	// h = Conv1(h) → tmp2
	if err := gpuConv2d(tmp2, tmp1, &grb.Conv1, H, W); err != nil {
		return fmt.Errorf("resnet conv1: %w", err)
	}
	gpu.Barrier()

	// h2 = GroupNorm(h) + SiLU → tmp1
	if err := gpu.GroupNorm(tmp1, tmp2, grb.Norm2W, grb.Norm2B, outCh, spatial, numGroups, eps); err != nil {
		return fmt.Errorf("resnet norm2: %w", err)
	}
	gpu.Barrier()
	if err := gpu.SiLU(tmp1, outCh*spatial); err != nil {
		return fmt.Errorf("resnet silu2: %w", err)
	}
	gpu.Barrier()

	// h2 = Conv2(h2) → tmp2
	if err := gpuConv2d(tmp2, tmp1, &grb.Conv2, H, W); err != nil {
		return fmt.Errorf("resnet conv2: %w", err)
	}
	gpu.Barrier()

	// Shortcut
	if grb.NinSC != nil {
		// NinSC is 1×1 conv: tmp1 = NinSC(act)
		if err := gpuConv2d(tmp1, act, grb.NinSC, H, W); err != nil {
			return fmt.Errorf("resnet ninsc: %w", err)
		}
		gpu.Barrier()
		// output = tmp2 + tmp1 → act
		if err := gpu.Add(act, tmp2, tmp1, outCh*spatial); err != nil {
			return fmt.Errorf("resnet residual: %w", err)
		}
	} else {
		// output = tmp2 + act → act (in-place add to act)
		if err := gpu.Add(act, tmp2, act, outCh*spatial); err != nil {
			return fmt.Errorf("resnet residual: %w", err)
		}
	}

	return nil
}

// gpuAttnBlock runs the mid-block attention on GPU.
// act holds input [C, H*W], writes output back to act.
func gpuAttnBlock(act, tmp1 gpu.Buf, attn *GpuAttnBlock, aq, ak, av, aout gpu.Buf,
	C, H, W, numGroups int, eps float32) error {
	spatial := H * W

	// GroupNorm → tmp1
	if err := gpu.GroupNorm(tmp1, act, attn.NormW, attn.NormB, C, spatial, numGroups, eps); err != nil {
		return fmt.Errorf("attn norm: %w", err)
	}
	gpu.Barrier()

	// Q, K, V projections (1×1 conv)
	if err := gpuConv2d(aq, tmp1, &attn.Q, H, W); err != nil {
		return fmt.Errorf("attn Q: %w", err)
	}
	if err := gpuConv2d(ak, tmp1, &attn.K, H, W); err != nil {
		return fmt.Errorf("attn K: %w", err)
	}
	if err := gpuConv2d(av, tmp1, &attn.V, H, W); err != nil {
		return fmt.Errorf("attn V: %w", err)
	}
	gpu.Barrier()

	// Spatial self-attention
	scale := float32(1.0 / math.Sqrt(float64(C)))
	if err := gpu.SpatialAttention(aout, aq, ak, av, C, spatial, scale); err != nil {
		return fmt.Errorf("attn: %w", err)
	}
	gpu.Barrier()

	// Output projection → tmp1
	if err := gpuConv2d(tmp1, aout, &attn.ProjOut, H, W); err != nil {
		return fmt.Errorf("attn proj: %w", err)
	}
	gpu.Barrier()

	// Residual: act = tmp1 + act
	if err := gpu.Add(act, tmp1, act, C*spatial); err != nil {
		return fmt.Errorf("attn residual: %w", err)
	}

	return nil
}

// GpuVAEDecode runs the full VAE decode on GPU.
// latent: [z_channels, H, W] CPU float32
// Returns: [3, H*8, W*8] CPU float32 in [0, 1] range
func GpuVAEDecode(d *VAEDecoder, gm *GpuVAEModel, grs *GpuVAERunState,
	latent []float32, H, W int) []float32 {
	cfg := d.Config

	// Un-scale latent on CPU (cheap, 16×32×32 = 16K elements)
	z := make([]float32, len(latent))
	for i := range latent {
		z[i] = latent[i]/cfg.ScaleFactor + cfg.ShiftFactor
	}

	// Upload latent to GPU
	gpu.UploadF32(grs.Act, z)

	curH, curW := H, W

	gpu.BeginBatch()

	// conv_in: [16, H, W] → [512, H, W]
	gpuConv2d(grs.Tmp1, grs.Act, &gm.ConvIn, curH, curW)
	gpu.Barrier()
	// Copy conv_in output to act
	n := gm.ConvIn.OutCh * curH * curW
	gpu.CopyRegion(grs.Act, 0, grs.Tmp1, 0, uint64(n)*4)
	gpu.Barrier()

	log.Printf("[vae/gpu] conv_in → [%d, %d, %d]", gm.ConvIn.OutCh, curH, curW)

	// Mid blocks
	gpuResnetBlock(grs.Act, grs.Tmp1, grs.Tmp2, &gm.MidBlock1, curH, curW, cfg.NumGroups, cfg.Eps)
	gpu.Barrier()

	topCh := cfg.BaseCh * cfg.ChMult[len(cfg.ChMult)-1]
	gpuAttnBlock(grs.Act, grs.Tmp1, &gm.MidAttn1,
		grs.AttnQ, grs.AttnK, grs.AttnV, grs.AttnOut,
		topCh, curH, curW, cfg.NumGroups, cfg.Eps)
	gpu.Barrier()

	gpuResnetBlock(grs.Act, grs.Tmp1, grs.Tmp2, &gm.MidBlock2, curH, curW, cfg.NumGroups, cfg.Eps)
	gpu.Barrier()

	gpu.EndBatch()
	log.Printf("[vae/gpu] mid → [%d, %d, %d]", topCh, curH, curW)

	// Up blocks: i = 3, 2, 1, 0
	for i := len(cfg.ChMult) - 1; i >= 0; i-- {
		gpu.BeginBatch()

		for j := 0; j < cfg.NumResBlocks+1; j++ {
			gpuResnetBlock(grs.Act, grs.Tmp1, grs.Tmp2, &gm.UpBlocks[i][j], curH, curW, cfg.NumGroups, cfg.Eps)
			gpu.Barrier()
		}

		if i > 0 && gm.UpSample[i] != nil {
			outCh := gm.UpBlocks[i][cfg.NumResBlocks].Conv2.OutCh
			// Upsample: act → ups
			gpu.UpsampleNearest(grs.Ups, grs.Act, outCh, curH, curW)
			curH *= 2
			curW *= 2
			gpu.Barrier()
			// Conv after upsample: ups → act
			gpuConv2d(grs.Act, grs.Ups, gm.UpSample[i], curH, curW)
			gpu.Barrier()
		}

		gpu.EndBatch()

		outCh := gm.UpBlocks[i][cfg.NumResBlocks].Conv2.OutCh
		log.Printf("[vae/gpu] up.%d → [%d, %d, %d]", i, outCh, curH, curW)
	}

	// norm_out + silu + conv_out
	gpu.BeginBatch()

	outCh := gm.UpBlocks[0][cfg.NumResBlocks].Conv2.OutCh
	spatial := curH * curW
	gpu.GroupNorm(grs.Tmp1, grs.Act, gm.NormOutW, gm.NormOutB, outCh, spatial, cfg.NumGroups, cfg.Eps)
	gpu.Barrier()
	gpu.SiLU(grs.Tmp1, outCh*spatial)
	gpu.Barrier()
	gpuConv2d(grs.Act, grs.Tmp1, &gm.ConvOut, curH, curW)

	gpu.EndBatch()
	log.Printf("[vae/gpu] conv_out → [%d, %d, %d]", cfg.OutCh, curH, curW)

	// Download result
	outSize := cfg.OutCh * curH * curW
	result := make([]float32, outSize)
	gpu.DownloadF32(grs.Act, result)

	// Post-process on CPU (cheap: 3 × 256 × 256 = 196K elements)
	for i := range result {
		v := (result[i] + 1.0) * 0.5
		if v < 0 {
			v = 0
		}
		if v > 1 {
			v = 1
		}
		result[i] = v
	}

	return result
}
