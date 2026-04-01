package diffusion

import (
	"fmt"
	"log"
	"strconv"
	"strings"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/format/gguf"
	"github.com/computerex/dlgo/mmap"
	"github.com/computerex/dlgo/quant"
)

// DiTLayer holds weights for one JointTransformerBlock.
type DiTLayer struct {
	// Attention
	AttnQKV  *core.QuantizedTensor // [hiddenSize, (numHeads+numKVHeads*2)*headDim]
	AttnOut  *core.QuantizedTensor // [numHeads*headDim, hiddenSize]
	QNorm    []float32             // [headDim]
	KNorm    []float32             // [headDim]
	AttnNorm1 []float32            // [hiddenSize] pre-attn norm
	AttnNorm2 []float32            // [hiddenSize] post-attn norm

	// FFN (SwiGLU)
	FFNGate  *core.QuantizedTensor // w1: [hiddenSize, ffnDim] gate
	FFNDown  *core.QuantizedTensor // w2: [ffnDim, hiddenSize] down
	FFNUp    *core.QuantizedTensor // w3: [hiddenSize, ffnDim] up
	FFNNorm1 []float32             // [hiddenSize]
	FFNNorm2 []float32             // [hiddenSize]

	// AdaLN modulation (nil for context_refiner layers)
	AdaLNWeight *core.QuantizedTensor // [adaLNEmbedDim, 4*hiddenSize]
	AdaLNBias   []float32             // [4*hiddenSize]
}

// DiTModel holds the complete Z-Image DiT model weights.
type DiTModel struct {
	Config ZImageConfig

	// Patch embedder
	XEmbedWeight *core.QuantizedTensor // [patchDim, hiddenSize]
	XEmbedBias   []float32             // [hiddenSize]
	XPadToken    []float32             // [hiddenSize]

	// Timestep embedder: sinusoidal → MLP(256→1024→256)
	TEmbedMLP0Weight *core.QuantizedTensor // [256, 1024]
	TEmbedMLP0Bias   []float32             // [1024]
	TEmbedMLP2Weight *core.QuantizedTensor // [1024, 256]
	TEmbedMLP2Bias   []float32             // [256]

	// Caption embedder: RMSNorm → Linear(capFeatDim→hiddenSize)
	CapEmbedNormWeight []float32             // [capFeatDim]
	CapEmbedLinWeight  *core.QuantizedTensor // [capFeatDim, hiddenSize]
	CapEmbedLinBias    []float32             // [hiddenSize]
	CapPadToken        []float32             // [hiddenSize]

	// Layers
	ContextRefiner []DiTLayer // 2 layers, no adaLN
	NoiseRefiner   []DiTLayer // 2 layers, with adaLN
	MainLayers     []DiTLayer // 30 layers, with adaLN

	// Final layer
	FinalAdaLNWeight *core.QuantizedTensor // [adaLNEmbedDim, hiddenSize]
	FinalAdaLNBias   []float32             // [hiddenSize]
	FinalLinWeight   *core.QuantizedTensor // [hiddenSize, patchDim]
	FinalLinBias     []float32             // [patchDim]

	// Backing mmap file
	MmapFile *mmap.MappedFile
}

// LoadDiTModel loads a Z-Image DiT model from a GGUF file.
func LoadDiTModel(path string) (*DiTModel, error) {
	gf, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open GGUF: %w", err)
	}

	arch, _ := gf.Metadata["general.architecture"].(string)
	if arch != "lumina2" {
		log.Printf("Warning: expected architecture 'lumina2', got %q", arch)
	}

	mf, err := mmap.Open(path)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}

	cfg := DefaultZImageConfig()
	m := &DiTModel{
		Config:   cfg,
		MmapFile: mf,
	}

	// Build tensor map: name → TensorInfo
	tensorMap := make(map[string]gguf.TensorInfo, len(gf.Tensors))
	for _, t := range gf.Tensors {
		tensorMap[t.Name] = t
	}

	getQT := func(name string) *core.QuantizedTensor {
		t, ok := tensorMap[name]
		if !ok {
			log.Printf("Warning: tensor %q not found", name)
			return nil
		}
		rows := int(t.Dimensions[len(t.Dimensions)-1])
		cols := 1
		for i := 0; i < len(t.Dimensions)-1; i++ {
			cols *= int(t.Dimensions[i])
		}
		dataSize := quant.BytesForN(uint32(t.Type), rows*cols)
		offset := gf.DataOffset + int64(t.Offset)
		data := mf.Data[offset : offset+int64(dataSize)]
		qt, err := core.NewQuantizedTensor(data, uint32(t.Type), rows, cols)
		if err != nil {
			log.Printf("Warning: tensor %q: %v", name, err)
			return nil
		}
		return qt
	}

	getF32 := func(name string) []float32 {
		t, ok := tensorMap[name]
		if !ok {
			log.Printf("Warning: tensor %q not found", name)
			return nil
		}
		numel := 1
		for _, d := range t.Dimensions {
			numel *= int(d)
		}
		dataSize := quant.BytesForN(uint32(t.Type), numel)
		offset := gf.DataOffset + int64(t.Offset)
		data := mf.Data[offset : offset+int64(dataSize)]
		out := make([]float32, numel)
		quant.DequantizeInto(out, data, uint32(t.Type), numel)
		return out
	}

	// Patch embedder
	m.XEmbedWeight = getQT("x_embedder.weight")
	m.XEmbedBias = getF32("x_embedder.bias")
	m.XPadToken = getF32("x_pad_token")

	// Timestep embedder
	m.TEmbedMLP0Weight = getQT("t_embedder.mlp.0.weight")
	m.TEmbedMLP0Bias = getF32("t_embedder.mlp.0.bias")
	m.TEmbedMLP2Weight = getQT("t_embedder.mlp.2.weight")
	m.TEmbedMLP2Bias = getF32("t_embedder.mlp.2.bias")

	// Caption embedder
	m.CapEmbedNormWeight = getF32("cap_embedder.0.weight")
	m.CapEmbedLinWeight = getQT("cap_embedder.1.weight")
	m.CapEmbedLinBias = getF32("cap_embedder.1.bias")
	m.CapPadToken = getF32("cap_pad_token")

	// Load layer function
	loadLayer := func(prefix string, hasAdaLN bool) DiTLayer {
		l := DiTLayer{
			AttnQKV:   getQT(prefix + ".attention.qkv.weight"),
			AttnOut:   getQT(prefix + ".attention.out.weight"),
			QNorm:     getF32(prefix + ".attention.q_norm.weight"),
			KNorm:     getF32(prefix + ".attention.k_norm.weight"),
			AttnNorm1: getF32(prefix + ".attention_norm1.weight"),
			AttnNorm2: getF32(prefix + ".attention_norm2.weight"),
			FFNGate:   getQT(prefix + ".feed_forward.w1.weight"),
			FFNDown:   getQT(prefix + ".feed_forward.w2.weight"),
			FFNUp:     getQT(prefix + ".feed_forward.w3.weight"),
			FFNNorm1:  getF32(prefix + ".ffn_norm1.weight"),
			FFNNorm2:  getF32(prefix + ".ffn_norm2.weight"),
		}
		if hasAdaLN {
			l.AdaLNWeight = getQT(prefix + ".adaLN_modulation.0.weight")
			l.AdaLNBias = getF32(prefix + ".adaLN_modulation.0.bias")
		}
		return l
	}

	// Context refiner layers (no adaLN)
	m.ContextRefiner = make([]DiTLayer, cfg.NumRefinerLayers)
	for i := 0; i < cfg.NumRefinerLayers; i++ {
		m.ContextRefiner[i] = loadLayer("context_refiner."+strconv.Itoa(i), false)
	}

	// Noise refiner layers (with adaLN)
	m.NoiseRefiner = make([]DiTLayer, cfg.NumRefinerLayers)
	for i := 0; i < cfg.NumRefinerLayers; i++ {
		m.NoiseRefiner[i] = loadLayer("noise_refiner."+strconv.Itoa(i), true)
	}

	// Main layers (with adaLN)
	m.MainLayers = make([]DiTLayer, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		m.MainLayers[i] = loadLayer("layers."+strconv.Itoa(i), true)
	}

	// Final layer
	m.FinalAdaLNWeight = getQT("final_layer.adaLN_modulation.1.weight")
	m.FinalAdaLNBias = getF32("final_layer.adaLN_modulation.1.bias")
	m.FinalLinWeight = getQT("final_layer.linear.weight")
	m.FinalLinBias = getF32("final_layer.linear.bias")

	// Verify critical tensors loaded
	missing := []string{}
	if m.XEmbedWeight == nil {
		missing = append(missing, "x_embedder.weight")
	}
	if m.TEmbedMLP0Weight == nil {
		missing = append(missing, "t_embedder.mlp.0.weight")
	}
	if len(m.MainLayers) > 0 && m.MainLayers[0].AttnQKV == nil {
		missing = append(missing, "layers.0.attention.qkv.weight")
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("missing critical tensors: %s", strings.Join(missing, ", "))
	}

	log.Printf("Loaded Z-Image DiT: %d context_refiner + %d noise_refiner + %d main layers",
		len(m.ContextRefiner), len(m.NoiseRefiner), len(m.MainLayers))

	return m, nil
}

// PreDequantizeAll dequantizes all weights to float32 for faster batch GEMM.
// Uses more RAM but avoids repeated dequantization during inference.
func (m *DiTModel) PreDequantizeAll() {
	pool := blas.DefaultPool()
	_ = pool
	dq := func(qt *core.QuantizedTensor) {
		if qt != nil {
			blas.PreDequantize(qt)
		}
	}
	dq(m.XEmbedWeight)
	dq(m.TEmbedMLP0Weight)
	dq(m.TEmbedMLP2Weight)
	dq(m.CapEmbedLinWeight)
	for i := range m.ContextRefiner {
		l := &m.ContextRefiner[i]
		dq(l.AttnQKV)
		dq(l.AttnOut)
		dq(l.FFNGate)
		dq(l.FFNDown)
		dq(l.FFNUp)
	}
	for i := range m.NoiseRefiner {
		l := &m.NoiseRefiner[i]
		dq(l.AttnQKV)
		dq(l.AttnOut)
		dq(l.FFNGate)
		dq(l.FFNDown)
		dq(l.FFNUp)
		dq(l.AdaLNWeight)
	}
	for i := range m.MainLayers {
		l := &m.MainLayers[i]
		dq(l.AttnQKV)
		dq(l.AttnOut)
		dq(l.FFNGate)
		dq(l.FFNDown)
		dq(l.FFNUp)
		dq(l.AdaLNWeight)
	}
	dq(m.FinalAdaLNWeight)
	dq(m.FinalLinWeight)
}
