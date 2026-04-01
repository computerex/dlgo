package diffusion

// ZImageConfig holds the architecture parameters for a Z-Image (Lumina2) DiT model.
type ZImageConfig struct {
	PatchSize       int
	HiddenSize      int
	InChannels      int
	OutChannels     int
	NumLayers       int
	NumRefinerLayers int
	HeadDim         int
	NumHeads        int
	NumKVHeads      int
	MultipleOf      int
	FFNDimMult      float32
	NormEps         float32
	QKNorm          bool
	CapFeatDim      int
	Theta           int
	AxesDim         [3]int
	AdaLNEmbedDim   int
	SeqMultiOf      int
}

// DefaultZImageConfig returns the default config for Z-Image-Turbo.
func DefaultZImageConfig() ZImageConfig {
	return ZImageConfig{
		PatchSize:        2,
		HiddenSize:       3840,
		InChannels:       16,
		OutChannels:      16,
		NumLayers:        30,
		NumRefinerLayers: 2,
		HeadDim:          128,
		NumHeads:         30,
		NumKVHeads:       30,
		MultipleOf:       256,
		FFNDimMult:       8.0 / 3.0,
		NormEps:          1e-5,
		QKNorm:           true,
		CapFeatDim:       2560,
		Theta:            256,
		AxesDim:          [3]int{32, 48, 48},
		AdaLNEmbedDim:    256,
		SeqMultiOf:       32,
	}
}

// FFNHiddenDim returns the computed FFN hidden dimension respecting MultipleOf.
func (c *ZImageConfig) FFNHiddenDim() int {
	hidden := int(c.FFNDimMult * float32(c.HiddenSize))
	return c.MultipleOf * ((hidden + c.MultipleOf - 1) / c.MultipleOf)
}
