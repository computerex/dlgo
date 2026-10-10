// Package vision implements image tokenization for multimodal models: loading
// a llama.cpp-format mmproj GGUF (CLIP-style vision transformer + projector),
// preprocessing images, and producing per-patch embedding vectors that can be
// spliced into the LLM's token stream at placeholder positions.
//
// Supported v1 encoders: CLIP ViT and SigLIP ViT (identical tensor naming in
// the mmproj format; differences handled via optional tensors: class
// embedding, pre-LN, biases) with LLaVA MLP projectors ("mlp" and "mlp_norm"
// projector types, including Yi variants).
package vision

import (
	"fmt"
	"strings"

	"github.com/computerex/dlgo/blas"
	"github.com/computerex/dlgo/format/gguf"
	"github.com/computerex/dlgo/mmap"
	"github.com/computerex/dlgo/quant"
)

// Config holds the vision encoder hyperparameters parsed from mmproj metadata.
type Config struct {
	ProjectorType string // "mlp", "mlp_norm"
	ImageSize     int    // e.g. 336
	PatchSize     int    // e.g. 14
	EmbeddingDim  int    // vision hidden size (e.g. 1024)
	ProjectionDim int    // CLIP projection dim (0 = unused)
	NumHeads      int    // attention heads
	NumLayers     int    // transformer blocks
	FFNDim        int    // feed-forward size
	LayerNormEps  float32
	ImageMean     [3]float32
	ImageStd      [3]float32
	MergeType     string // "flat" or "spatial_unpad"
	HasClassEmbed bool
	HasPreLN      bool
}

// Linear is a dense layer: out[i] = sum_j W[i*Cols+j]*x[j] + b[i].
type Linear struct {
	W    []float32 // Rows x Cols, row-major
	B    []float32 // Rows (may be nil)
	Rows int
	Cols int
}

// Layer holds one transformer block's weights.
type Layer struct {
	AttnQ, AttnK, AttnV, AttnOut Linear
	FFNUp, FFNDown               Linear
	LN1W, LN1B                   []float32 // pre-attention layer norm
	LN2W, LN2B                   []float32 // pre-FFN layer norm
}

// Encoder is a loaded vision encoder ready to tokenize images.
type Encoder struct {
	Config Config
	mf     *mmap.MappedFile

	PatchEmbed Linear    // 2D flattened layout (legacy): rows = [OC], cols = [IC*KH*KW], C-fastest
	Patch4DW   []float32 // non-nil when 4D conv layout: GGUF order [KH, KW, IC, OC]
	PatchKH    int
	PatchKW    int
	PatchIC    int
	PatchBias  []float32 // optional

	PosEmbed []float32 // [n_pos, EmbeddingDim]
	ClassEmb []float32 // optional
	PreLNW   []float32 // optional
	PreLNB   []float32
	PostLNW  []float32
	PostLNB  []float32
	Layers   []Layer

	MM0, MM2     Linear    // standard LLaVA projector: mm.0 -> GELU -> mm.2
	MM1, MM3     Linear    // Yi variant: mm.0/mm.1 -> norm -> mm.3
	ImageNewline []float32 // optional (LLaVA-1.6 spatial unpad)
	YiNorm       bool      // mm.1 present => Yi-style projector

	// Content extent (in pixels) of the last spatial-unpad encode; used to
	// trim padded patch rows/cols deterministically.
	lastContentW, lastContentH int

	// pool parallelizes the vision transformer across CPU cores.
	pool *blas.Pool
}

// LoadMmproj loads a llama.cpp mmproj GGUF vision encoder from path.
// All weights are dequantized to float32 at load time (mmproj files are
// typically F16/F32; typical encoders are 100-600M params).
func LoadMmproj(path string) (*Encoder, error) {
	gf, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("vision: parse GGUF: %w", err)
	}
	md := gf.Metadata

	if hasV, ok := md["clip.has_vision_encoder"]; ok && !toBool(hasV) {
		return nil, fmt.Errorf("vision: %s has no vision encoder", path)
	}

	enc := &Encoder{pool: blas.DefaultPool()}
	cfg := &enc.Config

	cfg.ProjectorType = toString(md, "clip.projector_type")
	if cfg.ProjectorType == "" {
		cfg.ProjectorType = toString(md, "clip.vision.projector_type")
	}
	switch cfg.ProjectorType {
	case "mlp", "mlp_norm", "":
		if cfg.ProjectorType == "" {
			cfg.ProjectorType = "mlp" // legacy files omit the key
		}
	default:
		return nil, fmt.Errorf("vision: unsupported projector type %q (v1 supports mlp, mlp_norm)", cfg.ProjectorType)
	}

	cfg.ImageSize = toInt(md, "clip.vision.image_size")
	cfg.PatchSize = toInt(md, "clip.vision.patch_size")
	if cfg.ImageSize <= 0 || cfg.PatchSize <= 0 {
		return nil, fmt.Errorf("vision: missing clip.vision.image_size/patch_size")
	}
	cfg.EmbeddingDim = toInt(md, "clip.vision.embedding_length")
	cfg.NumHeads = toInt(md, "clip.vision.attention.head_count")
	cfg.NumLayers = toInt(md, "clip.vision.block_count")
	cfg.FFNDim = toInt(md, "clip.vision.feed_forward_length")
	cfg.LayerNormEps = float32(toFloat(md, "clip.vision.attention.layer_norm_epsilon"))
	if cfg.LayerNormEps <= 0 {
		cfg.LayerNormEps = 1e-5
	}
	if cfg.EmbeddingDim <= 0 || cfg.NumHeads <= 0 || cfg.NumLayers <= 0 || cfg.FFNDim <= 0 {
		return nil, fmt.Errorf("vision: missing core clip.vision.* hyperparameters")
	}
	cfg.ProjectionDim = toInt(md, "clip.vision.projection_dim")

	mean, std := imageNormStats(md)
	cfg.ImageMean = mean
	cfg.ImageStd = std

	cfg.MergeType = toString(md, "clip.vision.mm_patch_merge_type")
	if cfg.MergeType == "" {
		cfg.MergeType = "flat"
	}

	tensors := make(map[string]gguf.TensorInfo, len(gf.Tensors))
	for _, ti := range gf.Tensors {
		tensors[ti.Name] = ti
	}

	// mf is used for zero-copy slices into quant.Dequantize.
	mf, err := mmap.Open(path)
	if err != nil {
		return nil, fmt.Errorf("vision: mmap: %w", err)
	}
	enc.mf = mf

	deq := func(name string) ([]float32, bool, error) {
		ti, ok := tensors[name]
		if !ok {
			return nil, false, nil
		}
		total := int64(1)
		for _, d := range ti.Dimensions {
			total *= d
		}
		data := mf.Slice(gf.DataOffset+int64(ti.Offset), int64(quant.BytesForN(uint32(ti.Type), int(total))))
		out, err := quant.Dequantize(data, uint32(ti.Type), int(total))
		if err != nil {
			return nil, true, fmt.Errorf("vision: dequantize %s: %w", name, err)
		}
		return out, true, nil
	}
	get := func(what string, names ...string) ([]float32, error) {
		for _, n := range names {
			out, found, err := deq(n)
			if err != nil {
				return nil, err
			}
			if found {
				return out, nil
			}
		}
		return nil, nil
	}
	getRequired := func(what string, names ...string) ([]float32, error) {
		out, err := get(what, names...)
		if err != nil {
			return nil, err
		}
		if out == nil {
			return nil, fmt.Errorf("vision: missing required tensor %s (tried %v)", what, names)
		}
		return out, nil
	}

	// Patch embedding: modern files are 4D GGUF order [KH, KW, IC, OC]
	// (convert_hf_to_gguf permutes the HF [OC, IC, KH, KW] conv weight);
	// legacy files are 2D [IC*KH*KW, OC] with HF flatten order (C fastest).
	patchNames := []string{"v.patch_embd.weight", "mm.patch_embd.weight"}
	if ti, ok := tensors[firstExisting(tensors, patchNames...)]; ok && len(ti.Dimensions) == 4 {
		enc.PatchKH = int(ti.Dimensions[0])
		enc.PatchKW = int(ti.Dimensions[1])
		enc.PatchIC = int(ti.Dimensions[2])
		w, err := getRequired("patch_embd (4D)", patchNames...)
		if err != nil {
			return nil, err
		}
		enc.Patch4DW = w
		if enc.PatchIC != 3 {
			return nil, fmt.Errorf("vision: patch conv input channels = %d, want 3", enc.PatchIC)
		}
		if enc.PatchKH != cfg.PatchSize || enc.PatchKW != cfg.PatchSize {
			return nil, fmt.Errorf("vision: patch conv kernel %dx%d != patch_size %d",
				enc.PatchKH, enc.PatchKW, cfg.PatchSize)
		}
	} else {
		w, err := getRequired("patch_embd (2D)", patchNames...)
		if err != nil {
			return nil, err
		}
		feat := 3 * cfg.PatchSize * cfg.PatchSize
		if len(w) != feat*cfg.EmbeddingDim {
			return nil, fmt.Errorf("vision: patch embd weight size %d, want %d x %d", len(w), feat, cfg.EmbeddingDim)
		}
		enc.PatchEmbed = Linear{W: w, Rows: cfg.EmbeddingDim, Cols: feat}
	}
	enc.PatchBias, _ = get("patch_embd bias", "v.patch_embd.bias", "mm.patch_embd.bias")

	pos, err := getRequired("position_embd", "v.position_embd.weight", "mm.position_embd.weight")
	if err != nil {
		return nil, err
	}
	enc.PosEmbed = pos

	enc.ClassEmb, _ = get("class_embd", "v.class_embd", "mm.class_embd")
	cfg.HasClassEmbed = enc.ClassEmb != nil

	enc.PreLNW, _ = get("pre_ln", "v.pre_ln.weight", "mm.pre_ln.weight")
	enc.PreLNB, _ = get("pre_ln bias", "v.pre_ln.bias", "mm.pre_ln.bias")
	cfg.HasPreLN = enc.PreLNW != nil

	dim := cfg.EmbeddingDim

	enc.PostLNW, _ = get("post_ln", "v.post_ln.weight", "mm.model_norm.weight", "mm.post_ln.weight")
	if enc.PostLNW == nil {
		// LLaVA-1.5 CLIP has no separate post-LN; the pre_ln acts as the input
		// norm and the encoder output is normalized by the projector path. Use
		// an identity-friendly default: no post-LN scaling/bias.
		enc.PostLNW = make([]float32, dim)
		for i := range enc.PostLNW {
			enc.PostLNW[i] = 1
		}
		enc.PostLNB = make([]float32, dim)
	}
	enc.PostLNB, _ = get("post_ln bias", "v.post_ln.bias", "mm.model_norm.bias", "mm.post_ln.bias")
	if enc.PostLNB == nil {
		enc.PostLNB = make([]float32, dim)
	}

	// Transformer blocks.
	enc.Layers = make([]Layer, cfg.NumLayers)
	for i := range enc.Layers {
		l := &enc.Layers[i]
		pfx := fmt.Sprintf("v.blk.%d.", i)
		lpfx := fmt.Sprintf("blk.%d.", i)
		readLinDims := func(what, w1, w2 string) (Linear, bool, error) {
			name := firstExisting(tensors, w1, w2)
			if name == "" {
				return Linear{}, false, nil
			}
			ti := tensors[name]
			rows, cols := inferDimensions(ti.Dimensions)
			w, err := getRequired(what, w1, w2)
			if err != nil {
				return Linear{}, false, err
			}
			if len(w) != rows*cols {
				return Linear{}, false, fmt.Errorf("vision: %s size %d, want %dx%d", what, len(w), rows, cols)
			}
			lin := Linear{W: w, Rows: rows, Cols: cols}
			lin.B, _ = get(what+" bias", strings.TrimSuffix(w1, ".weight")+".bias", strings.TrimSuffix(w2, ".weight")+".bias")
			return lin, true, nil
		}
		readLin := func(what, w1, w2 string, rows, cols int) (Linear, error) {
			w, err := getRequired(what, w1, w2)
			if err != nil {
				return Linear{}, err
			}
			if len(w) != rows*cols {
				return Linear{}, fmt.Errorf("vision: %s size %d, want %dx%d", what, len(w), rows, cols)
			}
			lin := Linear{W: w, Rows: rows, Cols: cols}
			lin.B, _ = get(what+" bias", strings.TrimSuffix(w1, ".weight")+".bias", strings.TrimSuffix(w2, ".weight")+".bias")
			return lin, nil
		}
		if l.AttnQ, err = readLin(fmt.Sprintf("blk%d attn_q", i), pfx+"attn_q.weight", lpfx+"attn_q.weight", dim, dim); err != nil {
			return nil, err
		}
		if l.AttnK, err = readLin(fmt.Sprintf("blk%d attn_k", i), pfx+"attn_k.weight", lpfx+"attn_k.weight", dim, dim); err != nil {
			return nil, err
		}
		if l.AttnV, err = readLin(fmt.Sprintf("blk%d attn_v", i), pfx+"attn_v.weight", lpfx+"attn_v.weight", dim, dim); err != nil {
			return nil, err
		}
		if l.AttnOut, err = readLin(fmt.Sprintf("blk%d attn_out", i), pfx+"attn_out.weight", lpfx+"attn_out.weight", dim, dim); err != nil {
			return nil, err
		}
		// FFN up/down: derive dims from the tensors and identify which maps
		// embd->ffn regardless of the file's naming (llama.cpp swaps by shape).
		up, hasUp, err := readLinDims(fmt.Sprintf("blk%d ffn_up", i), pfx+"ffn_up.weight", lpfx+"ffn_up.weight")
		if err != nil {
			return nil, err
		}
		down, hasDown, err := readLinDims(fmt.Sprintf("blk%d ffn_down", i), pfx+"ffn_down.weight", lpfx+"ffn_down.weight")
		if err != nil {
			return nil, err
		}
		if !hasUp || !hasDown {
			return nil, fmt.Errorf("vision: blk%d missing ffn tensors", i)
		}
		// up should be (rows=ffn, cols=dim); down (rows=dim, cols=ffn).
		if up.Rows == cfg.FFNDim && up.Cols == dim && down.Rows == dim && down.Cols == cfg.FFNDim {
			l.FFNUp, l.FFNDown = up, down
		} else if up.Rows == dim && up.Cols == cfg.FFNDim && down.Rows == cfg.FFNDim && down.Cols == dim {
			// stored swapped
			l.FFNUp, l.FFNDown = down, up
		} else {
			return nil, fmt.Errorf("vision: blk%d unexpected ffn dims up=%dx%d down=%dx%d (dim=%d ffn=%d)",
				i, up.Rows, up.Cols, down.Rows, down.Cols, dim, cfg.FFNDim)
		}
		if l.LN1W, err = getRequired(fmt.Sprintf("blk%d ln1", i), pfx+"ln1.weight", lpfx+"ln1.weight"); err != nil {
			return nil, err
		}
		if l.LN1B, err = get(fmt.Sprintf("blk%d ln1 bias", i), pfx+"ln1.bias", lpfx+"ln1.bias"); err != nil {
			return nil, err
		}
		if l.LN2W, err = getRequired(fmt.Sprintf("blk%d ln2", i), pfx+"ln2.weight", lpfx+"ln2.weight"); err != nil {
			return nil, err
		}
		if l.LN2B, err = get(fmt.Sprintf("blk%d ln2 bias", i), pfx+"ln2.bias", lpfx+"ln2.bias"); err != nil {
			return nil, err
		}
	}

	// Projector: LLaVA MLP. Standard: mm.0 -> GELU -> mm.2 (into LLM dim).
	// Yi variant: mm.0/mm.1 -> LayerNorm -> mm.3.
	readProj := func(idx int) (Linear, bool, error) {
		w, found, err := deq(fmt.Sprintf("mm.%d.weight", idx))
		if err != nil || !found {
			return Linear{}, false, err
		}
		lin := Linear{W: w}
		if b, found, err := deq(fmt.Sprintf("mm.%d.bias", idx)); err == nil && found {
			lin.B = b
		}
		return lin, true, nil
	}
	if e0, ok, err := readProj(0); err != nil {
		return nil, err
	} else if ok {
		e0.Rows, e0.Cols = len(e0.W)/dim, dim
		enc.MM0 = e0
	}
	if e1, ok, err := readProj(1); err != nil {
		return nil, err
	} else if ok {
		// Yi: mm.1 is the second input-dim linear (bias-less).
		e1.Rows, e1.Cols = len(e1.W)/dim, dim
		enc.MM1 = e1
		enc.YiNorm = true
	}
	outIdx := 2
	if e2, ok, err := readProj(2); err != nil {
		return nil, err
	} else if ok {
		inDim := dim
		if enc.MM0.W != nil {
			inDim = enc.MM0.Rows
		}
		e2.Rows, e2.Cols = len(e2.W)/inDim, inDim
		enc.MM2 = e2
	} else {
		outIdx = 3
		if e3, ok, err := readProj(3); err != nil {
			return nil, err
		} else if ok {
			inDim := dim
			if enc.MM1.W != nil {
				inDim = enc.MM1.Rows
			} else if enc.MM0.W != nil {
				inDim = enc.MM0.Rows
			}
			e3.Rows, e3.Cols = len(e3.W)/inDim, inDim
			enc.MM2 = e3 // normalize: output projection lives in MM2
			_ = outIdx
		} else {
			return nil, fmt.Errorf("vision: no mm.N output projector tensor found")
		}
	}
	if enc.MM0.W == nil && enc.MM1.W == nil {
		return nil, fmt.Errorf("vision: no mm.N input projector tensor found")
	}
	enc.ImageNewline, _ = get("image_newline", "v.image_newline", "mm.image_newline")

	if err := enc.validate(); err != nil {
		return nil, err
	}
	return enc, nil
}

// Close releases the underlying mmap.
func (e *Encoder) Close() error {
	if e.mf != nil {
		return e.mf.Close()
	}
	return nil
}

// validate checks the assembled encoder for shape coherence.
func (e *Encoder) validate() error {
	c := &e.Config
	if e.Patch4DW == nil && e.PatchEmbed.W == nil {
		return fmt.Errorf("vision: no patch embedding weights")
	}
	if c.ImageSize%c.PatchSize != 0 {
		return fmt.Errorf("vision: image_size %d not divisible by patch_size %d", c.ImageSize, c.PatchSize)
	}
	nPos := len(e.PosEmbed) / c.EmbeddingDim
	patches := (c.ImageSize / c.PatchSize) * (c.ImageSize / c.PatchSize)
	if nPos < patches {
		return fmt.Errorf("vision: position embeddings %d too small for %d patches", nPos, patches)
	}
	headDim := c.EmbeddingDim / c.NumHeads
	if c.EmbeddingDim%c.NumHeads != 0 || headDim == 0 {
		return fmt.Errorf("vision: embedding_length %d not divisible by head_count %d", c.EmbeddingDim, c.NumHeads)
	}
	return nil
}

// NumImageTokens returns how many embedding rows a native square image
// produces (the exact count varies for spatial_unpad models by aspect ratio).
func (e *Encoder) NumImageTokens() int {
	c := &e.Config
	return (c.ImageSize / c.PatchSize) * (c.ImageSize / c.PatchSize)
}

func imageNormStats(md map[string]interface{}) ([3]float32, [3]float32) {
	// OpenAI CLIP defaults.
	var mean = [3]float32{0.48145466, 0.4578275, 0.40821073}
	var std = [3]float32{0.26862954, 0.26130258, 0.27577711}
	read3 := func(key string, dst *[3]float32) {
		switch arr := md[key].(type) {
		case []float32:
			if len(arr) >= 3 {
				copy(dst[:], arr[:3])
			}
		case []interface{}:
			if len(arr) >= 3 {
				for i := 0; i < 3; i++ {
					dst[i] = toFloat32(arr[i])
				}
			}
		}
	}
	read3("clip.vision.image_mean", &mean)
	read3("clip.vision.image_std", &std)
	return mean, std
}

func toFloat32(v interface{}) float32 {
	switch x := v.(type) {
	case float32:
		return x
	case float64:
		return float32(x)
	case int:
		return float32(x)
	case int32:
		return float32(x)
	case int64:
		return float32(x)
	case uint32:
		return float32(x)
	case uint64:
		return float32(x)
	}
	return 0
}

// inferDimensions maps a GGUF tensor shape to (rows, cols) for a linear layer
// using the ggml convention: ne0 = dims[0] is the fast (input) dimension,
// ne_last is the slow (output) dimension. For a >2D tensor (e.g. a conv
// weight) rows/cols are flattened accordingly.
func inferDimensions(dims []int64) (int, int) {
	if len(dims) == 0 {
		return 1, 1
	}
	if len(dims) == 1 {
		return int(dims[0]), 1
	}
	rows := int(dims[len(dims)-1])
	cols := int(dims[0])
	return rows, cols
}

func firstExisting(tensors map[string]gguf.TensorInfo, names ...string) string {
	for _, n := range names {
		if _, ok := tensors[n]; ok {
			return n
		}
	}
	return ""
}

func toBool(v interface{}) bool {
	switch x := v.(type) {
	case bool:
		return x
	case uint8:
		return x != 0
	case int:
		return x != 0
	}
	return false
}

func toString(md map[string]interface{}, key string) string {
	if s, ok := md[key].(string); ok {
		return strings.TrimSpace(s)
	}
	return ""
}

func toInt(md map[string]interface{}, key string) int {
	switch x := md[key].(type) {
	case uint32:
		return int(x)
	case uint64:
		return int(x)
	case int32:
		return int(x)
	case int64:
		return int(x)
	case int:
		return x
	case float32:
		return int(x)
	case float64:
		return int(x)
	}
	return 0
}

func toFloat(md map[string]interface{}, key string) float64 {
	switch x := md[key].(type) {
	case float32:
		return float64(x)
	case float64:
		return x
	case uint32:
		return float64(x)
	case int32:
		return float64(x)
	}
	return 0
}
