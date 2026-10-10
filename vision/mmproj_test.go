package vision

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/computerex/dlgo/format/gguf"
)

// ---------------------------------------------------------------------------
// Synthetic CLIP-style mmproj builder
// ---------------------------------------------------------------------------

type tinyCfg struct {
	embd, heads, layers, ffn, patch, image int
	class                                  bool
	preLN                                  bool
	projectorOut                           int
}

func (c tinyCfg) gridSize() int { return c.image / c.patch }

// buildTinyMmproj writes a CLIP-shaped mmproj GGUF and returns the path plus
// the weight matrices (so the test can compute an independent reference).
func buildTinyMmproj(t *testing.T, cfg tinyCfg, seed int64) (string, map[string][]float32) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	W := map[string][]float32{}
	randMat := func(name string, rows, cols int) []float32 {
		v := make([]float32, rows*cols)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * 0.1
		}
		W[name] = v
		return v
	}
	randVec := func(name string, n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * 0.1
		}
		W[name] = v
		return v
	}

	md := map[string]interface{}{
		"general.architecture":                        "clip",
		"clip.has_vision_encoder":                     true,
		"clip.projector_type":                         "mlp",
		"clip.vision.image_size":                      uint32(cfg.image),
		"clip.vision.patch_size":                      uint32(cfg.patch),
		"clip.vision.embedding_length":                uint32(cfg.embd),
		"clip.vision.attention.head_count":            uint32(cfg.heads),
		"clip.vision.block_count":                     uint32(cfg.layers),
		"clip.vision.feed_forward_length":             uint32(cfg.ffn),
		"clip.vision.attention.layer_norm_epsilon":    float32(1e-5),
		"clip.vision.image_mean":                      []float32{0.5, 0.5, 0.5},
		"clip.vision.image_std":                       []float32{0.5, 0.5, 0.5},
	}

	var tensors []gguf.WriteTensor
	add := func(name string, dims []int64, data []float32) {
		tensors = append(tensors, gguf.WriteTensor{
			Name: name, Dimensions: dims, Type: gguf.GGMLTypeF32, Data: gguf.Float32Bytes(data),
		})
	}

	// Patch embedding in modern 4D GGUF order [KH, KW, IC, OC].
	// Values are laid out in ggml memory order: idx = kh + KW*(kw + IC*(ic + 3*oc))
	ps := cfg.patch
	patch := make([]float32, ps*ps*3*cfg.embd)
	for i := range patch {
		patch[i] = float32(rng.NormFloat64()) * 0.1
	}
	W["patch"] = patch
	add("v.patch_embd.weight", []int64{int64(ps), int64(ps), 3, int64(cfg.embd)}, patch)

	nPos := cfg.gridSize()*cfg.gridSize() + 1 // +1 always (class slot) in this fixture
	pos := randMat("pos", nPos, cfg.embd)
	add("v.position_embd.weight", []int64{int64(nPos), int64(cfg.embd)}, pos)

	if cfg.class {
		add("v.class_embd", []int64{int64(cfg.embd)}, randVec("class", cfg.embd))
	}
	if cfg.preLN {
		add("v.pre_ln.weight", []int64{int64(cfg.embd)}, randVec("pre_ln_w", cfg.embd))
		add("v.pre_ln.bias", []int64{int64(cfg.embd)}, randVec("pre_ln_b", cfg.embd))
	}
	add("v.post_ln.weight", []int64{int64(cfg.embd)}, randVec("post_ln_w", cfg.embd))
	add("v.post_ln.bias", []int64{int64(cfg.embd)}, randVec("post_ln_b", cfg.embd))

	for i := 0; i < cfg.layers; i++ {
		p := fmt.Sprintf("v.blk.%d.", i)
		q := fmt.Sprintf("blk%d_", i)
		add(p+"ln1.weight", []int64{int64(cfg.embd)}, randVec(q+"ln1w", cfg.embd))
		add(p+"ln1.bias", []int64{int64(cfg.embd)}, randVec(q+"ln1b", cfg.embd))
		add(p+"ln2.weight", []int64{int64(cfg.embd)}, randVec(q+"ln2w", cfg.embd))
		add(p+"ln2.bias", []int64{int64(cfg.embd)}, randVec(q+"ln2b", cfg.embd))
		for _, proj := range []string{"attn_q", "attn_k", "attn_v", "attn_out"} {
			add(p+proj+".weight", []int64{int64(cfg.embd), int64(cfg.embd)}, randMat(q+proj+"w", cfg.embd, cfg.embd))
		} 		// no biases for this fixture (SigLIP-like)
		add(p+"ffn_up.weight", []int64{int64(cfg.embd), int64(cfg.ffn)}, randMat(q+"ffnup", cfg.ffn, cfg.embd))
		add(p+"ffn_down.weight", []int64{int64(cfg.ffn), int64(cfg.embd)}, randMat(q+"ffndown", cfg.embd, cfg.ffn))
	}

	// Projector: mm.0 (embd -> projHidden) -> GELU -> mm.2 (projHidden -> out)
	projHidden := cfg.projectorOut
	add("mm.0.weight", []int64{int64(cfg.embd), int64(projHidden)}, randMat("mm0", projHidden, cfg.embd))
	add("mm.0.bias", []int64{int64(projHidden)}, randVec("mm0b", projHidden))
	add("mm.2.weight", []int64{int64(projHidden), int64(cfg.projectorOut)}, randMat("mm2", cfg.projectorOut, projHidden))
	add("mm.2.bias", []int64{int64(cfg.projectorOut)}, randVec("mm2b", cfg.projectorOut))

	path := filepath.Join(t.TempDir(), "mmproj.gguf")
	if err := gguf.WriteGGUF(path, md, tensors, 32); err != nil {
		t.Fatalf("WriteGGUF: %v", err)
	}
	return path, W
}

// ---------------------------------------------------------------------------
// Independent naive reference for the CLIP forward
// ---------------------------------------------------------------------------

type ref struct {
	cfg tinyCfg
	W   map[string][]float32
}

func (r ref) lin(name string, x []float32, rows, cols int) []float32 {
	w := r.W[name]
	out := make([]float32, rows)
	for i := 0; i < rows; i++ {
		var s float32
		for j := 0; j < cols; j++ {
			s += w[i*cols+j] * x[j]
		}
		out[i] = s
	}
	return out
}

func (r ref) ln(x []float32, wName, bName string) []float32 {
	n := len(x)
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(n)
	var varr float64
	for _, v := range x {
		d := float64(v) - mean
		varr += d * d
	}
	varr /= float64(n)
	inv := 1 / math.Sqrt(varr+1e-5)
	out := make([]float32, n)
	for i := range x {
		v := (x[i] - float32(mean)) * float32(inv)
		if wName != "" {
			v *= r.W[wName][i]
		}
		if bName != "" {
			v += r.W[bName][i]
		}
		out[i] = v
	}
	return out
}

// Encode reproduces the expected pipeline using plain loops over the same
// weights, with the patch gather done in a straightforward nested-loop order.
func (r ref) Encode(imgPix []float32, W, H int) [][]float32 {
	c := r.cfg
	ps := c.patch
	grid := c.gridSize()

	// 1. patch features, (C,Y,X) order per patch
	rows := make([][]float32, 0, grid*grid)
	for py := 0; py < grid; py++ {
		for px := 0; px < grid; px++ {
			f := make([]float32, 3*ps*ps)
			for ch := 0; ch < 3; ch++ {
				for y := 0; y < ps; y++ {
					for x := 0; x < ps; x++ {
						sx, sy := px*ps+x, py*ps+y
						f[ch*ps*ps+y*ps+x] = (imgPix[(sy*W+sx)*3+ch] - 0.5) / 0.5
					}
				}
			}
			rows = append(rows, f)
		}
	}

	// 2. patch linear: out[oc] = sum_{y,x,c} W[(y*KW+x)*IC+c][oc] * f[c*ps*ps+y*ps+x]
	pw := r.W["patch"]
	toks := make([][]float32, len(rows))
	for i, f := range rows {
		v := make([]float32, c.embd)
		for y := 0; y < ps; y++ {
			for x := 0; x < ps; x++ {
				for ch := 0; ch < 3; ch++ {
					wRow := (y*ps + x) * 3 + ch
					fv := f[ch*ps*ps+y*ps+x]
					for oc := 0; oc < c.embd; oc++ {
						v[oc] += pw[wRow+(ps*ps*3)*oc] * fv
					}
				}
			}
		}
		toks[i] = v
	}

	// 3. class + position
	all := toks
	if c.class {
		cls := make([]float32, c.embd)
		copy(cls, r.W["class"])
		all = append([][]float32{cls}, toks...)
	}
	for i, tk := range all {
		pe := r.W["pos"][i*c.embd : (i+1)*c.embd]
		for d := range tk {
			tk[d] += pe[d]
		}
	}

	if c.preLN {
		for _, tk := range all {
			copy(tk, r.ln(tk, "pre_ln_w", "pre_ln_b"))
		}
	}

	// 4. blocks
	headDim := c.embd / c.heads
	for l := 0; l < c.layers; l++ {
		p := fmt.Sprintf("blk%d_", l)
		n := len(all)
		nrm := make([][]float32, n)
		for i := range all {
			nrm[i] = r.ln(all[i], p+"ln1w", p+"ln1b")
		}
		Q := make([][]float32, n)
		K := make([][]float32, n)
		V := make([][]float32, n)
		for i := range nrm {
			Q[i] = r.lin(p+"attn_qw", nrm[i], c.embd, c.embd)
			K[i] = r.lin(p+"attn_kw", nrm[i], c.embd, c.embd)
			V[i] = r.lin(p+"attn_vw", nrm[i], c.embd, c.embd)
		}
		sc := float32(1 / math.Sqrt(float64(headDim)))
		for i := range all {
			acc := make([]float32, c.embd)
			for h := 0; h < c.heads; h++ {
				scores := make([]float32, n)
				mx := float32(-1e30)
				for j := 0; j < n; j++ {
					var s float32
					for t := 0; t < headDim; t++ {
						s += Q[i][h*headDim+t] * K[j][h*headDim+t]
					}
					s *= sc
					scores[j] = s
					if s > mx {
						mx = s
					}
				}
				var sum float32
				for j := range scores {
					scores[j] = float32(math.Exp(float64(scores[j] - mx)))
					sum += scores[j]
				}
				for j := 0; j < n; j++ {
					w := scores[j] / sum
					for t := 0; t < headDim; t++ {
						acc[h*headDim+t] += w * V[j][h*headDim+t]
					}
				}
			}
			proj := r.lin(p+"attn_outw", acc, c.embd, c.embd)
			for d := 0; d < c.embd; d++ {
				all[i][d] += proj[d]
			}
			// FFN
			n2 := r.ln(all[i], p+"ln2w", p+"ln2b")
			h := r.lin(p+"ffnup", n2, c.ffn, c.embd)
			for j := range h {
				h[j] = 0.5 * h[j] * (1 + float32(math.Erf(float64(h[j])/math.Sqrt2)))
			}
			down := r.lin(p+"ffndown", h, c.embd, c.ffn)
			for d := 0; d < c.embd; d++ {
				all[i][d] += down[d]
			}
		}
	}

	// 5. post LN, drop class
	out := make([][]float32, 0, len(all))
	for _, tk := range all {
		copy(tk, r.ln(tk, "post_ln_w", "post_ln_b"))
	}
	if c.class {
		out = append(out, all[1:]...)
	} else {
		out = append(out, all...)
	}

	// 6. projector
	res := make([][]float32, len(out))
	for i, tk := range out {
		h := r.lin("mm0", tk, c.projectorOut, c.embd)
		for j := range h {
			h[j] += r.W["mm0b"][j]
			h[j] = 0.5 * h[j] * (1 + float32(math.Erf(float64(h[j])/math.Sqrt2)))
		}
		o := r.lin("mm2", h, c.projectorOut, c.projectorOut)
		for j := range o {
			o[j] += r.W["mm2b"][j]
		}
		res[i] = o
	}
	return res
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

func solidImage(v uint8, n int) (image.Image, []float32) {
	img := image.NewNRGBA(image.Rect(0, 0, n, n))
	pix := make([]float32, n*n*3)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			img.Set(x, y, color.NRGBA{v, v, v, 255})
			f := float32(v) / 255
			pix[(y*n+x)*3+0] = f
			pix[(y*n+x)*3+1] = f
			pix[(y*n+x)*3+2] = f
		}
	}
	return img, pix
}

func TestEncodeMatchesNaiveReference(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  tinyCfg
	}{
		{"class_no_preln", tinyCfg{embd: 8, heads: 2, layers: 2, ffn: 16, patch: 2, image: 8, class: true, projectorOut: 12}},
		{"noclass_preln", tinyCfg{embd: 8, heads: 2, layers: 1, ffn: 16, patch: 4, image: 8, class: false, preLN: true, projectorOut: 8}},
		{"class_preln_4layer", tinyCfg{embd: 12, heads: 3, layers: 3, ffn: 24, patch: 2, image: 8, class: true, preLN: true, projectorOut: 16}},
		{"attn_only_1layer", tinyCfg{embd: 8, heads: 2, layers: 1, ffn: 16, patch: 2, image: 8, class: true, projectorOut: 12}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path, W := buildTinyMmproj(t, tc.cfg, 7)
			enc, err := LoadMmproj(path)
			if err != nil {
				t.Fatalf("LoadMmproj: %v", err)
			}
			defer enc.Close()

			img, pix := solidImage(200, tc.cfg.image)
			got, err := enc.Encode(img)
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}
			want := ref{cfg: tc.cfg, W: W}.Encode(pix, tc.cfg.image, tc.cfg.image)

			grid := tc.cfg.gridSize()
			if len(got) != grid*grid {
				t.Fatalf("token count = %d, want %d", len(got), grid*grid)
			}
			if len(want) != len(got) {
				t.Fatalf("reference token count = %d, want %d", len(want), len(got))
			}
			var maxAbs, refL2 float64
			for i := range got {
				for j := range got[i] {
					d := math.Abs(float64(got[i][j] - want[i][j]))
					if d > maxAbs {
						maxAbs = d
					}
					refL2 += float64(want[i][j]) * float64(want[i][j])
				}
			}
			rel := maxAbs / math.Sqrt(refL2/float64(len(got)*len(got[0])))
			if rel > 1e-4 {
				t.Fatalf("relative max deviation %.3e exceeds 1e-4 (maxAbs=%.3e)", rel, maxAbs)
			}
			t.Logf("tokens=%d dim=%d maxAbs=%.3e relL2=%.3e", len(got), len(got[0]), maxAbs, rel)
		})
	}
}

func TestEncodeDeterministic(t *testing.T) {
	cfg := tinyCfg{embd: 8, heads: 2, layers: 2, ffn: 16, patch: 2, image: 8, class: true, projectorOut: 12}
	path, _ := buildTinyMmproj(t, cfg, 11)
	enc, err := LoadMmproj(path)
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()

	img, _ := solidImage(90, cfg.image)
	a, err := enc.Encode(img)
	if err != nil {
		t.Fatal(err)
	}
	b, err := enc.Encode(img)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				t.Fatalf("non-deterministic at [%d][%d]: %v vs %v", i, j, a[i][j], b[i][j])
			}
		}
	}
}

// TestImageNormStatsFromArray verifies that mean/std arrays survive the GGUF
// round-trip (read back as []interface{}) and are not silently replaced by the
// CLIP defaults — a bug that made every encoder output wrong by ~2x.
func TestImageNormStatsFromArray(t *testing.T) {
	mean, std := imageNormStats(map[string]interface{}{
		"clip.vision.image_mean": []interface{}{0.5, 0.5, 0.5},
		"clip.vision.image_std":  []interface{}{0.5, 0.5, 0.5},
	})
	for i := 0; i < 3; i++ {
		if mean[i] != 0.5 || std[i] != 0.5 {
			t.Fatalf("mean/std not parsed from []interface{}: mean=%v std=%v", mean, std)
		}
	}
}

func TestEncodeRejectsUnsupportedProjector(t *testing.T) {
	cfg := tinyCfg{embd: 8, heads: 2, layers: 1, ffn: 16, patch: 2, image: 8, projectorOut: 8}
	// rebuild fixture then patch metadata is awkward; instead write a minimal
	// file with a qwen2vl projector type.
	md := map[string]interface{}{
		"general.architecture":             "clip",
		"clip.has_vision_encoder":          true,
		"clip.projector_type":              "qwen2vl",
		"clip.vision.image_size":           uint32(cfg.image),
		"clip.vision.patch_size":           uint32(cfg.patch),
		"clip.vision.embedding_length":     uint32(cfg.embd),
		"clip.vision.attention.head_count": uint32(cfg.heads),
		"clip.vision.block_count":          uint32(cfg.layers),
		"clip.vision.feed_forward_length":  uint32(cfg.ffn),
	}
	path := filepath.Join(t.TempDir(), "bad.gguf")
	if err := gguf.WriteGGUF(path, md, nil, 32); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadMmproj(path); err == nil {
		t.Fatal("expected error for unsupported projector type")
	}
}

func TestResizeBilinearAntialias(t *testing.T) {
	// 4x4 bright/black checkerboard downscaled to 1x1 must average to ~0.5.
	src := &f32Image{W: 4, H: 4, Pix: make([]float32, 4*4*3)}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			v := float32(0)
			if (x+y)%2 == 0 {
				v = 1
			}
			off := (y*4 + x) * 3
			src.Pix[off], src.Pix[off+1], src.Pix[off+2] = v, v, v
		}
	}
	out := resizeBilinear(src, 1, 1)
	for c := 0; c < 3; c++ {
		if math.Abs(float64(out.Pix[c]-0.5)) > 1e-6 {
			t.Fatalf("channel %d = %v, want 0.5", c, out.Pix[c])
		}
	}
}
