package vision

import (
	"fmt"
	"image"
	"math"
)

// Encode runs the full pipeline: preprocess -> patch embed -> ViT ->
// projector, returning one embedding row per image token in the LLM's
// embedding space (llmDim = MM2.Rows).
func (e *Encoder) Encode(src image.Image) ([][]float32, error) {
	gridH, gridW, feats, err := e.PrepareImage(src)
	if err != nil {
		return nil, err
	}

	// Patch embedding.
	var tokens [][]float32
	if e.Patch4DW != nil {
		tokens = e.convPatches(feats)
	} else {
		tokens = make([][]float32, len(feats))
		for i, f := range feats {
			tokens[i] = e.PatchEmbed.MulVecBias(f)
		}
	}

	// Class token prepend (always position 0 when present).
	nPos := len(e.PosEmbed) / e.Config.EmbeddingDim
	embd := tokens
	if e.ClassEmb != nil {
		cls := make([]float32, len(e.ClassEmb))
		copy(cls, e.ClassEmb)
		embd = append([][]float32{cls}, tokens...)
	}
	if len(embd) > nPos {
		return nil, fmt.Errorf("vision: %d tokens exceed %d position embeddings", len(embd), nPos)
	}

	// Position embeddings: add per position; for non-square grids (spatial
	// unpad) interpolate the learned square grid bilinearly, skipping the
	// class slot when present.
	learnedOff := 0
	if e.ClassEmb != nil {
		learnedOff = 1
	}
	learnedGrid := nPos - learnedOff
	posGrid := int(math.Round(math.Sqrt(float64(learnedGrid))))
	directGrid := posGrid*posGrid == learnedGrid
	for i, tok := range embd {
		var pe []float32
		switch {
		case e.ClassEmb != nil && i == 0:
			pe = e.PosEmbed[0:e.Config.EmbeddingDim]
		case directGrid && gridH == posGrid && gridW == posGrid:
			pe = e.PosEmbed[i*e.Config.EmbeddingDim : (i+1)*e.Config.EmbeddingDim]
		default:
			pe = e.interpolatePos(i, gridH, gridW, posGrid, learnedOff)
		}
		if pe != nil {
			for d := range tok {
				tok[d] += pe[d]
			}
		}
	}

	// Pre-LN (SigLIP-style encoders).
	if e.PreLNW != nil {
		for _, tok := range embd {
			layerNormInPlace(tok, e.PreLNW, e.PreLNB, e.Config.LayerNormEps)
		}
	}

	// Transformer blocks.
	x := embd
	for li := range e.Layers {
		if err := e.forwardLayer(&e.Layers[li], x); err != nil {
			return nil, fmt.Errorf("vision: block %d: %w", li, err)
		}
	}

	// Post-LN (CLIP).
	for _, tok := range x {
		layerNormInPlace(tok, e.PostLNW, e.PostLNB, e.Config.LayerNormEps)
	}

	// Drop class token output.
	if e.ClassEmb != nil {
		x = x[1:]
	}

	// Spatial unpad: trim padded patch rows/cols and insert newline embeddings
	// at end of each row (llava-1.6), before the projector.
	rowsOut := x
	if e.Config.MergeType == "spatial_unpad" {
		rowsOut = e.unpadGrid(x, gridH, gridW)
	} else if e.ImageNewline != nil && gridH > 1 {
		rowsOut = insertNewlines(x, gridH, gridW, e.ImageNewline)
	}

	// Projector.
	out := make([][]float32, len(rowsOut))
	for i, tok := range rowsOut {
		out[i] = e.project(tok)
	}
	return out, nil
}

// interpolatePos bilinearly samples the learned square position grid for a
// non-square patch grid. i is the token index (row-major within the target
// grid); posGrid is the learned grid side.
func (e *Encoder) interpolatePos(i, gridH, gridW, posGrid, offset int) []float32 {
	d := e.Config.EmbeddingDim
	out := make([]float32, d)
	patchIdx := i - offset // token index within the patch grid (class slot removed)
	row := patchIdx / gridW
	col := patchIdx % gridW
	fy := float64(row) * float64(posGrid-1) / float64(maxInt(gridH-1, 1))
	fx := float64(col) * float64(posGrid-1) / float64(maxInt(gridW-1, 1))
	y0, x0 := int(fy), int(fx)
	y1, x1 := minInt(y0+1, posGrid-1), minInt(x0+1, posGrid-1)
	y0, x0 = minInt(y0, posGrid-1), minInt(x0, posGrid-1)
	wy := fy - float64(y0)
	wx := fx - float64(x0)
	idx := func(y, x int) int {
		return (offset + y*posGrid + x) * d
	}
	for dd := 0; dd < d; dd++ {
		p00 := e.PosEmbed[idx(y0, x0)+dd]
		p01 := e.PosEmbed[idx(y0, x1)+dd]
		p10 := e.PosEmbed[idx(y1, x0)+dd]
		p11 := e.PosEmbed[idx(y1, x1)+dd]
		top := p00*(1-float32(wx)) + p01*float32(wx)
		bot := p10*(1-float32(wx)) + p11*float32(wx)
		out[dd] = top*(1-float32(wy)) + bot*float32(wy)
	}
	return out
}

// unpadGrid removes padded patch rows/cols and inserts newline embeddings.
// Padding is bottom/right; the actual content grid is derived from the
// resized (unpadded) dims, which we recompute from config: the caller passed
// the full square grid; content extent is gridH x gridW when the image filled
// the width, otherwise fewer rows are valid. We use the convention that the
// last full row/col boundary is where mean-color padding starts.
func (e *Encoder) unpadGrid(x [][]float32, gridH, gridW int) [][]float32 {
	// Detect mean-color padded columns/rows by comparing patch embeddings to
	// the projection of the pure-mean patch — too fragile. Instead use the
	// deterministic rule: content occupies (gridH x gridW) from PrepareImage's
	// fitInside; we recompute here.
	// NOTE: Encode's PrepareImage returns grid dims for the padded square;
	// recompute valid extents from the resize step by re-fitting.
	// The grid for content is (contentH/patch) x (contentW/patch); we recover
	// it from the last non-padding row/col by scanning for the newline anchor.
	// Simpler correct approach: PrepareImage records the content dims.
	if e.lastContentW > 0 {
		cw := e.lastContentW / e.Config.PatchSize
		ch := e.lastContentH / e.Config.PatchSize
		out := make([][]float32, 0, ch*(cw+1))
		newline := e.ImageNewline
		for r := 0; r < ch; r++ {
			row := x[r*gridW : r*gridW+cw]
			out = append(out, row...)
			if newline != nil && r < ch-1 {
				out = append(out, append([]float32(nil), newline...))
			}
		}
		return out
	}
	return x
}

// insertNewlines appends a newline embedding at the end of each patch-grid row.
func insertNewlines(x [][]float32, gridH, gridW int, newline []float32) [][]float32 {
	out := make([][]float32, 0, gridH*(gridW+1))
	for r := 0; r < gridH; r++ {
		out = append(out, x[r*gridW:(r+1)*gridW]...)
		if r < gridH-1 {
			out = append(out, append([]float32(nil), newline...))
		}
	}
	return out
}

func (e *Encoder) forwardLayer(l *Layer, x [][]float32) error {
	n := len(x)
	d := e.Config.EmbeddingDim
	heads := e.Config.NumHeads
	headDim := d / heads
	eps := e.Config.LayerNormEps
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// LN1
	normed := make([][]float32, n)
	for i := range x {
		normed[i] = layerNorm(x[i], l.LN1W, l.LN1B, eps)
	}

	// Q, K, V projections.
	q := make([][]float32, n)
	k := make([][]float32, n)
	v := make([][]float32, n)
	for i := range normed {
		q[i] = l.AttnQ.MulVecBias(normed[i])
		k[i] = l.AttnK.MulVecBias(normed[i])
		v[i] = l.AttnV.MulVecBias(normed[i])
	}

	// Bidirectional attention per head.
	attnOut := make([][]float32, n)
	for i := 0; i < n; i++ {
		acc := make([]float32, d)
		for h := 0; h < heads; h++ {
			// scores
			scores := make([]float32, n)
			maxS := float32(-math.MaxFloat32)
			for j := 0; j < n; j++ {
				var s float64
				for t := 0; t < headDim; t++ {
					s += float64(q[i][h*headDim+t]) * float64(k[j][h*headDim+t])
				}
				sf := float32(s) * scale
				scores[j] = sf
				if sf > maxS {
					maxS = sf
				}
			}
			// softmax
			sum := float32(0)
			for j := range scores {
				scores[j] = float32(math.Exp(float64(scores[j] - maxS)))
				sum += scores[j]
			}
			// weighted V
			for j := 0; j < n; j++ {
				w := scores[j] / sum
				if w == 0 {
					continue
				}
				for t := 0; t < headDim; t++ {
					acc[h*headDim+t] += w * v[j][h*headDim+t]
				}
			}
		}
		attnOut[i] = acc
	}

	// Out projection + residual.
	for i := 0; i < n; i++ {
		proj := l.AttnOut.MulVecBias(attnOut[i])
		for ddx := 0; ddx < d; ddx++ {
			x[i][ddx] += proj[ddx]
		}
	}

	// LN2 + FFN (exact GELU) + residual.
	for i := 0; i < n; i++ {
		n2 := layerNorm(x[i], l.LN2W, l.LN2B, eps)
		h := l.FFNUp.MulVecBias(n2)
		for j := range h {
			h[j] = gelu(h[j])
		}
		down := l.FFNDown.MulVecBias(h)
		for ddx := 0; ddx < d; ddx++ {
			x[i][ddx] += down[ddx]
		}
	}
	return nil
}

// project applies the LLaVA MLP projector: input-dim -> GELU -> llm-dim.
// Yi variant (MM1 present): mm.0 -> GELU -> mm.1 -> LN -> mm.2.
func (e *Encoder) project(tok []float32) []float32 {
	if e.YiNorm {
		h := geluVec(e.MM0.MulVecBias(tok))
		h = e.MM1.MulVecBias(h)
		// Yi-1.5: LlamaRMSNorm on the projector hidden
		h = rmsNormVec(h, 1e-5)
		return e.MM2.MulVecBias(h)
	}
	h := geluVec(e.MM0.MulVecBias(tok))
	return e.MM2.MulVecBias(h)
}

// MulVecBias computes W*x + b.
func (l *Linear) MulVecBias(x []float32) []float32 {
	out := make([]float32, l.Rows)
	if l.B != nil {
		copy(out, l.B)
	}
	matVecAccum(out, l.W, x, l.Rows, l.Cols)
	return out
}

func matVecAccum(out, w, x []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		row := w[r*cols : (r+1)*cols]
		var s float64
		for c := 0; c < cols; c++ {
			s += float64(row[c]) * float64(x[c])
		}
		out[r] += float32(s)
	}
}

func layerNorm(x, w, b []float32, eps float32) []float32 {
	out := make([]float32, len(x))
	copy(out, x)
	layerNormInPlace(out, w, b, eps)
	return out
}

func layerNormInPlace(x, w, b []float32, eps float32) {
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))
	var variance float64
	for _, v := range x {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= float64(len(x))
	inv := float32(1.0 / math.Sqrt(variance+float64(eps)))
	for i := range x {
		v := (x[i]-float32(mean))*inv
		if w != nil {
			v *= w[i]
		}
		if b != nil {
			v += b[i]
		}
		x[i] = v
	}
}

func rmsNormVec(x []float32, eps float32) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv
	}
	return out
}

func geluVec(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = gelu(v)
	}
	return out
}

// gelu is the exact erf-based GELU used by CLIP/SigLIP vision FFNs.
func gelu(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
