package vision

import (
	"fmt"
	"image"
	"math"
	"os"
	"time"

	"github.com/computerex/dlgo/quant"
)

// Encode runs the full pipeline: preprocess -> patch embed -> ViT ->
// projector, returning one embedding row per image token in the LLM's
// embedding space (llmDim = MM2.Rows).
func (e *Encoder) Encode(src image.Image) ([][]float32, error) {
	dbgTiming := os.Getenv("DLGO_VISION_TIMING") != ""
	stage := time.Now()
	mark := func(name string) {
		if dbgTiming {
			fmt.Fprintf(os.Stderr, "[vision] %-12s %s\n", name, time.Since(stage))
			stage = time.Now()
		}
	}
	gridH, gridW, feats, err := e.PrepareImage(src)
	if err != nil {
		return nil, err
	}
	mark("preprocess")

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
	mark("patchembed")

	// Class token prepend (always position 0 when present).
	nPos := len(e.PosEmbed) / e.Config.EmbeddingDim
	embd := tokens
	// Qwen3 VL runs attention in 2x2-tile order so each consecutive group is
	// later consumed by its merger. Keep the source-grid index for absolute and
	// rotary positions.
	posIndex := make([]int, len(tokens))
	for i := range posIndex {
		posIndex[i] = i
	}
	if e.Config.ProjectorType == "qwen3vl_merger" {
		embd, posIndex = reorderMergeTiles(tokens, gridH, gridW, e.Config.SpatialMerge)
		e.activePosIndex, e.activeGridW = posIndex, gridW
	} else {
		e.activePosIndex, e.activeGridW = nil, 0
	}
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
	// class slot when present. Qwen3 VL applies its M-RoPE in attention; its
	// learned absolute position table remains indexed in normal grid order.
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
			pos := i
			if e.activePosIndex != nil {
				pos = e.activePosIndex[i]
			}
			pe = e.PosEmbed[pos*e.Config.EmbeddingDim : (pos+1)*e.Config.EmbeddingDim]
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
	mark("blocks")

	// Post-LN (CLIP).
	for _, tok := range x {
		layerNormInPlace(tok, e.PostLNW, e.PostLNB, e.Config.LayerNormEps)
	}

	// Drop class token output.
	if e.ClassEmb != nil {
		x = x[1:]
	}

	// Qwen3 VL merges each 2x2 patch block, in raster order, before its
	// two-layer merger MLP. This turns a 48x48 patch grid into 24x24 LLM rows.
	rowsOut := x
	if e.Config.ProjectorType == "qwen3vl_merger" {
		var err error
		rowsOut, err = mergeSpatial2x2(x, gridH, gridW, e.Config.SpatialMerge)
		if err != nil {
			return nil, err
		}
	} else if e.Config.MergeType == "spatial_unpad" {
		rowsOut = e.unpadGrid(x, gridH, gridW)
	} else if e.ImageNewline != nil && gridH > 1 {
		rowsOut = insertNewlines(x, gridH, gridW, e.ImageNewline)
	}

	// Projector (parallel over tokens).
	out := make([][]float32, len(rowsOut))
	e.parallelFor(len(rowsOut), func(i int) {
		out[i] = e.project(rowsOut[i])
	})
	mark("projector")
	return out, nil
}

// reorderMergeTiles makes consecutive tokens represent each merge tile. This
// is the order Qwen3 VL uses both during ViT attention and merger projection.
func reorderMergeTiles(x [][]float32, gridH, gridW, merge int) ([][]float32, []int) {
	if merge <= 0 {
		merge = 2
	}
	out := make([][]float32, 0, len(x))
	idx := make([]int, 0, len(x))
	for y := 0; y < gridH; y += merge {
		for z := 0; z < gridW; z += merge {
			for dy := 0; dy < merge; dy++ {
				for dx := 0; dx < merge; dx++ {
					i := (y+dy)*gridW + z + dx
					out, idx = append(out, x[i]), append(idx, i)
				}
			}
		}
	}
	return out, idx
}

// mergeSpatial2x2 concatenates patches in each merge-by-merge tile. Qwen's
// projector expects order (top-left, top-right, bottom-left, bottom-right).
func mergeSpatial2x2(x [][]float32, gridH, gridW, merge int) ([][]float32, error) {
	if merge <= 0 {
		merge = 2
	}
	if gridH%merge != 0 || gridW%merge != 0 || len(x) != gridH*gridW {
		return nil, fmt.Errorf("vision: cannot merge %dx%d grid with factor %d", gridH, gridW, merge)
	}
	d := len(x[0])
	out := make([][]float32, 0, len(x)/(merge*merge))
	for y := 0; y < gridH; y += merge {
		for z := 0; z < gridW; z += merge {
			row := make([]float32, d*merge*merge)
			at := 0
			for dy := 0; dy < merge; dy++ {
				for dx := 0; dx < merge; dx++ {
					copy(row[at:at+d], x[(y+dy)*gridW+z+dx])
					at += d
				}
			}
			out = append(out, row)
		}
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

// applyQwenVisionRoPE applies Qwen's 2-D vision M-RoPE. Head dimensions are
// partitioned evenly into four sections: row, column, row, column. GGML's
// vision RoPE rotates first-half/second-half pairs inside each section.
func applyQwenVisionRoPE(x []float32, heads, headDim, row, col int) {
	quarter := headDim / 4
	for h := 0; h < heads; h++ {
		v := x[h*headDim : (h+1)*headDim]
		for sec := 0; sec < 4; sec++ {
			p := row
			if sec%2 == 1 {
				p = col
			}
			base := sec * quarter
			half := quarter / 2
			for i := 0; i < half; i++ {
				theta := float64(p) / math.Pow(10000, float64(2*i)/float64(quarter))
				c, s := float32(math.Cos(theta)), float32(math.Sin(theta))
				a, b := v[base+i], v[base+half+i]
				v[base+i], v[base+half+i] = a*c-b*s, a*s+b*c
			}
		}
	}
}

func (e *Encoder) forwardLayer(l *Layer, x [][]float32) error {
	n := len(x)
	d := e.Config.EmbeddingDim
	heads := e.Config.NumHeads
	headDim := d / heads
	eps := e.Config.LayerNormEps
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Flatten tokens to a row-major [n][d] buffer once per layer.
	xf := make([]float32, n*d)
	for i := 0; i < n; i++ {
		copy(xf[i*d:(i+1)*d], x[i])
	}
	// LN1 (parallel over tokens).
	nf := make([]float32, n*d)
	e.parallelFor(n, func(i int) {
		layerNormInto(nf[i*d:(i+1)*d], xf[i*d:(i+1)*d], l.LN1W, l.LN1B, eps)
	})

	// Batched Q, K, V.
	qf := make([]float32, n*d)
	kf := make([]float32, n*d)
	vf := make([]float32, n*d)
	e.gemmBT(l.AttnQ.W, l.AttnQ.B, nf, qf, n, d, d)
	e.gemmBT(l.AttnK.W, l.AttnK.B, nf, kf, n, d, d)
	e.gemmBT(l.AttnV.W, l.AttnV.B, nf, vf, n, d, d)
	if e.Config.ProjectorType == "qwen3vl_merger" {
		for i := 0; i < n; i++ {
			pos := e.activePosIndex[i]
			row, col := pos/e.activeGridW, pos%e.activeGridW
			applyQwenVisionRoPE(qf[i*d:(i+1)*d], heads, headDim, row, col)
			applyQwenVisionRoPE(kf[i*d:(i+1)*d], heads, headDim, row, col)
		}
	}

	// Bidirectional attention per query token (parallel).
	af := make([]float32, n*d)
	e.parallelFor(n, func(i int) {
		scores := make([]float32, n)
		acc := af[i*d : (i+1)*d]
		for h := 0; h < heads; h++ {
			qh := qf[i*d+h*headDim : i*d+h*headDim+headDim]
			maxS := float32(-math.MaxFloat32)
			for j := 0; j < n; j++ {
				khj := kf[j*d+h*headDim : j*d+h*headDim+headDim]
				var sc float32
				for t := 0; t < headDim; t++ {
					sc += qh[t] * khj[t]
				}
				sc *= scale
				scores[j] = sc
				if sc > maxS {
					maxS = sc
				}
			}
			sum := float32(0)
			for j := 0; j < n; j++ {
				scores[j] = float32(math.Exp(float64(scores[j] - maxS)))
				sum += scores[j]
			}
			accH := acc[h*headDim : h*headDim+headDim]
			for j := 0; j < n; j++ {
				w := scores[j] / sum
				if w == 0 {
					continue
				}
				vj := vf[j*d+h*headDim : j*d+h*headDim+headDim]
				for t := 0; t < headDim; t++ {
					accH[t] += w * vj[t]
				}
			}
		}
	})

	// Batched attn-out projection.
	pf := make([]float32, n*d)
	e.gemmBT(l.AttnOut.W, l.AttnOut.B, af, pf, n, d, d)

	// Residual + LN2 (parallel over tokens).
	nf2 := make([]float32, n*d)
	e.parallelFor(n, func(i int) {
		for ddx := 0; ddx < d; ddx++ {
			xf[i*d+ddx] += pf[i*d+ddx]
		}
		layerNormInto(nf2[i*d:(i+1)*d], xf[i*d:(i+1)*d], l.LN2W, l.LN2B, eps)
	})

	// Batched FFN up (d -> ffn), GELU, down (ffn -> d), residual.
	ffn := l.FFNUp.Rows
	hf := make([]float32, n*ffn)
	e.gemmBT(l.FFNUp.W, l.FFNUp.B, nf2, hf, n, ffn, d)
	e.parallelFor(n, func(i int) {
		for j := 0; j < ffn; j++ {
			hf[i*ffn+j] = gelu(hf[i*ffn+j])
		}
	})
	df := make([]float32, n*d)
	e.gemmBT(l.FFNDown.W, l.FFNDown.B, hf, df, n, d, ffn)
	e.parallelFor(n, func(i int) {
		out := x[i]
		for ddx := 0; ddx < d; ddx++ {
			out[ddx] = xf[i*d+ddx] + df[i*d+ddx]
		}
	})
	return nil
}

// parallelForScratch is parallelFor with a per-worker scratch buffer reused
// layerNormInto writes LayerNorm(x) * w + b into out (len >= len(x)).
func layerNormInto(out, x, w, b []float32, eps float32) {
	out = out[:len(x)]
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))
	var variance float64
	for _, v := range x {
		dd := float64(v) - mean
		variance += dd * dd
	}
	variance /= float64(len(x))
	inv := float32(1.0 / math.Sqrt(variance+float64(eps)))
	for i := range x {
		v := (x[i] - float32(mean)) * inv
		if w != nil {
			v *= w[i]
		}
		if b != nil {
			v += b[i]
		}
		out[i] = v
	}
}

// parallelFor runs fn(0..n) across the encoder's worker pool (falls back to a
// serial loop when no pool is configured).
func (e *Encoder) parallelFor(n int, fn func(i int)) {
	if e.pool != nil && n > 1 {
		e.pool.ParallelFor(n, fn)
		return
	}
	for i := 0; i < n; i++ {
		fn(i)
	}
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

// MulVecInto computes W*x + b into the caller-provided out (len >= Rows),
// avoiding a per-call allocation. Used by the parallel encoder hot path.
func (l *Linear) MulVecInto(out, x []float32) {
	out = out[:l.Rows]
	if l.B != nil {
		copy(out, l.B)
	} else {
		for i := range out {
			out[i] = 0
		}
	}
	matVecAccum(out, l.W, x, l.Rows, l.Cols)
}

func matVecAccum(out, w, x []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		row := w[r*cols : (r+1)*cols]
		var s float32
		for c := 0; c < cols; c++ {
			s += row[c] * x[c]
		}
		out[r] += s
	}
}

// gemmBT computes out[i][o] = sum_c W[o][c] * X[i][c] (+ bias), i.e. Y = X*W^T.
// Parallel over tokens; each row dot uses a SIMD kernel when available.
//
// W is row-major [rows][cols]; X is [n][cols]; out is [n][rows].
func (e *Encoder) gemmBT(w, bias []float32, x, out []float32, n, rows, cols int) {
	e.parallelFor(n, func(i int) {
		xRow := x[i*cols : (i+1)*cols]
		o := out[i*rows : (i+1)*rows]
		for r := 0; r < rows; r++ {
			wRow := w[r*cols : (r+1)*cols]
			s := quant.SIMDDotF32(wRow, xRow, cols)
			if bias != nil {
				s += bias[r]
			}
			o[r] = s
		}
	})
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
		v := (x[i] - float32(mean)) * inv
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
