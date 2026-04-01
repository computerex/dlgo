package diffusion

import "math"

// GenZImagePE generates positional embeddings for Z-Image DiT.
// Returns a flat float32 slice representing PE of shape [posLen, headDim/2, 2, 2].
func GenZImagePE(h, w, patchSize, bs, contextLen, seqMultiOf, theta int, axesDim [3]int) []float32 {
	ids := genZImageIDs(h, w, patchSize, bs, contextLen, seqMultiOf)
	axesThetas := []float32{float32(theta), float32(theta), float32(theta)}
	axesDimSlice := axesDim[:]
	return embedND(ids, bs, axesThetas, axesDimSlice)
}

// boundMod returns (m - (a % m)) % m — the padding needed to reach next multiple of m.
func boundMod(a, m int) int {
	return (m - (a % m)) % m
}

// genZImageIDs generates the 3D position IDs for text and image tokens.
// Returns [bs*(txtPadLen+imgPadLen)][3] position IDs.
func genZImageIDs(h, w, patchSize, bs, contextLen, seqMultiOf int) [][]float32 {
	paddedContextLen := contextLen + boundMod(contextLen, seqMultiOf)

	// Text IDs: [0..paddedContextLen-1] with first axis = position+1
	txtIDs := make([][]float32, bs*paddedContextLen)
	for i := range txtIDs {
		txtIDs[i] = []float32{float32(i%paddedContextLen) + 1, 0, 0}
	}

	// Image IDs: grid of (index, row, col)
	index := paddedContextLen + 1
	imgIDs := genFluxImgIDs(h, w, patchSize, bs, index)

	// Pad image tokens to multiple of seqMultiOf
	imgPerBatch := len(imgIDs) / bs
	imgPadLen := boundMod(imgPerBatch, seqMultiOf)
	if imgPadLen > 0 {
		padIDs := make([][]float32, bs*imgPadLen)
		for i := range padIDs {
			padIDs[i] = []float32{0, 0, 0}
		}
		imgIDs = concatIDs(imgIDs, padIDs, bs)
	}

	return concatIDs(txtIDs, imgIDs, bs)
}

// genFluxImgIDs generates 3D position IDs for image patch tokens.
func genFluxImgIDs(h, w, patchSize, bs, index int) [][]float32 {
	hLen := h / patchSize
	wLen := w / patchSize

	ids := make([][]float32, hLen*wLen)
	for i := 0; i < hLen; i++ {
		for j := 0; j < wLen; j++ {
			ids[i*wLen+j] = []float32{float32(index), float32(i), float32(j)}
		}
	}

	// Repeat for batch
	repeated := make([][]float32, bs*len(ids))
	for b := 0; b < bs; b++ {
		for j := 0; j < len(ids); j++ {
			repeated[b*len(ids)+j] = ids[j]
		}
	}
	return repeated
}

// concatIDs concatenates two ID arrays along the sequence dimension, respecting batch.
func concatIDs(a, b [][]float32, bs int) [][]float32 {
	aLen := len(a) / bs
	bLen := len(b) / bs
	result := make([][]float32, bs*(aLen+bLen))
	for i := 0; i < bs; i++ {
		for j := 0; j < aLen; j++ {
			result[i*(aLen+bLen)+j] = a[i*aLen+j]
		}
		for j := 0; j < bLen; j++ {
			result[i*(aLen+bLen)+aLen+j] = b[i*bLen+j]
		}
	}
	return result
}

// embedND computes N-dimensional RoPE embeddings from position IDs.
// ids: [bs*posLen][numAxes], axesThetas: [numAxes], axesDim: [numAxes]
// Returns flat array of shape [bs*posLen, sum(axesDim)/2, 2, 2].
func embedND(ids [][]float32, bs int, axesThetas []float32, axesDim []int) []float32 {
	numAxes := len(axesDim)
	posLen := len(ids) / bs

	// Transpose IDs: [numAxes][bs*posLen]
	transIDs := make([][]float32, numAxes)
	for i := 0; i < numAxes; i++ {
		transIDs[i] = make([]float32, len(ids))
		for j := range ids {
			transIDs[i][j] = ids[j][i]
		}
	}

	// Compute total embedding dimension
	var embDim int
	for _, d := range axesDim {
		embDim += d / 2
	}
	// Each position has embDim * 4 values (cos, -sin, sin, cos per pair)
	totalPerPos := embDim * 4

	result := make([]float32, bs*posLen*totalPerPos)

	offset := 0
	for ax := 0; ax < numAxes; ax++ {
		axRope := ropeForAxis(transIDs[ax], axesDim[ax], axesThetas[ax])
		// axRope: [bs*posLen][axisDim[ax]/2 * 4]
		ropeWidth := axesDim[ax] / 2 * 4
		for b := 0; b < bs; b++ {
			for j := 0; j < posLen; j++ {
				srcIdx := j // Position within single-batch (IDs repeat per batch)
				dstBase := (b*posLen + j) * totalPerPos
				for k := 0; k < ropeWidth; k++ {
					result[dstBase+offset+k] = axRope[srcIdx][k]
				}
			}
		}
		offset += ropeWidth
	}

	return result
}

// ropeForAxis computes RoPE cos/sin pairs for one axis.
// pos: [posLen] positions, dim: axis dimension, theta: frequency base
// Returns [posLen][dim/2 * 4].
func ropeForAxis(pos []float32, dim int, theta float32) [][]float32 {
	halfDim := dim / 2
	posLen := len(pos)

	// omega[j] = 1 / theta^(2j/dim)
	omega := make([]float64, halfDim)
	for j := 0; j < halfDim; j++ {
		scale := float64(2*j) / float64(dim)
		omega[j] = 1.0 / math.Pow(float64(theta), scale)
	}

	result := make([][]float32, posLen)
	for i := 0; i < posLen; i++ {
		row := make([]float32, halfDim*4)
		for j := 0; j < halfDim; j++ {
			angle := float64(pos[i]) * omega[j]
			c := float32(math.Cos(angle))
			s := float32(math.Sin(angle))
			row[4*j] = c
			row[4*j+1] = -s
			row[4*j+2] = s
			row[4*j+3] = c
		}
		result[i] = row
	}
	return result
}

// ApplyRoPE3D applies precomputed 3D rotary embeddings to Q or K vectors in-place.
// vec: [nPos * dim] where dim = nHeads * headDim
// pe: flat PE array, [posLen * headDim * 2] (headDim/2 pairs × 4 values each)
// peOffset: which position in the PE array to start from (for txt/img slicing)
func ApplyRoPE3D(vec []float32, pe []float32, nPos, nHeads, headDim, peOffset int) {
	halfDim := headDim / 2
	dim := nHeads * headDim
	peStride := headDim * 2 // halfDim * 4

	for p := 0; p < nPos; p++ {
		peBase := (peOffset + p) * peStride
		for h := 0; h < nHeads; h++ {
			for d := 0; d < halfDim; d++ {
				peIdx := peBase + d*4
				cosVal := pe[peIdx]
				sinVal := pe[peIdx+2]

				idx0 := p*dim + h*headDim + 2*d
				idx1 := idx0 + 1

				re := vec[idx0]
				im := vec[idx1]

				// Forward rotation R(θ): [cos -sin; sin cos]
				// sd.cpp: out_re = re*pe[0] + im*pe[1] = re*cos + im*(-sin) = re*cos - im*sin
				// sd.cpp: out_im = re*pe[2] + im*pe[3] = re*sin + im*cos
				vec[idx0] = re*cosVal - im*sinVal
				vec[idx1] = re*sinVal + im*cosVal
			}
		}
	}
}
