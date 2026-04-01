package diffusion

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// --- genZImageIDs tests ---

func TestGenZImageIDsShape(t *testing.T) {
	h, w, patchSize, bs, contextLen, seqMultiOf := 128, 128, 2, 1, 30, 32

	ids := genZImageIDs(h, w, patchSize, bs, contextLen, seqMultiOf)

	// Expect: paddedTxt + paddedImg tokens, each with 3 axes
	paddedContextLen := contextLen + boundMod(contextLen, seqMultiOf)
	hPatches := (h + patchSize/2) / patchSize // 64
	wPatches := (w + patchSize/2) / patchSize // 64
	nImgTokens := hPatches * wPatches          // 4096
	imgPadLen := boundMod(nImgTokens, seqMultiOf)
	nImgPadded := nImgTokens + imgPadLen

	expectedLen := bs * (paddedContextLen + nImgPadded)
	if len(ids) != expectedLen {
		t.Fatalf("genZImageIDs: got %d IDs, want %d (paddedTxt=%d + paddedImg=%d)",
			len(ids), expectedLen, paddedContextLen, nImgPadded)
	}

	// Each ID should have exactly 3 axes
	for i, id := range ids {
		if len(id) != 3 {
			t.Errorf("ID[%d] has %d axes, want 3", i, len(id))
		}
	}
}

func TestGenZImageIDsTextStructure(t *testing.T) {
	ids := genZImageIDs(128, 128, 2, 1, 30, 32)
	paddedContextLen := 32 // 30 + 2 (padded to 32)

	// Text tokens: first paddedContextLen IDs
	for i := 0; i < paddedContextLen; i++ {
		// First axis = position+1 (for actual tokens), second/third = 0
		if ids[i][1] != 0 || ids[i][2] != 0 {
			t.Errorf("text ID[%d] = %v, expected axes 1,2 to be 0", i, ids[i])
		}
	}
	// First 30 text tokens should have first axis = i+1
	for i := 0; i < 30; i++ {
		if ids[i][0] != float32(i+1) {
			t.Errorf("text ID[%d][0] = %f, want %f", i, ids[i][0], float32(i+1))
		}
	}
}

func TestGenZImageIDsImageTokensSameIndex(t *testing.T) {
	ids := genZImageIDs(128, 128, 2, 1, 30, 32)
	paddedContextLen := 32

	// All real image tokens should have the same first axis (paddedContextLen+1)
	expectedIdx := float32(paddedContextLen + 1)
	hPatches := 64
	wPatches := 64
	for i := 0; i < hPatches*wPatches; i++ {
		imgID := ids[paddedContextLen+i]
		if imgID[0] != expectedIdx {
			t.Errorf("img ID[%d] axis 0 = %f, want %f", i, imgID[0], expectedIdx)
		}
	}
}

func TestGenZImageIDsImageGrid(t *testing.T) {
	ids := genZImageIDs(16, 16, 2, 1, 4, 32)
	paddedContextLen := 32 // 4 padded→32

	hPatches := 8
	wPatches := 8
	for row := 0; row < hPatches; row++ {
		for col := 0; col < wPatches; col++ {
			idx := paddedContextLen + row*wPatches + col
			imgID := ids[idx]
			if imgID[1] != float32(row) || imgID[2] != float32(col) {
				t.Errorf("img ID at (%d,%d): got axes=[%f,%f], want [%d,%d]",
					row, col, imgID[1], imgID[2], row, col)
			}
		}
	}
}

// --- embedND / RoPE tests ---

func TestGenZImagePEShape(t *testing.T) {
	axesDim := [3]int{32, 48, 48}
	headDim := 128 // sum of axesDim
	pe := GenZImagePE(128, 128, 2, 1, 30, 32, 256, axesDim)

	// Expected: paddedTxt + paddedImg positions
	paddedTxt := 32
	nImg := 4096
	imgPad := boundMod(nImg, 32) // 0
	totalPos := paddedTxt + nImg + imgPad

	peStride := headDim * 2 // headDim/2 pairs × 4 values = headDim*2
	expectedLen := totalPos * peStride
	if len(pe) != expectedLen {
		t.Errorf("PE length = %d, want %d (totalPos=%d × stride=%d)",
			len(pe), expectedLen, totalPos, peStride)
	}
}

func TestRoPERotationMatrix(t *testing.T) {
	// Each PE entry at position p for frequency pair d should be a 2×2 rotation matrix:
	// [cos, -sin, sin, cos]
	// Verify: cos²+sin² = 1
	pe := GenZImagePE(8, 8, 2, 1, 4, 32, 256, [3]int{32, 48, 48})
	peStride := 128 * 2
	totalPos := len(pe) / peStride
	halfDim := 64 // (32+48+48)/2

	for p := 0; p < min(totalPos, 5); p++ { // check first 5 positions
		for d := 0; d < min(halfDim, 10); d++ { // check first 10 pairs
			base := p*peStride + d*4
			cosV := pe[base]
			negSin := pe[base+1]
			sinV := pe[base+2]
			cosV2 := pe[base+3]

			// cos should equal cos2
			if math.Abs(float64(cosV-cosV2)) > 1e-6 {
				t.Errorf("pos=%d, d=%d: cos mismatch %f vs %f", p, d, cosV, cosV2)
			}
			// -sin should be negative of sin
			if math.Abs(float64(negSin+sinV)) > 1e-6 {
				t.Errorf("pos=%d, d=%d: sin mismatch %f vs %f", p, d, negSin, sinV)
			}
			// cos²+sin²=1
			norm := cosV*cosV + sinV*sinV
			if math.Abs(float64(norm-1.0)) > 1e-5 {
				t.Errorf("pos=%d, d=%d: cos²+sin² = %f, want 1.0", p, d, norm)
			}
		}
	}
}

func TestApplyRoPE3DPreservesNorm(t *testing.T) {
	// RoPE is a rotation — it should preserve the L2 norm of each head vector.
	nPos, nHeads, headDim := 4, 2, 8
	dim := nHeads * headDim

	// Create simple PE (identity rotations would be all cos=1, sin=0)
	peStride := headDim * 2
	pe := make([]float32, (nPos+2)*peStride) // extra 2 for offset
	for p := 0; p < nPos+2; p++ {
		for d := 0; d < headDim/2; d++ {
			angle := float64(p) * 0.1 * float64(d+1) // varying angles
			c := float32(math.Cos(angle))
			s := float32(math.Sin(angle))
			base := p*peStride + d*4
			pe[base] = c
			pe[base+1] = -s
			pe[base+2] = s
			pe[base+3] = c
		}
	}

	// Create test vector
	vec := make([]float32, nPos*dim)
	for i := range vec {
		vec[i] = float32(i+1) * 0.1
	}

	// Compute pre-rotation norms per (position, head)
	preNorms := make([]float64, nPos*nHeads)
	for p := 0; p < nPos; p++ {
		for h := 0; h < nHeads; h++ {
			sum := float64(0)
			for d := 0; d < headDim; d++ {
				v := float64(vec[p*dim+h*headDim+d])
				sum += v * v
			}
			preNorms[p*nHeads+h] = math.Sqrt(sum)
		}
	}

	ApplyRoPE3D(vec, pe, nPos, nHeads, headDim, 0)

	// Compute post-rotation norms
	for p := 0; p < nPos; p++ {
		for h := 0; h < nHeads; h++ {
			sum := float64(0)
			for d := 0; d < headDim; d++ {
				v := float64(vec[p*dim+h*headDim+d])
				sum += v * v
			}
			postNorm := math.Sqrt(sum)
			preNorm := preNorms[p*nHeads+h]
			if math.Abs(preNorm-postNorm) > 1e-4 {
				t.Errorf("norm changed at pos=%d, head=%d: %.6f → %.6f",
					p, h, preNorm, postNorm)
			}
		}
	}
}

func TestApplyRoPE3DIdentityRotation(t *testing.T) {
	// With all angles = 0 (cos=1, sin=0), RoPE should be identity
	nPos, nHeads, headDim := 2, 1, 4
	peStride := headDim * 2

	pe := make([]float32, nPos*peStride)
	for p := 0; p < nPos; p++ {
		for d := 0; d < headDim/2; d++ {
			base := p*peStride + d*4
			pe[base] = 1.0 // cos
			pe[base+1] = 0  // -sin
			pe[base+2] = 0  // sin
			pe[base+3] = 1.0 // cos
		}
	}

	vec := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	orig := make([]float32, len(vec))
	copy(orig, vec)

	ApplyRoPE3D(vec, pe, nPos, nHeads, headDim, 0)

	for i := range vec {
		if math.Abs(float64(vec[i]-orig[i])) > 1e-6 {
			t.Errorf("identity rotation changed vec[%d]: %f → %f", i, orig[i], vec[i])
		}
	}
}

func TestApplyRoPE3DPeOffset(t *testing.T) {
	// Verify peOffset correctly shifts which PE positions are used
	nPos, nHeads, headDim := 2, 1, 4
	dim := nHeads * headDim
	peStride := headDim * 2

	// Create PE with distinct rotations per position
	totalPEPos := 5
	pe := make([]float32, totalPEPos*peStride)
	for p := 0; p < totalPEPos; p++ {
		for d := 0; d < headDim/2; d++ {
			angle := float64(p+1) * 0.5 * float64(d+1)
			base := p*peStride + d*4
			pe[base] = float32(math.Cos(angle))
			pe[base+1] = float32(-math.Sin(angle))
			pe[base+2] = float32(math.Sin(angle))
			pe[base+3] = float32(math.Cos(angle))
		}
	}

	// Apply with offset=0
	vec1 := make([]float32, nPos*dim)
	for i := range vec1 {
		vec1[i] = float32(i + 1)
	}
	ApplyRoPE3D(vec1, pe, nPos, nHeads, headDim, 0)

	// Apply with offset=2
	vec2 := make([]float32, nPos*dim)
	for i := range vec2 {
		vec2[i] = float32(i + 1)
	}
	ApplyRoPE3D(vec2, pe, nPos, nHeads, headDim, 2)

	// They should be different (different PE positions used)
	same := true
	for i := range vec1 {
		if vec1[i] != vec2[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different PE offsets produced identical results")
	}
}

// --- scaledMultiHeadAttention tests ---

func TestAttentionIdentityValues(t *testing.T) {
	// With seqLen=1, attention is trivially softmax(score)*V = V
	seqLen := 1
	numHeads := 2
	headDim := 4
	qDim := numHeads * headDim
	scale := float32(1.0 / float64(headDim))

	q := make([]float32, seqLen*qDim)
	k := make([]float32, seqLen*qDim)
	v := []float32{1, 2, 3, 4, 5, 6, 7, 8} // [1, 8]
	out := make([]float32, seqLen*qDim)

	scores := make([]float32, seqLen)
	scaledMultiHeadAttention(out, q, k, v, seqLen, numHeads, numHeads, headDim, scale, scores)

	for i := range v {
		if math.Abs(float64(out[i]-v[i])) > 1e-6 {
			t.Errorf("single-token attention out[%d] = %f, want %f (V passthrough)", i, out[i], v[i])
		}
	}
}

func TestAttentionUniformScores(t *testing.T) {
	// With identical Q and K, attention weights should be uniform
	seqLen := 4
	numHeads := 1
	headDim := 2
	qDim := numHeads * headDim
	scale := float32(1.0)

	// All Q and K the same → all scores identical → uniform weights = 1/seqLen
	q := make([]float32, seqLen*qDim)
	k := make([]float32, seqLen*qDim)
	v := make([]float32, seqLen*qDim)
	for i := range q {
		q[i] = 1.0
		k[i] = 1.0
	}
	// V: each position has a distinct value
	for i := 0; i < seqLen; i++ {
		for j := 0; j < headDim; j++ {
			v[i*qDim+j] = float32(i*headDim + j)
		}
	}
	out := make([]float32, seqLen*qDim)
	scores := make([]float32, seqLen)
	scaledMultiHeadAttention(out, q, k, v, seqLen, numHeads, numHeads, headDim, scale, scores)

	// Each output should be the mean of all V vectors
	for j := 0; j < headDim; j++ {
		mean := float32(0)
		for i := 0; i < seqLen; i++ {
			mean += v[i*qDim+j]
		}
		mean /= float32(seqLen)

		for i := 0; i < seqLen; i++ {
			if diff := math.Abs(float64(out[i*qDim+j] - mean)); diff > 1e-4 {
				t.Errorf("uniform attention pos=%d, dim=%d: got %f, want mean=%f",
					i, j, out[i*qDim+j], mean)
			}
		}
	}
}

func TestAttentionOutputDimensions(t *testing.T) {
	seqLen := 10
	numHeads := 4
	numKVHeads := 2
	headDim := 8
	qDim := numHeads * headDim
	kvDim := numKVHeads * headDim

	q := make([]float32, seqLen*qDim)
	k := make([]float32, seqLen*kvDim)
	v := make([]float32, seqLen*kvDim)
	out := make([]float32, seqLen*qDim)

	for i := range q {
		q[i] = 0.01
	}
	for i := range k {
		k[i] = 0.01
	}
	for i := range v {
		v[i] = float32(i) * 0.001
	}

	scaledMultiHeadAttention(out, q, k, v, seqLen, numHeads, numKVHeads, headDim, 1.0/float32(headDim), make([]float32, seqLen))

	// Verify output is not all zeros (sanity)
	allZero := true
	for _, val := range out {
		if val != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("attention output is all zeros")
	}
}

// --- conv2d tests ---

func TestConv2dIdentityKernel(t *testing.T) {
	// 1×1 conv with weight=identity should be a passthrough
	inCh, outCh := 2, 2
	H, W := 3, 3
	input := make([]float32, inCh*H*W)
	for i := range input {
		input[i] = float32(i)
	}

	// Identity 1×1 kernel: w[oc][ic] = delta(oc,ic)
	w := Conv2DWeight{
		Weight: []float32{1, 0, 0, 1}, // [outCh, inCh, 1, 1]
		Bias:   nil,
		InCh:   inCh,
		OutCh:  outCh,
		KH:     1,
		KW:     1,
	}

	out := conv2d(input, w, H, W, 1)
	if len(out) != len(input) {
		t.Fatalf("conv2d output len=%d, want %d", len(out), len(input))
	}
	for i := range input {
		if math.Abs(float64(out[i]-input[i])) > 1e-6 {
			t.Errorf("identity conv out[%d] = %f, want %f", i, out[i], input[i])
		}
	}
}

func TestConv2dBias(t *testing.T) {
	// 1×1 conv with zero weights and non-zero bias
	inCh, outCh := 1, 2
	H, W := 2, 2
	input := make([]float32, inCh*H*W)

	w := Conv2DWeight{
		Weight: make([]float32, outCh*inCh*1*1), // all zero
		Bias:   []float32{10, 20},
		InCh:   inCh,
		OutCh:  outCh,
		KH:     1,
		KW:     1,
	}

	out := conv2d(input, w, H, W, 1)
	// Channel 0: all 10, Channel 1: all 20
	for i := 0; i < H*W; i++ {
		if out[i] != 10 {
			t.Errorf("out ch0[%d] = %f, want 10", i, out[i])
		}
		if out[H*W+i] != 20 {
			t.Errorf("out ch1[%d] = %f, want 20", i, out[H*W+i])
		}
	}
}

func TestConv2dOutputSize(t *testing.T) {
	// 3×3 conv with pad=1 should preserve spatial dims
	inCh, outCh := 3, 8
	H, W := 16, 16
	input := make([]float32, inCh*H*W)
	w := Conv2DWeight{
		Weight: make([]float32, outCh*inCh*3*3),
		Bias:   make([]float32, outCh),
		InCh:   inCh,
		OutCh:  outCh,
		KH:     3,
		KW:     3,
	}
	out := conv2d(input, w, H, W, 1)
	if len(out) != outCh*H*W {
		t.Errorf("conv2d 3×3 output len=%d, want %d", len(out), outCh*H*W)
	}
}

// --- upsample2x tests ---

func TestUpsample2x(t *testing.T) {
	C, H, W := 1, 2, 2
	input := []float32{1, 2, 3, 4}
	out := upsample2x(input, C, H, W)

	if len(out) != C*H*2*W*2 {
		t.Fatalf("upsample output len=%d, want %d", len(out), C*H*2*W*2)
	}

	// Expected: each pixel repeated in a 2×2 block
	expected := []float32{
		1, 1, 2, 2,
		1, 1, 2, 2,
		3, 3, 4, 4,
		3, 3, 4, 4,
	}
	for i := range expected {
		if out[i] != expected[i] {
			t.Errorf("upsample[%d] = %f, want %f", i, out[i], expected[i])
		}
	}
}

// --- groupNormBatch tests ---

func TestGroupNormBatchIdentity(t *testing.T) {
	// With weight=1, bias=0, output should be normalized
	C := 4
	spatial := 4
	numGroups := 2
	eps := float32(1e-6)

	input := make([]float32, C*spatial)
	for i := range input {
		input[i] = float32(i)
	}
	w := []float32{1, 1, 1, 1}
	b := []float32{0, 0, 0, 0}
	out := make([]float32, C*spatial)

	groupNormBatch(out, input, w, b, C, spatial, numGroups, eps)

	// Each group should have mean≈0 after normalization
	groupSize := C / numGroups
	for g := 0; g < numGroups; g++ {
		sum := float64(0)
		n := groupSize * spatial
		for c := 0; c < groupSize; c++ {
			ch := g*groupSize + c
			for s := 0; s < spatial; s++ {
				sum += float64(out[ch*spatial+s])
			}
		}
		mean := sum / float64(n)
		if math.Abs(mean) > 1e-4 {
			t.Errorf("group %d: mean after normalization = %f, want ~0", g, mean)
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// --- conv2d parity tests: im2col vs naive ---

func TestConv2dParitySmall(t *testing.T) {
	// Small 3×3 conv: 4 inCh → 8 outCh, 8×8 spatial
	rng := rand.New(rand.NewSource(42))
	inCh, outCh, kH, kW := 4, 8, 3, 3
	H, W := 8, 8

	input := make([]float32, inCh*H*W)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}
	weight := make([]float32, outCh*inCh*kH*kW)
	for i := range weight {
		weight[i] = rng.Float32()*0.1 - 0.05
	}
	bias := make([]float32, outCh)
	for i := range bias {
		bias[i] = rng.Float32()*0.2 - 0.1
	}

	w := Conv2DWeight{Weight: weight, Bias: bias, InCh: inCh, OutCh: outCh, KH: kH, KW: kW}

	fast := conv2d(input, w, H, W, 1)
	naive := conv2dNaive(input, w, H, W, 1)

	if len(fast) != len(naive) {
		t.Fatalf("length mismatch: fast=%d naive=%d", len(fast), len(naive))
	}
	for i := range fast {
		diff := math.Abs(float64(fast[i] - naive[i]))
		if diff > 1e-4 {
			t.Errorf("index %d: fast=%f naive=%f diff=%e", i, fast[i], naive[i], diff)
		}
	}
}

func TestConv2dParityLarger(t *testing.T) {
	// Larger conv matching VAE's inner dimensions: 128 inCh → 128 outCh, 32×32 spatial
	rng := rand.New(rand.NewSource(123))
	inCh, outCh, kH, kW := 128, 128, 3, 3
	H, W := 32, 32

	input := make([]float32, inCh*H*W)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}
	weight := make([]float32, outCh*inCh*kH*kW)
	for i := range weight {
		weight[i] = rng.Float32()*0.02 - 0.01
	}
	bias := make([]float32, outCh)
	for i := range bias {
		bias[i] = rng.Float32()*0.1 - 0.05
	}

	w := Conv2DWeight{Weight: weight, Bias: bias, InCh: inCh, OutCh: outCh, KH: kH, KW: kW}

	fast := conv2d(input, w, H, W, 1)
	naive := conv2dNaive(input, w, H, W, 1)

	if len(fast) != len(naive) {
		t.Fatalf("length mismatch: fast=%d naive=%d", len(fast), len(naive))
	}
	maxDiff := float64(0)
	for i := range fast {
		diff := math.Abs(float64(fast[i] - naive[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > 1e-3 {
		t.Errorf("max diff=%e (want < 1e-3)", maxDiff)
	}
	t.Logf("parity: %d outputs, max diff=%e", len(fast), maxDiff)
}

func TestConv2dParity1x1(t *testing.T) {
	// 1×1 conv (used for attention projections): 512 inCh → 512 outCh, 16×16 spatial
	rng := rand.New(rand.NewSource(77))
	inCh, outCh, kH, kW := 512, 512, 1, 1
	H, W := 16, 16

	input := make([]float32, inCh*H*W)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}
	weight := make([]float32, outCh*inCh*kH*kW)
	for i := range weight {
		weight[i] = rng.Float32()*0.02 - 0.01
	}

	w := Conv2DWeight{Weight: weight, Bias: nil, InCh: inCh, OutCh: outCh, KH: kH, KW: kW}

	fast := conv2d(input, w, H, W, 1)
	naive := conv2dNaive(input, w, H, W, 1)

	if len(fast) != len(naive) {
		t.Fatalf("length mismatch: fast=%d naive=%d", len(fast), len(naive))
	}
	maxDiff := float64(0)
	for i := range fast {
		diff := math.Abs(float64(fast[i] - naive[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > 1e-4 {
		t.Errorf("1×1 conv max diff=%e (want < 1e-4)", maxDiff)
	}
	t.Logf("1×1 parity: %d outputs, max diff=%e", len(fast), maxDiff)
}

func TestConv2dSpeedup(t *testing.T) {
	// Benchmark-style test: verify the fast path is actually faster
	rng := rand.New(rand.NewSource(99))
	inCh, outCh, kH, kW := 256, 256, 3, 3
	H, W := 32, 32

	input := make([]float32, inCh*H*W)
	for i := range input {
		input[i] = rng.Float32()
	}
	weight := make([]float32, outCh*inCh*kH*kW)
	for i := range weight {
		weight[i] = rng.Float32() * 0.01
	}
	bias := make([]float32, outCh)

	w := Conv2DWeight{Weight: weight, Bias: bias, InCh: inCh, OutCh: outCh, KH: kH, KW: kW}

	// Warm up
	conv2d(input, w, H, W, 1)

	start := time.Now()
	conv2d(input, w, H, W, 1)
	fastTime := time.Since(start)

	start = time.Now()
	conv2dNaive(input, w, H, W, 1)
	naiveTime := time.Since(start)

	speedup := float64(naiveTime) / float64(fastTime)
	t.Logf("256ch 32×32: fast=%v naive=%v speedup=%.1fx", fastTime, naiveTime, speedup)

	if speedup < 1.0 {
		t.Logf("WARNING: fast path not faster (%.1fx). May improve at larger sizes.", speedup)
	}
}
