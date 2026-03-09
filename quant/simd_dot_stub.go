//go:build !amd64 || !cgo

package quant

import "math"

func SIMDDotProduct(data []byte, ggmlType uint32, x []float32, n int) float32 {
	return FusedDotProduct(data, ggmlType, x, n)
}

func SIMDDotProductMany(data []byte, ggmlType uint32, xFlat []float32, cols int, out []float32, nvecs int) {
	for i := 0; i < nvecs; i++ {
		out[i] = FusedDotProduct(data, ggmlType, xFlat[i*cols:(i+1)*cols], cols)
	}
}

func SIMDDotProductManyRows(data []byte, ggmlType uint32, xFlat []float32, cols int, out []float32, nrows int, bytesPerRow int, nvecs int) {
	for r := 0; r < nrows; r++ {
		rowData := data[r*bytesPerRow : (r+1)*bytesPerRow]
		SIMDDotProductMany(rowData, ggmlType, xFlat, cols, out[r*nvecs:(r+1)*nvecs], nvecs)
	}
}

func SIMDDotProductQ8K(data []byte, ggmlType uint32, xQ8K []byte, cols int) float32 {
	x := DequantizeQ8_K(xQ8K, cols)
	return FusedDotProduct(data, ggmlType, x, cols)
}

func SIMDDotBatchQ8K(data []byte, ggmlType uint32, xQ8K []byte, cols int, out []float32, nrows int, bytesPerRow int) {
	x := DequantizeQ8_K(xQ8K, cols)
	SIMDDotBatch(data, ggmlType, x, cols, out, nrows, bytesPerRow)
}

func SIMDDotProductQ8KManyRows(data []byte, ggmlType uint32, xQ8KFlat []byte, cols int, out []float32, nrows int, bytesPerRow int, nvecs int, xBytes int) {
	for r := 0; r < nrows; r++ {
		rowData := data[r*bytesPerRow : (r+1)*bytesPerRow]
		for v := 0; v < nvecs; v++ {
			out[r*nvecs+v] = SIMDDotProductQ8K(rowData, ggmlType, xQ8KFlat[v*xBytes:(v+1)*xBytes], cols)
		}
	}
}

func SIMDDotBatch(data []byte, ggmlType uint32, x []float32, cols int, out []float32, nrows int, bytesPerRow int) {
	if nrows <= 0 {
		return
	}
	for r := 0; r < nrows; r++ {
		rowData := data[r*bytesPerRow : (r+1)*bytesPerRow]
		out[r] = FusedDotProduct(rowData, ggmlType, x, cols)
	}
}

func SIMDDotF32(a, b []float32, n int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	limit := n - 7
	for ; i < limit; i += 8 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
		s4 += a[i+4] * b[i+4]
		s5 += a[i+5] * b[i+5]
		s6 += a[i+6] * b[i+6]
		s7 += a[i+7] * b[i+7]
	}
	sum := s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func SIMDDotF32Many(a []float32, xFlat []float32, cols int, out []float32, nvecs int) {
	for i := 0; i < nvecs; i++ {
		out[i] = SIMDDotF32(a, xFlat[i*cols:(i+1)*cols], cols)
	}
}

func SIMDDotF32Batch(aFlat []float32, x []float32, cols int, out []float32, nrows int) {
	for r := 0; r < nrows; r++ {
		out[r] = SIMDDotF32(aFlat[r*cols:(r+1)*cols], x, cols)
	}
}

func SIMDScaleAdd(out []float32, scale float32, src []float32, n int) {
	for i := 0; i < n; i++ {
		out[i] += scale * src[i]
	}
}

func SIMDSoftmax(x []float32) {
	// Fallback: use standard softmax
	if len(x) == 0 {
		return
	}
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float32
	for i, v := range x {
		e := float32(math.Exp(float64(v - maxVal)))
		x[i] = e
		sum += e
	}
	for i := range x {
		x[i] /= sum
	}
}

func SIMDSwiGLU(out, gate, up []float32, n int) {
	for i := 0; i < n; i++ {
		g := gate[i]
		out[i] = (g / (1.0 + float32(math.Exp(float64(-g))))) * up[i]
	}
}

func CausalAttnHead(q []float32, headDim int, kBase, vBase []float32, kvOffset, kvStride, seqLen int, scale float32, scores, out []float32) {
}

func HasCausalAttn() bool { return false }

func HasSIMDDot(ggmlType uint32) bool {
	switch ggmlType {
	case 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 34, 35, 39:
		return true
	default:
		return false
	}
}
