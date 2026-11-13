//go:build cgo && windows

package cuda

import (
	"math"
	"testing"
	"time"
)

// Measures the standalone Q4_K GEMV kernel throughput (GB/s) so we can tell
// whether decode is kernel-bandwidth-bound or launch-overhead-bound.
// Weight shape ~ a Qwen3-4B FFN gate (9728x2560) and the LM head (151936x2560).
func benchGEMV(t *testing.T, rows, cols int) {
	if err := Init(); err != nil {
		t.Skipf("init: %v", err)
	}
	nblk := cols / 256
	wbytes := make([]byte, rows*nblk*144)
	for i := range wbytes {
		wbytes[i] = byte(i*31 + 7)
	}
	// patch d/dmin to small finite values so dequant is finite
	for r := 0; r < rows; r++ {
		off := r * nblk * 144
		for b := 0; b < nblk; b++ {
			bo := off + b*144
			wbytes[bo+0] = 0x00
			wbytes[bo+1] = 0x2C // d ~ 0.06
			wbytes[bo+2] = 0x00
			wbytes[bo+3] = 0x20 // dmin ~ 0.03
		}
	}
	W, _ := Alloc(uint64(len(wbytes)))
	Upload(W, wbytes)
	xv := make([]float32, cols)
	xb := make([]byte, cols*4)
	for i := range xv {
		xv[i] = float32(math.Sin(float64(i) * 0.03))
		bits := math.Float32bits(xv[i])
		xb[i*4] = byte(bits)
		xb[i*4+1] = byte(bits >> 8)
		xb[i*4+2] = byte(bits >> 16)
		xb[i*4+3] = byte(bits >> 24)
	}
	X, _ := Alloc(uint64(len(xb)))
	Upload(X, xb)
	O, _ := Alloc(uint64(rows * 4))
	defer Free(W)
	defer Free(X)
	defer Free(O)

	q8, dx, sx, _ := ensureQ8(cols)

	runFloat := func() {
		gemv(kernels.GemvQ4K, W, X, O, rows, cols)
	}
	runBC := func() {
		launch(kernels.QuantizeX, cols/32, 32, [][]byte{
			ptrb(X), i32b(cols), ptrb(q8), ptrb(dx), ptrb(sx),
		})
		launch(kernels.GemvQ4KBc, rows, 256, [][]byte{
			ptrb(W), ptrb(q8), ptrb(dx), ptrb(sx), ptrb(O), i32b(rows), i32b(cols),
		})
	}

	measure := func(name string, f func()) {
		f()
		Sync()
		const N = 200
		start := time.Now()
		for i := 0; i < N; i++ {
			f()
		}
		Sync()
		el := time.Since(start)
		per := el.Seconds() / N
		gbs := float64(len(wbytes)) / per / 1e9
		t.Logf("rows=%d cols=%d %-6s  %.3f ms/call  %.1f GB/s", rows, cols, name, per*1e3, gbs)
	}
	measure("float", runFloat)
	measure("bc-dp4a", runBC)
}

func TestGEMVBandwidth(t *testing.T) {
	benchGEMV(t, 9728, 2560)   // FFN gate
	benchGEMV(t, 151936, 2560) // LM head
}

func benchGEMVQ6K(t *testing.T, rows, cols int) {
	if err := Init(); err != nil {
		t.Skipf("init: %v", err)
	}
	nblk := cols / 256
	wbytes := make([]byte, rows*nblk*210)
	for i := range wbytes {
		wbytes[i] = byte(i*17 + 3)
	}
	for r := 0; r < rows; r++ {
		off := r * nblk * 210
		for b := 0; b < nblk; b++ {
			bo := off + b*210
			wbytes[bo+208] = 0x00
			wbytes[bo+209] = 0x2C
		}
	}
	W, _ := Alloc(uint64(len(wbytes)))
	Upload(W, wbytes)
	xb := make([]byte, cols*4)
	for i := 0; i < cols; i++ {
		bits := math.Float32bits(float32(math.Sin(float64(i) * 0.03)))
		xb[i*4] = byte(bits)
		xb[i*4+1] = byte(bits >> 8)
		xb[i*4+2] = byte(bits >> 16)
		xb[i*4+3] = byte(bits >> 24)
	}
	X, _ := Alloc(uint64(len(xb)))
	Upload(X, xb)
	O, _ := Alloc(uint64(rows * 4))
	defer Free(W)
	defer Free(X)
	defer Free(O)
	run := func() { gemv(kernels.GemvQ6K, W, X, O, rows, cols) }
	run()
	Sync()
	const N = 200
	start := time.Now()
	for i := 0; i < N; i++ {
		run()
	}
	Sync()
	per := time.Since(start).Seconds() / N
	t.Logf("Q6K rows=%d cols=%d  %.3f ms  %.1f GB/s", rows, cols, per*1e3, float64(len(wbytes))/per/1e9)
}

func TestGEMVBandwidthQ6K(t *testing.T) {
	benchGEMVQ6K(t, 9728, 2560)   // FFN down/gate Q6_K
	benchGEMVQ6K(t, 151936, 2560) // LM head Q6_K
}

func benchAttention(t *testing.T, numHeads, numKVHeads, headDim, seqLen int) {
	if err := Init(); err != nil {
		t.Skipf("init: %v", err)
	}
	kvDim := numKVHeads * headDim
	qb := make([]byte, numHeads*headDim*4)
	for i := range qb {
		b := math.Float32bits(float32(math.Sin(float64(i) * 0.05)))
		qb[i] = byte(b)
	}
	kvbytes := make([]byte, seqLen*kvDim*2)
	for i := range kvbytes {
		kvbytes[i] = byte(i * 13)
	}
	Q, _ := Alloc(uint64(len(qb)))
	Upload(Q, qb)
	K, _ := Alloc(uint64(len(kvbytes)))
	Upload(K, kvbytes)
	V, _ := Alloc(uint64(len(kvbytes)))
	Upload(V, kvbytes)
	O, _ := Alloc(uint64(numHeads * headDim * 4))
	defer Free(Q)
	defer Free(K)
	defer Free(V)
	defer Free(O)
	run := func() { AttentionF16(Q, K, V, O, numHeads, numKVHeads, headDim, kvDim, seqLen, 0, 0.088, 0, 0) }
	run()
	Sync()
	const N = 200
	start := time.Now()
	for i := 0; i < N; i++ {
		run()
	}
	Sync()
	per := time.Since(start).Seconds() / N
	bytes := float64(seqLen*numKVHeads*headDim*2*2) // K + V, folded over GQA
	gbps := bytes / per / 1e9
	t.Logf("attn heads=%d kv=%d hd=%d seq=%d  %.4f ms/call  %.0f GB/s",
		numHeads, numKVHeads, headDim, seqLen, per*1e3, gbps)
}

func TestAttentionBandwidth(t *testing.T) {
	for _, n := range []int{512, 1024, 2048, 5000, 12000} {
		benchAttention(t, 32, 8, 128, n)
	}
}

// Measures the batched prefill attention kernel by itself, in TFLOP/s of causal
// attention work (QK^T + P*V). This is the number that has to move for prefill
// parity: the same work costs llama.cpp about 2.3 s at 16k, i.e. ~55 TFLOP/s.
//
// Causal FLOPs per head = 2 * 2 * headDim * (npos*(npos+1)/2), so
//   FLOPs = 2 * numHeads * headDim * npos * (npos+1).
func benchPrefillAttn(t *testing.T, numHeads, numKVHeads, headDim, npos, iters int) {
	if err := Init(); err != nil {
		t.Skipf("init: %v", err)
	}
	kvDim := numKVHeads * headDim
	qf := make([]float32, npos*numHeads*headDim)
	for i := range qf {
		qf[i] = float32(math.Sin(float64(i) * 0.001))
	}
	qb := make([]byte, len(qf)*4)
	for i, v := range qf {
		bits := math.Float32bits(v)
		qb[i*4] = byte(bits)
		qb[i*4+1] = byte(bits >> 8)
		qb[i*4+2] = byte(bits >> 16)
		qb[i*4+3] = byte(bits >> 24)
	}
	kvbytes := make([]byte, npos*kvDim*2)
	for i := range kvbytes {
		kvbytes[i] = byte(i*7 + 1)
	}
	Q, err := Alloc(uint64(len(qb)))
	if err != nil {
		t.Fatal(err)
	}
	defer Free(Q)
	K, err := Alloc(uint64(len(kvbytes)))
	if err != nil {
		t.Fatal(err)
	}
	defer Free(K)
	V, err := Alloc(uint64(len(kvbytes)))
	if err != nil {
		t.Fatal(err)
	}
	defer Free(V)
	O, err := Alloc(uint64(len(qb)))
	if err != nil {
		t.Fatal(err)
	}
	defer Free(O)
	if err := Upload(Q, qb); err != nil {
		t.Fatal(err)
	}
	if err := Upload(K, kvbytes); err != nil {
		t.Fatal(err)
	}
	if err := Upload(V, kvbytes); err != nil {
		t.Fatal(err)
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	run := func() {
		if err := AttnPrefillF16(O, Q, K, V, numHeads, numKVHeads, headDim, kvDim,
			0, npos, scale, 0); err != nil {
			t.Fatal(err)
		}
	}
	run()
	if err := Sync(); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	for i := 0; i < iters; i++ {
		run()
	}
	if err := Sync(); err != nil {
		t.Fatal(err)
	}
	per := time.Since(start).Seconds() / float64(iters)
	flops := 2.0 * float64(numHeads) * float64(headDim) * float64(npos) * float64(npos+1)
	t.Logf("prefill-attn heads=%d kv=%d hd=%d npos=%d  %.3f ms/call  %.1f TFLOP/s",
		numHeads, numKVHeads, headDim, npos, per*1e3, flops/per/1e12)
}

func TestPrefillAttnThroughput(t *testing.T) {
	for _, n := range []int{4096, 16384} {
		benchPrefillAttn(t, 32, 8, 128, n, 3)
	}
}

// TestLaunchOverhead measures the cost of dispatching one (trivial) kernel.
// Decode fires ~13 kernels per layer, so this dominates when it is ~10 us.
func TestLaunchOverhead(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("init: %v", err)
	}
	a, err := Alloc(16)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	defer Free(a)
	if err := Probe7(a, a, a, a, 1, 2, 3); err != nil {
		t.Fatalf("probe: %v", err)
	}
	Sync()
	const N = 2000
	start := time.Now()
	for i := 0; i < N; i++ {
		Probe7(a, a, a, a, 1, 2, 3)
	}
	Sync()
	t.Logf("launch overhead: %.2f us/launch (%d launches)", time.Since(start).Seconds()/N*1e6, N)
}
