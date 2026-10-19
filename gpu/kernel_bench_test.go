//go:build cgo && vulkan

package gpu

import (
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/computerex/dlgo/quant"
)

// This file is a standalone microbenchmark harness for the DiT hot kernels.
// It isolates a single kernel from the rest of the graph so a kernel change can
// be measured in seconds rather than in a ~90 s end-to-end image generation.
//
// Run with:
//   go test -tags vulkan -run TestBenchAttentionDiT -v -timeout 600s ./gpu/
//   go test -tags vulkan -run TestBenchGemmQ6K -v -timeout 600s ./gpu/
//
// Environment overrides (all optional):
//   BENCH_SEQ   query sequence length       (default 4175)
//   BENCH_HEADS number of heads             (default 30)
//   BENCH_HD    head dim                    (default 128)
//   BENCH_REPS  timed repetitions           (default 20)
//   BENCH_GEMM_ROWS / BENCH_GEMM_COLS / BENCH_GEMM_NPOS

func benchEnvInt(name string, def int) int {
	if s := os.Getenv(name); s != "" {
		if v, err := strconv.Atoi(s); err == nil && v > 0 {
			return v
		}
	}
	return def
}

// benchKernel times a single-dispatch closure. It warms up, then reports the
// best (min) time over reps runs — min is the most robust statistic against the
// heavy background load on this machine.
func benchKernel(t *testing.T, name string, reps int, fn func()) time.Duration {
	t.Helper()
	// Warm-up (compiles pipelines, populates caches).
	for i := 0; i < 3; i++ {
		fn()
	}
	best := time.Duration(math.MaxInt64)
	var total time.Duration
	for i := 0; i < reps; i++ {
		start := time.Now()
		fn()
		d := time.Since(start)
		if d < best {
			best = d
		}
		total += d
	}
	t.Logf("%-28s best=%9v  avg=%9v  (n=%d)", name, best, total/time.Duration(reps), reps)
	return best
}

func TestBenchAttentionDiT(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	seq := benchEnvInt("BENCH_SEQ", 4175)
	heads := benchEnvInt("BENCH_HEADS", 30)
	hd := benchEnvInt("BENCH_HD", 128)
	reps := benchEnvInt("BENCH_REPS", 20)

	rng := rand.New(rand.NewSource(1))
	mk := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = (rng.Float32()*2 - 1) * 0.5
		}
		return s
	}
	alloc := func(n int) Buf {
		b := Alloc(uint64(n) * 4)
		if b == 0 {
			t.Fatal("alloc failed")
		}
		t.Cleanup(func() { Free(b) })
		return b
	}

	qd := seq * heads * hd
	qb := alloc(qd)
	kb := alloc(qd)
	vb := alloc(qd)
	ob := alloc(qd)
	if err := UploadF32(qb, mk(qd)); err != nil {
		t.Fatal(err)
	}
	if err := UploadF32(kb, mk(qd)); err != nil {
		t.Fatal(err)
	}
	if err := UploadF32(vb, mk(qd)); err != nil {
		t.Fatal(err)
	}

	scale := float32(1.0 / math.Sqrt(float64(hd)))
	fn := func() {
		BeginBatch()
		if err := AttentionFullF32(ob, qb, kb, vb, heads, heads, hd, hd, seq, scale); err != nil {
			t.Fatal(err)
		}
		EndBatch()
		Sync()
	}
	best := benchKernel(t, "attention_full", reps, fn)

	// Report achieved throughput (QK^T + softmax*V ≈ 2*seq^2*heads*hd*2 flops).
	flops := 2.0 * float64(seq) * float64(seq) * float64(heads) * float64(hd) * 2.0
	t.Logf("attention: seq=%d heads=%d hd=%d  %.2f GFLOP/call  %.2f TFLOPS",
		seq, heads, hd, flops/1e9, flops/1e12/best.Seconds())
}

func TestBenchGemmQ6K(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	rows := benchEnvInt("BENCH_GEMM_ROWS", 10240)
	cols := benchEnvInt("BENCH_GEMM_COLS", 3840)
	npos := benchEnvInt("BENCH_GEMM_NPOS", 4175)
	reps := benchEnvInt("BENCH_REPS", 20)
	if os.Getenv("BENCH_NOCOOPMAT") == "1" {
		SetNoCoopmatGemm(true)
		t.Cleanup(func() { SetNoCoopmatGemm(false) })
	}

	// Q6_K weight buffer: rows * (cols/256) blocks * 210 bytes.
	wbytes := rows * (cols / 256) * 210
	w := make([]byte, wbytes)
	rng := rand.New(rand.NewSource(7))
	rng.Read(w)

	x := make([]float32, npos*cols)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 0.05
	}

	alloc := func(n uint64) Buf {
		b := Alloc(n)
		if b == 0 {
			t.Fatal("alloc failed")
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	wb := alloc(uint64(wbytes))
	Upload(wb, w)
	xb := alloc(uint64(len(x)) * 4)
	if err := UploadF32(xb, x); err != nil {
		t.Fatal(err)
	}
	ob := alloc(uint64(npos) * uint64(rows) * 4)

	const qtypeQ6K = 14
	fn := func() {
		BeginBatch()
		if err := BatchMatVec(ob, wb, xb, rows, cols, npos, qtypeQ6K); err != nil {
			t.Fatal(err)
		}
		EndBatch()
		Sync()
	}
	best := benchKernel(t, "gemm_q6k", reps, fn)

	// Batched: many dispatches in one submit, to separate any fixed per-submit
	// (fence/dispatch) overhead from the kernel's own time.
	const nb = 8
	fnB := func() {
		BeginBatch()
		for i := 0; i < nb; i++ {
			if err := BatchMatVec(ob, wb, xb, rows, cols, npos, qtypeQ6K); err != nil {
				t.Fatal(err)
			}
		}
		EndBatch()
		Sync()
	}
	bestB := benchKernel(t, "gemm_q6k_x8", 5, fnB)
	t.Logf("gemm_q6k batched: %.3f ms/dispatch (vs %.3f ms single-submit)",
		float64(bestB.Microseconds())/float64(nb)/1000, float64(best.Microseconds())/1000)


	flops := 2.0 * float64(rows) * float64(cols) * float64(npos)
	t.Logf("gemm_q6k: rows=%d cols=%d npos=%d  %.2f GFLOP/call  %.2f TFLOPS",
		rows, cols, npos, flops/1e9, flops/1e12/best.Seconds())
}

func TestBenchGemmQ5K(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	rows := benchEnvInt("BENCH_GEMM_ROWS", 10240)
	cols := benchEnvInt("BENCH_GEMM_COLS", 3840)
	npos := benchEnvInt("BENCH_GEMM_NPOS", 4175)
	reps := benchEnvInt("BENCH_REPS", 20)

	// Q5_K weight buffer: rows * (cols/256) blocks * 176 bytes.
	wbytes := rows * (cols / 256) * 176
	w := make([]byte, wbytes)
	rng := rand.New(rand.NewSource(7))
	rng.Read(w)
	// Finite super-scales: float16(1.0) at offset 0, float16(0.0) at offset 2.
	for i := 0; i+176 <= len(w); i += 176 {
		w[i+0], w[i+1] = 0x00, 0x3C // d = 1.0
		w[i+2], w[i+3] = 0x00, 0x00 // dmin = 0.0
	}
	x := make([]float32, npos*cols)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 0.05
	}

	alloc := func(n uint64) Buf {
		b := Alloc(n)
		if b == 0 {
			t.Fatal("alloc failed")
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	wb := alloc(uint64(wbytes))
	Upload(wb, w)
	xb := alloc(uint64(len(x)) * 4)
	if err := UploadF32(xb, x); err != nil {
		t.Fatal(err)
	}
	ob := alloc(uint64(npos) * uint64(rows) * 4)

	const qtypeQ5K = 13
	fn := func() {
		BeginBatch()
		if err := BatchMatVec(ob, wb, xb, rows, cols, npos, qtypeQ5K); err != nil {
			t.Fatal(err)
		}
		EndBatch()
		Sync()
	}
	best := benchKernel(t, "gemm_q5k", reps, fn)

	flops := 2.0 * float64(rows) * float64(cols) * float64(npos)
	t.Logf("gemm_q5k: rows=%d cols=%d npos=%d  %.2f GFLOP/call  %.2f TFLOPS",
		rows, cols, npos, flops/1e9, flops/1e12/best.Seconds())
}

// TestBenchGemmAffine8 exercises the unified canonical-Affine8 GEMM: benchmarks
// it on the DiT hot shape and checks parity against a CPU f32 reference.
func TestBenchGemmAffine8(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	rows := benchEnvInt("BENCH_GEMM_ROWS", 10240)
	cols := benchEnvInt("BENCH_GEMM_COLS", 3840)
	npos := benchEnvInt("BENCH_GEMM_NPOS", 4175)
	reps := benchEnvInt("BENCH_REPS", 20)

	rng := rand.New(rand.NewSource(11))
	wf := make([]float32, rows*cols)
	for i := range wf {
		wf[i] = (rng.Float32()*2 - 1) * 0.05
	}
	w := quant.QuantizeAffine8(wf)
	x := make([]float32, npos*cols)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 0.05
	}

	alloc := func(n uint64) Buf {
		b := Alloc(n)
		if b == 0 {
			t.Fatal("alloc failed")
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	wb := alloc(uint64(len(w)))
	if err := Upload(wb, w); err != nil {
		t.Fatal(err)
	}
	xb := alloc(uint64(len(x)) * 4)
	if err := UploadF32(xb, x); err != nil {
		t.Fatal(err)
	}
	ob := alloc(uint64(npos) * uint64(rows) * 4)

	fn := func() {
		BeginBatch()
		if err := BatchMatVec(ob, wb, xb, rows, cols, npos, QuantTypeAffine8); err != nil {
			t.Fatal(err)
		}
		EndBatch()
		Sync()
	}
	best := benchKernel(t, "gemm_affine8", reps, fn)
	flops := 2.0 * float64(rows) * float64(cols) * float64(npos)
	t.Logf("gemm_affine8: rows=%d cols=%d npos=%d  %.2f GFLOP/call  %.2f TFLOPS",
		rows, cols, npos, flops/1e9, flops/1e12/best.Seconds())

	// Parity on a small shape vs an exact f32 CPU GEMM over the decoded weights.
	pRows, pCols, pN := 256, 256, 64
	pwf := make([]float32, pRows*pCols)
	for i := range pwf {
		pwf[i] = (rng.Float32()*2 - 1) * 0.1
	}
	pw := quant.QuantizeAffine8(pwf)
	pwd := quant.DequantizeAffine8(pw, pRows*pCols)
	px := make([]float32, pN*pCols)
	for i := range px {
		px[i] = (rng.Float32()*2 - 1) * 0.1
	}
	pwb := alloc(uint64(len(pw)))
	if err := Upload(pwb, pw); err != nil {
		t.Fatal(err)
	}
	pxb := alloc(uint64(len(px)) * 4)
	if err := UploadF32(pxb, px); err != nil {
		t.Fatal(err)
	}
	pob := alloc(uint64(pN) * uint64(pRows) * 4)
	BeginBatch()
	if err := BatchMatVec(pob, pwb, pxb, pRows, pCols, pN, QuantTypeAffine8); err != nil {
		t.Fatal(err)
	}
	EndBatch()
	Sync()
	got := make([]float32, pN*pRows)
	if err := DownloadF32(pob, got); err != nil {
		t.Fatal(err)
	}
	// CPU reference: out[n*pRows+m] = sum_k W[m,k] * x[n,k]
	ref := make([]float32, pN*pRows)
	for n := 0; n < pN; n++ {
		for m := 0; m < pRows; m++ {
			var s float32
			for k := 0; k < pCols; k++ {
				s += pwd[m*pCols+k] * px[n*pCols+k]
			}
			ref[n*pRows+m] = s
		}
	}
	var num, den float64
	for i := range ref {
		d := float64(got[i] - ref[i])
		num += d * d
		den += float64(ref[i]) * float64(ref[i])
	}
	rl := math.Sqrt(num / den)
	t.Logf("gemm_affine8 parity relL2 = %.3e", rl)
	if rl > 0.05 {
		t.Fatalf("gemm_affine8 parity relL2 %.3e too large", rl)
	}
}

// TestBenchGemmAffine8Stream measures the unified GEMM when the weights are NOT
// L2-resident: each rep uses a different weight buffer, so every load misses L2
// and streams from DRAM. This is the realistic per-step regime and the number
// the hot-L2 bench hides.
func TestBenchGemmAffine8Stream(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	rows := benchEnvInt("BENCH_GEMM_ROWS", 10240)
	cols := benchEnvInt("BENCH_GEMM_COLS", 3840)
	npos := benchEnvInt("BENCH_GEMM_NPOS", 4175)
	reps := benchEnvInt("BENCH_REPS", 20)
	nbuf := benchEnvInt("BENCH_STREAM_NBUF", 12)

	rng := rand.New(rand.NewSource(21))
	wf := make([]float32, rows*cols)
	for i := range wf {
		wf[i] = (rng.Float32()*2 - 1) * 0.05
	}
	w := quant.QuantizeAffine8(wf)
	x := make([]float32, npos*cols)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 0.05
	}

	alloc := func(n uint64) Buf {
		b := Alloc(n)
		if b == 0 {
			t.Fatal("alloc failed")
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	// nb identical weight buffers; cycling through them defeats L2 reuse.
	wbufs := make([]Buf, nbuf)
	for i := range wbufs {
		wbufs[i] = alloc(uint64(len(w)))
		if err := Upload(wbufs[i], w); err != nil {
			t.Fatal(err)
		}
	}
	xb := alloc(uint64(len(x)) * 4)
	if err := UploadF32(xb, x); err != nil {
		t.Fatal(err)
	}
	ob := alloc(uint64(npos) * uint64(rows) * 4)

	i := 0
	fn := func() {
		wb := wbufs[i%nbuf]
		i++
		BeginBatch()
		if err := BatchMatVec(ob, wb, xb, rows, cols, npos, QuantTypeAffine8); err != nil {
			t.Fatal(err)
		}
		EndBatch()
		Sync()
	}
	best := benchKernel(t, "gemm_affine8_stream", reps, fn)
	flops := 2.0 * float64(rows) * float64(cols) * float64(npos)
	t.Logf("gemm_affine8_stream (%d bufs): rows=%d cols=%d npos=%d  %.2f TFLOPS",
		nbuf, rows, cols, npos, flops/1e12/best.Seconds())
}

// TestAttentionF16Parity checks that the FP16 attention kernel agrees with the
// FP32 kernel on the DiT shape. The FP16 kernel uses f16 shared memory and f16
// output accumulators, so a small difference is expected; a large one is a bug.
func TestAttentionF16Parity(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)
	t.Cleanup(func() { SetAttentionF16(true) })

	seq := benchEnvInt("BENCH_SEQ", 4175)
	heads := benchEnvInt("BENCH_HEADS", 30)
	hd := benchEnvInt("BENCH_HD", 128)

	rng := rand.New(rand.NewSource(5))
	mk := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = (rng.Float32()*2 - 1) * 0.5
		}
		return s
	}
	alloc := func(n int) Buf {
		b := Alloc(uint64(n) * 4)
		if b == 0 {
			t.Fatal("alloc failed")
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	qd := seq * heads * hd
	qb, kb, vb, ob := alloc(qd), alloc(qd), alloc(qd), alloc(qd)
	if err := UploadF32(qb, mk(qd)); err != nil {
		t.Fatal(err)
	}
	if err := UploadF32(kb, mk(qd)); err != nil {
		t.Fatal(err)
	}
	if err := UploadF32(vb, mk(qd)); err != nil {
		t.Fatal(err)
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	run := func() []float32 {
		BeginBatch()
		if err := AttentionFullF32(ob, qb, kb, vb, heads, heads, hd, hd, seq, scale); err != nil {
			t.Fatal(err)
		}
		EndBatch()
		Sync()
		out := make([]float32, qd)
		if err := DownloadF32(ob, out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	SetAttentionF16(false)
	ref := run()
	SetAttentionF16(true)
	got := run()

	maxAbs, maxRef, ident := 0.0, 0.0, true
	nonfinite := 0
	for i := range ref {
		if math.Abs(float64(ref[i])) > maxRef {
			maxRef = math.Abs(float64(ref[i]))
		}
		if ref[i] != got[i] {
			ident = false
		}
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			nonfinite++
		}
		if d := math.Abs(float64(ref[i] - got[i])); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("parity: maxAbs=%.3e maxRef=%.3e relAbs=%.3e bitIdentical=%v nonfinite=%d/%d",
		maxAbs, maxRef, maxAbs/maxRef, ident, nonfinite, len(ref))
	if ident {
		t.Fatalf("FP16 and FP32 outputs are bit-identical — the toggle likely did not take effect")
	}
	if maxAbs/maxRef > 1e-2 {
		t.Fatalf("FP16 attention deviates too far: relAbs=%.3e", maxAbs/maxRef)
	}
}

// TestAttentionCoopmatParity cross-checks the tensor-core flash-attention kernel
// against the independent FP32 SIMT attention path. Uses head_dim=128 (required
// by the coopmat kernel) and seq intentionally NOT a multiple of the 64-row /
// 32-key tiles so the partial-tail masking is exercised too.
func TestAttentionCoopmatParity(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)
	t.Cleanup(func() { SetAttentionCoopmat(false); SetAttentionF16(true) })

	seq := benchEnvInt("BENCH_SEQ", 130)
	heads := benchEnvInt("BENCH_HEADS", 2)
	hd := 128

	rng := rand.New(rand.NewSource(7))
	mk := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = (rng.Float32()*2 - 1) * 0.5
		}
		return s
	}
	alloc := func(n int) Buf {
		b := Alloc(uint64(n) * 4)
		if b == 0 {
			t.Fatal("alloc failed")
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	qd := seq * heads * hd
	qb, kb, vb, ob := alloc(qd), alloc(qd), alloc(qd), alloc(qd)
	if err := UploadF32(qb, mk(qd)); err != nil {
		t.Fatal(err)
	}
	if err := UploadF32(kb, mk(qd)); err != nil {
		t.Fatal(err)
	}
	if err := UploadF32(vb, mk(qd)); err != nil {
		t.Fatal(err)
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	run := func() []float32 {
		BeginBatch()
		if err := AttentionFullF32(ob, qb, kb, vb, heads, heads, hd, hd, seq, scale); err != nil {
			t.Fatal(err)
		}
		EndBatch()
		Sync()
		out := make([]float32, qd)
		if err := DownloadF32(ob, out); err != nil {
			t.Fatal(err)
		}
		return out
	}

	SetAttentionCoopmat(false)
	SetAttentionF16(false)
	ref := run()
	SetAttentionCoopmat(true)
	got := run()

	maxAbs, maxRef := 0.0, 0.0
	nonfinite := 0
	for i := range ref {
		if a := math.Abs(float64(ref[i])); a > maxRef {
			maxRef = a
		}
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			nonfinite++
		}
		if d := math.Abs(float64(ref[i] - got[i])); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("coopmat attention parity (seq=%d heads=%d): maxAbs=%.3e maxRef=%.3e relAbs=%.3e nonfinite=%d/%d",
		seq, heads, maxAbs, maxRef, maxAbs/maxRef, nonfinite, len(ref))
	if nonfinite > 0 {
		t.Fatalf("coopmat attention produced %d non-finite outputs", nonfinite)
	}
	if maxAbs/maxRef > 1e-2 {
		t.Fatalf("coopmat attention deviates too far: relAbs=%.3e", maxAbs/maxRef)
	}
}

// TestGemmQ6KParity cross-checks the coopmat Q6_K GEMM (default) against the
// independent FP32 tiled path on the same random weights/activations. Both
// dequantize the same Q6_K blocks but with separate shader code, so a bug in the
// (recently vectorized) coopmat dequant shows up as a large relL2 error.
func TestGemmQ6KParity(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)
	t.Cleanup(func() { SetNoCoopmatGemm(false) })

	rows := benchEnvInt("BENCH_GEMM_ROWS", 512)
	cols := benchEnvInt("BENCH_GEMM_COLS", 768)
	npos := benchEnvInt("BENCH_GEMM_NPOS", 128)

	wbytes := rows * (cols / 256) * 210
	w := make([]byte, wbytes)
	rng := rand.New(rand.NewSource(11))
	rng.Read(w)
	// Give every Q6_K super-block a finite, modest super-scale. Uniform random
	// bytes can decode to a float16 Inf/NaN in the trailing `d` field (offset
	// 208), which would poison BOTH GEMM paths and make the comparison
	// meaningless. float16(1.0) = 0x3C00, stored little-endian.
	for i := 0; i+210 <= len(w); i += 210 {
		w[i+208] = 0x00
		w[i+209] = 0x3C
	}
	x := make([]float32, npos*cols)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 0.1
	}

	alloc := func(n uint64) Buf {
		b := Alloc(n)
		if b == 0 {
			t.Fatal("alloc failed")
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	wb := alloc(uint64(wbytes))
	Upload(wb, w)
	xb := alloc(uint64(len(x)) * 4)
	if err := UploadF32(xb, x); err != nil {
		t.Fatal(err)
	}
	ob := alloc(uint64(npos) * uint64(rows) * 4)

	const qtypeQ6K = 14
	run := func() []float32 {
		BeginBatch()
		if err := BatchMatVec(ob, wb, xb, rows, cols, npos, qtypeQ6K); err != nil {
			t.Fatal(err)
		}
		EndBatch()
		Sync()
		out := make([]float32, npos*rows)
		if err := DownloadF32(ob, out); err != nil {
			t.Fatal(err)
		}
		return out
	}

	SetNoCoopmatGemm(false)
	got := run()
	SetNoCoopmatGemm(true)
	ref := run()
	SetNoCoopmatGemm(false)

	var num, den, maxAbs float64
	nonfinite := 0
	for i := range ref {
		d := float64(got[i]) - float64(ref[i])
		num += d * d
		den += float64(ref[i]) * float64(ref[i])
		if math.Abs(d) > maxAbs {
			maxAbs = math.Abs(d)
		}
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			nonfinite++
		}
	}
	relL2 := math.Sqrt(num) / math.Sqrt(den)
	t.Logf("q6k parity: relL2=%.3e maxAbs=%.3e nonfinite=%d/%d", relL2, maxAbs, nonfinite, len(ref))
	if nonfinite > 0 {
		t.Fatalf("non-finite outputs: %d", nonfinite)
	}
	if relL2 > 5e-3 {
		t.Fatalf("coopmat vs f32 relL2 too large: %.3e", relL2)
	}
}

// TestMatVecAffine8Parity validates the unified single-token (decode) matvec over
// canonical Affine8 weights (matvec_affine8) against an exact f32 CPU reference
// built from the dequantized weights. Shapes are chosen so cols/32 (blocks per
// row) is not a multiple of 128 and rows/2 has a tail, exercising both striding
// and partial-row masking.
func TestMatVecAffine8Parity(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	rows := benchEnvInt("BENCH_MV_ROWS", 130)
	cols := benchEnvInt("BENCH_MV_COLS", 96) // 3 blocks/row
	reps := benchEnvInt("BENCH_REPS", 50)

	rng := rand.New(rand.NewSource(3))
	wf := make([]float32, rows*cols)
	for i := range wf {
		wf[i] = (rng.Float32()*2 - 1) * 0.1
	}
	w := quant.QuantizeAffine8(wf)
	wd := quant.DequantizeAffine8(w, rows*cols)
	x := make([]float32, cols)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 0.5
	}

	alloc := func(n uint64) Buf {
		b := Alloc(n)
		if b == 0 {
			t.Fatal("alloc failed")
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	wb := alloc(uint64(len(w)))
	if err := Upload(wb, w); err != nil {
		t.Fatal(err)
	}
	xb := alloc(uint64(len(x)) * 4)
	if err := UploadF32(xb, x); err != nil {
		t.Fatal(err)
	}
	ob := alloc(uint64(rows) * 4)

	run := func() []float32 {
		BeginBatch()
		if err := MatVec(ob, wb, xb, rows, cols, QuantTypeAffine8); err != nil {
			t.Fatal(err)
		}
		EndBatch()
		Sync()
		out := make([]float32, rows)
		if err := DownloadF32(ob, out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	got := run()

	ref := make([]float32, rows)
	for m := 0; m < rows; m++ {
		var s float32
		for k := 0; k < cols; k++ {
			s += wd[m*cols+k] * x[k]
		}
		ref[m] = s
	}
	var num, den, maxAbs float64
	nonfinite := 0
	for i := range ref {
		d := float64(got[i]) - float64(ref[i])
		num += d * d
		den += float64(ref[i]) * float64(ref[i])
		if math.Abs(d) > maxAbs {
			maxAbs = math.Abs(d)
		}
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			nonfinite++
		}
	}
	relL2 := math.Sqrt(num / den)
	t.Logf("matvec_affine8 parity: relL2=%.3e maxAbs=%.3e nonfinite=%d/%d", relL2, maxAbs, nonfinite, rows)
	if nonfinite > 0 {
		t.Fatalf("non-finite outputs: %d", nonfinite)
	}
	if relL2 > 1e-4 {
		t.Fatalf("matvec_affine8 relL2 too large: %.3e", relL2)
	}

	// Decode-shape throughput (isolated).
	dRows, dCols := benchEnvInt("BENCH_MV_DROWS", 4096), benchEnvInt("BENCH_MV_DCOLS", 4096)
	dwf := make([]float32, dRows*dCols)
	for i := range dwf {
		dwf[i] = (rng.Float32()*2 - 1) * 0.05
	}
	dw := quant.QuantizeAffine8(dwf)
	dwb := alloc(uint64(len(dw)))
	if err := Upload(dwb, dw); err != nil {
		t.Fatal(err)
	}
	dx := make([]float32, dCols)
	for i := range dx {
		dx[i] = (rng.Float32()*2 - 1) * 0.5
	}
	dxb := alloc(uint64(len(dx)) * 4)
	if err := UploadF32(dxb, dx); err != nil {
		t.Fatal(err)
	}
	dob := alloc(uint64(dRows) * 4)
	best := benchKernel(t, "matvec_affine8", reps, func() {
		BeginBatch()
		if err := MatVec(dob, dwb, dxb, dRows, dCols, QuantTypeAffine8); err != nil {
			t.Fatal(err)
		}
		EndBatch()
		Sync()
	})
	gb := float64(len(dw)) / 1e9
	t.Logf("matvec_affine8 decode-shape: rows=%d cols=%d  %v  %.1f GB/s (weight bytes %.3f GB)",
		dRows, dCols, best, gb/best.Seconds(), gb)
}

// TestAffine4Parity validates the canonical Affine4 family: the single-token
// decode matvec (matvec_affine4) and the batched prefill GEMM
// (gemm_affine4_coopmat), both against exact f32 CPU references built from the
// dequantized weights. Shapes exercise partial tiles in m, k(block), and n.
func TestAffine4Parity(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	rows, cols, npos := 130, 96, 8
	rng := rand.New(rand.NewSource(9))
	wf := make([]float32, rows*cols)
	for i := range wf {
		wf[i] = (rng.Float32()*2 - 1) * 0.1
	}
	w := quant.QuantizeAffine4(wf)
	wd := quant.DequantizeAffine4(w, rows*cols)
	x := make([]float32, npos*cols)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 0.5
	}

	alloc := func(n uint64) Buf {
		b := Alloc(n)
		if b == 0 {
			t.Fatal("alloc failed")
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	wb := alloc(uint64(len(w)))
	if err := Upload(wb, w); err != nil {
		t.Fatal(err)
	}
	xb := alloc(uint64(len(x)) * 4)
	if err := UploadF32(xb, x); err != nil {
		t.Fatal(err)
	}

	// --- decode: matvec (npos=1) ---
	ob1 := alloc(uint64(rows) * 4)
	BeginBatch()
	if err := MatVec(ob1, wb, xb, rows, cols, QuantTypeAffine4); err != nil {
		t.Fatal(err)
	}
	EndBatch()
	Sync()
	got1 := make([]float32, rows)
	if err := DownloadF32(ob1, got1); err != nil {
		t.Fatal(err)
	}
	ref1 := make([]float32, rows)
	for m := 0; m < rows; m++ {
		var s float32
		for k := 0; k < cols; k++ {
			s += wd[m*cols+k] * x[k]
		}
		ref1[m] = s
	}
	if rl, nf := relL2(ref1, got1); nf > 0 || rl > 1e-4 {
		t.Fatalf("affine4 matvec: relL2=%.3e nonfinite=%d", rl, nf)
	} else {
		t.Logf("affine4 matvec parity relL2=%.3e", rl)
	}

	// --- prefill: batched GEMM (npos=8) ---
	ob8 := alloc(uint64(rows) * uint64(npos) * 4)
	BeginBatch()
	if err := BatchMatVec(ob8, wb, xb, rows, cols, npos, QuantTypeAffine4); err != nil {
		t.Fatal(err)
	}
	EndBatch()
	Sync()
	got8 := make([]float32, npos*rows)
	if err := DownloadF32(ob8, got8); err != nil {
		t.Fatal(err)
	}
	ref8 := make([]float32, npos*rows)
	for p := 0; p < npos; p++ {
		for m := 0; m < rows; m++ {
			var s float32
			for k := 0; k < cols; k++ {
				s += wd[m*cols+k] * x[p*cols+k]
			}
			ref8[p*rows+m] = s
		}
	}
	if rl, nf := relL2(ref8, got8); nf > 0 || rl > 5e-3 {
		t.Fatalf("affine4 gemm: relL2=%.3e nonfinite=%d", rl, nf)
	} else {
		t.Logf("affine4 gemm parity relL2=%.3e", rl)
	}
}

func relL2(ref, got []float32) (float64, int) {
	var num, den float64
	nf := 0
	for i := range ref {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			nf++
		}
		d := float64(got[i]) - float64(ref[i])
		num += d * d
		den += float64(ref[i]) * float64(ref[i])
	}
	return math.Sqrt(num / den), nf
}

// TestMoEMatVecAffineParity validates the fused MoE matvec over the canonical
// formats (matvec_affine4_moe / matvec_affine8_moe): a single dispatch computes
// n_used experts selected by an on-GPU index buffer, with weights gathered from a
// packed expert tensor. Covers both shared_input modes (gate/up use a shared
// activation; down uses a per-slot hidden).
func TestMoEMatVecAffineParity(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	const E = 4
	rows, cols := 64, 128
	nUsed := 2
	idx := []float32{1, 3}

	check := func(qt uint32, enc func([]float32) []byte, dec func([]byte, int) []float32,
		name string, sharedInput int) {
		rng := rand.New(rand.NewSource(5))
		wf := make([]float32, E*rows*cols)
		for i := range wf {
			wf[i] = (rng.Float32()*2 - 1) * 0.1
		}
		w := enc(wf)
		wd := dec(w, E*rows*cols)

		var x []float32
		if sharedInput != 0 {
			x = make([]float32, cols)
			for i := range x {
				x[i] = (rng.Float32()*2 - 1) * 0.5
			}
		} else {
			x = make([]float32, nUsed*cols)
			for i := range x {
				x[i] = (rng.Float32()*2 - 1) * 0.5
			}
		}

		alloc := func(n uint64) Buf {
			b := Alloc(n)
			if b == 0 {
				t.Fatal("alloc failed")
			}
			t.Cleanup(func() { Free(b) })
			return b
		}
		wb := alloc(uint64(len(w)))
		if err := Upload(wb, w); err != nil {
			t.Fatal(err)
		}
		xb := alloc(uint64(len(x)) * 4)
		if err := UploadF32(xb, x); err != nil {
			t.Fatal(err)
		}
		ib := alloc(uint64(nUsed) * 4)
		if err := UploadF32(ib, idx); err != nil {
			t.Fatal(err)
		}
		ob := alloc(uint64(nUsed*rows) * 4)

		stride := rows * (cols / 32) // affine blocks per expert
		BeginBatch()
		if err := MoEMatVecNative(ob, wb, xb, ib, rows, cols, qt, stride, 0, sharedInput, nUsed); err != nil {
			t.Fatal(err)
		}
		EndBatch()
		Sync()
		got := make([]float32, nUsed*rows)
		if err := DownloadF32(ob, got); err != nil {
			t.Fatal(err)
		}

		ref := make([]float32, nUsed*rows)
		for s := 0; s < nUsed; s++ {
			e := int(idx[s])
			for r := 0; r < rows; r++ {
				var sum float32
				for k := 0; k < cols; k++ {
					xv := x[k]
					if sharedInput == 0 {
						xv = x[s*cols+k]
					}
					sum += wd[e*rows*cols+r*cols+k] * xv
				}
				ref[s*rows+r] = sum
			}
		}
		rl, nf := relL2(ref, got)
		t.Logf("%s shared_input=%d: relL2=%.3e nonfinite=%d", name, sharedInput, rl, nf)
		if nf > 0 || rl > 1e-4 {
			t.Fatalf("%s shared_input=%d: relL2=%.3e nonfinite=%d", name, sharedInput, rl, nf)
		}
	}
	check(QuantTypeAffine4, quant.QuantizeAffine4, quant.DequantizeAffine4, "affine4", 1)
	check(QuantTypeAffine4, quant.QuantizeAffine4, quant.DequantizeAffine4, "affine4", 0)
	check(QuantTypeAffine8, quant.QuantizeAffine8, quant.DequantizeAffine8, "affine8", 1)
	check(QuantTypeAffine8, quant.QuantizeAffine8, quant.DequantizeAffine8, "affine8", 0)
}

// TestBenchAffineStream measures the unified affine kernels with weights that are
// NOT L2-resident: a pool of buffers is cycled every rep so each dispatch streams
// from DRAM. This is the realistic decode/prefill regime (isolated single-buffer
// benches are L2-resident and non-predictive). Reports effective GB/s + TFLOPS.
func TestBenchAffineStream(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)

	mvRows := benchEnvInt("BENCH_MV_ROWS", 4096)
	mvCols := benchEnvInt("BENCH_MV_COLS", 4096)
	gRows := benchEnvInt("BENCH_GEMM_ROWS", 10240)
	gCols := benchEnvInt("BENCH_GEMM_COLS", 3840)
	npos := benchEnvInt("BENCH_GEMM_NPOS", 512)
	nbuf := benchEnvInt("BENCH_STREAM_NBUF", 12)
	streamBatch := benchEnvInt("BENCH_STREAM_BATCH", 16) // dispatches per submit

	alloc := func(n uint64) Buf {
		b := Alloc(n)
		if b == 0 {
			t.Fatal("alloc failed")
		}
		t.Cleanup(func() { Free(b) })
		return b
	}
	timeit := func(name string, weightBytes int, flops float64, reps, div int, fn func()) {
		for k := 0; k < 4; k++ {
			fn()
		}
		start := time.Now()
		for k := 0; k < reps; k++ {
			fn()
		}
		per := time.Since(start) / time.Duration(reps*div)
		sec := per.Seconds()
		if weightBytes > 0 && flops > 0 {
			t.Logf("%-16s %10v/call  %6.1f GB/s  %6.1f TFLOPS", name, per,
				float64(weightBytes)/1e9/sec, flops/sec/1e12)
		} else if weightBytes > 0 {
			t.Logf("%-16s %10v/call  %6.1f GB/s", name, per, float64(weightBytes)/1e9/sec)
		} else {
			t.Logf("%-16s %10v/call  %6.1f TFLOPS", name, per, flops/sec/1e12)
		}
	}

	rng := rand.New(rand.NewSource(41))

	// ---- single-token decode matvec (npos=1) ----
	mvX := make([]float32, mvCols)
	for i := range mvX {
		mvX[i] = (rng.Float32()*2 - 1) * 0.5
	}
	mvxb := alloc(uint64(len(mvX)) * 4)
	if err := UploadF32(mvxb, mvX); err != nil {
		t.Fatal(err)
	}
	mvob := alloc(uint64(mvRows) * 4)
	mvwf := make([]float32, mvRows*mvCols)
	for i := range mvwf {
		mvwf[i] = (rng.Float32()*2 - 1) * 0.05
	}
	for _, f := range []struct {
		name string
		enc  []byte
		qt   uint32
	}{
		{"matvec_affine4", quant.QuantizeAffine4(mvwf), QuantTypeAffine4},
		{"matvec_affine8", quant.QuantizeAffine8(mvwf), QuantTypeAffine8},
	} {
		bufs := make([]Buf, nbuf)
		for i := range bufs {
			bufs[i] = alloc(uint64(len(f.enc)))
			if err := Upload(bufs[i], f.enc); err != nil {
				t.Fatal(err)
			}
		}
		i := 0
		fn := func() {
			BeginBatch()
			for b := 0; b < streamBatch; b++ {
				if err := MatVec(mvob, bufs[(i+b)%nbuf], mvxb, mvRows, mvCols, f.qt); err != nil {
					t.Fatal(err)
				}
			}
			EndBatch()
			Sync()
			i += streamBatch
		}
		timeit(f.name, len(f.enc), 0, 60, streamBatch, fn)
	}

	// ---- batched prefill GEMM ----
	gx := make([]float32, npos*gCols)
	for i := range gx {
		gx[i] = (rng.Float32()*2 - 1) * 0.05
	}
	gxb := alloc(uint64(len(gx)) * 4)
	if err := UploadF32(gxb, gx); err != nil {
		t.Fatal(err)
	}
	gob := alloc(uint64(npos*gRows) * 4)
	gwf := make([]float32, gRows*gCols)
	for i := range gwf {
		gwf[i] = (rng.Float32()*2 - 1) * 0.05
	}
	flops := 2.0 * float64(gRows) * float64(gCols) * float64(npos)
	for _, f := range []struct {
		name string
		enc  []byte
		qt   uint32
	}{
		{"gemm_affine4", quant.QuantizeAffine4(gwf), QuantTypeAffine4},
		{"gemm_affine8", quant.QuantizeAffine8(gwf), QuantTypeAffine8},
	} {
		bufs := make([]Buf, 4)
		for i := range bufs {
			bufs[i] = alloc(uint64(len(f.enc)))
			if err := Upload(bufs[i], f.enc); err != nil {
				t.Fatal(err)
			}
		}
		i := 0
		fn := func() {
			BeginBatch()
			for b := 0; b < streamBatch; b++ {
				if err := BatchMatVec(gob, bufs[(i+b)%4], gxb, gRows, gCols, npos, f.qt); err != nil {
					t.Fatal(err)
				}
			}
			EndBatch()
			Sync()
			i += streamBatch
		}
		timeit(f.name, len(f.enc), flops, 10, streamBatch, fn)
	}
}

