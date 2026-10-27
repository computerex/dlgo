//go:build cgo && vulkan

package gpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestAttentionKV4ParityGPU compares the KV-grouped attention (4x less KV
// traffic) against the default per-head kernel at Qwen3.5-9B shapes. Dots and
// PV chains are bit-identical by construction; softmax sums use the same tree,
// so the tolerance is tight. Reports isolated kernel timing.
func TestAttentionKV4ParityGPU(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)
	t.Cleanup(func() { SetAttentionKV4(false) })
	for _, shape := range []struct{ hd, n, start int }{{256, 64, 4097}, {256, 16, 10753}} {
		t.Run(fmt.Sprintf("hd%d_n%d_start%d", shape.hd, shape.n, shape.start), func(t *testing.T) {
			const heads, kvheads = 16, 4
			hd, n, start := shape.hd, shape.n, shape.start
			rng := rand.New(rand.NewSource(31))
			alloc := func(bytes int) Buf {
				b := Alloc(uint64(bytes))
				if b == 0 {
					t.Fatal("alloc")
				}
				t.Cleanup(func() { Free(b) })
				return b
			}
			check := func(err error) {
				t.Helper()
				if err != nil {
					t.Fatal(err)
				}
			}
			packed := func(count int) []byte {
				b := make([]byte, count*2)
				for i := 0; i < count; i++ {
					binary.LittleEndian.PutUint16(b[2*i:], packF32ToF16((rng.Float32()*2-1)*.7))
				}
				return b
			}
			seq := start + n - 1
			kb, vb := alloc(seq*kvheads*hd*2), alloc(seq*kvheads*hd*2)
			check(Upload(kb, packed(seq*kvheads*hd)))
			check(Upload(vb, packed(seq*kvheads*hd)))
			q := make([]float32, n*heads*hd)
			for i := range q {
				q[i] = (rng.Float32()*2 - 1) * .8
			}
			qb, ob := alloc(len(q)*4), alloc(len(q)*4)
			check(UploadF32(qb, q))
			run := func(kv4 bool) []float32 {
				SetAttentionKV4(kv4)
				BeginBatch()
				check(BatchAttentionF16(ob, qb, kb, vb, heads, kvheads, hd, kvheads*hd, start,
					float32(1/math.Sqrt(float64(hd))), 0, n, 0))
				EndBatch()
				Sync()
				out := make([]float32, len(q))
				check(DownloadF32(ob, out))
				return out
			}
			base := run(false)
			// Warm both pipelines before timing.
			run(false)
			run(true)
			bench := func(kv4 bool, reps int) time.Duration {
				start := time.Now()
				for i := 0; i < reps; i++ {
					run(kv4)
				}
				return time.Since(start) / time.Duration(reps)
			}
			opt := run(true)
			maxAbs := 0.0
			for i := range base {
				if math.IsNaN(float64(opt[i])) || math.IsInf(float64(opt[i]), 0) {
					t.Fatalf("nonfinite at %d", i)
				}
				if a := math.Abs(float64(base[i] - opt[i])); a > maxAbs {
					maxAbs = a
				}
			}
			if maxAbs > 0 {
				t.Fatalf("expected bit-identical output, maxAbs=%g", maxAbs)
			}
			off, on := bench(false, 5), bench(true, 5)
			t.Logf("maxAbs=%g baseline=%s kv4=%s speedup=%.3fx", maxAbs, off, on, float64(off)/float64(on))
		})
	}
}
