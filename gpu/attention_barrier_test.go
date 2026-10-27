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

func TestAttentionBarrierParityGPU(t *testing.T) {
	if err := Init(); err != nil {
		t.Skip(err)
	}
	t.Cleanup(Shutdown)
	t.Cleanup(func() { SetAttentionBarrier(false) })
	for _, shape := range []struct{ hd, n, start int }{{128, 7, 1}, {256, 33, 249}, {256, 64, 4097}, {256, 16, 10753}} {
		t.Run(fmt.Sprintf("hd%d_n%d_start%d", shape.hd, shape.n, shape.start), func(t *testing.T) {
			const heads, kvheads = 16, 4
			hd, n, start := shape.hd, shape.n, shape.start
			rng := rand.New(rand.NewSource(11))
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
			for _, softcap := range []float32{0, 10} {
				run := func(on bool) []float32 {
					SetAttentionBarrier(on)
					BeginBatch()
					check(BatchAttentionF16(ob, qb, kb, vb, heads, kvheads, hd, kvheads*hd, start, float32(1/math.Sqrt(float64(hd))), softcap, n, 0))
					EndBatch()
					Sync()
					out := make([]float32, len(q))
					check(DownloadF32(ob, out))
					return out
				}
				a, b := run(false), run(true)
				maxAbs := float64(0)
				bits := 0
				for i := range a {
					if math.IsNaN(float64(b[i])) || math.IsInf(float64(b[i]), 0) {
						t.Fatal("nonfinite")
					}
					maxAbs = math.Max(maxAbs, math.Abs(float64(a[i]-b[i])))
					if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
						bits++
					}
				}
				if maxAbs > 1e-6 {
					t.Fatalf("maxAbs=%g changed=%d", maxAbs, bits)
				}
				// Alternate warmed variants to avoid first-pipeline compilation artifacts.
				bench := func(on bool) time.Duration {
					run(on)
					s := time.Now()
					for i := 0; i < 3; i++ {
						run(on)
					}
					return time.Since(s) / 3
				}
				off, on := bench(false), bench(true)
				t.Logf("softcap=%g maxAbs=%g changed=%d baseline=%s opt=%s speedup=%.3fx", softcap, maxAbs, bits, off, on, float64(off)/float64(on))
			}
		})
	}
}
