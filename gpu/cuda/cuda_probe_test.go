//go:build cgo && windows

package cuda

import "testing"

func TestProbes(t *testing.T) {
	mustInit(t)
	requireKernels(t)

	// Tiny KV store: kvDim=8, one row, linear. Should be trivially in-bounds.
	kk, _ := Alloc(8 * 4)
	vv, _ := Alloc(8 * 4)
	ko, _ := Alloc(4 * 4)
	vo, _ := Alloc(4 * 4)
	if err := KVStoreF16(kk, vv, ko, vo, 0, 8, 0); err != nil {
		t.Fatalf("KVStoreF16 tiny: %v", err)
	}
	t.Logf("tiny KVStoreF16 OK")

	// Probe7: trivial 7-arg kernel.
	a, _ := Alloc(16)
	b, _ := Alloc(16)
	c, _ := Alloc(16)
	d, _ := Alloc(16)
	if err := Probe7(a, b, c, d, 1, 2, 3); err != nil {
		t.Fatalf("Probe7 launch: %v", err)
	}
	raw := make([]byte, 4)
	if err := Download(raw, c); err != nil {
		t.Fatal(err)
	}
	v := uint32(raw[0]) | uint32(raw[1])<<8 | uint32(raw[2])<<16 | uint32(raw[3])<<24
	if v != 9 {
		t.Fatalf("Probe7 got %d want 9", v)
	}

	// HalfProbe: __half conversion kernel.
	in := []float32{1.0, 2.0, 0.5}
	din := upF32(t, in)
	dout, _ := Alloc(uint64(len(in) * 4))
	if err := HalfProbe(din, dout, len(in)); err != nil {
		t.Fatalf("HalfProbe launch: %v", err)
	}
	raw2 := make([]byte, len(in)*4)
	if err := Download(raw2, dout); err != nil {
		t.Fatal(err)
	}
	want := []uint32{0x3C00, 0x4000, 0x3800}
	for i := 0; i < len(in); i++ {
		g := uint32(raw2[i*4]) | uint32(raw2[i*4+1])<<8 | uint32(raw2[i*4+2])<<16 | uint32(raw2[i*4+3])<<24
		if g != want[i] {
			t.Fatalf("half[%d] got 0x%04x want 0x%04x", i, g, want[i])
		}
	}
}
