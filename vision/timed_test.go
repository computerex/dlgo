package vision

import (
	"os"
	"testing"
	"time"
)

// TestEncodeRealTimed times a single encode on a real mmproj (set DLGO_MMPROJ).
func TestEncodeRealTimed(t *testing.T) {
	path := os.Getenv("DLGO_MMPROJ")
	if path == "" {
		t.Skip("set DLGO_MMPROJ to an mmproj .gguf")
	}
	enc, err := LoadMmproj(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer enc.Close()
	img, _ := solidImage(200, enc.Config.ImageSize)
	t0 := time.Now()
	rows, err := enc.Encode(img)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	t.Logf("encode: %d tokens x %d dims in %s (pool workers=%d)", len(rows), len(rows[0]), time.Since(t0), enc.pool.NumWorkers())
}
