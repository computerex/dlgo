package vision

import (
	"os"
	"testing"
)

// BenchmarkEncodeReal runs the encoder on a real mmproj if available (set
// DLGO_MMPROJ). It reports the per-encode cost so regressions in the vision
// transformer speed are visible.
func BenchmarkEncodeReal(b *testing.B) {
	path := os.Getenv("DLGO_MMPROJ")
	if path == "" {
		b.Skip("set DLGO_MMPROJ to an mmproj .gguf")
	}
	enc, err := LoadMmproj(path)
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	defer enc.Close()
	img, _ := solidImage(200, enc.Config.ImageSize)
	if _, err := enc.Encode(img); err != nil {
		b.Fatalf("encode: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := enc.Encode(img); err != nil {
			b.Fatalf("encode: %v", err)
		}
	}
}
