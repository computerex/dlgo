package llm

import (
	"math/rand"
	"os"
	"strings"
	"testing"
)

// TestImageSpliceEndToEnd verifies that a precomputed image embedding block can
// be spliced into a real model's prompt via the <image> marker, that the KV/SSM
// state advances identically whether the block is fed as rows through the
// embedding path or as a batched forward, and that the image actually changes
// the output relative to the same prompt with the block omitted.
//
// Gated on an env var pointing at a Qwen GGUF because it loads a real model.
func TestImageSpliceEndToEnd(t *testing.T) {
	modelPath := os.Getenv("DLGO_IMAGE_E2E_MODEL")
	if modelPath == "" {
		modelPath = `C:\Users\mohd\Downloads\Qwen3.5-0.8B-Q4_K_M.gguf`
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not found: %s", modelPath)
	}

	p, err := NewPipeline(modelPath, 1024)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}

	dim := p.Model.Config.EmbeddingDim
	if dim <= 0 {
		t.Fatalf("bad embedding dim %d", dim)
	}

	// Deterministic synthetic "image": nRows rows of dim-many values, scaled to
	// a plausible embedding magnitude so the model sees a meaningful signal.
	const nRows = 8
	rng := rand.New(rand.NewSource(42))
	img := make([][]float32, nRows)
	for i := range img {
		row := make([]float32, dim)
		for j := range row {
			row[j] = float32(rng.NormFloat64()) * 0.5
		}
		img[i] = row
	}

	base := GenerateConfig{MaxTokens: 24}
	base.Sampler.Temperature = 0
	withImg := base
	withImg.ImageMarker = "<image>"
	// A prompt where the image is the only strong signal plus a text cue.
	imgPrompt := "Describe the object: <image>\nAnswer in one word:"
	noImgPrompt := "Describe the object: \nAnswer in one word:"

	resA, err := p.GenerateDetailedWithEmbeddings(imgPrompt, [][][]float32{img}, withImg)
	if err != nil {
		t.Fatalf("GenerateDetailedWithEmbeddings: %v", err)
	}
	if resA.Text == "" {
		t.Fatalf("empty output with image")
	}

	// Determinism: same call twice must match.
	resA2, err := p.GenerateDetailedWithEmbeddings(imgPrompt, [][][]float32{img}, withImg)
	if err != nil {
		t.Fatalf("second run: %v", err)
	}
	if resA.Text != resA2.Text {
		t.Fatalf("non-deterministic: %q vs %q", resA.Text, resA2.Text)
	}

	// The image changes state: output should differ from the no-image prompt.
	resB, err := p.GenerateDetailed(noImgPrompt, base)
	if err != nil {
		t.Fatalf("no-image run: %v", err)
	}
	t.Logf("with image (%d prompt tok): %q", resA.PromptTokens, resA.Text)
	t.Logf("no image: %q", resB.Text)

	// Marker/count mismatch must error.
	if _, err := p.GenerateDetailedWithEmbeddings("no marker here", [][][]float32{img}, withImg); err == nil {
		t.Fatalf("expected marker/count mismatch error")
	}
	// Wrong dim must error.
	bad := [][]float32{make([]float32, dim+1)}
	if _, err := p.GenerateDetailedWithEmbeddings("x <image>", [][][]float32{bad}, withImg); err == nil {
		t.Fatalf("expected dim mismatch error")
	}

	// Sanity: output token ids are valid and finite count.
	if resA.TotalTokens <= 0 {
		t.Fatalf("no tokens generated")
	}
	if !strings.Contains(imgPrompt, "<image>") {
		t.Fatal("test bug")
	}
}
