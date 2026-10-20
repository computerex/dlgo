//go:build !cgo || !vulkan

package diffusion

import "fmt"

// GenerateImageBatch requires GPU support compiled in. Without it, callers
// should fall back to calling GenerateImagePixels once per prompt.
func GenerateImageBatch(ditPath, vaePath, llmPath string, prompts []string, cfg ImageGenConfig) ([][]float32, error) {
	return nil, fmt.Errorf("batch rendering requires GPU support (build with -tags 'cgo vulkan')")
}
