//go:build cgo && vulkan

package gpu

import (
	"os"
	"strings"
)

// Op-class elision for the CUDA batched prefill.
//
// Sync()-based timing is useless here: a cuStreamSynchronize on this WDDM box
// costs ~800 us, which dwarfs every op. Instead, DLGO_CUDA_SKIP_OPS names the op
// classes to *not* launch, and the drop in total prefill time is that class's
// exact contribution. The happy path is untouched (both knobs are read once and
// are off by default).
//
//	DLGO_CUDA_SKIP_OPS=down-gemm,gate-gemm,up-gemm,qkv-gemm,o-proj-gemm dlgo ...
var cbpEnabled = os.Getenv("DLGO_CUDA_PREFILL_PROFILE") == "1"

const (
	cbpQKV = iota
	cbpQKNorm
	cbpRoPE
	cbpKVStore
	cbpAttn
	cbpOProj
	cbpAddNorm
	cbpGate
	cbpUp
	cbpSwiGLU
	cbpDown
	cbpLogits
	cbpClasses
)

var cbpNames = [cbpClasses]string{
	"qkv-gemm",
	"qk-norm",
	"rope",
	"kv-store",
	"attention",
	"o-proj-gemm",
	"add+rmsnorm",
	"gate-gemm",
	"up-gemm",
	"swiglu",
	"down-gemm",
	"logits",
}

var cbpSkip [cbpClasses]bool

func init() {
	v := os.Getenv("DLGO_CUDA_SKIP_OPS")
	if v == "" {
		return
	}
	for _, name := range strings.Split(v, ",") {
		name = strings.TrimSpace(name)
		for i := 0; i < cbpClasses; i++ {
			if cbpNames[i] == name {
				cbpSkip[i] = true
			}
		}
	}
}

// cbpE runs f, unless its class is elided.
func cbpE(i int, f func() error) error {
	if cbpSkip[i] {
		return nil
	}
	return f()
}
