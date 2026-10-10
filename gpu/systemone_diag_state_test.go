//go:build cgo && vulkan

package gpu

import (
	"bytes"
	"math"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// gpuDownloadState captures every persistent state buffer (per-layer SSM state,
// conv history, KV rows [0, npos)) as raw bytes for comparison.
func gpuDownloadState(p *GpuPipeline, npos int) [][]byte {
	cfg := p.CPUModel.Config
	numHeads := cfg.SSMTimeStepRank
	numKVGroups := cfg.SSMGroupCount
	if numKVGroups <= 0 {
		numKVGroups = numHeads
	}
	headVDim := cfg.SSMInnerSize / numHeads
	headKDim := cfg.SSMStateSize
	keyDim := numKVGroups * headKDim
	qkvDim := keyDim*2 + numHeads*headVDim
	convK := cfg.SSMConvKernel
	kvBytes := npos * p.KVCache.KVDim * p.KVCache.BytesPerElem

	var blobs [][]byte
	for l := 0; l < cfg.NumLayers; l++ {
		gl := &p.GpuModel.Layers[l]
		if gl.SSMState != 0 {
			s := make([]float32, numHeads*headKDim*headVDim)
			DownloadF32(gl.SSMState, s)
			blobs = append(blobs, f32Bytes(s))
			c := make([]float32, convK*qkvDim)
			DownloadF32(gl.SSMConvBuf, c)
			blobs = append(blobs, f32Bytes(c))
		}
		if p.KVCache.KeyBufs[l] != 0 {
			kb := make([]byte, kvBytes)
			Download(p.KVCache.KeyBufs[l], kb)
			blobs = append(blobs, kb)
			vb := make([]byte, kvBytes)
			Download(p.KVCache.ValBufs[l], vb)
			blobs = append(blobs, vb)
		}
	}
	return blobs
}

func f32Bytes(f []float32) []byte {
	out := make([]byte, len(f)*4)
	for i, v := range f {
		bits := math.Float32bits(v)
		out[i*4] = byte(bits)
		out[i*4+1] = byte(bits >> 8)
		out[i*4+2] = byte(bits >> 16)
		out[i*4+3] = byte(bits >> 24)
	}
	return out
}

// TestSystemOneGPUStateDiagE2E pinpoints which persistent state component
// diverges between a single-call prefill and a two-call (prefix+suffix) prefill
// of the same tokens. Component order per layer: [SSM state, SSM conv, K, V].
func TestSystemOneGPUStateDiagE2E(t *testing.T) {
	_, gpuPipe, _, _ := newSystemOneTestPipelines(t)

	state := "I was charged twice for my subscription this month, the app crashes when I open the invoice page, and I cannot log in from my new phone."
	q := llm.SystemOneQuestion{ID: "route", Prompt: "Which team should handle this customer message?", Choices: []llm.Choice{
		{Value: "billing", Label: "Billing, charges, refunds, invoices, subscriptions, or payments."},
		{Value: "technical", Label: "Technical software, device, service, or product malfunction."},
		{Value: "account", Label: "Account access, identity, password, login, or authentication issue."},
	}}
	prompt, _, err := llm.CompileSystemOnePrompt(gpuPipe.CPUModel.Config, state, q, llm.SystemOneConfig{})
	if err != nil {
		t.Fatal(err)
	}
	tokens := gpuPipe.Tokenizer.Encode(prompt)
	npos := len(tokens)
	dim := gpuPipe.CPUModel.Config.EmbeddingDim
	split := npos / 2
	if err := gpuPipe.ensurePrefillBatchState(npos); err != nil {
		t.Fatal(err)
	}

	hybrid := func(toks []int32, startPos int, last bool, hidden []float32) {
		GpuForwardPrefillBatchHybrid(gpuPipe.CPUModel, gpuPipe.GpuModel, toks, gpuPipe.KVCache, gpuPipe.RunState,
			gpuPipe.BatchState, nil, gpuPipe.BatchLayerConfs, gpuPipe, startPos, last, hidden)
		Sync()
	}

	// A: full prompt, single call.
	gpuPipe.ResetState()
	hybrid(tokens, 0, true, make([]float32, dim))
	stateA := gpuDownloadState(gpuPipe, npos)

	// D: prefix (lastChunk=false) + suffix (lastChunk=true).
	gpuPipe.ResetState()
	hybrid(tokens[:split], 0, false, nil)
	hybrid(tokens[split:], split, true, make([]float32, dim))
	stateD := gpuDownloadState(gpuPipe, npos)

	// E1/E2: identical lastChunk=true prefix calls — determinism check.
	gpuPipe.ResetState()
	hybrid(tokens[:split], 0, true, make([]float32, dim))
	stateE1 := gpuDownloadState(gpuPipe, split)
	gpuPipe.ResetState()
	hybrid(tokens[:split], 0, true, make([]float32, dim))
	stateE2 := gpuDownloadState(gpuPipe, split)

	// B: lastChunk=false prefix call.
	gpuPipe.ResetState()
	hybrid(tokens[:split], 0, false, nil)
	stateB := gpuDownloadState(gpuPipe, split)

	names := []string{"ssmState", "ssmConv", "kvK", "kvV"}
	layer := 0
	comp := 0
	report := func(tag string, a, b [][]byte) {
		if len(a) != len(b) {
			t.Logf("%s: blob count %d vs %d", tag, len(a), len(b))
			return
		}
		mismatches := 0
		for i := range a {
			if !bytes.Equal(a[i], b[i]) {
				// count differing bytes and find first offset
				first := -1
				diff := 0
				for j := range a[i] {
					if a[i][j] != b[i][j] {
						diff++
						if first < 0 {
							first = j
						}
					}
				}
				mismatches++
				if mismatches <= 6 {
					t.Logf("%s: blob %d (layer %d %s) differs: %d/%d bytes, first at %d",
						tag, i, layer+comp, names[comp%4], diff, len(a[i]), first)
				}
			}
			comp++
			if comp%4 == 0 {
				layer++
			}
		}
		if mismatches == 0 {
			t.Logf("%s: all %d state blobs IDENTICAL", tag, len(a))
		} else {
			t.Logf("%s: %d/%d state blobs differ", tag, mismatches, len(a))
		}
	}

	report("E1 vs E2 (same call twice: determinism)", stateE1, stateE2)
	report("B vs E1 (lastChunk=false vs true)", stateB, stateE1)
	report("full: single call vs prefix+suffix", stateA, stateD)
}