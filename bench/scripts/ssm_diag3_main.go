//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

func main() {
	path := `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`
	pipe, err := llm.NewPipeline(path, 512)
	if err != nil {
		fmt.Printf("Load fail: %v\n", err)
		os.Exit(1)
	}

	cfg := pipe.Model.Config
	m := pipe.Model
	fmt.Printf("=== Model Config ===\n")
	fmt.Printf("  Arch:               %s\n", cfg.Architecture)
	fmt.Printf("  EmbeddingDim:       %d\n", cfg.EmbeddingDim)
	fmt.Printf("  NumLayers:          %d\n", cfg.NumLayers)
	fmt.Printf("  FFNDim:             %d\n", cfg.FFNDim)
	fmt.Printf("  NumHeads:           %d\n", cfg.NumHeads)
	fmt.Printf("  NumKVHeads:         %d\n", cfg.NumKVHeads)
	fmt.Printf("  HeadDim:            %d\n", cfg.HeadDim)
	fmt.Printf("  RMSNormEps:         %e\n", cfg.RMSNormEps)
	fmt.Printf("  RopeFreqBase:       %e\n", cfg.RopeFreqBase)
	fmt.Printf("  RopeNeox:           %v\n", cfg.RopeNeox)
	fmt.Printf("  RopeDim:            %d\n", cfg.RopeDim)
	fmt.Printf("  BOS:                %d\n", cfg.BOS)
	fmt.Printf("  AddBOS:             %v\n", cfg.AddBOS)
	fmt.Printf("  FullAttnInterval:   %d\n", cfg.FullAttentionInterval)
	fmt.Printf("  SSMConvKernel:      %d\n", cfg.SSMConvKernel)
	fmt.Printf("  SSMInnerSize:       %d\n", cfg.SSMInnerSize)
	fmt.Printf("  SSMStateSize:       %d\n", cfg.SSMStateSize)
	fmt.Printf("  SSMTimeStepRank:    %d\n", cfg.SSMTimeStepRank)
	fmt.Printf("  SSMGroupCount:      %d\n", cfg.SSMGroupCount)
	fmt.Printf("  ChatTemplate:       %s\n", cfg.ChatTemplate)

	// Derived SSM dims
	numHeads := cfg.SSMTimeStepRank     // 32
	numKVGroups := cfg.SSMGroupCount    // 16
	headVDim := cfg.SSMInnerSize / numHeads // 128
	headKDim := cfg.SSMStateSize        // 128
	valueDim := numHeads * headVDim     // 4096
	keyDim := numKVGroups * headKDim    // 2048
	qkvDim := keyDim*2 + valueDim      // 8192
	fmt.Printf("\n=== Derived SSM Dims ===\n")
	fmt.Printf("  numHeads=%d numKVGroups=%d headKDim=%d headVDim=%d\n", numHeads, numKVGroups, headKDim, headVDim)
	fmt.Printf("  keyDim=%d valueDim=%d qkvDim=%d\n", keyDim, valueDim, qkvDim)
	fmt.Printf("  headsPerGroup=%d\n", numHeads/numKVGroups)

	// Check layer 0 (SSM)
	fmt.Printf("\n=== Layer 0 (SSM) Weight Check ===\n")
	l0 := &m.Layers[0]
	fmt.Printf("  Spec.Core:     %v\n", l0.Spec.Core)
	fmt.Printf("  Spec.GatedQ:   %v\n", l0.Spec.GatedQ)
	fmt.Printf("  Spec.QKNorm:   %v\n", l0.Spec.QKNorm)
	fmt.Printf("  Spec.Residual: %v\n", l0.Spec.Residual)
	fmt.Printf("  SSMInProj:     rows=%d cols=%d (expected %d x %d)\n", l0.SSMInProj.Rows, l0.SSMInProj.Cols, qkvDim, cfg.EmbeddingDim)
	fmt.Printf("  AttnGate:      rows=%d cols=%d (expected %d x %d)\n", l0.AttnGate.Rows, l0.AttnGate.Cols, valueDim, cfg.EmbeddingDim)
	fmt.Printf("  SSMAlpha:      rows=%d cols=%d (expected %d x %d)\n", l0.SSMAlpha.Rows, l0.SSMAlpha.Cols, numHeads, cfg.EmbeddingDim)
	fmt.Printf("  SSMBeta:       rows=%d cols=%d (expected %d x %d)\n", l0.SSMBeta.Rows, l0.SSMBeta.Cols, numHeads, cfg.EmbeddingDim)
	fmt.Printf("  SSMOut:        rows=%d cols=%d (expected %d x %d)\n", l0.SSMOut.Rows, l0.SSMOut.Cols, cfg.EmbeddingDim, valueDim)
	fmt.Printf("  SSMNorm:       len=%d (expected %d)\n", len(l0.SSMNorm), headVDim)
	fmt.Printf("  SSMA:          len=%d (expected %d)\n", len(l0.SSMA), numHeads)
	fmt.Printf("  SSMDtBias:     len=%d (expected %d)\n", len(l0.SSMDtBias), numHeads)
	fmt.Printf("  SSMConv1dW:    len=%d (expected %d)\n", len(l0.SSMConv1dW), cfg.SSMConvKernel*qkvDim)
	fmt.Printf("  AttnNorm:      len=%d (expected %d)\n", len(l0.AttnNorm), cfg.EmbeddingDim)
	fmt.Printf("  PostAttnNorm:  %v\n", l0.PostAttnNorm != nil)
	fmt.Printf("  FFNNorm:       %v\n", l0.FFNNorm != nil)

	// Check layer 3 (attention)
	fmt.Printf("\n=== Layer 3 (Attention) Weight Check ===\n")
	l3 := &m.Layers[3]
	fmt.Printf("  Spec.Core:     %v\n", l3.Spec.Core)
	fmt.Printf("  Spec.GatedQ:   %v\n", l3.Spec.GatedQ)
	fmt.Printf("  Spec.QKNorm:   %v\n", l3.Spec.QKNorm)
	fmt.Printf("  Wq:            rows=%d cols=%d\n", l3.Wq.Rows, l3.Wq.Cols)
	fmt.Printf("  Wk:            rows=%d cols=%d\n", l3.Wk.Rows, l3.Wk.Cols)
	fmt.Printf("  Wv:            rows=%d cols=%d\n", l3.Wv.Rows, l3.Wv.Cols)
	fmt.Printf("  Wo:            rows=%d cols=%d\n", l3.Wo.Rows, l3.Wo.Cols)
	if l3.AttnQNorm != nil {
		fmt.Printf("  AttnQNorm:     len=%d\n", len(l3.AttnQNorm))
	}
	if l3.AttnKNorm != nil {
		fmt.Printf("  AttnKNorm:     len=%d\n", len(l3.AttnKNorm))
	}

	// Run single token and check SSM state
	fmt.Printf("\n=== SSM State After Token 0 ===\n")
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	rs := llm.NewRunState(cfg, 512)
	kv := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)

	tok := int32(760) // "The"
	llm.Forward(m, tok, 0, kv, rs)

	// Check SSM state for layer 0
	ssmL0 := rs.SSMState.Layers[0]
	fmt.Printf("  State size: %d (expected %d)\n", len(ssmL0.State), numHeads*headKDim*headVDim)
	fmt.Printf("  ConvBuf size: %d (expected %d)\n", len(ssmL0.ConvBuf), cfg.SSMConvKernel*qkvDim)

	// Check state magnitude per head
	for h := 0; h < numHeads; h++ {
		off := h * headKDim * headVDim
		var sumSq float64
		for i := off; i < off+headKDim*headVDim; i++ {
			sumSq += float64(ssmL0.State[i]) * float64(ssmL0.State[i])
		}
		l2 := math.Sqrt(sumSq)
		if h < 4 || h >= numHeads-2 {
			fmt.Printf("  Head %d state L2: %.6f\n", h, l2)
		} else if h == 4 {
			fmt.Printf("  ... (heads 4-%d omitted) ...\n", numHeads-3)
		}
	}

	// Also check SSMA values and DtBias
	fmt.Printf("\n=== SSMA (per head) ===\n")
	for h := 0; h < numHeads; h++ {
		if h < 4 || h >= numHeads-2 {
			fmt.Printf("  Head %d: SSMA=%.6f DtBias=%.6f\n", h, l0.SSMA[h], l0.SSMDtBias[h])
		} else if h == 4 {
			fmt.Printf("  ...\n")
		}
	}

	// Top prediction after first token
	top := argmax(rs.Logits)
	fmt.Printf("\nAfter token 'The' (pos=0): top=%d %q logit=%.4f\n", top, pipe.Tokenizer.DecodeToken(int32(top)), rs.Logits[top])
}

func argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}
