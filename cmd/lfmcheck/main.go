// Command lfmcheck loads an LFM2 / LFM2MoE GGUF, dumps the resolved architecture
// (per-layer core and FFN kinds, KV-cache footprint) and runs a short greedy
// generation so the forward pass can be checked for coherence.
//
//	go run ./cmd/lfmcheck -model <path.gguf> -ctx 2048 -tokens 32
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func coreName(c llm.CoreKind) string {
	switch c {
	case llm.CoreAttention:
		return "attn"
	case llm.CoreSSM:
		return "ssm"
	case llm.CoreMLA:
		return "mla"
	case llm.CoreShortConv:
		return "sconv"
	}
	return "?"
}

func ffnName(f llm.FFNKind) string {
	switch f {
	case llm.FFNSwiGLU:
		return "swiglu"
	case llm.FFNGeGLU:
		return "geglu"
	case llm.FFNPlain:
		return "plain"
	case llm.FFNMoE:
		return "moe"
	case llm.FFNMoESwiOAI:
		return "moe-oai"
	}
	return "?"
}

func main() {
	model := flag.String("model", `C:\Users\mohd\Downloads\LFM2.5-24B-A2B-QAD-Q4_0.gguf`, "GGUF path")
	ctx := flag.Int("ctx", 2048, "context length")
	tokens := flag.Int("tokens", 32, "tokens to generate")
	prompt := flag.String("prompt", "The capital of France is", "prompt text")
	chat := flag.Bool("chat", false, "wrap the prompt in the model's chat template")
	enc := flag.Bool("enc", false, "print the tokenization of the prompt and exit")
	first := flag.Bool("first", false, "print the top logits for the first generated token")
	bos := flag.Bool("bos", false, "force BOS to be prepended when tokenizing")
	dumpOnly := flag.Bool("dump", false, "only dump the architecture, no generation")
	flag.Parse()

	t0 := time.Now()
	p, err := llm.NewPipeline(*model, *ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load failed: %v\n", err)
		os.Exit(1)
	}
	cfg := p.Model.Config

	fmt.Printf("== loaded in %.2fs\n", time.Since(t0).Seconds())
	fmt.Printf("arch=%s layers=%d dim=%d heads=%d kvHeads=%d headDim=%d ffn=%d vocab=%d\n",
		cfg.Architecture, cfg.NumLayers, cfg.EmbeddingDim, cfg.NumHeads, cfg.NumKVHeads,
		cfg.HeadDim, cfg.FFNDim, cfg.VocabSize)
	fmt.Printf("rope: neox=%v dim=%d base=%.1f eps=%.1e\n", cfg.RopeNeox, cfg.RopeDim, cfg.RopeFreqBase, cfg.RMSNormEps)
	fmt.Printf("moe: experts=%d used=%d expFFN=%d gatingFunc=%d weightsNorm=%v weightsScale=%v leadingDense=%d\n",
		cfg.ExpertCount, cfg.ExpertUsedCount, cfg.ExpertFFNDim, cfg.ExpertGatingFunc,
		cfg.ExpertWeightsNorm, cfg.ExpertWeightsScale, cfg.LeadingDenseCount)
	fmt.Printf("shortconv: l_cache=%d hasShortConv=%v attnLayers=%d kvBytesPerTok=%d\n",
		cfg.ShortConvLCache, cfg.HasShortConv(), cfg.AttentionLayerCount(), cfg.KVBytesPerToken())
	if len(cfg.KVHeadsPerLayer) > 0 {
		fmt.Printf("kvHeadsPerLayer=%v\n", cfg.KVHeadsPerLayer)
	}
	fmt.Printf("chatTemplate=%s bos=%d eos=%d addBOS=%v\n", cfg.ChatTemplate, cfg.BOS, cfg.EOS, cfg.AddBOS)
	fmt.Printf("outputNorm len=%d, output tied=%v\n", len(p.Model.OutputNorm), p.Model.Output == p.Model.TokenEmbed)

	// Per-layer summary.
	coreCount := map[string]int{}
	ffnCount := map[string]int{}
	var noKV []int
	for l := range p.Model.Layers {
		spec := p.Model.Layers[l].Spec
		coreCount[coreName(spec.Core)]++
		ffnCount[ffnName(spec.FFN)]++
		if p.Model.Layers[l].ShortConvConv == nil && spec.Core == llm.CoreShortConv {
			fmt.Printf("  !! layer %d is CoreShortConv but has no conv taps\n", l)
		}
		if p.KVCache != nil && p.KVCache.Layers[l] == nil {
			noKV = append(noKV, l)
		}
	}
	fmt.Printf("cores=%v ffns=%v\n", coreCount, ffnCount)
	fmt.Printf("layers without KV cache (%d): %v\n", len(noKV), noKV)

	// Spot-check the first shortconv and first attention layer.
	for l := 0; l < cfg.NumLayers; l++ {
		L := p.Model.Layers[l]
		if L.Spec.Core == llm.CoreShortConv {
			fmt.Printf("shortconv layer %d: conv len=%d in_proj=%dx%d out_proj=%dx%d attnNorm=%d ffnNorm=%d selBias=%d\n",
				l, len(L.ShortConvConv),
				L.ShortConvInProj.Rows, L.ShortConvInProj.Cols,
				L.ShortConvOutProj.Rows, L.ShortConvOutProj.Cols,
				len(L.AttnNorm), len(L.FFNNorm), len(L.FFNExpertSelBias))
			break
		}
	}
	for l := 0; l < cfg.NumLayers; l++ {
		L := p.Model.Layers[l]
		if L.Spec.Core == llm.CoreAttention {
			fmt.Printf("attn layer %d: q=%dx%d k=%dx%d v=%dx%d o=%dx%d qnorm=%d knorm=%d\n",
				l, L.Wq.Rows, L.Wq.Cols, L.Wk.Rows, L.Wk.Cols,
				L.Wv.Rows, L.Wv.Cols, L.Wo.Rows, L.Wo.Cols,
				len(L.AttnQNorm), len(L.AttnKNorm))
			break
		}
	}
	for l := 0; l < cfg.NumLayers; l++ {
		L := p.Model.Layers[l]
		if L.Spec.FFN == llm.FFNMoE {
			fmt.Printf("moe layer %d: router=%dx%d gateExps=%dx%d upExps=%dx%d downExps=%dx%d selBias=%d\n",
				l, L.FFNRouter.Rows, L.FFNRouter.Cols,
				L.FFNGateExps.Rows, L.FFNGateExps.Cols,
				L.FFNUpExps.Rows, L.FFNUpExps.Cols,
				L.FFNDownExps.Rows, L.FFNDownExps.Cols, len(L.FFNExpertSelBias))
			break
		}
	}

	if *dumpOnly {
		return
	}

	gcfg := llm.DefaultGenerateConfig()
	gcfg.MaxTokens = *tokens
	gcfg.Sampler = ops.SamplerConfig{Temperature: 0, TopK: 1, TopP: 1, RepetitionPenalty: 1.0}

	t1 := time.Now()
	text := *prompt
	if *chat {
		text = llm.FormatChat(cfg, "", text)
		fmt.Printf("\n== chat prompt:\n%s\n", text)
	}
	if *enc {
		ids := p.Tokenizer.Encode(text)
		fmt.Printf("\n== tokenization: %d tokens\n", len(ids))
		for i, id := range ids {
			fmt.Printf("  %3d %7d %q\n", i, id, p.Tokenizer.DecodeToken(id))
		}
		if !*first {
			return
		}
	}

	if *bos {
		p.Tokenizer.AddBOS = true
		p.Model.Config.AddBOS = true
		fmt.Printf("forced AddBOS=true (bos id=%d)\n", cfg.BOS)
	}

	if *first {
		// Mirror the server's CPU path exactly: prefill the chat prompt, apply
		// the EOG suppression list, then greedy-pick the first token.
		ids := p.Tokenizer.Encode(text)
		p.ResetSequenceState()
		llm.ForwardBatch(p.Model, ids, 0, p.KVCache, p.RunState, p.BatchState)
		logits := p.RunState.Logits
		fmt.Printf("\n== prefill done, prompt=%d tokens\n", len(ids))
		llm.ApplySuppressTokens(logits, cfg.SuppressTokens)
		fmt.Printf("suppressTokens=%v eos=%d stopTokens=%v\n", cfg.SuppressTokens, cfg.EOS, cfg.StopTokens)

		type pair struct {
			id int
			l  float32
		}
		best := make([]pair, 0, 8)
		for i, v := range logits {
			if len(best) < 8 {
				best = append(best, pair{i, v})
			} else {
				worst := 0
				for j := 1; j < len(best); j++ {
					if best[j].l < best[worst].l {
						worst = j
					}
				}
				if v > best[worst].l {
					best[worst] = pair{i, v}
				}
			}
		}
		for i := 1; i < len(best); i++ {
			for j := i; j > 0 && best[j].l > best[j-1].l; j-- {
				best[j], best[j-1] = best[j-1], best[j]
			}
		}
		for _, b := range best {
			fmt.Printf("  logit %9.4f  id=%7d  %q\n", b.l, b.id, p.Tokenizer.DecodeToken(int32(b.id)))
		}
		got := ops.SampleToken(logits, ops.SamplerConfig{Temperature: 0}, nil, nil)
		fmt.Printf("argmax -> id=%d %q  isEOS=%v\n", got, p.Tokenizer.DecodeToken(int32(got)), int32(got) == cfg.EOS)
		return
	}
	out, tps, err := p.GenerateText(text, gcfg)
	elapsed := time.Since(t1).Seconds()
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("\n== generation: %.2fs, %.2f tok/s\n", elapsed, tps)
	fmt.Printf("----8<----\n%s\n----8<----\n", out)

	// Sanity: flag obviously broken output (empty, or control-character soup).
	trimmed := strings.TrimSpace(out)
	if len(trimmed) == 0 {
		fmt.Println("VERDICT: FAIL (empty output)")
		os.Exit(2)
	}
	nonPrint := 0
	for _, r := range trimmed {
		if r < 0x20 && r != '\n' && r != '\t' && r != '\r' {
			nonPrint++
		}
	}
	if nonPrint*10 > len([]rune(trimmed)) {
		fmt.Println("VERDICT: FAIL (mostly control characters)")
		os.Exit(2)
	}
	fmt.Println("VERDICT: output is printable text")
}
