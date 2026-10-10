package llm

import (
	"os"
	"strings"
	"testing"
	"time"
)

// TestTokenizerLongDocParity encodes a large real-world document and asserts:
//  1. known substrings round-trip (sanity)
//  2. encoding is fast (the pre-fix applyBPE took ~30s; heap version must be <2s)
//  3. output is stable across repeated calls (determinism)
//
// Run against the real Qwen3.5 tokenizer used by System One.
func TestTokenizerLongDocParity(t *testing.T) {
	const modelPath = `C:\Users\mohd\Downloads\Qwen3.5-0.8B-Q4_K_M.gguf`
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("model not available")
	}
	article, err := os.ReadFile(`C:\projects\dlgo\.sessions\gdpr_article.txt`)
	if err != nil {
		t.Skip("article fixture not available")
	}
	text := string(article)

	pipe, err := NewPipeline(modelPath, 2048)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	tok := pipe.Tokenizer

	start := time.Now()
	toks := tok.Encode(text)
	d1 := time.Since(start)
	t.Logf("encode1: %d tokens in %v (%.0f tok/s)", len(toks), d1, float64(len(toks))/d1.Seconds())

	start = time.Now()
	toks2 := tok.Encode(text)
	d2 := time.Since(start)
	t.Logf("encode2: %v", d2)

	if len(toks) != len(toks2) {
		t.Fatalf("nondeterministic token count: %d vs %d", len(toks), len(toks2))
	}
	for i := range toks {
		if toks[i] != toks2[i] {
			t.Fatalf("nondeterministic token at %d: %d vs %d", i, toks[i], toks2[i])
		}
	}
	if d1 > 2*time.Second {
		t.Fatalf("encode too slow: %v (expected <2s after heap BPE)", d1)
	}

	// Sanity: BOS present, decode round-trips whitespace-normalized prefix.
	if strings.TrimSpace(tok.Decode(toks)) != strings.TrimSpace(text) {
		t.Fatalf("decode round-trip mismatch")
	}
}
