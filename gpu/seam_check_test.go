//go:build cgo && vulkan

package gpu

import (
	"os"
	"strings"
	"testing"

	"github.com/computerex/dlgo/models/llm"
)

// seamCase is one seam test: a prefix, a suffix, and whether the concatenation
// must tokenize identically to the parts (safe) or not (unsafe).
type seamCase struct {
	prefix string
	suffix string
}

// TestCheckSeamBoundaryParity pins the seam-local boundary check against the
// exact definition: encode(prefix)+encode(suffix) == encode(prefix+suffix).
// For every case, checkSeamBoundary must return tokens exactly when the
// definition holds, and nil exactly when it does not.
func TestCheckSeamBoundaryParity(t *testing.T) {
	const modelPath = `C:\Users\mohd\Downloads\Qwen3.5-0.8B-Q4_K_M.gguf`
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("model not available")
	}
	if err := Init(); err != nil {
		t.Skipf("Vulkan init failed: %v", err)
	}
	cpuPipe, err := llm.NewPipeline(modelPath, 2048)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	p, err := NewGpuPipeline(cpuPipe)
	if err != nil {
		t.Fatalf("NewGpuPipeline: %v", err)
	}
	defer p.FreeAll()

	doc := ""
	if b, err := os.ReadFile(`C:\projects\dlgo\.sessions\gdpr_article.txt`); err == nil {
		doc = string(b)
	}

	cases := []seamCase{
		{prefix: doc, suffix: "\n\nQuestion: Is this true?\nA. yes\nB. no"},
		{prefix: doc, suffix: "Question: Is this true?"},
		{prefix: strings.Repeat("The quick brown fox jumps over the lazy dog. ", 500), suffix: " What next?"},
		{prefix: "Short prefix.", suffix: " Short suffix."},
		{prefix: "unicode: café über 日本語 ok", suffix: "more unicode: ñ and emoji 🎉 end"},
		{prefix: "ends with number 12345", suffix: "678 more"},
		{prefix: "ends with partial word supercalifragilistic", suffix: "expialidocious"},
		{prefix: "trailing space ", suffix: "leading text"},
		{prefix: "trailing newline\n", suffix: "\n\nQuestion after blank lines?"},
		{prefix: "code block:\n\nfunc main() {}\n\n", suffix: " extra fence"},
		{prefix: strings.Repeat("x", 3), suffix: strings.Repeat("y", 3)},
	}
	if doc != "" {
		// nasty seams against the real article: cut mid-word, mid-number,
		// right after a heading newline, right before a markdown table row.
		cut := len(doc) / 2
		cases = append(cases,
			seamCase{prefix: doc[:cut], suffix: doc[cut:] + "\n\nQ?"},
			seamCase{prefix: doc[:cut-1], suffix: doc[cut-1:][:1] + " appended"},
		)
	}

	for i, c := range cases {
		pt := p.Tokenizer.Encode(c.prefix)
		st := p.Tokenizer.Encode(c.suffix)
		full := p.Tokenizer.Encode(c.prefix + c.suffix)
		want := append(append([]int32(nil), pt...), st...)
		exact := len(full) == len(want)
		if exact {
			for j := range full {
				if full[j] != want[j] {
					exact = false
					break
				}
			}
		}
		got := p.checkSeamBoundary(c.prefix, pt, c.suffix)
		gotOK := got != nil && len(got) == len(st)
		if gotOK && exact {
			for j := range got {
				if got[j] != st[j] {
					gotOK = false
					break
				}
			}
		}
		if gotOK != exact {
			t.Errorf("case %d: definition(exact=%v) but checkSeamBoundary(ok=%v) prefix=%q... suffix=%q",
				i, exact, gotOK, tailStr(c.prefix, 40), headStr(c.suffix, 40))
			continue
		}
		// When the check passes, the returned suffix tokens must be the true
		// standalone suffix tokens (they drive the shared-prefill continuation).
		if gotOK && exact {
			for j := range got {
				if got[j] != st[j] {
					t.Errorf("case %d: suffix token mismatch at %d", i, j)
					break
				}
			}
		}
	}
}

// TestCheckSeamBoundaryAgainstFullEncodeOnCorpus fuzzes the seam check across
// many cut points of the real article (the actual production seam shape).
func TestCheckSeamBoundaryAgainstFullEncodeOnCorpus(t *testing.T) {
	const modelPath = `C:\Users\mohd\Downloads\Qwen3.5-0.8B-Q4_K_M.gguf`
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("model not available")
	}
	raw, err := os.ReadFile(`C:\projects\dlgo\.sessions\gdpr_article.txt`)
	if err != nil {
		t.Skip("article fixture not available")
	}
	doc := string(raw)
	if err := Init(); err != nil {
		t.Skipf("Vulkan init failed: %v", err)
	}
	cpuPipe, err := llm.NewPipeline(modelPath, 2048)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	p, err := NewGpuPipeline(cpuPipe)
	if err != nil {
		t.Fatalf("NewGpuPipeline: %v", err)
	}
	defer p.FreeAll()

	suffix := "\n\nQuestion: Rate compliance burden 0-4.\n0 none\n4 extreme"
	checked := 0
	for cut := 1000; cut < len(doc)-len(suffix)-1; cut += 997 {
		prefix := doc[:cut]
		pt := p.Tokenizer.Encode(prefix)
		got := p.checkSeamBoundary(prefix, pt, suffix)
		full := p.Tokenizer.Encode(prefix + suffix)
		want := append(append([]int32(nil), pt...), p.Tokenizer.Encode(suffix)...)
		exact := len(full) == len(want)
		if exact {
			for j := range full {
				if full[j] != want[j] {
					exact = false
					break
				}
			}
		}
		if (got != nil) != exact {
			t.Fatalf("cut %d: definition(exact=%v) but check(nil=%v)", cut, exact, got == nil)
		}
		checked++
		if checked >= 40 {
			break
		}
	}
	if checked == 0 {
		t.Fatal("no seams checked")
	}
}

func tailStr(s string, n int) string {
	if len(s) > n {
		return s[len(s)-n:]
	}
	return s
}

func headStr(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

var _ = llm.SystemOneConfig{}
