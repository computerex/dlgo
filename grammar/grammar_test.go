package grammar

import (
	"math"
	"testing"
)

func TestParseSimpleGrammar(t *testing.T) {
	g, err := Parse(`root ::= "hello"`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(g.Rules) == 0 {
		t.Fatal("no rules parsed")
	}
	if len(g.Stacks) == 0 {
		t.Fatal("no initial stacks")
	}
}

func TestParseJSONGrammar(t *testing.T) {
	g, err := Parse(JSONGrammar)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(g.Rules) == 0 {
		t.Fatal("no rules parsed")
	}
	if len(g.Stacks) == 0 {
		t.Fatal("no initial stacks")
	}
	t.Logf("JSON grammar: %d rules, %d initial stacks", len(g.Rules), len(g.Stacks))
}

func TestAcceptSimpleString(t *testing.T) {
	g, err := Parse(`root ::= "hi"`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	g.AcceptToken("h")
	if len(g.Stacks) == 0 {
		t.Fatal("stacks empty after accepting 'h'")
	}

	g.AcceptToken("i")
	if !g.IsComplete() {
		t.Fatal("grammar should be complete after 'hi'")
	}
}

func TestRejectInvalidChar(t *testing.T) {
	g, err := Parse(`root ::= "ab"`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	// "a" should be accepted
	g.AcceptToken("a")
	if len(g.Stacks) == 0 {
		t.Fatal("stacks empty after 'a'")
	}

	// now only "b" should be valid, not "x"
	logits := []float32{1.0, 1.0, 1.0} // tokens: 0="x", 1="b", 2="c"
	pieces := []string{"x", "b", "c"}
	eos := map[int32]bool{}

	g.ApplyToLogits(logits, pieces, eos)

	if logits[0] != float32(math.Inf(-1)) {
		t.Errorf("token 'x' should be masked, got %f", logits[0])
	}
	if logits[1] == float32(math.Inf(-1)) {
		t.Error("token 'b' should NOT be masked")
	}
	if logits[2] != float32(math.Inf(-1)) {
		t.Errorf("token 'c' should be masked, got %f", logits[2])
	}
}

func TestCharClass(t *testing.T) {
	g, err := Parse(`root ::= [a-z]`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	logits := []float32{1.0, 1.0, 1.0, 1.0}
	pieces := []string{"a", "m", "z", "5"}
	eos := map[int32]bool{}

	g.ApplyToLogits(logits, pieces, eos)

	for i := 0; i < 3; i++ {
		if logits[i] == float32(math.Inf(-1)) {
			t.Errorf("token '%s' should NOT be masked", pieces[i])
		}
	}
	if logits[3] != float32(math.Inf(-1)) {
		t.Errorf("token '5' should be masked")
	}
}

func TestNegatedCharClass(t *testing.T) {
	g, err := Parse(`root ::= [^0-9]`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	logits := []float32{1.0, 1.0, 1.0}
	pieces := []string{"a", "5", "!"}
	eos := map[int32]bool{}

	g.ApplyToLogits(logits, pieces, eos)

	if logits[0] == float32(math.Inf(-1)) {
		t.Error("'a' should NOT be masked (not a digit)")
	}
	if logits[1] != float32(math.Inf(-1)) {
		t.Error("'5' should be masked (is a digit)")
	}
	if logits[2] == float32(math.Inf(-1)) {
		t.Error("'!' should NOT be masked (not a digit)")
	}
}

func TestAlternation(t *testing.T) {
	g, err := Parse(`root ::= "a" | "b"`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	logits := []float32{1.0, 1.0, 1.0}
	pieces := []string{"a", "b", "c"}
	eos := map[int32]bool{}

	g.ApplyToLogits(logits, pieces, eos)

	if logits[0] == float32(math.Inf(-1)) {
		t.Error("'a' should be allowed")
	}
	if logits[1] == float32(math.Inf(-1)) {
		t.Error("'b' should be allowed")
	}
	if logits[2] != float32(math.Inf(-1)) {
		t.Error("'c' should be masked")
	}
}

func TestRepetitionStar(t *testing.T) {
	g, err := Parse(`root ::= "a"*`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	// Should be immediately complete (zero repetitions allowed)
	if !g.IsComplete() {
		t.Error("a* grammar should be complete at start (zero repetitions)")
	}

	// Accept "a"
	g.AcceptToken("a")
	if !g.IsComplete() {
		t.Error("a* grammar should still be complete after 'a'")
	}

	// Accept another "a"
	g.AcceptToken("a")
	if !g.IsComplete() {
		t.Error("a* grammar should still be complete after 'aa'")
	}
}

func TestRepetitionPlus(t *testing.T) {
	g, err := Parse(`root ::= "a"+`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	// Not complete at start (need at least one)
	if g.IsComplete() {
		t.Error("a+ grammar should NOT be complete at start")
	}

	g.AcceptToken("a")
	if !g.IsComplete() {
		t.Error("a+ grammar should be complete after 'a'")
	}
}

func TestOptional(t *testing.T) {
	g, err := Parse(`root ::= "a"? "b"`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	// Should accept "b" directly or "a" then "b"
	logits := []float32{1.0, 1.0, 1.0}
	pieces := []string{"a", "b", "c"}
	eos := map[int32]bool{}

	g.ApplyToLogits(logits, pieces, eos)

	if logits[0] == float32(math.Inf(-1)) {
		t.Error("'a' should be allowed (optional prefix)")
	}
	if logits[1] == float32(math.Inf(-1)) {
		t.Error("'b' should be allowed (skip optional)")
	}
	if logits[2] != float32(math.Inf(-1)) {
		t.Error("'c' should be masked")
	}
}

func TestRuleReference(t *testing.T) {
	g, err := Parse(`root ::= greeting
greeting ::= "hi"`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	g.AcceptToken("h")
	g.AcceptToken("i")
	if !g.IsComplete() {
		t.Error("should be complete after 'hi'")
	}
}

func TestEOSHandling(t *testing.T) {
	g, err := Parse(`root ::= "a"`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	// Before accepting anything, EOS should be blocked
	logits := []float32{1.0, 1.0}
	pieces := []string{"a", "<eos>"}
	eos := map[int32]bool{1: true}

	g.ApplyToLogits(logits, pieces, eos)
	if logits[1] != float32(math.Inf(-1)) {
		t.Error("EOS should be masked before grammar is complete")
	}

	// After accepting "a", EOS should be allowed
	g.AcceptToken("a")
	logits = []float32{1.0, 1.0}
	g.ApplyToLogits(logits, pieces, eos)
	if logits[1] == float32(math.Inf(-1)) {
		t.Error("EOS should be allowed after grammar is complete")
	}
}

func TestJSONObjectAccept(t *testing.T) {
	g, err := Parse(JSONGrammar)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	// Accept a minimal JSON object: {}
	jsonStr := `{}`
	for _, ch := range jsonStr {
		before := len(g.Stacks)
		g.AcceptChar(uint32(ch))
		if len(g.Stacks) == 0 {
			t.Fatalf("stacks empty after accepting '%c' (had %d stacks before)", ch, before)
		}
	}

	if !g.IsComplete() {
		t.Error("grammar should be complete after '{}'")
	}
}

func TestJSONSimpleObject(t *testing.T) {
	g, err := Parse(JSONGrammar)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	jsonStr := `{"key": "value"}`
	for i, ch := range jsonStr {
		before := len(g.Stacks)
		g.AcceptChar(uint32(ch))
		if len(g.Stacks) == 0 {
			t.Fatalf("stacks empty at char %d '%c' (had %d stacks before)", i, ch, before)
		}
	}

	if !g.IsComplete() {
		t.Error("grammar should be complete after valid JSON object")
	}
}

func TestJSONRejectsInvalid(t *testing.T) {
	g, err := Parse(JSONGrammar)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	// First character must be '{' for an object
	logits := []float32{1.0, 1.0, 1.0, 1.0}
	pieces := []string{"{", "a", "[", "\""}
	eos := map[int32]bool{}

	g.ApplyToLogits(logits, pieces, eos)

	if logits[0] == float32(math.Inf(-1)) {
		t.Error("'{' should be allowed at start of JSON object")
	}
	if logits[1] != float32(math.Inf(-1)) {
		t.Error("'a' should be masked at start of JSON object")
	}
}

func TestGrouping(t *testing.T) {
	g, err := Parse(`root ::= ("a" "b") | "c"`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	logits := []float32{1.0, 1.0, 1.0}
	pieces := []string{"a", "c", "x"}
	eos := map[int32]bool{}

	g.ApplyToLogits(logits, pieces, eos)

	if logits[0] == float32(math.Inf(-1)) {
		t.Error("'a' should be allowed (start of 'ab' alternative)")
	}
	if logits[1] == float32(math.Inf(-1)) {
		t.Error("'c' should be allowed (second alternative)")
	}
	if logits[2] != float32(math.Inf(-1)) {
		t.Error("'x' should be masked")
	}
}

func TestMultiCharToken(t *testing.T) {
	g, err := Parse(`root ::= "hello"`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	// A token that contains the whole word
	logits := []float32{1.0, 1.0, 1.0}
	pieces := []string{"hello", "world", "h"}
	eos := map[int32]bool{}

	g.ApplyToLogits(logits, pieces, eos)

	if logits[0] == float32(math.Inf(-1)) {
		t.Error("'hello' token should be allowed")
	}
	if logits[1] != float32(math.Inf(-1)) {
		t.Error("'world' token should be masked")
	}
	if logits[2] == float32(math.Inf(-1)) {
		t.Error("'h' token should be allowed (partial match)")
	}
}

func TestAnyChar(t *testing.T) {
	g, err := Parse(`root ::= .`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	logits := []float32{1.0, 1.0, 1.0}
	pieces := []string{"a", "5", "!"}
	eos := map[int32]bool{}

	g.ApplyToLogits(logits, pieces, eos)

	for i, p := range pieces {
		if logits[i] == float32(math.Inf(-1)) {
			t.Errorf("'%s' should be allowed with '.' rule", p)
		}
	}
}

func TestUndefinedRuleError(t *testing.T) {
	_, err := Parse(`root ::= undefined_rule`)
	if err == nil {
		t.Error("expected error for undefined rule reference")
	}
}

func TestLeftRecursionError(t *testing.T) {
	_, err := Parse(`root ::= root "a"`)
	if err == nil {
		t.Error("expected error for left-recursive grammar")
	}
}

func TestDecodeUTF8(t *testing.T) {
	cps, partial := decodeUTF8("hello", 0, 0)
	// should be [104, 101, 108, 108, 111, 0]
	expected := []uint32{'h', 'e', 'l', 'l', 'o', 0}
	if len(cps) != len(expected) {
		t.Fatalf("codepoints length = %d, want %d", len(cps), len(expected))
	}
	for i := range expected {
		if cps[i] != expected[i] {
			t.Errorf("cp[%d] = %d, want %d", i, cps[i], expected[i])
		}
	}
	if partial.nRemain != 0 {
		t.Errorf("partial.nRemain = %d, want 0", partial.nRemain)
	}
}

func TestRepetitionCount(t *testing.T) {
	g, err := Parse(`root ::= [0-9]{3}`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	g.AcceptToken("1")
	if g.IsComplete() {
		t.Error("should not be complete after 1 digit")
	}
	g.AcceptToken("2")
	if g.IsComplete() {
		t.Error("should not be complete after 2 digits")
	}
	g.AcceptToken("3")
	if !g.IsComplete() {
		t.Error("should be complete after 3 digits")
	}
}

func TestClone(t *testing.T) {
	g, err := Parse(`root ::= "ab"`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	// Clone before accepting anything
	g2 := g.Clone()

	g.AcceptToken("a")
	// g should have advanced, g2 should still be at start
	if g.IsComplete() {
		t.Error("g should not be complete after 'a'")
	}

	// g2 should still accept 'a'
	logits := []float32{1.0, 1.0}
	pieces := []string{"a", "b"}
	eos := map[int32]bool{}
	g2.ApplyToLogits(logits, pieces, eos)
	if logits[0] == float32(math.Inf(-1)) {
		t.Error("cloned grammar should still accept 'a'")
	}
}