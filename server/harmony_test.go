package server

import (
	"strings"
	"testing"
)

// runHarmony feeds text to a splitter in chunks of the given size and returns
// the concatenated content and reasoning.
func runHarmony(text string, chunk int) (content, reasoning string) {
	h := newHarmonySplitter()
	var c, r strings.Builder
	emit := func(isReasoning bool, piece string) {
		if isReasoning {
			r.WriteString(piece)
		} else {
			c.WriteString(piece)
		}
	}
	for i := 0; i < len(text); {
		j := i + chunk
		if j > len(text) {
			j = len(text)
		}
		h.feed(text[i:j], emit)
		i = j
	}
	h.close(emit)
	return c.String(), r.String()
}

func TestHarmonySplitterAnalysisFinal(t *testing.T) {
	stream := "<|start|>assistant<|channel|>analysis<|message|>" +
		"The user asks: 2+2. So answer: 4.<|end|>" +
		"<|start|>assistant<|channel|>final<|message|>4<|return|>"

	// Every chunk size must produce the same split, including 1 byte at a time
	// (which cuts every marker in half).
	for chunk := 1; chunk <= len(stream)+1; chunk++ {
		content, reasoning := runHarmony(stream, chunk)
		if want := "4"; content != want {
			t.Fatalf("chunk=%d content = %q, want %q", chunk, content, want)
		}
		if !strings.Contains(reasoning, "The user asks: 2+2.") {
			t.Fatalf("chunk=%d reasoning = %q, want the analysis text", chunk, reasoning)
		}
		if strings.Contains(content, "analysis") || strings.Contains(content, "<|") {
			t.Fatalf("chunk=%d content leaked protocol markup: %q", chunk, content)
		}
	}
}

func TestHarmonySplitterFinalOnly(t *testing.T) {
	stream := "<|start|>assistant<|channel|>final<|message|>Hello there.<|return|>"
	for chunk := 1; chunk <= len(stream)+1; chunk++ {
		content, reasoning := runHarmony(stream, chunk)
		if content != "Hello there." {
			t.Fatalf("chunk=%d content = %q", chunk, content)
		}
		if reasoning != "" {
			t.Fatalf("chunk=%d reasoning = %q, want empty", chunk, reasoning)
		}
	}
}

func TestHarmonySplitterCommentaryIsReasoning(t *testing.T) {
	stream := "<|start|>assistant<|channel|>analysis<|message|>think<|end|>" +
		"<|start|>assistant<|channel|>commentary to=functions.f <|constrain|>json<|message|>" +
		`{"x":1}` +
		"<|call|>" +
		"<|start|>assistant<|channel|>final<|message|>done<|return|>"

	for chunk := 1; chunk <= len(stream)+1; chunk++ {
		content, reasoning := runHarmony(stream, chunk)
		if content != "done" {
			t.Fatalf("chunk=%d content = %q, want %q", chunk, content, "done")
		}
		if !strings.Contains(reasoning, `{"x":1}`) {
			t.Fatalf("chunk=%d reasoning = %q, want the commentary body", chunk, reasoning)
		}
	}
}

func TestHarmonySplitterTruncatedFinal(t *testing.T) {
	// Generation cut off mid-answer: the partial text must still reach the
	// client, and a half-written marker must not be emitted as content.
	stream := "<|start|>assistant<|channel|>analysis<|message|>hmm<|end|>" +
		"<|start|>assistant<|channel|>final<|message|>partial ans<|ret"
	content, _ := runHarmony(stream, 1)
	if content != "partial ans" {
		t.Fatalf("content = %q, want %q", content, "partial ans")
	}
}

func TestHarmonySplitterEmptyAnalysis(t *testing.T) {
	stream := "<|start|>assistant<|channel|>analysis<|message|><|end|>" +
		"<|start|>assistant<|channel|>final<|message|>Answer<|return|>"
	for chunk := 1; chunk <= len(stream)+1; chunk++ {
		content, reasoning := runHarmony(stream, chunk)
		if content != "Answer" {
			t.Fatalf("chunk=%d content = %q", chunk, content)
		}
		if reasoning != "" {
			t.Fatalf("chunk=%d reasoning = %q, want empty", chunk, reasoning)
		}
	}
}

func TestHarmonySplitterNoChannelMarkers(t *testing.T) {
	// A stream that never opens a channel must not be swallowed.
	stream := "just plain text without any marker"
	content, _ := runHarmony(stream, 4)
	if content != stream {
		t.Fatalf("content = %q, want %q", content, stream)
	}
}

func TestHarmonySplitterUnknownChannelIsReasoning(t *testing.T) {
	stream := "<|channel|>mystery<|message|>secret<|end|>" +
		"<|start|>assistant<|channel|>final<|message|>visible<|return|>"
	content, reasoning := runHarmony(stream, 1)
	if content != "visible" {
		t.Fatalf("content = %q, want %q", content, "visible")
	}
	if !strings.Contains(reasoning, "secret") {
		t.Fatalf("reasoning = %q, want the unknown-channel body", reasoning)
	}
}

func TestHarmonyChannelName(t *testing.T) {
	cases := map[string]string{
		"analysis":                             harmonyChanAnalysis,
		"final":                                harmonyChanFinal,
		"commentary":                           harmonyChanCommentary,
		"commentary to=functions.get_weather ": harmonyChanCommentary,
		"final ":                               harmonyChanFinal,
		"":                                     harmonyChanFinal,
		"something-else":                       harmonyChanCommentary,
	}
	for header, want := range cases {
		if got := harmonyChannelName(header); got != want {
			t.Errorf("harmonyChannelName(%q) = %q, want %q", header, got, want)
		}
	}
}
