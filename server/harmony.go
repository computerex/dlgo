package server

import "strings"

// Harmony is OpenAI's structured chat protocol (used by gpt-oss). Instead of a
// single free-text stream it multiplexes named "channels" inside one turn:
//
//	<|start|>assistant<|channel|>analysis<|message|>…reasoning…<|end|>
//	<|start|>assistant<|channel|>final<|message|>…answer…<|return|>
//
// Only the `final` channel is the assistant's answer. `analysis` is the
// reasoning trace and `commentary` carries tool calls / intermediate notes.
// A front end that streams the raw decoded text leaks the entire reasoning
// chain into `content`, which is what this splitter prevents.
//
// The splitter is a streaming state machine: it is fed decoded token pieces one
// at a time (they can cut a marker in half) and reports each piece together
// with whether it belongs to the reasoning side.
const (
	harmonyChanAnalysis   = "analysis"
	harmonyChanCommentary = "commentary"
	harmonyChanFinal      = "final"
)

// harmonyOpen is the only possible start of a control token. Any "<" at the end
// of the buffer may be the first byte of one, so it is held back one token.
const harmonyOpen = "<|"

// harmonyMsgOpen terminates a channel header and starts its body.
const harmonyMsgOpen = "<|message|>"

// harmonyNoHeaderLimit bounds how much text we are willing to swallow while
// waiting for a channel header to open. A conforming harmony model emits the
// header as its first tokens; this only guards against a malformed/foreign
// stream so output is never silently dropped.
const harmonyNoHeaderLimit = 512

type harmonyState int

const (
	harmonySeek   harmonyState = iota // waiting for <|channel|>
	harmonyHeader                     // inside <|channel|> … <|message|>
	harmonyBody                       // inside the channel's content
)

// harmonySplitter converts a raw harmony token stream into
// (isReasoning, text) pieces.
type harmonySplitter struct {
	state   harmonyState
	buf     string
	channel string

	raw         strings.Builder // everything seen, for the degenerate fallback
	reasoning   strings.Builder // concatenated reasoning-channel content
	emitted     bool            // whether any piece has been reported
	seenControl bool            // whether any complete control token was parsed
}

func newHarmonySplitter() *harmonySplitter {
	return &harmonySplitter{state: harmonySeek}
}

// Reasoning returns the concatenated analysis/commentary text seen so far.
func (h *harmonySplitter) Reasoning() string {
	return h.reasoning.String()
}

// feed appends a decoded piece and drains everything that is now unambiguous.
func (h *harmonySplitter) feed(token string, emit func(reasoning bool, text string)) {
	if token == "" {
		return
	}
	h.raw.WriteString(token)
	h.buf += token
	h.drain(emit)
}

// close flushes whatever is left once generation has finished. Anything still
// buffered is either the tail of the current channel's content or a marker cut
// in half by truncation (which is dropped).
func (h *harmonySplitter) close(emit func(reasoning bool, text string)) {
	body := h.buf
	h.buf = ""

	if !h.seenControl {
		// No harmony structure was ever recognised, so the stream cannot be a
		// multiplexed turn: it is all answer, none of it reasoning.
		h.channel = harmonyChanFinal
		h.emitBody(body, emit)
		return
	}

	switch h.state {
	case harmonyBody:
		h.emitBody(harmonyTrimPartial(body), emit)
	case harmonyHeader:
		// Generation stopped inside a channel header. The header is protocol,
		// not content.
	}
	if !h.emitted {
		// Safety net: never hand the client an empty answer when the stream did
		// look like harmony but nothing was classified.
		h.channel = harmonyChanFinal
		h.emitBody(h.raw.String(), emit)
	}
}

// harmonyTrimPartial drops a control token that was cut in half by the end of
// generation, so "<|ret" never reaches the client as text.
func harmonyTrimPartial(s string) string {
	if i := strings.LastIndex(s, harmonyOpen); i >= 0 {
		if !strings.Contains(s[i:], "|>") {
			return s[:i]
		}
	}
	return s
}

func (h *harmonySplitter) drain(emit func(reasoning bool, text string)) {
	for {
		switch h.state {
		case harmonySeek:
			i := strings.Index(h.buf, harmonyOpen)
			if i < 0 {
				// No control token in sight. A trailing "<" could still start
				// one, so hold it back; anything older than the safety limit is
				// treated as ordinary content so it is not lost.
				keep := 0
				if strings.HasSuffix(h.buf, "<") {
					keep = 1
				}
				if len(h.buf)-keep > harmonyNoHeaderLimit {
					body := h.buf[:len(h.buf)-keep]
					h.buf = h.buf[len(h.buf)-keep:]
					h.emitBody(body, emit)
				}
				return
			}
			// Text before the first control token is the addressee
			// ("assistant") or whitespace, never answer content.
			h.buf = h.buf[i:]

			name, rest, ok := harmonyControl(h.buf)
			if !ok {
				return
			}
			h.buf = rest
			h.seenControl = true
			switch name {
			case "channel":
				h.state = harmonyHeader
			case "start", "end", "return", "call", "endoftext", "message", "constrain":
				// Structural only.
			default:
				// Unknown control token: drop it rather than leak it.
			}

		case harmonyHeader:
			j := strings.Index(h.buf, harmonyMsgOpen)
			if j < 0 {
				if len(h.buf) > harmonyNoHeaderLimit {
					// Malformed header: give up on structure and treat the
					// remainder as answer content.
					body := h.buf
					h.buf = ""
					h.channel = harmonyChanFinal
					h.state = harmonyBody
					h.emitBody(body, emit)
				}
				return
			}
			header := h.buf[:j]
			h.buf = h.buf[j+len(harmonyMsgOpen):]
			h.channel = harmonyChannelName(header)
			h.state = harmonyBody

		case harmonyBody:
			i := strings.Index(h.buf, harmonyOpen)
			if i < 0 {
				keep := 0
				if strings.HasSuffix(h.buf, "<") {
					keep = 1
				}
				if len(h.buf) > keep {
					body := h.buf[:len(h.buf)-keep]
					h.buf = h.buf[len(h.buf)-keep:]
					h.emitBody(body, emit)
				}
				return
			}
			if i > 0 {
				body := h.buf[:i]
				h.buf = h.buf[i:]
				h.emitBody(body, emit)
				continue
			}

			name, rest, ok := harmonyControl(h.buf)
			if !ok {
				return
			}
			h.buf = rest
			h.seenControl = true
			switch name {
			case "end", "return", "endoftext", "call", "start":
				h.state = harmonySeek
				h.channel = ""
			case "channel":
				h.state = harmonyHeader
			case "message", "constrain":
				// Harmless inside a body.
			default:
				// Unknown control token: drop it.
			}
		}
	}
}

func (h *harmonySplitter) emitBody(body string, emit func(reasoning bool, text string)) {
	if body == "" {
		return
	}
	reasoning := h.channel != harmonyChanFinal
	if reasoning {
		h.reasoning.WriteString(body)
	}
	h.emitted = true
	emit(reasoning, body)
}

// harmonyControl splits a complete "<|name|>" control token off the front of s.
// ok is false when s does not (yet) hold a complete token.
func harmonyControl(s string) (name, rest string, ok bool) {
	if !strings.HasPrefix(s, harmonyOpen) {
		return "", s, false
	}
	j := strings.Index(s[len(harmonyOpen):], "|>")
	if j < 0 {
		return "", s, false
	}
	end := len(harmonyOpen) + j + len("|>")
	return s[len(harmonyOpen) : len(harmonyOpen)+j], s[end:], true
}

// harmonyChannelName extracts the channel name from a channel header. The
// header may carry a recipient and a constraint before <|message|>, e.g.
// "commentary to=functions.get_weather <|constrain|>json".
func harmonyChannelName(header string) string {
	header = strings.TrimSpace(header)
	if i := strings.IndexAny(header, " \t\r\n"); i >= 0 {
		header = header[:i]
	}
	switch header {
	case harmonyChanAnalysis, harmonyChanCommentary, harmonyChanFinal:
		return header
	case "":
		// No channel named: treat as the answer rather than dropping it.
		return harmonyChanFinal
	}
	// Unknown channel name: route to the reasoning side so it cannot leak into
	// the visible answer.
	return harmonyChanCommentary
}
