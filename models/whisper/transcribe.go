package whisper

import "github.com/computerex/dlgo/ops"

// Token IDs for Whisper special tokens
const (
	TokenSOT  = 50257 // Start of transcript
	TokenEOT  = 50256 // End of transcript
	TokenSOL  = 50361 // Start of language (optional)
	TokenNot  = 50362 // No speech
	TokenPrev = 50363 // Previous segment
)

// Transcribe runs the full Whisper pipeline: encode audio, then greedy decode until EOT.
// samples: raw audio samples (mono, typically 16kHz)
// For now, samples are assumed to be pre-processed mel spectrogram features [nMels * nFrames].
// If samples are raw audio, the caller must convert to mel first.
//
// Returns the decoded text as a string. For a minimal implementation, returns token IDs
// as a string representation; a full implementation would use a vocabulary to map to text.
func (m *WhisperModel) Transcribe(samples []float32) (string, error) {
	// Encode: samples are mel features [nFrames * nMels]
	encOut := EncodeAudio(m, samples)
	if encOut == nil {
		return "", nil
	}

	cfg := m.Config
	dim := cfg.DModel
	encLen := len(encOut) / dim

	kc := NewKVCache(cfg)
	kc.FillCross(encOut, encLen, dim, m.DecLayers)

	// Prefill with SOT (start of transcript)
	tokens := []int32{TokenSOT}
	pos := 0

	logits := DecoderStep(m, encOut, encLen, TokenSOT, pos, kc)
	pos++

	// Greedy decode until EOT or max length
	maxTokens := cfg.NTextCtx - 1
	for pos < maxTokens {
		nextToken := int32(ops.Argmax(logits))
		if nextToken == TokenEOT {
			break
		}
		tokens = append(tokens, nextToken)

		logits = DecoderStep(m, encOut, encLen, nextToken, pos, kc)
		pos++
	}

	// Convert token IDs to string (simplified: just return count for now)
	// A full implementation would use the Whisper vocabulary to decode to text
	return tokensToString(tokens), nil
}

// tokensToString converts token IDs to a string.
// Minimal implementation: returns placeholder; real impl would use vocab.
func tokensToString(tokens []int32) string {
	if len(tokens) == 0 {
		return ""
	}
	// Placeholder: in production, use Whisper's vocabulary (e.g. multilingual)
	// to map token IDs to UTF-8 text
	var buf []byte
	for _, t := range tokens {
		if t == TokenSOT || t == TokenEOT || t == TokenSOL || t == TokenNot || t == TokenPrev {
			continue // skip special tokens in output
		}
		if t < 0 || t >= 256 {
			continue
		}
		buf = append(buf, byte(t))
	}
	return string(buf)
}
