package server

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// decodeImageRef turns an OpenAI-style image reference into an image.Image.
// Supported forms:
//
//	data:image/png;base64,<b64>
//	file:///abs/path or a bare local path
//	http(s):// URL
func decodeImageRef(ref string) (image.Image, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return nil, fmt.Errorf("empty image reference")
	}

	var raw []byte
	switch {
	case strings.HasPrefix(ref, "data:"):
		comma := strings.IndexByte(ref, ',')
		if comma < 0 {
			return nil, fmt.Errorf("malformed data URI")
		}
		meta, data := ref[:comma], ref[comma+1:]
		if strings.Contains(meta, ";base64") {
			b, err := base64.StdEncoding.DecodeString(data)
			if err != nil {
				return nil, fmt.Errorf("base64 decode: %w", err)
			}
			raw = b
		} else {
			raw = []byte(data)
		}
	case strings.HasPrefix(ref, "http://"), strings.HasPrefix(ref, "https://"):
		client := &http.Client{Timeout: 30 * time.Second}
		resp, err := client.Get(ref)
		if err != nil {
			return nil, fmt.Errorf("fetch image: %w", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("fetch image: status %d", resp.StatusCode)
		}
		b, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
		if err != nil {
			return nil, fmt.Errorf("read image: %w", err)
		}
		raw = b
	default:
		path := strings.TrimPrefix(ref, "file://")
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read image file: %w", err)
		}
		raw = b
	}

	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}
