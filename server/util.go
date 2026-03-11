package server

import (
	"crypto/rand"
	"encoding/hex"
)

func randomHex(n int) string {
	b := make([]byte, n)
	rand.Read(b)
	return hex.EncodeToString(b)
}
