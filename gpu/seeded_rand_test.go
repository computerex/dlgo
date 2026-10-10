//go:build cgo && vulkan

package gpu

import "math/rand"

func seededRand() *rand.Rand { return rand.New(rand.NewSource(4242)) }
