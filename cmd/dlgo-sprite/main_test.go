package main

import "testing"

func TestSlug(t *testing.T) {
	cases := map[string]string{
		"pixel art sprite of a knight":       "knight",
		"pixel art sprite of a blue slime":   "blue-slime",
		"Pixel Art Sprite of a Treasure Chest": "treasure-chest",
		"a wizard":                           "wizard",
		"sprite":                             "sprite", // all filler -> fallback keeps words
		"":                                   "sprite",
		"!!!":                                "sprite",
	}
	for in, want := range cases {
		if got := slug(in); got != want {
			t.Errorf("slug(%q) = %q, want %q", in, got, want)
		}
	}
}
