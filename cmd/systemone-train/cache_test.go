package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFeatureCacheKeyStableAndSensitive(t *testing.T) {
	base := example{ID: "a", State: "state text", Instructions: "do", Target: "x"}
	k1 := featureCacheKey(base, "model.gguf", 512, "cpu")
	k2 := featureCacheKey(base, "model.gguf", 512, "cpu")
	if k1 != k2 {
		t.Fatal("same inputs must produce the same cache key")
	}
	if len(k1) != 64 {
		t.Fatalf("key length = %d, want 64 (sha256 hex)", len(k1))
	}
	// The empty device is treated as cpu: legacy call sites keep old keys.
	if featureCacheKey(base, "model.gguf", 512, "") != k1 {
		t.Fatal("empty device must share the cpu cache key")
	}

	// Any meaningful change must change the key.
	cases := []example{
		{ID: "b", State: "state text", Instructions: "do", Target: "x"},
		{ID: "a", State: "state text 2", Instructions: "do", Target: "x"},
		{ID: "a", State: "state text", Instructions: "do2", Target: "x"},
		{ID: "a", State: "state text", Instructions: "do", Target: "y"},
	}
	for i, c := range cases {
		if featureCacheKey(c, "model.gguf", 512, "cpu") == k1 {
			t.Fatalf("case %d did not change the cache key", i)
		}
	}
	if featureCacheKey(base, "other.gguf", 512, "cpu") == k1 {
		t.Fatal("model path change did not change the cache key")
	}
	if featureCacheKey(base, "model.gguf", 1024, "cpu") == k1 {
		t.Fatal("context change did not change the cache key")
	}

	// Code assignments must change the key (they alter the prompt), and the
	// canonical key must not depend on map iteration order: two logically
	// identical examples share one key regardless of how their Codes maps were
	// built.
	withCodes := base
	withCodes.Codes = map[string]string{"x": "B", "y": "A"}
	if featureCacheKey(withCodes, "model.gguf", 512, "cpu") == k1 {
		t.Fatal("codes change did not change the cache key")
	}
	sameCodesOtherBuild := example{ID: "a", State: "state text", Instructions: "do", Target: "x", Codes: map[string]string{"y": "A", "x": "B"}}
	if featureCacheKey(sameCodesOtherBuild, "model.gguf", 512, "cpu") != featureCacheKey(withCodes, "model.gguf", 512, "cpu") {
		t.Fatal("identical code assignments must produce identical keys")
	}

	// GPU features are a different distribution: the device must change the
	// key so GPU entries can never collide with CPU entries.
	if featureCacheKey(base, "model.gguf", 512, "gpu") == k1 {
		t.Fatal("device change did not change the cache key")
	}
}

// TestLookupFeatureToleratesLegacyKeyOrder verifies that features stored under
// pre-canonical keys (randomized code-map orderings) are still found, and that
// new entries land under the canonical key.
func TestLookupFeatureToleratesLegacyKeyOrder(t *testing.T) {
	base := example{ID: "a", State: "state text", Instructions: "do", Target: "x", Codes: map[string]string{"x": "B", "y": "A"}}
	keys := featureCacheKeys(base, "model.gguf", 512, "cpu")
	if len(keys) != 2 {
		t.Fatalf("two code values must yield two candidate keys, got %d", len(keys))
	}
	if keys[0] != featureCacheKey(base, "model.gguf", 512, "cpu") {
		t.Fatal("the canonical key must come first")
	}
	dim := 3
	// Store under the LEGACY (second) key only, as an old cache would.
	cache := &cachedFeatures{Version: featureCacheVersion, Dim: dim, Entries: map[string]cacheEntry{
		keys[1]: {Feature: []float32{7, 8, 9}},
	}}
	feature, ok := lookupFeature(cache, base, "model.gguf", 512, dim, "cpu")
	if !ok {
		t.Fatal("lookupFeature must find entries stored under legacy key orderings")
	}
	if len(feature) != dim {
		t.Fatalf("feature length = %d, want %d", len(feature), dim)
	}
	// A wrong-dimension entry must not count as a hit.
	cache.Entries[keys[1]] = cacheEntry{Feature: []float32{1}}
	if _, ok := lookupFeature(cache, base, "model.gguf", 512, dim, "cpu"); ok {
		t.Fatal("lookupFeature must reject entries with the wrong dimension")
	}
}

func TestFeatureCacheRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "features.json")
	cache := &cachedFeatures{Version: featureCacheVersion, Dim: 4, Entries: map[string]cacheEntry{
		"key1": {Feature: []float32{1, 2, 3, 4}},
		"key2": {Feature: []float32{-1, -2, -3, -4}},
	}}
	if err := saveFeatureCache(path, cache); err != nil {
		t.Fatalf("saveFeatureCache: %v", err)
	}
	loaded := loadFeatureCache(path)
	if loaded.Version != featureCacheVersion || loaded.Dim != 4 || len(loaded.Entries) != 2 {
		t.Fatalf("loaded = %+v", loaded)
	}
	if got := loaded.Entries["key1"].Feature; len(got) != 4 || got[0] != 1 || got[3] != 4 {
		t.Fatalf("key1 feature = %v", got)
	}

	// A missing file yields a usable empty cache.
	empty := loadFeatureCache(filepath.Join(t.TempDir(), "missing.json"))
	if empty.Entries == nil || len(empty.Entries) != 0 {
		t.Fatalf("missing-file cache = %+v, want empty usable cache", empty)
	}

	// Corrupt JSON yields an empty cache, not a crash.
	corrupt := filepath.Join(t.TempDir(), "corrupt.json")
	os.WriteFile(corrupt, []byte("{not json"), 0o644)
	bad := loadFeatureCache(corrupt)
	if bad.Entries == nil || len(bad.Entries) != 0 {
		t.Fatalf("corrupt cache = %+v, want empty usable cache", bad)
	}

	// A wrong version is rejected.
	wrongVersion := filepath.Join(t.TempDir(), "wrongver.json")
	os.WriteFile(wrongVersion, []byte(`{"version":999,"dim":4,"entries":{}}`), 0o644)
	stale := loadFeatureCache(wrongVersion)
	if len(stale.Entries) != 0 {
		t.Fatalf("stale version cache = %+v, want empty usable cache", stale)
	}
}