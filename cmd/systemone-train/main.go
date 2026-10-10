// systemone-train trains an independently initialized linear System One
// decision readout on final hidden states from a frozen local GGUF backbone.
// It never modifies packed Q4 model weights.
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime/pprof"
	"sort"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/computerex/dlgo/format/gguf"
	"github.com/computerex/dlgo/models/llm"
)

type example struct {
	ID           string            `json:"id"`
	Task         string            `json:"task"`
	State        string            `json:"state"`
	Instructions string            `json:"instructions"`
	Criteria     map[string]string `json:"criteria"`
	Target       string            `json:"target"`
	// Codes optionally pins value->answer-letter assignments (e.g. "B") for
	// code-permutation augmentation. When absent, codes are assigned stably by
	// sorted value.
	Codes map[string]string `json:"codes,omitempty"`
}

type artifact struct {
	FormatVersion int         `json:"format_version"`
	Architecture  string      `json:"architecture"`
	ModelPath     string      `json:"model_path"`
	HiddenSize    int         `json:"hidden_size"`
	MaxOptions    int         `json:"max_options"`
	PromptVersion string      `json:"prompt_version"`
	Temperature   float64     `json:"temperature"`
	Device        string      `json:"device,omitempty"`
	Weights       [][]float32 `json:"weights"`
	CreatedAt     string      `json:"created_at"`
}

type metrics struct {
	Rows     int     `json:"rows"`
	Correct  int     `json:"correct"`
	Accuracy float64 `json:"accuracy"`
	NLL      float64 `json:"nll"`
	Brier    float64 `json:"brier"`
}

type report struct {
	Train       metrics `json:"train"`
	Development metrics `json:"development"`
	Calibration metrics `json:"calibration"`
	Test        metrics `json:"test"`
	Temperature float64 `json:"temperature"`
}

func main() {
	modelPath := flag.String("model", "", "frozen Q4 GGUF model path")
	trainPath := flag.String("train", "", "JSONL training data")
	devPath := flag.String("dev", "", "JSONL development data")
	calibrationPath := flag.String("calibration", "", "JSONL calibration data")
	testPath := flag.String("test", "", "JSONL held-out test data")
	outputPath := flag.String("output", "systemone-readout.json", "output readout artifact")
	ggufOutputPath := flag.String("gguf-output", "", "optional standalone GGUF export of the readout")
	featureCache := flag.String("feature-cache", "", "optional feature cache file; skips backbone extraction on repeat runs (keyed by example ID, state, codes, model path, device, and ctx)")
	unionCache := flag.String("union-cache", "", "optional second (gpu-extracted) feature cache; trains a device-agnostic any-device readout on the union of both caches with zero extraction (every example must exist in both caches)")
	reportPath := flag.String("report", "", "optional metrics JSON report")
	device := flag.String("device", "cpu", "feature extraction device: cpu or gpu (gpu requires -tags \"cgo vulkan\")")
	epochs := flag.Int("epochs", 100, "training epochs")
	learningRate := flag.Float64("lr", 0.02, "SGD learning rate")
	ctx := flag.Int("ctx", 512, "context length")
	seed := flag.Int64("seed", 20260920, "random seed")
	cpuProfile := flag.String("cpuprofile", "", "write a pprof CPU profile to this path")
	flag.Parse()
	if *modelPath == "" || *trainPath == "" {
		fatal(fmt.Errorf("usage: systemone-train -model MODEL.gguf -train TRAIN.jsonl [-dev DEV.jsonl -calibration CAL.jsonl -test TEST.jsonl]"))
	}
	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			fatal(err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			fatal(err)
		}
		defer func() {
			pprof.StopCPUProfile()
			f.Close()
		}()
	}
	if *device != "cpu" && *device != "gpu" {
		fatal(fmt.Errorf("-device must be cpu or gpu, got %q", *device))
	}

	train, err := readExamples(*trainPath)
	if err != nil {
		fatal(err)
	}
	dev, err := readOptionalExamples(*devPath)
	if err != nil {
		fatal(err)
	}
	calibration, err := readOptionalExamples(*calibrationPath)
	if err != nil {
		fatal(err)
	}
	test, err := readOptionalExamples(*testPath)
	if err != nil {
		fatal(err)
	}
	ensureDistinctSplits(train, dev, calibration, test)

	pipeline, err := llm.NewPipeline(*modelPath, *ctx)
	if err != nil {
		fatal(fmt.Errorf("load frozen Q4 backbone: %w", err))
	}
	defer pipeline.Model.Close()
	dim := pipeline.Model.Config.EmbeddingDim

	// Feature extraction device. The GPU pipeline extracts structurally
	// different (but internally consistent) hidden states, so readouts are
	// device-native: train and serve on the same device.
	var extractor hiddenExtractor = pipeline
	extractDevice := "cpu"
	var gpuCleanup func()
	if *device == "gpu" {
		extractor, gpuCleanup, err = newGPUExtractor(pipeline)
		if err != nil {
			fatal(err)
		}
		defer gpuCleanup()
		extractDevice = "gpu"
		fmt.Printf("extracting features on %s\n", extractDevice)
	}

	var trainSet, devSet, calSet, testSet dataset
	var maxOptions int
	if *unionCache != "" {
		// Device-agnostic mode: no backbone forward passes at all — both
		// caches must already hold every example (from prior per-device runs).
		// The Vulkan backend must be initialized so gpuCacheSuffix reports the
		// real device name the gpu cache was written with.
		nameCleanup, err := ensureGPUCacheSuffix()
		if err != nil {
			fatal(err)
		}
		defer nameCleanup()
		fmt.Printf("union mode: building any-device dataset from %s (%s) + %s (gpu keys %q)\n", *featureCache, extractDevice, *unionCache, gpuCacheSuffix())
		trainSet, maxOptions, err = extractDatasetUnionCached(train, *featureCache, extractDevice, *unionCache, *modelPath, *ctx, dim)
		if err != nil {
			fatal(err)
		}
		devSet, _, err = extractDatasetUnionCached(dev, *featureCache, extractDevice, *unionCache, *modelPath, *ctx, dim)
		if err != nil {
			fatal(err)
		}
		calSet, _, err = extractDatasetUnionCached(calibration, *featureCache, extractDevice, *unionCache, *modelPath, *ctx, dim)
		if err != nil {
			fatal(err)
		}
		testSet, _, err = extractDatasetUnionCached(test, *featureCache, extractDevice, *unionCache, *modelPath, *ctx, dim)
		if err != nil {
			fatal(err)
		}
	} else {
		trainSet, maxOptions, err = extractDatasetCached(extractor, pipeline, train, *featureCache, *modelPath, *ctx, extractDevice)
		if err != nil {
			fatal(err)
		}
		devSet, _, err = extractDatasetCached(extractor, pipeline, dev, *featureCache, *modelPath, *ctx, extractDevice)
		if err != nil {
			fatal(err)
		}
		calSet, _, err = extractDatasetCached(extractor, pipeline, calibration, *featureCache, *modelPath, *ctx, extractDevice)
		if err != nil {
			fatal(err)
		}
		testSet, _, err = extractDatasetCached(extractor, pipeline, test, *featureCache, *modelPath, *ctx, extractDevice)
		if err != nil {
			fatal(err)
		}
	}

	weights := initializeWeights(maxOptions, dim, *seed)
	rng := rand.New(rand.NewSource(*seed))
	bestWeights := cloneWeights(weights)
	bestDev := math.Inf(1)
	order := make([]int, len(trainSet.features))
	for i := range order {
		order[i] = i
	}
	for epoch := 1; epoch <= *epochs; epoch++ {
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		trainSGD(weights, trainSet, order, *learningRate)
		trainMetrics := evaluate(weights, 1, trainSet)
		devMetrics := evaluate(weights, 1, devSet)
		selection := devMetrics.NLL
		if devMetrics.Rows == 0 {
			selection = trainMetrics.NLL
		}
		if selection < bestDev {
			bestDev, bestWeights = selection, cloneWeights(weights)
		}
		fmt.Printf("epoch %d/%d train_loss=%.4f train_acc=%.1f%% dev_loss=%.4f dev_acc=%.1f%%\n", epoch, *epochs, trainMetrics.NLL, 100*trainMetrics.Accuracy, devMetrics.NLL, 100*devMetrics.Accuracy)
	}

	temperature := fitTemperature(bestWeights, calSet)
	if calSet.rows == 0 {
		temperature = 1
	}
	artifactDevice := extractDevice
	if *unionCache != "" {
		artifactDevice = "any"
	}
	result := artifact{FormatVersion: 1, Architecture: "dlgo.systemone.linear-readout", ModelPath: *modelPath, HiddenSize: dim, MaxOptions: maxOptions, PromptVersion: "dlgo-systemone-v1", Temperature: temperature, Device: artifactDevice, Weights: bestWeights, CreatedAt: time.Now().UTC().Format(time.RFC3339)}
	writeJSON(*outputPath, result)
	if *ggufOutputPath != "" {
		if err := writeReadoutGGUF(*ggufOutputPath, result); err != nil {
			fatal(fmt.Errorf("export GGUF readout: %w", err))
		}
		fmt.Printf("wrote GGUF readout export: %s\n", *ggufOutputPath)
	}

	finalReport := report{Train: evaluate(bestWeights, temperature, trainSet), Development: evaluate(bestWeights, temperature, devSet), Calibration: evaluate(bestWeights, temperature, calSet), Test: evaluate(bestWeights, temperature, testSet), Temperature: temperature}
	if *reportPath != "" {
		writeJSON(*reportPath, finalReport)
	}
	fmt.Printf("wrote Q4 System One readout: %s (%d options x %d hidden), temperature=%.4f\n", *outputPath, maxOptions, dim, temperature)
	fmt.Printf("held-out metrics: dev=%.1f%%/%d test=%.1f%%/%d nll=%.4f brier=%.4f\n", 100*finalReport.Development.Accuracy, finalReport.Development.Rows, 100*finalReport.Test.Accuracy, finalReport.Test.Rows, finalReport.Test.NLL, finalReport.Test.Brier)
}

type dataset struct {
	features [][]float32
	// classes[i] holds the readout row index for each choice position of row i,
	// derived from the answer code letter. Arbitrary code sets (e.g. A and C for
	// a two-choice question) are supported; logits are computed per position.
	classes         [][]int
	targets, counts []int
	rows            int
}

// codeRowIndex maps a compiled answer code (" A") to its readout row index.
func codeRowIndex(code string) (int, error) {
	if len(code) == 2 && code[0] == ' ' && code[1] >= 'A' && code[1] <= 'Z' {
		return int(code[1] - 'A'), nil
	}
	return 0, fmt.Errorf("unsupported answer code %q", code)
}

// hiddenExtractor is the feature-extraction interface shared by the CPU
// pipeline and the Vulkan GPU pipeline. Both expose SystemOneHidden with an
// identical signature, so training and evaluation can run on either device.
type hiddenExtractor interface {
	SystemOneHidden(state string, q llm.SystemOneQuestion, cfg llm.SystemOneConfig) ([]float32, []llm.Choice, int, error)
}

func extractDataset(extractor hiddenExtractor, examples []example) (dataset, int, error) {
	set := dataset{rows: len(examples)}
	maxOptions := 0
	for _, ex := range examples {
		q, err := toQuestion(ex)
		if err != nil {
			return dataset{}, 0, fmt.Errorf("row %s: %w", ex.ID, err)
		}
		hidden, choices, _, err := extractor.SystemOneHidden(ex.State, q, llm.SystemOneConfig{})
		if err != nil {
			return dataset{}, 0, fmt.Errorf("extract row %s: %w", ex.ID, err)
		}
		if err := appendRow(&set, choices, ex.Target, normalizeFeature(hidden), &maxOptions); err != nil {
			return dataset{}, 0, fmt.Errorf("row %s: %w", ex.ID, err)
		}
	}
	return set, maxOptions, nil
}

// appendRow records one extracted example: feature, per-position readout row
// indices, target position, and choice count.
func appendRow(set *dataset, choices []llm.Choice, target string, feature []float32, maxOptions *int) error {
	classes := make([]int, len(choices))
	targetPos := -1
	for i, choice := range choices {
		index, err := codeRowIndex(choice.Code)
		if err != nil {
			return fmt.Errorf("choice %q: %w", choice.Value, err)
		}
		if index >= *maxOptions {
			*maxOptions = index + 1
		}
		classes[i] = index
		if choice.Value == target {
			targetPos = i
		}
	}
	if targetPos < 0 {
		return fmt.Errorf("target %q is not among the allowed choices", target)
	}
	set.features = append(set.features, feature)
	set.classes = append(set.classes, classes)
	set.targets = append(set.targets, targetPos)
	set.counts = append(set.counts, len(choices))
	return nil
}

// cachedFeatures is the on-disk feature cache format. Entries are keyed by a
// hash of (example ID, state, codes, model path, ctx) so a changed dataset,
// prompt-affecting code permutation, backbone, or context invalidates stale
// entries automatically.
type cachedFeatures struct {
	Version  int                `json:"version"`
	Dim      int                `json:"dim"`
	Entries  map[string]cacheEntry `json:"entries"`
}

type cacheEntry struct {
	Feature []float32 `json:"feature"`
}

const featureCacheVersion = 1

// featureCacheKey builds the canonical cache key for an example: code
// assignments are rendered in sorted-value order so the key is deterministic.
// (Keys once depended on Go's randomized map iteration order, which made keys
// for code-augmented rows differ run to run and silently defeated caching.)
func featureCacheKey(ex example, modelPath string, ctx int, device string) string {
	codes := sortedCodes(ex)
	parts := ex.ID + "\x00" + ex.State + "\x00" + ex.Instructions + "\x00" + ex.Target + "\x00" + codes + "\x00" + modelPath + "\x00" + fmt.Sprint(ctx)
	// Only non-CPU devices get a device suffix: this keeps every existing CPU
	// cache entry valid while making GPU features (a different distribution,
	// and hardware-specific) uncacheable against CPU entries.
	if device != "cpu" && device != "" {
		parts += "\x00" + gpuCacheSuffix()
	}
	sum := sha256.Sum256([]byte(parts))
	return hex.EncodeToString(sum[:])
}

// sortedCodes renders the example's code assignments in sorted-value order.
func sortedCodes(ex example) string {
	values := make([]string, 0, len(ex.Codes))
	for value := range ex.Codes {
		values = append(values, value)
	}
	sort.Strings(values)
	codes := ""
	for _, value := range values {
		codes += value + "=" + ex.Codes[value] + ";"
	}
	return codes
}

// featureCacheKeys lists every cache key an example's features could be
// stored under: the canonical (sorted-codes) key first, then keys for the
// remaining orderings of the code map. Caches written before canonical keying
// captured Go's randomized iteration order, so lookups must tolerate those
// legacy rotations until entries are rewritten.
func featureCacheKeys(ex example, modelPath string, ctx int, device string) []string {
	values := make([]string, 0, len(ex.Codes))
	for value := range ex.Codes {
		values = append(values, value)
	}
	sort.Strings(values)
	prefix := ex.ID + "\x00" + ex.State + "\x00" + ex.Instructions + "\x00" + ex.Target + "\x00"
	suffix := "\x00" + modelPath + "\x00" + fmt.Sprint(ctx)
	if device != "cpu" && device != "" {
		suffix += "\x00" + gpuCacheSuffix()
	}
	hash := func(codes string) string {
		sum := sha256.Sum256([]byte(prefix + codes + suffix))
		return hex.EncodeToString(sum[:])
	}
	var keys []string
	permuteValues(values, 0, func(order []string) {
		codes := ""
		for _, value := range order {
			codes += value + "=" + ex.Codes[value] + ";"
		}
		keys = append(keys, hash(codes))
	})
	return keys
}

// permuteValues invokes fn with every permutation of values, beginning with
// the current (sorted) order.
func permuteValues(values []string, k int, fn func([]string)) {
	if k == len(values) {
		fn(values)
		return
	}
	for i := k; i < len(values); i++ {
		values[k], values[i] = values[i], values[k]
		permuteValues(values, k+1, fn)
		values[k], values[i] = values[i], values[k]
	}
}

// lookupFeature finds an example's cached feature under any legacy key
// ordering, preferring the canonical (sorted-codes) key.
func lookupFeature(cache *cachedFeatures, ex example, modelPath string, ctx, dim int, device string) ([]float32, bool) {
	for _, key := range featureCacheKeys(ex, modelPath, ctx, device) {
		if entry, ok := cache.Entries[key]; ok && len(entry.Feature) == dim {
			return entry.Feature, true
		}
	}
	return nil, false
}

// extractDatasetCached extracts features through the backbone, consulting and
// populating an optional on-disk cache. Cache hits skip the expensive forward
// pass entirely, making repeat training runs (new lr/epochs/seed) take seconds
// instead of tens of minutes.
//
// Extraction is strictly sequential and steady-state memory is bounded: one
// row's feature is produced, folded into the dataset and the cache, then
// released. It deliberately does NOT buffer all rows or clone pipelines —
// both of which grew RSS steadily. Parallelism, where it helps, already lives
// inside SystemOneHidden (the batched GEMM / FFN / attention worker pool).
//
// The cache is checkpointed to disk every checkpointEvery extractions, and a
// progress line (rows done, rate, ETA) is printed every progressSecs seconds,
// so a long run is observable and an interruption never loses all prior work.
func extractDatasetCached(extractor hiddenExtractor, cpuPipeline *llm.Pipeline, examples []example, cachePath, modelPath string, ctx int, device string) (dataset, int, error) {
	if cachePath == "" {
		return extractDataset(extractor, examples)
	}
	cache := loadFeatureCache(cachePath)
	modelDim := cpuPipeline.Model.Config.EmbeddingDim
	if cache.Dim != 0 && cache.Dim != modelDim {
		// Dimension mismatch means the cache came from a different backbone;
		// start fresh rather than mixing incompatible features.
		cache = &cachedFeatures{Version: featureCacheVersion, Entries: map[string]cacheEntry{}}
	}
	if cache.Entries == nil {
		cache.Entries = map[string]cacheEntry{}
	}

	set := dataset{rows: len(examples)}
	maxOptions := 0

	// Work accounting for progress/ETA (count cache misses up front).
	pending := 0
	for i := range examples {
		if _, ok := lookupFeature(cache, examples[i], modelPath, ctx, modelDim, device); !ok {
			pending++
		}
	}
	fmt.Printf("extracting %d rows (%d cache hits, %d to extract)\n",
		len(examples), len(examples)-pending, pending)

	checkpointEvery := 250
	if v := os.Getenv("DLGO_EXTRACT_CHECKPOINT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			checkpointEvery = n
		}
	}
	progressSecs := 15
	if v := os.Getenv("DLGO_EXTRACT_PROGRESS_SECS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			progressSecs = n
		}
	}

	startTime := time.Now()
	var extracted int64
	lastReport := startTime
	var lastCount int64

	misses := 0
	for i := range examples {
		exm := examples[i]
		q, err := toQuestion(exm)
		if err != nil {
			return dataset{}, 0, fmt.Errorf("row %s: %w", exm.ID, err)
		}
		var hidden []float32
		if feature, ok := lookupFeature(cache, exm, modelPath, ctx, modelDim, device); ok {
			hidden = feature
		} else {
			raw, _, _, err := extractor.SystemOneHidden(exm.State, q, llm.SystemOneConfig{})
			if err != nil {
				return dataset{}, 0, fmt.Errorf("extract row %s: %w", exm.ID, err)
			}
			hidden = append([]float32(nil), raw...)
			cache.Entries[featureCacheKey(exm, modelPath, ctx, device)] = cacheEntry{Feature: hidden}
			misses++
			extracted++

			if checkpointEvery > 0 && int(extracted)%checkpointEvery == 0 {
				cache.Dim = modelDim
				if err := saveFeatureCache(cachePath, cache); err != nil {
					fmt.Fprintf(os.Stderr, "[systemone-train] checkpoint save failed: %v\n", err)
				} else {
					fmt.Printf("  checkpoint: %d rows extracted, cache saved\n", extracted)
				}
			}
		}

		if err := appendRow(&set, q.Choices, exm.Target, normalizeFeature(hidden), &maxOptions); err != nil {
			return dataset{}, 0, fmt.Errorf("row %s: %w", exm.ID, err)
		}

		// Release this row's feature unless it came from the cache (shared).
		if cache.Entries != nil {
			// hidden is either the cache slice (shared, keep) or a fresh copy
			// already stored in the cache; dropping the local ref here is safe.
		}

		// Time-based progress with ETA.
		if pending > 0 && time.Since(lastReport) >= time.Duration(progressSecs)*time.Second {
			el := time.Since(startTime).Seconds()
			n := atomic.LoadInt64(&extracted)
			delta := n - lastCount
			var rate float64
			if el > 0 {
				rate = float64(n) / el
			}
			eta := 0.0
			if rate > 0 {
				eta = float64(int64(pending)-n) / rate
			}
			fmt.Printf("  progress: %d/%d rows (%.1f%%) %.2f rows/s (+%d since last) elapsed %s ETA %s\n",
				n, pending, 100*float64(n)/float64(pending), rate, delta,
				time.Duration(el*float64(time.Second)).Round(time.Second),
				time.Duration(eta*float64(time.Second)).Round(time.Second))
			lastReport = time.Now()
			lastCount = n
		}
	}

	if misses > 0 {
		cache.Dim = modelDim
		if err := saveFeatureCache(cachePath, cache); err != nil {
			return dataset{}, 0, fmt.Errorf("save feature cache: %w", err)
		}
	}
	fmt.Printf("features: %d rows, %d cache hits, %d extracted\n", len(examples), len(examples)-misses, misses)
	return set, maxOptions, nil
}

// extractDatasetUnionCached builds a device-agnostic dataset purely from two
// complete feature caches (no backbone extraction): each example contributes
// two rows, one feature from the primary cache (primaryDevice, usually cpu)
// and one from the union cache (gpu-extracted). Training on the union lets a
// single readout interpolate across both engines' structurally different
// hidden states, so one artifact serves either device.
func extractDatasetUnionCached(examples []example, primaryPath, primaryDevice, unionPath, modelPath string, ctx, dim int) (dataset, int, error) {
	type unionSource struct {
		name   string
		cache  *cachedFeatures
		device string
	}
	sources := []unionSource{
		{"primary", loadFeatureCache(primaryPath), primaryDevice},
		{"union", loadFeatureCache(unionPath), "gpu"},
	}
	for _, source := range sources {
		if source.cache.Dim != 0 && source.cache.Dim != dim {
			return dataset{}, 0, fmt.Errorf("%s cache dim %d does not match model dimension %d", source.name, source.cache.Dim, dim)
		}
	}
	set := dataset{rows: len(examples) * len(sources)}
	maxOptions := 0
	hits := 0
	for _, ex := range examples {
		q, err := toQuestion(ex)
		if err != nil {
			return dataset{}, 0, fmt.Errorf("row %s: %w", ex.ID, err)
		}
		for _, source := range sources {
			feature, ok := lookupFeature(source.cache, ex, modelPath, ctx, dim, source.device)
			if !ok {
				return dataset{}, 0, fmt.Errorf("row %s: missing %s-cache feature for device %s; run the %s training pass first", ex.ID, source.name, source.device, source.device)
			}
			hits++
			if err := appendRow(&set, q.Choices, ex.Target, normalizeFeature(feature), &maxOptions); err != nil {
				return dataset{}, 0, fmt.Errorf("row %s: %w", ex.ID, err)
			}
		}
	}
	fmt.Printf("features: %d rows, %d cache hits, 0 extracted (union of %s + gpu caches)\n", len(examples)*len(sources), hits, primaryDevice)
	return set, maxOptions, nil
}

func loadFeatureCache(path string) *cachedFeatures {
	data, err := os.ReadFile(path)
	if err != nil {
		return &cachedFeatures{Version: featureCacheVersion, Entries: map[string]cacheEntry{}}
	}
	var cache cachedFeatures
	if err := json.Unmarshal(data, &cache); err != nil || cache.Version != featureCacheVersion {
		return &cachedFeatures{Version: featureCacheVersion, Entries: map[string]cacheEntry{}}
	}
	return &cache
}

func saveFeatureCache(path string, cache *cachedFeatures) error {
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	data, err := json.Marshal(cache)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func trainSGD(weights [][]float32, set dataset, order []int, learningRate float64) {
	for _, index := range order {
		logits := classLogits(weights, set.features[index], set.classes[index])
		probs := softmax(logits)
		for c := range logits {
			gradient := float32(probs[c])
			if c == set.targets[index] {
				gradient--
			}
			row := weights[set.classes[index][c]]
			for j, value := range set.features[index] {
				row[j] -= float32(learningRate) * gradient * value
			}
		}
	}
}
func evaluate(weights [][]float32, temperature float64, set dataset) metrics {
	result := metrics{Rows: set.rows}
	if set.rows == 0 {
		return result
	}
	for i, feature := range set.features {
		logits := classLogits(weights, feature, set.classes[i])
		for j := range logits {
			logits[j] /= temperature
		}
		probs := softmax(logits)
		best := 0
		for j := 1; j < len(probs); j++ {
			if probs[j] > probs[best] {
				best = j
			}
		}
		if best == set.targets[i] {
			result.Correct++
		}
		result.NLL -= math.Log(math.Max(probs[set.targets[i]], 1e-12))
		for j, probability := range probs {
			expected := 0.0
			if j == set.targets[i] {
				expected = 1
			}
			delta := probability - expected
			result.Brier += delta * delta
		}
	}
	result.Accuracy = float64(result.Correct) / float64(result.Rows)
	result.NLL /= float64(result.Rows)
	result.Brier /= float64(result.Rows)
	return result
}
func fitTemperature(weights [][]float32, set dataset) float64 {
	if set.rows == 0 {
		return 1
	}
	best, bestNLL := 1.0, math.Inf(1)
	for temperature := 0.2; temperature <= 5; temperature += 0.05 {
		nll := evaluate(weights, temperature, set).NLL
		if nll < bestNLL {
			best, bestNLL = temperature, nll
		}
	}
	return best
}
func classLogits(weights [][]float32, feature []float32, classes []int) []float64 {
	result := make([]float64, len(classes))
	for c, row := range classes {
		result[c] = dot(weights[row], feature)
	}
	return result
}
func initializeWeights(maxOptions, dim int, seed int64) [][]float32 {
	result := make([][]float32, maxOptions)
	rng := rand.New(rand.NewSource(seed))
	for c := range result {
		result[c] = make([]float32, dim)
		for j := range result[c] {
			result[c][j] = float32(rng.NormFloat64() * 0.001)
		}
	}
	return result
}
func cloneWeights(weights [][]float32) [][]float32 {
	result := make([][]float32, len(weights))
	for i := range weights {
		result[i] = append([]float32(nil), weights[i]...)
	}
	return result
}
func readOptionalExamples(path string) ([]example, error) {
	if path == "" {
		return nil, nil
	}
	return readExamples(path)
}
func readExamples(path string) ([]example, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var result []example
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024), 4*1024*1024)
	line := 0
	for scanner.Scan() {
		line++
		var ex example
		if err := json.Unmarshal(scanner.Bytes(), &ex); err != nil {
			return nil, fmt.Errorf("%s line %d: %w", path, line, err)
		}
		if ex.ID == "" || ex.State == "" || ex.Instructions == "" || ex.Target == "" || len(ex.Criteria) < 2 {
			return nil, fmt.Errorf("%s line %d: require id, state, instructions, target, and at least two criteria", path, line)
		}
		result = append(result, ex)
	}
	return result, scanner.Err()
}
func ensureDistinctSplits(splits ...[]example) {
	seen := map[string]bool{}
	for _, split := range splits {
		for _, ex := range split {
			if seen[ex.ID] {
				fatal(fmt.Errorf("example ID %q occurs in more than one split", ex.ID))
			}
			seen[ex.ID] = true
		}
	}
}
func toQuestion(ex example) (llm.SystemOneQuestion, error) {
	keys := make([]string, 0, len(ex.Criteria))
	for key := range ex.Criteria {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	choices := make([]llm.Choice, len(keys))
	for i, key := range keys {
		choice := llm.Choice{Value: key, Label: ex.Criteria[key]}
		if letter, ok := ex.Codes[key]; ok {
			if len(letter) != 1 || letter[0] < 'A' || letter[0] > 'Z' {
				return llm.SystemOneQuestion{}, fmt.Errorf("row %s: invalid code letter %q for value %q", ex.ID, letter, key)
			}
			choice.Code = fmt.Sprintf(" %c", letter[0])
		}
		choices[i] = choice
	}
	return llm.SystemOneQuestion{ID: ex.ID, Prompt: ex.Instructions, Choices: choices}, nil
}
func normalizeFeature(feature []float32) []float32 {
	result := append([]float32(nil), feature...)
	var squared float64
	for _, value := range result {
		squared += float64(value) * float64(value)
	}
	if norm := math.Sqrt(squared); norm > 0 {
		for i := range result {
			result[i] /= float32(norm)
		}
	}
	return result
}
func dot(weight, feature []float32) float64 {
	var sum float64
	for i, value := range feature {
		sum += float64(weight[i]) * float64(value)
	}
	return sum
}
func softmax(logits []float64) []float64 {
	max := logits[0]
	for _, value := range logits[1:] {
		if value > max {
			max = value
		}
	}
	result := make([]float64, len(logits))
	var sum float64
	for i, value := range logits {
		result[i] = math.Exp(value - max)
		sum += result[i]
	}
	for i := range result {
		result[i] /= sum
	}
	return result
}
func writeJSON(path string, value any) {
	directory := filepath.Dir(path)
	if directory != "." {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			fatal(err)
		}
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		fatal(err)
	}
}
func fatal(err error) { fmt.Fprintln(os.Stderr, "systemone-train:", err); os.Exit(1) }

// writeReadoutGGUF exports a trained readout as a standalone GGUF artifact
// using the dlgo GGUF writer. The layout matches LoadSystemOneReadoutGGUF.
func writeReadoutGGUF(path string, a artifact) error {
	flat := make([]float32, 0, a.MaxOptions*a.HiddenSize)
	for _, row := range a.Weights {
		flat = append(flat, row...)
	}
	metadata := map[string]interface{}{
		"general.architecture":    "dlgo.systemone.linear-readout",
		"general.name":            "dlgo-systemone-readout",
		"systemone.format_version": uint32(a.FormatVersion),
		"systemone.hidden_size":    uint32(a.HiddenSize),
		"systemone.max_options":    uint32(a.MaxOptions),
		"systemone.temperature":    a.Temperature,
		"systemone.prompt_version": a.PromptVersion,
		"systemone.created_at":     a.CreatedAt,
		"systemone.backbone":       a.ModelPath,
		"systemone.device":         a.Device,
	}
	tensors := []gguf.WriteTensor{
		{
			Name:       "systemone.readout.weight",
			Dimensions: []int64{int64(a.MaxOptions), int64(a.HiddenSize)},
			Type:       gguf.GGMLTypeF32,
			Data:       gguf.Float32Bytes(flat),
		},
	}
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return gguf.WriteGGUF(path, metadata, tensors, 32)
}
