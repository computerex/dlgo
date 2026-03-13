package ops

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
)

type SamplerConfig struct {
	Temperature       float32
	TopK              int
	TopP              float32
	MinP              float32
	RepetitionPenalty float32
}

func DefaultSamplerConfig() SamplerConfig {
	return SamplerConfig{
		Temperature:       0.7,
		TopK:              40,
		TopP:              0.9,
		MinP:              0,
		RepetitionPenalty: 1.1,
	}
}

type tokenProb struct {
	idx  int
	logit float32
}

type minHeap []tokenProb

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool   { return h[i].logit < h[j].logit }
func (h minHeap) Swap(i, j int)        { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{})  { *h = append(*h, x.(tokenProb)) }
func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func SampleToken(logits []float32, cfg SamplerConfig, recentTokens []int32, rng *rand.Rand) int {
	n := len(logits)

	ApplyRepetitionPenalty(logits, recentTokens, cfg.RepetitionPenalty)

	if cfg.Temperature <= 0 || rng == nil {
		return Argmax(logits)
	}

	invTemp := float32(1.0) / cfg.Temperature
	if cfg.Temperature != 1.0 {
		for i := range logits {
			logits[i] *= invTemp
		}
	}

	k := cfg.TopK
	if k <= 0 || k > n {
		k = n
	}

	candidates := topKHeap(logits, k)

	sort.Slice(candidates, func(a, b int) bool {
		return candidates[a].logit > candidates[b].logit
	})

	nc := len(candidates)
	probs := make([]float32, nc)
	maxLogit := candidates[0].logit
	var sum float32
	for i := 0; i < nc; i++ {
		probs[i] = fastExpf(candidates[i].logit - maxLogit)
		sum += probs[i]
	}
	invSum := float32(1.0) / sum
	for i := range probs {
		probs[i] *= invSum
	}

	if cfg.MinP > 0 {
		threshold := cfg.MinP * probs[0]
		for i := 0; i < nc; i++ {
			if probs[i] < threshold {
				nc = i
				break
			}
		}
		if nc == 0 {
			nc = 1
		}
		probs = probs[:nc]
		candidates = candidates[:nc]
	}

	if cfg.TopP > 0 && cfg.TopP < 1.0 {
		var cumSum float32
		cutoff := nc
		for i := 0; i < nc; i++ {
			cumSum += probs[i]
			if cumSum >= cfg.TopP {
				cutoff = i + 1
				break
			}
		}
		nc = cutoff
		probs = probs[:nc]
		candidates = candidates[:nc]
	}

	pSum := float32(0)
	for _, p := range probs {
		pSum += p
	}
	invPSum := float32(1.0) / pSum

	r := rng.Float32()
	var cumSum float32
	for i, p := range probs {
		cumSum += p * invPSum
		if cumSum >= r {
			return candidates[i].idx
		}
	}
	return candidates[nc-1].idx
}

func topKHeap(logits []float32, k int) []tokenProb {
	h := make(minHeap, 0, k+1)

	negInf := float32(math.Inf(-1))
	for i, v := range logits {
		if v == negInf {
			continue
		}
		if h.Len() < k {
			heap.Push(&h, tokenProb{i, v})
		} else if v > h[0].logit {
			h[0] = tokenProb{i, v}
			heap.Fix(&h, 0)
		}
	}

	result := make([]tokenProb, len(h))
	copy(result, h)
	return result
}

func ApplyRepetitionPenalty(logits []float32, recentTokens []int32, penalty float32) {
	if penalty <= 1.0 || len(recentTokens) == 0 {
		return
	}
	seen := make(map[int32]bool, len(recentTokens))
	for _, tok := range recentTokens {
		if tok < 0 || int(tok) >= len(logits) || seen[tok] {
			continue
		}
		seen[tok] = true
		if logits[tok] > 0 {
			logits[tok] /= penalty
		} else {
			logits[tok] *= penalty
		}
	}
}

func ApplyTemperature(logits []float32, temp float32) {
	if temp <= 0 || temp == 1.0 {
		return
	}
	invTemp := 1.0 / temp
	for i := range logits {
		logits[i] *= invTemp
	}
}

func ApplyTopK(logits []float32, k int) {
	if k <= 0 || k >= len(logits) {
		return
	}
	candidates := topKHeap(logits, k)
	allowed := make(map[int]bool, len(candidates))
	for _, c := range candidates {
		allowed[c.idx] = true
	}
	negInf := float32(math.Inf(-1))
	for i := range logits {
		if !allowed[i] {
			logits[i] = negInf
		}
	}
}

func ApplyTopP(logits []float32, p float32) {
	if p >= 1.0 {
		return
	}
	n := len(logits)
	probs := make([]float32, n)
	copy(probs, logits)
	Softmax(probs)

	type iv struct {
		idx  int
		prob float32
	}
	items := make([]iv, n)
	for i, v := range probs {
		items[i] = iv{i, v}
	}
	sort.Slice(items, func(a, b int) bool { return items[a].prob > items[b].prob })

	var cumSum float32
	cutoffIdx := n - 1
	for i, item := range items {
		cumSum += item.prob
		if cumSum >= p {
			cutoffIdx = i
			break
		}
	}

	allowed := make(map[int]bool, cutoffIdx+1)
	for i := 0; i <= cutoffIdx; i++ {
		allowed[items[i].idx] = true
	}
	negInf := float32(math.Inf(-1))
	for i := range logits {
		if !allowed[i] {
			logits[i] = negInf
		}
	}
}

func ApplyMinP(logits []float32, minP float32) {
	if minP <= 0 {
		return
	}
	probs := make([]float32, len(logits))
	copy(probs, logits)
	Softmax(probs)

	maxProb := probs[0]
	for _, p := range probs[1:] {
		if p > maxProb {
			maxProb = p
		}
	}

	threshold := minP * maxProb
	negInf := float32(math.Inf(-1))
	for i := range logits {
		if probs[i] < threshold {
			logits[i] = negInf
		}
	}
}
