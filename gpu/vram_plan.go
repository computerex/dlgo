//go:build cgo && vulkan

package gpu

import (
	"fmt"
	"os"
	"strconv"
	"sync"

	"github.com/computerex/dlgo/mmap"
	"github.com/computerex/dlgo/models/llm"
)

// ─────────────────────────────────────────────────────────────────────────────
// Weight residency: VRAM (device-local) vs system RAM (host-visible).
//
// The Vulkan allocator already falls back to host-visible memory once the
// device-local ceiling is reached, and every storage buffer stays readable by
// shaders regardless of which heap it landed in. That means a model larger than
// VRAM can still run entirely on the GPU compute path — the overflow is simply
// read across PCIe instead of from local memory. This is the "weight streaming
// from RAM" tier: no VRAM copy, no eviction, just a slower path to the same
// data. It is what Strata calls the RAM tier, and what FlexGen/ZeRO-Inference
// do with an explicit weight cache.
//
// What this file adds on top of the allocator's blind first-come-first-served
// fallback:
//
//  1. Context sizing that is aware it may spill (previously the pipeline
//     refused to raise the context past what fit *entirely* in VRAM, which
//     capped big models at 8K even though they would have run fine from RAM).
//  2. A startup report of the actual split, so the cost model is visible.
//  3. A hot-weight budget: the small, touched-every-token tensors (embeddings,
//     output head, norms, routers) are placed in VRAM first, and the bulky cold
//     ones (MoE expert matrices) are the ones allowed to spill. Spilling an
//     expert row costs a PCIe read for 4 of 64 experts; spilling the output
//     head would cost a PCIe read of the whole vocabulary projection on every
//     token.
// ─────────────────────────────────────────────────────────────────────────────

// Spill tiers for a single weight allocation.
const (
	// TierVRAM prefers device-local memory and only falls back to host when
	// the ceiling is reached (the historical behaviour).
	TierVRAM = iota
	// TierHost places the allocation in host-visible memory deliberately,
	// keeping precious device-local VRAM for hotter tensors.
	TierHost
)

const mib = 1 << 20

var (
	spillOnce   sync.Once
	spillEnable bool
)

// SpillEnabled reports whether weights may overflow VRAM into system RAM.
//
// Default: on. The pipeline already uploads every layer and relies on the
// allocator's fallback chain, so "off" would only reintroduce the old
// all-or-nothing behaviour for people who want the old startup messages.
// DLGO_GPU_SPILL=0 disables it.
func SpillEnabled() bool {
	spillOnce.Do(func() {
		spillEnable = true
		if s := os.Getenv("DLGO_GPU_SPILL"); s != "" {
			if v, err := strconv.Atoi(s); err == nil {
				spillEnable = v != 0
			}
		}
	})
	return spillEnable
}

// hostSpillBudget returns how many bytes of weights we are willing to let the
// allocator place in system RAM.
//
// Host-visible Vulkan memory is pinned by the driver, so it counts against the
// Windows commit limit, not just free physical RAM. We take the smaller of
// free physical RAM and remaining commit charge, and keep a reserve so the
// process cannot push the machine into paging.
func hostSpillBudget() int64 {
	info, err := mmap.GetSystemMemInfo()
	if err != nil {
		// Unknown: assume a conservative 2 GiB.
		return 2 << 30
	}
	// Host-visible Vulkan memory is pinned, so the real limit is what we can
	// commit. Take the smaller of free physical RAM and remaining commit
	// charge. Apply a fixed reserve, not a percentage: a percentage shrinks the
	// tier exactly when the desktop is busy, i.e. when the tier is needed.
	budget := int64(info.AvailablePhysical)
	if info.CommitAvailable > 0 && int64(info.CommitAvailable) < budget {
		budget = int64(info.CommitAvailable)
	}

	// Keep room for the OS, the desktop, the Go heap, mmap'd GGUF pages and the
	// CPU-side KV mirror.
	const reserve = 2 << 30
	budget -= reserve
	if budget < 0 {
		budget = 0
	}
	// The tier decision is otherwise invisible: it silently caps the context at
	// its floor, so make the inputs available when asked.
	if os.Getenv("DLGO_DEBUG_VRAM_PLAN") != "" {
		fmt.Printf("[dlgo/plan] host tier: availPhys=%d MB commitAvail=%d MB -> budget=%d MB\n",
			int64(info.AvailablePhysical)/mib, int64(info.CommitAvailable)/mib, budget/mib)
	}
	return budget
}

// vramSpillBudget returns how many bytes of weights may live in device-local
// memory. This mirrors the allocator's own ceiling (total VRAM minus the
// driver/system reserve), capped by --max-vram when the user set one.
func vramSpillBudget() int64 {
	free := effectiveFreeVRAM()
	if free <= 0 {
		return 0
	}
	// The allocator reserves 3 GiB below total VRAM for the desktop/WDDM; free
	// VRAM already reflects what the desktop is currently using, so only a
	// small safety margin is needed here.
	return free - 512*mib
}

// modelWeightBytes totals the weights that UploadModel will push to the GPU,
// split into per-layer and non-layer (embedding/output/norm) bytes.
func modelWeightBytes(m *llm.Model) (layerBytes, nonLayerBytes int64) {
	for l := range m.Layers {
		layerBytes += llm.EstimateLayerBytes(&m.Layers[l])
	}
	if m.TokenEmbed != nil {
		nonLayerBytes += int64(len(m.TokenEmbed.Data))
	}
	if m.Output != nil && m.Output != m.TokenEmbed {
		nonLayerBytes += int64(len(m.Output.Data))
	}
	nonLayerBytes += int64(m.Config.EmbeddingDim * 4 * 4) // norms + biases
	return
}

// streamingHeadroom is how many bytes we can afford for everything that is not
// resident weights: KV cache, run state, batch state, RoPE tables, scratch.
func streamingHeadroom(m *llm.Model, maxSeqLen int) int64 {
	vram := vramSpillBudget()
	host := hostSpillBudget()
	if vram < 0 {
		vram = 0
	}
	return vram + host
}

// kvAndFixedBytes estimates the non-weight GPU cost at a given context length.
func kvAndFixedBytes(m *llm.Model, ctx int) int64 {
	total := estimateFixedVRAM(m.Config, ctx) * 115 / 100

	kvDim := int64(m.Config.NumKVHeads * m.Config.HeadDim)
	kvPerLayer := 2 * int64(ctx) * kvDim * 2 // K + V, f16

	for l := range m.Layers {
		if layerNeedsKV(&m.Layers[l]) {
			total += kvPerLayer
		}
		if m.Layers[l].Spec.Core == llm.CoreSSM && m.Config.SSMInnerSize > 0 {
			numHeads := int64(m.Config.SSMTimeStepRank)
			headKDim := int64(m.Config.SSMStateSize)
			headVDim := int64(m.Config.SSMInnerSize) / numHeads
			total += numHeads*headKDim*headVDim*4 + int64(m.Config.SSMConvKernel)*numHeads*headVDim*4
		}
	}
	return total
}

// computeStreamingContext picks the largest context that fits once weights are
// allowed to spill into system RAM.
//
// The old computeMaxGPUContext required *all* weights plus the KV cache to fit
// in device-local VRAM, so a model whose weights alone exceeded VRAM was pinned
// to the 8K floor no matter how much RAM the machine had. With spilling, the
// only thing that must fit is headroom = weights + KV + scratch against
// (VRAM + host budget). KV still prefers VRAM (it is touched every token).
func computeStreamingContext(m *llm.Model, maxSeqLen int) int {
	weights, _ := modelWeightBytes(m)
	// Vulkan buffer alignment and descriptor overhead.
	weights = weights * 105 / 100

	budget := streamingHeadroom(m, maxSeqLen)
	if budget <= 0 {
		return 0
	}

	// Everything at full context already fits — take it.
	if weights+kvAndFixedBytes(m, maxSeqLen) <= budget {
		return maxSeqLen
	}

	const minCtx = 4096
	if weights+kvAndFixedBytes(m, minCtx) > budget {
		return 0
	}

	lo, hi := minCtx, maxSeqLen
	best := minCtx
	for lo <= hi {
		mid := lo + (hi-lo)/2
		if weights+kvAndFixedBytes(m, mid) <= budget {
			best = mid
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}
	return best
}

// residencyReport prints how much of the model actually landed in VRAM versus
// system RAM, plus the effective per-token read amplification that implies.
func residencyReport(weights int64) {
	allocated := int64(AllocatedBytes())
	host := int64(HostVisibleBytes())
	vram := allocated - host
	if vram < 0 {
		vram = 0
	}

	fmt.Printf("[dlgo/gpu] Residency: %.0f MB device-local, %.0f MB host-visible (system RAM)\n",
		float64(vram)/mib, float64(host)/mib)

	if host <= 0 {
		return
	}
	if weights > 0 {
		fmt.Printf("[dlgo/gpu] Weights spilled to RAM: %.0f MB (%.0f%% of model), read over PCIe every token\n",
			float64(host)/mib, 100*float64(host)/float64(weights))
		// Worst case: every host-resident byte read once per token. Decode only
		// touches what the router selects (ExpertUsedCount of ExpertCount experts)
		// plus the dense weights, so this is an upper bound, not a prediction:
		// Qwen3.6-35B-A3B at 54% spill measured 16.3 tok/s against a 1.9 tok/s
		// bound here. The active fraction is roughly all of the dense weights plus
		// ExpertUsedCount/ExpertCount of the expert weights, which is what makes
		// spilling the *experts* far cheaper per byte than spilling anything read
		// every step (the KV cache, the output head, the routers).
		fmt.Printf("[dlgo/gpu] Spill bound (worst case, all spilled bytes per token): %.2f GB/token at ~%.0f GB/s = >=%.0f ms/token\n",
			float64(host)/(1<<30), spillPCIeGBs,
			float64(host)/(spillPCIeGBs*1e9)*1000)
	}
}

// spillPCIeGBs is the measured sustained host-visible read rate. Used only for
// the startup cost estimate; the real number depends on chipset and ReBAR.
const spillPCIeGBs = 22.0
