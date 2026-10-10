# System One kernel optimization plan

Status: **proposal / not yet implemented**
Scope: GPU (Vulkan) execution path for System One decision requests
Companion documents: `docs/systemone-parity-plan.md` (accuracy + latency parity),
`docs/gpu-ssm-recurrence-optimization.md` (fused SSM recurrence, already landed)

---

## 0. Summary

System One is **prefill-dominated**: every decision is one forward pass over a prompt
with no autoregressive decoding. The measured 9B profile puts **~75% of categorized
time in FFN**, and the reason is now confirmed at the tensor level rather than inferred.

The single highest-value finding in this document:

> **The 9B's two largest FFN matrices (`ffn_gate`, `ffn_up`, 4096×12288 each) are
> stored as Q3_K, and `gpu_batch_matvec` has no batched GEMM path for Q3_K.** Those
> projections therefore fall back to `matvec_q3_k.comp`, which is indexed by token
> position (`gl_WorkGroupID.y = pos_id`) and **re-reads the weight matrix once per
> token**. For a 120-token request that is ~120× redundant weight traffic; for a
> 10.7K-token document it is ~10,700×.

| ID | Workstream | Priority | Risk | Expected impact |
|---|---|---|---|---|
| **K1** | Q3_K batched GEMM (**tiled f32 done, opt-in**; coopmat TBD) | **P0** | Low | Largest single win; attacks cold long-doc and warm short |
| **K2** | FFN fusion + shape specialization | P1 | Low–Med | Dispatch/barrier + intermediate-traffic reduction |
| **K3** | Branch-batched question suffixes | P1 | Med–High | Warm multi-question throughput; genuinely System One-specific |
| **K4** | Chunk-parallel delta-rule SSM | P2 | High | Becomes the bottleneck after K1 |
| **K5** | Shared-prefix attention kernels | P2 | Med | Grows with document length × question count |
| **K6** | Smaller items (persistent scratch, readout on GPU, npos-tuned configs) | P2 | Low | Incremental |

**Recommended first move:** K1, gated behind an opt-in env flag, validated against the
existing corpus and hidden-state parity tests. Do not start K3–K5 until K1's effect is
measured, because K1 changes where the bottleneck is.

**Implementation progress.** K1's **tiled f32** path is implemented and numerically
validated (see K1 below); it is opt-in via `DLGO_GEMM_Q3K=1` and the default path is
unchanged. The coopmat variant and the real-model corpus/_latency_ gates remain.

---

## 1. Scope and non-goals

### In scope
- Vulkan compute kernels and the C dispatch layer that serves System One requests.
- The `SystemOneMulti` / `SystemOneWithReadout` orchestration in `gpu/pipeline.go`.
- Instrumentation needed to attribute time to kernels, quantization formats and phases.
- Validation methodology sufficient to ship these changes without silent accuracy drift.

### Out of scope (explicitly)
- **Backbone fine-tuning.** Packed Q3_K/Q4_K/Q5_K weights are frozen and must not be
  modified or re-quantized. All work here is readout-only / runtime-only.
- **Approximation as an optimization.** Early exit, layer skipping, token pruning and
  speculative shortcuts are model changes, not kernel optimizations. They are not
  proposed here.
- **Readout-head optimization.** The 9B readout is 5×4096 floats (~80 KB). Moving it to
  the GPU saves a small download; it cannot fix second-scale latency.
- **CPU path.** Unchanged.
- **Accuracy workstreams** (data hygiene, split fixes, option-count capacity) — those
  live in `docs/systemone-parity-plan.md` and are orthogonal.

### Standing constraints (carried from the project)
1. GPU builds use `-tags vulkan -ldflags "-linkmode internal"` (external linker
   produces an invalid PE on this toolchain).
2. One server at a time; kill active servers and verify VRAM is clean before
   onboarding or benchmarking a model.
3. **The default path must stay bit-identical.** Any change that reorders
   floating-point reductions (GEMM vs matvec, coopmat vs scalar, chunked vs sequential
   recurrence) must be **opt-in via an env flag and validated on the corpus** before
   it can become a default.
4. Never revert the user's pre-existing uncommitted changes.

---

## 2. Why the System One workload is different

A generic LLM serving optimization assumes **decode dominates**: single-token steps,
KV-cache-bound, latency-sensitive. System One inverts this.

| Property | Generic chat serving | System One |
|---|---|---|
| Dominant phase | decode (1 token/step) | **prefill** (whole prompt at once) |
| Per request | many steps | **one forward pass** |
| Output | long token stream | **one hidden vector → linear readout** |
| Vocabulary projection | required, expensive | **skipped entirely** |
| Cross-request reuse | prefix cache (text match) | **exact-prefix state snapshot** |
| Multi-question | separate requests | **one shared prefix, N suffixes** |

Consequences that shape this plan:

1. **Single-token decode kernels are the wrong target.** Optimizing `npos == 1`
   matvec paths does little for System One. The `npos > 1` batched paths are what
   matter.
2. **The batched path is where the money is.** `gpu_batch_matvec` with `npos` up to 512
   is the hot function. Its per-format coverage is therefore the most important
   property in the codebase for this workload.
3. **Prefix prefill and suffix prefill have different shapes.** The prefix is long
   (up to `MaxSeqLen`, chunked at 512); suffixes are short (tens of tokens). They
   deserve different kernel configurations.
4. **Multi-question requests are embarrassingly parallel at the question level**, but
   the current implementation runs them sequentially (see K3).

---

## 3. Current GPU execution path (grounded)

### 3.1 The batched dispatch point

`gpu/csrc/vulkan_gpu.c:2630` — `gpu_batch_matvec(out, weights, x, rows, cols, npos, qtype)`

Current routing for `npos > 1`:

| qtype | Path | Notes |
|---|---|---|
| BF16 | `gpu_gemm_bf16` (coopmat) | only if `g.has_coopmat` |
| Q4_K | `gpu_gemm_q4k` (coopmat) → `gpu_gemm_q4k_tiled` fallback | `vulkan_gpu.c:3340`, `:2647` |
| Q5_K | `gpu_gemm_q5k` (coopmat) | `:3360` |
| Q6_K | `gpu_gemm_q6k` (coopmat) → `gpu_gemm_q6k_f32` | `:3380`, `:3400` |
| **Q3_K** | **falls through to `PIPE_MATVEC_Q3_K`** | **no GEMM path — this is the gap** |
| Q2_K, IQ*, MXFP4, Q4_0, Q5_0, Q8_0, F32, F16 | matvec pipelines | `rows_per_wg = 2` for most |

The matvec fallback is not merely "scalar instead of tensor cores" — it is
**algorithmically wrong for a batch**. `gpu/shaders/matvec_q3_k.comp:74`:

```glsl
uint pos_id = gl_WorkGroupID.y;      // one workgroup per token position
uint pos_x_off = pos_id * ncols;
```

Each workgroup owns **2 weight rows for exactly one token**. The weight block is read
from global memory per token, with no reuse across tokens. Workgroup count for one
`ffn_gate` projection at `npos = 120` is `(12288/2) × 120 = 737,280`; each re-reads
its slice of a 21.6 MB matrix.

A tiled GEMM instead loads a weight tile into shared memory **once** and reuses it
across every token in the N-tile.

### 3.2 Confirmed per-tensor quantization map (Qwen3.5-9B-Q3_K_M)

Produced with `gguf_info.exe <model> -names` (427 tensors total). This is the evidence
that makes K1 the top priority.

**Type histogram:** F32 177 · Q8_0 48 · **Q3_K 105** · Q4_K 92 · Q5_K 4 · Q6_K 1

**FFN (present in all 32 layers):**

| Tensor | dims | type | batched path today |
|---|---|---|---|
| `blk.N.ffn_gate.weight` | 4096 × 12288 | **Q3_K** | **matvec fallback** |
| `blk.N.ffn_up.weight` | 4096 × 12288 | **Q3_K** | **matvec fallback** |
| `blk.N.ffn_down.weight` | 12288 × 4096 | Q4_K (blk.2+), Q5_K (blk.0–1) | coopmat GEMM ✓ |

**SSM layers** (24 of 32; e.g. blk.0,1,2,4,5,6,8):

| Tensor | dims | type | batched path today |
|---|---|---|---|
| `blk.N.attn_qkv.weight` | 4096 × 8192 | Q4_K | coopmat GEMM ✓ |
| `blk.N.attn_gate.weight` | 4096 × 4096 | **Q3_K** | **matvec fallback** |
| `blk.N.ssm_out.weight` | 4096 × 4096 | Q4_K | coopmat GEMM ✓ |
| `blk.N.ssm_alpha.weight` | 4096 × 32 | Q8_0 | matvec (tiny, negligible) |
| `blk.N.ssm_beta.weight` | 4096 × 32 | Q8_0 | matvec (tiny, negligible) |

**Full-attention layers** (8 of 32; e.g. blk.3, blk.7):

| Tensor | dims | type | batched path today |
|---|---|---|---|
| `blk.N.attn_q.weight` | 4096 × 8192 | **Q3_K** | **matvec fallback** |
| `blk.N.attn_k.weight` | 4096 × 1024 | **Q3_K** | **matvec fallback** |
| `blk.N.attn_v.weight` | 4096 × 1024 | Q5_K | coopmat GEMM ✓ |
| `blk.N.attn_output.weight` | 4096 × 4096 | Q4_K | coopmat GEMM ✓ |

`token_embd.weight` is Q3_K but is consumed as a row lookup
(`TokenEmbed.DequantizeRow`), not a matvec, so its type is irrelevant to dispatch.

**Conclusion:** the two largest matrices in the model — the FFN gate and up
projections, which together are **two-thirds of all FFN weight bytes** — are exactly
the format with no batched GEMM. This is not a coincidence of the profile; it is the
profile's cause.

### 3.3 Weight traffic arithmetic (analytical, not measured)

Q3_K: 256-element super-block → 110 bytes (hmask 32 + qs 64 + scales 12 + d 2)
= **3.4375 bits/element**.
Q4_K: 144 bytes/256 = **4.5 bits/element**. Q5_K: 176 bytes/256 = **5.5 bits/element**.

FFN weight bytes per layer:
- `ffn_gate` Q3_K: 50,331,648 × 0.4297 B = **21.6 MB**
- `ffn_up` Q3_K: **21.6 MB**
- `ffn_down` Q4_K: 50,331,648 × 0.5625 B = **28.3 MB**
- **Total ≈ 71.6 MB/layer → 2.29 GB across 32 layers**

FFN FLOPs per token: 2 × 3 × 50,331,648 × 32 = **9.66 GFLOP/token**.

| Quantity | Value |
|---|---|
| Weight bytes, one full pass | 2.29 GB |
| Bandwidth floor (read weights once), RTX 4070 Ti SUPER ≈ 672 GB/s | **≈ 3.4 ms** |
| Compute floor at a hypothetical 40 TFLOPS effective | ≈ 29 ms (for 120 tokens) |
| **Measured FFN time, ~120-token request** | **958.7 ms** |
| Implied achieved rate | ≈ 1.2 TFLOPS |

The measured time is consistent with **per-token weight re-read** (120 × 2.29 GB
≈ 275 GB → ≈ 410 ms at 672 GB/s, plus activation traffic and dequant overhead). The
exact multiple is not the point; the point is that the current path is **one to two
orders of magnitude away from its own memory roofline**, and the fix is a tiling
change, not a micro-optimization.

> These figures are analytical estimates from the tensor map and published device
> bandwidth. They are **not** measurements and must be replaced by the per-tensor
> profiling described in §6 before any speedup is claimed.

### 3.4 SSM batched path (already reasonably optimized)

`gpu/ssm_batch.go:10` gates it:

```go
ssmBatchEnabled && !ssmReference && !cpuSSMDelta && !cpuSSMFull &&
heads > 0 && kd > 0 && kd <= 128 && vd > 0 && vd <= 128 &&
keyDim > 0 && keyDim%kd == 0 && keyDim/kd <= heads && npos > 1 && npos <= 512
```

`ssmBatchPrefill` (`gpu/ssm_batch.go:14`) runs four stages —
conv1d → preprocess → recurrence → norm/gate — each a single dispatch:
`ssm_conv1d_batch.comp`, `ssm_preprocess_batch.comp`, `ssm_recurrence_batch.comp`,
`ssm_norm_gate_batch.comp`.

`ssm_recurrence_batch.comp` is **multi-token**: one workgroup per (head, value tile of
32), each lane owns a value column and keeps its K-dim column of recurrent state in
registers for the whole token loop. State traffic is one load/store per chunk instead
of per token. Its header states the arithmetic order per lane is identical to the
single-workgroup version, so **results are bit-identical**.

The 9B sits exactly on the limit: `vd = 4096/32 = 128` (`vd <= 128` ✓), `kd = 128`
(`kd <= 128` ✓). Any change to these limits must be validated on the 9B specifically.

This path is already the "fuse the dispatches" step. The next step (K4) is a
different algorithm, not more fusion.

### 3.5 Attention

Full-attention layers use `attention_tiled`, `attention_tiled_f32`,
`attention_full_f16/f32`, `flash_attn_f32`, `paged_attention` (see
`gpu/shaders/compile.ps1:74–85`). Measured share on the 9B short request: **86.7 ms
(≈7%)**. Not the first target, but see K5.

### 3.6 `SystemOneMulti` orchestration and the prefix cache

`gpu/pipeline.go:1643` — `SystemOneMulti`. Structure:

1. Compile every question up front (`CompileSystemOneParts`), require a shared
   `system + state` prefix.
2. Token-boundary safety: `checkSeamBoundary` (`:1898`) re-encodes a 16-token window
   at the seam; on any mismatch the question falls back to a standalone prefill. This
   preserves correctness and must not be weakened by any optimization.
3. Fast-path gate (`:1655`):
   ```go
   if p.HasMoE || p.IsPartialGPU || p.hasCPUAttn || p.CPURunState != nil ||
      !isHybridSSMModel(p.CPUModel) {
       return p.systemOneMultiPerQuestion(...)
   }
   ```
   The fast shared-prefix path requires **dense + fully GPU-resident + hybrid SSM**.
4. Prefix prefill, chunked at 512 tokens (`:1762`), then `SnapshotState` →
   `storePrefixCache` (4-entry LRU, `prefixCacheMaxEntries = 4`, `:82`).
5. Per question, in a loop (`:1789`): restore state, run the suffix prefill, extract
   the final hidden vector, apply the readout.
   - `prefixResident` (`:1836`) uses `RestoreRecurrentState` instead of the full
     `RestoreState` when the prefix KV is still live — a real, already-landed win.
   - Suffix passes are **strictly sequential**.

**The structural opportunity:** steps 5's per-question work is independent given the
snapshot. Nothing in the math requires sequential execution. That is K3.

### 3.7 What is already optimized (do not redo)

- Shared document prefill across questions.
- Exact-prefix state snapshot cache with LRU eviction.
- Recurrent-only branch restore when prefix KV is resident.
- Final-hidden extraction without the vocabulary projection.
- Batched multi-token SSM recurrence (bit-identical, measured 2.3× isolated /
  1.5× E2E in `docs/gpu-ssm-recurrence-optimization.md`).
- Coopmat GEMM for Q4_K, Q5_K, Q6_K, BF16.

---

## 4. Measured baseline (for regression comparison)

From `bench/systemone/RESULTS-qwen35-9b-2026-10-07.md` and
`bench/systemone/RESULTS-qwen38-2b-2026-10-07.md`:

| Workload | 0.8B | 2B | 9B |
|---|---:|---:|---:|
| 3q short cold (p50) | 194 ms | 244 ms | **2,558 ms** |
| 3q short warm (p50) | 109 ms | 148 ms | **1,546 ms** |
| 3q 10.7K-doc cold (p50) | 5.35 s | 6.87 s | **87.85 s** |
| 3q 10.7K-doc warm (p50) | 248 ms | 292 ms | **1.84 s** |

9B phase profile, `DLGO_SSM_TIME=1`, ~120-token request, batched path confirmed
engaged:

| Phase | ms | share |
|---|---:|---:|
| FFN | 958.7 | 75% |
| SSM | 224.9 | 18% |
| attention | 86.7 | 7% |
| embed | 4.5 | 0.4% |

Caveats that must be repeated in any future report:
- Historical cross-model comparisons, not matched-nonce.
- `DLGO_SSM_TIME` brackets each category with **submit + wait**, so it includes host
  recording and synchronization overhead, not pure GPU execution. It is a coarse
  attribution tool, not a GPU timestamp.
- The profile identifies FFN as the target. It does **not** by itself prove Q3_K is the
  cause; that proof is the tensor map in §3.2 plus the profiling in §6.

Accuracy baselines that must not regress: 9B test **93.07%** (693 rows), NLL 0.2208,
Brier 0.1111, ECE 0.0372; Yu-Gi-Oh **9/10**.

---

## 5. Workstreams

### K1 — Q3_K batched GEMM (P0) — **tiled f32 path implemented (opt-in)**

**Status.** The **tiled f32 GEMM** (`gemm_q3k_tiled.comp`, `gpu_gemm_q3k_tiled`) is
implemented, gated behind `DLGO_GEMM_Q3K=1` (default off). It is numerically validated
against the CPU reference (`quant.DequantizeQ3_K`) at **relErr 4.7e-07**
(`gpu/gemm_q3k_test.go`, `TestGemmA3KTiledGPU`). The coopmat variant is **not** yet
implemented — see "Coopmat follow-up" below.

**Problem.** Q3_K has no `npos > 1` GEMM. `ffn_gate`, `ffn_up` (all 32 layers),
`attn_gate` (24 layers), `attn_q`, `attn_k` (8 layers) all take the per-token
matvec fallback.

**Goal.** A tiled GEMM that loads a Q3_K weight tile into shared memory once and
reuses it across the token tile.

**Design notes (as implemented).**

Q3_K super-block layout (256 elements, 110 bytes), from `quant/q3_k.go` (authoritative)
and `matvec_q3_k.comp:14`:

```glsl
struct block_q3_k {
    uint8_t hmask[32];   // 256 high bits, one per element
    uint8_t qs[64];      // 2-bit low parts, 4 per byte
    uint8_t scales[12];  // 16 × 6-bit scales
    float16_t d;         // super-block scale
};
```

Dequantization (ported verbatim from `quant/q3_k.go`):

```
is = local_k / 16                              // scale group (16 elements)
dl = d * (scale[is] - 32)                       // 6-bit scale, biased by 32
qs_byte = (local_k/128)*32 + (local_k%32)       // 2-bit field = (local_k/32)%4
v  = (qs[qs_byte] >> (2*((local_k/32)%4))) & 3
v -= 4 if (hmask[local_k%32] bit (local_k/32)) is clear    // → v ∈ {-4..3}
w  = dl * v
```

The 16 six-bit scales are unpacked from the 12-byte `scales` array exactly as
`quant/q3_k.go` does (four u32 words via the `kmask1`/`kmask2` recombination).

**Key structural difference from Q4_K:** Q3_K has **no `dmin`** and its 16 scales each
cover **16 elements**, not 32.

> **Correction to an earlier draft of this plan.** The draft claimed "coopmat `TK = 16`
> aligns exactly with one Q3_K scale group". That is **wrong**: a coopmat A tile of
> `TK` 16-bit columns covers `TK/2 = 8` elements, so one 16-element Q3_K group needs
> `TK = 32`. The implemented tiled f32 kernel sidesteps this entirely by dequantizing
> the whole `BK = 32` slice per row into shared f32 (no tensor cores), so scale-group
> alignment is not a constraint.

**Implemented structure** (`gemm_q3k_tiled.comp`):
- `TM = 64` tokens × `TN = 64` rows, 256 threads, `TK = 32` K-tile.
- Dequantize the Q3_K tile to **f32** in shared memory (`smB`), reuse across the token
  tile; input tile `smA` is f32.
- Push constants `(M, N, K, out_stride) = (npos, rows, cols, rows)`, matching
  `gemm_f16.comp`.
- Workgroup grid: `((npos+63)/64) × ((rows+63)/64)`; requires `cols % 256 == 0`
  (whole super-blocks), else falls through to the matvec.
- Kept f32 (not f16) staging to avoid a second lossy rounding on Q3_K tensors with
  large outliers (`attn_q`/`attn_k`) — see "Coopmat follow-up".

**Coopmat follow-up (not yet done).** A tensor-core variant still needs to be written
and benchmarked against this tiled f32 path; the plan's open question #2 (is coopmat
even faster here, given the dequant cost?) applies. If added, mirror
`gemm_q4k_coopmat.comp` with `BK = 32`, dequantizing the whole K-tile to f16 staging
and using `TK = 16, N_K_TILES = 2`, and gate it behind the same `DLGO_GEMM_Q3K`.

**Alternative considered: Q3_K dp4a GEMM.** `matvec_q3_k_dp4a.comp` exists but is a
matvec (wrong shape), and dp4a requires Q8_1-quantized activations, which changes
numerics. Viable second choice if coopmat is blocked; must be opt-in for the same
reason.

**Touch points (as implemented).**
- `gpu/shaders/gemm_q3k_tiled.comp` (new)
- `gpu/shaders/compile.ps1` — registered; generates `spv_gemm_q3k_tiled`
- `gpu/csrc/shaders_spirv.h` — embedded SPIR-V (regenerated)
- `gpu/csrc/vulkan_gpu.h` — `PIPE_GEMM_Q3K_TILED` (appended at the end of `PipelineID`
  so existing IDs stay stable) + prototype + `gpu_set_gemm_q3k`
- `gpu/csrc/vulkan_gpu.c` — `gpu_gemm_q3k_tiled()`, the gated dispatch branch in
  `gpu_batch_matvec` (before the matvec switch), and the env flag helper
- `gpu/gpu_cgo.go` / `gpu/gpu_stub.go` — `SetGemmQ3K` test hook
- `gpu/gemm_q3k_test.go` (new) — numerical parity test
- `gpu/forward.go:1795` `supportsGPUQType` — Q3_K already accepted; no change

**Gating.** Env-flag helper in `vulkan_gpu.c` (mirrors `fused_q4k_available`,
`vulkan_gpu.c:4112`):

```c
static int g_gemm_q3k_enabled = -1;
static int gemm_q3k_available(void) {
    if (g_gemm_q3k_enabled < 0) {
        const char* env = getenv("DLGO_GEMM_Q3K");
        g_gemm_q3k_enabled = (env && env[0] == '1') ? 1 : 0;
    }
    return g_gemm_q3k_enabled;
}
```

Default **off**; with the flag unset the matvec fallback is byte-for-byte the default
behavior. Flip to default-on only after §7 gates pass.

**Acceptance.**
- [x] Numerical parity vs the CPU reference within tolerance (relErr 4.7e-07 on
  synthetic Q3_K blocks; reduction order differs from the matvec, so bit-identity to
  the *matvec* is not expected).
- [x] **Isolated kernel speedup ≥5×** — measured `TestGemmA3KBatchTimingGPU` on the real
  FFN gate/up shape (4096×12288): **5.25× @ npos=64, 5.72× @128, 5.95× @256, 6.04× @512**
  (npos=512: 43.14 ms matvec → 7.15 ms GEMM).
- [x] Decision-level parity on a real model — 120-row subset, flag off vs on: **0 decision
  changes**; max confidence Δ 0.001, mean abs Δ 0.00011 (C6).
- [x] **Full-corpus accuracy unchanged** — 693-row splits-gp2 test, flag off vs on:
  **93.07% both**, NLL 0.2208 both, Brier 0.1111 both, ECE 0.0372 both; zero-shot 92.65%
  both, seen 94.32% both. §7.3 budget fully met.
- [x] **Cold long-doc p50 materially reduced; warm short p50 improved** — matched-nonce
  HTTP latency (9B, 3q): short cold 2,558 → **920 ms (2.72×)**, short warm 1,546 →
  **577 ms (2.68×)**, 10.8K-doc cold 87,901 → **25,325 ms (3.47×)**, 10.8K-doc warm
  1,826 → **841 ms (2.17×)**. Answers identical; only confidence drifts ~0.0008.
  Full numbers: `bench/systemone/RESULTS-k1-q3k-gemm-2026-10-08.md`.
- [x] Default (flag-off) path verified unchanged; pre-existing GPU test failures are
  unrelated (confirmed against a pristine HEAD worktree).

---

### K2 — FFN fusion and shape specialization (P1)

Once K1 lands, the FFN is a set of well-shaped GEMMs and the remaining costs are
dispatch count, barriers and intermediate traffic.

**Targets, in order of expected value:**

1. **Fused gate/up epilogue.** `ffn_gate` and `ffn_up` share the same input
   (`ffn_norm`) and are consumed together as `SiLU(gate) * up`. A dual-weight GEMM
   variant that binds both weight buffers and writes the *activated product* directly
   avoids materializing two 12288-wide intermediates. For `npos = 512` that is
   ~50 MB of write+read traffic removed per layer.
2. **Down-projection residual epilogue.** Fold the residual add into the
   `ffn_down` GEMM epilogue instead of a separate `add` dispatch.
3. **Shape-specialized configs.** The prefix (up to 512 tokens) and the suffix (tens of
   tokens) want different `BN`/`BK` and workgroup counts. A single configuration for
   all `npos` leaves performance on the table at both ends. Add a small
   `npos`-dependent selection in the C dispatch layer.
4. **Persistent scratch.** `GpuForwardPrefillBatchHybrid` allocates and frees the SSM
   `processed` buffer per invocation (`gpu/forward.go:1007–1010`). Move it to
   `GpuBatchState` so it is allocated once.

**Sequencing note:** do this **after** K1. Fusing epilogues around a kernel that
re-reads weights 120× is polishing the wrong layer.

---

### K3 — Branch-batched question suffixes (P1, System One-specific)

**Problem.** `SystemOneMulti` runs question suffixes strictly sequentially
(`gpu/pipeline.go:1789`). For a 3-question warm request, the suffix work is done three
times, one after another, even though the three branches are independent given the
prefix snapshot.

**Goal.** Execute the N independent suffix branches in a branch-aware batched pass:

```
prefix prefill once → snapshot
        ↓
batched pass over all N suffixes, layer by layer
  - dense projections / FFN: tokens from all branches in one GEMM
  - attention: shared read-only prefix KV; per-branch suffix KV + causal mask
  - SSM: per-branch recurrent + conv state
        ↓
N final hidden vectors → readout per question
```

**Why this is the genuinely System One-specific opportunity.** Generic serving has no
notion of "one shared prefix, N independent short continuations that only need their
final hidden state". This structure exists only here.

**Hard requirements — this is where the risk lives.**
- **Branch independence must be exact.** Separate SSM state, separate conv state,
  separate suffix KV per branch. Sharing any mutable buffer across branches is a
  correctness bug, not a performance trade.
- **Attention masking.** Each branch attends to the *whole* shared prefix plus *its own*
  suffix, and must not attend to another branch's suffix tokens.
- **Positions.** Each branch's suffix tokens must carry the correct absolute positions
  (prefix length + local offset), not a packed batch offset.
- **Variable lengths.** Branches have different suffix lengths. Padding must not
  contribute to any reduction (mask it, or use per-branch lengths).
- **SSM batch limit.** `ssmBatchSupported` caps `npos <= 512`; a branch-batched pass
  concatenates suffixes, so the combined token count must be checked against this and
  the `vd <= 128` / `kd <= 128` limits.
- **Do not drive the existing shared mutable buffers concurrently.** The current
  pipeline has one `BatchState` and one `RunState`. A branch-batched pass needs an
  explicit multi-branch layout, not concurrent use of the single-branch buffers.

**Honest expectation.** This does not make three questions three times faster. It makes
the *same* work more efficient by batching it. The win is real but bounded by how much
of the suffix work is memory-bound (large, after K1) versus latency-bound.

**Prerequisite:** K1. Without it, the batched suffix pass would still hit the Q3_K
matvec fallback and the win would be small.

**Benchmark requirement:** measure as a function of question count (1, 4, 13, 32) and
report p50/p95 plus peak VRAM, per `docs/systemone-parity-plan.md` §Priority 1.3.

---

### K4 — Chunk-parallel delta-rule SSM (P2)

**Why P2 and not P1:** the SSM is 18% of the measured profile *today*. After K1 it
becomes a much larger share, and this is the natural next step — but starting it now
would optimize a minority cost.

**Problem.** `ssm_recurrence_batch.comp` already batches across tokens in the sense
that one workgroup processes the whole chunk, but the token loop is still sequential
inside the workgroup (token N depends on token N−1). Parallelism is across
(head, value tile), not across time.

**Goal.** A chunked gated-delta-rule formulation: split the sequence into chunks,
compute chunk-local delta-rule contributions, and propagate the incoming recurrent
state across chunk boundaries. This exposes parallelism across chunks.

**Why this is high risk.**
- The current shader is **bit-identical** to the reference by construction (same
  decay/lr order, compensated Kahan prediction and output sums, private state column
  per lane — see the shader header and `docs/gpu-ssm-recurrence-optimization.md`).
  A chunked reformulation **cannot** preserve bit-identity; it reorders reductions.
- The recurrence is matrix-valued. A naive elementwise prefix sum is **not
  equivalent** (this is called out explicitly in `docs/systemone-parity-plan.md`).
- Chunk boundaries, partial chunks, prefix restore and long contexts all need
  independent validation.

**Required validation (from the parity plan, restated):** compare hidden states,
recurrent state, logits and decisions against the sequential reference across chunk
boundaries and partial chunks. Report **tolerance and accuracy drift**, not cosine
similarity alone. Keep the sequential path as a reference fallback (`DLGO_SSM_REFERENCE`
already exists as the pattern).

**Lower-risk intermediate (do this first):** fuse decay/predict/update/output per
(head, value tile) while preserving the required synchronization and reduction
semantics. Measure whether dispatch savings justify the kernel before committing to
the chunked version.

---

### K5 — Shared-prefix attention kernels (P2)

**Structure to exploit:** N short suffixes each attend to one long, immutable prefix.

**Targets:**
- Avoid duplicating prefix KV reads across branches (pairs with K3).
- Split-KV reduction over the prefix length for long documents, so the reduction is
  parallelized rather than serialized in one workgroup.
- Layout tuned for the "many short queries × one long KV" shape, which is different
  from both standard prefill and standard decode.

**Value scales with** document length × question count. For short-document,
single-question requests the gain is small; do not prioritize it over K1/K3.

---

### K6 — Smaller items (P2)

- **Readout on GPU.** The 9B readout is 5×4096 floats. Applying it on-device removes
  a hidden-vector download and one host round trip. Small but cheap and it completes
  the "everything stays on the GPU" story. Must not change the readout math
  (`llm.ApplySystemOneReadout` semantics: L2 normalize, linear map,
  temperature-scaled masked softmax).
- **`npos`-tuned dispatch configs.** See K2 item 3.
- **Prefix cache sizing.** `prefixCacheMaxEntries = 4` is a fixed constant
  (`gpu/pipeline.go:82`). A VRAM-budgeted cache would help multi-document workloads.
  This is a serving/memory change, not a kernel change, but it belongs in the same
  latency effort.
- **Q8_0 GEMM for `ssm_alpha`/`ssm_beta`.** 4096×32 each — negligible; listed only for
  completeness of the dispatch-coverage audit.

---

## 6. Instrumentation and profiling plan (prerequisite for all of the above)

The existing `DLGO_SSM_TIME` instrumentation (`gpu/forward.go:30–86`) is insufficient
to attribute cost to kernels and formats. It:
- buckets into 4 coarse categories (embed / SSM / attn / FFN),
- **brackets each category with submit + wait**, so it measures host + sync + GPU
  together,
- covers only the batched path,
- counts only the first N chunks (`DLGO_SSM_TIME_CHUNKS`, default 3).

**Required additions before claiming any speedup:**

1. **Per-tensor / per-qtype attribution.** For each `BatchMatVec` call site, record
   `(layer, tensor_name, qtype, rows, cols, npos, elapsed)`. This is what turns
   "FFN is 75%" into "Q3_K gate+up is X% and Q4_K down is Y%".
2. **GPU timestamps, not wall-clock brackets.** Use Vulkan timestamp queries around
   dispatches so kernel execution is separated from host recording and
   `submit_and_wait` overhead. Without this, dispatch-bound and compute-bound
   optimizations are indistinguishable.
3. **Dispatch and barrier counters** per request, per phase. The Q3_K matvec fallback
   should show a workgroup count proportional to `npos × rows / 2`; a GEMM should show
   `(rows/64) × (npos/64)`. This is a direct, falsifiable check that K1 engaged.
4. **A same-backbone Q3-vs-Q4 comparison.** Re-quantize the same model to Q4_K_M (or
   use an existing Q4_K_M sibling), run the identical workload, and compare. This
   isolates quantization format from model size — the caveat already recorded in
   `bench/systemone/RESULTS-qwen35-9b-2026-10-07.md`.
5. **Microbenchmark harness.** A standalone benchmark that runs a single
   `gpu_batch_matvec` for a given `(qtype, rows, cols, npos)` and reports achieved
   GB/s and TFLOPS against the roofline. This is how K1's improvement gets measured
   without confounds from the rest of the model.

**Explicitly avoid:** reporting a single end-to-end number as evidence that a specific
kernel improved. End-to-end is the acceptance test; the kernel benchmark is the
diagnostic.

---

## 7. Correctness and validation gates

Every workstream must clear these before its flag can default to on.

### 7.1 Numerical
- **Bit-identity is required** for changes that do not reorder reductions (K2 item 4,
  K6 readout-on-GPU if implemented with the same order, K3's per-branch state
  handling).
- **Bit-identity is not expected** for K1 (GEMM vs matvec) and K4 (chunked vs
  sequential). For these, report:
  - max absolute and relative error on the final hidden vector,
  - cosine similarity,
  - recurrent-state divergence over the sequence,
  - and **accuracy drift on the corpus**, which is the number that actually matters.
- Reject non-finite results explicitly (the existing tests already do this — keep the
  pattern).
- Keep the reference path available: `DLGO_SSM_REFERENCE` for SSM; a
  `DLGO_NO_GEMM_Q3K`-style escape hatch for K1.

### 7.2 Behavioral
- Hidden-state, answer-code, confidence and calibration checks — not just a faster
  stopwatch.
- Cold and warm paths evaluated **independently**; a warm-cache win must not mask a
  cold-path regression.
- The `checkSeamBoundary` standalone fallback (`gpu/pipeline.go:1898`) must still
  trigger on the same inputs. Any K3 work must not bypass it.
- Degraded (too-wide) questions must still route to CPU continuation scoring
  (`gpu/pipeline.go:1802`).

### 7.3 Accuracy regression budget
Baseline to protect (9B, `splits-gp2` test, 693 rows):

| Metric | Baseline | Gate |
|---|---:|---|
| Overall accuracy | 93.07% | must not drop below 93.07% − 1 row (≈92.9%) |
| NLL | 0.2208 | not worse by >2% relative |
| Brier | 0.1111 | not worse by >2% relative |
| ECE | 0.0372 | not worse |
| Yu-Gi-Oh battery | 9/10 | not worse |

### 7.4 Test hygiene
- Development data for tuning; a separate final evaluation set for the verdict.
  (Note the existing caveat: the zero-shot group is **not** an untouched holdout —
  see `docs/systemone-parity-plan.md`.)
- Do not use test errors for kernel/config selection.
- Existing suites must stay green:
  `go test ./models/llm ./server ./cmd/systemone-train ./cmd/systemone-eval ./format/gguf`
  and the GPU numerical tests (`go test -tags vulkan ./gpu -run '^TestSSMRecurrence'`).

---

## 8. Phased roadmap

| Phase | Work | Depends on | Exit criterion |
|---|---|---|---|
| **0** | Per-tensor profiling + GPU timestamps + matvec microbenchmark | — | Q3_K share of FFN time **measured**, not inferred |
| **1** | K1: Q3_K tiled f32 GEMM (**done, opt-in**) + coopmat variant | 0 | Microbench shows ≥5× on the isolated kernel; corpus unchanged; long-doc cold p50 materially down |
| **2** | K2: FFN fusion (gate/up epilogue, down residual), persistent scratch | 1 | Dispatch count and intermediate traffic down; no accuracy change |
| **3** | K3: branch-batched suffixes | 1, 2 | Warm 3q p50 down; VRAM bounded; per-branch independence proven by test |
| **4** | Re-profile | 1–3 | New bottleneck identified with evidence |
| **5** | K4 and/or K5, chosen by the phase-4 profile | 4 | Tolerance + accuracy-drift gates passed |
| **6** | K6 cleanup; flip K1 default-on if all gates green | 1–5 | Default path documented and reproducible |

**Rule:** each phase ends with a written result (numbers, caveats, artifact hashes) in
`bench/systemone/`, matching the existing `RESULTS-*.md` convention.

---

## 9. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| GEMM reorders reductions → accuracy drift | High | Opt-in flag; corpus gate; report tolerance and drift, not just cosine |
| Q3_K 16-element scale groups mis-aligned with coopmat `TK` | Med | The tiled f32 path dequantizes the whole `BK=32` slice, so no alignment constraint; assert `cols % 256 == 0`. A coopmat variant needs `TK=32` (a `TK=16` A-tile is only 8 elements) |
| Coopmat unavailable on some device | Med | Tiled f32 fallback mirroring `gpu_gemm_q4k_tiled` |
| Branch batching shares mutable state → wrong answers | High | Per-branch state buffers; explicit independence test; never drive the single-branch `BatchState` concurrently |
| Chunked SSM breaks bit-identity | High | Keep sequential reference; validate across chunk boundaries and partial chunks; tolerance + drift gate |
| Profiling overhead distorts results | Med | Timestamp queries, not wall-clock brackets; warmup excluded; measure with instrumentation off for headline numbers |
| Optimizing a minority cost (SSM today, FFN tomorrow) | Med | Phase 0 profiling first; re-profile after each phase |
| VRAM growth from extra buffers (branch batching, persistent scratch) | Med | Budget against the 16 GB card; measure peak VRAM per phase; one server at a time |
| Building with the wrong linker → invalid PE | Med | Always `-tags vulkan -ldflags "-linkmode internal"` |

---

## 10. Open questions

1. **Is Q3_K the right format for this model at all?** If a Q4_K_M sibling exists,
   the same-backbone comparison (Phase 0 item 4) may show that re-quantizing is
   cheaper than writing a new kernel. This should be answered **before** Phase 1
   commits to the shader work.
2. **Is coopmat Q3_K actually faster than a well-tiled f32 Q3_K GEMM** on this GPU,
   given the dequant cost? The `gemm_q4k_dequant_coopmat` comment
   (`vulkan_gpu.c:3200`) notes it exists to avoid "scattered byte reads" — the same
   concern applies to Q3_K's `hmask`/`qs` bit-unpacking. Benchmark both.
3. **How much of the 958.7 ms is dequant ALU vs memory?** Determines whether K1's
   ceiling is bandwidth or compute.
4. **Does the prefix cache hit rate justify K3's complexity?** If real workloads
   mostly miss the cache, cold prefill (K1) dominates and K3's value drops.
5. **What is the actual production question count distribution?** K3's payoff scales
   with it; 1-question requests get nothing from it.
6. **Should the prefix chunk size stay at 512?** It was chosen for VRAM and fence-timeout
   reasons (`gpu/forward.go:1046–1055` comments). Larger chunks might feed a GEMM
   better; the fence timeout is the hard constraint.

---

## 11. Appendix — file map

**Dispatch / C layer**
- `gpu/csrc/vulkan_gpu.c:2630` — `gpu_batch_matvec` (the routing decision)
- `gpu/csrc/vulkan_gpu.c:3133` — `gpu_gemm_q4k_tiled` (fallback pattern)
- `gpu/csrc/vulkan_gpu.c:3300` — `gpu_gemm_q4k_coopmat`
- `gpu/csrc/vulkan_gpu.c:3340` — `gpu_gemm_q4k` (dispatched)
- `gpu/csrc/vulkan_gpu.c:3360` — `gpu_gemm_q5k`
- `gpu/csrc/vulkan_gpu.c:3380` — `gpu_gemm_q6k`
- `gpu/csrc/shaders_spirv.h` — embedded SPIR-V arrays

**Shaders**
- `gpu/shaders/matvec_q3_k.comp` — the fallback being replaced (per-token indexing)
- `gpu/shaders/gemm_q3k_tiled.comp` — **implemented** fused Q3_K→f32 tiled GEMM (K1)
- `gpu/shaders/gemm_q4k_coopmat.comp` — the template for a future coopmat Q3_K shader
- `gpu/gemm_q3k_test.go` — K1 numerical parity test (CPU reference vs GPU)
- `gpu/shaders/ssm_recurrence_batch.comp` — batched SSM recurrence (K4 target)
- `gpu/shaders/ssm_{conv1d,preprocess,norm_gate}_batch.comp` — the other SSM stages
- `gpu/shaders/compile.ps1` — shader registry (source of truth)
- `gpu/shaders/compile_ssm.ps1`, `compile_ssm_batch.ps1` — selective rebuilds

**Go orchestration**
- `gpu/pipeline.go:1643` — `SystemOneMulti` (K3 target)
- `gpu/pipeline.go:82` — `prefixCacheMaxEntries = 4`
- `gpu/pipeline.go:1655` — fast-path eligibility gate
- `gpu/pipeline.go:1757` — prefix cache lookup / fresh prefill
- `gpu/pipeline.go:1836` — `RestoreRecurrentState` vs `RestoreState`
- `gpu/pipeline.go:1898` — `checkSeamBoundary` (correctness-critical)
- `gpu/pipeline.go:1498` — `NewGpuBatchState` call site (K2 persistent scratch)
- `gpu/model.go:298` — `NewGpuBatchState` definition (batch scratch layout)
- `gpu/systemone_state.go:28` — `SnapshotState`
- `gpu/systemone_state.go:141` — `RestoreRecurrentState`
- `gpu/forward.go:975` — `GpuForwardPrefillBatchHybrid` (the hot loop)
- `gpu/forward.go:1007` — per-invocation SSM scratch alloc/free
- `gpu/forward.go:30–86` — `DLGO_SSM_TIME` instrumentation
- `gpu/forward.go:1795` — `supportsGPUQType`
- `gpu/forward.go:1810` — `hasDp4aQType`
- `gpu/ssm_batch.go:10` — `ssmBatchSupported` limits
- `gpu/ssm_batch.go:14` — `ssmBatchPrefill` four-stage sequence
- `gpu/ssm_recurrence.go:9` — `DLGO_SSM_REFERENCE` pattern

**Model / readout**
- `models/llm/systemone.go:86` — `SystemOneHidden`
- `models/llm/systemone_readout.go` — `LoadSystemOneReadout`, `NormalizeDevice`
- `models/llm/arch.go` — `archRegistry` (`qwen35` hybrid SSM)

**Evidence / reports**
- `bench/systemone/RESULTS-qwen35-9b-2026-10-07.md`
- `bench/systemone/RESULTS-qwen38-2b-2026-10-07.md`
- `bench/systemone/meta9b-names.txt` — per-tensor quant map (§3.2 evidence)
- `docs/systemone-parity-plan.md`
- `docs/gpu-ssm-recurrence-optimization.md`

**Env flags referenced**
- `DLGO_SSM_BATCH=1` — batched SSM prefill (already used by the 9B server)
- `DLGO_SSM_TIME=1`, `DLGO_SSM_TIME_CHUNKS` — coarse phase timing
- `DLGO_SSM_REFERENCE=1` — force the decomposed SSM reference
- `DLGO_GEMM_Q3K=1` — **proposed** for K1
- `DLGO_SYSTEMONE_E2E=1` — opt-in model E2E GPU test
