# DiT GEMM: quant abstraction + performance plan

Status: **core implemented** (2026-06). See "Implementation status" at the end.
Scope: `models/diffusion` (Z-Image DiT) GEMM path on the Vulkan backend.
Goal: (a) remove the "one hand-written kernel per quant type" coupling, and
(b) close the gap to the BF16/FP8 ComfyUI pipeline (currently ~12x slower on a
4070-class card).

---

## 1. Where the coupling actually lives

The DiT forward calls **one** function per projection and passes the layer's
weight quant type as an argument:

    models/diffusion/gpu_forward.go (per layer)
        gpu.BatchMatVec(grs.QKV,  gl.AttnQKV.Buf, grs.XNorm, rows, cols, seqLen, gl.AttnQKV.Type)
        gpu.BatchMatVec(grs.Proj, gl.AttnOut.Buf, ...)
        gpu.BatchMatVec(grs.Gate, gl.FFNGate.Buf, ...)   // + FFNUp, FFNDown
        gpu.BatchMatVec(grs.FFNOut, gl.FFNDown.Buf, ...)

    gpu/gpu_cgo.go:940  BatchMatVec -> C.gpu_batch_matvec(..., qtype)

All quant diversity is funnelled through **`gpu_batch_matvec`**
(`gpu/csrc/vulkan_gpu.c:2657`). That function is the entire problem in one
place. It has **two tiers**:

### Tier 1 — fast coopmat GEMM, but only 4 types

    gpu/csrc/vulkan_gpu.c:2665-2695
        QTYPE_BF16 -> gpu_gemm_bf16()
        QTYPE_Q4_K -> gpu_gemm_q4k()     (PIPE_GEMM_Q4K_COOPMAT)
        QTYPE_Q5_K -> gpu_gemm_q5k()     (PIPE_GEMM_Q5K_COOPMAT)
        QTYPE_Q6_K -> gpu_gemm_q6k()     (PIPE_GEMM_Q6K_COOPMAT)
        QTYPE_Q3_K -> gpu_gemm_q3k_*()   (opt-in, default off)

Each of those is a **separate hand-written shader** with its own copy of the
load / dequant / coopmat-FMA / epilogue body:

    gpu/shaders/gemm_q4k_coopmat.comp     gemm_q5k_coopmat.comp
    gpu/shaders/gemm_q6k_coopmat.comp     gemm_q3k_coopmat_recovered.comp
    (+ legacy q4k: gemm_q4k.comp, _fused, _smem, _dp4a, gemm_q4_k)

So *every* time we tune the GEMM (BN=128, vec4 B loads, hoisted quadrant
branch, …) it has to be re-applied by hand to N near-identical files. That is
the maintenance tax you are feeling.

### Tier 2 — the cliff

Every other type (`Q4_0`, `Q8_0`, `Q2_K`, `Q5_0`, `Q5_1`, `TQ1_0`, all `IQ*`,
`MXFP4`, …) falls through to the **matvec** kernel, dispatched as
`groups_y = npos`:

    gpu/csrc/vulkan_gpu.c:2758-2759
        dp.groups_x = (rows + rows_per_wg - 1) / rows_per_wg;  // rows_per_wg = 2..8
        dp.groups_y = npos;                                    // one WG per token

i.e. it re-does a **mat-vec per token** instead of a real GEMM. There is no
weight reuse: every row of W is re-read and re-dequantized for every one of the
4175 tokens. This is why the engine is "dependent on specific quants" — you get
GEMM speed **iff** someone hand-wrote a coopmat kernel for that exact block
type, otherwise you get the matvec cliff.

**That is the abstraction leak: the numeric format is baked into the kernel
family and into the dispatcher instead of being confined to one converter.**

### The right shape already exists (half-built)

There is already a **format-agnostic** path: dequantize to f16 once, then run a
single coopmat GEMM:

    gpu/csrc/vulkan_gpu.c:3260  gpu_gemm_q4k_dequant_coopmat()
        PIPE_DEQUANT_Q4K_F16   (gpu/shaders/dequant_q4k_f16.comp)
            -> f16 scratch
        PIPE_GEMM_F16_COOPMAT  (format-agnostic f16 GEMM)
        gpu_gemm_f16()         gpu/csrc/vulkan_gpu.c:3146

It proves the concept but is wired only for Q4_K and re-dequantises into a
per-call scratch **every forward pass** — so it loses to the fused kernel. The
plan below turns this into the primary path.

---

## 2. The clean abstraction: one canonical runtime format

Split **storage format** (many) from **GEMM kernel** (one):

```
                load time (once)                         runtime (hot, every step)
   GGUF block type ──────────────►  GemmReady tensor ──────────────►  ONE coopmat kernel
   Q4_K/Q5_K/Q6_K/Q8_0/...          (canonical layout)                gemm_coopmat.comp
```

* All per-format logic (bit unpacking, scale/min decoding, IBM-float reversal,
  K-quant sub-block scales, …) moves into a **host-side converter**
  `ConvertToGemmReady(quantTensor) -> GemmReadyTensor`. Host code is easy to
  unit-test against the existing `quant` package.
* The runtime sees exactly **one weight type**. The dispatcher collapses to a
  single `gpu_gemm_gemmready(...)`; `gpu_batch_matvec`'s 130-line switch and the
  per-quant coopmat shaders disappear.
* New quant support = a new function in the converter, **zero** shader work.

### What should `GemmReady` be?

Two viable targets. They are not mutually exclusive — do the first, keep the
second as an opt-in "max speed" mode.

| | **A. Repacked 4-bit block** (recommended) | **B. f16 upcast** |
|---|---|---|
| bytes/weight | ~0.56 (4.5 bpw) | 2.0 |
| model size | ~4.7 GB (unchanged) | ~12 GB |
| decode on hot path | 1 tiny, uniform, aligned | none |
| aligned A loads | **yes** (fixed-width blocks) | yes |
| VRAM fit (16 GB) | comfortable | tight; needs text-encoder freed |
| kernel count | 1 | 1 |

**Option A** is the best default: it keeps the memory footprint that lets the
text encoder stay resident (and larger batch/seq), *and* it fixes the #1 GEMM
bottleneck we measured — the A-path byte-gather (see §3). Repack so each block
is a fixed width and **4/8/16-byte aligned**, so the A load becomes a few wide
contiguous reads instead of 8 scalar `uchar` gathers.

The canonical block I'd standardise on is deliberately dumb, e.g. a 32-element
group of signed 4-bit ints + one f16 scale (+ optionally one f16 min), stored as
`{ int8_t q[16]; float16_t d; }` = 18 B, padded to a 4-aligned 20 B stride. Both
Q4_K and Q5_K/Q6_K convert down into it. That gives one decode function that is
literal SWAR on aligned `uint32` loads — no branch, no bitfield gymnastics.

**Option B (f16)** is worth a flag because it's the closest thing to the
ComfyUI data path (tensor cores with no decode at all). It is only attractive if
we can guarantee the DiT fits alongside nothing else in 16 GB. Measure before
committing.

---

## 3. Why this also makes us faster (perf levers, ordered by ROI)

We measured (Q6_K coopmat, rows=10240 cols=3840 npos=4175):

    MMA + coopmatLoad + barrier floor .........  3.0 ms   (108 TFLOPS)
    A-path (weight loads + decode) ...........  ~7.0 ms  of which ALU ~0.3 ms
    B-path (activation loads) ................  ~0.5 ms
    current tuned kernel ......................  ~9.0 ms

ILD profile of a real 1024^2 step: ffn_gate_up 33%, attention 27%,
attn_qkv 12%, ffn_down 11%, attn_out 7%, cpu ~10%.

1. **Kill the A-path byte-gather (biggest single lever).** Repack to aligned
   fixed-width blocks (Option A) so the weight load is wide and vectorized. We
   measured that the A path is *loads*, not ALU — so this is where the ~7 ms
   goes. Target: fold most of it into the MMA floor. Expected GEMM: ~9 -> ~4-5 ms.
2. **One kernel to tune, not N.** Every scheduling win (double-buffered /
   pipelined shared loads, larger BM to cut A re-reads, BN=256 with a shrunken
   epilogue stage) lands once and helps every layer.
3. **Attention: 27% and LDS-bound — replace with a flash/online-softmax kernel.**
   Today `gpu.AttentionFullF32` materialises the full S x S score matrix. An
   online softmax (running max + rescale) avoids that buffer and its shared
   traffic entirely. This is the second-biggest op.
4. **Fusion to remove launches and round-trips:** qkv+rope; ffn_gate+up+swiglu;
   attn_out + gate-residual. `-debug` shows the launch count already matters
   (374 extra submits inflate the step).
5. **Delete the power-of-two scale hack.** The DiT wraps every GEMM in
   `Scale(1/64)` / `Scale(64)` (gpu_forward.go:657-658, 768-865) purely to keep
   activations inside the f16 coopmat range. If `GemmReady` carry a per-tensor
   `d_scale` that the kernel folds into the MMA accumulator scale, those `Scale`
   passes (and the whole `g_no_coopmat_gemm` escape hatch + its overflow bug
   class) go away.
6. **coopmat2 (`VK_NV_cooperative_matrix2`)** is exposed by this driver and
   untried — flexible M/N/K and better scheduling; a possible later 1.2-1.5x.

---

## 4. Migration plan (incremental, each step shippable)

1. **Converter + `GemmReady` type.** Add `core.GemmReady` (or a field on
   `QuantizedTensor`) and a host converter with round-trip tests against the
   `quant` package reference. No behaviour change yet.
2. **One canonical coopmat kernel** `gemm_gemmready_coopmat.comp` — copy the
   tuned `gemm_q6k_coopmat.comp` body, swap its decode block for the canonical
   4-bit decode, done. Validate bit-parity vs the existing kernels on a layer.
3. **Wire it in behind the dispatcher**: in `gpu_batch_matvec`, route *every*
   quant type through `ConvertToGemmReady` (cached per weight tensor) ->
   `gpu_gemm_gemmready`. Keep the old kernels selectable by env until parity is
   proven, then delete them (`gemm_q{3,4,5,6}k_*`, the q4k legacy family).
4. **Then optimise the single kernel** (§3.1, 3.2) — one target.
5. **Attention flash kernel** (§3.3) as an independent track.
6. **Fusion pass** (§3.4) last.

Repack cost is paid once at load (~4.7 GB read + write, a few hundred ms) and
cached; it does not affect per-step time.

---

## 5. Expected outcome

* **Complexity:** quant-format diversity moves out of `gpu/csrc` + `gpu/shaders`
  and into one host converter. Runtime has one weight type, one GEMM.
* **Speed:** GEMM ~9 -> ~4-5 ms per big op (aligned single kernel); attention
  ~2x from the flash rewrite; plus launch savings from fusion. On the measured
  step that is a realistic path into the ~15-25 s / 8-step range from 44 s,
  without changing the model or the numeric result beyond f16 rounding.

---

## 6. Open questions to settle before step 2

* Q4_K's per-sub-block `min` (used as `d*scale - dmin*min`) — preserve exactly in
  the canonical format (carry the offset), or drop to symmetric 4-bit and accept
  a small quality delta? (Carry it: fidelity first.)
* Option A vs B: does the f16 DiT + text encoder actually fit in 16 GB with the
  encoder freed? If yes, prototype both on one layer and pick by measured kernel
  time.

---

## Implementation status (2026-06)

Steps 1-3 are done. The DiT now runs on a single unified GEMM kernel.

**Canonical format — Affine8** (`quant/affine8.go`): 32 weights / 36 bytes,
`{uint32 q32[8]; f16 d; f16 m}`, `w = d*q + m`. Chosen over f16 because a full
f16 DiT (≈7.8B params) is ~15.6 GB and does not fit in 16 GB. ~8-bit is finer
than any source format, so the conversion is effectively lossless. 36-byte
blocks are 4-aligned so the kernel does one aligned `uint32` load per 4 codes
(the fix for the measured A-path byte-gather bottleneck).

**Converter** (`quant.RepackAffine8` / `QuantizeAffine8` / `DequantizeAffine8`):
reuses the universal `quant.Dequantize` for every scheme, so all per-format
knowledge is confined to one tested host function. Round-trip relL2 ≈ 4.6e-3.

**Kernel** (`gpu/shaders/gemm_affine8_coopmat.comp`, `PIPE_GEMM_AFFINE8_COOPMAT`):
the tuned Q6_K coopmat body (BM=64, BN=128, epilogue staging, tile_m-fastest)
with the decode swapped for Affine8.

**Wiring**: `gpu.UploadTensorAffine8` repacks at load; `gpu_batch_matvec`
routes `QTYPE_AFFINE8` (60) to `gpu_gemm_affine8`. `models/diffusion` uses it
for the five big projections (QKV, AttnOut, FFNGate, FFNDown, FFNUp). AdaLN
(single-token, npos=1) stays on the original path. Escape hatch:
`DLGO_DIT_NO_AFFINE8=1`.

**Verified**
* Kernel parity vs exact f32 CPU GEMM: **relL2 = 2.9e-4** (f16 rounding only).
* Kernel speed on the DiT hot shape (10240×3840×4175): **9.45 ms / 34.7 TFLOPS**
  — equal to the hand-tuned Q5_K/Q6_K kernels, now from one kernel for all types.
* End-to-end: sampling **42.1 s** (baseline 43.8 s), `nonfinite = 0/262144`.
* Image vs reference: **PSNR 34.7 dB, meanAbs 1.49/255** (weight-requantization
  noise only).
* Escape hatch reproduces the reference image **byte-for-byte** (identical MD5),
  proving the legacy path is untouched.

**Not yet done**
* Unify the LLM path too (it still uses the per-quant GEMM selection).
* Delete the now-redundant per-quant coopmat shaders once the LLM is migrated and
  the Affine8 path is the sole option (the escape hatch still needs them).
* §3 perf levers: pipeline/double-buffer the Affine8 kernel (single tuning
  target now), flash-attention, fusion, drop the power-of-two `Scale` hack.
