# 9B System One: fresh-context GPU latency below 3 seconds

> **SUPERSEDED.** The sub-3s goal for the 10.8K-token workload was dropped after a
> first-principles roofline showed it is unattainable on the RTX 4070 Ti SUPER (FFN alone
> costs 104 TFLOP, which is 2.36 s at 100% of the 44 TFLOPS tensor peak). See
> `docs/systemone-9b-latency-plan.md` for the current practical plan. This document is kept
> only for its measurement methodology, gates, and the historical profile below.

## Objective and contract

Target the existing Qwen3.5-9B-Q3_K_M backbone with the device-agnostic union readout, fully GPU-resident on the RTX 4070 Ti SUPER. Preserve all input tokens, all layers, independent question semantics, and the existing public API. No retraining, truncation, approximate attention, layer skipping, or prefix-cache hits counted as fresh-context success.

Primary workload: existing `bench/systemone/long-request.json`, approximately 10.8K input tokens, three questions. Secondary release workload: ten distinct short questions on the same context; freeze a fixture before collecting its baseline. Also test short, 1K, 4K, 8K and maximum supported context lengths and 1/3/10 questions.

Acceptance target: warmed, model-resident, single-client fresh-context HTTP p95 below 3,000 ms for the primary workload and separately for the ten-question fixture. Report initialization separately. This is an ambitious engineering target, not a demonstrated feasibility claim or a guarantee for every input or GPU.

## Verified baseline

Historical K1 result: three questions, fresh long context 25,325 ms p50, cached 841 ms. K1 tiled float32 Q3_K GEMM is already enabled in these figures; baseline before K1 was 87,901 ms.

Unified-head diagnostic: `bench/systemone/latency-9b-uni-long-profile.json`, two runs, fresh mean 25,264 ms and cached mean 838 ms. Timing instrumentation introduces submit/wait boundaries, and two samples are insufficient for reliable tail estimates.

Coarse categorized GPU-path profile (host synchronization included):

| Block | Seconds | Share |
|---|---:|---:|
| FFN | 11.275 | 45.8% |
| Full attention block | 7.479 | 30.4% |
| SSM block | 5.707 | 23.2% |
| Embed/upload | 0.144 | 0.6% |

These are block costs, not isolated operator times. Attention/SSM blocks include projections. Do not attribute all attention-block time to attention or all SSM-block time to recurrence. This supersedes the pre-K1 short-input 75%-FFN profile.

Live unified-head validation, 693 examples: GPU accuracy 93.65%, NLL .2109, Brier .1045, ECE .0365; CPU accuracy 93.51%. Current test results are reference baselines, not tuning targets.

## Why one kernel change is insufficient

25.26 / 3 = 8.42x total acceleration required. Eliminating the entire FFN category would still leave about 13.3 seconds of categorized work. Even a 10x FFN improvement alone cannot reach the target. Multiple blocks must improve together.

Planning budget, not predicted measurements:

| Phase | Target budget |
|---|---:|
| Context FFN | 0.85 s |
| Context attention blocks, including projections | 0.65 s |
| Context SSM blocks, including projections | 0.65 s |
| Ten question branches and readouts | 0.45 s |
| Host, tokenization, transfers, snapshot management | 0.15 s |
| Tail/measurement margin | 0.25 s |
| Total | 3.00 s |

The coarse baseline includes prefix and some suffix work, so this budget is not a direct operator-by-operator speedup forecast. Replace it with measured operator budgets in phase 0.

## Phase 0 — trustworthy attribution and baseline

1. Capture exact executable/source state, device/driver, clocks and power, shader capabilities, context limits, enabled flags, input token counts and fixture hashes.
2. Measure unified-head long-context baseline with profiling off. Warm model once, use matched unique state nonces to invalidate prefix cache, measure at least 30 fresh requests interleaved with their cached counterparts. Confirm zero degraded answers and no accidental standalone fallback.
3. Add Vulkan timestamp query instrumentation at operator/category boundaries, retrieving queries after normal submission completion rather than inserting a wait after each operator. Check timestamp support, valid bits and timestamp period. Sampling must be bounded and opt-in.
4. Attribute by layer, operator, quantization type, M/N/K shape and prefix versus suffix: Q3/Q4/Q5/Q6 projections, FFN elementwise work, SSM preprocessing/recurrence/norm, attention score/softmax/value, KV writes, snapshot/restore, dispatch/submit/wait, host tokenization and transfers.
5. Record memory high-water marks, descriptor/command allocation counts and chunk count. Use the coarse Sync profiler only as a cross-check, not a production latency baseline.
6. Microbench real model shapes at 16/32/64/128/256/512 tokens. Distinguish compilation/first-touch setup from steady execution.

Deliverable: ranked operator table and latency scaling curves. Proceed on measured cost, not the historical pre-K1 profile.

## Phase 1 — accelerate quantized projections (highest-confidence first implementation)

### 1A. Q3_K cooperative-matrix GEMM

`gpu/shaders/gemm_q3k_tiled.comp` explicitly uses float32 shared tiles and no tensor cores. Implement a separate cooperative-matrix shader based on the existing Q4_K pipeline structure, preserving Q3_K's 110-byte block layout and scale/sign decoding.

- Stage dequantized Q3_K tiles and activations in supported operand types, with float32 accumulation where the device supports it.
- Avoid repeating full scale unpacking for each scalar element; unpack per block/tile and share the result.
- Support tails, output stride, chunked output rows, and all real Q3_K projection shapes.
- Tune tile sizes/subgroups for long prefix and short suffix independently. Do not assume a long-prefix winner benefits short branches.
- Route only on positively detected cooperative-matrix shape/type support; retain tiled float32 and original fallbacks. Vendor-name checks are not capability checks.
- Packed GGUF weights remain unchanged. FP16 operand staging changes arithmetic and requires opt-in rollout and accuracy validation; it is not bit-identical.

This improves FFN gate/up and Q3_K projections inside both attention and SSM blocks. Count the total projection saving once, not independently in each category.

### 1B. Whole-projection and FFN efficiency

After 1A, profile Q4/Q5 down and other projections again. Optimize operand conversion, tile reuse, padding and dispatch shapes across those paths as needed. Explore gate/up joint scheduling or fusion where it actually reduces input loads; avoid concatenating weights through an expensive per-request copy.

Fuse SiLU(gate) times up and appropriate residual/norm epilogues where mathematically valid. Eliminate intermediate writes only when the next projection's consumption remains efficient. Bench register pressure and occupancy, not just dispatch count.

Gate: real-shape kernel speedup, complete pipeline improvement, acceptable numerical drift, no out-of-bounds tails, no unsupported-device regression.

## Phase 2 — long-context attention

First subtract projection time from the attention-block profile.

- Inspect the actual dispatch selected for the 9B's head dimensions, GQA layout, FP16 KV and causal chunk offsets. Verify existing flash/tiled attention paths are actually applicable before writing another kernel.
- Optimize a flash-style prefill path with tiled QK and probability-V operations, online softmax and bounded on-chip score storage. Do not materialize the entire long-context score matrix.
- Preserve causal masking, RoPE positions, GQA mapping and prior-chunk KV semantics. Chunk boundaries are not independent attention windows.
- Tune query/key block sizes and sequence-length specializations; use existing hardware matrix support where beneficial. Check accumulation precision and softmax stability for long contexts.
- Avoid rereading historical KV unnecessarily; retain compatible FP16 cache unless a measured precision requirement prevents it.

Gate: standalone-vs-multi and full-vs-chunked numerical parity within justified tolerances, long-context adversarial tests, lower attention execution time and lower full-request latency.

## Phase 3 — SSM recurrence and preprocessing

First separate recurrence from QKV/gate/output projections. Existing `ssm_recurrence_batch.comp` loops sequentially over tokens, retains 128-element columns, and uses ordered compensated prediction/output sums. This is already fused; simply fusing it again is not the next algorithmic improvement.

Two tracks:

1. Low-risk tuning: benchmark workgroup/value tile sizes, register occupancy/spills, shared Q/K staging and preprocessing/norm fusion. The current shader uses 16 lanes despite older documentation describing 32; use code and compiled measurements as authoritative.
2. If recurrence remains a dominant floor: implement a model-correct chunkwise gated delta-rule formulation. Parallelize intra-chunk work through matrix operations, propagate the correct inter-chunk recurrent state, and preserve the convolution state. Do not apply a naive associative scan to the rank-one delta update or erase initial-state contributions.

Use float32 state accumulation initially; any mixed-precision change is separate and validated. Algebraically equivalent algorithms reorder floating-point operations and must remain gated. Test arbitrary nonzero initial states, head/KV-group mapping, extreme decay/beta values, full/chunk/suffix parity, and long-state drift.

Gate: isolated recurrence speedup and complete-request gain; state snapshot/restore correctness and accuracy retained.

## Phase 4 — chunking, scheduling and memory traffic

- Tune prefill chunk sizes only after extending applicable kernels and scratch capacities. Existing SSM support is limited to npos <=512; changing a loop constant alone is unsafe.
- Compare 128/256/512 and, when supported, 1024-token chunks for latency, occupancy and memory. Larger chunks reduce overhead but can regress attention/SSM or exceed scratch limits.
- Reuse bounded scratch/descriptor resources and remove demonstrably redundant host/device transfers and waits. Preserve required Vulkan barriers and buffer lifetimes.
- Track command submission cost; batch commands where dependency ordering permits. Avoid global synchronization between independent operator preparation steps.
- Treat one-time persistent weight repacking/dequantization as an optional experiment only after a VRAM budget. Expanding all 9B weights to FP16 is not assumed to fit alongside long-context KV, scratch, and branch state on 16 GB VRAM.

Gate: no increasing RAM/VRAM over repeated unique contexts, allocation reuse under cancellation, no race or barrier-related corruption.

## Phase 5 — ten independent question branches

The API already shares context prefill. Current optimized multi orchestration runs suffixes sequentially. Build branch-aware batching after the expensive fresh-prefix path has improved.

- Share immutable prefix KV; allocate independent branch suffix KV, recurrent/convolution states and positions.
- Batch compatible suffix projections across questions; keep attention masks and SSM recurrence branch-local. Concatenating questions into one causal stream is incorrect.
- Bucket variable suffix lengths with correct masking; support different choices and maintain result order and question IDs.
- Retain seam-safety checks, fallback semantics and cancellation handling.
- Avoid full-prefix duplication per branch. Include memory budget and safe fallback when branch capacity is exceeded.

Gate: results match standalone question inference within numerical tolerances, no cross-question leakage, and actual 10-question p95 below budget. One-question performance must not regress materially.

## Correctness and rollout gates

- Preserve the reference default while developing capability-gated opt-in paths; CPU path and union artifact format remain unchanged.
- Synthetic kernels plus real tensors: max absolute/relative error, hidden-state cosine, finite values, tail shapes, FP16 under/overflow. Select tolerances based on model behavior, not arbitrary convenient thresholds.
- Use development/calibration sets for tuning. Run the fixed 693-example held-out test as a release gate rather than repeatedly tuning to it. No more than one additional GPU test error versus the 649/693 reference without explicit review; aim for zero. NLL/Brier no more than 2% relative degradation; ECE must be reviewed for worsening.
- Re-run unseen-domain fixtures including chess and Yu-Gi-Oh, plus long-context retrieval/contradiction fixtures beyond the short corpus. Record confidence changes, not only hard answer matches.
- Benchmark profiling off: p50/p95, at least 30 fresh requests, matched nonces, separate cached times, question counts and token lengths. Increase sample counts before claiming robust tail service guarantees.
- Validate optimized multi against standalone at nonzero state, after cache eviction, cancellation and repeated requests; assert no degraded answers.
- AMD/Intel capability fallbacks must remain functional. NVIDIA latency success is not an AMD/Intel latency claim; arrange vendor testing before broad performance claims.

## Milestones and decisions

1. M0: clean unified-head baseline and operator timestamps.
2. M1: Q3_K hardware GEMM and shape tuning; report complete-request improvement.
3. M2: attention and SSM follow-up selected by residual measured costs.
4. M3: allocation/chunk tuning and ten-branch batching.
5. M4: sub-3s fresh-context acceptance on both fixed workloads, accuracy gates, and bounded-memory soak.

Do not declare success from warm-cache sub-second measurements. If efficient projections/attention and a tested chunkwise SSM still leave a measured floor above the budget, report that floor and reassess hardware or model size explicitly. Smaller backbones/distillation are separate product choices with new accuracy validation, not silent substitutes for 9B.

Jev parity is a separate matched-service experiment: identical state/question semantics and context lengths, cache conditions, output requirements, network accounting, and documented hardware/service tier. Achieving a local target does not independently establish vendor parity.
