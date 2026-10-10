# 9B System One: practical GPU latency plan

Supersedes `systemone-9b-sub3s-plan.md`. The sub-3s target is dropped for the 10.8K-token
fixture: it is not a useful near-term engineering target at the current measured throughput.
The earlier physical-impossibility argument had unverified peak and unit errors; it is withdrawn.
This plan instead targets gains justified by operator profiles and matched benchmarks.

## 0. Why sub-3s was dropped (keep this reasoning)

FFN FLOPs are fixed by the architecture:

```
2 x 4096 x 12288 x 3 projections x 32 layers = 9.66 GFLOP/token
x 10,816 tokens                              = 104 TFLOP
```

RTX 4070 Ti SUPER throughput must be measured for the active kernels, not inferred from a mislabeled peak.
The previous 44/88 TFLOPS Tensor Core figures were not verified and must not be used to claim a
physical lower bound. The fixed FFN arithmetic above is valid; measured K1 throughput is about
7.2 TFLOPS for the gate/up shape, with a historical coopmat result around 10.4 TFLOPS.
These rates describe existing implementations, not an upper bound on future implementations.

Dropping the sub-3s goal is a practical prioritization decision based on current ~25 s latency,
not a proven physical impossibility. The 12-15 s goal is provisional until operator profiles and
matched uninstrumented benchmarks establish attainable gains.

## 1. Corrected facts (fix earlier plan errors)
- **Bandwidth diagnosis is unresolved.** 22 x 2.29 GB ~= 50 GB, and 50 GB / 672 GB/s ~=
  **0.075 s**, not 75 s. Neither that calculation nor block timings establish L2 hit rates or
  whether a kernel is limited by memory, dequantization, instruction throughput, or occupancy.
- **Whole-block timings include projections.** SSM 5.71 s and attention 7.48 s are not timings
  of recurrence or score kernels alone. Operator profiles are required before prioritizing.
- **Aggregate FLOP/s is descriptive, not a roofline proof.** The FFN count is derived, but
  attention/SSM FLOP counts need the actual model dimensions and operations. Causal chunked
  attention still attends to preceding context; chunking does not eliminate quadratic work.
- **No guaranteed format speedup.** Less aggressive quantization may simplify dequantization,
  but must be timed at matched shapes. Dequantizing existing Q3_K weights to resident FP16 is
  distinct from loading a differently quantized artifact; both have memory/numerical gates.

## 2. Where the achievable time is

Measured 10.8K-token, 3-question fresh-context baseline (**v2-uni head, plain K1**):
`bench/systemone/latency-9b-uni-long-profile.json`, cold p50 **25,010 ms** / warm 838 ms.
Categorized: FFN 11.28 s, attention 7.48 s, SSM 5.71 s, embed 0.14 s. (The older v1-gpu head
measured ~88 s cold — `bench/systemone/9b-long-latency.json` — do not confuse the two.)

A *correct* coopmat revision previously reached **21,729 ms cold / 722 ms warm** with identical
answers (`bench/systemone/latency-9b-uni-long-coopmat.json`); recovering it is the W4a target.

Provisional stretch targets, **not validated forecasts**; revise after operator profiling:

| Block | Now | Practical target | Basis |
|---|---:|---:|---|
| Attention | 7.48 s | 1.5-2.5 s | overhead removal + tiled/flash rewrite |
| SSM | 5.71 s | 1.0-2.0 s | occupancy + reduced per-token overhead |
| FFN | 11.28 s | 8-10 s | better Q3_K kernel efficiency only |
| Embed/overhead | 0.14 s | unchanged | negligible |

That is roughly **25 s -> 12-15 s** on the long fixture without changing the model, plus
larger relative wins on shorter contexts where FFN has less weight.

## 3. Work items, cheapest first

Ordering is by (gain / risk). Each item is independently shippable and independently
measurable. Do not start the next until the current item's effect is measured end-to-end.

### W1. Finish the already-validated SSM occupancy win (in progress)
- 32-lane recurrence kernel `ssm_recurrence_batch_t32.comp` is implemented and proven
  **bit-identical** to the 16-lane kernel (output, recurrent state, conv state) by
  `TestSSMBatchT32ParityGPU`. Gated by `DLGO_SSM_REC_T32=1`.
- Remaining: measure end-to-end on the long fixture. Kernel-level parity is done; the E2E
  number is the deliverable.
- Risk: none (bit-identical, opt-in). Cost: one benchmark run.

### W2. Profile the SSM sub-operators before changing recurrence
The 5.71 s measurement includes input/gate/alpha/beta and output projections, not just recurrence.
The active `DLGO_SSM_BATCH=1` path already uses four dispatches per chunk. The earlier
200-500 ns/token serial-floor estimate was an assumption, not a hardware-derived bound, and
cannot establish a 100-400x overhead gap for the whole block.
- W2a. Measure projections, conv, preprocess, recurrence, norm, and output separately.
- W2b. If recurrence dominates, test larger value tiles or specialized state dimensions while
  preserving per-column Kahan ordering; measure occupancy/register pressure as well as timing.
- W2c. Consider fusion only if the measured stage/submission costs justify it.
- `ssmFlushInterval` affects the per-token fallback **only**; do not tune it for the batched baseline.
- Risk: medium. Gate: state/output parity and the fixed corpus for numeric changes.

### W3. Attention: measure first, then one targeted kernel change
- W3a. Produce a real operator breakdown of the attention block (QKV projections vs RoPE vs
  KV store vs score/softmax/PV). Some projections are Q3_K and use tiled FP32 K1, not coopmat.
  Do not presume that score/softmax/PV accounts for the 7.48 s. No change without this breakdown.
- W3b. If scores dominate: replace scalar unpack-and-FMA scoring with a tiled path that
  keeps scores in shared memory and uses cooperative matrices for QK^T and PV where the
  shape allows. Preserve exact masking/softmax semantics; validate long-context and
  chunk-boundary parity.
- Risk: medium-high (attention correctness). Gate: standalone-vs-multi and full-vs-chunked
  parity, adversarial long-context tests, and corpus accuracy unchanged.

### W4. FFN efficiency only (no format change)
- W4a (**promoted to high priority**). Recover a numerically validated cooperative-matrix Q3_K path.
  The historical `latency-9b-uni-long-coopmat.json` reports 21.7 s versus ~25 s K1, making recovery
  worth investigating. However `answers_identical` compares cold vs cached responses **within**
  a run, not coopmat against K1, and the configs/nonces were not pinned in those files. Treat it
  as a candidate performance result to reproduce, not a release-grade correctness proof.
  Later revisions produced zeros; any reconstructed shader requires independent CPU-reference
  tests, matched E2E requests, and the held-out corpus before promotion. Do not silently enable it.
- W4b. Cheap independent gains: hoist scale unpack out of the inner loop, use 32-bit weight
  loads instead of byte loads, and tune tile sizes for occupancy. The earlier v2 attempt
  regressed because it doubled shared memory (occupancy loss); treat shared-memory budget as
  a first-class tuning parameter and measure occupancy, not just timing.
- Risk: low each, but only ~10-20% total. Do not let this block W2/W3.

## 4. The one lever that actually changes the FFN wall (decision required)
FFN FLOPs are fixed, but achieved TFLOPS is capped by Q3_K dequant ALU. A **less aggressive
quantization with a pure tensor-core path** (Q4_K/Q5_K coopmat, or FP16 weight residency)
would run the same FLOPs at materially higher throughput, i.e. **shorter wall time**, paid
for in VRAM.
- This is a trade, not a free win: it changes memory footprint and requires re-validation of
  the readout head against the new feature distribution.
- Action: measure the achieved TFLOPS of the existing Q4_K coopmat path (`ffn_down` uses it)
  versus Q3_K on the same shape. If Q4_K is >=1.8x faster at equal FLOPs, quantify the VRAM
  cost of converting `ffn_gate`/`ffn_up` to Q4_K and present it as a product decision.
- Do **not** silently substitute a different quant: it changes numerics and the artifact.

## 5. What is explicitly out of scope
- Truncating or subsampling context; layer skipping; early exit; approximate attention;
  smaller-model substitution. These change answers and are not latency work.
- Counting prefix-cache hits as fresh-context success.
- Redesigning the model to chase 3 s at 10.8K tokens. This goal was dropped as a practical priority,
  not disproven by a valid hardware lower bound.
- Rebuilding working code. K1 tiled Q3_K GEMM is correct and stays; the T32 SSM kernel is
  proven bit-identical and stays. (Recovering the coopmat Q3_K revision in W4a is *not* a rebuild —
  a correct revision existed and delivered 21.7 s; it is re-derivation of lost untracked work.)

## 6. Scoping by workload (state this in release notes)
- **Short/decision prompts (~512 tokens, the real corpus): already ~0.9 s cold / ~0.6 s warm.**
  This is the band the product lives in; protect it from regressions.
- **1-4K tokens: W2 + W3 should bring this comfortably under ~2-3 s**, which is the
  practical target for "large context" decisions.
- **10.8K tokens: provisional target ~12-15 s**; release notes must report measured latency,
  not describe this target as an established physical limit.

## 7. Measurement and gates (unchanged discipline)
- Baseline and every result: warm model, matched unique-state nonces to force fresh prefix,
  at least 30 fresh requests, report p50 and p95, profiling instrumentation off.
- Correctness gates for any numeric change: kernel numerical tests (max abs/rel error,
  hidden-state cosine), standalone-vs-multi parity, SSM state/conv parity, and the fixed
  693-row held-out test - no more than one additional GPU error versus the 649/693
  reference, NLL/Brier within 2% relative, ECE reviewed if it worsens.
- Memory: no growth across repeated unique contexts; allocation reuse under cancellation.
- Vendors: AMD/Intel fallbacks must remain functional; NVIDIA numbers are not vendor claims.

## 8. Next actions (concrete, in order)
1. Implement the pending cleanup so the tree builds: remove the reverted cooperative-matrix
   Q3_K remnants (enum entry, C functions and gate, `SetGemmQ3KCoopmat`, compile.ps1 entry)
   and delete the stray non-test `gpu/gemm_q3k_test_timing.go`.
2. Regenerate `gpu/csrc/shaders_spirv.h`, `go vet`/`go build` the vulkan target, run
   `TestSSMBatchT32ParityGPU` and `TestGemmA3KTiledGPU`.
3. Rebuild `bench/systemone/server-9b.exe`; run the long-fixture latency benchmark with
   `DLGO_GEMM_Q3K=1` and `DLGO_SSM_REC_T32=1`; record the E2E delta for W1.
4. Run the W3a attention breakdown (extend `DLGO_SSM_TIME` or add attention sub-timers).
5. Re-derive the Q3_K coopmat revision (W4a, now high priority): reproduce the working 1.45x
   revision from the `gemm_q4k_coopmat.comp` template, validate numerically, and re-measure the E2E
   delta; then find the zero-output regression cause. Target: re-attain the 21.7 s
   `latency-9b-uni-long-coopmat.json` number from a clean build.
6. Pick W2 or W3 by whichever the breakdown says is bigger, implement, validate, measure.

> See `systemone-9b-latency-plan-validation.md` for the independent re-derivation of every number
> in §0-§2 and the evidence behind the W4a re-prioritization.
