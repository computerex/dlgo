# 9B latency plan execution — 2026-10-09

## Model facts verified from the artifact (gguf_info, not recalled)
- 32 layers: 24 SSM (blk 0-2, 4-6...) use fused `attn_qkv` (4096x8192, Q4_K) + `attn_gate`
  (4096x4096, Q3_K) + `ssm_out` (4096x4096, Q4_K); 8 attention layers (3,7,11,15,19,23,27,31)
  use `attn_q` 4096x8192 (Q3_K), `attn_k` 4096x1024 (Q3_K), `attn_v` 4096x1024 (Q4_K/Q5_K),
  `attn_output` 4096x4096 (Q4_K).
- All 32 `ffn_gate` are Q3_K (4096x12288). `ffn_down` is Q4_K/Q5_K (12288x4096).
- SSM per layer: `ssm_alpha`/`ssm_beta` 4096x32 (Q8_0), `ssm_a`/`ssm_dt.bias` 32, conv 4x8192,
  `ssm_norm` 128. Confirms heads=32, key groups=16, state 128x128 used by the kernels.
- The earlier "attention ~4 TFLOP could be ~15" note applies to the 8 attention layers only;
  chunked causal scoring spans full preceding context (seq up to ~10.8K at the last chunk).

## Operator profile (diagnostic; includes submit/wait per bracket)
23 counted chunks, long 3-question fixture, plain K1/T16, v2-uni head
(latency-plan-operator-profile-server.log / .json / -manifest.json):

| Operator | Time |
|---|---:|
| SSM input/gate/alpha/beta projections | 2.401 s |
| SSM conv + preprocess + norm | 0.138 s |
| SSM recurrence | 2.694 s |
| SSM output projection | 0.801 s |
| Attention QKV/Q/K/V projections (8 layers) | 0.900 s |
| Attention norm/RoPE/KV store | 0.007 s |
| Attention score/softmax/PV (8 layers) | 6.636 s |
| Attention gate/output | 0.241 s |
| FFN whole block | 11.604 s |

Focus ranking: FFN 11.6 s > attention score/PV 6.6 s > SSM recurrence 2.7 s > SSM projections 3.2 s.

## Completed
- Deleted coopmat Q3_K references removed from enum/dispatch/gate/Go hook/compile registry.
- Stray duplicate timing-test source removed (original preserved in gemm_q3k_test.go).
- Full shader regeneration; clean GPU parity tests, GPU vet, server build; CPU tests passed.
- SSM + attention sub-operator profiling behind DLGO_SSM_TIME; report now prints after the
  final counted chunk rather than before executing it.
- Isolated-server matched-nonce benchmark harness with env manifests and logs.
- Opt-in barrier-reduced attention (`DLGO_ATTN_BARRIER`): identical dot/reduction order,
  fewer shared broadcasts/barriers; compiled, spirv-val clean.
- Fast surgical SPIR-V embedder (embed_spirv.py) validating name/registry/enum counts.
- Recovered Q3_K coopmat candidate generated from the Q4_K template plus the verbatim tiled
  Q3_K dequantization; compile-registered as PIPE_GEMM_Q3K_RECOVERED, **never selected by
  inference**, reachable only through a numerical test, skipped for non-coopmat devices.
- Expanded T32 test to the real model shape (32 heads, 128x128, 512-token chunks).

## Rejected candidates (measured, do not re-derive)
- **KV-grouped attention (DLGO_ATTN_KV4)**: bit-identical output (maxAbs=0) but **3.5-4.2x
  SLOWER** (15.1 ms vs 4.3 ms at seq 4K; 11.7 ms vs 2.8 ms at seq 10.7K). First revision had a
  redundant-work bug (all 256 threads computed every head's dots); the deeper finding survives
  the bug: baseline attention is **compute/latency-bound, not KV-memory-bound** — per layer the
  KV working set (~44 MB at seq 10.8K) fits the 48 MB L2 and cross-head reuse hits. Cutting
  traffic 4x cannot win; only raising FLOP throughput can. Kernel + test kept default-off as a
  documented negative result (`attention_tiled_kv4.comp`, `attention_kv4_test.go`).
- **Barrier-reduced attention (DLGO_ATTN_BARRIER)**: noise-level timings (0.93-1.51x). Dead.
- **"Attention is overhead-dominated" claim**: corrected. Cross-checking the microbench scaling
  (~2.8 ms per call at n=16, seq 10.7K) against real chunk sizes (~npos 512, seq to 10.8K, 8
  layers, 22 chunks) reproduces the ~6.6 s score/PV block as genuine kernel compute at roughly
  1-2 TFLOPS effective. The 6.6 s is real work, not dispatch overhead.

## Attention: the remaining lever
- **W3b coopmat flash-attention**: QK^T and PV via cooperative matrix f16xf16->f32 (head_dim 256,
  position tiles 256 — shapes fit the existing `gemm_q4k_coopmat.comp` pattern). Baseline runs
  ~1-2 TFLOPS effective on scalar chains; a working coopmat path plausibly reaches 3-5x on the
  6.6 s. High risk (online softmax on tensor-core tiles, chunk-seam masking, npos=1 fallback);
  needs a design doc + gates before implementation. This is the decision point for significant
  attention savings; everything cheaper has been measured and rejected.

## Recovered GEMM E2E (3+3 paired, long fixture)
- Cold p50: 25,258 ms (off) vs **24,925 ms (on)** → ~333 ms (~1.3%), within 3-sample noise;
  warm slightly worse (851 vs 841 ms). Does **not** replicate the historical 21.7 s result —
  consistent with the kernel timing: my recovered revision is only **1.02x** vs K1 tiled
  (6.54 vs 6.64 ms at npos=512); the historical 1.45x kernel likely hoisted the per-element
  `extract_scales()` (my re-derivation recomputes it 256x per block). Parked.
- Decisions: all 9 identical (type/choice). Confidence drift <=0.0022 (K1-precedent reduction
  reorder). The earlier "3 answer mismatches" was my comparator comparing probability floats;
  fixed in `run_pair.py` (decision-level gate).

## In flight## In flight
- T32 corpus gate: 693-row validation with T32 on (validate-9b-uni-t32.ps1). Gate: no more than
  one additional GPU error vs 649/693; NLL/Brier within 2%; ECE reviewed.
- Next: W3b coopmat flash-attention scope doc (go/no-go decision for the user).

## T32 paired E2E — PROMOTED CANDIDATE (3+3 matched nonces, long fixture)
- Cold p50 **25,335 ms (off) vs 24,972 ms (on)** = **363 ms (1.45%)**; p95 25,401 -> 25,036 ms.
- **Zero decision changes; probability delta exactly 0.0** (bit-identical features, as designed).
- Matches the scaled kernel prediction (recurrence 2.69 s x ~1.33x -> ~0.7 s; observed 0.36 s
  cold — recurrence's share is evidently smaller than the profile bracket suggested).

## Corrections recorded (do not reintroduce)
- 50 GB / 672 GB/s = 0.075 s, not 75 s; that arithmetic cannot diagnose a bandwidth bottleneck.
- 44/88 TFLOPS Tensor Core peaks unverified; "physical impossibility" claim withdrawn.
- SSM batch path: four dispatches per chunk; ssmFlushInterval affects fallback only.
- Block timings include projections; assumed serial floors cannot diagnose overhead.
- Chunked causal attention retains quadratic total work across the context.
- HTTP answers_identical is cold-vs-cached within a config, not optimized-vs-baseline proof.
- 12-15 s is a provisional target, not an achieved result or proven lower bound.
