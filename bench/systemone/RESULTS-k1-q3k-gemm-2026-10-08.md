# K1: Q3_K batched GEMM (System One kernel optimization) — results (2026-10-08)

Implementation of **K1** from `docs/systemone-kernel-optimization-plan.md`: a fused
Q3_K → FP32 tiled GEMM that replaces the per-token Q3_K matvec fallback for the
`npos > 1` (batched prefill / System One) path. Gated behind `DLGO_GEMM_Q3K=1`
(**default off** — the default path is unchanged).

## Why this exists

`gguf_info Qwen3.5-9B-Q3_K_M.gguf -names` shows the two largest FFN matrices
(`ffn_gate`, `ffn_up`, 4096×12288, all 32 layers) are **Q3_K**, and
`gpu_batch_matvec` had no Q3_K GEMM — so they took the per-token matvec fallback
(`matvec_q3_k.comp`), which re-reads the weight matrix once per token. The 9B's
`DLGO_SSM_TIME` profile put **75% of time in FFN**, consistent with that.

## What was built

- `gpu/shaders/gemm_q3k_tiled.comp` — TM=64 tokens × TN=64 rows, 256 threads,
  TK=32 K-tile; dequantizes the Q3_K tile to **f32** in shared memory once and
  reuses it across the token tile. Dequant ported verbatim from
  `quant/q3_k.go` (`DequantizeQ3_K`), the authoritative CPU reference.
- `gpu_gemm_q3k_tiled()` + gated dispatch branch in `gpu_batch_matvec`.
- `PIPE_GEMM_Q3K_TILED` (appended last; existing pipeline IDs unchanged).
- `gpu/gemm_q3k_test.go` — numerical parity test + kernel timing test.

Build: `-tags vulkan -ldflags "-linkmode internal"`. Run with `DLGO_GEMM_Q3K=1`.

## Correctness

**Numerical parity (synthetic Q3_K vs `quant.DequantizeQ3_K`)** —
`TestGemmA3KTiledGPU`: **maxAbsErr 9.2e-05, maxRef 194.98, relErr 4.7e-07**.

**Decision-level parity (real model, 120-row subset of splits-gp2 test, flag off vs on):**

| Metric | Flag OFF | Flag ON | Δ |
|---|---:|---:|---:|
| Overall accuracy | 88.33% (106/120) | 88.33% (106/120) | 0 |
| NLL | 0.2788 | 0.2787 | −0.0001 |
| Brier | 0.1650 | 0.1650 | 0 |
| ECE | 0.0766 | 0.0766 | 0 |
| Zero-shot (85 rows) | 85.88% | 85.88% | 0 |
| Seen (35 rows) | 94.29% | 94.29% | 0 |

**Per-row: 0 decision changes across 120 rows.** Only confidences shift — reduction
reorder only: **max Δ0.001, mean abs Δ0.00011.**

## Speedup (isolated kernel)

`TestGemmA3KBatchTimingGPU` on the real FFN gate/up shape (4096×12288):

| npos | matvec fallback | K1 GEMM | speedup |
|---:|---:|---:|---:|
| 64 | 5.640 ms | 1.075 ms | **5.25×** |
| 128 | 11.218 ms | 1.960 ms | **5.72×** |
| 256 | 21.765 ms | 3.657 ms | **5.95×** |
| 512 | 43.144 ms | 7.145 ms | **6.04×** |

The matvec scales ~linearly with `npos` (per-token weight re-read); K1 is sublinear.
This clears the plan's Phase-1 "≥5× on the isolated kernel" exit criterion.

## Default-path isolation

The only change to the default dispatch is one guarded `if` requiring
`gemm_q3k_available()` (returns 0 unless `DLGO_GEMM_Q3K=1`); with the flag unset the
Q3_K matvec fallback is byte-for-byte the default behaviour. Verified against a
pristine `git worktree` at HEAD that the two failing GPU tests
(`TestAllocSSMScratch_ExactAllocCount`, `TestAttentionFullF32`) are **pre-existing**
and unrelated.

## Full-corpus accuracy (693-row splits-gp2 test, flag off vs on)

| Metric | Baseline | K1 ON | Gate | Status |
|---|---:|---:|---|---|
| Overall accuracy | 93.07% (693) | **93.07%** | ≥92.9% | ✅ |
| NLL | 0.2208 | **0.2208** | ≤+2% rel | ✅ |
| Brier | 0.1111 | **0.1111** | ≤+2% rel | ✅ |
| ECE | 0.0372 | **0.0372** | not worse | ✅ |
| Zero-shot (517) | 92.65% | **92.65%** | — | ✅ |
| Seen (176) | 94.32% | **94.32%** | — | ✅ |

Identical to 4 decimals on NLL/Brier/ECE; identical accuracy. The plan's §7.3
regression budget is fully met.

## End-to-end server latency (HTTP, 9B, matched nonces, 3 questions)

`http_latency.py` against `server-9b.exe` with `DLGO_GEMM_Q3K` off vs on.
(`serve-9b-q3k.ps1` is the on variant.)

| Workload | K1 OFF | K1 ON | Speedup |
|---|---:|---:|---:|
| 3q short cold (p50) | 2,558 ms | **920 ms** | **2.72×** |
| 3q short warm (p50) | 1,546 ms | **577 ms** | **2.68×** |
| 3q 10.8K-doc cold (p50) | 87,901 ms | **25,325 ms** | **3.47×** |
| 3q 10.8K-doc warm (p50) | 1,826 ms | **841 ms** | **2.17×** |

Answers identical across configs (only the `noul` confidence shifts by ~0.0008 —
reduction reorder). `answers_identical=true` within every run; 0 degraded questions.

> Note: short-workload speedup (2.7×) is larger than a naive FFN-only estimate would
> suggest; the Q3_K GEMM also accelerates the `attn_gate` (24 SSM layers) and
> `attn_q`/`attn_k` (8 attn layers) projections, not just FFN gate/up.

## Not yet done

- Coopmat tensor-core variant of Q3_K (benchmark against this tiled f32 path first).
- Flipping the flag to default-on (all correctness + latency gates now pass; a
  matched vendor-parity benchmark and a decision on default-on remain).
