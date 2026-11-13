# gpu/cuda — dlgo CUDA backend

A CUDA backend for dlgo that is selectable at runtime alongside the Vulkan
backend. It talks to the CUDA **Driver API** in `nvcuda.dll` directly (no cudart
link at build time), loads an nvcc-compiled `kernels.cubin` embedded via
`go:embed`, and exposes the primitives the dense forward path needs.

The `gpu` package routes each primitive to CUDA when `DLGO_GPU_BACKEND=cuda`;
otherwise it uses Vulkan unchanged. Because the higher-level orchestration
(`gpu/forward.go`, `gpu/pipeline.go`, …) only calls these primitives, the *same
Go inference code runs on either backend* with no change.

## Enabling it

```powershell
$env:DLGO_GPU_BACKEND = "cuda"     # pick CUDA instead of Vulkan
$env:DLGO_NO_FUSED    = "1"        # route decode AND prefill through the Go primitive path
$env:DLGO_UNIFIED_QUANT = "0"      # keep native Q4_K/Q6_K (this backend's GEMV handles those)
dlgo run model.gguf -gpu
```

Without `DLGO_GPU_BACKEND=cuda` nothing changes — Vulkan stays the default
(`gpu.BackendName()` reports which is active).

> Why the two extra flags: the CUDA primitives implement the **native GGML**
> quant formats (Q4_K, Q6_K, F32, F16). The default pipeline uses the *fused*
> C dispatcher (Vulkan-only) and the *unified canonical* Affine4/Affine8 weights.
> `DLGO_NO_FUSED=1` + `DLGO_UNIFIED_QUANT=0` keep everything on the Go primitive
> path with native weights, which this backend serves. (Affine4/8 + fused batch
> + prefill GEMM are the next milestones — see Roadmap.)

## Architecture

- `cuda_shim.c` — thin C shim over the driver API. `LoadLibraryA("nvcuda.dll")`
  + `GetProcAddress`; prefers the `*_v2` entry points. Retains the device primary
  context and re-binds it (`cuCtxSetCurrent`) at the top of every entry, because
  Go goroutines migrate between OS threads.
- `cuda_shim.h` — C prototypes.
- `kernels.cu` → `kernels.cubin` — the kernels, built by `build_cuda.ps1`
  (`nvcc -cubin -arch=sm_89`, default RTX 4070 Ti SUPER).
- `cuda.go` — Go wrapper: `Init` loads the module and resolves kernel handles;
  `launch()` marshals args into one reusable C buffer (no per-launch malloc) and
  calls `cuLaunchKernel`.
- Stub (`cuda_stub.go`, `!cgo || !windows`) keeps non-Windows builds compiling.

## Kernels (validated)

| kernel | op | parity test |
|---|---|---|
| `dl_gemv_q4k` | `out[r]=Σ dequantQ4K(W[r,c])·x[c]` | `TestGEMVQ4KParity` vs `quant.DequantizeQ4_K` |
| `dl_gemv_q6k` | Q6_K GEMV | `TestGEMVQ6KParity` |
| `dl_gemv_f32` / `dl_gemv_f16` | dense GEMV | `TestGEMVF32Parity` |
| `dl_attention_f16` | single-query causal GQA over f16 KV, two-pass block-per-head (short ctx) | `TestAttentionF16Parity` |
| `dl_attention_split_f16` | split-KV + GQA-folded flash decode: grid=(kv_heads, splits), online softmax, softcap, window, ring, writes per-split (m,l,acc) | `TestAttentionF16Parity`, `TestAttentionSplitPartials`, `TestAttentionF16WindowParity` |
| `dl_attention_combine` | merges split partials into the attention output | ditto |
| `dl_kv_store_f16` | f32 K/V → packed half2 cache | bit-exact vs shader |
| `dl_rmsnorm_f32`, `dl_rmsnorm_heads`, `dl_layernorm_f32` | norms | |
| `dl_add_rmsnorm` | fused residual add + RMSNorm | |
| `dl_add`, `dl_scale`, `dl_silu_mul` | elementwise / SwiGLU | |
| `dl_rope` | interleaved / neox RoPE | |
| `dl_vector_add`, `dl_softmax_row`, probes | smoke/diag | |

`go test ./gpu/cuda/` — all green.

## Measured (Qwen3-4B Q4_K_M, RTX 4070 Ti SUPER, decode tok/s, greedy, 96 tok)
| backend | short ctx (~128 tok) | long ctx |
|---|---|---|
| Vulkan (default) | ~121 tok/s | **83.5 @ 2.2k, 122.9 @ 5k** tok/s |
| CUDA (this backend) | ~112 tok/s | **90.9 @ ~5k** tok/s (was 48.5) |

Sequence-length sweep of the decode attention kernel alone (32 heads / 8 KV
heads / head_dim 128, one call = one layer at that context). Current kernel is
v2 (`dl_attn_split_v2_d4`, warp-per-query-head); v1 (`dl_attention_split_f16`,
key-per-thread) is retained for head_dims other than 64/128:

| ctx | v1 | v2 + combine | GB/s (KV), v2 |
|---|---|---|---|
| 512 | 0.0181 ms | 0.0120 ms | 175 |
| 1024 | 0.0243 | 0.0157 | 268 |
| 2048 | 0.0253 | 0.0212 | 395 |
| 5000 | 0.0542 | 0.0367 | 558 |
| 12000 | 0.2135 | 0.1191 | 413 |

v2 exists because v1 scores one key per thread: a warp load touches 32 different
256-byte K rows and its P*V pass reads V two bytes at a time, which measured
230 GB/s (34% of peak) and was latency-bound, not bandwidth-bound. v2 puts one
warp on one (query head, key split) with the 32 lanes spread across head_dim, so
a single load covers a whole K/V row, and it consumes keys 16 at a time so 16 K
loads (pass 1) and then 16 V loads (pass 2) are in flight per warp.

Two hard-won constraints:

1. The split count dominates everything (the kernel is parallelism-limited):
   at 12000 ctx, 8 -> 16 -> 32 -> 64 splits gave 0.638 / 0.355 / 0.2155 /
   0.1191 ms. 128 is no better (0.1261) and hurts short contexts.
2. It MUST NOT be derived from the live key count. A captured CUDA graph bakes
   the grid into the replay, so an n-derived split count makes the graph replay
   whatever tiny grid was live at capture time -- capturing at n=1 and replaying
   at n=4114 cost 10x decode (111 -> 11 tok/s).

End-to-end decode (Qwen3-4B Q4_K_M, `cmd/llmbench`, 128 tokens), in-session A/B:
530 ctx 154.9 tok/s (was 154.1), 4114 ctx 123.8 (was 111.7), 16392 ctx 79.7
(was 59.8). The 16k needle is still answered correctly.

The v1-only numbers that this replaced, for reference:

| ctx | `dl_attention_split_f16` + combine | GB/s (KV) |
|---|---|---|
| 512 | 0.0237 ms | 88 |
| 1024 | 0.0221 | 190 |
| 2048 | 0.0392 | 214 |
| 5000 | 0.0857 | 239 |
| 12000 | 0.2001 | 246 |

That is **3–4× faster than the previous block-per-head kernel**
(0.147 ms @ 2048, ~0.36 ms @ 5000) and it is what took CUDA long-context decode
from 48.5 → 90.9 tok/s. Short-context numbers are launch-bound (2 kernels ≈
20 µs of a ~24 µs call), which is why 512 and 1024 look flat.

Important correction: earlier notes here said "Vulkan TDR-crashes at long context."
That was a **real pre-existing bug**, not inherent — the dense prefill path
(`GpuForwardPrefillBatch`) did not chunk: it sized batch buffers to
`prefillChunkSize` (≤1024 rows) but passed *all* `npos` tokens, so prompts over
1024 tokens did an out-of-bounds GPU write → device fault → Windows TDR ("device
lost"). Fixed (`startPos` param + chunking in both callers).

**Where the remaining CUDA deficit is (measured, not guessed).** At 5k ctx CUDA
spends ~11.0 ms/token: **~4.9 ms kernel dispatch**, ~4.4 ms GEMV (≈530 GB/s, at
peak), ~1.4 ms attention, ~0.3 ms the rest. `TestLaunchOverhead` measures
**10.12 µs per `cuLaunchKernel`** on this WDDM driver, and decode fires ~13
kernels × 36 layers ≈ 480 of them per token. Vulkan avoids this with its fused
C monolith (≈1 launch/layer). So CUDA's attention is now *faster* than Vulkan's,
and the entire remaining gap is dispatch overhead — which is exactly what the
first roadmap item below fixes.

Optimization log (measured; keep before "optimizing" again):

- **Won** — vectorized Q4_K/Q6_K GEMV: uint32 qs load/lane + coalesced `float4` x.
  Microbenchmark: Q4_K GEMV **544–593 GB/s**, Q6_K **534–612 GB/s** (near the
  ~672 GB/s peak) — so the matvec kernels are *not* the bottleneck anymore.
- **Won** — attention rewrite #1: two-pass block-per-head flash decode, 0.289 →
  0.039 ms/layer @ seq512 (7.5×) over the old warp-per-head kernel.
- **Won (big)** — attention rewrite #2: **split-KV + GQA fold**
  (`dl_attention_split_f16` + `dl_attention_combine`). One block serves all
  `num_heads/num_kv_heads` query heads that share a KV head, so each K/V row is
  read **once instead of once per query head**, and the key range is split across
  32 blocks so 256 blocks fill the GPU. A further **3–4×** at every long context;
  2048: 0.147 → 0.039 ms, 5000: ~0.36 → 0.086 ms. Parity kept on the sliding
  window and ring-buffer paths too (`TestAttentionF16WindowParity`).
- **Two real bugs found while landing it** (worth remembering):
  1. Threads that own no output pair (`kv_mul*head_dim < blockDim`) were still
     indexing `sfa[h]`/`sp[h*TILE]` with `h = tid/head_dim`, i.e. reading past
     the block's dynamic shared-memory window → silent context loss (the kernel
     "ran" and every later download returned zeros). Guard the pair loop.
  2. The V-row walker advanced `r` only inside `if (kv_cap > 0)`, so without a
     ring buffer every key of a split read the *same* V row. `TestAttentionSplitPartials`
     (all-zero K ⇒ all scores equal ⇒ `l` must equal the split's key count and
     `acc[d] == Σ V[j][d]`) pins both down now.
- **Lost** — int8 dp4a GEMV (block-coop 304 GB/s, warp 8-lane 81 tok/s): the
  Q8 requant + reductions cost more than the int8 ALU saved at 256-wide blocks.
  Kept behind `DLGO_CUDA_Q4K` / `DLGO_CUDA_DP4A`.
- **Lost** — 4-rows/warp register blocking (reg pressure); per-thread cuCtxSetCurrent
  cache (driver drops current ctx; must re-bind every call).
- **Not the problem (measured)** — the split count. Swept 4/8/16/32 at 512…12000:
  32 is best or tied everywhere, so it is now the fixed default
  (`DLGO_CUDA_ATTN_SPLITS` overrides for experiments).

## Roadmap to surpass llama.cpp / Strata

1. **CUDA-graph the layer loop** — now the *only* significant CUDA deficit:
   4.9 ms of 11.0 ms per token at 5k ctx (10.12 µs × 480 launches, measured) and
   4.9 ms of 8.9 ms at short ctx. Capture the 36-layer sequence once and replay
   per token. Needs (a) a non-default stream used by every launch/upload/sync —
   the legacy default stream cannot be captured, (b) `pos`/`seqLen`/`win_start`
   read from a small device int buffer (rope, kvstore, attention variants),
   (c) the per-token x upload kept outside the graph. Expected: 5k 11.0 → ~6.2 ms
   (~160 tok/s) and short ctx → ~200 tok/s, i.e. faster than Vulkan at both.
2. **Attention bandwidth** — the split kernel runs at 240/672 GB/s (~36%): the
   K/V rows are read thread-private (each thread walks one 256-byte row, relying
   on L1), which is latency- not throughput-bound. A warp-cooperative layout
   (32 lanes across one key's dims + shuffle reduce, per-warp accumulators)
   should reach ~500 GB/s → ~0.045 ms/layer @ 5000. Worth doing after graphs.
3. **cuBLAS/tensor-core f16 prefill GEMM** — per-token prefill is why a 5k-token
   prompt takes ~58 s; a batched GEMM prefill fixes TTFT.
4. **Affine4/Affine8 GEMV** so the default unified-quant path works on CUDA.
5. Extend to MoE / paged attention / diffusion kernels (currently Vulkan-only).

## Build / test

```powershell
powershell -File gpu/cuda/build_cuda.ps1          # rebuild kernels.cubin (needs VS2022 + CUDA)
go test ./gpu/cuda/                                # parity + smoke tests
```
