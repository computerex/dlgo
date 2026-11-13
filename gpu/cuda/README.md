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

## Prefill attention: what actually made it faster (and what did not)

`dl_attn_prefill_f16` was rewritten from "4 warps share a 32-key K/V tile staged
in shared memory" to "one warp per 16-query tile, mma fragments loaded straight
from the cache, no `__syncthreads()` in the key loop, each warp running to its
own causal end". Measured on `TestPrefillAttnThroughput`
(`go test ./gpu/cuda/ -run TestPrefillAttnThroughput -v`):

| version | 4k | 16k |
|---|---|---|
| staged K/V, block-wide tiles | 11.2 TFLOP/s | 12.5 TFLOP/s |
| no staging, per-warp causal end | 11.9 | 13.1 |
| ...and no per-iteration `__syncwarp()` | 13.7 | 15.0 |
| ...and fully unrolled d-loops (loads pipelined) | **16.3** | **16.6** |

Two lessons, because the obvious diagnosis was wrong:

1. **Occupancy was not the limit.** The no-staging rewrite cut static shared
   memory from 44 KB to 28 KB (2 -> 3 blocks/SM), removed two block-wide
   barriers per 32-key tile and let each warp stop at its own causal end. All of
   that bought 5%. The staged version was already reading K/V once per block and
   therefore had *better* L2 reuse than the per-warp version, which reads each
   row once per warp.
2. **Memory-level parallelism was the limit.** The win came from making the
   d-loops fully unrollable (`for (d=0; d<DLATT_HD; d+=16) { if (d>=headDim) break; ... }`
   with `#pragma unroll`, so headDim stays a runtime bound while the compiler
   still sees 8 static iterations) and from deleting the redundant `__syncwarp()`
   that had been sitting *inside* them. The compiler could then hoist several K/V
   fragment loads above the mma chain instead of each mma waiting ~500 cycles for
   its own load: 13.1 -> 16.6 TFLOP/s.

End-to-end prefill: 16k 9519 -> 8601 ms (1722 -> 1906 tok/s), 4k 1350 -> 1322 ms,
512 unchanged (it is GEMM-bound). llama.cpp is 2322 ms, so the gap is 3.7x.

Still open, in order: attention is at 16.6 TFLOP/s where llama is at ~55 (the
remaining structural gap needs higher arithmetic intensity per KV byte -- bigger
query tiles, or applying the softmax to the mma accumulator fragments instead of
round-tripping scores through shared memory -- and/or int8 tensor cores), then
int8/int4 MMA for the GEMM itself.


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

## Prefill: where the time actually goes (measured, not inferred)

`DLGO_CUDA_SKIP_OPS=<op-class>` elides one op class from the batched prefill, so the drop in total prefill time is that class's exact cost. Use it instead of `Sync()`-bracketing: a `cuStreamSynchronize` on this WDDM box costs ~800 us, which dwarfs every op and once inflated a 164 ms prefill into 472 ms of profile.

At 16392 tokens (7739 ms) on Qwen3-4B-Q4_K_M: prefill attention 4781 ms (62%, 17 TFLOP/s), the five GEMMs 2204 ms (28%, 54 TFLOP/s), everything else 755 ms. At 530 tokens (127 ms): GEMM 107 ms (84%, 36 TFLOP/s), attention 11 ms, other 9 ms.

So the two bottlenecks differ, and neither is 'flash attention at long context' alone:

* **Attention is DRAM-bound on K/V re-reads.** One warp owns 16 queries and reads the whole K/V range for them, so a 64-query block re-reads K/V once per tile: 69 GB at 16k = 519 GB/s = 77% of peak. The 17 TFLOP/s is the bandwidth wall. Raising it needs 128+ queries per K/V load; doing so by doubling the query tile and halving the key tile to fit the 48 KB static shared limit made it worse (17.0 -> 10.9 TFLOP/s), so it needs dynamic shared memory or a flash-decoding split over keys.
* **The GEMM is limited by block count at small batch, not by tiling.** The 128x128 tile reaches 47 TFLOP/s at 512 tokens and 60 at 16k in isolation, but `down` at 512 is only 80 blocks for 66 SMs (~1.2 waves), so the second wave is 21% full. Split-K is the fix, not a bigger tile.
* **llama.cpp sustains ~83 TFLOP/s of the same work** at pp512, above what a dequantise-to-f16 GEMM reaches here, so the remaining gap is the int8 MMA path (MMQ). dlgo's Affine4 block is already one 32-weight group with f16 scale `d` and offset `m`, i.e. weights are `d*q + m`; the int8 route needs the `m` term as `coef = 8d + m` times the group sum of X, a secondary K=cols/32 GEMM worth about 3% of the main one.

Harnesses: `gpu/gemm_probe_test.go` (the five Qwen3-4B GEMM shapes) and `gpu/attn_probe_test.go` (one layer of prefill attention, whole-prompt versus the chunked sequence the pipeline issues). `DLGO_DEBUG_ATTN=1` logs the launch geometry.
