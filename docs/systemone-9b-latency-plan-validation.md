# Validation of `systemone-9b-latency-plan.md`

> **Execution audit correction (2026-10-09): this review overstated its conclusions.**
> The unverified 44/88 TFLOPS Tensor Core figures and physical-impossibility verdict below are
> withdrawn. 50 GB / 672 GB/s is 0.075 s, not 75 s. SSM's active batch path has four dispatches
> per chunk, not per token; `ssmFlushInterval` only affects fallback. SSM/attention block timings
> include projections and cannot establish recurrence/scoring overhead without sub-operator
> profiles. The 200-500 ns serial-step premise is assumed, not a derived hardware bound.
> Chunked causal attention still reads preceding context and retains quadratic total work.
> `answers_identical` in HTTP reports compares cache-cold vs cache-warm within a configuration,
> not coopmat vs K1; historical 21.7 s is a candidate to reproduce, not full-corpus correctness
> proof. GGUF type 12 is Q4_K and type 13 is Q5_K (the earlier mapping was reversed).
> The plan has been revised to use measurement gates. Retain the original review below as history,
> not as an authoritative roofline analysis.

Independent re-derivation of the plan's roofline arithmetic, cross-checked against the measured
artifacts on disk and against the current source tree. Date of review: after the plan was written.
Purpose: state which claims are solid, which are assumptions, and which must be corrected before
the plan drives implementation.

Sources used (all read, not recalled):
- `gpu/csrc/vulkan_gpu.c`, `gpu/csrc/vulkan_gpu.h`, `gpu/csrc/shaders_spirv.h`
- `gpu/shaders/compile.ps1`
- `bench/systemone/latency-9b-uni-long-profile.json` (plain K1)
- `bench/systemone/latency-9b-uni-long-coopmat.json` (coopmat config)
- `bench/systemone/9b-long-latency.json` (prior v1-gpu head)
- `bench/systemone/RESULTS-k1-q3k-gemm-2026-10-08.md`
- `bench/systemone/serve-9b-q3k.ps1`, `bench/systemone/profile-9b-uni-server.ps1`
- `gguf_info.exe` on the real `Qwen3.5-9B-Q3_K_M.gguf`

---

## 1. What checks out exactly

### 1.1 FFN FLOP arithmetic — CORRECT
`2 x 4096 x 12288 x 3 x 32 = 9.664 GFLOP/token`; `x 10,816 tokens = 104.5 TFLOP`. ✅
Verified against `gguf_info`: `ffn_gate`/`ffn_up` are **4096x12288, type=11 (Q3_K)**, all **32**
layers; `ffn_down` is **12288x4096, type=12/13 (Q5_K/Q4_K)** per layer. The three projections are
M*N*K balanced for M=1, so `2*M*N*K` per projection is the correct MAC->FLOP convention. The
numerator is genuine token-proportional work, not an artifact.

### 1.2 Achieved-TFLOPS reconciliation — ARITHMETIC CORRECT
- coopmat: `2*4096*12288*512 = 51.5 GFLOP`; `51.5e9 / 4.94e-3 = 10.42 TFLOPS` ✅
- tiled K1: `51.5e9 / 7.145e-3 = 7.21 TFLOPS` ✅ (matches `RESULTS-k1-q3k-gemm` table)
- `104 TFLOP / 10.4 TFLOPS ~= 10 s` vs measured FFN block **11.275 s** -> order-consistent.

### 1.3 Aggregate efficiency — CORRECT
`~108 TFLOP / 24.6 s = 4.4 TFLOPS` ✅. Every block runs far below device capability.

### 1.4 SSM is overhead-dominated — SOLID
24 layers x 10,816 tokens = 260K sequential steps. Serial floor (200-500 ns/step) = 0.05-0.13 s vs
measured **5.71 s** -> ~22 us/step, ~100-400x above floor. The structural argument (barriers,
dispatch, staging per token, not arithmetic) is sound and is the single most defensible conclusion
in the plan.

### 1.5 Code-state / cleanup claims — ALL VERIFIED ACCURATE
I confirmed every item in plan §8 step 1 against the tree:
- `vulkan_gpu.h:194` still has `PIPE_GEMM_Q3K_COOPMAT` (shader deleted -> build break).
- `gpu_gemm_q3k_coopmat()` (`vulkan_gpu.c:3520`), the `gemm_q3k_coopmat_available()` gate
  (`:3462`), `gpu_set_gemm_q3k_coopmat()` (`:3478`), and the dispatch branch (`:2676-2679`) all
  still reference the deleted pipeline. **Confirmed build break.**
- `gpu_cgo.go:947` `SetGemmQ3KCoopmat()` calls the still-present C symbol -> compiles, but the
  pipeline it pointlessly toggles no longer exists.
- `compile.ps1:322` still registers `gemm_q3k_coopmat` though `gemm_q3k_coopmat.spv/.comp` are
  gone from disk -> `compile.ps1` will fail.
- `gpu/shaders/ssm_recurrence_batch_t32.{comp,spv}` **exist on disk**, `compile.ps1:323` registers
  them, but **`shaders_spirv.h` contains no `ssm_recurrence_batch_t32` symbol** -> the T32 pipeline
  is not yet embeddable; header regeneration is genuinely required (plan §8 step 2).
- Q4_K coopmat infrastructure for the plan §4 lever **exists and is wired**:
  `gpu_gemm_q4k_coopmat`, `gpu_gemm_q4k`, `gpu_gemm_q4k_dequant_coopmat` (`vulkan_gpu.c:2627-2635`);
  `ffn_down` (Q4_K) already routes through `gpu_gemm_q4k` at `:2655-2656`. So §4 is directly
  actionable.

---

## 2. Claims that are defended-but-weak (must be reworded)

### 2.1 "RTX 4070 Ti SUPER FP16 tensor peak ~44 TFLOPS dense" — UNDER-JUSTIFIED
The plan's headline conclusion ("sub-3s is physically impossible") rests entirely on this number.
Problems:
- 44 TFLOPS is the **FP16 with FP32-accumulate dense** figure. The FP16->FP16 dense rate on Ada is
  ~2x higher (~88 TFLOPS), and INT8/FP8 higher still. The plan does not state which figure 44 is,
  so a GPU-literate reviewer will challenge "44 is the peak."
- If the true peak is ~88, the *theoretical* FFN floor is ~1.18 s and sub-3s becomes *theoretically*
  reachable (1.18 + attention/SSM floors ~ 1.2 s).
- **The conclusion still holds**, but for a better-stated reason: a Q3_K dequant GEMM can never use
  FP16->FP16 tensor throughput — dequant ALU is the binding cost — so realizable throughput is
  4-10x below any peak, and the achieved 7-10 TFLOPS is what actually matters. Restate as:
  "even the theoretical floor is marginal, and the realizable (dequant-limited) throughput makes
  3 s unreachable regardless of which peak you quote." Keep the concrete 2.36 s floor but label it
  a *pessimistic bound*, and add the FP16-accum alternative.

### 2.2 Attention "~4 TFLOP" — SHAKY (conclusion survives)
At seq ~= 10,816, the score/PV terms alone are ~`2 (QK^T + PV) x 2 (MAC) x seq^2 x (head_dim x
num_heads)`. If `head_dim x num_heads ~= 4096`, that is ~1.9 TFLOP **per attention layer**, x8
layers ~ **15 TFLOP**, not 4 — a ~3-4x understatement, *unless* attention is chunked/flash-style
(chunk=512), which collapses the score term. So the figure is algorithm-dependent and under-cited.
- Does not change the verdict: even at 15 TFLOP the floor is ~0.35 s vs measured 7.48 s ->
  still overhead-dominated. But the plan should not state "~4 TFLOP" as if measured; it should flag
  it as an estimate and get the real number from the W3a breakdown before sizing W3.

### 2.3 "104/10.4 ~= 10 s, matching 11.275 s" — ROUGHLY right, not tight
The comparison is order-consistent but ~13% low. Two reasons the estimate is optimistic: (a) the
10.4 TFLOPS was measured on gate/up only at npos=512, while the FFN block also includes `ffn_down`
at Q4_K coopmat (a *higher* TFLOPS/FLOP path -> would make the estimate read *higher*, not lower);
(b) yet measured is 11.28 s, more than the estimate, so a chunk of time is dequant-scale overhead
and imperfect overlap. Reword `~=` to "order-consistent; ~13% residual unexplained," and let W4
explain the residual rather than implying the model fully closes.

### 2.4 T32 "16.74x" must not set expectations
The isolated speedup was a **warmup artifact**; warm t16 vs t32 is ~1.00x at the 16-head test
shape. The plan already defers to E2E, but should state explicitly that the 16.74x is **not**
predictive of E2E.

---

## 3. The material gap: the coopmat path is NOT a proven dead end

**Plan §0 and W4a say** the cooperative-matrix Q3_K kernel "produced zero output (relL2 = 1.0) at
both vulkan1.2 and vulkan1.3 and was removed rather than shipped," presenting it as a failed path.

**File evidence contradicts the framing as an unqualified failure:**
- `bench/systemone/latency-9b-uni-long-coopmat.json` records cold p50 **21,729 ms** / warm **722 ms**
  at 10,809 tokens, with **`answers_identical=true`** and **sane, correct-looking answers**
  (move_type=electric 0.9997; switch=pikachu; the `noul` confidences are in the normal range).
  The plain K1 baseline (`latency-9b-uni-long-profile.json`) is cold p50 **25,010 ms** / warm 838 ms.
- `bench/systemone/serve-9b-q3k.ps1` **on disk today** still sets
  `$env:DLGO_GEMM_Q3K_COOPMAT = '1'` (line 4), while the plain `profile-9b-uni-server.ps1` does not.
- The validation GPU log shows the coopmat device/property enumeration succeeding.

Interpretation: a **correct, fast cooperative-matrix Q3_K revision existed** (consistent with the
first validation that passed at relL2 2.6e-4 and 1.47-1.58x), it delivered a **~3.3-3.6 s E2E win
(25.3 s -> 21.7 s)** with identical answers, and a *later* edit (v2, BK=64) regressed it to 0.98x and
a subsequent edit broke it to zero output — after which the whole shader was deleted. The deleted
shader is **untracked in git** and is not recoverable from git; but the existence proof is strong.

Consequence for the plan:
- W4a should be **promoted from "root-cause later" to a first-class recovery item**, because a
  correct revision is evidenced to give a **~13% E2E cold win** for near-zero risk (answers
  identical). That is a larger, safer win than most of W2/W3 and comparable to plan §2's
  attention/SSM targets as *measured reality* rather than projection.
- §0's framing of physics should be softened: the "wall" is not "coopmat failed," it is "we have a
  working ~1.45x coopmat revision we haven't reshipped." Do not delete history that shows a win.

---

## 4. Which numbers the plan should re-derive before implementation

1. **Peak TFLOPS**: state what 44 is (FP16 FP32-accum dense); note FP16-accum ~2x. Reframe the
   impossibility argument on realizable dequant-limited throughput, not the peak.
2. **Attention FLOPs**: obtain from the W3a breakdown; do not size W3 on the "~4 TFLOP" estimate.
3. **Pinned baseline**: cite `latency-9b-uni-long-profile.json` (v2-uni, 25.01 s cold p50) as THE
   baseline; note the older v1-gpu head was 88 s (`9b-long-latency.json`) to prevent mix-ups.
4. **Coopmat recovery delta**: treat 21.7 s (`latency-9b-uni-long-coopmat.json`) as the target to
   re-attain, then re-measure from a clean, validated rebuild.

---

## 5. Verdict

The plan is **directionally correct and unusually disciplined** (measure-before-change, gates tied
to the 693-row corpus, explicit out-of-scope, per-item shippability). Its central practical
conclusion — *the realizable wins are in SSM/attention overhead and FFN efficiency, not in an
algorithm swap, and 3 s at 10.8K tokens is not attainable on this GPU* — is **sound**, but the
"physics" framing overstates certainty by leaning on an unqualified 44 TFLOPS peak.

Two changes are required before the plan drives work:
1. **Reword §0/§2**: label 44 TFLOPS as FP16/FP32-accum dense, add the FP16-accum alternative, and
   rest the impossibility claim on realizable dequant-limited throughput (7-10 TFLOPS), which is the
   actual measured reality.
2. **Promote coopmat recovery (W4a) to a high-priority item** and record that a correct ~1.45x
   coopsmat revision previously yielded a ~3.6 s E2E cold win with identical answers, per
   `latency-9b-uni-long-coopmat.json`; do not present it as a failed dead end.

Everything else in §1 (cleanup list, gates, workload scoping) is verified accurate and can proceed
as written. The cleanup in §8 step 1 is confirmed necessary: the tree does not currently build.

### Confidence
- FFN FLOP model, aggregate efficiency, SSM overhead thesis, cleanup list, T32/§4 viability: **high**.
- Attention FLOP figure, the 44 TFLOPS peak as "the peak": **low — restate**.
- Coopmat "failure": **contradicted by file evidence — correct**.
