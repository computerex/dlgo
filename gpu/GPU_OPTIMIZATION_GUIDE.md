# GPU Inference Optimization Guide

## Architecture Overview

dlgo runs quantized LLM inference on Vulkan GPUs via Go + CGo + GLSL compute shaders.

**Key files:**
- `gpu/csrc/vulkan_gpu.c` — C dispatch logic, barriers, profiling, pipeline selection
- `gpu/csrc/vulkan_gpu.h` — `PipelineID` enum, C function declarations
- `gpu/shaders/compile.ps1` — Shader compilation script + `shader_registry[]` (hardcoded)
- `gpu/shaders/*.comp` — GLSL compute shaders (compiled to SPIR-V)
- `gpu/csrc/shaders_spirv.h` — Auto-generated: embedded SPIR-V + `shader_registry[]`
- `gpu/forward.go` — Go-side layer dispatch, dp4a type checks (`hasDp4aQType`)
- `gpu/gpu_cgo.go` — Go FFI wrappers (SetDP4A, MoEConf, LayerConf)
- `gpu/model.go` — Tensor upload to GPU
- `gpu/pipeline.go` — High-level generation loop, profiling env vars

## The #1 Optimization: dp4a Integer Dot Product

The single most impactful optimization is ensuring all quantization types use the **dp4a path** (`dotPacked4x8EXT` / `VK_KHR_shader_integer_dot_product`). This maps to hardware dp4a instructions that compute 4 int8 multiply-adds per clock — dramatically faster than float FMA for quantized weights.

**How dp4a works:**
1. Input vector is pre-quantized to Q8_1 format (one `gpu_quantize_q8_1` dispatch)
2. Weight nibbles are converted to int8 via lookup table
3. `dotPacked4x8EXT(packed_weight_int8, packed_input_int8)` does the dot product
4. Scale factors applied: `result = weight_scale * q8_1_scale * int_dot_product`

**Performance impact:** MXFP4 went from 155 µs → 41.6 µs per dispatch (73% faster), overall 61.9 → 118 tok/s.

## CRITICAL: The 7-Location Registration Checklist

When adding dp4a support for a new quantization type, you MUST update ALL of these locations. Missing even ONE causes silent fallback to the slow float path:

### 1. `gpu/shaders/matvec_<type>_dp4a.comp` — Create the shader
### 2. `gpu/shaders/matvec_<type>_dp4a_moe.comp` — Create MoE variant (if type used in MoE)
### 3. `gpu/shaders/compile.ps1` — Add to BOTH:
   - The `$shaders` array (compilation list)
   - The `shader_registry[]` array (must match enum order exactly)
### 4. `gpu/csrc/vulkan_gpu.h` — Add `PIPE_MATVEC_<TYPE>_DP4A` and `PIPE_MATVEC_<TYPE>_DP4A_MOE` to `PipelineID` enum (before `PIPE_COUNT`, order must match `shader_registry[]`)
### 5. `gpu/csrc/vulkan_gpu.c` — Add case to ALL of:
   - `qtype_has_dp4a()` — general dp4a support check
   - `gate_has_dp4a` / `down_has_dp4a` switch blocks in `gpu_moe_ffn()`
   - `dispatch_moe_matvec_dp4a()` — pipeline + rows_per_wg selection
   - `gpu_matvec_dp4a()` — non-MoE pipeline selection
### 6. `gpu/forward.go` — Add type ID to `hasDp4aQType()` switch
### 7. Compile shaders: `cd gpu/shaders && powershell -ExecutionPolicy Bypass -File compile.ps1`

## Profiling Methodology

### Step 1: Enable per-dispatch GPU timestamps
```powershell
$env:DLGO_GPU_PROFILE="1"
$env:DLGO_GPU_PROFILE_DETAIL="1"
```
This adds `vkCmdWriteTimestamp` around every dispatch and prints an aggregated table showing each pipeline's total time, count, avg, and percentage.

### Step 2: Identify the hottest kernel
Look at the `Pct` column. The top kernel is your optimization target. For MoE models, `matvec_*_moe` variants dominate (typically 60-80% of GPU time).

### Step 3: Check if dp4a is active
If you see `matvec_<type>_moe` instead of `matvec_<type>_dp4a_moe`, the dp4a path is NOT being used. Check all 7 registration points above.

### Step 4: Clean measurement
Always measure final performance with profiling OFF:
```powershell
$env:DLGO_GPU_PROFILE="0"
$env:DLGO_GPU_PROFILE_DETAIL="0"
```

### Step 5: Use the test harness
`profile_moe_main.go` — Temperature=0, Seed=42 for reproducible output. Compare text output between float and dp4a paths; slight differences are expected (Q8_1 rounding) but output should be coherent.

## Build Gotchas

### Stale CGo builds (CRITICAL)
`go run` does NOT always rebuild CGo components when only `.h` files change. The `compile.ps1` script touches `vulkan_gpu.c` to help, but after any shader or C code change, ALWAYS use:
```bash
go run -a -tags "cgo vulkan" <file>
```
The `-a` flag forces full rebuild. Without it, you may test stale shader code and draw wrong conclusions.

### Shader compilation takes ~3-4 minutes
The `compile.ps1` script compiles all ~90 shaders sequentially. Budget for this. It must complete fully (check for "Generated csrc/shaders_spirv.h" at the end).

### Pipeline enum ordering
The `PipelineID` enum in `vulkan_gpu.h` and the `shader_registry[]` in `compile.ps1` MUST be in exactly the same order. A mismatch silently loads the wrong shader for a pipeline ID.

## dp4a Shader Template (for new quant types)

For a new quant type `<TYPE>` with MoE support, follow the pattern from `matvec_mxfp4_dp4a_moe.comp`:

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_integer_dot_product : require
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 128) in;

struct block_q8_1 {
    float16_t d;      // scale = amax / 127
    float16_t s;      // sum = d * sum(qs)
    int32_t qs[8];    // 32 int8 values packed as 8 int32
};

// Bindings: 0=output, 1=weights, 2=Q8_1 input, 3=expert indices (MoE)
// Push constants: rows, cols, expert_stride, base_offset, shared_input

// Key pattern:
// - 128 threads = 4 subgroups, each handles 1 row (4 rows/WG)
// - b_qs_idx = sub_tid & 3 selects which 8 elements within a 32-element block
// - dotPacked4x8EXT(packed_weight_int8, data_b[b_ib].qs[b_qs_idx]) for low half
// - dotPacked4x8EXT(packed_weight_int8, data_b[b_ib].qs[b_qs_idx+4]) for high half
// - subgroupAdd for reduction
```

### Scale factor formulas by quant type:
- **Q4_0**: `da * (q_sum * dsb.x - 2.0 * dsb.y)` — zero-point 8 correction
- **MXFP4**: `da * q_sum * dsb.x` — signed kvalues, no bias needed
- **Q3_K/Q4_K/Q5_K/Q6_K**: Type-specific scale + min component handling
- Check existing dp4a shaders in `gpu/shaders/` for each type's formula

## Performance Targets and Bounds

### RTX 4070 Ti SUPER specs:
- Memory bandwidth: 672 GB/s theoretical, ~470-570 GB/s practical
- 66 SMs, 8448 CUDA cores
- L2 cache: 48 MB

### Per-model estimation:
```
weight_data_per_token = n_expert_used × expert_dim × blocks_per_row × bytes_per_block
theoretical_min_ms = weight_data_per_token / bandwidth
```
If observed time >> theoretical, check: dp4a enabled? Memory access coalescing? Occupancy?

## Quick Diagnosis Checklist

When a model is slow:
1. Run with `DLGO_GPU_PROFILE_DETAIL=1` — which kernel dominates?
2. Is it a `*_dp4a_moe` or just `*_moe`? If no dp4a → check 7-point registration
3. What's the WG size in the profile? Match expected `rows_per_wg`
4. Compare dispatch count with expected (`n_layers × dispatches_per_layer`)
5. Check `quantize_q8_1` appears in the profile (confirms dp4a path active)
6. Run `go run -a` to rule out stale builds

## Quant Type ID Reference

| ID | Type | Block size | Bytes/block | Has dp4a? |
|----|------|-----------|-------------|-----------|
| 2  | Q4_0 | 32 | 18 | Yes |
| 6  | Q5_0 | 32 | 22 | Yes |
| 8  | Q8_0 | 32 | 34 | Yes |
| 10 | Q3_K | 256 | 110 | Yes |
| 12 | Q4_K | 256 | 144 | Yes |
| 13 | Q5_K | 256 | 176 | Yes |
| 14 | Q6_K | 256 | 210 | Yes |
| 20 | IQ4_NL | 32 | 18 | No (TODO) |
| 39 | MXFP4 | 32 | 17 | Yes |
| 30 | BF16 | 1 | 2 | N/A |

## llama.cpp Reference

A local clone lives at `C:\projects\llama.cpp`. Their Vulkan shaders are the gold-standard reference:
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq.comp` — main dp4a matvec kernel
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl` — per-type `repack()` + `mmvq_dot_product()` + `mul_q8_1()`
- `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl` — block struct definitions, shared memory lookup tables
- `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl` — float dequant path (for comparison)
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — dispatch logic, workgroup size heuristics, barrier strategy

When implementing dp4a for a new quant type, read the corresponding `repack()` and `mul_q8_1()` in `mul_mat_vecq_funcs.glsl` first — it shows exactly how to extract int8 values from the block and what scale correction factor to use.

## Lessons Learned

1. **Always check llama.cpp first** — before profiling or micro-optimizing, see if llama.cpp uses a fundamentally different approach (like dp4a) for the quant type in question.
2. **dp4a >> shader micro-optimization** — Don't waste time on shared memory tricks, prefetching, or loop unrolling before ensuring dp4a is active. dp4a gave 73% speedup; all shader tweaks combined gave ~6%.
3. **Profiling overhead is real** — GPU timestamp queries add ~10-15% overhead. Always measure final perf with profiling off.
4. **Output text comparison is a correctness test** — Temperature=0, fixed seed. dp4a output will differ slightly from float (Q8_1 rounding) but must be coherent.
5. **The compile script is the source of truth** — `shader_registry[]` in `compile.ps1` defines what gets embedded. The generated `shaders_spirv.h` is just output.
6. **Silent fallback is the enemy** — The system never errors when dp4a isn't registered for a type. It silently uses the float path. The ONLY way to catch this is profiling output (check pipeline names for `_dp4a_`) or noticing `quantize_q8_1` is absent from the dispatch list.
