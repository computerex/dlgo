param([string[]]$Only = @())

# -Only accepts either repeated flags or a single comma-separated list:
#   compile.ps1 -Only attention_decode_split,attention_decode_merge
$onlyNames = @()
foreach ($o in $Only) {
    foreach ($part in ($o -split ',')) {
        $t = $part.Trim()
        if ($t) { $onlyNames += $t }
    }
}

$shaders = @(
    @{name="matvec_f32"; file="matvec_f32.comp"},
    @{name="matvec_q4_0"; file="matvec_q4_0.comp"},
    @{name="matvec_q8_0"; file="matvec_q8_0.comp"},
    @{name="matvec_q3_k"; file="matvec_q3_k.comp"},
    @{name="matvec_q4_k"; file="matvec_q4_k.comp"},
    @{name="matvec_q5_k"; file="matvec_q5_k.comp"},
    @{name="matvec_q5_0"; file="matvec_q5_0.comp"},
    @{name="matvec_q6_k"; file="matvec_q6_k.comp"},
    @{name="attention"; file="attention.comp"},
    @{name="rmsnorm"; file="rmsnorm.comp"},
    @{name="rmsnorm_heads"; file="rmsnorm_heads.comp"},
    @{name="softmax"; file="softmax.comp"},
    @{name="rope"; file="rope.comp"},
    @{name="swiglu"; file="swiglu.comp"},
    @{name="geglu"; file="geglu.comp"},
    @{name="gelu"; file="gelu.comp"},
    @{name="add"; file="add.comp"},
    @{name="scale"; file="scale.comp"},
    @{name="add_rmsnorm"; file="add_rmsnorm.comp"},
    @{name="quantize_q8_1"; file="quantize_q8_1.comp"},
    @{name="matvec_q4_0_dp4a"; file="matvec_q4_0_dp4a.comp"},
    @{name="matvec_q5_0_dp4a"; file="matvec_q5_0_dp4a.comp"},
    @{name="matvec_q8_0_dp4a"; file="matvec_q8_0_dp4a.comp"},
    @{name="matvec_q4_k_dp4a"; file="matvec_q4_k_dp4a.comp"},
    @{name="matvec_q6_k_dp4a"; file="matvec_q6_k_dp4a.comp"},
    @{name="matvec_q3_k_dp4a"; file="matvec_q3_k_dp4a.comp"},
    @{name="matvec_q5_k_dp4a"; file="matvec_q5_k_dp4a.comp"},
    @{name="ssm_conv1d_silu"; file="ssm_conv1d_silu.comp"},
    @{name="ssm_preprocess"; file="ssm_preprocess.comp"},
    @{name="ssm_delta_rule"; file="ssm_delta_rule.comp"},
    @{name="ssm_conv1d_batch"; file="ssm_conv1d_batch.comp"},
    @{name="ssm_preprocess_batch"; file="ssm_preprocess_batch.comp"},
    @{name="ssm_recurrence_batch"; file="ssm_recurrence_batch.comp"},
    @{name="ssm_norm_gate_batch"; file="ssm_norm_gate_batch.comp"},
    @{name="ssm_decay"; file="ssm_decay.comp"},
    @{name="ssm_predict"; file="ssm_predict.comp"},
    @{name="ssm_update"; file="ssm_update.comp"},
    @{name="ssm_output"; file="ssm_output.comp"},
    @{name="ssm_norm_gate"; file="ssm_norm_gate.comp"},
    @{name="deinterleave_qgate"; file="deinterleave_qgate.comp"},
    @{name="sigmoid_gate"; file="sigmoid_gate.comp"},
    @{name="paged_attention"; file="paged_attention.comp"},
    @{name="matvec_q2_k"; file="matvec_q2_k.comp"},
    @{name="matvec_iq1_s"; file="matvec_iq1_s.comp"},
    @{name="matvec_iq1_m"; file="matvec_iq1_m.comp"},
    @{name="matvec_tq1_0"; file="matvec_tq1_0.comp"},
    @{name="matvec_iq2_xxs"; file="matvec_iq2_xxs.comp"},
    @{name="matvec_iq2_s"; file="matvec_iq2_s.comp"},
    @{name="matvec_iq3_xxs"; file="matvec_iq3_xxs.comp"},
    @{name="matvec_iq3_s"; file="matvec_iq3_s.comp"},
    @{name="matvec_iq4_xs"; file="matvec_iq4_xs.comp"},
    @{name="matvec_iq4_nl"; file="matvec_iq4_nl.comp"},
    @{name="matvec_mxfp4"; file="matvec_mxfp4.comp"},
    @{name="attention_sinks"; file="attention_sinks.comp"},
    @{name="swiglu_oai"; file="swiglu_oai.comp"},
    @{name="add_offset"; file="add_offset.comp"},
    @{name="scale_add"; file="scale_add.comp"},
    @{name="swiglu_oai_bias"; file="swiglu_oai_bias.comp"},
    @{name="moe_topk"; file="moe_topk.comp"},
    @{name="matvec_mxfp4_dp4a"; file="matvec_mxfp4_dp4a.comp"},
    @{name="matvec_q4_0_dp4a_moe"; file="matvec_q4_0_dp4a_moe.comp"},
    @{name="matvec_q5_0_dp4a_moe"; file="matvec_q5_0_dp4a_moe.comp"},
    @{name="matvec_q8_0_dp4a_moe"; file="matvec_q8_0_dp4a_moe.comp"},
    @{name="matvec_q4_k_dp4a_moe"; file="matvec_q4_k_dp4a_moe.comp"},
    @{name="matvec_q6_k_dp4a_moe"; file="matvec_q6_k_dp4a_moe.comp"},
    @{name="matvec_q3_k_dp4a_moe"; file="matvec_q3_k_dp4a_moe.comp"},
    @{name="matvec_q5_k_dp4a_moe"; file="matvec_q5_k_dp4a_moe.comp"},
    @{name="matvec_mxfp4_dp4a_moe"; file="matvec_mxfp4_dp4a_moe.comp"},
    @{name="matvec_iq3_xxs_moe"; file="matvec_iq3_xxs_moe.comp"},
    @{name="moe_accumulate"; file="moe_accumulate.comp"},
    @{name="swiglu_oai_bias_moe"; file="swiglu_oai_bias_moe.comp"},
    @{name="moe_bias_add"; file="moe_bias_add.comp"},
    @{name="attention_tiled"; file="attention_tiled.comp"},
    @{name="kv_store_f16"; file="kv_store_f16.comp"},
    @{name="kv_store_batch_f16"; file="kv_store_batch_f16.comp"},
    @{name="attention_tiled_f32"; file="attention_tiled_f32.comp"},
    @{name="layernorm"; file="layernorm.comp"},
    @{name="matvec_bf16"; file="matvec_bf16.comp"},
    @{name="broadcast_mul"; file="broadcast_mul.comp"},
    @{name="tanh_gate_residual"; file="tanh_gate_residual.comp"},
    @{name="rope_3d"; file="rope_3d.comp"},
    @{name="attention_full_f32"; file="attention_full_f32.comp"},
    @{name="attention_full_f16"; file="attention_full_f16.comp"},
    @{name="flash_attn_f32"; file="flash_attn_f32.comp"},
    @{name="conv2d_f32"; file="conv2d_f32.comp"},
    @{name="group_norm"; file="group_norm.comp"},
    @{name="silu"; file="silu.comp"},
    @{name="upsample_nearest"; file="upsample_nearest.comp"},
    @{name="spatial_attention"; file="spatial_attention.comp"},
    @{name="gemm_q4_k"; file="gemm_q4_k.comp"},
    @{name="split_qkv"; file="split_qkv.comp"},
    @{name="dequant_q4k_f16"; file="dequant_q4k_f16.comp"},
    @{name="gemm_f16"; file="gemm_f16.comp"},
    @{name="gemm_q4k_dp4a"; file="gemm_q4k_dp4a.comp"},
    @{name="gemm_q4k_coopmat"; file="gemm_q4k_coopmat.comp"; target="vulkan1.3"},
    @{name="gemm_q5k_coopmat"; file="gemm_q5k_coopmat.comp"; target="vulkan1.3"},
    @{name="gemm_q6k_coopmat"; file="gemm_q6k_coopmat.comp"; target="vulkan1.3"},
    @{name="gemm_q6k_f32"; file="gemm_q6k_f32.comp"; target="vulkan1.3"},
    @{name="gemm_bf16"; file="gemm_bf16.comp"; target="vulkan1.3"},
    @{name="gemm_f16_coopmat"; file="gemm_f16_f32.comp"; target="vulkan1.3"},
    @{name="matvec_f32_moe"; file="matvec_f32_moe.comp"},
    @{name="matvec_q4_0_moe"; file="matvec_q4_0_moe.comp"},
    @{name="matvec_q8_0_moe"; file="matvec_q8_0_moe.comp"},
    @{name="matvec_q3_k_moe"; file="matvec_q3_k_moe.comp"},
    @{name="matvec_q4_k_moe"; file="matvec_q4_k_moe.comp"},
    @{name="matvec_q5_k_moe"; file="matvec_q5_k_moe.comp"},
    @{name="matvec_q5_0_moe"; file="matvec_q5_0_moe.comp"},
    @{name="matvec_q6_k_moe"; file="matvec_q6_k_moe.comp"},
    @{name="matvec_q2_k_moe"; file="matvec_q2_k_moe.comp"},
    @{name="matvec_tq1_0_moe"; file="matvec_tq1_0_moe.comp"},
    @{name="matvec_iq1_s_moe"; file="matvec_iq1_s_moe.comp"},
    @{name="matvec_iq1_m_moe"; file="matvec_iq1_m_moe.comp"},
    @{name="matvec_iq2_xxs_moe"; file="matvec_iq2_xxs_moe.comp"},
    @{name="matvec_iq2_s_moe"; file="matvec_iq2_s_moe.comp"},
    @{name="matvec_iq3_s_moe"; file="matvec_iq3_s_moe.comp"},
    @{name="matvec_iq4_xs_moe"; file="matvec_iq4_xs_moe.comp"},
    @{name="matvec_iq4_nl_moe"; file="matvec_iq4_nl_moe.comp"},
    @{name="matvec_mxfp4_moe"; file="matvec_mxfp4_moe.comp"},
    @{name="shared_expert_gate"; file="shared_expert_gate.comp"},
    @{name="matvec_q4_k_dp4a_fused"; file="matvec_q4_k_dp4a_fused.comp"},
    @{name="matvec_iq3_xxs_dp4a"; file="matvec_iq3_xxs_dp4a.comp"},
    @{name="matvec_iq4_xs_dp4a"; file="matvec_iq4_xs_dp4a.comp"},
    @{name="matvec_iq4_nl_dp4a"; file="matvec_iq4_nl_dp4a.comp"},
    @{name="matvec_iq3_s_dp4a"; file="matvec_iq3_s_dp4a.comp"},
    @{name="matvec_iq3_xxs_dp4a_moe"; file="matvec_iq3_xxs_dp4a_moe.comp"},
    @{name="matvec_iq4_xs_dp4a_moe"; file="matvec_iq4_xs_dp4a_moe.comp"},
    @{name="matvec_iq4_nl_dp4a_moe"; file="matvec_iq4_nl_dp4a_moe.comp"},
    @{name="matvec_iq3_s_dp4a_moe"; file="matvec_iq3_s_dp4a_moe.comp"},
    @{name="attention_tiled_sinks"; file="attention_tiled_sinks.comp"},
    @{name="matvec_q5_1"; file="matvec_q5_1.comp"},
    @{name="gemm_q3k_tiled"; file="gemm_q3k_tiled.comp"},
    @{name="ssm_recurrence_batch_t32"; file="ssm_recurrence_batch_t32.comp"},
    @{name="attention_tiled_barrier"; file="attention_tiled_barrier.comp"},
    @{name="attention_tiled_kv4"; file="attention_tiled_kv4.comp"},
    @{name="gemm_q3k_coopmat_recovered"; file="gemm_q3k_coopmat_recovered.comp"; target="vulkan1.3"},
    @{name="gemm_affine8_coopmat"; file="gemm_affine8_coopmat.comp"; target="vulkan1.3"},
    @{name="attention_flash_f16_coopmat"; file="attention_flash_f16_coopmat.comp"; target="vulkan1.3"},
    @{name="matvec_affine8"; file="matvec_affine8.comp"},
    @{name="matvec_affine4"; file="matvec_affine4.comp"},
    @{name="gemm_affine4_coopmat"; file="gemm_affine4_coopmat.comp"; target="vulkan1.3"},
    @{name="matvec_affine4_moe"; file="matvec_affine4_moe.comp"},
    @{name="matvec_affine8_moe"; file="matvec_affine8_moe.comp"},
    @{name="attention_decode_split"; file="attention_decode_split.comp"},
    @{name="attention_decode_gqa"; file="attention_decode_gqa.comp"},
    @{name="attention_decode_merge"; file="attention_decode_merge.comp"},
    @{name="convert_to_affine"; file="convert_to_affine.comp"},
    @{name="shortconv"; file="shortconv.comp"},
    @{name="gather_rows"; file="gather_rows.comp"},
    @{name="scatter_rows"; file="scatter_rows.comp"},
    @{name="gemm_affine4_moe_grouped"; file="gemm_affine4_moe_grouped.comp"; target="vulkan1.3"},
    @{name="gemm_affine8_moe_grouped"; file="gemm_affine8_moe_grouped.comp"; target="vulkan1.3"},
    @{name="moe_slot_reduce"; file="moe_slot_reduce.comp"},
    @{name="moe_expert_bias_add"; file="moe_expert_bias_add.comp"}
)

$header = @"
// Auto-generated SPIR-V shader data. Do not edit.
#ifndef DLGO_SHADERS_SPIRV_H
#define DLGO_SHADERS_SPIRV_H

#include <stdint.h>
#include <stddef.h>

"@

foreach ($s in $shaders) {
    $spvFile = Join-Path $PSScriptRoot "$($s.name).spv"
    $env = if ($s.target) { $s.target } else { "vulkan1.2" }
    # -Only <names> skips glslc for unchanged shaders but still regenerates the
    # header from their existing .spv, so a single-shader edit costs seconds
    # instead of recompiling all of them.
    $recompile = ($onlyNames.Count -eq 0) -or ($onlyNames -contains $s.name)
    if (-not (Test-Path $spvFile)) { $recompile = $true }
    if ($recompile) {
        Write-Host "Compiling $($s.file) -> $spvFile ($env)"
        & glslc --target-env=$env -O (Join-Path $PSScriptRoot $s.file) -o $spvFile
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to compile $($s.file)"
            exit 1
        }
    }

    $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $spvFile))
    $header += "static const uint32_t spv_$($s.name)[] = {`n"

    for ($i = 0; $i -lt $bytes.Length; $i += 4) {
        $val = [BitConverter]::ToUInt32($bytes, $i)
        $header += "    0x$($val.ToString('x8')),"
        if (($i / 4 + 1) % 8 -eq 0) { $header += "`n" } else { $header += " " }
    }
    $header += "`n};`n"
    $header += "static const size_t spv_$($s.name)_size = sizeof(spv_$($s.name));`n`n"
}

$header += @"

typedef struct {
    const char* name;
    const uint32_t* code;
    size_t code_size;
    int num_buffers;
    int push_const_size;
} ShaderInfo;

static const ShaderInfo shader_registry[] = {
    {"matvec_f32",  spv_matvec_f32,  spv_matvec_f32_size,  3, 8},   // PIPE_MATVEC_F32
    {"matvec_f32",  spv_matvec_f32,  spv_matvec_f32_size,  3, 8},   // PIPE_MATVEC_F16 (placeholder)
    {"matvec_q4_0", spv_matvec_q4_0, spv_matvec_q4_0_size, 3, 8},   // PIPE_MATVEC_Q4_0
    {"matvec_q8_0", spv_matvec_q8_0, spv_matvec_q8_0_size, 3, 8},   // PIPE_MATVEC_Q8_0
    {"matvec_q3_k", spv_matvec_q3_k, spv_matvec_q3_k_size, 3, 8},   // PIPE_MATVEC_Q3_K
    {"matvec_q4_k", spv_matvec_q4_k, spv_matvec_q4_k_size, 3, 8},   // PIPE_MATVEC_Q4_K
    {"matvec_q5_k", spv_matvec_q5_k, spv_matvec_q5_k_size, 3, 8},   // PIPE_MATVEC_Q5_K
    {"matvec_q5_0", spv_matvec_q5_0, spv_matvec_q5_0_size, 3, 8},   // PIPE_MATVEC_Q5_0
    {"matvec_q6_k", spv_matvec_q6_k, spv_matvec_q6_k_size, 3, 8},   // PIPE_MATVEC_Q6_K
    {"matvec_q4_0", spv_matvec_q4_0, spv_matvec_q4_0_size, 3, 8},   // PIPE_DEQUANT_Q4_0 (placeholder)
    {"matvec_q8_0", spv_matvec_q8_0, spv_matvec_q8_0_size, 3, 8},   // PIPE_DEQUANT_Q8_0 (placeholder)
    {"matvec_q4_0", spv_matvec_q4_0, spv_matvec_q4_0_size, 3, 8},   // PIPE_DEQUANT_Q4_K (placeholder)
    {"matvec_q4_0", spv_matvec_q4_0, spv_matvec_q4_0_size, 3, 8},   // PIPE_DEQUANT_Q5_0 (placeholder)
    {"matvec_q4_0", spv_matvec_q4_0, spv_matvec_q4_0_size, 3, 8},   // PIPE_DEQUANT_Q6_K (placeholder)
    {"rmsnorm",     spv_rmsnorm,     spv_rmsnorm_size,     3, 8},   // PIPE_RMSNORM
    {"softmax",     spv_softmax,     spv_softmax_size,     1, 4},   // PIPE_SOFTMAX
    {"rope",        spv_rope,        spv_rope_size,        4, 24},  // PIPE_ROPE
    {"swiglu",      spv_swiglu,      spv_swiglu_size,      3, 4},   // PIPE_SWIGLU
    {"geglu",       spv_geglu,       spv_geglu_size,       3, 4},   // PIPE_GEGLU
    {"gelu",        spv_gelu,        spv_gelu_size,        1, 4},   // PIPE_GELU
    {"add",         spv_add,         spv_add_size,         3, 4},   // PIPE_ADD
    {"add",         spv_add,         spv_add_size,         3, 4},   // PIPE_ADD_SCALED (placeholder)
    {"scale",       spv_scale,       spv_scale_size,       1, 8},   // PIPE_SCALE
    {"scale",       spv_scale,       spv_scale_size,       1, 8},   // PIPE_MUL (placeholder)
    {"scale",       spv_scale,       spv_scale_size,       1, 8},   // PIPE_COPY_F32 (placeholder)
    {"attention",   spv_attention,   spv_attention_size,   4, 24},  // PIPE_ATTENTION
    {"rmsnorm_heads", spv_rmsnorm_heads, spv_rmsnorm_heads_size, 2, 8}, // PIPE_RMSNORM_HEADS
    {"add_rmsnorm", spv_add_rmsnorm, spv_add_rmsnorm_size, 5, 8}, // PIPE_ADD_RMSNORM
    {"quantize_q8_1", spv_quantize_q8_1, spv_quantize_q8_1_size, 2, 4}, // PIPE_QUANTIZE_Q8_1
    {"matvec_q4_0_dp4a", spv_matvec_q4_0_dp4a, spv_matvec_q4_0_dp4a_size, 3, 8}, // PIPE_MATVEC_Q4_0_DP4A
    {"matvec_q5_0_dp4a", spv_matvec_q5_0_dp4a, spv_matvec_q5_0_dp4a_size, 3, 8}, // PIPE_MATVEC_Q5_0_DP4A
    {"matvec_q8_0_dp4a", spv_matvec_q8_0_dp4a, spv_matvec_q8_0_dp4a_size, 3, 8}, // PIPE_MATVEC_Q8_0_DP4A
    {"matvec_q4_k_dp4a", spv_matvec_q4_k_dp4a, spv_matvec_q4_k_dp4a_size, 3, 8}, // PIPE_MATVEC_Q4_K_DP4A
    {"matvec_q6_k_dp4a", spv_matvec_q6_k_dp4a, spv_matvec_q6_k_dp4a_size, 3, 8}, // PIPE_MATVEC_Q6_K_DP4A
    {"matvec_q3_k_dp4a", spv_matvec_q3_k_dp4a, spv_matvec_q3_k_dp4a_size, 3, 8}, // PIPE_MATVEC_Q3_K_DP4A
    {"matvec_q5_k_dp4a", spv_matvec_q5_k_dp4a, spv_matvec_q5_k_dp4a_size, 3, 8}, // PIPE_MATVEC_Q5_K_DP4A
    {"ssm_conv1d_silu", spv_ssm_conv1d_silu, spv_ssm_conv1d_silu_size, 3, 8},   // PIPE_SSM_CONV1D_SILU
    {"ssm_preprocess",  spv_ssm_preprocess,  spv_ssm_preprocess_size,  5, 20},  // PIPE_SSM_PREPROCESS
    {"ssm_delta_rule",  spv_ssm_delta_rule,  spv_ssm_delta_rule_size,  5, 16},  // PIPE_SSM_DELTA_RULE
    {"ssm_decay",       spv_ssm_decay,       spv_ssm_decay_size,       2, 12},  // PIPE_SSM_DECAY
    {"ssm_predict",     spv_ssm_predict,     spv_ssm_predict_size,     3, 16},  // PIPE_SSM_PREDICT
    {"ssm_update",      spv_ssm_update,      spv_ssm_update_size,      4, 16},  // PIPE_SSM_UPDATE
    {"ssm_output",      spv_ssm_output,      spv_ssm_output_size,      3, 16},  // PIPE_SSM_OUTPUT
    {"ssm_norm_gate",   spv_ssm_norm_gate,   spv_ssm_norm_gate_size,   3, 8},   // PIPE_SSM_NORM_GATE
    {"deinterleave_qgate", spv_deinterleave_qgate, spv_deinterleave_qgate_size, 3, 8}, // PIPE_DEINTERLEAVE_QGATE
    {"sigmoid_gate",    spv_sigmoid_gate,    spv_sigmoid_gate_size,    2, 4},   // PIPE_SIGMOID_GATE
    {"paged_attention", spv_paged_attention, spv_paged_attention_size, 5, 28},  // PIPE_PAGED_ATTENTION
    {"matvec_q2_k",    spv_matvec_q2_k,    spv_matvec_q2_k_size,    3, 8},   // PIPE_MATVEC_Q2_K
    {"matvec_iq1_s",   spv_matvec_iq1_s,   spv_matvec_iq1_s_size,   4, 8},   // PIPE_MATVEC_IQ1_S
    {"matvec_iq1_m",   spv_matvec_iq1_m,   spv_matvec_iq1_m_size,   4, 8},   // PIPE_MATVEC_IQ1_M
    {"matvec_tq1_0",   spv_matvec_tq1_0,   spv_matvec_tq1_0_size,   3, 8},   // PIPE_MATVEC_TQ1_0
    {"matvec_iq2_xxs", spv_matvec_iq2_xxs, spv_matvec_iq2_xxs_size, 4, 8},   // PIPE_MATVEC_IQ2_XXS
    {"matvec_iq2_s",   spv_matvec_iq2_s,   spv_matvec_iq2_s_size,   4, 8},   // PIPE_MATVEC_IQ2_S
    {"matvec_iq3_xxs", spv_matvec_iq3_xxs, spv_matvec_iq3_xxs_size, 3, 8},   // PIPE_MATVEC_IQ3_XXS
    {"matvec_iq3_s",   spv_matvec_iq3_s,   spv_matvec_iq3_s_size,   4, 8},   // PIPE_MATVEC_IQ3_S
    {"matvec_iq4_xs",  spv_matvec_iq4_xs,  spv_matvec_iq4_xs_size,  3, 8},   // PIPE_MATVEC_IQ4_XS
    {"matvec_iq4_nl",  spv_matvec_iq4_nl,  spv_matvec_iq4_nl_size,  3, 8},   // PIPE_MATVEC_IQ4_NL
    {"matvec_mxfp4",   spv_matvec_mxfp4,   spv_matvec_mxfp4_size,   3, 8},   // PIPE_MATVEC_MXFP4
    {"attention_sinks", spv_attention_sinks, spv_attention_sinks_size, 5, 24}, // PIPE_ATTENTION_SINKS
    {"swiglu_oai",     spv_swiglu_oai,     spv_swiglu_oai_size,     3, 12},  // PIPE_SWIGLU_OAI
    {"add_offset",     spv_add_offset,     spv_add_offset_size,     2, 8},   // PIPE_ADD_OFFSET
    {"scale_add",      spv_scale_add,      spv_scale_add_size,      2, 8},   // PIPE_SCALE_ADD
    {"swiglu_oai_bias", spv_swiglu_oai_bias, spv_swiglu_oai_bias_size, 5, 20}, // PIPE_SWIGLU_OAI_BIAS
    {"moe_topk",       spv_moe_topk,       spv_moe_topk_size,       4, 24},  // PIPE_MOE_TOPK
    {"matvec_mxfp4_dp4a", spv_matvec_mxfp4_dp4a, spv_matvec_mxfp4_dp4a_size, 3, 8}, // PIPE_MATVEC_MXFP4_DP4A
    {"matvec_q4_0_dp4a_moe", spv_matvec_q4_0_dp4a_moe, spv_matvec_q4_0_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q4_0_DP4A_MOE
    {"matvec_q5_0_dp4a_moe", spv_matvec_q5_0_dp4a_moe, spv_matvec_q5_0_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q5_0_DP4A_MOE
    {"matvec_q8_0_dp4a_moe", spv_matvec_q8_0_dp4a_moe, spv_matvec_q8_0_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q8_0_DP4A_MOE
    {"matvec_q4_k_dp4a_moe", spv_matvec_q4_k_dp4a_moe, spv_matvec_q4_k_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q4_K_DP4A_MOE
    {"matvec_q6_k_dp4a_moe", spv_matvec_q6_k_dp4a_moe, spv_matvec_q6_k_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q6_K_DP4A_MOE
    {"matvec_q3_k_dp4a_moe", spv_matvec_q3_k_dp4a_moe, spv_matvec_q3_k_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q3_K_DP4A_MOE
    {"matvec_q5_k_dp4a_moe", spv_matvec_q5_k_dp4a_moe, spv_matvec_q5_k_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q5_K_DP4A_MOE
    {"matvec_mxfp4_dp4a_moe", spv_matvec_mxfp4_dp4a_moe, spv_matvec_mxfp4_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_MXFP4_DP4A_MOE
    {"matvec_iq3_xxs_moe", spv_matvec_iq3_xxs_moe, spv_matvec_iq3_xxs_moe_size, 4, 20}, // PIPE_MATVEC_IQ3_XXS_MOE
    {"moe_accumulate", spv_moe_accumulate, spv_moe_accumulate_size, 5, 12}, // PIPE_MOE_ACCUMULATE
    {"swiglu_oai_bias_moe", spv_swiglu_oai_bias_moe, spv_swiglu_oai_bias_moe_size, 6, 16}, // PIPE_SWIGLU_OAI_BIAS_MOE
    {"moe_bias_add", spv_moe_bias_add, spv_moe_bias_add_size, 3, 8}, // PIPE_MOE_BIAS_ADD
    {"attention_tiled", spv_attention_tiled, spv_attention_tiled_size, 4, 40}, // PIPE_ATTENTION_TILED
    {"kv_store_f16", spv_kv_store_f16, spv_kv_store_f16_size, 4, 12}, // PIPE_KV_STORE_F16
    {"kv_store_batch_f16", spv_kv_store_batch_f16, spv_kv_store_batch_f16_size, 4, 12}, // PIPE_KV_STORE_BATCH_F16
    {"attention_tiled_f32", spv_attention_tiled_f32, spv_attention_tiled_f32_size, 4, 28}, // PIPE_ATTENTION_TILED_F32
    {"layernorm", spv_layernorm, spv_layernorm_size, 4, 8}, // PIPE_LAYERNORM
    {"matvec_bf16", spv_matvec_bf16, spv_matvec_bf16_size, 3, 8}, // PIPE_MATVEC_BF16
    {"broadcast_mul", spv_broadcast_mul, spv_broadcast_mul_size, 2, 8}, // PIPE_BROADCAST_MUL
    {"tanh_gate_residual", spv_tanh_gate_residual, spv_tanh_gate_residual_size, 4, 8}, // PIPE_TANH_GATE_RESIDUAL
    {"rope_3d", spv_rope_3d, spv_rope_3d_size, 2, 20}, // PIPE_ROPE_3D
    {"attention_full_f32", spv_attention_full_f32, spv_attention_full_f32_size, 4, 24}, // PIPE_ATTENTION_FULL_F32
    {"attention_full_f16", spv_attention_full_f16, spv_attention_full_f16_size, 4, 24}, // PIPE_ATTENTION_FULL_F16
    {"flash_attn_f32", spv_flash_attn_f32, spv_flash_attn_f32_size, 4, 24}, // PIPE_FLASH_ATTN_F32
    {"conv2d_f32", spv_conv2d_f32, spv_conv2d_f32_size, 4, 40}, // PIPE_CONV2D_F32
    {"group_norm", spv_group_norm, spv_group_norm_size, 4, 16}, // PIPE_GROUP_NORM
    {"silu", spv_silu, spv_silu_size, 1, 4}, // PIPE_SILU
    {"upsample_nearest", spv_upsample_nearest, spv_upsample_nearest_size, 2, 12}, // PIPE_UPSAMPLE_NEAREST
    {"spatial_attention", spv_spatial_attention, spv_spatial_attention_size, 4, 12}, // PIPE_SPATIAL_ATTENTION
    {"gemm_q4_k", spv_gemm_q4_k, spv_gemm_q4_k_size, 3, 12}, // PIPE_GEMM_Q4_K
    {"split_qkv", spv_split_qkv, spv_split_qkv_size, 4, 12}, // PIPE_SPLIT_QKV
    {"dequant_q4k_f16", spv_dequant_q4k_f16, spv_dequant_q4k_f16_size, 2, 8}, // PIPE_DEQUANT_Q4K_F16
    {"gemm_f16", spv_gemm_f16, spv_gemm_f16_size, 3, 16}, // PIPE_GEMM_F16
    {"gemm_q4k_dp4a", spv_gemm_q4k_dp4a, spv_gemm_q4k_dp4a_size, 3, 12}, // PIPE_GEMM_Q4K_DP4A
    {"gemm_q4k_dp4a", spv_gemm_q4k_dp4a, spv_gemm_q4k_dp4a_size, 3, 12}, // PIPE_GEMM_Q4K_FUSED (placeholder)
    {"gemm_q4k_dp4a", spv_gemm_q4k_dp4a, spv_gemm_q4k_dp4a_size, 3, 12}, // PIPE_GEMM_Q4K_SMEM (placeholder)
    {"gemm_q4k_coopmat", spv_gemm_q4k_coopmat, spv_gemm_q4k_coopmat_size, 3, 12}, // PIPE_GEMM_Q4K_COOPMAT
    {"gemm_q5k_coopmat", spv_gemm_q5k_coopmat, spv_gemm_q5k_coopmat_size, 3, 12}, // PIPE_GEMM_Q5K_COOPMAT
    {"gemm_q6k_coopmat", spv_gemm_q6k_coopmat, spv_gemm_q6k_coopmat_size, 3, 12}, // PIPE_GEMM_Q6K_COOPMAT
    {"gemm_q6k_f32", spv_gemm_q6k_f32, spv_gemm_q6k_f32_size, 3, 12}, // PIPE_GEMM_Q6K_F32
    {"gemm_bf16", spv_gemm_bf16, spv_gemm_bf16_size, 3, 12}, // PIPE_GEMM_BF16
    {"gemm_f16_coopmat", spv_gemm_f16_coopmat, spv_gemm_f16_coopmat_size, 3, 12}, // PIPE_GEMM_F16_COOPMAT
    {"matvec_f32_moe", spv_matvec_f32_moe, spv_matvec_f32_moe_size, 4, 20}, // PIPE_MATVEC_F32_MOE
    {"matvec_q4_0_moe", spv_matvec_q4_0_moe, spv_matvec_q4_0_moe_size, 4, 20}, // PIPE_MATVEC_Q4_0_MOE
    {"matvec_q8_0_moe", spv_matvec_q8_0_moe, spv_matvec_q8_0_moe_size, 4, 20}, // PIPE_MATVEC_Q8_0_MOE
    {"matvec_q3_k_moe", spv_matvec_q3_k_moe, spv_matvec_q3_k_moe_size, 4, 20}, // PIPE_MATVEC_Q3_K_MOE
    {"matvec_q4_k_moe", spv_matvec_q4_k_moe, spv_matvec_q4_k_moe_size, 4, 20}, // PIPE_MATVEC_Q4_K_MOE
    {"matvec_q5_k_moe", spv_matvec_q5_k_moe, spv_matvec_q5_k_moe_size, 4, 20}, // PIPE_MATVEC_Q5_K_MOE
    {"matvec_q5_0_moe", spv_matvec_q5_0_moe, spv_matvec_q5_0_moe_size, 4, 20}, // PIPE_MATVEC_Q5_0_MOE
    {"matvec_q6_k_moe", spv_matvec_q6_k_moe, spv_matvec_q6_k_moe_size, 4, 20}, // PIPE_MATVEC_Q6_K_MOE
    {"matvec_q2_k_moe", spv_matvec_q2_k_moe, spv_matvec_q2_k_moe_size, 4, 20}, // PIPE_MATVEC_Q2_K_MOE
    {"matvec_tq1_0_moe", spv_matvec_tq1_0_moe, spv_matvec_tq1_0_moe_size, 4, 20}, // PIPE_MATVEC_TQ1_0_MOE
    {"matvec_iq1_s_moe", spv_matvec_iq1_s_moe, spv_matvec_iq1_s_moe_size, 5, 20}, // PIPE_MATVEC_IQ1_S_MOE
    {"matvec_iq1_m_moe", spv_matvec_iq1_m_moe, spv_matvec_iq1_m_moe_size, 5, 20}, // PIPE_MATVEC_IQ1_M_MOE
    {"matvec_iq2_xxs_moe", spv_matvec_iq2_xxs_moe, spv_matvec_iq2_xxs_moe_size, 5, 20}, // PIPE_MATVEC_IQ2_XXS_MOE
    {"matvec_iq2_s_moe", spv_matvec_iq2_s_moe, spv_matvec_iq2_s_moe_size, 5, 20}, // PIPE_MATVEC_IQ2_S_MOE
    {"matvec_iq3_s_moe", spv_matvec_iq3_s_moe, spv_matvec_iq3_s_moe_size, 5, 20}, // PIPE_MATVEC_IQ3_S_MOE
    {"matvec_iq4_xs_moe", spv_matvec_iq4_xs_moe, spv_matvec_iq4_xs_moe_size, 4, 20}, // PIPE_MATVEC_IQ4_XS_MOE
    {"matvec_iq4_nl_moe", spv_matvec_iq4_nl_moe, spv_matvec_iq4_nl_moe_size, 4, 20}, // PIPE_MATVEC_IQ4_NL_MOE
    {"matvec_mxfp4_moe", spv_matvec_mxfp4_moe, spv_matvec_mxfp4_moe_size, 4, 20}, // PIPE_MATVEC_MXFP4_MOE
    {"shared_expert_gate", spv_shared_expert_gate, spv_shared_expert_gate_size, 3, 4}, // PIPE_SHARED_EXPERT_GATE
    {"matvec_q4_k_dp4a_fused", spv_matvec_q4_k_dp4a_fused, spv_matvec_q4_k_dp4a_fused_size, 3, 8}, // PIPE_MATVEC_Q4_K_DP4A_FUSED
    {"matvec_iq3_xxs_dp4a", spv_matvec_iq3_xxs_dp4a, spv_matvec_iq3_xxs_dp4a_size, 3, 8}, // PIPE_MATVEC_IQ3_XXS_DP4A
    {"matvec_iq4_xs_dp4a", spv_matvec_iq4_xs_dp4a, spv_matvec_iq4_xs_dp4a_size, 3, 8}, // PIPE_MATVEC_IQ4_XS_DP4A
    {"matvec_iq4_nl_dp4a", spv_matvec_iq4_nl_dp4a, spv_matvec_iq4_nl_dp4a_size, 3, 8}, // PIPE_MATVEC_IQ4_NL_DP4A
    {"matvec_iq3_s_dp4a", spv_matvec_iq3_s_dp4a, spv_matvec_iq3_s_dp4a_size, 4, 8}, // PIPE_MATVEC_IQ3_S_DP4A
    {"matvec_iq3_xxs_dp4a_moe", spv_matvec_iq3_xxs_dp4a_moe, spv_matvec_iq3_xxs_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_IQ3_XXS_DP4A_MOE
    {"matvec_iq4_xs_dp4a_moe", spv_matvec_iq4_xs_dp4a_moe, spv_matvec_iq4_xs_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_IQ4_XS_DP4A_MOE
    {"matvec_iq4_nl_dp4a_moe", spv_matvec_iq4_nl_dp4a_moe, spv_matvec_iq4_nl_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_IQ4_NL_DP4A_MOE
    {"matvec_iq3_s_dp4a_moe", spv_matvec_iq3_s_dp4a_moe, spv_matvec_iq3_s_dp4a_moe_size, 5, 20}, // PIPE_MATVEC_IQ3_S_DP4A_MOE
    {"attention_tiled_sinks", spv_attention_tiled_sinks, spv_attention_tiled_sinks_size, 5, 40}, // PIPE_ATTENTION_TILED_SINKS
    {"matvec_q5_1", spv_matvec_q5_1, spv_matvec_q5_1_size, 3, 8}, // PIPE_MATVEC_Q5_1
    {"ssm_conv1d_batch", spv_ssm_conv1d_batch, spv_ssm_conv1d_batch_size, 4, 12},
    {"ssm_preprocess_batch", spv_ssm_preprocess_batch, spv_ssm_preprocess_batch_size, 5, 24},
    {"ssm_recurrence_batch", spv_ssm_recurrence_batch, spv_ssm_recurrence_batch_size, 5, 20},
    {"ssm_norm_gate_batch", spv_ssm_norm_gate_batch, spv_ssm_norm_gate_batch_size, 3, 12},
    {"gemm_q3k_tiled", spv_gemm_q3k_tiled, spv_gemm_q3k_tiled_size, 3, 16}, // PIPE_GEMM_Q3K_TILED
    {"ssm_recurrence_batch_t32", spv_ssm_recurrence_batch_t32, spv_ssm_recurrence_batch_t32_size, 5, 20}, // PIPE_SSM_RECURRENCE_BATCH_T32
    {"attention_tiled_barrier", spv_attention_tiled_barrier, spv_attention_tiled_barrier_size, 4, 28}, // PIPE_ATTENTION_TILED_BARRIER
    {"attention_tiled_kv4", spv_attention_tiled_kv4, spv_attention_tiled_kv4_size, 4, 28}, // PIPE_ATTENTION_TILED_KV4
    {"gemm_q3k_coopmat_recovered", spv_gemm_q3k_coopmat_recovered, spv_gemm_q3k_coopmat_recovered_size, 3, 12}, // PIPE_GEMM_Q3K_RECOVERED
    {"gemm_affine8_coopmat", spv_gemm_affine8_coopmat, spv_gemm_affine8_coopmat_size, 3, 12}, // PIPE_GEMM_AFFINE8_COOPMAT
    {"attention_flash_f16_coopmat", spv_attention_flash_f16_coopmat, spv_attention_flash_f16_coopmat_size, 4, 24}, // PIPE_ATTENTION_FLASH_COOPMAT
    {"matvec_affine8", spv_matvec_affine8, spv_matvec_affine8_size, 3, 12}, // PIPE_MATVEC_AFFINE8
    {"matvec_affine4", spv_matvec_affine4, spv_matvec_affine4_size, 3, 12}, // PIPE_MATVEC_AFFINE4
    {"gemm_affine4_coopmat", spv_gemm_affine4_coopmat, spv_gemm_affine4_coopmat_size, 3, 12}, // PIPE_GEMM_AFFINE4_COOPMAT
    {"matvec_affine4_moe", spv_matvec_affine4_moe, spv_matvec_affine4_moe_size, 4, 20}, // PIPE_MATVEC_AFFINE4_MOE
    {"matvec_affine8_moe", spv_matvec_affine8_moe, spv_matvec_affine8_moe_size, 4, 20}, // PIPE_MATVEC_AFFINE8_MOE
    {"attention_decode_split", spv_attention_decode_split, spv_attention_decode_split_size, 4, 32}, // PIPE_ATTENTION_DECODE_SPLIT
    {"attention_decode_gqa", spv_attention_decode_gqa, spv_attention_decode_gqa_size, 4, 32}, // PIPE_ATTENTION_DECODE_GQA
    {"attention_decode_merge", spv_attention_decode_merge, spv_attention_decode_merge_size, 2, 12}, // PIPE_ATTENTION_DECODE_MERGE
    {"convert_to_affine", spv_convert_to_affine, spv_convert_to_affine_size, 2, 12}, // PIPE_CONVERT_TO_AFFINE
    {"shortconv", spv_shortconv, spv_shortconv_size, 4, 12}, // PIPE_SHORT_CONV
    {"gather_rows", spv_gather_rows, spv_gather_rows_size, 3, 8}, // PIPE_GATHER_ROWS
    {"scatter_rows", spv_scatter_rows, spv_scatter_rows_size, 4, 12}, // PIPE_SCATTER_ROWS
    {"gemm_affine4_moe_grouped", spv_gemm_affine4_moe_grouped, spv_gemm_affine4_moe_grouped_size, 5, 16}, // PIPE_GEMM_AFFINE4_MOE_GROUPED
    {"gemm_affine8_moe_grouped", spv_gemm_affine8_moe_grouped, spv_gemm_affine8_moe_grouped_size, 5, 16}, // PIPE_GEMM_AFFINE8_MOE_GROUPED
    {"moe_slot_reduce", spv_moe_slot_reduce, spv_moe_slot_reduce_size, 4, 8}, // PIPE_MOE_SLOT_REDUCE
    {"moe_expert_bias_add", spv_moe_expert_bias_add, spv_moe_expert_bias_add_size, 3, 16}, // PIPE_MOE_EXPERT_BIAS_ADD
};

#endif // DLGO_SHADERS_SPIRV_H
"@

$header | Out-File -FilePath (Join-Path $PSScriptRoot '..\csrc\shaders_spirv.h') -Encoding ascii
Write-Host "Generated csrc/shaders_spirv.h"

# Touch vulkan_gpu.c so Go's CGo build cache detects the shader change.
# Go tracks .c file modification times but not transitively-included .h files,
# so without this, `go build` may silently reuse stale cached shader data.
(Get-Item (Join-Path $PSScriptRoot '..\csrc\vulkan_gpu.c')).LastWriteTime = Get-Date
Write-Host "Touched csrc/vulkan_gpu.c (cache invalidation)"

# Neither touching vulkan_gpu.c nor editing it invalidates the Go build cache
# in practice: cgo's action ID does not cover the shader table, so a shader edit
# silently has no effect and the stale SPIR-V keeps running (this burned a full
# debugging cycle). Emit the table's hash into a *Go* file in the same package,
# whose content change is always part of the build action.
$headerPath = Join-Path $PSScriptRoot '..\csrc\shaders_spirv.h'
$hash = (Get-FileHash $headerPath -Algorithm SHA256).Hash
$stampPath = Join-Path $PSScriptRoot '..\shader_stamp.go'
$stamp = @"
//go:build cgo && vulkan

// Code generated by gpu/shaders/compile.ps1. DO NOT EDIT.

package gpu

// ShaderTableSHA256 fingerprints csrc/shaders_spirv.h. It exists only so the
// Go build cache sees a change when the embedded SPIR-V changes: cgo's action
// ID does not include the transitively-included shader table, so without this
// a shader edit reuses the stale binary and appears to do nothing.
const ShaderTableSHA256 = "$hash"
"@
[System.IO.File]::WriteAllText($stampPath, $stamp, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "Wrote gpu/shader_stamp.go ($hash) for Go cache invalidation"
