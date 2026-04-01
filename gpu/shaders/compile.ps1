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
    @{name="conv2d_f32"; file="conv2d_f32.comp"},
    @{name="group_norm"; file="group_norm.comp"},
    @{name="silu"; file="silu.comp"},
    @{name="upsample_nearest"; file="upsample_nearest.comp"},
    @{name="spatial_attention"; file="spatial_attention.comp"}
)

$header = @"
// Auto-generated SPIR-V shader data. Do not edit.
#ifndef DLGO_SHADERS_SPIRV_H
#define DLGO_SHADERS_SPIRV_H

#include <stdint.h>
#include <stddef.h>

"@

foreach ($s in $shaders) {
    $spvFile = "$($s.name).spv"
    Write-Host "Compiling $($s.file) -> $spvFile"
    & glslc --target-env=vulkan1.2 -O $s.file -o $spvFile
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to compile $($s.file)"
        exit 1
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
    {"matvec_iq3_xxs", spv_matvec_iq3_xxs, spv_matvec_iq3_xxs_size, 4, 8},   // PIPE_MATVEC_IQ3_XXS
    {"matvec_iq3_s",   spv_matvec_iq3_s,   spv_matvec_iq3_s_size,   4, 8},   // PIPE_MATVEC_IQ3_S
    {"matvec_iq4_xs",  spv_matvec_iq4_xs,  spv_matvec_iq4_xs_size,  3, 8},   // PIPE_MATVEC_IQ4_XS
    {"matvec_iq4_nl",  spv_matvec_iq4_nl,  spv_matvec_iq4_nl_size,  3, 8},   // PIPE_MATVEC_IQ4_NL
    {"matvec_mxfp4",   spv_matvec_mxfp4,   spv_matvec_mxfp4_size,   3, 8},   // PIPE_MATVEC_MXFP4
    {"attention_sinks", spv_attention_sinks, spv_attention_sinks_size, 5, 24}, // PIPE_ATTENTION_SINKS
    {"swiglu_oai",     spv_swiglu_oai,     spv_swiglu_oai_size,     3, 12},  // PIPE_SWIGLU_OAI
    {"add_offset",     spv_add_offset,     spv_add_offset_size,     2, 8},   // PIPE_ADD_OFFSET
    {"scale_add",      spv_scale_add,      spv_scale_add_size,      2, 8},   // PIPE_SCALE_ADD
    {"swiglu_oai_bias", spv_swiglu_oai_bias, spv_swiglu_oai_bias_size, 5, 20}, // PIPE_SWIGLU_OAI_BIAS
    {"moe_topk",       spv_moe_topk,       spv_moe_topk_size,       3, 12},  // PIPE_MOE_TOPK
    {"matvec_mxfp4_dp4a", spv_matvec_mxfp4_dp4a, spv_matvec_mxfp4_dp4a_size, 3, 8}, // PIPE_MATVEC_MXFP4_DP4A
    {"matvec_q4_0_dp4a_moe", spv_matvec_q4_0_dp4a_moe, spv_matvec_q4_0_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q4_0_DP4A_MOE
    {"matvec_q5_0_dp4a_moe", spv_matvec_q5_0_dp4a_moe, spv_matvec_q5_0_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q5_0_DP4A_MOE
    {"matvec_q8_0_dp4a_moe", spv_matvec_q8_0_dp4a_moe, spv_matvec_q8_0_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q8_0_DP4A_MOE
    {"matvec_q4_k_dp4a_moe", spv_matvec_q4_k_dp4a_moe, spv_matvec_q4_k_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q4_K_DP4A_MOE
    {"matvec_q6_k_dp4a_moe", spv_matvec_q6_k_dp4a_moe, spv_matvec_q6_k_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q6_K_DP4A_MOE
    {"matvec_q3_k_dp4a_moe", spv_matvec_q3_k_dp4a_moe, spv_matvec_q3_k_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q3_K_DP4A_MOE
    {"matvec_q5_k_dp4a_moe", spv_matvec_q5_k_dp4a_moe, spv_matvec_q5_k_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_Q5_K_DP4A_MOE
    {"matvec_mxfp4_dp4a_moe", spv_matvec_mxfp4_dp4a_moe, spv_matvec_mxfp4_dp4a_moe_size, 4, 20}, // PIPE_MATVEC_MXFP4_DP4A_MOE
    {"moe_accumulate", spv_moe_accumulate, spv_moe_accumulate_size, 5, 12}, // PIPE_MOE_ACCUMULATE
    {"swiglu_oai_bias_moe", spv_swiglu_oai_bias_moe, spv_swiglu_oai_bias_moe_size, 6, 16}, // PIPE_SWIGLU_OAI_BIAS_MOE
    {"moe_bias_add", spv_moe_bias_add, spv_moe_bias_add_size, 3, 8}, // PIPE_MOE_BIAS_ADD
    {"attention_tiled", spv_attention_tiled, spv_attention_tiled_size, 4, 24}, // PIPE_ATTENTION_TILED
    {"kv_store_f16", spv_kv_store_f16, spv_kv_store_f16_size, 4, 8}, // PIPE_KV_STORE_F16
    {"kv_store_batch_f16", spv_kv_store_batch_f16, spv_kv_store_batch_f16_size, 4, 8}, // PIPE_KV_STORE_BATCH_F16
    {"attention_tiled_f32", spv_attention_tiled_f32, spv_attention_tiled_f32_size, 4, 24}, // PIPE_ATTENTION_TILED_F32
    {"layernorm", spv_layernorm, spv_layernorm_size, 4, 8}, // PIPE_LAYERNORM
    {"matvec_bf16", spv_matvec_bf16, spv_matvec_bf16_size, 3, 8}, // PIPE_MATVEC_BF16
    {"broadcast_mul", spv_broadcast_mul, spv_broadcast_mul_size, 2, 8}, // PIPE_BROADCAST_MUL
    {"tanh_gate_residual", spv_tanh_gate_residual, spv_tanh_gate_residual_size, 4, 8}, // PIPE_TANH_GATE_RESIDUAL
    {"rope_3d", spv_rope_3d, spv_rope_3d_size, 2, 20}, // PIPE_ROPE_3D
    {"attention_full_f32", spv_attention_full_f32, spv_attention_full_f32_size, 4, 24}, // PIPE_ATTENTION_FULL_F32
    {"conv2d_f32", spv_conv2d_f32, spv_conv2d_f32_size, 4, 40}, // PIPE_CONV2D_F32
    {"group_norm", spv_group_norm, spv_group_norm_size, 4, 16}, // PIPE_GROUP_NORM
    {"silu", spv_silu, spv_silu_size, 1, 4}, // PIPE_SILU
    {"upsample_nearest", spv_upsample_nearest, spv_upsample_nearest_size, 2, 12}, // PIPE_UPSAMPLE_NEAREST
    {"spatial_attention", spv_spatial_attention, spv_spatial_attention_size, 4, 12}, // PIPE_SPATIAL_ATTENTION
};

#endif // DLGO_SHADERS_SPIRV_H
"@

$header | Out-File -FilePath "..\csrc\shaders_spirv.h" -Encoding ascii
Write-Host "Generated csrc/shaders_spirv.h"

# Touch vulkan_gpu.c so Go's CGo build cache detects the shader change.
# Go tracks .c file modification times but not transitively-included .h files,
# so without this, `go build` may silently reuse stale cached shader data.
(Get-Item "..\csrc\vulkan_gpu.c").LastWriteTime = Get-Date
Write-Host "Touched csrc/vulkan_gpu.c (cache invalidation)"
