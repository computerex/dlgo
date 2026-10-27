#ifndef DLGO_VULKAN_GPU_H
#define DLGO_VULKAN_GPU_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
#define GPU_OK                0
#define GPU_ERR_NO_VULKAN    -1
#define GPU_ERR_NO_DEVICE    -2
#define GPU_ERR_INIT_FAIL    -3
#define GPU_ERR_OOM          -4
#define GPU_ERR_SHADER       -5
#define GPU_ERR_DISPATCH     -6

// Buffer usage flags
#define GPU_BUF_STORAGE      1
#define GPU_BUF_UNIFORM      2

// Quantization type IDs (matching GGML)
#define QTYPE_F32    0
#define QTYPE_F16    1
#define QTYPE_Q4_0   2
#define QTYPE_Q4_1   3
#define QTYPE_Q5_0   6
#define QTYPE_Q5_1   7
#define QTYPE_Q8_0   8
#define QTYPE_Q2_K  10
#define QTYPE_Q3_K  11
#define QTYPE_Q4_K  12
#define QTYPE_Q5_K  13
#define QTYPE_Q6_K  14
#define QTYPE_IQ2_XXS 16
#define QTYPE_IQ1_S   19
#define QTYPE_IQ3_XXS 18
#define QTYPE_IQ2_S   22
#define QTYPE_IQ3_S   21
#define QTYPE_IQ4_XS  23
#define QTYPE_IQ1_M   29
#define QTYPE_BF16    30
#define QTYPE_TQ1_0   34
// Canonical unified weight format (see quant/affine8.go). Not a GGML type;
// markers >39 are free. Weights uploaded in this format route to the single
// gpu_gemm_affine8 kernel regardless of their original quantization scheme.
#define QTYPE_AFFINE8 60
// Byte-lean canonical format for sub-8-bit sources (see quant/affine4.go).
#define QTYPE_AFFINE4 61

// Shader pipeline IDs
typedef enum {
    PIPE_MATVEC_F32 = 0,
    PIPE_MATVEC_F16,
    PIPE_MATVEC_Q4_0,
    PIPE_MATVEC_Q8_0,
    PIPE_MATVEC_Q3_K,
    PIPE_MATVEC_Q4_K,
    PIPE_MATVEC_Q5_K,
    PIPE_MATVEC_Q5_0,
    PIPE_MATVEC_Q6_K,
    PIPE_DEQUANT_Q4_0,
    PIPE_DEQUANT_Q8_0,
    PIPE_DEQUANT_Q4_K,
    PIPE_DEQUANT_Q5_0,
    PIPE_DEQUANT_Q6_K,
    PIPE_RMSNORM,
    PIPE_SOFTMAX,
    PIPE_ROPE,
    PIPE_SWIGLU,
    PIPE_GEGLU,
    PIPE_GELU,
    PIPE_ADD,
    PIPE_ADD_SCALED,
    PIPE_SCALE,
    PIPE_MUL,
    PIPE_COPY_F32,
    PIPE_ATTENTION,
    PIPE_RMSNORM_HEADS,
    PIPE_ADD_RMSNORM,
    PIPE_QUANTIZE_Q8_1,
    PIPE_MATVEC_Q4_0_DP4A,
    PIPE_MATVEC_Q5_0_DP4A,
    PIPE_MATVEC_Q8_0_DP4A,
    PIPE_MATVEC_Q4_K_DP4A,
    PIPE_MATVEC_Q6_K_DP4A,
    PIPE_MATVEC_Q3_K_DP4A,
    PIPE_MATVEC_Q5_K_DP4A,
    PIPE_SSM_CONV1D_SILU,
    PIPE_SSM_PREPROCESS,
    PIPE_SSM_DELTA_RULE,
    PIPE_SSM_DECAY,
    PIPE_SSM_PREDICT,
    PIPE_SSM_UPDATE,
    PIPE_SSM_OUTPUT,
    PIPE_SSM_NORM_GATE,
    PIPE_DEINTERLEAVE_QGATE,
    PIPE_SIGMOID_GATE,
    PIPE_PAGED_ATTENTION,
    PIPE_MATVEC_Q2_K,
    PIPE_MATVEC_IQ1_S,
    PIPE_MATVEC_IQ1_M,
    PIPE_MATVEC_TQ1_0,
    PIPE_MATVEC_IQ2_XXS,
    PIPE_MATVEC_IQ2_S,
    PIPE_MATVEC_IQ3_XXS,
    PIPE_MATVEC_IQ3_S,
    PIPE_MATVEC_IQ4_XS,
    PIPE_MATVEC_IQ4_NL,
    PIPE_MATVEC_MXFP4,
    PIPE_ATTENTION_SINKS,
    PIPE_SWIGLU_OAI,
    PIPE_ADD_OFFSET,
    PIPE_SCALE_ADD,
    PIPE_SWIGLU_OAI_BIAS,
    PIPE_MOE_TOPK,
    PIPE_MATVEC_MXFP4_DP4A,
    PIPE_MATVEC_Q4_0_DP4A_MOE,
    PIPE_MATVEC_Q5_0_DP4A_MOE,
    PIPE_MATVEC_Q8_0_DP4A_MOE,
    PIPE_MATVEC_Q4_K_DP4A_MOE,
    PIPE_MATVEC_Q6_K_DP4A_MOE,
    PIPE_MATVEC_Q3_K_DP4A_MOE,
    PIPE_MATVEC_Q5_K_DP4A_MOE,
    PIPE_MATVEC_MXFP4_DP4A_MOE,
    PIPE_MATVEC_IQ3_XXS_MOE,
    PIPE_MOE_ACCUMULATE,
    PIPE_SWIGLU_OAI_BIAS_MOE,
    PIPE_MOE_BIAS_ADD,
    PIPE_ATTENTION_TILED,
    PIPE_KV_STORE_F16,
    PIPE_KV_STORE_BATCH_F16,
    PIPE_ATTENTION_TILED_F32,
    PIPE_LAYERNORM,
    PIPE_MATVEC_BF16,
    PIPE_BROADCAST_MUL,
    PIPE_TANH_GATE_RESIDUAL,
    PIPE_ROPE_3D,
    PIPE_ATTENTION_FULL_F32,
    PIPE_ATTENTION_FULL_F16,
    PIPE_FLASH_ATTN_F32,
    PIPE_CONV2D_F32,
    PIPE_GROUP_NORM,
    PIPE_SILU,
    PIPE_UPSAMPLE_NEAREST,
    PIPE_SPATIAL_ATTENTION,
    PIPE_GEMM_Q4_K,
    PIPE_SPLIT_QKV,
    PIPE_DEQUANT_Q4K_F16,
    PIPE_GEMM_F16,
    PIPE_GEMM_Q4K_DP4A,
    PIPE_GEMM_Q4K_FUSED,
    PIPE_GEMM_Q4K_SMEM,
    PIPE_GEMM_Q4K_COOPMAT,
    PIPE_GEMM_Q5K_COOPMAT,
    PIPE_GEMM_Q6K_COOPMAT,
    PIPE_GEMM_Q6K_F32,
    PIPE_GEMM_BF16,
    PIPE_GEMM_F16_COOPMAT,
    // Native (float-input) MoE matvec variants
    PIPE_MATVEC_F32_MOE,
    PIPE_MATVEC_Q4_0_MOE,
    PIPE_MATVEC_Q8_0_MOE,
    PIPE_MATVEC_Q3_K_MOE,
    PIPE_MATVEC_Q4_K_MOE,
    PIPE_MATVEC_Q5_K_MOE,
    PIPE_MATVEC_Q5_0_MOE,
    PIPE_MATVEC_Q6_K_MOE,
    PIPE_MATVEC_Q2_K_MOE,
    PIPE_MATVEC_TQ1_0_MOE,
    PIPE_MATVEC_IQ1_S_MOE,
    PIPE_MATVEC_IQ1_M_MOE,
    PIPE_MATVEC_IQ2_XXS_MOE,
    PIPE_MATVEC_IQ2_S_MOE,
    PIPE_MATVEC_IQ3_S_MOE,
    PIPE_MATVEC_IQ4_XS_MOE,
    PIPE_MATVEC_IQ4_NL_MOE,
    PIPE_MATVEC_MXFP4_MOE,
    PIPE_SHARED_EXPERT_GATE,
    PIPE_MATVEC_Q4_K_DP4A_FUSED,
    PIPE_MATVEC_IQ3_XXS_DP4A,
    PIPE_MATVEC_IQ4_XS_DP4A,
    PIPE_MATVEC_IQ4_NL_DP4A,
    PIPE_MATVEC_IQ3_S_DP4A,
    PIPE_MATVEC_IQ3_XXS_DP4A_MOE,
    PIPE_MATVEC_IQ4_XS_DP4A_MOE,
    PIPE_MATVEC_IQ4_NL_DP4A_MOE,
    PIPE_MATVEC_IQ3_S_DP4A_MOE,
    PIPE_ATTENTION_TILED_SINKS,
    PIPE_MATVEC_Q5_1,
    PIPE_SSM_CONV1D_BATCH,
    PIPE_SSM_PREPROCESS_BATCH,
    PIPE_SSM_RECURRENCE_BATCH,
    PIPE_SSM_NORM_GATE_BATCH,
    // Q3_K batched GEMM (opt-in via DLGO_GEMM_Q3K; see gpu_batch_matvec).
    // Appended at the end so existing pipeline IDs stay stable.
    PIPE_GEMM_Q3K_TILED,
    PIPE_SSM_RECURRENCE_BATCH_T32,
    PIPE_ATTENTION_TILED_BARRIER,
    PIPE_ATTENTION_TILED_KV4,
    PIPE_GEMM_Q3K_RECOVERED,
    // Unified GEMM: canonical Affine8 weights × f32 input (see quant/affine8.go).
    // Appended at the end so existing pipeline IDs stay stable.
    PIPE_GEMM_AFFINE8_COOPMAT,
    // Tensor-core (cooperative matrix) bidirectional flash attention. Requires
    // head_dim == 128. Appended at the end so existing pipeline IDs stay stable.
    PIPE_ATTENTION_FLASH_COOPMAT,
    // Unified single-token matvec over canonical Affine8 weights (dense decode).
    // Appended at the end so existing pipeline IDs stay stable.
    PIPE_MATVEC_AFFINE8,
    // Byte-lean canonical Affine4 family (sub-8-bit sources; see quant/affine4.go):
    // one decode matvec + one prefill GEMM. Appended at the end (ID stability).
    PIPE_MATVEC_AFFINE4,
    PIPE_GEMM_AFFINE4_COOPMAT,
    // Fused MoE matvec over the canonical formats (n_used experts / dispatch).
    PIPE_MATVEC_AFFINE4_MOE,
    PIPE_MATVEC_AFFINE8_MOE,
    // Split-KV decode attention (npos == 1, head_dim == 128): one dispatch
    // produces per-split partials, the second merges them. Appended at the end
    // so existing pipeline IDs stay stable.
    PIPE_ATTENTION_DECODE_SPLIT,
    // GQA-folded split-KV decode attention: one workgroup per (KV head, split)
    // covering every q-head that shares that KV head, so the KV cache is read
    // once per position instead of once per q-head.
    PIPE_ATTENTION_DECODE_GQA,
    PIPE_ATTENTION_DECODE_MERGE,
    // Load-time in-VRAM weight converter: decodes a GGML K-quant (Q4_K/Q5_K/Q6_K)
    // block tensor and requantizes it into the canonical Affine4/Affine8 format on
    // the device (gpu/shaders/convert_to_affine.comp). Appended at the end so
    // existing pipeline IDs stay stable.
    PIPE_CONVERT_TO_AFFINE,
    // LFM2/LFM2MoE causal depthwise short convolution over the token axis
    // (gpu/shaders/shortconv.comp). Appended at the end so existing pipeline
    // IDs stay stable.
    PIPE_SHORT_CONV,
    // Position-batched MoE row remap helpers.
    // gather_rows (gpu/shaders/gather_rows.comp): dst[i] = src[idx[i]] whole
    // rows, used to pack the tokens that selected one expert into a dense
    // buffer so a single batched matvec per expert is a real GEMM.
    // scatter_rows (gpu/shaders/scatter_rows.comp): dst[idx[i]] (+)= w[i]*src[i],
    // the matching weighted accumulate back into the token rows.
    // Appended at the end so existing pipeline IDs stay stable.
    PIPE_GATHER_ROWS,
    PIPE_SCATTER_ROWS,
    // Grouped MoE GEMM: one dispatch per projection covering every expert, with
    // the expert as the z dimension of the grid and the token axis read through
    // the expert's token list (gpu/shaders/gemm_affine4_moe_grouped.comp). This
    // replaces the per-expert loop's ~192 dispatches per layer with 3. The
    // companion moe_slot_reduce does the weighted top-k sum in slot order.
    // Appended at the end so existing pipeline IDs stay stable.
    PIPE_GEMM_AFFINE4_MOE_GROUPED,
    PIPE_GEMM_AFFINE8_MOE_GROUPED,
    PIPE_MOE_SLOT_REDUCE,
    // Per-expert bias add for the position-batched MoE: one expert's [n_rows]
    // block of exp_dim-wide activations gets that expert's bias row added
    // (gpu/shaders/moe_expert_bias_add.comp). gpt-oss carries
    // ffn_{gate,up,down}_exps.bias, which the per-token path applies through
    // moe_bias_add; this is the batched-by-expert equivalent and needs no GPU
    // id list because the expert is a host-side loop variable.
    // Appended at the end so existing pipeline IDs stay stable.
    PIPE_MOE_EXPERT_BIAS_ADD,
    PIPE_COUNT
} PipelineID;

typedef uint64_t GpuBuf;

// Initialize Vulkan compute: creates instance, picks best device, creates queue
int gpu_init(void);
void gpu_shutdown(void);

// Device info
const char* gpu_device_name(void);
uint64_t gpu_vram_bytes(void);
uint64_t gpu_vram_free_bytes(void);
uint64_t gpu_allocated_bytes(void);
uint64_t gpu_host_visible_bytes(void);

// Device-local VRAM tiering control.
// gpu_set_vram_ceiling pins the device-local ceiling (0 = default rule). It can
// only lower the ceiling, never raise it above the VRAM_RESERVE_BYTES rule.
// Allocations past the ceiling go to host-visible memory and are still read
// directly by shaders over PCIe, so they stay on the GPU compute path.
void gpu_set_vram_ceiling(uint64_t bytes);
// gpu_min_ssbo_offset_align returns minStorageBufferOffsetAlignment, i.e. the
// largest power of two that a storage-buffer descriptor offset (used to point a
// matvec at one MoE expert inside a packed tensor) must be a multiple of.
uint64_t gpu_min_ssbo_offset_align(void);

uint64_t gpu_vram_ceiling_bytes(void);
uint64_t gpu_device_local_bytes(void);
int gpu_is_initialized(void);
int gpu_has_dp4a(void);

int gpu_matvec_offset_dp4a(GpuBuf out_buf, int out_offset_bytes,
                           GpuBuf weights_buf, int weights_offset_bytes,
                           GpuBuf q8_1_buf, int rows, int cols, int qtype);

int gpu_moe_matvec_dp4a(GpuBuf out_buf, GpuBuf weights_buf,
                        GpuBuf q8_1_buf, GpuBuf indices_buf,
                        int rows, int cols, int qtype,
                        int expert_stride, int base_offset,
                        int shared_input, int n_used);

// table_buf may be 0; auto-resolved from qtype using uploaded IQ tables.
int gpu_moe_matvec_iq(GpuBuf out_buf, GpuBuf weights_buf,
                      GpuBuf x_buf, GpuBuf table_buf, GpuBuf indices_buf,
                      int rows, int cols, int qtype,
                      int expert_stride, int base_offset,
                      int shared_input, int n_used);

int gpu_moe_accumulate(GpuBuf out_buf, GpuBuf exp_outs_buf, GpuBuf weights_buf,
                       GpuBuf bias_buf, GpuBuf indices_buf,
                       int dim, int n_used, int has_bias);

int gpu_swiglu_oai_bias_moe(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf,
                            GpuBuf gate_bias_buf, GpuBuf up_bias_buf, GpuBuf indices_buf,
                            int total_n, float alpha, float limit, int exp_dim);

int gpu_moe_bias_add(GpuBuf data_buf, GpuBuf bias_buf, GpuBuf indices_buf,
                     int exp_dim, int n_used);

int gpu_swiglu_at(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf,
                  int out_off, int gate_off, int up_off, int n);
int gpu_swiglu_oai_at(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf,
                      int out_off, int gate_off, int up_off,
                      int n, float alpha, float limit);
int gpu_quantize_q8_1_at(GpuBuf q8_1_buf, int q8_off, GpuBuf f32_buf, int f32_off, int n_elements);

// Buffer management
GpuBuf gpu_alloc(uint64_t size_bytes, int usage);
void gpu_free(GpuBuf buf);
void gpu_reset_buffer_table(void);
int gpu_pool_create(uint64_t size_bytes);
void gpu_pool_seal(void);
void gpu_pool_destroy(void);
int gpu_upload(GpuBuf dst, const void* src, uint64_t size_bytes, uint64_t offset);
int gpu_download(void* dst, GpuBuf src, uint64_t size_bytes, uint64_t offset);

// Matrix-vector multiply: out[r] = dot(weights[r,:], x) for r in [0,rows)
// weights_buf: quantized weight matrix on GPU
// x_buf: input vector on GPU (float32)
// out_buf: output vector on GPU (float32)
int gpu_matvec(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
               int rows, int cols, int qtype);
int gpu_matvec_offset(GpuBuf out_buf, int out_offset_bytes,
                      GpuBuf weights_buf, int weights_offset_bytes,
                      GpuBuf x_buf, int rows, int cols, int qtype);

// Batched matvec against a weight slice starting at weights_offset_bytes inside
// a packed tensor (one MoE expert). Uses the same per-format matvec kernels,
// walking input columns by workgroup.y and shifting the weight binding, so a
// per-expert GEMM needs no dedicated kernel. Returns GPU_ERR_DISPATCH when the
// quant type has no plain matvec pipeline.
int gpu_batch_matvec_offset(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                            int rows, int cols, int npos, int qtype,
                            uint64_t weights_offset_bytes);

// Position-batched MoE row remap.
// gather:  dst[i] = src[idx[i]]
// scatter: dst[idx[i]] (+)= w[i] * src[i]   (add != 0 accumulates)
int gpu_gather_rows(GpuBuf dst, GpuBuf src, GpuBuf idx, int n, int dim);
int gpu_scatter_rows(GpuBuf dst, GpuBuf src, GpuBuf idx, GpuBuf weights,
                     int n, int dim, int add);

// Grouped MoE GEMM: one dispatch covers every expert of the layer.
//
// Grid is (ceil(rows/64), ceil(npos/128), n_experts), so a workgroup whose expert
// received no tokens returns immediately. The token axis is read through
// tok_idx[starts[e] .. starts[e+1]-1], i.e. the positions that routed to expert
// e, which is why no gathered activation buffer is needed. The result is written
// pair-major: out[(starts[e] + n) * rows + m]. a_blocks is the Affine block count
// of one expert slice and a_base an extra block offset (fused [gate|up] slices);
// expressing the slice in blocks rather than bytes keeps it independent of
// minStorageBufferOffsetAlignment.
int gpu_gemm_affine4_moe_grouped(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                                 GpuBuf tok_idx, GpuBuf starts, int rows, int cols,
                                 int npos, int n_experts, int a_blocks, int a_base);
int gpu_gemm_affine8_moe_grouped(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                                 GpuBuf tok_idx, GpuBuf starts, int rows, int cols,
                                 int npos, int n_experts, int a_blocks, int a_base);

// Deterministic top-k slot reduce for the grouped MoE:
// out[t*dim + c] = sum_s slot_w[t*nUsed+s] * pair[slot_pair[t*nUsed+s]*dim + c].
// Slot order is ascending, matching the per-token path's accumulate order, so the
// result never depends on scheduling.
int gpu_moe_slot_reduce(GpuBuf out_buf, GpuBuf pair_buf, GpuBuf slot_pair_buf,
                        GpuBuf slot_w_buf, int dim, int n_used, int npos);

// Per-expert bias add over a [n_rows][exp_dim] activation block:
// data[i] += bias[base + (i % exp_dim)], where base is either bias_base (one
// expert, from the loop variable) or eidx[i / exp_dim] * exp_dim (the grouped
// path, whose pair-major activations mix experts). idx_buf must always be a
// valid int32 buffer; it is only read when use_index != 0.
int gpu_moe_expert_bias_add(GpuBuf data_buf, GpuBuf bias_buf, GpuBuf idx_buf,
                            int n, int exp_dim, int use_index, int bias_base);

// Batch matrix-vector: out[p*rows+r] = dot(W[r,:], x[p*cols...]) for each position p
int gpu_batch_matvec(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                     int rows, int cols, int npos, int qtype);

// Unified GEMM over the canonical Affine8 weight format (every GGUF quant scheme
// is repacked to Affine8 at load time; see quant/affine8.go). One kernel, one
// weight type, one optimization surface.
int gpu_gemm_affine8(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                     int rows, int cols, int npos);

// Unified GEMM over the canonical Affine4 weight format (byte-lean companion of
// Affine8 for sub-8-bit sources; see quant/affine4.go).
int gpu_gemm_affine4(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                     int rows, int cols, int npos);

// Element-wise operations
int gpu_rmsnorm(GpuBuf out_buf, GpuBuf x_buf, GpuBuf weight_buf, int n, float eps);
int gpu_layernorm(GpuBuf out_buf, GpuBuf x_buf, GpuBuf weight_buf, GpuBuf bias_buf, int n, float eps);
int gpu_rmsnorm_heads(GpuBuf data_buf, GpuBuf weight_buf, int num_heads, int head_dim, float eps);

// npos-token forms: same shaders, groups_y = position. Used by the batched
// prefill for the q/k norm and the fused FFN residual + attention norm.
int gpu_batch_rmsnorm_heads(GpuBuf data_buf, GpuBuf weight_buf,
                            int num_heads, int head_dim, float eps, int npos);
int gpu_batch_add_rmsnorm(GpuBuf norm_out, GpuBuf sum_out,
                          GpuBuf a_buf, GpuBuf b_buf, GpuBuf weight_buf,
                          int n, float eps, int npos);
int gpu_softmax(GpuBuf buf, int n);
int gpu_rope(GpuBuf q_buf, GpuBuf k_buf, GpuBuf cos_table, GpuBuf sin_table,
             int num_heads, int num_kv_heads, int head_dim, int rope_dim,
             int pos, int neox);
int gpu_swiglu(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf, int n);
int gpu_swiglu_oai(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf, int n, float alpha, float limit);
int gpu_geglu(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf, int n);
int gpu_add_offset(GpuBuf out_buf, GpuBuf bias_buf, int n, int offset);
int gpu_scale_add(GpuBuf out_buf, GpuBuf in_buf, int n, float scale);
int gpu_swiglu_oai_bias(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf,
                        GpuBuf gate_bias_buf, GpuBuf up_bias_buf,
                        int n, float alpha, float limit,
                        int gate_bias_offset, int up_bias_offset);
int gpu_moe_topk(GpuBuf logits_buf, GpuBuf out_indices_buf, GpuBuf out_weights_buf,
                 GpuBuf sel_bias_buf, int n_experts, int k, int gating_func,
                 int weights_norm, float weights_scale, int has_sel_bias);
int gpu_gelu(GpuBuf buf, int n);
int gpu_add(GpuBuf out_buf, GpuBuf a_buf, GpuBuf b_buf, int n);
int gpu_add_rmsnorm(GpuBuf norm_out, GpuBuf sum_out,
                    GpuBuf a_buf, GpuBuf b_buf, GpuBuf weight_buf,
                    int n, float eps);
int gpu_add_bias(GpuBuf buf, GpuBuf bias_buf, int n);
int gpu_scale(GpuBuf buf, float s, int n);
int gpu_copy_f32(GpuBuf dst, GpuBuf src, int n);

// Fused multi-head attention: dispatches one workgroup per head
// q_buf: [num_heads * head_dim], k_cache/v_cache: [max_seq_len * kv_dim]
// out_buf: [num_heads * head_dim]
int gpu_attention(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                  int num_heads, int num_kv_heads, int head_dim, int kv_dim,
                  int seq_len, float scale, int start_pos);
int gpu_attention_sinks(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                        GpuBuf sinks_buf, int num_heads, int num_kv_heads, int head_dim,
                        int kv_dim, int seq_len, float scale, int start_pos);

// KV cache operations
int gpu_kv_store(GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                 GpuBuf k_buf, GpuBuf v_buf,
                 int pos, int kv_dim);

// FP16 KV cache operations: float32 → packed half2 quantization
int gpu_kv_store_f16(GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                     GpuBuf k_buf, GpuBuf v_buf,
                     int pos, int kv_dim);
int gpu_batch_kv_store_f16(GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                           GpuBuf k_buf, GpuBuf v_buf,
                           int start_pos, int kv_dim, int npos);

// Tiled attention with FP16 KV cache (online softmax, no seq_len limit)
int gpu_attention_f16(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                      int num_heads, int num_kv_heads, int head_dim, int kv_dim,
                      int seq_len, float scale, float softcap, int start_pos);
int gpu_batch_attention_f16(GpuBuf out, GpuBuf q, GpuBuf k_cache, GpuBuf v_cache,
                            int num_heads, int num_kv_heads, int head_dim,
                            int kv_dim, int start_seq_len, float scale, float softcap,
                            int npos, int window);

// Split-KV decode attention (npos == 1). Splits the KV range across num_splits
// workgroups per head to fill the GPU, stages K/V tiles in shared memory and
// writes per-split partials into `partial_buf`, which must hold at least
// num_heads * num_splits * (2 + head_dim) floats. The second dispatch merges
// those partials into `out_buf`.
int gpu_attention_decode(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                         GpuBuf partial_buf, int num_heads, int num_kv_heads, int head_dim,
                         int kv_dim, int seq_len, float scale, float softcap, int start_pos);

// GQA-folded variant of the above: one workgroup per (KV head, split) that
// covers all num_heads/num_kv_heads q-heads sharing that KV head, so the KV
// cache is staged and read once per position instead of once per q-head.
// Requires 2 <= kv_mul <= 8 and kv_mul * (head_dim/8) <= 256; otherwise it
// returns GPU_ERR_DISPATCH and the caller keeps gpu_attention_decode.
int gpu_attention_decode_gqa(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                             GpuBuf partial_buf, int num_heads, int num_kv_heads, int head_dim,
                             int kv_dim, int seq_len, float scale, float softcap, int start_pos);

// Maximum num_splits gpu_attention_decode[_gqa] will pick (bounds the partial buffer).
#define GPU_ATTENTION_DECODE_MAX_SPLITS 16

// Select the split-KV decode kernel for gpu_attention_f16 (default on for
// supported head_dim). Exposed so tests can compare against the tiled kernel.
void gpu_set_attention_decode(int on);

// Select the GQA-folded split-KV decode kernel (default on when kv_mul > 1).
void gpu_set_attention_decode_gqa(int on);

// Which kernel the most recent single-token decode attention dispatch used:
// 0 = tiled, 1 = split (per q-head), 2 = GQA-folded split. Diagnostics/tests.
int gpu_attention_decode_used(void);

// Tiled attention with FP32 KV cache (online softmax, no seq_len limit)
int gpu_attention_tiled_f32(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                            int num_heads, int num_kv_heads, int head_dim, int kv_dim,
                            int seq_len, float scale, float softcap, int start_pos);
int gpu_batch_attention_tiled_f32(GpuBuf out, GpuBuf q, GpuBuf k_cache, GpuBuf v_cache,
                                  int num_heads, int num_kv_heads, int head_dim,
                                  int kv_dim, int start_seq_len, float scale, float softcap, int npos);

// Paged KV store: write K/V into block pool at computed block offset.
// effective_pos = physical_block * block_size + slot_in_block (computed on CPU)
int gpu_paged_kv_store(GpuBuf k_pool, GpuBuf v_pool,
                       GpuBuf k_buf, GpuBuf v_buf,
                       int effective_pos, int kv_dim);

// Paged attention: block-table-indexed KV access for PagedAttention.
// block_table_buf contains int32 physical block IDs.
int gpu_paged_attention(GpuBuf out_buf, GpuBuf q_buf,
                        GpuBuf k_pool_buf, GpuBuf v_pool_buf,
                        GpuBuf block_table_buf,
                        int num_heads, int num_kv_heads, int head_dim, int kv_dim,
                        int seq_len, float scale, int block_size);

// Dequantize a buffer from quantized format to float32
int gpu_dequantize(GpuBuf out_f32_buf, GpuBuf quant_buf, int n, int qtype);

// Diffusion-specific operations
int gpu_broadcast_mul(GpuBuf data_buf, GpuBuf scale_buf, int total_n, int dim);
int gpu_tanh_gate_residual(GpuBuf out_buf, GpuBuf residual_buf, GpuBuf data_buf,
                           GpuBuf gate_buf, int total_n, int dim);
int gpu_rope_3d(GpuBuf vec_buf, GpuBuf pe_buf, int n_pos, int n_heads,
                int head_dim, int pe_offset, int pe_stride);
int gpu_attention_full_f32(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_buf, GpuBuf v_buf,
                           int num_heads, int num_kv_heads, int head_dim, int kv_dim,
                           int seq_len, float scale);
int gpu_rmsnorm_heads_batch(GpuBuf data_buf, GpuBuf weight_buf,
                            int num_heads, int head_dim, int npos, float eps);

// VAE-specific operations
int gpu_conv2d_f32(GpuBuf out_buf, GpuBuf in_buf, GpuBuf weight_buf, GpuBuf bias_buf,
                   int inCh, int H, int W, int kH, int kW, int padH, int padW,
                   int stride, int outH, int outW, int outCh);
int gpu_group_norm(GpuBuf out_buf, GpuBuf in_buf, GpuBuf weight_buf, GpuBuf bias_buf,
                   int C, int spatialSize, int numGroups, float eps);
int gpu_silu(GpuBuf data_buf, int n);
int gpu_upsample_nearest(GpuBuf out_buf, GpuBuf in_buf, int C, int H, int W);
int gpu_spatial_attention(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_buf, GpuBuf v_buf,
                          int C, int spatial, float scale);
int gpu_gemm_q4_k(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                  int M, int K, int N);
int gpu_split_qkv(GpuBuf q_buf, GpuBuf k_buf, GpuBuf v_buf, GpuBuf qkv_buf,
                   int qDim, int kvDim, int seqLen);
int gpu_dequant_q4k_f16(GpuBuf out_f16, GpuBuf weights_q4k, int rows, int cols);
// Convert n elements of a GGML K-quant tensor (src, already uploaded) into the
// canonical Affine format in out. out_bits = 8 -> Affine8, 4 -> Affine4.
// qtype is the GGML id: 12=Q4_K, 13=Q5_K, 14=Q6_K. One load-time dispatch.
int gpu_convert_quant(GpuBuf out, GpuBuf src, int n, int qtype, int out_bits);
int gpu_gemm_f16(GpuBuf out_buf, GpuBuf weight_f16, GpuBuf x_buf,
                 int M, int N, int K, int out_stride);
int gpu_gemm_bf16(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                  int rows, int cols, int npos);
int gpu_gemm_q4k(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                 int rows, int cols, int npos);
int gpu_gemm_q5k(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                 int rows, int cols, int npos);
int gpu_gemm_q6k(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                 int rows, int cols, int npos);
int gpu_gemm_q3k_tiled(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                       int rows, int cols, int npos);

// Test-only: force Q3_K GEMM availability (0=off, 1=on). Not used by inference.
void gpu_set_gemm_q3k(int on);

// Disable cooperative-matrix GEMMs (0=allow, 1=disable). Their kernels narrow
// activations to FP16, which overflows for diffusion DiT activations; disabling
// selects the FP32-activation GEMM paths instead.
void gpu_set_no_coopmat_gemm(int on);

// Precision override for the DiT full-attention kernel:
//   -1 = follow DLGO_ATTN_F16 env (default: f16 ON), 0 = force f32, 1 = force f16.
void gpu_set_attention_f16(int v);

// Tensor-core (cooperative matrix) full-attention override:
//   -1 = follow DLGO_ATTN_COOPMAT env (default OFF), 1 = force ON. head_dim==128 only.
void gpu_set_attention_coopmat(int v);

// Synchronize: wait for all GPU work to complete
void gpu_sync(void);

// Batch mode: record all dispatches into one command buffer, submit once
void gpu_begin_batch(void);
void gpu_end_batch(void);
void gpu_barrier(void);

// Quantize F32 → Q8_1 for dp4a integer dot product path
int gpu_quantize_q8_1(GpuBuf q8_1_buf, GpuBuf f32_buf, int n_elements);

// MatVec using dp4a (dotPacked4x8EXT): weights × Q8_1 input
int gpu_matvec_dp4a(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf q8_1_buf,
                    int rows, int cols, int qtype);

// Fused Q4_K dp4a matvec: quantizes float32 input to Q8_1 in shared memory,
// then performs dp4a dot product. Eliminates separate quantize dispatch + barrier.
int gpu_matvec_dp4a_fused(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_f32_buf,
                          int rows, int cols);

// Performance counters
void gpu_perf_reset(void);
void gpu_perf_print(void);
double gpu_perf_get_gpu_us(void);
int    gpu_perf_get_dispatches(void);
int    gpu_perf_get_barriers(void);

// Forward declaration for inline MoE conf in layer conf.
typedef struct GpuMoEConf_s GpuMoEConf;

// Fused layer: records all dispatches for one transformer layer in a single
// CGo call, eliminating per-operation Go→C overhead.
typedef struct {
    GpuBuf x, x_norm, q, k, v, attn_out, attn_proj;
    GpuBuf ffn_norm, ffn_in, gate, up, hidden, ffn_out;

    GpuBuf attn_norm_w;
    GpuBuf wq, wk, wv, wo;
    int wq_rows, wq_cols, wq_type;
    int wk_rows, wk_cols, wk_type;
    int wv_rows, wv_cols, wv_type;
    int wo_rows, wo_cols, wo_type;

    GpuBuf bq, bk, bv, bo;
    GpuBuf q_norm_w, k_norm_w;

    GpuBuf ffn_norm_w;
    GpuBuf ffn_gate_w, ffn_up_w, ffn_down_w;
    int gate_rows, gate_cols, gate_type;
    int up_rows, up_cols, up_type;
    int down_rows, down_cols, down_type;

    GpuBuf post_attn_norm_w, post_ffn_norm_w;
    GpuBuf k_cache, v_cache;

    int dim, head_dim, num_heads, num_kv_heads, kv_dim;
    float rms_eps;
    int rope_dim;
    int rope_neox;
    GpuBuf rope_cos_table, rope_sin_table;
    int ffn_type;       // 0=swiglu, 1=geglu, 2=plain, 3=moe_skip (CPU handles FFN)
    int residual_type;  // 0=standard, 1=parallel

    GpuBuf q8_1_scratch; // Q8_1 scratch buffer (reused for each quantize step)
    int use_dp4a;        // 1 to use dp4a path, 0 for float path
    int core_type;       // 0=attention, 1=SSM (skip attn, Go fills attn_proj)
    GpuBuf attn_sinks;   // learned sink logits per head (0 if not used)
    int sliding_window;  // ISWA: max window size for this layer (0 = full attention)
    float attn_logit_softcap; // attention logit soft-capping (0 = disabled)

    // Inline MoE FFN (ffn_type=3): if non-NULL, runs MoE FFN (+ residual
    // if moe_has_shared=0) in the same C call instead of returning to Go.
    const GpuMoEConf* moe_conf;
    int moe_use_native;  // 1=native matvec, 0=dp4a
    int moe_has_shared;  // 1=shared expert exists, Go handles shared+residual
    int f32_kv;          // 1=use F32 KV cache (debug: DLGO_F32_KV=1)
} GpuLayerConf;

// Records all dispatches for one transformer layer.
// next_attn_norm: weight buffer for next layer's attn norm (fused FFN residual
// + next RMSNorm). Pass 0 for the last layer.
int gpu_forward_layer(const GpuLayerConf* lc, int pos, int seq_len, float scale,
                      GpuBuf next_attn_norm);

// Batched layer forward: processes npos tokens through one transformer layer.
// All scratch buffers in lc must be npos× the single-token size.
int gpu_forward_layer_batch(const GpuLayerConf* lc, int npos, int start_pos,
                            float scale, GpuBuf next_attn_norm);

// Batch RMS normalization (groups_y = npos).
int gpu_batch_rmsnorm(GpuBuf out_buf, GpuBuf x_buf, GpuBuf weight_buf, int n, int npos, float eps);

// Copy a region between GPU buffers.
int gpu_copy_region(GpuBuf dst, uint64_t dst_offset, GpuBuf src, uint64_t src_offset, uint64_t size);

// SSM (Gated Delta Net) operations
int gpu_ssm_conv1d_silu(GpuBuf qkv, GpuBuf conv_state, GpuBuf conv_w, int channels, int conv_k);

// LFM2/LFM2MoE causal depthwise short convolution (kernel conv_k, taps
// oldest-first) with a rolling (conv_k-1)-row state over the token axis.
// bcx holds npos rows of [3*dim] in_proj output, y receives npos rows of
// [dim] c-gated conv output. state is updated in place.
int gpu_shortconv(GpuBuf bcx_buf, GpuBuf y_buf, GpuBuf state_buf, GpuBuf conv_w_buf,
                  int dim, int conv_k, int npos);
int gpu_ssm_preprocess(GpuBuf alpha, GpuBuf beta, GpuBuf ssma, GpuBuf dt_bias,
                       GpuBuf qkv, int num_heads, int head_k_dim, int key_dim,
                       float rms_eps, int has_dt_bias);
int gpu_ssm_delta_rule(GpuBuf state, GpuBuf qkv, GpuBuf alpha, GpuBuf beta,
                       GpuBuf y, int num_heads, int head_k_dim, int head_v_dim, int key_dim);
int gpu_ssm_decay(GpuBuf state, GpuBuf alpha, int num_heads, int head_k_dim, int head_v_dim);
int gpu_ssm_predict(GpuBuf state, GpuBuf qkv, GpuBuf pred,
                    int num_heads, int head_k_dim, int head_v_dim, int key_dim);
int gpu_ssm_update(GpuBuf state, GpuBuf qkv, GpuBuf beta, GpuBuf pred,
                   int num_heads, int head_k_dim, int head_v_dim, int key_dim);
int gpu_ssm_output(GpuBuf state, GpuBuf qkv, GpuBuf y,
                   int num_heads, int head_k_dim, int head_v_dim, int key_dim);
int gpu_ssm_norm_gate(GpuBuf y, GpuBuf z, GpuBuf norm_w,
                      int num_heads, int head_v_dim, float eps);

// GatedQ attention operations
int gpu_deinterleave_qgate(GpuBuf qfull, GpuBuf q, GpuBuf qgate,
                           int num_heads, int head_dim);
int gpu_sigmoid_gate(GpuBuf out_buf, GpuBuf gate_buf, int n);
int gpu_shared_expert_gate(GpuBuf out_buf, GpuBuf gate_w_buf, GpuBuf input_buf, int n);

// Fused MoE FFN: runs entire MoE FFN in a single CGo call.
// Eliminates ~18 CGo calls per layer → 1 call.
struct GpuMoEConf_s {
    // Scratch buffers
    GpuBuf ffn_norm;         // input (RMS-normed hidden state)
    GpuBuf ffn_out;          // output (weighted expert sum)
    GpuBuf moe_logits;       // [n_experts] scratch
    GpuBuf moe_topk_idx;     // [n_used] scratch (float-encoded indices)
    GpuBuf moe_topk_w;       // [n_used] scratch (softmax weights)
    GpuBuf q8_input;         // Q8_1 quantized input scratch
    GpuBuf gate_scratch;     // [n_used * exp_dim] gate projections
    GpuBuf up_scratch;       // [n_used * exp_dim] up projections
    GpuBuf q8_down_packed;   // Q8_1 packed hidden for down proj
    GpuBuf out_scratch;      // [n_used * dim] per-expert outputs

    // Router
    GpuBuf router_w;
    int router_rows, router_cols, router_type;
    GpuBuf router_bias;      // 0 if none
    // Selection-only expert bias (LFM2MoE / gpt-oss): added AFTER the gating
    // activation and used only to choose which experts run. 0 if none.
    GpuBuf sel_bias;

    // Expert weights
    GpuBuf gate_w, up_w, down_w;
    int gate_type, up_type, down_type;
    int gate_stride, gate_base;    // block-unit strides
    int up_stride, up_base;
    int down_stride;

    // Expert biases (0 if none)
    GpuBuf gate_bias, up_bias, down_bias;

    // Dimensions
    int dim, exp_dim, n_experts, n_used;
    int gating_func;
    int weights_norm;
    float weights_scale;

    // Activation mode: 0=SwiGLU, 1=SwiGLU_OAI
    int is_oai;
    float alpha, limit;

    // Shared expert (all 0 if not present)
    int has_shared;
    GpuBuf sh_gate_w, sh_up_w, sh_down_w;
    int sh_gate_rows, sh_gate_cols, sh_gate_type;
    int sh_up_rows, sh_up_cols, sh_up_type;
    int sh_down_rows, sh_down_cols, sh_down_type;
    GpuBuf sh_gate_scratch, sh_up_scratch, sh_hidden_scratch, sh_out_scratch;
    GpuBuf sh_router_w;
};

int gpu_forward_moe_ffn(const GpuMoEConf* mc);

// Native (float-input) fused MoE FFN: same as gpu_forward_moe_ffn but uses
// native matvec pipelines instead of dp4a. Skips Q8_1 quantization.
// Works for ALL quant types including IQ types.
int gpu_forward_moe_ffn_native(const GpuMoEConf* mc);

// Batched native MoE matvec: dispatches for all n_used experts in one call.
// Uses float input (not Q8_1). Supports all quant types.
int gpu_moe_matvec_native(GpuBuf out_buf, GpuBuf weights_buf,
                          GpuBuf x_buf, GpuBuf indices_buf,
                          int rows, int cols, int qtype,
                          int expert_stride, int base_offset,
                          int shared_input, int n_used);

// IQ lookup table management for I-quant GPU matvec.
// Call once after gpu_init to register grid table buffers.
// iq1s_buf: iq1s_grid[2048] as uint32[4096]
// iq2xxs_buf: iq2xxs_grid[256] as uint32[512] + ksigns[128] as uint32[32]
// iq2s_buf: iq2s_grid[1024] as uint32[2048]
int gpu_set_iq_tables(GpuBuf iq1s_buf, GpuBuf iq2xxs_buf, GpuBuf iq2s_buf, GpuBuf iq3xxs_buf, GpuBuf iq3s_buf);

#ifdef __cplusplus
}
#endif

#endif // DLGO_VULKAN_GPU_H
