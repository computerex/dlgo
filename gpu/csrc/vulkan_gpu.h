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
    PIPE_CONV2D_F32,
    PIPE_GROUP_NORM,
    PIPE_SILU,
    PIPE_UPSAMPLE_NEAREST,
    PIPE_SPATIAL_ATTENTION,
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

// Batch matrix-vector: out[p*rows+r] = dot(W[r,:], x[p*cols...]) for each position p
int gpu_batch_matvec(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                     int rows, int cols, int npos, int qtype);

// Element-wise operations
int gpu_rmsnorm(GpuBuf out_buf, GpuBuf x_buf, GpuBuf weight_buf, int n, float eps);
int gpu_layernorm(GpuBuf out_buf, GpuBuf x_buf, GpuBuf weight_buf, GpuBuf bias_buf, int n, float eps);
int gpu_rmsnorm_heads(GpuBuf data_buf, GpuBuf weight_buf, int num_heads, int head_dim, float eps);
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
                 int n_experts, int k, int gating_func,
                 int weights_norm, float weights_scale);
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
                            int kv_dim, int start_seq_len, float scale, float softcap, int npos);

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
int gpu_ssm_preprocess(GpuBuf alpha, GpuBuf beta, GpuBuf ssma, GpuBuf dt_bias,
                       GpuBuf qkv, int num_heads, int head_k_dim, int key_dim,
                       float rms_eps, int has_dt_bias);
int gpu_ssm_delta_rule(GpuBuf state, GpuBuf qkv, GpuBuf alpha, GpuBuf beta,
                       GpuBuf y, int num_heads, int head_k_dim, int head_v_dim, int key_dim);
int gpu_ssm_norm_gate(GpuBuf y, GpuBuf z, GpuBuf norm_w,
                      int num_heads, int head_v_dim, float eps);

// GatedQ attention operations
int gpu_deinterleave_qgate(GpuBuf qfull, GpuBuf q, GpuBuf qgate,
                           int num_heads, int head_dim);
int gpu_sigmoid_gate(GpuBuf out_buf, GpuBuf gate_buf, int n);

// Fused MoE FFN: runs entire MoE FFN in a single CGo call.
// Eliminates ~18 CGo calls per layer → 1 call.
typedef struct {
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
} GpuMoEConf;

int gpu_forward_moe_ffn(const GpuMoEConf* mc);

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
