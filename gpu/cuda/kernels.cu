// dlgo CUDA kernels.
//
// Compiled to a cubin by build_cuda.ps1 (nvcc -cubin -arch=sm_89) and loaded at
// run time with cuModuleLoadData. Names use extern "C" so cuModuleGetFunction
// can find them by their plain names.
//
// Semantics mirror the GLSL compute shaders / ops package so results can be
// compared against the Vulkan backend.

#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

// ---------------------------------------------------------------------------
// vector_add: out[i] = a[i] + b[i]   (smoke-test kernel)
// ---------------------------------------------------------------------------
extern "C" __global__ void dl_vector_add(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ out,
                                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// kv_store_f16: quantize float32 K/V to packed half2 (uint32) in the KV cache.
// Exact port of gpu/shaders/kv_store_f16.comp.
// ---------------------------------------------------------------------------
extern "C" __global__ void dl_kv_store_f16(const float* __restrict__ k_in,
                                           const float* __restrict__ v_in,
                                           unsigned int* __restrict__ k_out,
                                           unsigned int* __restrict__ v_out,
                                           int pos, int kv_dim, int ring_cap,
                                           const int* __restrict__ sc) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_dim = (unsigned int)kv_dim >> 1;
    if (tid >= half_dim) return;

    unsigned int src = tid << 1;
    if (sc) pos = sc[0];        // graph replay: position comes from device memory
    unsigned int row = (unsigned int)pos;
    if (ring_cap > 0) row = row % (unsigned int)ring_cap;
    unsigned int dst = row * half_dim + tid;

    unsigned short klo = __half_as_ushort(__float2half(k_in[src]));
    unsigned short khi = __half_as_ushort(__float2half(k_in[src + 1]));
    k_out[dst] = (unsigned int)klo | ((unsigned int)khi << 16);

    unsigned short vlo = __half_as_ushort(__float2half(v_in[src]));
    unsigned short vhi = __half_as_ushort(__float2half(v_in[src + 1]));
    v_out[dst] = (unsigned int)vlo | ((unsigned int)vhi << 16);
}

// ---------------------------------------------------------------------------
// rmsnorm: out[i] = x[i] * w[i] * rsqrt(mean(x^2) + eps)   (one block per row)
// ---------------------------------------------------------------------------
extern "C" __global__ void dl_rmsnorm_f32(const float* __restrict__ x,
                                          const float* __restrict__ w,
                                          float* __restrict__ out,
                                          int n, float eps) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = x[i];
        acc += v * v;
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float mean = sdata[0] / (float)n;
    float scale = rsqrtf(mean + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = x[i] * scale * (w ? w[i] : 1.0f);
    }
}

// ---------------------------------------------------------------------------
// layernorm: out[i] = (x[i]-mean) * rsqrt(var+eps) * w[i] + b[i]
// ---------------------------------------------------------------------------
extern "C" __global__ void dl_layernorm_f32(const float* __restrict__ x,
                                            const float* __restrict__ w,
                                            const float* __restrict__ b,
                                            float* __restrict__ out,
                                            int n, float eps) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    float s = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) s += x[i];
    sdata[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float mean = sdata[0] / (float)n;
    float v = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float d = x[i] - mean;
        v += d * d;
    }
    __syncthreads();
    sdata[tid] = v;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float rstd = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        float r = (x[i] - mean) * rstd * (w ? w[i] : 1.0f);
        out[i] = r + (b ? b[i] : 0.0f);
    }
}

// ---------------------------------------------------------------------------
// rmsnorm_heads: per-head RMSNorm, in place. grid = num_heads, one block/head.
//   data[head*hd + i] *= w[i] * rsqrt(mean(data[head*hd .. +hd]^2)+eps)
// ---------------------------------------------------------------------------
extern "C" __global__ void dl_rmsnorm_heads(float* __restrict__ data,
                                            const float* __restrict__ w,
                                            int head_dim, float eps) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    int base = blockIdx.x * head_dim;
    float acc = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = data[base + i];
        acc += v * v;
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        data[base + i] = data[base + i] * scale * w[i];
    }
}

// ---------------------------------------------------------------------------
// silu_mul (SwiGLU): out[i] = silu(gate[i]) * up[i]
// ---------------------------------------------------------------------------
extern "C" __global__ void dl_silu_mul(const float* __restrict__ gate,
                                       const float* __restrict__ up,
                                       float* __restrict__ out,
                                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float s = g / (1.0f + expf(-g));
        out[i] = s * up[i];
    }
}

// ---------------------------------------------------------------------------
// add: out[i] = a[i] + b[i]
// ---------------------------------------------------------------------------
extern "C" __global__ void dl_add(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// scale: buf[i] *= s  (in place)
// ---------------------------------------------------------------------------
extern "C" __global__ void dl_scale(float* __restrict__ buf, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] *= s;
}

// ---------------------------------------------------------------------------
// add_rmsnorm: sum[i]=a[i]+b[i]; out[i]=sum[i]*w[i]*rsqrt(mean(sum^2)+eps)
// One block; writes both sumOut and normOut.
// ---------------------------------------------------------------------------
extern "C" __global__ void dl_add_rmsnorm(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          const float* __restrict__ w,
                                          float* __restrict__ sum,
                                          float* __restrict__ out,
                                          int n, float eps) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = a[i] + b[i];
        sum[i] = v;
        acc += v * v;
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float scale = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = sum[i] * scale * w[i];
    }
}

// ---------------------------------------------------------------------------
// rope: rotary position embedding (standard interleaved or neox split-half).
// grid = max(num_heads, num_kv_heads); block = half_dim threads.
//   table_base = actual_pos*half_dim + tid ; cos/sin tables are [maxpos][half_dim]
// ---------------------------------------------------------------------------
extern "C" __global__ void dl_rope(float* __restrict__ q,
                                   float* __restrict__ k,
                                   const float* __restrict__ cosT,
                                   const float* __restrict__ sinT,
                                   int num_heads, int num_kv_heads,
                                   int head_dim, int rope_dim,
                                   int pos, int neox,
                                   const int* __restrict__ sc) {
    int head = blockIdx.x;
    int tid = threadIdx.x;
    if (sc) pos = sc[0];        // graph replay: position comes from device memory
    int rd = rope_dim > 0 ? rope_dim : head_dim;
    int half_dim = rd >> 1;
    if (tid >= half_dim) return;

    int table_base = pos * half_dim + tid;
    float c = cosT[table_base];
    float s = sinT[table_base];

    if (head < num_heads) {
        float* base = q + (size_t)head * head_dim;
        if (neox) {
            int i0 = tid, i1 = tid + half_dim;
            float a = base[i0], b = base[i1];
            base[i0] = a * c - b * s;
            base[i1] = a * s + b * c;
        } else {
            int i0 = 2 * tid, i1 = 2 * tid + 1;
            float a = base[i0], b = base[i1];
            base[i0] = a * c - b * s;
            base[i1] = a * s + b * c;
        }
    }
    if (head < num_kv_heads) {
        float* base = k + (size_t)head * head_dim;
        if (neox) {
            int i0 = tid, i1 = tid + half_dim;
            float a = base[i0], b = base[i1];
            base[i0] = a * c - b * s;
            base[i1] = a * s + b * c;
        } else {
            int i0 = 2 * tid, i1 = 2 * tid + 1;
            float a = base[i0], b = base[i1];
            base[i0] = a * c - b * s;
            base[i1] = a * s + b * c;
        }
    }
}

// ---------------------------------------------------------------------------
// Block reduction helper: reduce val across `nthreads` (== blockDim.x) and
// leave the total in *result (valid on thread 0). Shared buffer is supplied by
// the caller so multiple reduce phases can reuse it.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_sum(float val, float* sdata) {
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    return sdata[0];
}

// ---------------------------------------------------------------------------
// quantized / typed GEMV: out[row] = sum_c dequant(W[row,c]) * x[c]
// Launch: grid.x = ceil(rows/8), block = 256 (8 warps -> 8 rows/block).
// One warp per output row: coalesced weight/x reads, warp-shuffle reduce, no
// __syncthreads. This is the decode hot path and must saturate DRAM bandwidth.
// ---------------------------------------------------------------------------
#define GEMV_ROWS_PER_BLOCK 16

__device__ __forceinline__ void q4k_load_scales(const unsigned char* blk, float sc[8], float mn[8]) {
    float d = __half2float(*(const __half*)(blk + 0));
    float dmin = __half2float(*(const __half*)(blk + 2));
    const unsigned char* s = blk + 4;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        sc[i] = d * (float)(s[i] & 0x3F);
        mn[i] = dmin * (float)(s[4 + i] & 0x3F);
        int scHi = (s[i] >> 6) & 0x03;
        int scLo = s[8 + i] & 0x0F;
        int mnHi = (s[4 + i] >> 6) & 0x03;
        int mnLo = (s[8 + i] >> 4) & 0x0F;
        sc[4 + i] = d * (float)(scLo | (scHi << 4));
        mn[4 + i] = dmin * (float)(mnLo | (mnHi << 4));
    }
}

// Compute the two Q4_K scale pairs a lane needs (low/high nibble sub-block) for
// one block, given grp = (4*lane)/32. Kept as a plain device function.
__device__ __forceinline__ void q4k_pair_scales(const unsigned char* blk, int grp,
                                                float& scL, float& mnL, float& scH, float& mnH) {
    float d = __half2float(*(const __half*)(blk + 0));
    float dmin = __half2float(*(const __half*)(blk + 2));
    const unsigned char* s = blk + 4;
    int iL = grp * 2, iH = grp * 2 + 1;
    if (iL < 4) { scL = d * (float)(s[iL] & 0x3F); mnL = dmin * (float)(s[4 + iL] & 0x3F); }
    else { int j = iL - 4; scL = d * (float)(((s[j] >> 6 & 3) << 4) | (s[8 + j] & 0x0F));
           mnL = dmin * (float)((((s[4 + j] >> 6) & 3) << 4) | ((s[8 + j] >> 4) & 0x0F)); }
    if (iH < 4) { scH = d * (float)(s[iH] & 0x3F); mnH = dmin * (float)(s[4 + iH] & 0x3F); }
    else { int j = iH - 4; scH = d * (float)(((s[j] >> 6 & 3) << 4) | (s[8 + j] & 0x0F));
           mnH = dmin * (float)((((s[4 + j] >> 6) & 3) << 4) | ((s[8 + j] >> 4) & 0x0F)); }
}

// Q4_K GEMV: one warp per row. Lane reads qs bytes [4*lane, 4*lane+4) (one
// uint32); 32 lanes cover the full 128-byte qs. Those 4 bytes yield 4 low-nibble
// values at consecutive x positions (grp*64+rem0..+3) and 4 high-nibble values
// (+32..+35), so x is pulled in with two coalesced float4 loads and only 2 scale
// factors are needed per lane per block.
extern "C" __global__ void dl_gemv_q4k(const unsigned char* __restrict__ W,
                                       const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int rows, int cols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    int nblk = cols >> 8;
    const size_t rowbase = (size_t)row * nblk * 144;
    int j0 = 4 * lane;
    int grp = j0 >> 5;          // 0..3
    int rem0 = j0 & 31;         // 0,4,...,28
    float acc = 0.0f;
    for (int b = 0; b < nblk; b++) {
        const unsigned char* blk = W + rowbase + (size_t)b * 144;
        float scL, mnL, scH, mnH;
        q4k_pair_scales(blk, grp, scL, mnL, scH, mnH);
        unsigned int qw = *((const unsigned int*)(blk + 16 + j0));
        const float* xb = x + b * 256;
        float4 xl = *((const float4*)(xb + grp * 64 + rem0));
        float4 xh = *((const float4*)(xb + grp * 64 + 32 + rem0));
        int lo0 = qw & 0xF, lo1 = (qw >> 8) & 0xF, lo2 = (qw >> 16) & 0xF, lo3 = (qw >> 24) & 0xF;
        int hi0 = (qw >> 4) & 0xF, hi1 = (qw >> 12) & 0xF, hi2 = (qw >> 20) & 0xF, hi3 = (qw >> 28) & 0xF;
        acc += (scL * (float)lo0 - mnL) * xl.x;
        acc += (scL * (float)lo1 - mnL) * xl.y;
        acc += (scL * (float)lo2 - mnL) * xl.z;
        acc += (scL * (float)lo3 - mnL) * xl.w;
        acc += (scH * (float)hi0 - mnH) * xh.x;
        acc += (scH * (float)hi1 - mnH) * xh.y;
        acc += (scH * (float)hi2 - mnH) * xh.z;
        acc += (scH * (float)hi3 - mnH) * xh.w;
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

extern "C" __global__ void dl_gemv_q6k(const unsigned char* __restrict__ W,
                                       const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int rows, int cols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    int nblk = cols >> 8;
    const size_t rowbase = (size_t)row * nblk * 210;
    float acc = 0.0f;
    for (int b = 0; b < nblk; b++) {
        const unsigned char* blk = W + rowbase + (size_t)b * 210;
        const unsigned char* ql = blk;          // 128 bytes
        const unsigned char* qh = blk + 128;    // 64 bytes
        const signed char* scal = (const signed char*)(blk + 192);
        float d = __half2float(*(const __half*)(blk + 208));
        const float* xb = x + b * 256;
        int is = 0;
#pragma unroll
        for (int n128 = 0; n128 < 2; n128++) {
            int qlA = ql[n128 * 64 + lane];
            int qlB = ql[n128 * 64 + lane + 32];
            int qhb = qh[n128 * 32 + lane];
            int sidx = 8 * n128 + (lane >> 4);
            int pos0 = n128 * 128 + lane;
            float ds = d;
            int q0 = (qlA & 0x0F) | ((qhb & 3) << 4);
            int q1 = (qlB & 0x0F) | (((qhb >> 2) & 3) << 4);
            int q2 = (qlA >> 4) | (((qhb >> 4) & 3) << 4);
            int q3 = (qlB >> 4) | (((qhb >> 6) & 3) << 4);
            acc += ds * (float)scal[sidx + 0] * (float)(q0 - 32) * xb[pos0 + 0];
            acc += ds * (float)scal[sidx + 2] * (float)(q1 - 32) * xb[pos0 + 32];
            acc += ds * (float)scal[sidx + 4] * (float)(q2 - 32) * xb[pos0 + 64];
            acc += ds * (float)scal[sidx + 6] * (float)(q3 - 32) * xb[pos0 + 96];
        }
        (void)is;
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

extern "C" __global__ void dl_gemv_f32(const float* __restrict__ W,
                                       const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int rows, int cols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    const float* wrow = W + (size_t)row * cols;
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) acc += wrow[c] * x[c];
#pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

extern "C" __global__ void dl_gemv_f16(const __half* __restrict__ W,
                                       const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int rows, int cols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    const __half* wrow = W + (size_t)row * cols;
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) acc += __half2float(wrow[c]) * x[c];
#pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// attention_f16: single-query (decode) causal GQA attention over an f16 KV
// cache. KV cache layout: contiguous __half[kv_dim] per position row (the
// half2 packing in kv_store_f16 is byte-equivalent). One warp per head; each
// lane owns head_dim/32 output dims. Online (flash) softmax, no barriers.
//   grid.x = ceil(num_heads/4); block = 128 (4 warps -> 4 heads)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// attention_f16: single-query (decode) causal GQA attention over an f16 KV cache.
//
// Two-pass, block-per-head design (replaces a warp-per-head kernel that only
// used ~8 of 66 SMs and did a serial shuffle-reduce per key -> it dominated
// decode time). grid.x = num_heads, block = 256 threads:
//   pass 1: each thread scores a strided subset of keys (q . k over head_dim)
//   block reduce max -> block reduce exp-sum -> softmax weights in shared
//   pass 2: each thread owns output dims, sums p[k]*V[k] with coalesced V reads.
// KV layout: contiguous __half[kv_dim] per position row (byte-equal to the
// half2 packing of kv_store_f16). Win_start/kv_cap(ring)/softcap supported.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float attn_block_max(float v, float* red, int nt, int tid) {
    red[tid] = v;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float r = red[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float attn_block_sum(float v, float* red, int nt, int tid) {
    red[tid] = v;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float r = red[0];
    __syncthreads();
    return r;
}

#define ATTN_MAXSEQ 8192

extern "C" __global__ void dl_attention_f16(const float* __restrict__ q,
                                            const __half* __restrict__ kc,
                                            const __half* __restrict__ vc,
                                            float* __restrict__ out,
                                            int num_heads, int num_kv_heads,
                                            int head_dim, int kv_dim,
                                            int seq_len, float scale,
                                            float softcap, int win_start,
                                            int kv_cap) {
    __shared__ float sp[ATTN_MAXSEQ];
    __shared__ float red[256];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    int kv_mul = num_heads / num_kv_heads;
    int kv_head = head / kv_mul;
    const float* qh = q + head * head_dim;
    int n = seq_len - win_start;
    if (n <= 0) {
        for (int d = tid; d < head_dim; d += nt) out[head * head_dim + d] = 0.0f;
        return;
    }
    if (n > ATTN_MAXSEQ) n = ATTN_MAXSEQ;

    // Pass 1: scores.
    float lmax = -1e30f;
    for (int i = tid; i < n; i += nt) {
        int t = win_start + i;
        int row = kv_cap > 0 ? (t % kv_cap) : t;
        const __half* krow = kc + (size_t)row * kv_dim + kv_head * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) s += qh[d] * __half2float(krow[d]);
        s *= scale;
        if (softcap > 0.0f) s = softcap * tanhf(s / softcap);
        sp[i] = s;
        if (s > lmax) lmax = s;
    }
    lmax = attn_block_max(lmax, red, nt, tid);

    // Pass 2: exp + sum.
    float lsum = 0.0f;
    for (int i = tid; i < n; i += nt) {
        float e = __expf(sp[i] - lmax);
        sp[i] = e;
        lsum += e;
    }
    lsum = attn_block_sum(lsum, red, nt, tid);
    float inv = lsum > 0.0f ? 1.0f / lsum : 0.0f;

    // Pass 3: PV — each thread owns output dims, coalesced V reads.
    for (int d = tid; d < head_dim; d += nt) {
        float acc = 0.0f;
        for (int i = 0; i < n; i++) {
            int t = win_start + i;
            int row = kv_cap > 0 ? (t % kv_cap) : t;
            acc += sp[i] * __half2float(vc[(size_t)row * kv_dim + kv_head * head_dim + d]);
        }
        out[head * head_dim + d] = acc * inv;
    }
}

// ---------------------------------------------------------------------------
// attention_split_f16: split-KV, GQA-folded causal decode attention (+ combine).
//
// dl_attention_f16 above runs one block per *query* head, so an f16 step pays
// the KV traffic once per query head (with GQA 4:1 that is 4x the useful bytes)
// and launches only num_heads blocks. This version folds the whole GQA group of
// one KV head into a single block and splits the key range across blocks:
//
//   grid = (num_kv_heads, num_splits), block = 256 threads
//
// Each block owns the kv_mul query heads that share KV head `blockIdx.x` and the
// key window [i0, i1) of the visible range. It walks that window in 256-key
// tiles with an online (flash) softmax — scores for all kv_mul heads come from
// one K row read, and the P*V pass reads each V row once for all kv_mul heads —
// then writes a partial (m, l, acc) triple per query head. dl_attention_combine
// merges the split partials into the final output.
//
// Requirements (checked on the Go side): num_heads % num_kv_heads == 0,
// kv_mul <= 8, 256 % kv_mul == 0, head_dim even, kv_mul*head_dim <= 512.
// ---------------------------------------------------------------------------
#define ATTN_TILE 256
#define ATTN_MAXSPLITS 32

__device__ __forceinline__ float attn_warp_max(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float attn_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

extern "C" __global__ void dl_attention_split_f16(
        const float* __restrict__ q,
        const __half* __restrict__ kc,
        const __half* __restrict__ vc,
        float* __restrict__ part,
        int num_heads, int num_kv_heads, int head_dim, int kv_dim,
        int seq_len, float scale, float softcap, int win_start,
        int kv_cap, int num_splits,
        const int* __restrict__ sc) {
    extern __shared__ float sh[];
    const int nt = blockDim.x;              // 256
    const int tid = threadIdx.x;
    const int kh = blockIdx.x;              // KV head
    const int sid = blockIdx.y;             // key-range split
    if (sc) { seq_len = sc[1]; win_start = sc[2]; kv_cap = sc[3]; }  // graph replay
    const int kv_mul = num_heads / num_kv_heads;
    const int tph = nt / kv_mul;            // threads scoring one query head
    const int nw = tph >> 5;                // warps per query head
    const int hp = tid / tph;               // own query head inside the group
    const int lt = tid - hp * tph;          // index inside the head's threads
    const int lane = tid & 31;
    const int wrp = lt >> 5;

    float* sq = sh;                          // kv_mul*head_dim   (Q of the group)
    float* sp = sq + kv_mul * head_dim;      // kv_mul*ATTN_TILE  (scores / probs)
    float* sww = sp + kv_mul * ATTN_TILE;    // kv_mul*nw         (warp partials)
    float* smm = sww + kv_mul * nw;          // running max
    float* sml = smm + kv_mul;               // running sum
    float* smt = sml + kv_mul;               // tile max (then new running max)
    float* sma = smt + kv_mul;               // tile sum
    float* sfa = sma + kv_mul;               // rescale factor

    const int npair = kv_mul * head_dim;
    for (int i = tid; i < npair; i += nt) sq[i] = q[kh * kv_mul * head_dim + i];
    if (tid < kv_mul) { smm[tid] = -INFINITY; sml[tid] = 0.0f; }
    __syncthreads();

    // Output accumulators: pair p = head*head_dim + dim, owned by thread p and
    // p + nt (npair <= 2*nt, so at most two pairs per thread). When npair < nt
    // many threads own no pair at all - they must not touch sfa[]/sp[] with an
    // out-of-range head index (that reads outside the block's shared window and
    // faults the context).
    const int p0 = tid, p1 = tid + nt;
    const bool one0 = (p0 < npair);
    const bool two = (p1 < npair);
    const int h0 = one0 ? p0 / head_dim : 0;
    const int d0 = one0 ? p0 - h0 * head_dim : 0;
    const int h1 = two ? p1 / head_dim : 0;
    const int d1 = two ? p1 - h1 * head_dim : 0;
    float acc0 = 0.0f, acc1 = 0.0f;

    int n = seq_len - win_start;
    if (n < 0) n = 0;
    const int per = (n + num_splits - 1) / num_splits;
    int i0 = sid * per;
    if (i0 > n) i0 = n;
    int i1 = i0 + per;
    if (i1 > n) i1 = n;

    const int vb = kh * head_dim;

    for (int tile = i0; tile < i1; tile += ATTN_TILE) {
        const int ntk = min(ATTN_TILE, i1 - tile);

        // -- stage A: scores q.k for this thread's query head -----------------
        float lmax = -INFINITY;
        for (int j = lt; j < ntk; j += tph) {
            const int t = win_start + tile + j;
            const int row = kv_cap > 0 ? (t % kv_cap) : t;
            const __half* krow = kc + (size_t)row * kv_dim + vb;
            const float* qh = sq + hp * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; d += 2) {
                float2 f2 = __half22float2(*(const __half2*)(krow + d));
                s = fmaf(qh[d], f2.x, s);
                s = fmaf(qh[d + 1], f2.y, s);
            }
            s *= scale;
            if (softcap > 0.0f) s = softcap * tanhf(s / softcap);
            sp[hp * ATTN_TILE + j] = s;
            if (s > lmax) lmax = s;
        }

        // -- stage B: tile max over the head's thread group -------------------
        lmax = attn_warp_max(lmax);
        if (lane == 0) sww[hp * nw + wrp] = lmax;
        __syncthreads();
        if (tid == hp * tph) {
            float v = -INFINITY;
            for (int i = 0; i < nw; i++) v = fmaxf(v, sww[hp * nw + i]);
            smt[hp] = v;
        }
        __syncthreads();

        // -- stage C: running max, rescale factor for the accumulators --------
        if (tid < kv_mul) {
            float mprev = smm[tid];
            float mnew = fmaxf(mprev, smt[tid]);
            smm[tid] = mnew;
            float f = isinf(mprev) ? 1.0f : __expf(mprev - mnew);
            sfa[tid] = f;
            sml[tid] *= f;
            smt[tid] = mnew;   // stage D needs the new running max
        }
        __syncthreads();
        acc0 *= sfa[h0];
        acc1 = two ? acc1 * sfa[h1] : 0.0f;

        // -- stage D: probabilities + tile sum --------------------------------
        {
            float mh = smt[hp];
            float lsum = 0.0f;
            for (int j = lt; j < ntk; j += tph) {
                const int idx = hp * ATTN_TILE + j;
                float e = __expf(sp[idx] - mh);
                sp[idx] = e;
                lsum += e;
            }
            lsum = attn_warp_sum(lsum);
            if (lane == 0) sww[hp * nw + wrp] = lsum;
            __syncthreads();
            if (tid == hp * tph) {
                float v = 0.0f;
                for (int i = 0; i < nw; i++) v += sww[hp * nw + i];
                sma[hp] = v;
            }
            __syncthreads();
            if (tid < kv_mul) sml[tid] += sma[tid];
        }

        // -- stage E: P*V, one V row read per key for all kv_mul heads --------
        {
            const float* ph = sp + h0 * ATTN_TILE;
            const float* pl = sp + h1 * ATTN_TILE;
            int r = kv_cap > 0 ? ((win_start + tile) % kv_cap) : (win_start + tile);
            const __half* vrow = vc + (size_t)r * kv_dim + vb;
            float s0 = 0.0f, s1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
            int j = 0;
            for (; j + 2 <= ntk; j += 2) {
                s0 = fmaf(ph[j], __half2float(vrow[d0]), s0);
                if (two) u0 = fmaf(pl[j], __half2float(vrow[d1]), u0);
                if (++r == kv_cap) r = 0;
                vrow = vc + (size_t)r * kv_dim + vb;
                s1 = fmaf(ph[j + 1], __half2float(vrow[d0]), s1);
                if (two) u1 = fmaf(pl[j + 1], __half2float(vrow[d1]), u1);
                if (++r == kv_cap) r = 0;
                vrow = vc + (size_t)r * kv_dim + vb;
            }
            if (j < ntk) {
                s0 = fmaf(ph[j], __half2float(vrow[d0]), s0);
                if (two) u0 = fmaf(pl[j], __half2float(vrow[d1]), u0);
            }
            if (one0) acc0 += s0 + s1;
            if (two) acc1 += u0 + u1;
        }
    }

    // -- write the split partial ---------------------------------------------
    {
        const int stride = kv_mul * (head_dim + 2);
        float* base = part + ((size_t)kh * num_splits + sid) * stride;
        if (tid < kv_mul) {
            base[tid] = smm[tid];
            base[kv_mul + tid] = sml[tid];
        }
        if (tid < npair) base[2 * kv_mul + tid] = acc0;
        if (two) base[2 * kv_mul + p1] = acc1;
    }
}

// ---------------------------------------------------------------------------
// attention_combine: merge the split partials of dl_attention_split_f16.
// grid.x = num_heads, block = head_dim-ish. Partial layout per (kv_head, split)
// is [m(kv_mul)][l(kv_mul)][acc(kv_mul*head_dim)].
// ---------------------------------------------------------------------------
extern "C" __global__ void dl_attention_combine(const float* __restrict__ part,
                                                float* __restrict__ out,
                                                int num_heads, int num_kv_heads,
                                                int head_dim, int num_splits) {
    __shared__ float sw[ATTN_MAXSPLITS];
    __shared__ float sden;
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int kv_mul = num_heads / num_kv_heads;
    const int kh = h / kv_mul;
    const int gj = h - kh * kv_mul;
    const int stride = kv_mul * (head_dim + 2);
    const float* base = part + (size_t)kh * num_splits * stride;

    float M = -INFINITY;
    for (int s = 0; s < num_splits; s++) {
        float m = base[s * stride + gj];
        if (m > M) M = m;
    }
    if (M == -INFINITY) {
        for (int d = tid; d < head_dim; d += nt) out[h * head_dim + d] = 0.0f;
        return;
    }
    if (tid == 0) {
        float den = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            float m = base[s * stride + gj];
            if (m == -INFINITY) { sw[s] = 0.0f; continue; }
            float w = __expf(m - M);
            sw[s] = w;
            den += w * base[s * stride + kv_mul + gj];
        }
        sden = den > 0.0f ? 1.0f / den : 0.0f;
    }
    __syncthreads();
    for (int d = tid; d < head_dim; d += nt) {
        float acc = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            if (sw[s] == 0.0f) continue;
            acc += sw[s] * base[s * stride + 2 * kv_mul + gj * head_dim + d];
        }
        out[h * head_dim + d] = acc * sden;
    }
}

// ---------------------------------------------------------------------------
// softmax_row: in-place softmax over one row of length n (kept for parity).
// ---------------------------------------------------------------------------
extern "C" __global__ void dl_softmax_row(float* __restrict__ x, int n) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;

    float m = -INFINITY;
    for (int i = tid; i < n; i += blockDim.x) m = fmaxf(m, x[i]);
    sdata[tid] = m;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    float rowmax = sdata[0];

    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) acc += expf(x[i] - rowmax);
    __syncthreads();
    sdata[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float inv = 1.0f / sdata[0];
    for (int i = tid; i < n; i += blockDim.x) x[i] = expf(x[i] - rowmax) * inv;
}

// ---------------------------------------------------------------------------
// Diagnostic probes (kept; harmless).
// ---------------------------------------------------------------------------
extern "C" __global__ void dl_probe7(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     unsigned int* __restrict__ c,
                                     unsigned int* __restrict__ d,
                                     int n1, int n2, int n3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        c[0] = (unsigned)n1 + (unsigned)n2 + (unsigned)n3
             + (a ? 1u : 0u) + (b ? 1u : 0u) + (d ? 1u : 0u);
    }
}

extern "C" __global__ void dl_half_probe(const float* __restrict__ in,
                                         unsigned int* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (unsigned int)__half_as_ushort(__float2half(in[i]));
}

// ===========================================================================
// int8 dp4a quantized GEMV (Q4_K). This is the decode hot path: it mirrors the
// llama.cpp / Vulkan dp4a approach that makes the Vulkan backend fast.
//
// For a 256-value Q4_K block, dequant value_i = sc_sub*q_i - mn_sub, where the
// 8 sub-blocks are 32 *contiguous* positions each (sub s = positions s*32..s*32+31,
// scale index s). Quantizing x to int8 per 32-group (scale dx_sub, integer sum
// sumxq_sub) gives, per sub-block:
//     sum_i (sc*q_i - mn) * x_i  ≈  dx_sub * ( sc_sub*dot(q,xq) - mn_sub*sumxq )
// dot(q,xq) is computed 4-wide with __dp4a.
// ===========================================================================

// Quantize x (length n, multiple of 32) into int8 xq[cols], with per-32-group
// scale dx[cols/32] and integer sum sumxq[cols/32]. One warp per 32-group.
extern "C" __global__ void dl_quantize_x32(const float* __restrict__ x, int n,
                                           signed char* __restrict__ xq,
                                           float* __restrict__ dx,
                                           int* __restrict__ sumxq) {
    int g = blockIdx.x;
    int lane = threadIdx.x;
    if (g * 32 >= n) return;
    float v = x[g * 32 + lane];

    float a = fabsf(v);
#pragma unroll
    for (int off = 16; off; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    float scale = a / 127.0f;
    if (scale == 0.0f) scale = 1.0f;

    int q = (int)lrintf(v / scale);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    xq[g * 32 + lane] = (signed char)q;

    int s = q;
#pragma unroll
    for (int off = 16; off; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    if (lane == 0) {
        dx[g] = scale;
        sumxq[g] = s;
    }
}

// Q4_K dp4a GEMV: out[row] = sum_c dequantQ4K(W[row,c]) * x[c].
// One warp per row. grid.x = ceil(rows/8), block = 256 (8 warps). Each lane, per
// iteration, owns one 4-position group of one sub-block; 8 lanes cover a
// sub-block (8*4=32). intdot is reduced within 8 lanes, then scaled per sub-block
// and the 8 sub-block contributions are reduced across the warp.
extern "C" __global__ void dl_gemv_q4k_dp4a(const unsigned char* __restrict__ W,
                                            const signed char* __restrict__ xq,
                                            const float* __restrict__ dx,
                                            const int* __restrict__ sumxq,
                                            float* __restrict__ out,
                                            int rows, int cols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    int nblk = cols >> 8;
    const size_t rowbase = (size_t)row * nblk * 144;
    int g = lane & 7;          // 0..7 : which 4-position group within a sub-block
    int subBase = lane >> 3;   // 0..3 : base sub-block for this lane's iteration
    float myV = 0.0f;

    for (int b = 0; b < nblk; b++) {
        const unsigned char* blk = W + rowbase + (size_t)b * 144;
        float sc[8], mn[8];
        q4k_load_scales(blk, sc, mn);
        const unsigned char* qs = blk + 16;
        const signed char* xqx = xq + b * 256;
        const float* dxb = dx + b * 8;
        const int* sxb = sumxq + b * 8;
#pragma unroll
        for (int it = 0; it < 2; it++) {
            int s = subBase + it * 4;      // sub-block 0..7
            int grp = s >> 1;
            int hi = s & 1;
            int pos = g * 4;
            unsigned int w4 = *((const unsigned int*)(qs + grp * 32 + pos));
            unsigned int qb = hi ? ((w4 >> 4) & 0x0F0F0F0Fu) : (w4 & 0x0F0F0F0Fu);
            int xqi = *((const int*)(xqx + s * 32 + pos));
            int d = __dp4a((int)qb, xqi, 0);
            d += __shfl_down_sync(0xffffffffu, d, 4, 8);
            d += __shfl_down_sync(0xffffffffu, d, 2, 8);
            d += __shfl_down_sync(0xffffffffu, d, 1, 8);
            if (g == 0) {
                myV += dxb[s] * (sc[s] * (float)d - mn[s] * (float)sxb[s]);
            }
        }
    }

#pragma unroll
    for (int off = 16; off; off >>= 1) myV += __shfl_xor_sync(0xffffffffu, myV, off);
    if (lane == 0) out[row] = myV;
}

// ===========================================================================
// Block-cooperative Q4_K dp4a GEMV (the "llama.cpp structure").
//
// One CUDA block computes ONE output row, so the whole block (blockDim.x
// threads) streams that row's column-blocks together -> far more memory-level
// parallelism than warp-per-row, and activations are Q8 int8 (from
// dl_quantize_x32) so the redundant activation re-reads are 4x smaller.
//
// Each thread owns WHOLE 32-value sub-blocks (no intra-sub-block reduction):
//   out[row] += sum_sub dx_sub * ( sc_sub * dot(q,xq) - mn_sub * sumxq_sub )
// where dot(q,xq) is 8x __dp4a over the 32 values. One block reduction at end.
//   grid.x = rows ; block = 256 threads.
// ===========================================================================
extern "C" __global__ void dl_gemv_q4k_bc(const unsigned char* __restrict__ W,
                                          const signed char* __restrict__ xq,
                                          const float* __restrict__ dx,
                                          const int* __restrict__ sumxq,
                                          float* __restrict__ out,
                                          int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    int nblk = cols >> 8;
    int nsub = nblk * 8;
    const size_t rowbase = (size_t)row * nblk * 144;
    float partial = 0.0f;

    for (int s = tid; s < nsub; s += nt) {
        int b = s >> 3;
        int sub = s & 7;
        const unsigned char* blk = W + rowbase + (size_t)b * 144;
        float scL, mnL, scH, mnH;
        q4k_pair_scales(blk, sub >> 1, scL, mnL, scH, mnH);
        int hi = sub & 1;
        float scv = hi ? scH : scL;
        float mnv = hi ? mnH : mnL;
        const unsigned char* qs = blk + 16 + (sub >> 1) * 32;
        const signed char* xqb = xq + b * 256 + sub * 32;
        int dot = 0;
#pragma unroll
        for (int g = 0; g < 8; g++) {
            unsigned int w = *((const unsigned int*)(qs + g * 4));
            int qp = hi ? (int)((w >> 4) & 0x0F0F0F0Fu) : (int)(w & 0x0F0F0F0Fu);
            int u = *((const int*)(xqb + g * 4));
            dot = __dp4a(qp, u, dot);
        }
        partial += dx[b * 8 + sub] * (scv * (float)dot - mnv * (float)sumxq[b * 8 + sub]);
    }

    __shared__ float red[256];
    red[tid] = partial;
    __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    if (tid == 0) out[row] = red[0];
}

// ---------------------------------------------------------------------------
// Canonical Affine4 / Affine8 weight formats (quant/affine4.go, affine8.go).
//   one block = 32 weights, w = d*q + m, q : uint
//   Affine4: 4 x uint32 codes (16 B) + f16 d + f16 m = 20 B  (0.625 B/weight)
//   Affine8: 8 x uint32 codes (32 B) + f16 d + f16 m = 36 B  (1.125 B/weight)
// Every source quantization scheme is repacked into one of these two, so a single
// GEMV and a single GEMM cover the whole model.
// ---------------------------------------------------------------------------

// Dequantize the 8 codes that thread `part` (0..3) owns of one 32-weight block.
template <bool AFF8>
__device__ __forceinline__ void dl_affine_part(const unsigned char* __restrict__ blk,
                                               int part, __half* __restrict__ dst) {
    const int boff = AFF8 ? 32 : 16;
    const float d = __half2float(*(const __half*)(blk + boff));
    const float m = __half2float(*(const __half*)(blk + boff + 2));
    const unsigned int w = AFF8 ? 0u : *(const unsigned int*)(blk + 4 * part);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int q = AFF8 ? (int)blk[part * 8 + i] : (int)((w >> (4 * i)) & 0xFu);
        dst[i] = __float2half(d * (float)q + m);
    }
}

// f32 -> f16 conversion, used to stage the prefill activation matrix once per
// GEMM instead of re-converting X for every output tile.
extern "C" __global__ void dl_f32_to_half(const float* __restrict__ x,
                                          __half* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

// Affine4 GEMV (single token): one warp per row. Lane (g0, w4) owns 8 consecutive
// codes of group g0, so a warp covers 8 groups = 256 weights per iteration with
// one 4-byte weight load and two coalesced float4 x loads.
extern "C" __global__ void dl_gemv_affine4(const unsigned char* __restrict__ W,
                                           const float* __restrict__ x,
                                           float* __restrict__ out,
                                           int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    const int ng = cols >> 5;
    const size_t rowbase = (size_t)row * ng * 20;
    const int g0 = lane >> 2;
    const int w4 = lane & 3;
    const int e0 = w4 << 3;
    float acc = 0.0f;
    for (int g = g0; g < ng; g += 8) {
        const unsigned char* blk = W + rowbase + (size_t)g * 20;
        const float d = __half2float(*(const __half*)(blk + 16));
        const float m = __half2float(*(const __half*)(blk + 18));
        const unsigned int qw = *(const unsigned int*)(blk + 4 * w4);
        const float4 xa = *(const float4*)(x + (size_t)g * 32 + e0);
        const float4 xb = *(const float4*)(x + (size_t)g * 32 + e0 + 4);
        acc += (d * (float)(qw & 0xFu) + m) * xa.x;
        acc += (d * (float)((qw >> 4) & 0xFu) + m) * xa.y;
        acc += (d * (float)((qw >> 8) & 0xFu) + m) * xa.z;
        acc += (d * (float)((qw >> 12) & 0xFu) + m) * xa.w;
        acc += (d * (float)((qw >> 16) & 0xFu) + m) * xb.x;
        acc += (d * (float)((qw >> 20) & 0xFu) + m) * xb.y;
        acc += (d * (float)((qw >> 24) & 0xFu) + m) * xb.z;
        acc += (d * (float)((qw >> 28) & 0xFu) + m) * xb.w;
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

// Affine8 GEMV: same shape as the Affine4 kernel, 8 code bytes per thread.
extern "C" __global__ void dl_gemv_affine8(const unsigned char* __restrict__ W,
                                           const float* __restrict__ x,
                                           float* __restrict__ out,
                                           int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    const int ng = cols >> 5;
    const size_t rowbase = (size_t)row * ng * 36;
    const int g0 = lane >> 2;
    const int w4 = lane & 3;
    const int e0 = w4 << 3;
    float acc = 0.0f;
    for (int g = g0; g < ng; g += 8) {
        const unsigned char* blk = W + rowbase + (size_t)g * 36;
        const float d = __half2float(*(const __half*)(blk + 32));
        const float m = __half2float(*(const __half*)(blk + 34));
        const unsigned char* qs = blk + (w4 << 3);
        const float4 xa = *(const float4*)(x + (size_t)g * 32 + e0);
        const float4 xb = *(const float4*)(x + (size_t)g * 32 + e0 + 4);
        acc += (d * (float)qs[0] + m) * xa.x;
        acc += (d * (float)qs[1] + m) * xa.y;
        acc += (d * (float)qs[2] + m) * xa.z;
        acc += (d * (float)qs[3] + m) * xa.w;
        acc += (d * (float)qs[4] + m) * xb.x;
        acc += (d * (float)qs[5] + m) * xb.y;
        acc += (d * (float)qs[6] + m) * xb.z;
        acc += (d * (float)qs[7] + m) * xb.w;
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Batched quantized GEMM: out[tok][row] = sum_c X[tok][c] * W[row][c]
//
// This is the prefill workhorse. out is token-major because every batch kernel in
// the layer (swiglu, add-rmsnorm, the next GEMM's X) indexes it as [tok][row].
//
// Shape per block: 64 tokens x 64 output rows, 256 threads, K step = 32 (one
// Affine block, so the dequantized (d,m) are uniform across a k-tile).
//
// wmma with A = X (row-major, m = tokens, k contiguous) and B = W (col_major,
// n = rows) produces C[tok][row] directly, so the epilogue needs no transpose.
// ---------------------------------------------------------------------------
#define DLG_TOK 64
#define DLG_ROW 64
#define DLG_KC  32

template <bool AFF8>
__device__ __forceinline__ void dl_gemm_stage(const unsigned char* __restrict__ W,
                                              const __half* __restrict__ X,
                                              __half* __restrict__ sw,
                                              __half* __restrict__ sx,
                                              int row0, int tok0, int rows, int npos,
                                              int cols, int ng, int k0,
                                              int lrow, int lpart) {
    const int block_bytes = AFF8 ? 36 : 20;
    const int grow = row0 + lrow;
    if (grow < rows) {
        const unsigned char* blk = W + ((size_t)grow * ng + (k0 >> 5)) * block_bytes;
        dl_affine_part<AFF8>(blk, lpart, sw + lrow * DLG_KC + lpart * 8);
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) sw[lrow * DLG_KC + lpart * 8 + i] = __float2half(0.f);
    }
    const int tok = tok0 + lrow;
    if (tok < npos) {
        *(uint4*)(sx + lrow * DLG_KC + lpart * 8) =
            *(const uint4*)(X + (size_t)tok * cols + k0 + lpart * 8);
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) sx[lrow * DLG_KC + lpart * 8 + i] = __float2half(0.f);
    }
}

template <bool AFF8>
__device__ __forceinline__ void dl_gemm_body(const unsigned char* __restrict__ W,
                                             const __half* __restrict__ X,
                                             float* __restrict__ out,
                                             int rows, int cols, int npos) {
    const int tid = threadIdx.x;          // 256 threads
    const int warp = tid >> 5;            // 0..7
    const int lane = tid & 31;
    const int row0 = blockIdx.y * DLG_ROW;
    const int tok0 = blockIdx.x * DLG_TOK;
    const int wt = warp & 3;              // token sub-tile (m) 0..3
    const int wr = warp >> 2;             // row sub-tiles 2*wr, 2*wr+1

    __align__(16) __shared__ __half sW[2][DLG_ROW * DLG_KC];
    __align__(16) __shared__ __half sX[2][DLG_TOK * DLG_KC];
    __shared__ float sC[8][16 * 16];

    const int lrow = tid >> 2;            // 0..63: row of this block's tile
    const int lpart = tid & 3;            // 0..3: 8 codes / 8 halves of that row
    const int ng = cols >> 5;
    const int nk = cols / DLG_KC;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[2];
    nvcuda::wmma::fill_fragment(acc[0], 0.0f);
    nvcuda::wmma::fill_fragment(acc[1], 0.0f);

    dl_gemm_stage<AFF8>(W, X, sW[0], sX[0], row0, tok0, rows, npos, cols, ng, 0, lrow, lpart);
    __syncthreads();

    int buf = 0;
    for (int ks = 0; ks < nk; ks++) {
        if (ks + 1 < nk) {
            dl_gemm_stage<AFF8>(W, X, sW[buf ^ 1], sX[buf ^ 1], row0, tok0, rows, npos,
                                cols, ng, (ks + 1) * DLG_KC, lrow, lpart);
        }
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half,
                               nvcuda::wmma::row_major> fa;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half,
                               nvcuda::wmma::col_major> fb;
#pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            nvcuda::wmma::load_matrix_sync(fa, sX[buf] + (wt * 16) * DLG_KC + kh * 16, DLG_KC);
#pragma unroll
            for (int sub = 0; sub < 2; sub++) {
                nvcuda::wmma::load_matrix_sync(fb, sW[buf] + (wr * 32 + sub * 16) * DLG_KC + kh * 16,
                                               DLG_KC);
                nvcuda::wmma::mma_sync(acc[sub], fa, fb, acc[sub]);
            }
        }
        __syncthreads();
        buf ^= 1;
    }

#pragma unroll
    for (int sub = 0; sub < 2; sub++) {
        nvcuda::wmma::store_matrix_sync(sC[warp], acc[sub], 16, nvcuda::wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) {
            const int t = tok0 + wt * 16 + (i >> 4);
            const int r = row0 + wr * 32 + sub * 16 + (i & 15);
            if (t < npos && r < rows) out[(size_t)t * rows + r] = sC[warp][i];
        }
        __syncwarp();
    }
}

extern "C" __global__ void dl_gemm_affine4_f16(const unsigned char* __restrict__ W,
                                               const __half* __restrict__ X,
                                               float* __restrict__ out,
                                               int rows, int cols, int npos) {
    dl_gemm_body<false>(W, X, out, rows, cols, npos);
}

extern "C" __global__ void dl_gemm_affine8_f16(const unsigned char* __restrict__ W,
                                               const __half* __restrict__ X,
                                               float* __restrict__ out,
                                               int rows, int cols, int npos) {
    dl_gemm_body<true>(W, X, out, rows, cols, npos);
}

// ---------------------------------------------------------------------------
// Batched (prompt-processing) layer kernels.
//
// The single-token kernels above stay the decode path; these take the whole
// prompt chunk at once (grid.y = position) so one dispatch covers npos tokens.
// All buffers are token-major: x[pos][elem].
// ---------------------------------------------------------------------------

// out[pos][i] = x[pos][i] / rms(x[pos]) * w[i]
extern "C" __global__ void dl_batch_rmsnorm(float* __restrict__ out,
                                            const float* __restrict__ x,
                                            const float* __restrict__ w,
                                            int n, int npos, float eps) {
    const int pos = blockIdx.x;
    if (pos >= npos) return;
    __shared__ float sdata[1024];
    const float* xp = x + (size_t)pos * n;
    float* op = out + (size_t)pos * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xp[i] * xp[i];
    ss = block_reduce_sum(ss, sdata);
    const float inv = rsqrtf(ss / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) op[i] = xp[i] * inv * w[i];
}

// sum[pos][i] = a[pos][i] + b[pos][i]; norm[pos][i] = rms(sum[pos]) * w[i]
extern "C" __global__ void dl_batch_add_rmsnorm(float* __restrict__ normOut,
                                                float* __restrict__ sumOut,
                                                const float* __restrict__ a,
                                                const float* __restrict__ b,
                                                const float* __restrict__ w,
                                                int n, int npos, float eps) {
    const int pos = blockIdx.x;
    if (pos >= npos) return;
    __shared__ float sdata[1024];
    const size_t off = (size_t)pos * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float s = a[off + i] + b[off + i];
        sumOut[off + i] = s;
        ss += s * s;
    }
    ss = block_reduce_sum(ss, sdata);
    const float inv = rsqrtf(ss / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        normOut[off + i] = sumOut[off + i] * inv * w[i];
    }
}

// Per-head RMS norm over data[pos][head][head_dim] (QK-norm).
// Launch: grid = (numHeads, npos), block = headDim (power of 2, <= 1024).
extern "C" __global__ void dl_batch_rmsnorm_heads(float* __restrict__ data,
                                                  const float* __restrict__ w,
                                                  int numHeads, int headDim,
                                                  int npos, float eps) {
    const int head = blockIdx.x, pos = blockIdx.y;
    if (pos >= npos) return;
    __shared__ float sdata[1024];
    float* base = data + (((size_t)pos * numHeads) + head) * headDim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) ss += base[i] * base[i];
    ss = block_reduce_sum(ss, sdata);
    const float inv = rsqrtf(ss / (float)headDim + eps);
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) base[i] = base[i] * inv * w[i];
}

// Batched RoPE. Launch: grid = (max(numHeads,numKVHeads), npos), block = ropeDim/2.
// pos = startPos + blockIdx.y, matching the single-token kernel's table layout.
extern "C" __global__ void dl_batch_rope(float* __restrict__ q,
                                         float* __restrict__ k,
                                         const float* __restrict__ cosT,
                                         const float* __restrict__ sinT,
                                         int numHeads, int numKVHeads,
                                         int headDim, int ropeDim,
                                         int startPos, int neox, int npos) {
    const int head = blockIdx.x;
    const int pos = startPos + blockIdx.y;
    if (blockIdx.y >= npos) return;
    const int tid = threadIdx.x;
    const int rd = ropeDim > 0 ? ropeDim : headDim;
    const int halfDim = rd >> 1;
    if (tid >= halfDim) return;
    const int table_base = pos * halfDim + tid;
    const float c = cosT[table_base];
    const float s = sinT[table_base];
    const int i0 = neox ? tid : 2 * tid;
    const int i1 = neox ? tid + halfDim : 2 * tid + 1;
    if (head < numHeads) {
        float* base = q + (((size_t)blockIdx.y * numHeads) + head) * headDim;
        const float a = base[i0], b = base[i1];
        base[i0] = a * c - b * s;
        base[i1] = a * s + b * c;
    }
    if (head < numKVHeads) {
        float* base = k + (((size_t)blockIdx.y * numKVHeads) + head) * headDim;
        const float a = base[i0], b = base[i1];
        base[i0] = a * c - b * s;
        base[i1] = a * s + b * c;
    }
}

// Append npos rows of k/v (float32) to the f16 KV cache at startPos.
// Launch: grid = (ceil(kvDim/256), npos), block = 256.
extern "C" __global__ void dl_batch_kv_store_f16(const float* __restrict__ kIn,
                                                 const float* __restrict__ vIn,
                                                 __half* __restrict__ kOut,
                                                 __half* __restrict__ vOut,
                                                 int startPos, int kvDim, int npos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= npos || i >= kvDim) return;
    const size_t src = (size_t)row * kvDim + i;
    const size_t dst = (size_t)(startPos + row) * kvDim + i;
    kOut[dst] = __float2half(kIn[src]);
    vOut[dst] = __float2half(vIn[src]);
}

// ---------------------------------------------------------------------------
// Causal flash attention for prompt chunks.
//
// One block per (head, 64-query tile); 128 threads = 4 warps, one 16-query row
// block each. Keys are streamed in 32-row tiles straight out of the f16 KV
// cache, so K/V traffic is amortised over 64 queries per head instead of being
// re-read per query (which is what makes the per-token path quadratic).
//
// Block-to-block L2 reuse covers GQA: the numHeads blocks for one query tile all
// read the same kv head's rows, so DRAM sees each row about once.
//
// Requires headDim <= DLATT_HD and headDim % 16 == 0 (both true for Qwen/Llama).
// ---------------------------------------------------------------------------
#define DLATT_QTOK 64
#define DLATT_KTOK 32
#define DLATT_HD   128
#define DLATT_THREADS 128

extern "C" __global__ void dl_attn_prefill_f16(float* __restrict__ out,
                                               const float* __restrict__ q,
                                               const __half* __restrict__ kCache,
                                               const __half* __restrict__ vCache,
                                               int numHeads, int numKVHeads,
                                               int headDim, int kvDim,
                                               int startPos, int npos,
                                               float scale, float softcap) {
    const int head = blockIdx.x;
    const int qt = blockIdx.y;                 // query tile index
    const int q0 = qt * DLATT_QTOK;            // chunk-relative first query
    if (q0 >= npos) return;
    const int nq = min(DLATT_QTOK, npos - q0);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int grp = numHeads / numKVHeads;     // query heads per kv head
    const int kvh = head / grp;

    __shared__ __align__(16) __half sQ[DLATT_QTOK * DLATT_HD];
    __shared__ __align__(16) __half sK[DLATT_KTOK * DLATT_HD];
    __shared__ __align__(16) __half sV[DLATT_KTOK * DLATT_HD];
    __shared__ __align__(16) __half sP[4][16 * DLATT_KTOK];
    __shared__ float sS[4][16 * DLATT_KTOK];
    __shared__ float sM[4][16], sL[4][16];

    const int qBase = warp * 16;               // this warp's first query (tile-local)

    // Stage this tile's queries for the whole block (f32 -> f16 once).
    for (int i = threadIdx.x; i < DLATT_QTOK * DLATT_HD; i += DLATT_THREADS) {
        const int r = i / DLATT_HD, c = i % DLATT_HD;
        sQ[i] = __float2half((r < nq && c < headDim)
                                 ? q[(((size_t)(q0 + r) * numHeads) + head) * headDim + c]
                                 : 0.0f);
    }

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> oAcc[DLATT_HD / 16];
#pragma unroll
    for (int i = 0; i < DLATT_HD / 16; i++) nvcuda::wmma::fill_fragment(oAcc[i], 0.0f);
    float mrun = -INFINITY, lrun = 0.0f;
    __syncthreads();

    // The loop bound must be block-uniform: the body has a __syncthreads(), so
    // per-warp trip counts would diverge and either deadlock or race on sK/sV.
    // Early warps just mask out the keys past their own last query.
    const int kEnd = startPos + q0 + nq - 1;
    for (int k0 = 0; k0 <= kEnd; k0 += DLATT_KTOK) {
        const int nk = min(DLATT_KTOK, kEnd + 1 - k0);
        for (int i = threadIdx.x; i < DLATT_KTOK * DLATT_HD; i += DLATT_THREADS) {
            const int r = i / DLATT_HD, c = i % DLATT_HD;
            const size_t src = (size_t)(k0 + r) * kvDim + kvh * headDim + c;
            sK[i] = (r < nk && c < headDim) ? kCache[src] : __float2half(0.f);
            sV[i] = (r < nk && c < headDim) ? vCache[src] : __float2half(0.f);
        }
        __syncthreads();

        // --- scores: S[16 queries][32 keys] = Q[warp] . K ---
        {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> sAcc[2];
            nvcuda::wmma::fill_fragment(sAcc[0], 0.0f);
            nvcuda::wmma::fill_fragment(sAcc[1], 0.0f);
            for (int d = 0; d < headDim; d += 16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half,
                                       nvcuda::wmma::row_major> fa;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half,
                                       nvcuda::wmma::col_major> fb[2];
                nvcuda::wmma::load_matrix_sync(fa, sQ + qBase * DLATT_HD + d, DLATT_HD);
                nvcuda::wmma::load_matrix_sync(fb[0], sK + d, DLATT_HD);
                nvcuda::wmma::load_matrix_sync(fb[1], sK + 16 * DLATT_HD + d, DLATT_HD);
                nvcuda::wmma::mma_sync(sAcc[0], fa, fb[0], sAcc[0]);
                nvcuda::wmma::mma_sync(sAcc[1], fa, fb[1], sAcc[1]);
                __syncwarp();
            }
            nvcuda::wmma::store_matrix_sync(sS[warp], sAcc[0], DLATT_KTOK, nvcuda::wmma::mem_row_major);
            nvcuda::wmma::store_matrix_sync(sS[warp] + 16, sAcc[1], DLATT_KTOK, nvcuda::wmma::mem_row_major);
        }

        // --- causal mask + online softmax over this key tile ---
        // Lane covers half a row: row = lane & 15, keys = half*16 .. +16.
        {
            const int row = lane & 15;
            const int halfk = lane >> 4;
            float* srow = sS[warp] + row * DLATT_KTOK + halfk * 16;
            const int p = startPos + q0 + qBase + row;     // absolute query position
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                float v = srow[j] * scale;
                if (softcap > 0.0f) v = softcap * tanhf(v / softcap);
                if (k0 + halfk * 16 + j > p) v = -INFINITY;
                srow[j] = v;
                mx = fmaxf(mx, v);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
            const float mnew = fmaxf(mrun, mx);
            const float corr = expf(mrun - mnew);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const float e = expf(srow[j] - mnew);
                srow[j] = e;
                sP[warp][row * DLATT_KTOK + halfk * 16 + j] = __float2half(e);
                sum += e;
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 16);
            lrun = lrun * corr + sum;
            mrun = mnew;
            // Rescale O by the per-row correction. A wmma accumulator's
            // element->(row,col) mapping is opaque and is NOT this lane's softmax
            // row, so the factor is broadcast through a 16x16 tile and applied
            // fragment-wise: both fragments share a layout, so the product is
            // row-correct. (sS is free here - P already lives in sP.)
            __syncwarp();
            if (halfk == 0) {
#pragma unroll
                for (int c = 0; c < 16; c++) sS[warp][row * 16 + c] = corr;
            }
            __syncwarp();
            {
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf;
                nvcuda::wmma::load_matrix_sync(cf, sS[warp], 16, nvcuda::wmma::mem_row_major);
#pragma unroll
                for (int t = 0; t < DLATT_HD / 16; t++) {
#pragma unroll
                    for (int e = 0; e < oAcc[t].num_elements; e++) oAcc[t].x[e] *= cf.x[e];
                }
            }
        }
        __syncwarp();

        // --- O += P[16][32] * V[32][headDim] ---
        {
            for (int d0 = 0; d0 < headDim; d0 += 16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half,
                                       nvcuda::wmma::row_major> fa[2];
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half,
                                       nvcuda::wmma::row_major> fb;
                nvcuda::wmma::load_matrix_sync(fa[0], sP[warp], DLATT_KTOK);
                nvcuda::wmma::load_matrix_sync(fa[1], sP[warp] + 16, DLATT_KTOK);
                nvcuda::wmma::load_matrix_sync(fb, sV + d0, DLATT_HD);
                nvcuda::wmma::mma_sync(oAcc[d0 / 16], fa[0], fb, oAcc[d0 / 16]);
                nvcuda::wmma::load_matrix_sync(fb, sV + 16 * DLATT_HD + d0, DLATT_HD);
                nvcuda::wmma::mma_sync(oAcc[d0 / 16], fa[1], fb, oAcc[d0 / 16]);
                __syncwarp();
            }
        }
        __syncthreads();
        if (k0 + DLATT_KTOK > kEnd) break;
    }

    // --- epilogue: O/l, written token-major as out[pos][head][dim] ---
    // lrun belongs to this lane's softmax row (lane & 15), not to the row this
    // lane writes, so publish 1/l per row before rewriting O from shared.
    if ((lane >> 4) == 0) sL[warp][lane & 15] = lrun > 0.0f ? 1.0f / lrun : 0.0f;
    __syncwarp();
#pragma unroll
    for (int t = 0; t < DLATT_HD / 16; t++) {
        if (t * 16 >= headDim) break;
        nvcuda::wmma::store_matrix_sync(sS[warp], oAcc[t], 16, nvcuda::wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) {
            const int r = i >> 4, c = i & 15;
            const int qq = q0 + qBase + r;
            const int dd = t * 16 + c;
            if (r < nq && qq < npos && dd < headDim) {
                out[(((size_t)qq * numHeads) + head) * headDim + dd] =
                    sS[warp][i] * sL[warp][r];
            }
        }
        __syncwarp();
    }
}

// Byte-exact device-to-device region copy. Go's CopyRegion moves float regions
// (logical positions, KV rows, hidden rows) by byte offset, and a CUDA backend
// has no memcpy entry point for it otherwise; the uint32 body keeps it near
// bandwidth while the tail handles any non-multiple-of-4 size.
extern "C" __global__ void dl_copy_region(unsigned char* __restrict__ dst,
                                          const unsigned char* __restrict__ src,
                                          long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n4 = n >> 2;
    if (i < n4) {
        ((unsigned int*)dst)[i] = ((const unsigned int*)src)[i];
        return;
    }
    const long long b = (n4 << 2) + (i - n4);
    if (b < n) dst[b] = src[b];
}
