//go:build amd64 && cgo

// simd_dot.c — AVX2+FMA fused dequantize-dot-product for GGUF quantized types.
// Called from Go via CGo. These replace the hot inner loop of quantized matmul.
//
// Each function computes: sum_i( dequant(q[i]) * x[i] ) for one row, using
// AVX2 integer/float conversions and FMA accumulation.

// Enable AVX2/FMA/F16C at the function level via pragmas,
// so we don't need -mavx2 -mfma -mf16c in CFLAGS (avoids CGo flag restrictions).
#pragma GCC target("avx2,fma,f16c,avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")
#pragma GCC optimize("O3")

#include <immintrin.h>
#include <stdint.h>
#include "iq_tables_c.h"
#include <string.h>
#include <math.h>

// ── Helper: convert float16 (binary16) to float32 ──────────────
// Uses F16C intrinsic (SSE extension, always present with AVX2).
static inline float f16_to_f32(uint16_t h) {
    __m128i v = _mm_set1_epi16(h);
    __m128 f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

static inline float hsum256_ps(__m256 v) {
    __m128 hi128 = _mm256_extractf128_ps(v, 1);
    __m128 lo128 = _mm256_castps256_ps128(v);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q4_0: 32 values per 18-byte block ──────────────────────────
// Layout: f16 scale + 16 nibble bytes
// dequant: (nibble - 8) * scale
float vec_dot_q4_0(const uint8_t* data, const float* x, int n) {
    int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();
    
    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 18;
        float d = f16_to_f32(*(const uint16_t*)block);
        __m256 vd = _mm256_set1_ps(d);
        const uint8_t* qs = block + 2;
        const float* xp = x + b * 32;
        
        // Process 32 values: first 16 from low nibbles, next 16 from high nibbles
        // But they interleave: byte j has value j (low nibble) and j+16 (high nibble)
        for (int j = 0; j < 16; j += 8) {
            // Load 8 bytes, extract low and high nibbles
            __m128i bytes = _mm_loadl_epi64((const __m128i*)(qs + j));
            __m128i lo = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
            __m128i hi = _mm_srli_epi16(bytes, 4);
            hi = _mm_and_si128(hi, _mm_set1_epi8(0x0F));
            
            // Convert to int32 and subtract 8
            __m256i lo32 = _mm256_cvtepu8_epi32(lo);
            __m256i hi32 = _mm256_cvtepu8_epi32(hi);
            __m256i eight = _mm256_set1_epi32(8);
            lo32 = _mm256_sub_epi32(lo32, eight);
            hi32 = _mm256_sub_epi32(hi32, eight);
            
            // Convert to float, multiply by scale and x
            __m256 flo = _mm256_cvtepi32_ps(lo32);
            __m256 fhi = _mm256_cvtepi32_ps(hi32);
            
            __m256 xlo = _mm256_loadu_ps(xp + j);
            __m256 xhi = _mm256_loadu_ps(xp + j + 16);
            
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, flo), xlo, acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, fhi), xhi, acc);
        }
    }
    
    // Horizontal sum
    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Batch version: process multiple rows in one CGo call ───────

// F16 row dot: n half-precision values against n float32 values.
float vec_dot_f16(const uint8_t* data, const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        uint16_t h = *(const uint16_t*)(data + i * 2);
        sum += f16_to_f32(h) * x[i];
    }
    return sum;
}

void vec_dot_f16_batch(const uint8_t* data, const float* x, int n,
                        float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_f16(data + (size_t)r * bpr, x, n);
    }
}
void vec_dot_q4_0_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q4_0(data + (size_t)r * bpr, x, n);
    }
}

// ── Q8_0: 32 values per 34-byte block ──────────────────────────
// Layout: f16 scale + 32 int8 quants
// dequant: int8(q) * scale
// Optimized: 2-block unrolled loop for better ILP. Each block computes
// dot(int8_quants, x) then scales once (saves 3 vmulps per block).
float vec_dot_q8_0(const uint8_t* data, const float* x, int n) {
    int nb = n / 32;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int b = 0;
    // Process 2 blocks at a time for better instruction-level parallelism
    for (; b <= nb - 2; b += 2) {
        // Block 0
        const uint8_t* block0 = data + b * 34;
        float d0 = f16_to_f32(*(const uint16_t*)block0);
        const int8_t* qs0 = (const int8_t*)(block0 + 2);
        const float* xp0 = x + b * 32;

        // Block 1
        const uint8_t* block1 = data + (b + 1) * 34;
        float d1 = f16_to_f32(*(const uint16_t*)block1);
        const int8_t* qs1 = (const int8_t*)(block1 + 2);
        const float* xp1 = x + (b + 1) * 32;

        // Prefetch next pair
        if (b + 3 < nb) {
            _mm_prefetch((const char*)(data + (b + 2) * 34), _MM_HINT_T0);
            _mm_prefetch((const char*)(data + (b + 3) * 34), _MM_HINT_T0);
        }

        __m256 bd0 = _mm256_setzero_ps();
        __m256 bd1 = _mm256_setzero_ps();
        for (int j = 0; j < 32; j += 8) {
            __m128i bytes0 = _mm_loadl_epi64((const __m128i*)(qs0 + j));
            __m256i i32_0 = _mm256_cvtepi8_epi32(bytes0);
            __m256 fval0 = _mm256_cvtepi32_ps(i32_0);
            bd0 = _mm256_fmadd_ps(fval0, _mm256_loadu_ps(xp0 + j), bd0);

            __m128i bytes1 = _mm_loadl_epi64((const __m128i*)(qs1 + j));
            __m256i i32_1 = _mm256_cvtepi8_epi32(bytes1);
            __m256 fval1 = _mm256_cvtepi32_ps(i32_1);
            bd1 = _mm256_fmadd_ps(fval1, _mm256_loadu_ps(xp1 + j), bd1);
        }
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), bd0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), bd1, acc1);
    }

    // Handle remaining single block
    for (; b < nb; b++) {
        const uint8_t* block = data + b * 34;
        float d = f16_to_f32(*(const uint16_t*)block);
        const int8_t* qs = (const int8_t*)(block + 2);
        const float* xp = x + b * 32;

        __m256 block_dot = _mm256_setzero_ps();
        for (int j = 0; j < 32; j += 8) {
            __m128i bytes = _mm_loadl_epi64((const __m128i*)(qs + j));
            __m256i i32 = _mm256_cvtepi8_epi32(bytes);
            __m256 fval = _mm256_cvtepi32_ps(i32);
            block_dot = _mm256_fmadd_ps(fval, _mm256_loadu_ps(xp + j), block_dot);
        }
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d), block_dot, acc0);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 hi128 = _mm256_extractf128_ps(acc0, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc0);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q4_1: 32 values per 20-byte block ──────────────────────────
// Layout: f16 scale + f16 min + 16 nibble bytes
// dequant: nibble * scale + min
float vec_dot_q4_1(const uint8_t* data, const float* x, int n) {
    int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 20;
        float d = f16_to_f32(*(const uint16_t*)block);
        float m = f16_to_f32(*(const uint16_t*)(block + 2));
        __m256 vd = _mm256_set1_ps(d);
        __m256 vm = _mm256_set1_ps(m);
        const uint8_t* qs = block + 4;
        const float* xp = x + b * 32;

        for (int j = 0; j < 16; j += 8) {
            __m128i bytes = _mm_loadl_epi64((const __m128i*)(qs + j));
            __m128i lo = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
            __m128i hi = _mm_srli_epi16(bytes, 4);
            hi = _mm_and_si128(hi, _mm_set1_epi8(0x0F));

            __m256i lo32 = _mm256_cvtepu8_epi32(lo);
            __m256i hi32 = _mm256_cvtepu8_epi32(hi);

            __m256 flo = _mm256_cvtepi32_ps(lo32);
            __m256 fhi = _mm256_cvtepi32_ps(hi32);

            __m256 xlo = _mm256_loadu_ps(xp + j);
            __m256 xhi = _mm256_loadu_ps(xp + j + 16);

            // val = d * q + m; dot += val * x = d*q*x + m*x
            acc = _mm256_fmadd_ps(_mm256_fmadd_ps(vd, flo, vm), xlo, acc);
            acc = _mm256_fmadd_ps(_mm256_fmadd_ps(vd, fhi, vm), xhi, acc);
        }
    }

    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q5_0: 32 values per 22-byte block ──────────────────────────
// Layout: f16 scale + 4 bytes qh (5th bits) + 16 nibble bytes
// dequant: ((4bits | 5thbit<<4) - 16) * scale
float vec_dot_q5_0(const uint8_t* data, const float* x, int n) {
    int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();
    __m256i sixteen = _mm256_set1_epi32(16);

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 22;
        float d = f16_to_f32(*(const uint16_t*)block);
        __m256 vd = _mm256_set1_ps(d);
        uint32_t qh;
        memcpy(&qh, block + 2, 4);
        const uint8_t* qs = block + 6;
        const float* xp = x + b * 32;

        for (int j = 0; j < 16; j += 8) {
            __m128i bytes = _mm_loadl_epi64((const __m128i*)(qs + j));
            __m128i lo_nib = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
            __m128i hi_nib = _mm_srli_epi16(bytes, 4);
            hi_nib = _mm_and_si128(hi_nib, _mm_set1_epi8(0x0F));

            __m256i lo32 = _mm256_cvtepu8_epi32(lo_nib);
            __m256i hi32 = _mm256_cvtepu8_epi32(hi_nib);

            // Extract 5th bits from qh for positions j..j+7 and j+16..j+23
            uint8_t h_lo[8], h_hi[8];
            for (int k = 0; k < 8; k++) {
                h_lo[k] = ((qh >> (j + k)) & 1) ? 16 : 0;
                h_hi[k] = ((qh >> (j + k + 16)) & 1) ? 16 : 0;
            }
            __m128i hlo_128 = _mm_loadl_epi64((const __m128i*)h_lo);
            __m128i hhi_128 = _mm_loadl_epi64((const __m128i*)h_hi);
            __m256i hlo32 = _mm256_cvtepu8_epi32(hlo_128);
            __m256i hhi32 = _mm256_cvtepu8_epi32(hhi_128);

            lo32 = _mm256_or_si256(lo32, hlo32);
            hi32 = _mm256_or_si256(hi32, hhi32);
            lo32 = _mm256_sub_epi32(lo32, sixteen);
            hi32 = _mm256_sub_epi32(hi32, sixteen);

            __m256 flo = _mm256_cvtepi32_ps(lo32);
            __m256 fhi = _mm256_cvtepi32_ps(hi32);
            __m256 xlo = _mm256_loadu_ps(xp + j);
            __m256 xhi = _mm256_loadu_ps(xp + j + 16);

            acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, flo), xlo, acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, fhi), xhi, acc);
        }
    }

    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q5_1: 32 values per 24-byte block ──────────────────────────
// Layout: f16 scale + f16 min + 4 bytes qh + 16 nibble bytes
// dequant: (4bits | 5thbit<<4) * scale + min
float vec_dot_q5_1(const uint8_t* data, const float* x, int n) {
    int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 24;
        float d = f16_to_f32(*(const uint16_t*)block);
        float m = f16_to_f32(*(const uint16_t*)(block + 2));
        __m256 vd = _mm256_set1_ps(d);
        __m256 vm = _mm256_set1_ps(m);
        uint32_t qh;
        memcpy(&qh, block + 4, 4);
        const uint8_t* qs = block + 8;
        const float* xp = x + b * 32;

        for (int j = 0; j < 16; j += 8) {
            __m128i bytes = _mm_loadl_epi64((const __m128i*)(qs + j));
            __m128i lo_nib = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
            __m128i hi_nib = _mm_srli_epi16(bytes, 4);
            hi_nib = _mm_and_si128(hi_nib, _mm_set1_epi8(0x0F));

            __m256i lo32 = _mm256_cvtepu8_epi32(lo_nib);
            __m256i hi32 = _mm256_cvtepu8_epi32(hi_nib);

            uint8_t h_lo[8], h_hi[8];
            for (int k = 0; k < 8; k++) {
                h_lo[k] = ((qh >> (j + k)) & 1) ? 16 : 0;
                h_hi[k] = ((qh >> (j + k + 16)) & 1) ? 16 : 0;
            }
            __m128i hlo_128 = _mm_loadl_epi64((const __m128i*)h_lo);
            __m128i hhi_128 = _mm_loadl_epi64((const __m128i*)h_hi);
            __m256i hlo32 = _mm256_cvtepu8_epi32(hlo_128);
            __m256i hhi32 = _mm256_cvtepu8_epi32(hhi_128);

            lo32 = _mm256_or_si256(lo32, hlo32);
            hi32 = _mm256_or_si256(hi32, hhi32);

            __m256 flo = _mm256_cvtepi32_ps(lo32);
            __m256 fhi = _mm256_cvtepi32_ps(hi32);
            __m256 xlo = _mm256_loadu_ps(xp + j);
            __m256 xhi = _mm256_loadu_ps(xp + j + 16);

            acc = _mm256_fmadd_ps(_mm256_fmadd_ps(vd, flo, vm), xlo, acc);
            acc = _mm256_fmadd_ps(_mm256_fmadd_ps(vd, fhi, vm), xhi, acc);
        }
    }

    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q2_K: 256 values per 84-byte block ─────────────────────────
float vec_dot_q2_k(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();
    
    for (int block = 0; block < nb; block++) {
        const uint8_t* bp = data + block * 84;
        const uint8_t* scales = bp;
        const uint8_t* qs = bp + 16;
        float d = f16_to_f32(*(const uint16_t*)(bp + 80));
        float dmin = f16_to_f32(*(const uint16_t*)(bp + 82));
        const float* xp = x + block * 256;
        
        int xi = 0;
        int is = 0;
        const uint8_t* qptr = qs;
        for (int n128 = 0; n128 < 2; n128++) {
            for (int j = 0; j < 4; j++) {
                uint32_t shift = j * 2;
                
                uint8_t sc = scales[is++];
                float dl = d * (float)(sc & 0xF);
                float ml = dmin * (float)(sc >> 4);
                __m256 vdl = _mm256_set1_ps(dl);
                __m256 vml = _mm256_set1_ps(ml);
                
                for (int l = 0; l < 16; l += 8) {
                    // Extract 2-bit quants
                    __m128i raw = _mm_loadl_epi64((const __m128i*)(qptr + l));
                    __m128i shifted = _mm_srli_epi16(raw, shift);
                    __m128i masked = _mm_and_si128(shifted, _mm_set1_epi8(3));
                    __m256i q32 = _mm256_cvtepu8_epi32(masked);
                    __m256 fq = _mm256_cvtepi32_ps(q32);
                    __m256 xv = _mm256_loadu_ps(xp + xi + l);
                    // val = dl * q - ml
                    __m256 val = _mm256_fmsub_ps(vdl, fq, vml);
                    acc = _mm256_fmadd_ps(val, xv, acc);
                }
                xi += 16;
                
                sc = scales[is++];
                dl = d * (float)(sc & 0xF);
                ml = dmin * (float)(sc >> 4);
                vdl = _mm256_set1_ps(dl);
                vml = _mm256_set1_ps(ml);
                
                for (int l = 0; l < 16; l += 8) {
                    __m128i raw = _mm_loadl_epi64((const __m128i*)(qptr + 16 + l));
                    __m128i shifted = _mm_srli_epi16(raw, shift);
                    __m128i masked = _mm_and_si128(shifted, _mm_set1_epi8(3));
                    __m256i q32 = _mm256_cvtepu8_epi32(masked);
                    __m256 fq = _mm256_cvtepi32_ps(q32);
                    __m256 xv = _mm256_loadu_ps(xp + xi + l);
                    __m256 val = _mm256_fmsub_ps(vdl, fq, vml);
                    acc = _mm256_fmadd_ps(val, xv, acc);
                }
                xi += 16;
            }
            qptr += 32;
        }
    }
    
    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q3_K: 256 values per 110-byte block ────────────────────────
float vec_dot_q3_k(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();
    
    for (int block = 0; block < nb; block++) {
        const uint8_t* bp = data + block * 110;
        float dAll = f16_to_f32(*(const uint16_t*)(bp + 108));
        
        const uint8_t* hm = bp;
        const uint8_t* qp = bp + 32;
        const uint8_t* scp = bp + 96;
        const float* xp = x + block * 256;
        
        // Unpack 16 x 6-bit scales
        uint32_t aux[4] = {0, 0, 0, 0};
        for (int i = 0; i < 12; i++) {
            aux[i/4] |= (uint32_t)scp[i] << ((i%4)*8);
        }
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
        aux[3] = ((aux[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
        aux[0] = (aux[0] & 0x0f0f0f0f) | (((tmp >> 0) & 0x03030303) << 4);
        aux[1] = (aux[1] & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4);
        
        int8_t scales[16];
        memcpy(scales, aux, 16);
        
        int xi = 0;
        int is = 0;
        uint8_t m = 1;
        
        for (int n128 = 0; n128 < 2; n128++) {
            for (int j = 0; j < 4; j++) {
                uint32_t shift = j * 2;
                float dl = dAll * (float)(scales[is] - 32);
                is++;
                __m256 vdl = _mm256_set1_ps(dl);
                
                // Process 16 values in 2 groups of 8
                for (int l = 0; l < 16; l += 8) {
                    // Load 8 bytes of qs, shift and mask to get 2-bit quants
                    __m128i raw = _mm_loadl_epi64((const __m128i*)(qp + l));
                    __m128i shifted = _mm_srli_epi16(raw, shift);
                    __m128i masked = _mm_and_si128(shifted, _mm_set1_epi8(3));
                    __m256i q32 = _mm256_cvtepu8_epi32(masked);
                    
                    // Load 8 hmask bytes, test bit m
                    __m128i hbytes = _mm_loadl_epi64((const __m128i*)(hm + l));
                    __m128i mbyte = _mm_set1_epi8(m);
                    __m128i htst = _mm_cmpeq_epi8(_mm_and_si128(hbytes, mbyte), _mm_setzero_si128());
                    // htst = 0xFF where hmask bit is 0 (subtract 4), 0 where set (subtract 0)
                    __m256i h32 = _mm256_cvtepi8_epi32(htst);
                    // h32 is -1 where we need to subtract 4, 0 otherwise
                    __m256i four = _mm256_set1_epi32(4);
                    __m256i hbits = _mm256_and_si256(h32, four); // 4 or 0
                    q32 = _mm256_sub_epi32(q32, hbits);
                    
                    __m256 fq = _mm256_cvtepi32_ps(q32);
                    __m256 xv = _mm256_loadu_ps(xp + xi + l);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(vdl, fq), xv, acc);
                }
                xi += 16;
                
                dl = dAll * (float)(scales[is] - 32);
                is++;
                vdl = _mm256_set1_ps(dl);
                
                for (int l = 0; l < 16; l += 8) {
                    __m128i raw = _mm_loadl_epi64((const __m128i*)(qp + 16 + l));
                    __m128i shifted = _mm_srli_epi16(raw, shift);
                    __m128i masked = _mm_and_si128(shifted, _mm_set1_epi8(3));
                    __m256i q32 = _mm256_cvtepu8_epi32(masked);
                    
                    __m128i hbytes = _mm_loadl_epi64((const __m128i*)(hm + 16 + l));
                    __m128i mbyte = _mm_set1_epi8(m);
                    __m128i htst = _mm_cmpeq_epi8(_mm_and_si128(hbytes, mbyte), _mm_setzero_si128());
                    __m256i h32 = _mm256_cvtepi8_epi32(htst);
                    __m256i four = _mm256_set1_epi32(4);
                    __m256i hbits = _mm256_and_si256(h32, four);
                    q32 = _mm256_sub_epi32(q32, hbits);
                    
                    __m256 fq = _mm256_cvtepi32_ps(q32);
                    __m256 xv = _mm256_loadu_ps(xp + xi + l);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(vdl, fq), xv, acc);
                }
                xi += 16;
                
                m <<= 1;
            }
            qp += 32;
        }
    }
    
    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q4_K: 256 values per 144-byte block ────────────────────────
float vec_dot_q4_k(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();
    
    for (int block = 0; block < nb; block++) {
        const uint8_t* bp = data + block * 144;
        float d = f16_to_f32(*(const uint16_t*)bp);
        float dmin = f16_to_f32(*(const uint16_t*)(bp + 2));
        
        const uint8_t* sp = bp + 4;
        float sc[8], mn[8];
        for (int i = 0; i < 4; i++) {
            sc[i] = d * (float)(sp[i] & 0x3F);
            mn[i] = dmin * (float)(sp[4+i] & 0x3F);
        }
        for (int i = 0; i < 4; i++) {
            uint8_t scHi = (sp[i] >> 6) & 0x03;
            uint8_t mnHi = (sp[4+i] >> 6) & 0x03;
            uint8_t scLo = sp[8+i] & 0x0F;
            uint8_t mnLo = (sp[8+i] >> 4) & 0x0F;
            sc[4+i] = d * (float)(scLo | (scHi << 4));
            mn[4+i] = dmin * (float)(mnLo | (mnHi << 4));
        }
        
        const uint8_t* qp = bp + 16;
        const float* xp = x + block * 256;
        int xi = 0;
        int is = 0;
        
        for (int grp = 0; grp < 4; grp++) {
            float d1 = sc[is], m1 = mn[is];
            float d2 = sc[is+1], m2 = mn[is+1];
            __m256 vd1 = _mm256_set1_ps(d1);
            __m256 vm1 = _mm256_set1_ps(m1);
            __m256 vd2 = _mm256_set1_ps(d2);
            __m256 vm2 = _mm256_set1_ps(m2);
            
            const uint8_t* qOff = qp + grp * 32;
            
            // First 32: low nibbles
            for (int l = 0; l < 32; l += 8) {
                __m128i bytes = _mm_loadl_epi64((const __m128i*)(qOff + l));
                __m128i lo = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
                __m256i q32 = _mm256_cvtepu8_epi32(lo);
                __m256 fq = _mm256_cvtepi32_ps(q32);
                __m256 xv = _mm256_loadu_ps(xp + xi + l);
                __m256 val = _mm256_fmsub_ps(vd1, fq, vm1);
                acc = _mm256_fmadd_ps(val, xv, acc);
            }
            xi += 32;
            
            // Next 32: high nibbles  
            for (int l = 0; l < 32; l += 8) {
                __m128i bytes = _mm_loadl_epi64((const __m128i*)(qOff + l));
                __m128i hi = _mm_srli_epi16(bytes, 4);
                hi = _mm_and_si128(hi, _mm_set1_epi8(0x0F));
                __m256i q32 = _mm256_cvtepu8_epi32(hi);
                __m256 fq = _mm256_cvtepi32_ps(q32);
                __m256 xv = _mm256_loadu_ps(xp + xi + l);
                __m256 val = _mm256_fmsub_ps(vd2, fq, vm2);
                acc = _mm256_fmadd_ps(val, xv, acc);
            }
            xi += 32;
            is += 2;
        }
    }
    
    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q5_K: 256 values per 176-byte block ────────────────────────
float vec_dot_q5_k(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();
    
    for (int block = 0; block < nb; block++) {
        const uint8_t* bp = data + block * 176;
        float d = f16_to_f32(*(const uint16_t*)bp);
        float dmin = f16_to_f32(*(const uint16_t*)(bp + 2));
        
        const uint8_t* sp = bp + 4;
        float sc[8], mn[8];
        for (int i = 0; i < 4; i++) {
            sc[i] = d * (float)(sp[i] & 0x3F);
            mn[i] = dmin * (float)(sp[4+i] & 0x3F);
        }
        for (int i = 0; i < 4; i++) {
            uint8_t scHi = (sp[i] >> 6) & 0x03;
            uint8_t mnHi = (sp[4+i] >> 6) & 0x03;
            uint8_t scLo = sp[8+i] & 0x0F;
            uint8_t mnLo = (sp[8+i] >> 4) & 0x0F;
            sc[4+i] = d * (float)(scLo | (scHi << 4));
            mn[4+i] = dmin * (float)(mnLo | (mnHi << 4));
        }
        
        const uint8_t* qh = bp + 16;
        const uint8_t* qs = bp + 48;
        const float* xp = x + block * 256;
        int xi = 0;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        
        for (int grp = 0; grp < 4; grp++) {
            float d1 = sc[is], m1 = mn[is];
            float d2 = sc[is+1], m2 = mn[is+1];
            __m256 vd1 = _mm256_set1_ps(d1);
            __m256 vm1 = _mm256_set1_ps(m1);
            __m256 vd2 = _mm256_set1_ps(d2);
            __m256 vm2 = _mm256_set1_ps(m2);
            const uint8_t* qlOff = qs + grp * 32;
            
            // First 32: low nibbles + 5th bit
            for (int l = 0; l < 32; l += 8) {
                __m128i raw = _mm_loadl_epi64((const __m128i*)(qlOff + l));
                __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
                __m256i q32 = _mm256_cvtepu8_epi32(lo);
                
                // Add 5th bit from qh
                __m128i hbytes = _mm_loadl_epi64((const __m128i*)(qh + l));
                __m128i hmask = _mm_set1_epi8(u1);
                __m128i htst = _mm_and_si128(hbytes, hmask);
                // Non-zero where bit set → need to OR 16
                __m128i hcmp = _mm_cmpeq_epi8(htst, _mm_setzero_si128());
                // hcmp = 0xFF where bit is NOT set, 0 where it IS set
                // We want 16 where set, 0 where not → invert
                __m128i hset = _mm_andnot_si128(hcmp, _mm_set1_epi8(16));
                __m256i h32 = _mm256_cvtepu8_epi32(hset);
                q32 = _mm256_or_si256(q32, h32);
                
                __m256 fq = _mm256_cvtepi32_ps(q32);
                __m256 xv = _mm256_loadu_ps(xp + xi + l);
                __m256 val = _mm256_fmsub_ps(vd1, fq, vm1);
                acc = _mm256_fmadd_ps(val, xv, acc);
            }
            xi += 32;
            
            // Next 32: high nibbles + 5th bit
            for (int l = 0; l < 32; l += 8) {
                __m128i raw = _mm_loadl_epi64((const __m128i*)(qlOff + l));
                __m128i hi = _mm_srli_epi16(raw, 4);
                hi = _mm_and_si128(hi, _mm_set1_epi8(0x0F));
                __m256i q32 = _mm256_cvtepu8_epi32(hi);
                
                __m128i hbytes = _mm_loadl_epi64((const __m128i*)(qh + l));
                __m128i hmask = _mm_set1_epi8(u2);
                __m128i htst = _mm_and_si128(hbytes, hmask);
                __m128i hcmp = _mm_cmpeq_epi8(htst, _mm_setzero_si128());
                __m128i hset = _mm_andnot_si128(hcmp, _mm_set1_epi8(16));
                __m256i h32 = _mm256_cvtepu8_epi32(hset);
                q32 = _mm256_or_si256(q32, h32);
                
                __m256 fq = _mm256_cvtepi32_ps(q32);
                __m256 xv = _mm256_loadu_ps(xp + xi + l);
                __m256 val = _mm256_fmsub_ps(vd2, fq, vm2);
                acc = _mm256_fmadd_ps(val, xv, acc);
            }
            xi += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    
    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q6_K: 256 values per 210-byte block ────────────────────────
float vec_dot_q6_k(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();
    
    for (int block = 0; block < nb; block++) {
        const uint8_t* bp = data + block * 210;
        float d_val = f16_to_f32(*(const uint16_t*)(bp + 208));
        const uint8_t* ql = bp;
        const uint8_t* qh_base = bp + 128;
        const int8_t* scales = (const int8_t*)(bp + 192);
        const float* xp = x + block * 256;
        __m256 vd = _mm256_set1_ps(d_val);
        __m256i thirty_two = _mm256_set1_epi32(32);
        
        for (int n128 = 0; n128 < 2; n128++) {
            const uint8_t* qlp = ql + n128 * 64;
            const uint8_t* qhp = qh_base + n128 * 32;
            int xoff = n128 * 128;
            
            // Process 8 positions at a time (each produces 4 values at l, l+32, l+64, l+96)
            for (int l = 0; l < 32; l += 8) {
                int sidx = 8 * n128 + l / 16;

                // Load 8 ql bytes at [l] and [l+32], and 8 qh bytes
                __m128i ql0_raw = _mm_loadl_epi64((const __m128i*)(qlp + l));
                __m128i ql32_raw = _mm_loadl_epi64((const __m128i*)(qlp + l + 32));
                __m128i qh_raw = _mm_loadl_epi64((const __m128i*)(qhp + l));
                
                __m128i mask4 = _mm_set1_epi8(0x0F);
                __m128i mask2 = _mm_set1_epi8(0x03);
                
                // q1 = (ql0 & 0xF) | ((qh >> 0) & 3) << 4 - 32
                __m128i q1_lo = _mm_and_si128(ql0_raw, mask4);
                __m128i q1_hi = _mm_slli_epi16(_mm_and_si128(qh_raw, mask2), 4);
                __m128i q1_8 = _mm_or_si128(q1_lo, q1_hi);
                __m256i q1_32 = _mm256_cvtepu8_epi32(q1_8);
                q1_32 = _mm256_sub_epi32(q1_32, thirty_two);
                
                // q2 = (ql32 & 0xF) | ((qh >> 2) & 3) << 4 - 32
                __m128i q2_lo = _mm_and_si128(ql32_raw, mask4);
                __m128i qh_s2 = _mm_srli_epi16(qh_raw, 2);
                __m128i q2_hi = _mm_slli_epi16(_mm_and_si128(qh_s2, mask2), 4);
                __m128i q2_8 = _mm_or_si128(q2_lo, q2_hi);
                __m256i q2_32 = _mm256_cvtepu8_epi32(q2_8);
                q2_32 = _mm256_sub_epi32(q2_32, thirty_two);
                
                // q3 = (ql0 >> 4) | ((qh >> 4) & 3) << 4 - 32
                __m128i q3_lo = _mm_and_si128(_mm_srli_epi16(ql0_raw, 4), mask4);
                __m128i qh_s4 = _mm_srli_epi16(qh_raw, 4);
                __m128i q3_hi = _mm_slli_epi16(_mm_and_si128(qh_s4, mask2), 4);
                __m128i q3_8 = _mm_or_si128(q3_lo, q3_hi);
                __m256i q3_32 = _mm256_cvtepu8_epi32(q3_8);
                q3_32 = _mm256_sub_epi32(q3_32, thirty_two);
                
                // q4 = (ql32 >> 4) | ((qh >> 6) & 3) << 4 - 32
                __m128i q4_lo = _mm_and_si128(_mm_srli_epi16(ql32_raw, 4), mask4);
                __m128i qh_s6 = _mm_srli_epi16(qh_raw, 6);
                __m128i q4_hi = _mm_slli_epi16(_mm_and_si128(qh_s6, mask2), 4);
                __m128i q4_8 = _mm_or_si128(q4_lo, q4_hi);
                __m256i q4_32 = _mm256_cvtepu8_epi32(q4_8);
                q4_32 = _mm256_sub_epi32(q4_32, thirty_two);
                
                // Scale broadcasts — all 8 values at position l..l+7 use the same scale
                // (since l/16 is constant within each group of 16)
                // First 8 values use sidx, next 8 may use sidx+1 if l crosses 16 boundary
                __m256 vs0 = _mm256_mul_ps(vd, _mm256_set1_ps((float)scales[sidx]));
                __m256 vs2 = _mm256_mul_ps(vd, _mm256_set1_ps((float)scales[sidx + 2]));
                __m256 vs4 = _mm256_mul_ps(vd, _mm256_set1_ps((float)scales[sidx + 4]));
                __m256 vs6 = _mm256_mul_ps(vd, _mm256_set1_ps((float)scales[sidx + 6]));
                
                // Load x at 4 scattered positions
                __m256 x0 = _mm256_loadu_ps(xp + xoff + l);
                __m256 x32 = _mm256_loadu_ps(xp + xoff + l + 32);
                __m256 x64 = _mm256_loadu_ps(xp + xoff + l + 64);
                __m256 x96 = _mm256_loadu_ps(xp + xoff + l + 96);
                
                // Accumulate: d * scale * q * x
                __m256 fq1 = _mm256_cvtepi32_ps(q1_32);
                __m256 fq2 = _mm256_cvtepi32_ps(q2_32);
                __m256 fq3 = _mm256_cvtepi32_ps(q3_32);
                __m256 fq4 = _mm256_cvtepi32_ps(q4_32);
                
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vs0, fq1), x0, acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vs2, fq2), x32, acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vs4, fq3), x64, acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vs6, fq4), x96, acc);
            }
        }
    }
    
    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ══════════════════════════════════════════════════════════════════
// Batch versions: process nrows in one CGo call to amortize overhead
// ══════════════════════════════════════════════════════════════════

void vec_dot_q8_0_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q8_0(data + (size_t)r * bpr, x, n);
    }
}

void vec_dot_q4_1_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q4_1(data + (size_t)r * bpr, x, n);
    }
}
void vec_dot_q5_0_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q5_0(data + (size_t)r * bpr, x, n);
    }
}
void vec_dot_q5_1_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q5_1(data + (size_t)r * bpr, x, n);
    }
}
void vec_dot_q2_k_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q2_k(data + (size_t)r * bpr, x, n);
    }
}

void vec_dot_q3_k_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q3_k(data + (size_t)r * bpr, x, n);
    }
}

void vec_dot_q4_k_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q4_k(data + (size_t)r * bpr, x, n);
    }
}

void vec_dot_q5_k_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q5_k(data + (size_t)r * bpr, x, n);
    }
}

void vec_dot_q6_k_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q6_k(data + (size_t)r * bpr, x, n);
    }
}

// ── IQ4_NL lookup table ─────────────────────────────────────────
static const int8_t kvalues_iq4nl_c[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// ── IQ4_XS: 256 values per 136-byte block ──────────────────────
// Fused dequant+dot using stack buffer and AVX2 dot product.
float vec_dot_iq4_xs(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();

    // Load kvalues_iq4nl lookup table into a SIMD register for vpshufb
    __m128i lut = _mm_loadu_si128((const __m128i*)kvalues_iq4nl_c);
    const __m128i lo_mask = _mm_set1_epi8(0x0F);

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 136;
        float d = f16_to_f32(*(const uint16_t*)block);
        uint16_t scales_h = *(const uint16_t*)(block + 2);
        const uint8_t* scales_l = block + 4;
        const uint8_t* qs = block + 8;
        const float* xp = x + b * 256;

        for (int ib = 0; ib < 8; ib++) {
            uint8_t lo = (scales_l[ib/2] >> (4*(ib%2))) & 0xf;
            uint8_t hi = ((scales_h >> (2*ib)) & 3) << 4;
            float dl = d * (float)((int)(lo | hi) - 32);
            __m256 vdl = _mm256_set1_ps(dl);

            const uint8_t* qp = qs + ib * 16;
            const float* xsub = xp + ib * 32;

            // Load 16 bytes of quantized data
            __m128i qbytes = _mm_loadu_si128((const __m128i*)qp);

            // Extract low nibbles and look up values via vpshufb
            __m128i lo_nibbles = _mm_and_si128(qbytes, lo_mask);
            __m128i lo_vals = _mm_shuffle_epi8(lut, lo_nibbles);

            // Extract high nibbles and look up values via vpshufb
            __m128i hi_nibbles = _mm_and_si128(_mm_srli_epi16(qbytes, 4), lo_mask);
            __m128i hi_vals = _mm_shuffle_epi8(lut, hi_nibbles);

            // Convert low nibble int8 values to float (two groups of 8)
            __m128i lo_0 = _mm_cvtepi8_epi32(lo_vals);
            __m128i lo_1 = _mm_cvtepi8_epi32(_mm_srli_si128(lo_vals, 4));
            __m256 flo_0 = _mm256_cvtepi32_ps(_mm256_set_m128i(lo_1, lo_0));
            __m256 vx_lo0 = _mm256_loadu_ps(xsub);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vdl, flo_0), vx_lo0, acc);

            __m128i lo_2 = _mm_cvtepi8_epi32(_mm_srli_si128(lo_vals, 8));
            __m128i lo_3 = _mm_cvtepi8_epi32(_mm_srli_si128(lo_vals, 12));
            __m256 flo_1 = _mm256_cvtepi32_ps(_mm256_set_m128i(lo_3, lo_2));
            __m256 vx_lo1 = _mm256_loadu_ps(xsub + 8);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vdl, flo_1), vx_lo1, acc);

            // Convert high nibble int8 values to float (two groups of 8)
            __m128i hi_0 = _mm_cvtepi8_epi32(hi_vals);
            __m128i hi_1 = _mm_cvtepi8_epi32(_mm_srli_si128(hi_vals, 4));
            __m256 fhi_0 = _mm256_cvtepi32_ps(_mm256_set_m128i(hi_1, hi_0));
            __m256 vx_hi0 = _mm256_loadu_ps(xsub + 16);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vdl, fhi_0), vx_hi0, acc);

            __m128i hi_2 = _mm_cvtepi8_epi32(_mm_srli_si128(hi_vals, 8));
            __m128i hi_3 = _mm_cvtepi8_epi32(_mm_srli_si128(hi_vals, 12));
            __m256 fhi_1 = _mm256_cvtepi32_ps(_mm256_set_m128i(hi_3, hi_2));
            __m256 vx_hi1 = _mm256_loadu_ps(xsub + 24);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vdl, fhi_1), vx_hi1, acc);
        }
    }
    return hsum256_ps(acc);
}

void vec_dot_iq4_xs_batch(const uint8_t* data, const float* x, int n,
                           float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_iq4_xs(data + (size_t)r * bpr, x, n);
    }
}

// ── IQ3_XXS: 256 values per 98-byte block ──────────────────────
// iq3xxs_grid[256]: each entry is 4 packed uint8 values
static const uint32_t iq3xxs_grid_c[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

static const uint8_t ksigns_c[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

float vec_dot_iq3_xxs(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();

    const __m256i bitmask = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
    const __m256i sign_flip = _mm256_set1_epi32(0x80000000);

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 98;
        float d = f16_to_f32(*(const uint16_t*)block);
        const uint8_t* grid_idx = block + 2;
        const uint8_t* ss = block + 2 + 64;
        const float* xp = x + b * 256;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32 = *(const uint32_t*)(ss + ib32 * 4);
            float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            __m256 vdb = _mm256_set1_ps(db);

            for (int l = 0; l < 4; l++) {
                uint8_t signIdx = (aux32 >> (7 * l)) & 127;
                uint8_t signs = ksigns_c[signIdx];
                uint32_t g1 = iq3xxs_grid_c[grid_idx[ib32*8 + 2*l]];
                uint32_t g2 = iq3xxs_grid_c[grid_idx[ib32*8 + 2*l + 1]];

                // SIMD byte unpack: extract 8 grid values from 2 uint32s
                uint64_t g12;
                memcpy(&g12, &g1, 4);
                uint32_t g2_copy = g2;
                memcpy((char*)&g12 + 4, &g2_copy, 4);
                __m128i vg_bytes = _mm_set_epi64x(0, g12);
                __m128i vg_lo = _mm_cvtepu8_epi32(vg_bytes);
                __m128i vg_hi = _mm_cvtepu8_epi32(_mm_srli_si128(vg_bytes, 4));
                __m256i vg = _mm256_set_m128i(vg_hi, vg_lo);
                __m256 vgf = _mm256_cvtepi32_ps(vg);

                // SIMD sign application: negate elements where sign bit is set
                __m256i vsigns = _mm256_set1_epi32(signs);
                __m256i vflags = _mm256_and_si256(vsigns, bitmask);
                __m256i vcmp = _mm256_cmpeq_epi32(vflags, bitmask);
                __m256i vxor = _mm256_and_si256(vcmp, sign_flip);
                __m256 vvals = _mm256_castsi256_ps(
                    _mm256_xor_si256(_mm256_castps_si256(vgf), vxor));

                __m256 vx = _mm256_loadu_ps(xp + ib32*32 + l*8);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vdb, vvals), vx, acc);
            }
        }
    }
    return hsum256_ps(acc);
}

void vec_dot_iq3_xxs_batch(const uint8_t* data, const float* x, int n,
                            float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_iq3_xxs(data + (size_t)r * bpr, x, n);
    }
}

// ── IQ4_NL: 32 values per 18-byte block ────────────────────────
// Layout: d:f16(2) + qs[16] (nibble pairs indexing kvalues_iq4nl)
// Same as IQ4_XS but without per-sub-block scales.
float vec_dot_iq4_nl(const uint8_t* data, const float* x, int n) {
    int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();

    __m128i lut = _mm_loadu_si128((const __m128i*)kvalues_iq4nl_c);
    const __m128i lo_mask = _mm_set1_epi8(0x0F);

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 18;
        float d = f16_to_f32(*(const uint16_t*)block);
        const float* xp = x + b * 32;
        __m256 vd = _mm256_set1_ps(d);

        __m128i qbytes = _mm_loadu_si128((const __m128i*)(block + 2));

        __m128i lo_nibbles = _mm_and_si128(qbytes, lo_mask);
        __m128i lo_vals = _mm_shuffle_epi8(lut, lo_nibbles);
        __m128i hi_nibbles = _mm_and_si128(_mm_srli_epi16(qbytes, 4), lo_mask);
        __m128i hi_vals = _mm_shuffle_epi8(lut, hi_nibbles);

        // Low nibbles: values 0-7
        __m128i lo_0 = _mm_cvtepi8_epi32(lo_vals);
        __m128i lo_1 = _mm_cvtepi8_epi32(_mm_srli_si128(lo_vals, 4));
        __m256 flo_0 = _mm256_cvtepi32_ps(_mm256_set_m128i(lo_1, lo_0));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, flo_0), _mm256_loadu_ps(xp), acc);

        // Low nibbles: values 8-15
        __m128i lo_2 = _mm_cvtepi8_epi32(_mm_srli_si128(lo_vals, 8));
        __m128i lo_3 = _mm_cvtepi8_epi32(_mm_srli_si128(lo_vals, 12));
        __m256 flo_1 = _mm256_cvtepi32_ps(_mm256_set_m128i(lo_3, lo_2));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, flo_1), _mm256_loadu_ps(xp + 8), acc);

        // High nibbles: values 16-23
        __m128i hi_0 = _mm_cvtepi8_epi32(hi_vals);
        __m128i hi_1 = _mm_cvtepi8_epi32(_mm_srli_si128(hi_vals, 4));
        __m256 fhi_0 = _mm256_cvtepi32_ps(_mm256_set_m128i(hi_1, hi_0));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, fhi_0), _mm256_loadu_ps(xp + 16), acc);

        // High nibbles: values 24-31
        __m128i hi_2 = _mm_cvtepi8_epi32(_mm_srli_si128(hi_vals, 8));
        __m128i hi_3 = _mm_cvtepi8_epi32(_mm_srli_si128(hi_vals, 12));
        __m256 fhi_1 = _mm256_cvtepi32_ps(_mm256_set_m128i(hi_3, hi_2));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, fhi_1), _mm256_loadu_ps(xp + 24), acc);
    }
    return hsum256_ps(acc);
}

void vec_dot_iq4_nl_batch(const uint8_t* data, const float* x, int n,
                           float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_iq4_nl(data + (size_t)r * bpr, x, n);
    }
}

// ── IQ3_S: 256 values per 110-byte block ────────────────────────
// Layout: d:f16(2) + qs[64] + qh[8] + signs[32] + scales[4]
// Grid: iq3s_grid_c[512] (9-bit index) → uint32 with 4 packed values
float vec_dot_iq3_s(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();

    const __m256i bitmask = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
    const __m256i sign_flip = _mm256_set1_epi32(0x80000000);

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 110;
        float d = f16_to_f32(*(const uint16_t*)block);
        const uint8_t* qs = block + 2;
        const uint8_t* qh = block + 2 + 64;
        const uint8_t* signs = block + 2 + 64 + 8;
        const uint8_t* sc = block + 2 + 64 + 8 + 32;
        const float* xp = x + b * 256;

        int qsOff = 0;
        int signsOff = 0;

        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            uint8_t scByte = sc[ib32 / 2];
            float db1 = d * (float)(1 + 2 * (int)(scByte & 0xf));
            float db2 = d * (float)(1 + 2 * (int)(scByte >> 4));
            __m256 vdb1 = _mm256_set1_ps(db1);
            __m256 vdb2 = _mm256_set1_ps(db2);

            uint8_t qh0 = qh[ib32];
            uint8_t qh1 = qh[ib32 + 1];

            // First group of 32 values
            for (int l = 0; l < 4; l++) {
                uint16_t gi1 = (uint16_t)qs[qsOff + 2*l] | (((uint16_t)qh0 << (8 - 2*l)) & 256);
                uint16_t gi2 = (uint16_t)qs[qsOff + 2*l + 1] | (((uint16_t)qh0 << (7 - 2*l)) & 256);
                uint32_t g1 = iq3s_grid_c[gi1];
                uint32_t g2 = iq3s_grid_c[gi2];
                uint8_t signByte = signs[signsOff + l];

                uint64_t g12;
                memcpy(&g12, &g1, 4);
                memcpy((char*)&g12 + 4, &g2, 4);
                __m128i vg_bytes = _mm_set_epi64x(0, g12);
                __m128i vg_lo = _mm_cvtepu8_epi32(vg_bytes);
                __m128i vg_hi = _mm_cvtepu8_epi32(_mm_srli_si128(vg_bytes, 4));
                __m256i vg = _mm256_set_m128i(vg_hi, vg_lo);
                __m256 vgf = _mm256_cvtepi32_ps(vg);

                __m256i vsigns = _mm256_set1_epi32(signByte);
                __m256i vflags = _mm256_and_si256(vsigns, bitmask);
                __m256i vcmp = _mm256_cmpeq_epi32(vflags, bitmask);
                __m256i vxor = _mm256_and_si256(vcmp, sign_flip);
                __m256 vvals = _mm256_castsi256_ps(
                    _mm256_xor_si256(_mm256_castps_si256(vgf), vxor));

                __m256 vx = _mm256_loadu_ps(xp + ib32*32 + l*8);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vdb1, vvals), vx, acc);
            }
            qsOff += 8;
            signsOff += 4;

            // Second group of 32 values
            for (int l = 0; l < 4; l++) {
                uint16_t gi1 = (uint16_t)qs[qsOff + 2*l] | (((uint16_t)qh1 << (8 - 2*l)) & 256);
                uint16_t gi2 = (uint16_t)qs[qsOff + 2*l + 1] | (((uint16_t)qh1 << (7 - 2*l)) & 256);
                uint32_t g1 = iq3s_grid_c[gi1];
                uint32_t g2 = iq3s_grid_c[gi2];
                uint8_t signByte = signs[signsOff + l];

                uint64_t g12;
                memcpy(&g12, &g1, 4);
                memcpy((char*)&g12 + 4, &g2, 4);
                __m128i vg_bytes = _mm_set_epi64x(0, g12);
                __m128i vg_lo = _mm_cvtepu8_epi32(vg_bytes);
                __m128i vg_hi = _mm_cvtepu8_epi32(_mm_srli_si128(vg_bytes, 4));
                __m256i vg = _mm256_set_m128i(vg_hi, vg_lo);
                __m256 vgf = _mm256_cvtepi32_ps(vg);

                __m256i vsigns = _mm256_set1_epi32(signByte);
                __m256i vflags = _mm256_and_si256(vsigns, bitmask);
                __m256i vcmp = _mm256_cmpeq_epi32(vflags, bitmask);
                __m256i vxor = _mm256_and_si256(vcmp, sign_flip);
                __m256 vvals = _mm256_castsi256_ps(
                    _mm256_xor_si256(_mm256_castps_si256(vgf), vxor));

                __m256 vx = _mm256_loadu_ps(xp + (ib32+1)*32 + l*8);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vdb2, vvals), vx, acc);
            }
            qsOff += 8;
            signsOff += 4;
        }
    }
    return hsum256_ps(acc);
}

void vec_dot_iq3_s_batch(const uint8_t* data, const float* x, int n,
                          float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_iq3_s(data + (size_t)r * bpr, x, n);
    }
}

// ── IQ2_XXS: 256 values per 66-byte block ──────────────────────
// Layout: d:f16(2) + qs[64] (8 groups × 8 bytes each)
// Grid: iq2xxs_grid_c[256] → uint64 with 8 packed values
float vec_dot_iq2_xxs(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();

    const __m256i bitmask = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
    const __m256i sign_flip = _mm256_set1_epi32(0x80000000);

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 66;
        float d = f16_to_f32(*(const uint16_t*)block);
        const uint8_t* qs = block + 2;
        const float* xp = x + b * 256;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            const uint8_t* grp = qs + ib32 * 8;
            uint32_t aux1;
            memcpy(&aux1, grp + 4, 4);

            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            __m256 vdb = _mm256_set1_ps(db);

            for (int l = 0; l < 4; l++) {
                uint8_t gridIdx = grp[l];
                uint64_t grid = iq2xxs_grid_c[gridIdx];

                uint8_t signIdx = (aux1 >> (7 * l)) & 127;
                uint8_t signs8 = ksigns_c[signIdx];

                __m128i vg_bytes = _mm_set_epi64x(0, grid);
                __m128i vg_lo = _mm_cvtepu8_epi32(vg_bytes);
                __m128i vg_hi = _mm_cvtepu8_epi32(_mm_srli_si128(vg_bytes, 4));
                __m256i vg = _mm256_set_m128i(vg_hi, vg_lo);
                __m256 vgf = _mm256_cvtepi32_ps(vg);

                __m256i vsigns = _mm256_set1_epi32(signs8);
                __m256i vflags = _mm256_and_si256(vsigns, bitmask);
                __m256i vcmp = _mm256_cmpeq_epi32(vflags, bitmask);
                __m256i vxor = _mm256_and_si256(vcmp, sign_flip);
                __m256 vvals = _mm256_castsi256_ps(
                    _mm256_xor_si256(_mm256_castps_si256(vgf), vxor));

                __m256 vx = _mm256_loadu_ps(xp + ib32*32 + l*8);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vdb, vvals), vx, acc);
            }
        }
    }
    return hsum256_ps(acc);
}

void vec_dot_iq2_xxs_batch(const uint8_t* data, const float* x, int n,
                             float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_iq2_xxs(data + (size_t)r * bpr, x, n);
    }
}

// ── IQ1_S: 256 values per 50-byte block ────────────────────────
// Layout: d:f16(2) + qs[32] + qh[16] (8×uint16)
// Grid: iq1s_grid_c[2048] → uint64 with 8 packed signed int8 {-1,0,+1}
// Delta: ±0.125 applied to all grid values in a sub-group
float vec_dot_iq1_s(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 50;
        float d = f16_to_f32(*(const uint16_t*)block);
        const uint8_t* qs = block + 2;
        const uint8_t* qhp = block + 2 + 32;
        const float* xp = x + b * 256;

        int qsOff = 0;

        for (int ib = 0; ib < 8; ib++) {
            uint16_t qh = (uint16_t)qhp[ib*2] | ((uint16_t)qhp[ib*2+1] << 8);

            float dl = d * (float)(2 * (int)((qh >> 12) & 7) + 1);
            float delta = (qh & 0x8000) ? -0.125f : 0.125f;
            __m256 vdl = _mm256_set1_ps(dl);
            __m256 vdelta = _mm256_set1_ps(dl * delta);

            for (int l = 0; l < 4; l++) {
                uint16_t gridIdx = (uint16_t)qs[qsOff + l] |
                                   (((qh >> (3*l)) & 7) << 8);
                uint64_t grid = iq1s_grid_c[gridIdx];

                __m128i vg_bytes = _mm_set_epi64x(0, grid);
                __m128i vg_lo_i8 = _mm_cvtepi8_epi32(vg_bytes);
                __m128i vg_hi_i8 = _mm_cvtepi8_epi32(_mm_srli_si128(vg_bytes, 4));
                __m256i vg = _mm256_set_m128i(vg_hi_i8, vg_lo_i8);
                __m256 vgf = _mm256_cvtepi32_ps(vg);

                // val = dl * (gridVal + delta) = dl*gridVal + dl*delta
                __m256 vx = _mm256_loadu_ps(xp + ib*32 + l*8);
                __m256 vdlg = _mm256_fmadd_ps(vdl, vgf, vdelta);
                acc = _mm256_fmadd_ps(vdlg, vx, acc);
            }
            qsOff += 4;
        }
    }
    return hsum256_ps(acc);
}

void vec_dot_iq1_s_batch(const uint8_t* data, const float* x, int n,
                          float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_iq1_s(data + (size_t)r * bpr, x, n);
    }
}

// ── IQ2_S: 256 values per 82-byte block ─────────────────────────
// Layout: d:f16(2) + qs[32](grid low) + signs[32] + qh[8] + scales[8]
// Grid: iq2s_grid_c[1024] (10-bit index) → uint64 with 8 packed values
float vec_dot_iq2_s(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();

    const __m256i bitmask = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
    const __m256i sign_flip = _mm256_set1_epi32(0x80000000);

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 82;
        float d = f16_to_f32(*(const uint16_t*)block);
        const uint8_t* gridLow = block + 2;
        const uint8_t* signBytes = block + 2 + 32;
        const uint8_t* qhp = block + 2 + 64;
        const uint8_t* scp = block + 2 + 64 + 8;
        const float* xp = x + b * 256;

        int glOff = 0;
        int sOff = 0;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint8_t scByte = scp[ib32];
            float db0 = d * (0.5f + (float)(scByte & 0xf)) * 0.25f;
            float db1 = d * (0.5f + (float)(scByte >> 4)) * 0.25f;
            uint8_t qhByte = qhp[ib32];

            for (int l = 0; l < 4; l++) {
                uint16_t lowByte = gridLow[glOff];
                uint16_t highBits = ((uint16_t)qhByte << (8 - 2*l)) & 0x300;
                uint16_t gridIdx = lowByte | highBits;
                uint64_t grid = iq2s_grid_c[gridIdx];
                uint8_t signs8 = signBytes[sOff];

                float db = (l < 2) ? db0 : db1;

                __m128i vg_bytes = _mm_set_epi64x(0, grid);
                __m128i vg_lo = _mm_cvtepu8_epi32(vg_bytes);
                __m128i vg_hi = _mm_cvtepu8_epi32(_mm_srli_si128(vg_bytes, 4));
                __m256i vg = _mm256_set_m128i(vg_hi, vg_lo);
                __m256 vgf = _mm256_cvtepi32_ps(vg);

                __m256i vsigns = _mm256_set1_epi32(signs8);
                __m256i vflags = _mm256_and_si256(vsigns, bitmask);
                __m256i vcmp = _mm256_cmpeq_epi32(vflags, bitmask);
                __m256i vxor = _mm256_and_si256(vcmp, sign_flip);
                __m256 vvals = _mm256_castsi256_ps(
                    _mm256_xor_si256(_mm256_castps_si256(vgf), vxor));

                __m256 vdb = _mm256_set1_ps(db);
                __m256 vx = _mm256_loadu_ps(xp + ib32*32 + l*8);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vdb, vvals), vx, acc);

                glOff++;
                sOff++;
            }
        }
    }
    return hsum256_ps(acc);
}

void vec_dot_iq2_s_batch(const uint8_t* data, const float* x, int n,
                          float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_iq2_s(data + (size_t)r * bpr, x, n);
    }
}

// ── MXFP4: 32 values per 17-byte block ─────────────────────────
// Layout: e:E8M0(1) + qs[16] (nibble pairs indexing kvalues_mxfp4)
// Low nibbles → values[0..15], high nibbles → values[16..31]
static const int8_t kvalues_mxfp4_c[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static inline float e8m0_to_fp32_half(uint8_t e) {
    uint32_t bits;
    if (e < 2) {
        bits = 0x00200000u << e;
    } else {
        bits = (uint32_t)(e - 1) << 23;
    }
    float result;
    memcpy(&result, &bits, 4);
    return result;
}

float vec_dot_mxfp4(const uint8_t* data, const float* x, int n) {
    int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();

    __m128i lut = _mm_loadu_si128((const __m128i*)kvalues_mxfp4_c);
    const __m128i lo_mask = _mm_set1_epi8(0x0F);

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 17;
        float d = e8m0_to_fp32_half(block[0]);
        const float* xp = x + b * 32;
        __m256 vd = _mm256_set1_ps(d);

        __m128i qbytes = _mm_loadu_si128((const __m128i*)(block + 1));

        __m128i lo_nibbles = _mm_and_si128(qbytes, lo_mask);
        __m128i lo_vals = _mm_shuffle_epi8(lut, lo_nibbles);
        __m128i hi_nibbles = _mm_and_si128(_mm_srli_epi16(qbytes, 4), lo_mask);
        __m128i hi_vals = _mm_shuffle_epi8(lut, hi_nibbles);

        __m128i lo_0 = _mm_cvtepi8_epi32(lo_vals);
        __m128i lo_1 = _mm_cvtepi8_epi32(_mm_srli_si128(lo_vals, 4));
        __m256 flo_0 = _mm256_cvtepi32_ps(_mm256_set_m128i(lo_1, lo_0));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, flo_0), _mm256_loadu_ps(xp), acc);

        __m128i lo_2 = _mm_cvtepi8_epi32(_mm_srli_si128(lo_vals, 8));
        __m128i lo_3 = _mm_cvtepi8_epi32(_mm_srli_si128(lo_vals, 12));
        __m256 flo_1 = _mm256_cvtepi32_ps(_mm256_set_m128i(lo_3, lo_2));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, flo_1), _mm256_loadu_ps(xp + 8), acc);

        __m128i hi_0 = _mm_cvtepi8_epi32(hi_vals);
        __m128i hi_1 = _mm_cvtepi8_epi32(_mm_srli_si128(hi_vals, 4));
        __m256 fhi_0 = _mm256_cvtepi32_ps(_mm256_set_m128i(hi_1, hi_0));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, fhi_0), _mm256_loadu_ps(xp + 16), acc);

        __m128i hi_2 = _mm_cvtepi8_epi32(_mm_srli_si128(hi_vals, 8));
        __m128i hi_3 = _mm_cvtepi8_epi32(_mm_srli_si128(hi_vals, 12));
        __m256 fhi_1 = _mm256_cvtepi32_ps(_mm256_set_m128i(hi_3, hi_2));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, fhi_1), _mm256_loadu_ps(xp + 24), acc);
    }
    return hsum256_ps(acc);
}

void vec_dot_mxfp4_batch(const uint8_t* data, const float* x, int n,
                          float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_mxfp4(data + (size_t)r * bpr, x, n);
    }
}

#define DEFINE_DOT_MANY(name) \
void name##_many(const uint8_t* data, const float* x_flat, int n, float* out, int nvecs) { \
    for (int v = 0; v < nvecs; v++) { \
        out[v] = name(data, x_flat + (size_t)v * n, n); \
    } \
}

#define DEFINE_DOT_MANY_ROWS(name) \
void name##_many_rows(const uint8_t* data, const float* x_flat, int n, float* out, int nrows, int bpr, int nvecs) { \
    for (int r = 0; r < nrows; r++) { \
        name##_many(data + (size_t)r * bpr, x_flat, n, out + (size_t)r * nvecs, nvecs); \
    } \
}

DEFINE_DOT_MANY(vec_dot_f16)
DEFINE_DOT_MANY(vec_dot_q4_0)
DEFINE_DOT_MANY(vec_dot_q4_1)
DEFINE_DOT_MANY(vec_dot_q5_0)
DEFINE_DOT_MANY(vec_dot_q5_1)
DEFINE_DOT_MANY(vec_dot_q8_0)
DEFINE_DOT_MANY(vec_dot_q2_k)
DEFINE_DOT_MANY(vec_dot_q3_k)
DEFINE_DOT_MANY(vec_dot_q5_k)

void vec_dot_q4_k_many(const uint8_t* data, const float* x_flat, int n, float* out, int nvecs) {
    int v = 0;
    for (; v + 3 < nvecs; v += 4) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        const float* x0 = x_flat + (size_t)(v + 0) * n;
        const float* x1 = x_flat + (size_t)(v + 1) * n;
        const float* x2 = x_flat + (size_t)(v + 2) * n;
        const float* x3 = x_flat + (size_t)(v + 3) * n;
        int nb = n / 256;

        for (int block = 0; block < nb; block++) {
            const uint8_t* bp = data + block * 144;
            float d = f16_to_f32(*(const uint16_t*)bp);
            float dmin = f16_to_f32(*(const uint16_t*)(bp + 2));
            const uint8_t* sp = bp + 4;
            float sc[8], mn[8];
            for (int i = 0; i < 4; i++) {
                sc[i] = d * (float)(sp[i] & 0x3F);
                mn[i] = dmin * (float)(sp[4 + i] & 0x3F);
            }
            for (int i = 0; i < 4; i++) {
                uint8_t sc_hi = (sp[i] >> 6) & 0x03;
                uint8_t mn_hi = (sp[4 + i] >> 6) & 0x03;
                uint8_t sc_lo = sp[8 + i] & 0x0F;
                uint8_t mn_lo = (sp[8 + i] >> 4) & 0x0F;
                sc[4 + i] = d * (float)(sc_lo | (sc_hi << 4));
                mn[4 + i] = dmin * (float)(mn_lo | (mn_hi << 4));
            }

            const uint8_t* qp = bp + 16;
            const float* xp0 = x0 + block * 256;
            const float* xp1 = x1 + block * 256;
            const float* xp2 = x2 + block * 256;
            const float* xp3 = x3 + block * 256;
            int xi = 0;
            int is = 0;

            for (int grp = 0; grp < 4; grp++) {
                __m256 vd1 = _mm256_set1_ps(sc[is]);
                __m256 vm1 = _mm256_set1_ps(mn[is]);
                __m256 vd2 = _mm256_set1_ps(sc[is + 1]);
                __m256 vm2 = _mm256_set1_ps(mn[is + 1]);
                const uint8_t* qoff = qp + grp * 32;

                for (int l = 0; l < 32; l += 8) {
                    __m128i bytes = _mm_loadl_epi64((const __m128i*)(qoff + l));
                    __m128i lo = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
                    __m256 val = _mm256_fmsub_ps(vd1, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), vm1);
                    acc0 = _mm256_fmadd_ps(val, _mm256_loadu_ps(xp0 + xi + l), acc0);
                    acc1 = _mm256_fmadd_ps(val, _mm256_loadu_ps(xp1 + xi + l), acc1);
                    acc2 = _mm256_fmadd_ps(val, _mm256_loadu_ps(xp2 + xi + l), acc2);
                    acc3 = _mm256_fmadd_ps(val, _mm256_loadu_ps(xp3 + xi + l), acc3);
                }
                xi += 32;

                for (int l = 0; l < 32; l += 8) {
                    __m128i bytes = _mm_loadl_epi64((const __m128i*)(qoff + l));
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(0x0F));
                    __m256 val = _mm256_fmsub_ps(vd2, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), vm2);
                    acc0 = _mm256_fmadd_ps(val, _mm256_loadu_ps(xp0 + xi + l), acc0);
                    acc1 = _mm256_fmadd_ps(val, _mm256_loadu_ps(xp1 + xi + l), acc1);
                    acc2 = _mm256_fmadd_ps(val, _mm256_loadu_ps(xp2 + xi + l), acc2);
                    acc3 = _mm256_fmadd_ps(val, _mm256_loadu_ps(xp3 + xi + l), acc3);
                }
                xi += 32;
                is += 2;
            }
        }

        out[v + 0] = hsum256_ps(acc0);
        out[v + 1] = hsum256_ps(acc1);
        out[v + 2] = hsum256_ps(acc2);
        out[v + 3] = hsum256_ps(acc3);
    }

    for (; v < nvecs; v++) {
        out[v] = vec_dot_q4_k(data, x_flat + (size_t)v * n, n);
    }
}

void vec_dot_q6_k_many(const uint8_t* data, const float* x_flat, int n, float* out, int nvecs) {
    int v = 0;
    for (; v + 3 < nvecs; v += 4) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        const float* x0 = x_flat + (size_t)(v + 0) * n;
        const float* x1 = x_flat + (size_t)(v + 1) * n;
        const float* x2 = x_flat + (size_t)(v + 2) * n;
        const float* x3 = x_flat + (size_t)(v + 3) * n;
        int nb = n / 256;

        for (int block = 0; block < nb; block++) {
            const uint8_t* bp = data + block * 210;
            float d_val = f16_to_f32(*(const uint16_t*)(bp + 208));
            const uint8_t* ql = bp;
            const uint8_t* qh_base = bp + 128;
            const int8_t* scales = (const int8_t*)(bp + 192);
            __m256 vd = _mm256_set1_ps(d_val);
            __m256i thirty_two = _mm256_set1_epi32(32);
            const float* xp0 = x0 + block * 256;
            const float* xp1 = x1 + block * 256;
            const float* xp2 = x2 + block * 256;
            const float* xp3 = x3 + block * 256;

            for (int n128 = 0; n128 < 2; n128++) {
                const uint8_t* qlp = ql + n128 * 64;
                const uint8_t* qhp = qh_base + n128 * 32;
                int xoff = n128 * 128;

                for (int l = 0; l < 32; l += 8) {
                    int sidx = 8 * n128 + l / 16;
                    __m128i ql0_raw = _mm_loadl_epi64((const __m128i*)(qlp + l));
                    __m128i ql32_raw = _mm_loadl_epi64((const __m128i*)(qlp + l + 32));
                    __m128i qh_raw = _mm_loadl_epi64((const __m128i*)(qhp + l));
                    __m128i mask4 = _mm_set1_epi8(0x0F);
                    __m128i mask2 = _mm_set1_epi8(0x03);

                    __m128i q1_lo = _mm_and_si128(ql0_raw, mask4);
                    __m128i q1_hi = _mm_slli_epi16(_mm_and_si128(qh_raw, mask2), 4);
                    __m256 fq1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_or_si128(q1_lo, q1_hi)), thirty_two));

                    __m128i q2_lo = _mm_and_si128(ql32_raw, mask4);
                    __m128i q2_hi = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_raw, 2), mask2), 4);
                    __m256 fq2 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_or_si128(q2_lo, q2_hi)), thirty_two));

                    __m128i q3_lo = _mm_and_si128(_mm_srli_epi16(ql0_raw, 4), mask4);
                    __m128i q3_hi = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_raw, 4), mask2), 4);
                    __m256 fq3 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_or_si128(q3_lo, q3_hi)), thirty_two));

                    __m128i q4_lo = _mm_and_si128(_mm_srli_epi16(ql32_raw, 4), mask4);
                    __m128i q4_hi = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_raw, 6), mask2), 4);
                    __m256 fq4 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_or_si128(q4_lo, q4_hi)), thirty_two));

                    __m256 mul0 = _mm256_mul_ps(_mm256_mul_ps(vd, _mm256_set1_ps((float)scales[sidx])), fq1);
                    __m256 mul2 = _mm256_mul_ps(_mm256_mul_ps(vd, _mm256_set1_ps((float)scales[sidx + 2])), fq2);
                    __m256 mul4 = _mm256_mul_ps(_mm256_mul_ps(vd, _mm256_set1_ps((float)scales[sidx + 4])), fq3);
                    __m256 mul6 = _mm256_mul_ps(_mm256_mul_ps(vd, _mm256_set1_ps((float)scales[sidx + 6])), fq4);

                    acc0 = _mm256_fmadd_ps(mul0, _mm256_loadu_ps(xp0 + xoff + l), acc0);
                    acc1 = _mm256_fmadd_ps(mul0, _mm256_loadu_ps(xp1 + xoff + l), acc1);
                    acc2 = _mm256_fmadd_ps(mul0, _mm256_loadu_ps(xp2 + xoff + l), acc2);
                    acc3 = _mm256_fmadd_ps(mul0, _mm256_loadu_ps(xp3 + xoff + l), acc3);

                    acc0 = _mm256_fmadd_ps(mul2, _mm256_loadu_ps(xp0 + xoff + l + 32), acc0);
                    acc1 = _mm256_fmadd_ps(mul2, _mm256_loadu_ps(xp1 + xoff + l + 32), acc1);
                    acc2 = _mm256_fmadd_ps(mul2, _mm256_loadu_ps(xp2 + xoff + l + 32), acc2);
                    acc3 = _mm256_fmadd_ps(mul2, _mm256_loadu_ps(xp3 + xoff + l + 32), acc3);

                    acc0 = _mm256_fmadd_ps(mul4, _mm256_loadu_ps(xp0 + xoff + l + 64), acc0);
                    acc1 = _mm256_fmadd_ps(mul4, _mm256_loadu_ps(xp1 + xoff + l + 64), acc1);
                    acc2 = _mm256_fmadd_ps(mul4, _mm256_loadu_ps(xp2 + xoff + l + 64), acc2);
                    acc3 = _mm256_fmadd_ps(mul4, _mm256_loadu_ps(xp3 + xoff + l + 64), acc3);

                    acc0 = _mm256_fmadd_ps(mul6, _mm256_loadu_ps(xp0 + xoff + l + 96), acc0);
                    acc1 = _mm256_fmadd_ps(mul6, _mm256_loadu_ps(xp1 + xoff + l + 96), acc1);
                    acc2 = _mm256_fmadd_ps(mul6, _mm256_loadu_ps(xp2 + xoff + l + 96), acc2);
                    acc3 = _mm256_fmadd_ps(mul6, _mm256_loadu_ps(xp3 + xoff + l + 96), acc3);
                }
            }
        }

        out[v + 0] = hsum256_ps(acc0);
        out[v + 1] = hsum256_ps(acc1);
        out[v + 2] = hsum256_ps(acc2);
        out[v + 3] = hsum256_ps(acc3);
    }

    for (; v < nvecs; v++) {
        out[v] = vec_dot_q6_k(data, x_flat + (size_t)v * n, n);
    }
}

DEFINE_DOT_MANY_ROWS(vec_dot_f16)
DEFINE_DOT_MANY_ROWS(vec_dot_q4_0)
DEFINE_DOT_MANY_ROWS(vec_dot_q4_1)
DEFINE_DOT_MANY_ROWS(vec_dot_q5_0)
DEFINE_DOT_MANY_ROWS(vec_dot_q5_1)
DEFINE_DOT_MANY_ROWS(vec_dot_q8_0)
DEFINE_DOT_MANY_ROWS(vec_dot_q2_k)
DEFINE_DOT_MANY_ROWS(vec_dot_q3_k)
DEFINE_DOT_MANY_ROWS(vec_dot_q4_k)
DEFINE_DOT_MANY_ROWS(vec_dot_q5_k)
DEFINE_DOT_MANY_ROWS(vec_dot_q6_k)

// ── AVX2 fast exp approximation ────────────────────────────────
// Range reduction: exp(x) = 2^n * exp(r) where n=round(x/ln2), r=x-n*ln2
// Polynomial minimax for exp(r), r in [-0.5*ln2, 0.5*ln2].
// Accuracy: ~5 decimal digits (sufficient for SiLU activation).
static inline __m256 fast_exp_avx2(__m256 x) {
    // Clamp to avoid overflow/underflow
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    __m256 log2e = _mm256_set1_ps(1.44269504088896f);
    __m256 ln2   = _mm256_set1_ps(0.6931471805599453f);

    // n = round(x / ln2)
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, log2e),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // r = x - n*ln2
    __m256 r = _mm256_fnmadd_ps(n, ln2, x);

    // Horner's: exp(r) ≈ 1 + r(1 + r(0.5 + r(1/6 + r/24)))
    __m256 p = _mm256_set1_ps(1.0f / 24.0f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));

    // 2^n via IEEE 754 bit manipulation
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);
    __m256 pow2n = _mm256_castsi256_ps(ni);

    return _mm256_mul_ps(p, pow2n);
}

// ── Fused SwiGLU: out[i] = SiLU(gate[i]) * up[i] ─────────────
// SiLU(x) = x / (1 + exp(-x))
// Uses fast AVX2 exp + Newton-Raphson reciprocal for high throughput.
void vec_swiglu(float* out, const float* gate, const float* up, int n) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_one = _mm256_set1_ps(-1.0f);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        // exp(-g)
        __m256 neg_g = _mm256_mul_ps(neg_one, g);
        __m256 exp_neg_g = fast_exp_avx2(neg_g);
        // sigmoid(g) = 1 / (1 + exp(-g)) — fast reciprocal + Newton-Raphson
        __m256 denom = _mm256_add_ps(one, exp_neg_g);
        __m256 rcp = _mm256_rcp_ps(denom);
        // One Newton-Raphson step: rcp = rcp * (2 - denom*rcp)
        __m256 two = _mm256_set1_ps(2.0f);
        rcp = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, two));
        // SiLU(g) * up = g * sigmoid(g) * up
        __m256 result = _mm256_mul_ps(_mm256_mul_ps(g, rcp), u);
        _mm256_storeu_ps(out + i, result);
    }
    for (; i < n; i++) {
        float g = gate[i];
        float eg = expf(-g);
        out[i] = g / (1.0f + eg) * up[i];
    }
}

// ── AVX2 Softmax: out[i] = exp(x[i]-max) / sum(exp(x[j]-max)) ──
// In-place softmax using fast exp approximation.
void vec_softmax(float* x, int n) {
    // Find max
    int i = 0;
    __m256 vmax = _mm256_set1_ps(-1e30f);
    for (; i <= n - 8; i += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
    }
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_movehdup_ps(lo));
    float maxVal = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        if (x[i] > maxVal) maxVal = x[i];
    }

    // exp(x[i] - max) and sum
    __m256 voff = _mm256_set1_ps(maxVal);
    __m256 vsum = _mm256_setzero_ps();
    i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 v = fast_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(x + i), voff));
        _mm256_storeu_ps(x + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }
    hi = _mm256_extractf128_ps(vsum, 1);
    lo = _mm256_castps256_ps128(vsum);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }

    // Normalize
    __m256 vinv = _mm256_set1_ps(1.0f / sum);
    i = 0;
    for (; i <= n - 8; i += 8) {
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
    }
    for (; i < n; i++) {
        x[i] /= sum;
    }
}

// ── Float32 dot product using AVX2+FMA ─────────────────────────
// Matches gollm's DotProductAVX2 assembly: processes 16 floats/iter
// with two FMA accumulators, then horizontal sum.
float vec_dot_f32(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    }
    for (; i <= n - 8; i += 8) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void vec_dot_f32_many(const float* a, const float* x_flat, int cols,
                      float* out, int nvecs) {
    for (int v = 0; v < nvecs; v++) {
        out[v] = vec_dot_f32(a, x_flat + (size_t)v * cols, cols);
    }
}

// Batch version: compute dot products for nrows rows of a matrix against x.
// a_flat is [nrows * cols] float32, x is [cols] float32, out is [nrows] float32.
void vec_dot_f32_batch(const float* a_flat, const float* x, int cols,
                       float* out, int nrows) {
    for (int r = 0; r < nrows; r++) {
        out[r] = vec_dot_f32(a_flat + r * cols, x, cols);
    }
}

// ── TQ1_0: 256 values per 54-byte block (ternary 1.6875 bpw) ──
// Layout: qs[48] base-3 encoded (5 trits/byte) + qh[4] (4 trits/byte) + f16 scale
// Values are {-1, 0, +1} * d
//
// Optimized: uses integer SIMD for trit extraction instead of scalar loops.
// For each group of 8 bytes: cvtepu8→epi32, multiply by pow3 (mod 256),
// extract trit via (q*3)>>8, subtract 1, convert to float, FMA with x.

static inline void tq1_extract_and_fma_8(
    const uint8_t* src, int p3, const float* xp, __m256* acc
) {
    __m128i bytes8 = _mm_loadl_epi64((const __m128i*)src);
    __m256i vals = _mm256_cvtepu8_epi32(bytes8);
    __m256i vp3 = _mm256_set1_epi32(p3);
    __m256i mask8 = _mm256_set1_epi32(0xFF);
    __m256i three = _mm256_set1_epi32(3);
    __m256i one = _mm256_set1_epi32(1);

    vals = _mm256_mullo_epi32(vals, vp3);
    vals = _mm256_and_si256(vals, mask8);
    vals = _mm256_mullo_epi32(vals, three);
    vals = _mm256_srli_epi32(vals, 8);
    vals = _mm256_sub_epi32(vals, one);

    __m256 trits = _mm256_cvtepi32_ps(vals);
    __m256 xv = _mm256_loadu_ps(xp);
    *acc = _mm256_fmadd_ps(trits, xv, *acc);
}

float vec_dot_tq1_0(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    static const int pow3[5] = {1, 3, 9, 27, 81};
    __m256 acc = _mm256_setzero_ps();
    float scalar_acc = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t* qs = data + b * 54;
        float d = f16_to_f32(*(const uint16_t*)(qs + 52));
        const float* xp = x + b * 256;

        __m256 block_acc = _mm256_setzero_ps();
        int outIdx = 0;

        // 32-byte chunk: 5 digit passes × 4 groups of 8 = 160 values
        for (int nn = 0; nn < 5; nn++) {
            int p3 = pow3[nn];
            for (int m = 0; m < 32; m += 8) {
                tq1_extract_and_fma_8(qs + m, p3, xp + outIdx, &block_acc);
                outIdx += 8;
            }
        }

        // 16-byte chunk: 5 digit passes × 2 groups of 8 = 80 values
        for (int nn = 0; nn < 5; nn++) {
            int p3 = pow3[nn];
            for (int m = 0; m < 16; m += 8) {
                tq1_extract_and_fma_8(qs + 32 + m, p3, xp + outIdx, &block_acc);
                outIdx += 8;
            }
        }

        // qh: 4 bytes × 4 digit passes = 16 values (scalar, small)
        const uint8_t* qh = qs + 48;
        float block_tail = 0.0f;
        for (int nn = 0; nn < 4; nn++) {
            int p3 = pow3[nn];
            for (int j = 0; j < 4; j++) {
                uint8_t q = (uint8_t)((uint16_t)qh[j] * p3);
                int xi = ((uint16_t)q * 3) >> 8;
                block_tail += (float)(xi - 1) * xp[outIdx];
                outIdx++;
            }
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, block_acc, acc);
        scalar_acc += d * block_tail;
    }
    return hsum256_ps(acc) + scalar_acc;
}

void vec_dot_tq1_0_batch(const uint8_t* data, const float* x, int n,
                          float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_tq1_0(data + (size_t)r * bpr, x, n);
    }
}

// ── TQ2_0: 256 values per 66-byte block (ternary 2.0625 bpw) ──
// Layout: qs[64] 2 bits per value (4 per byte) + f16 scale
// Values are {-1, 0, +1} * d, stored as {0, 1, 2} mapped to {-1, 0, +1}
//
// Optimized: uses integer SIMD for 2-bit extraction instead of scalar loops.
float vec_dot_tq2_0(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();
    __m256i mask2 = _mm256_set1_epi32(3);
    __m256i one = _mm256_set1_epi32(1);

    for (int b = 0; b < nb; b++) {
        const uint8_t* qs = data + b * 66;
        float d = f16_to_f32(*(const uint16_t*)(qs + 64));
        const float* xp = x + b * 256;

        __m256 block_acc = _mm256_setzero_ps();
        int outIdx = 0;

        for (int j = 0; j < 64; j += 32) {
            for (int shift = 0; shift < 4; shift++) {
                int s2 = shift * 2;
                for (int m = 0; m < 32; m += 8) {
                    __m128i bytes8 = _mm_loadl_epi64((const __m128i*)(qs + j + m));
                    __m256i vals = _mm256_cvtepu8_epi32(bytes8);
                    vals = _mm256_srli_epi32(vals, s2);
                    vals = _mm256_and_si256(vals, mask2);
                    vals = _mm256_sub_epi32(vals, one);

                    __m256 trits = _mm256_cvtepi32_ps(vals);
                    __m256 xv = _mm256_loadu_ps(xp + outIdx);
                    block_acc = _mm256_fmadd_ps(trits, xv, block_acc);
                    outIdx += 8;
                }
            }
        }
        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, block_acc, acc);
    }
    return hsum256_ps(acc);
}

void vec_dot_tq2_0_batch(const uint8_t* data, const float* x, int n,
                          float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_tq2_0(data + (size_t)r * bpr, x, n);
    }
}

// ── SIMD scale-add (axpy): out[i] += scale * src[i] ───────────
// Used for attention weighted value accumulation.
void vec_scale_add(float* out, float scale, const float* src, int n) {
    __m256 vs = _mm256_set1_ps(scale);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 o = _mm256_loadu_ps(out + i);
        __m256 s = _mm256_loadu_ps(src + i);
        o = _mm256_fmadd_ps(vs, s, o);
        _mm256_storeu_ps(out + i, o);
    }
    for (; i < n; i++) {
        out[i] += scale * src[i];
    }
}

// ── Fused causal attention for one head ─────────────────────────
// Computes dot(q, k[t]) * scale for t in [0, seqLen), softmax, then
// weighted sum of v[t]. All in a single C call to avoid Go/CGo overhead.
// k_base/v_base point to pre-gathered flat [maxSeqLen][kvDim] buffers.
// kv_offset is kvH*headDim; kv_stride is kvDim (distance between timesteps).
// scores is a scratch buffer of at least seqLen floats.
void causal_attn_head(
    const float* q, int head_dim,
    const float* k_base, const float* v_base,
    int kv_offset, int kv_stride,
    int seq_len, float scale,
    float* scores,
    float* out)
{
    for (int t = 0; t < seq_len; t++) {
        scores[t] = vec_dot_f32(q, k_base + (size_t)t * kv_stride + kv_offset, head_dim) * scale;
    }

    float maxv = scores[0];
    for (int t = 1; t < seq_len; t++) {
        if (scores[t] > maxv) maxv = scores[t];
    }
    float sum = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        scores[t] = expf(scores[t] - maxv);
        sum += scores[t];
    }
    float inv_sum = 1.0f / sum;
    for (int t = 0; t < seq_len; t++) {
        scores[t] *= inv_sum;
    }

    for (int i = 0; i < head_dim; i++) {
        out[i] = 0.0f;
    }

    for (int t = 0; t < seq_len; t++) {
        vec_scale_add(out, scores[t], v_base + (size_t)t * kv_stride + kv_offset, head_dim);
    }
}
