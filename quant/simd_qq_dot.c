//go:build amd64 && cgo

// simd_qq_dot.c — AVX2+FMA quantized×quantized dot products for GGUF types.
// Ported from ggml (llama.cpp) ggml-cpu-quants.c. Uses _mm256_maddubs_epi16
// for ~4x throughput vs the float-based dequant+dot path.
//
// Key insight: quantize the input activation vector to Q8 format once,
// then compute integer dot products against quantized weight rows.
// _mm256_maddubs_epi16 processes 32 values per instruction (vs 8 for float FMA).

#pragma GCC target("avx2,fma,f16c,avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")
#pragma GCC optimize("O3")

#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// ── Helpers from ggml ───────────────────────────────────────────

static inline float f16_to_f32_qq(uint16_t h) {
    __m128i v = _mm_set1_epi16(h);
    return _mm_cvtss_f32(_mm_cvtph_ps(v));
}

static inline uint16_t f32_to_f16_qq(float f) {
    __m128 v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, 0);
    return (uint16_t)_mm_extract_epi16(h, 0);
}

static inline float hsum_float_8_qq(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline int hsum_i32_4_qq(const __m128i x) {
    __m128i sum = _mm_add_epi32(x, _mm_srli_si128(x, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    return _mm_cvtsi128_si32(sum);
}

static inline float hsum_float_4_qq(const __m128 x) {
    __m128 sum = _mm_add_ps(x, _mm_movehl_ps(x, x));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

static inline int hsum_i32_8_qq(const __m256i x) {
    __m128i lo = _mm256_castsi256_si128(x);
    __m128i hi = _mm256_extracti128_si256(x, 1);
    return hsum_i32_4_qq(_mm_add_epi32(lo, hi));
}

static inline int dot_i8_16_qq(const __m128i x, const __m128i y) {
    const __m128i ax = _mm_sign_epi8(x, x);
    const __m128i sy = _mm_sign_epi8(y, x);
    const __m128i prod16 = _mm_maddubs_epi16(ax, sy);
    const __m128i prod32 = _mm_madd_epi16(prod16, _mm_set1_epi16(1));
    return hsum_i32_4_qq(prod32);
}

#define MM256_SET_M128I_QQ(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// Unpack 16 nibble bytes into 32 bytes in [0..15]
static inline __m256i bytes_from_nibbles_32_qq(const uint8_t * rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I_QQ(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}

// unsigned×signed dot product. Use VNNI when available, otherwise fall back to
// the AVX2 maddubs+madd sequence.
static inline __m256 dotbusd_ps_qq(const __m256i ax, const __m256i sy) {
#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)
    __m256i res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
    return _mm256_cvtepi32_ps(res);
#else
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i prod16 = _mm256_maddubs_epi16(ax, sy);
    __m256i prod32 = _mm256_madd_epi16(prod16, ones);
    return _mm256_cvtepi32_ps(prod32);
#endif
}

static inline __m256 mul_sum_us8_pairs_float_qq(const __m256i ax, const __m256i sy) {
    return dotbusd_ps_qq(ax, sy);
}

// signed×signed via sign trick → float
static inline __m256 mul_sum_i8_pairs_float_qq(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float_qq(ax, sy);
}

// Scale shuffle tables for K-quants
static inline __m256i get_scale_shuffle_k4_qq(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

static inline __m256i get_scale_shuffle_q3k_qq(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13, 14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

static inline __m128i get_scale_shuffle_qq(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15,
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}

// ══════════════════════════════════════════════════════════════════
// Quantize float32 → Q8_0 (32 values per 34-byte block)
// Layout: f16 scale + 32 int8 quants
// ══════════════════════════════════════════════════════════════════

void quantize_to_q8_0(const float* restrict x, uint8_t* restrict y, int n) {
    const int nb = n / 32;
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    for (int i = 0; i < nb; i++) {
        const float* xp = x + i * 32;
        uint8_t* bp = y + i * 34;

        __m256 v0 = _mm256_loadu_ps(xp);
        __m256 v1 = _mm256_loadu_ps(xp + 8);
        __m256 v2 = _mm256_loadu_ps(xp + 16);
        __m256 v3 = _mm256_loadu_ps(xp + 24);

        __m256 maxAbs = _mm256_max_ps(
            _mm256_max_ps(_mm256_andnot_ps(sign_mask, v0), _mm256_andnot_ps(sign_mask, v1)),
            _mm256_max_ps(_mm256_andnot_ps(sign_mask, v2), _mm256_andnot_ps(sign_mask, v3)));

        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        float amax = _mm_cvtss_f32(max4);

        float d = amax / 127.f;
        float id = (amax != 0.0f) ? 127.f / amax : 0.0f;

        *(uint16_t*)bp = f32_to_f16_qq(d);

        __m256 mul = _mm256_set1_ps(id);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, mul));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, mul));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, mul));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, mul));

        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);

        // Fix AVX2 lane-crossing pack order
        i0 = _mm256_permutevar8x32_epi32(i0, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i*)(bp + 2), i0);
    }
}

// ══════════════════════════════════════════════════════════════════
// Quantize float32 → Q8_K (256 values per block)
// Layout: float32 d (4B) + 256 int8 qs + 16 int16 bsums (32B) = 292B
// ══════════════════════════════════════════════════════════════════

void quantize_to_q8_K(const float* restrict x, uint8_t* restrict y, int n) {
    const int nb = n / 256;
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    for (int i = 0; i < nb; i++) {
        const float* xp = x + i * 256;
        uint8_t* bp = y + i * 292;

        // Find max absolute value using AVX2
        __m256 vmax = _mm256_setzero_ps();
        for (int j = 0; j < 256; j += 8) {
            vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(xp + j)));
        }
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(vmax, 1), _mm256_castps256_ps128(vmax));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        float amax = _mm_cvtss_f32(max4);

        if (amax == 0.0f) {
            memset(bp, 0, 292);
            continue;
        }

        // Find the actual value with max absolute value (need sign)
        float max_val = 0.0f;
        for (int j = 0; j < 256; j++) {
            if (fabsf(xp[j]) == amax) { max_val = xp[j]; break; }
        }

        float iscale = -127.f / max_val;
        float d = 1.0f / iscale;
        memcpy(bp, &d, 4);

        int8_t* qs = (int8_t*)(bp + 4);
        int16_t* bsums = (int16_t*)(bp + 4 + 256);
        __m256 vmul = _mm256_set1_ps(iscale);

        // Quantize 32 values at a time using AVX2
        for (int j = 0; j < 256; j += 32) {
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xp + j), vmul));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xp + j + 8), vmul));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xp + j + 16), vmul));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xp + j + 24), vmul));

            i0 = _mm256_packs_epi32(i0, i1);
            i2 = _mm256_packs_epi32(i2, i3);
            __m256i packed = _mm256_packs_epi16(i0, i2);
            packed = _mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256((__m256i*)(qs + j), packed);
        }

        // Compute bsums (sum of each group of 16 int8 values)
        for (int j = 0; j < 16; j++) {
            int sum = 0;
            for (int k = 0; k < 16; k++) sum += qs[j * 16 + k];
            bsums[j] = (int16_t)sum;
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// Q4_0 × Q8_0 dot product (32 values per block pair)
// ══════════════════════════════════════════════════════════════════

float qq_dot_q4_0_q8_0(const uint8_t* restrict xb, const uint8_t* restrict yb, int n) {
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const uint8_t* x = xb + i * 18;  // Q4_0: 18 bytes per block
        const uint8_t* y = yb + i * 34;  // Q8_0: 34 bytes per block

        const __m256 d = _mm256_set1_ps(f16_to_f32_qq(*(const uint16_t*)x) * f16_to_f32_qq(*(const uint16_t*)y));

        __m256i qx = bytes_from_nibbles_32_qq(x + 2);
        qx = _mm256_sub_epi8(qx, _mm256_set1_epi8(8));

        __m256i qy = _mm256_loadu_si256((const __m256i*)(y + 2));

        acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs_float_qq(qx, qy), acc);
    }
    return hsum_float_8_qq(acc);
}

// ══════════════════════════════════════════════════════════════════
// Q5_0 × Q8_0 dot product (32 values per block pair)
// Q5_0 block: f16 d (2B) + qh uint32 (4B) + qs (16B) = 22B
// ══════════════════════════════════════════════════════════════════

// Expand 32 bits → 32 bytes (0xFF where bit set, 0x00 where clear)
static inline __m256i bytes_from_bits_32_qq(const uint8_t* src) {
    uint32_t x32;
    memcpy(&x32, src, 4);
    const __m256i shuf_mask = _mm256_set_epi64x(
        0x0303030303030303LL, 0x0202020202020202LL,
        0x0101010101010101LL, 0x0000000000000000LL);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int)x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x8040201008040201LL);
    bytes = _mm256_and_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, bit_mask);
}

float qq_dot_q5_0_q8_0(const uint8_t* restrict xb, const uint8_t* restrict yb, int n) {
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const uint8_t* x = xb + i * 22;
        const uint8_t* y = yb + i * 34;

        const __m256 d = _mm256_set1_ps(
            f16_to_f32_qq(*(const uint16_t*)x) * f16_to_f32_qq(*(const uint16_t*)y));

        __m256i qx = bytes_from_nibbles_32_qq(x + 6);
        __m256i bxhi = bytes_from_bits_32_qq(x + 2);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);
        qx = _mm256_sub_epi8(qx, _mm256_set1_epi8(16));

        __m256i qy = _mm256_loadu_si256((const __m256i*)(y + 2));

        acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs_float_qq(qx, qy), acc);
    }
    return hsum_float_8_qq(acc);
}

// Q5_1 × Q8_1 dot product (32 values per block pair)
// Q5_1 block: f16 d (2B) + f16 m (2B) + qh uint32 (4B) + qs (16B) = 24B
// Q8_1 block: f32 d (4B) + f32 s (4B) + qs (32B) = 40B
float qq_dot_q5_1_q8_1(const uint8_t* restrict xb, const uint8_t* restrict yb, int n) {
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* x = xb + i * 24;
        const uint8_t* y = yb + i * 40;

        const float dx = f16_to_f32_qq(*(const uint16_t*)x);
        const float mx = f16_to_f32_qq(*(const uint16_t*)(x + 2));
        float dy;
        memcpy(&dy, y, 4);
        float sy;
        memcpy(&sy, y + 4, 4);

        summs += mx * sy;

        __m256i qx = bytes_from_nibbles_32_qq(x + 8);
        __m256i bxhi = bytes_from_bits_32_qq(x + 4);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);

        __m256i qy = _mm256_loadu_si256((const __m256i*)(y + 8));

        acc = _mm256_fmadd_ps(_mm256_set1_ps(dx * dy),
                              mul_sum_us8_pairs_float_qq(qx, qy), acc);
    }
    return hsum_float_8_qq(acc) + summs;
}

// ══════════════════════════════════════════════════════════════════
// Q8_0 × Q8_0 dot product
// ══════════════════════════════════════════════════════════════════

float qq_dot_q8_0_q8_0(const uint8_t* restrict xb, const uint8_t* restrict yb, int n) {
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const uint8_t* x = xb + i * 34;
        const uint8_t* y = yb + i * 34;

        const __m256 d = _mm256_set1_ps(f16_to_f32_qq(*(const uint16_t*)x) * f16_to_f32_qq(*(const uint16_t*)y));
        __m256i qx = _mm256_loadu_si256((const __m256i*)(x + 2));
        __m256i qy = _mm256_loadu_si256((const __m256i*)(y + 2));

        acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs_float_qq(qx, qy), acc);
    }
    return hsum_float_8_qq(acc);
}

// ══════════════════════════════════════════════════════════════════
// Q2_K × Q8_K dot product (256 values per super-block pair)
// ══════════════════════════════════════════════════════════════════

float qq_dot_q2_K_q8_K(const uint8_t* restrict xb, const uint8_t* restrict yb, int n) {
    const int nb = n / 256;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        // Q2_K block: scales[16] + qs[64] + d(f16) + dmin(f16) = 84 bytes
        const uint8_t* xp = xb + i * 84;
        // Q8_K block: d(f32) + qs[256] + bsums[32] = 292 bytes
        const uint8_t* yp = yb + i * 292;

        float y_d;
        memcpy(&y_d, yp, 4);
        const float d = y_d * f16_to_f32_qq(*(const uint16_t*)(xp + 80));
        const float dmin = -y_d * f16_to_f32_qq(*(const uint16_t*)(xp + 82));

        const uint8_t* q2 = xp + 16;  // qs at offset 16
        const int8_t* q8 = (const int8_t*)(yp + 4);

        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)xp); // scales at offset 0
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)(yp + 260)));

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I_QQ(l_scales, l_scales), MM256_SET_M128I_QQ(h_scales, h_scales)};

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < 2; j++) {  // QK_K/128 = 2
            const __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2); q2 += 32;

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
            const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k_qq(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k_qq(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k_qq(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k_qq(3)), p3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_float_8_qq(acc);
}

// ══════════════════════════════════════════════════════════════════
// Q3_K × Q8_K dot product
// ══════════════════════════════════════════════════════════════════

float qq_dot_q3_K_q8_K(const uint8_t* restrict xb, const uint8_t* restrict yb, int n) {
    const int nb = n / 256;
    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    const __m256i m3 = _mm256_set1_epi8(0x03);
    const __m256i zero = _mm256_setzero_si256();
    float sum = 0.0f;

    for (int block = 0; block < nb; block++) {
        const uint8_t* xp = xb + block * 110;
        const uint8_t* yp = yb + block * 292;

        float yd;
        memcpy(&yd, yp, 4);
        const float d_all = yd * f16_to_f32_qq(*(const uint16_t*)(xp + 108));

        uint32_t aux[4] = {0, 0, 0, 0};
        memcpy(aux, xp + 96, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int8_t scales[16];
        for (int i = 0; i < 16; i++) {
            scales[i] = (int8_t)((aux[i / 4] >> (8 * (i % 4))) & 0xff);
        }

        int is = 0;
        uint8_t m = 1;
        const uint8_t* q3 = xp + 32;
        const int8_t* q8 = (const int8_t*)(yp + 4);
        const __m256i hbits = _mm256_loadu_si256((const __m256i*)xp);
        __m256i sumi = _mm256_setzero_si256();

        for (int half = 0; half < 2; half++) {
            const __m256i qbits = _mm256_loadu_si256((const __m256i*)q3);
            for (int j = 0; j < 4; j++) {
                const int shift = 2 * j;
                __m256i q2 = shift == 0 ? _mm256_and_si256(qbits, m3)
                                        : _mm256_and_si256(_mm256_srli_epi16(qbits, shift), m3);
                const __m256i hmask = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, _mm256_set1_epi8((char)m)), zero);
                const __m256i qv = _mm256_sub_epi8(q2, _mm256_and_si256(hmask, _mm256_set1_epi8(4)));
                const __m256i abs_qv = _mm256_sign_epi8(qv, qv);
                const __m256i yv = _mm256_loadu_si256((const __m256i*)q8);
                __m256i p16 = _mm256_maddubs_epi16(abs_qv, _mm256_sign_epi8(yv, qv));
                const __m256i sc = MM256_SET_M128I_QQ(
                    _mm_set1_epi16((int16_t)(scales[is+1] - 32)),
                    _mm_set1_epi16((int16_t)(scales[is] - 32)));
                is += 2;
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(sc, p16));
                q8 += 32;
                m <<= 1;
            }
            q3 += 32;
        }
        sum += d_all * (float)hsum_i32_8_qq(sumi);
    }
    return sum;
}

// ══════════════════════════════════════════════════════════════════
// Q4_K × Q8_K dot product
// ══════════════════════════════════════════════════════════════════

float qq_dot_q4_K_q8_K(const uint8_t* restrict xb, const uint8_t* restrict yb, int n) {
    const int nb = n / 256;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; i++) {
        // Q4_K: d(f16) + dmin(f16) + scales[12] + qs[128] = 144 bytes
        const uint8_t* xp = xb + i * 144;
        const uint8_t* yp = yb + i * 292;

        float y_d;
        memcpy(&y_d, yp, 4);
        const float d = y_d * f16_to_f32_qq(*(const uint16_t*)xp);
        const float dmin = -y_d * f16_to_f32_qq(*(const uint16_t*)(xp + 2));

        memcpy(utmp, xp + 4, 12); // scales at offset 4
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t* q4 = xp + 16;  // qs at offset 16
        const int8_t* q8 = (const int8_t*)(yp + 4);

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)(yp + 260));
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I_QQ(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < 4; j++) { // QK_K/64 = 4
            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4_qq(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4_qq(2*j+1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    return hsum_float_8_qq(acc) + _mm_cvtss_f32(acc_m);
}

// ══════════════════════════════════════════════════════════════════
// Q5_K × Q8_K dot product
// ══════════════════════════════════════════════════════════════════

float qq_dot_q5_K_q8_K(const uint8_t* restrict xb, const uint8_t* restrict yb, int n) {
    const int nb = n / 256;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i mone = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.f;

    for (int i = 0; i < nb; i++) {
        // Q5_K: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128] = 176 bytes
        const uint8_t* xp = xb + i * 176;
        const uint8_t* yp = yb + i * 292;

        float y_d;
        memcpy(&y_d, yp, 4);
        const float d = y_d * f16_to_f32_qq(*(const uint16_t*)xp);
        const float dmin = -y_d * f16_to_f32_qq(*(const uint16_t*)(xp + 2));

        memcpy(utmp, xp + 4, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)(yp + 260));
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, _mm_setzero_si128()), _mm_setzero_si128());
        summs += dmin * (float)_mm_extract_epi32(hsum, 0);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I_QQ(sc128, sc128);

        const uint8_t* q5 = xp + 48; // qs at offset 48
        const int8_t* q8 = (const int8_t*)(yp + 4);
        const __m256i hbits = _mm256_loadu_si256((const __m256i*)(xp + 16)); // qh at offset 16
        __m256i hmask = mone;

        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;

        for (int j = 0; j < 4; j++) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4_qq(2*j+0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4_qq(2*j+1));

            const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5); q5 += 32;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_madd_epi16(scale_0, _mm256_maddubs_epi16(q5_0, q8_0));
            __m256i p16_1 = _mm256_madd_epi16(scale_1, _mm256_maddubs_epi16(q5_1, q8_1));

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_float_8_qq(acc) + summs;
}

// ══════════════════════════════════════════════════════════════════
// Q6_K × Q8_K dot product
// ══════════════════════════════════════════════════════════════════

float qq_dot_q6_K_q8_K(const uint8_t* restrict xb, const uint8_t* restrict yb, int n) {
    const int nb = n / 256;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        // Q6_K: ql[128] + qh[64] + scales[16] + d(f16) = 210 bytes
        const uint8_t* xp = xb + i * 210;
        const uint8_t* yp = yb + i * 292;

        float y_d;
        memcpy(&y_d, yp, 4);
        const float d = y_d * f16_to_f32_qq(*(const uint16_t*)(xp + 208));

        const uint8_t* q4 = xp;         // ql at offset 0
        const uint8_t* qh = xp + 128;   // qh at offset 128
        const int8_t* q8 = (const int8_t*)(yp + 4);

        const __m128i scales = _mm_loadu_si128((const __m128i*)(xp + 192));

        __m256i sumi = _mm256_setzero_si256();
        int is = 0;

        for (int j = 0; j < 2; j++) { // QK_K/128 = 2
            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle_qq(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle_qq(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle_qq(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle_qq(is + 3));
            is += 4;

            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
            const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
            const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

            __m256i p16_0 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_0, q8_0), q8s_0);
            __m256i p16_1 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_1, q8_1), q8s_1);
            __m256i p16_2 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_2, q8_2), q8s_2);
            __m256i p16_3 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_3, q8_3), q8s_3);

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p16_0, p16_1), _mm256_add_epi32(p16_2, p16_3)));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_float_8_qq(acc);
}

// ══════════════════════════════════════════════════════════════════
// Batch dispatcher: quantize input once, then run Q×Q for all rows
// ══════════════════════════════════════════════════════════════════

void qq_dot_batch(const uint8_t* w_data, uint32_t w_type,
                  const uint8_t* q_data, int cols,
                  float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        const uint8_t* row = w_data + (size_t)r * bpr;
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(w_data + (size_t)(r+1) * bpr), _MM_HINT_T0);

        switch (w_type) {
        case 2:  out[r] = qq_dot_q4_0_q8_0(row, q_data, cols); break;
        case 6:  out[r] = qq_dot_q5_0_q8_0(row, q_data, cols); break;
        case 8:  out[r] = qq_dot_q8_0_q8_0(row, q_data, cols); break;
        case 10: out[r] = qq_dot_q2_K_q8_K(row, q_data, cols); break;
        case 11: out[r] = qq_dot_q3_K_q8_K(row, q_data, cols); break;
        case 12: out[r] = qq_dot_q4_K_q8_K(row, q_data, cols); break;
        case 13: out[r] = qq_dot_q5_K_q8_K(row, q_data, cols); break;
        case 14: out[r] = qq_dot_q6_K_q8_K(row, q_data, cols); break;
        default: out[r] = 0; break;
        }
    }
}

// Q4_0 batch kernel: process one weight row against multiple Q8 positions,
// sharing nibble unpack across positions for better throughput.
static void qq_batch_row_q4_0(const uint8_t* restrict row, const uint8_t* restrict q8_flat,
                               int q8_stride, int n_inputs, int cols,
                               float* restrict out_flat, int out_stride, int r) {
    const int nb = cols / 32;
    int p = 0;
    // Process 4 positions at a time, sharing weight block unpack
    for (; p + 3 < n_inputs; p += 4) {
        const uint8_t* y0 = q8_flat + (size_t)p * q8_stride;
        const uint8_t* y1 = q8_flat + (size_t)(p+1) * q8_stride;
        const uint8_t* y2 = q8_flat + (size_t)(p+2) * q8_stride;
        const uint8_t* y3 = q8_flat + (size_t)(p+3) * q8_stride;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        for (int i = 0; i < nb; i++) {
            const uint8_t* xb = row + i * 18;
            float xd = f16_to_f32_qq(*(const uint16_t*)xb);
            __m256i qx = bytes_from_nibbles_32_qq(xb + 2);
            qx = _mm256_sub_epi8(qx, _mm256_set1_epi8(8));
            const __m256i ax = _mm256_sign_epi8(qx, qx);

            const uint8_t* yb0 = y0 + i * 34;
            __m256i sy0 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(yb0 + 2)), qx);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb0)),
                                    dotbusd_ps_qq(ax, sy0), acc0);

            const uint8_t* yb1 = y1 + i * 34;
            __m256i sy1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(yb1 + 2)), qx);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb1)),
                                    dotbusd_ps_qq(ax, sy1), acc1);

            const uint8_t* yb2 = y2 + i * 34;
            __m256i sy2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(yb2 + 2)), qx);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb2)),
                                    dotbusd_ps_qq(ax, sy2), acc2);

            const uint8_t* yb3 = y3 + i * 34;
            __m256i sy3 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(yb3 + 2)), qx);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb3)),
                                    dotbusd_ps_qq(ax, sy3), acc3);
        }
        out_flat[(size_t)p * out_stride + r] = hsum_float_8_qq(acc0);
        out_flat[(size_t)(p+1) * out_stride + r] = hsum_float_8_qq(acc1);
        out_flat[(size_t)(p+2) * out_stride + r] = hsum_float_8_qq(acc2);
        out_flat[(size_t)(p+3) * out_stride + r] = hsum_float_8_qq(acc3);
    }
    // Process 2 at a time for remainder
    for (; p + 1 < n_inputs; p += 2) {
        const uint8_t* y0 = q8_flat + (size_t)p * q8_stride;
        const uint8_t* y1 = q8_flat + (size_t)(p+1) * q8_stride;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (int i = 0; i < nb; i++) {
            const uint8_t* xb = row + i * 18;
            float xd = f16_to_f32_qq(*(const uint16_t*)xb);
            __m256i qx = bytes_from_nibbles_32_qq(xb + 2);
            qx = _mm256_sub_epi8(qx, _mm256_set1_epi8(8));

            const uint8_t* yb0 = y0 + i * 34;
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb0)),
                                    mul_sum_i8_pairs_float_qq(qx, _mm256_loadu_si256((const __m256i*)(yb0 + 2))), acc0);
            const uint8_t* yb1 = y1 + i * 34;
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb1)),
                                    mul_sum_i8_pairs_float_qq(qx, _mm256_loadu_si256((const __m256i*)(yb1 + 2))), acc1);
        }
        out_flat[(size_t)p * out_stride + r] = hsum_float_8_qq(acc0);
        out_flat[(size_t)(p+1) * out_stride + r] = hsum_float_8_qq(acc1);
    }
    for (; p < n_inputs; p++) {
        out_flat[(size_t)p * out_stride + r] = qq_dot_q4_0_q8_0(row, q8_flat + (size_t)p * q8_stride, cols);
    }
}

// Q8_0 batch kernel: same pattern for Q8_0 weights
static void qq_batch_row_q8_0(const uint8_t* restrict row, const uint8_t* restrict q8_flat,
                               int q8_stride, int n_inputs, int cols,
                               float* restrict out_flat, int out_stride, int r) {
    const int nb = cols / 32;
    int p = 0;
    for (; p + 3 < n_inputs; p += 4) {
        const uint8_t* y0 = q8_flat + (size_t)p * q8_stride;
        const uint8_t* y1 = q8_flat + (size_t)(p+1) * q8_stride;
        const uint8_t* y2 = q8_flat + (size_t)(p+2) * q8_stride;
        const uint8_t* y3 = q8_flat + (size_t)(p+3) * q8_stride;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        for (int i = 0; i < nb; i++) {
            const uint8_t* xb = row + i * 34;
            float xd = f16_to_f32_qq(*(const uint16_t*)xb);
            __m256i qx = _mm256_loadu_si256((const __m256i*)(xb + 2));
            const __m256i ax = _mm256_sign_epi8(qx, qx);

            const uint8_t* yb0 = y0 + i * 34;
            __m256i sy0 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(yb0 + 2)), qx);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb0)),
                                    dotbusd_ps_qq(ax, sy0), acc0);

            const uint8_t* yb1 = y1 + i * 34;
            __m256i sy1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(yb1 + 2)), qx);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb1)),
                                    dotbusd_ps_qq(ax, sy1), acc1);

            const uint8_t* yb2 = y2 + i * 34;
            __m256i sy2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(yb2 + 2)), qx);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb2)),
                                    dotbusd_ps_qq(ax, sy2), acc2);

            const uint8_t* yb3 = y3 + i * 34;
            __m256i sy3 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(yb3 + 2)), qx);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb3)),
                                    dotbusd_ps_qq(ax, sy3), acc3);
        }
        out_flat[(size_t)p * out_stride + r] = hsum_float_8_qq(acc0);
        out_flat[(size_t)(p+1) * out_stride + r] = hsum_float_8_qq(acc1);
        out_flat[(size_t)(p+2) * out_stride + r] = hsum_float_8_qq(acc2);
        out_flat[(size_t)(p+3) * out_stride + r] = hsum_float_8_qq(acc3);
    }
    for (; p + 1 < n_inputs; p += 2) {
        const uint8_t* y0 = q8_flat + (size_t)p * q8_stride;
        const uint8_t* y1 = q8_flat + (size_t)(p+1) * q8_stride;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (int i = 0; i < nb; i++) {
            const uint8_t* xb = row + i * 34;
            float xd = f16_to_f32_qq(*(const uint16_t*)xb);
            __m256i qx = _mm256_loadu_si256((const __m256i*)(xb + 2));
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)(y0 + i * 34))),
                                    mul_sum_i8_pairs_float_qq(qx, _mm256_loadu_si256((const __m256i*)(y0 + i * 34 + 2))), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)(y1 + i * 34))),
                                    mul_sum_i8_pairs_float_qq(qx, _mm256_loadu_si256((const __m256i*)(y1 + i * 34 + 2))), acc1);
        }
        out_flat[(size_t)p * out_stride + r] = hsum_float_8_qq(acc0);
        out_flat[(size_t)(p+1) * out_stride + r] = hsum_float_8_qq(acc1);
    }
    for (; p < n_inputs; p++) {
        out_flat[(size_t)p * out_stride + r] = qq_dot_q8_0_q8_0(row, q8_flat + (size_t)p * q8_stride, cols);
    }
}

// Q5_0 batch kernel: process one weight row against multiple Q8 positions
static void qq_batch_row_q5_0(const uint8_t* restrict row, const uint8_t* restrict q8_flat,
                               int q8_stride, int n_inputs, int cols,
                               float* restrict out_flat, int out_stride, int r) {
    const int nb = cols / 32;
    int p = 0;
    for (; p + 3 < n_inputs; p += 4) {
        const uint8_t* y0 = q8_flat + (size_t)p * q8_stride;
        const uint8_t* y1 = q8_flat + (size_t)(p+1) * q8_stride;
        const uint8_t* y2 = q8_flat + (size_t)(p+2) * q8_stride;
        const uint8_t* y3 = q8_flat + (size_t)(p+3) * q8_stride;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        for (int i = 0; i < nb; i++) {
            const uint8_t* xb = row + i * 22;
            float xd = f16_to_f32_qq(*(const uint16_t*)xb);
            __m256i qx = bytes_from_nibbles_32_qq(xb + 6);
            __m256i bxhi = bytes_from_bits_32_qq(xb + 2);
            bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
            qx = _mm256_or_si256(qx, bxhi);
            qx = _mm256_sub_epi8(qx, _mm256_set1_epi8(16));
            const __m256i ax = _mm256_sign_epi8(qx, qx);

            const uint8_t* yb0 = y0 + i * 34;
            __m256i sy0 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(yb0 + 2)), qx);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb0)),
                                    dotbusd_ps_qq(ax, sy0), acc0);
            const uint8_t* yb1 = y1 + i * 34;
            __m256i sy1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(yb1 + 2)), qx);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb1)),
                                    dotbusd_ps_qq(ax, sy1), acc1);
            const uint8_t* yb2 = y2 + i * 34;
            __m256i sy2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(yb2 + 2)), qx);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb2)),
                                    dotbusd_ps_qq(ax, sy2), acc2);
            const uint8_t* yb3 = y3 + i * 34;
            __m256i sy3 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(yb3 + 2)), qx);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(xd * f16_to_f32_qq(*(const uint16_t*)yb3)),
                                    dotbusd_ps_qq(ax, sy3), acc3);
        }
        out_flat[(size_t)p * out_stride + r] = hsum_float_8_qq(acc0);
        out_flat[(size_t)(p+1) * out_stride + r] = hsum_float_8_qq(acc1);
        out_flat[(size_t)(p+2) * out_stride + r] = hsum_float_8_qq(acc2);
        out_flat[(size_t)(p+3) * out_stride + r] = hsum_float_8_qq(acc3);
    }
    for (; p < n_inputs; p++) {
        out_flat[(size_t)p * out_stride + r] = qq_dot_q5_0_q8_0(row, q8_flat + (size_t)p * q8_stride, cols);
    }
}

// Q4_K batch kernel: process one weight row against multiple Q8_K positions,
// sharing scale unpack and q4 nibble decode across positions.
static void qq_batch_row_q4_K(const uint8_t* restrict row, const uint8_t* restrict q8_flat,
                              int q8_stride, int n_inputs, int cols,
                              float* restrict out_flat, int out_stride, int r) {
    const int nb = cols / 256;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    uint32_t utmp[4];

    int p = 0;
    for (; p + 3 < n_inputs; p += 4) {
        const uint8_t* y0 = q8_flat + (size_t)p * q8_stride;
        const uint8_t* y1 = q8_flat + (size_t)(p + 1) * q8_stride;
        const uint8_t* y2 = q8_flat + (size_t)(p + 2) * q8_stride;
        const uint8_t* y3 = q8_flat + (size_t)(p + 3) * q8_stride;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m128 accm0 = _mm_setzero_ps();
        __m128 accm1 = _mm_setzero_ps();
        __m128 accm2 = _mm_setzero_ps();
        __m128 accm3 = _mm_setzero_ps();

        for (int i = 0; i < nb; i++) {
            const uint8_t* xp = row + i * 144;
            const uint8_t* yp0 = y0 + i * 292;
            const uint8_t* yp1 = y1 + i * 292;
            const uint8_t* yp2 = y2 + i * 292;
            const uint8_t* yp3 = y3 + i * 292;

            memcpy(utmp, xp + 4, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            {
                const uint32_t uaux = utmp[1] & kmask1;
                utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                utmp[2] = uaux;
                utmp[0] &= kmask1;
            }

            const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
            const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
            const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            const __m256i scales = MM256_SET_M128I_QQ(sc128, sc128);
            const float wd = f16_to_f32_qq(*(const uint16_t*)xp);
            const float wdm = f16_to_f32_qq(*(const uint16_t*)(xp + 2));

            float yd0, yd1, yd2, yd3;
            memcpy(&yd0, yp0, 4);
            memcpy(&yd1, yp1, 4);
            memcpy(&yd2, yp2, 4);
            memcpy(&yd3, yp3, 4);
            const float d0 = yd0 * wd;
            const float d1 = yd1 * wd;
            const float d2 = yd2 * wd;
            const float d3 = yd3 * wd;
            const float dmin0 = -yd0 * wdm;
            const float dmin1 = -yd1 * wdm;
            const float dmin2 = -yd2 * wdm;
            const float dmin3 = -yd3 * wdm;

            const __m256i q8sums0 = _mm256_loadu_si256((const __m256i*)(yp0 + 260));
            const __m256i q8sums1 = _mm256_loadu_si256((const __m256i*)(yp1 + 260));
            const __m256i q8sums2 = _mm256_loadu_si256((const __m256i*)(yp2 + 260));
            const __m256i q8sums3 = _mm256_loadu_si256((const __m256i*)(yp3 + 260));
            const __m128i q8s0 = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums0, 0), _mm256_extracti128_si256(q8sums0, 1));
            const __m128i q8s1 = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums1, 0), _mm256_extracti128_si256(q8sums1, 1));
            const __m128i q8s2 = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums2, 0), _mm256_extracti128_si256(q8sums2, 1));
            const __m128i q8s3 = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums3, 0), _mm256_extracti128_si256(q8sums3, 1));
            accm0 = _mm_fmadd_ps(_mm_set1_ps(dmin0), _mm_cvtepi32_ps(_mm_madd_epi16(mins, q8s0)), accm0);
            accm1 = _mm_fmadd_ps(_mm_set1_ps(dmin1), _mm_cvtepi32_ps(_mm_madd_epi16(mins, q8s1)), accm1);
            accm2 = _mm_fmadd_ps(_mm_set1_ps(dmin2), _mm_cvtepi32_ps(_mm_madd_epi16(mins, q8s2)), accm2);
            accm3 = _mm_fmadd_ps(_mm_set1_ps(dmin3), _mm_cvtepi32_ps(_mm_madd_epi16(mins, q8s3)), accm3);

            const uint8_t* q4 = xp + 16;
            const int8_t* q80 = (const int8_t*)(yp0 + 4);
            const int8_t* q81 = (const int8_t*)(yp1 + 4);
            const int8_t* q82 = (const int8_t*)(yp2 + 4);
            const int8_t* q83 = (const int8_t*)(yp3 + 4);
            __m256i sumi0 = _mm256_setzero_si256();
            __m256i sumi1 = _mm256_setzero_si256();
            __m256i sumi2 = _mm256_setzero_si256();
            __m256i sumi3 = _mm256_setzero_si256();

            for (int j = 0; j < 4; j++) {
                const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4_qq(2*j+0));
                const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4_qq(2*j+1));
                const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
                const __m256i q4l = _mm256_and_si256(q4bits, m4);
                const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                const __m256i q8l0 = _mm256_loadu_si256((const __m256i*)q80); q80 += 32;
                const __m256i q8h0 = _mm256_loadu_si256((const __m256i*)q80); q80 += 32;
                const __m256i q8l1 = _mm256_loadu_si256((const __m256i*)q81); q81 += 32;
                const __m256i q8h1 = _mm256_loadu_si256((const __m256i*)q81); q81 += 32;
                const __m256i q8l2 = _mm256_loadu_si256((const __m256i*)q82); q82 += 32;
                const __m256i q8h2 = _mm256_loadu_si256((const __m256i*)q82); q82 += 32;
                const __m256i q8l3 = _mm256_loadu_si256((const __m256i*)q83); q83 += 32;
                const __m256i q8h3 = _mm256_loadu_si256((const __m256i*)q83); q83 += 32;

                sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(
                    _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l0)),
                    _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h0))));
                sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(
                    _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l1)),
                    _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h1))));
                sumi2 = _mm256_add_epi32(sumi2, _mm256_add_epi32(
                    _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l2)),
                    _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h2))));
                sumi3 = _mm256_add_epi32(sumi3, _mm256_add_epi32(
                    _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l3)),
                    _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h3))));
            }

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(sumi0), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(sumi1), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(d2), _mm256_cvtepi32_ps(sumi2), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(d3), _mm256_cvtepi32_ps(sumi3), acc3);
        }

        out_flat[(size_t)p * out_stride + r] = hsum_float_8_qq(acc0) + hsum_float_4_qq(accm0);
        out_flat[(size_t)(p + 1) * out_stride + r] = hsum_float_8_qq(acc1) + hsum_float_4_qq(accm1);
        out_flat[(size_t)(p + 2) * out_stride + r] = hsum_float_8_qq(acc2) + hsum_float_4_qq(accm2);
        out_flat[(size_t)(p + 3) * out_stride + r] = hsum_float_8_qq(acc3) + hsum_float_4_qq(accm3);
    }

    for (; p + 1 < n_inputs; p += 2) {
        const uint8_t* y0 = q8_flat + (size_t)p * q8_stride;
        const uint8_t* y1 = q8_flat + (size_t)(p + 1) * q8_stride;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m128 accm0 = _mm_setzero_ps();
        __m128 accm1 = _mm_setzero_ps();

        for (int i = 0; i < nb; i++) {
            const uint8_t* xp = row + i * 144;
            const uint8_t* yp0 = y0 + i * 292;
            const uint8_t* yp1 = y1 + i * 292;

            memcpy(utmp, xp + 4, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            {
                const uint32_t uaux = utmp[1] & kmask1;
                utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                utmp[2] = uaux;
                utmp[0] &= kmask1;
            }

            const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
            const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
            const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            const __m256i scales = MM256_SET_M128I_QQ(sc128, sc128);
            const float wd = f16_to_f32_qq(*(const uint16_t*)xp);
            const float wdm = f16_to_f32_qq(*(const uint16_t*)(xp + 2));

            float yd0, yd1;
            memcpy(&yd0, yp0, 4);
            memcpy(&yd1, yp1, 4);
            const float d0 = yd0 * wd;
            const float d1 = yd1 * wd;
            const float dmin0 = -yd0 * wdm;
            const float dmin1 = -yd1 * wdm;

            const __m256i q8sums0 = _mm256_loadu_si256((const __m256i*)(yp0 + 260));
            const __m256i q8sums1 = _mm256_loadu_si256((const __m256i*)(yp1 + 260));
            const __m128i q8s0 = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums0, 0), _mm256_extracti128_si256(q8sums0, 1));
            const __m128i q8s1 = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums1, 0), _mm256_extracti128_si256(q8sums1, 1));
            accm0 = _mm_fmadd_ps(_mm_set1_ps(dmin0), _mm_cvtepi32_ps(_mm_madd_epi16(mins, q8s0)), accm0);
            accm1 = _mm_fmadd_ps(_mm_set1_ps(dmin1), _mm_cvtepi32_ps(_mm_madd_epi16(mins, q8s1)), accm1);

            const uint8_t* q4 = xp + 16;
            const int8_t* q80 = (const int8_t*)(yp0 + 4);
            const int8_t* q81 = (const int8_t*)(yp1 + 4);
            __m256i sumi0 = _mm256_setzero_si256();
            __m256i sumi1 = _mm256_setzero_si256();

            for (int j = 0; j < 4; j++) {
                const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4_qq(2*j+0));
                const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4_qq(2*j+1));
                const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
                const __m256i q4l = _mm256_and_si256(q4bits, m4);
                const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                const __m256i q8l0 = _mm256_loadu_si256((const __m256i*)q80); q80 += 32;
                const __m256i q8h0 = _mm256_loadu_si256((const __m256i*)q80); q80 += 32;
                const __m256i q8l1 = _mm256_loadu_si256((const __m256i*)q81); q81 += 32;
                const __m256i q8h1 = _mm256_loadu_si256((const __m256i*)q81); q81 += 32;

                sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(
                    _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l0)),
                    _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h0))));
                sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(
                    _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l1)),
                    _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h1))));
            }

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(sumi0), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(sumi1), acc1);
        }

        out_flat[(size_t)p * out_stride + r] = hsum_float_8_qq(acc0) + hsum_float_4_qq(accm0);
        out_flat[(size_t)(p + 1) * out_stride + r] = hsum_float_8_qq(acc1) + hsum_float_4_qq(accm1);
    }

    for (; p < n_inputs; p++) {
        out_flat[(size_t)p * out_stride + r] = qq_dot_q4_K_q8_K(row, q8_flat + (size_t)p * q8_stride, cols);
    }
}

// Q6_K batch kernel: share weight unpack and scale decode across positions.
static void qq_batch_row_q6_K(const uint8_t* restrict row, const uint8_t* restrict q8_flat,
                              int q8_stride, int n_inputs, int cols,
                              float* restrict out_flat, int out_stride, int r) {
    const int nb = cols / 256;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);

    int p = 0;
    for (; p + 1 < n_inputs; p += 2) {
        const uint8_t* y0 = q8_flat + (size_t)p * q8_stride;
        const uint8_t* y1 = q8_flat + (size_t)(p + 1) * q8_stride;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        for (int i = 0; i < nb; i++) {
            const uint8_t* xp = row + i * 210;
            const uint8_t* yp0 = y0 + i * 292;
            const uint8_t* yp1 = y1 + i * 292;
            const float wd = f16_to_f32_qq(*(const uint16_t*)(xp + 208));

            float yd0, yd1;
            memcpy(&yd0, yp0, 4);
            memcpy(&yd1, yp1, 4);
            const float d0 = yd0 * wd;
            const float d1 = yd1 * wd;

            const uint8_t* q4 = xp;
            const uint8_t* qh = xp + 128;
            const int8_t* q80 = (const int8_t*)(yp0 + 4);
            const int8_t* q81 = (const int8_t*)(yp1 + 4);
            const __m128i scales = _mm_loadu_si128((const __m128i*)(xp + 192));

            __m256i sumi0 = _mm256_setzero_si256();
            __m256i sumi1 = _mm256_setzero_si256();
            int is = 0;

            for (int j = 0; j < 2; j++) {
                const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle_qq(is + 0));
                const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle_qq(is + 1));
                const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle_qq(is + 2));
                const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle_qq(is + 3));
                is += 4;

                const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
                const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
                const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

                const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
                const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
                const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
                const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

                const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
                const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

                const __m256i q8_00 = _mm256_loadu_si256((const __m256i*)q80); q80 += 32;
                const __m256i q8_01 = _mm256_loadu_si256((const __m256i*)q80); q80 += 32;
                const __m256i q8_02 = _mm256_loadu_si256((const __m256i*)q80); q80 += 32;
                const __m256i q8_03 = _mm256_loadu_si256((const __m256i*)q80); q80 += 32;
                const __m256i q8_10 = _mm256_loadu_si256((const __m256i*)q81); q81 += 32;
                const __m256i q8_11 = _mm256_loadu_si256((const __m256i*)q81); q81 += 32;
                const __m256i q8_12 = _mm256_loadu_si256((const __m256i*)q81); q81 += 32;
                const __m256i q8_13 = _mm256_loadu_si256((const __m256i*)q81); q81 += 32;

                __m256i q8s_00 = _mm256_maddubs_epi16(m32s, q8_00);
                __m256i q8s_01 = _mm256_maddubs_epi16(m32s, q8_01);
                __m256i q8s_02 = _mm256_maddubs_epi16(m32s, q8_02);
                __m256i q8s_03 = _mm256_maddubs_epi16(m32s, q8_03);
                __m256i q8s_10 = _mm256_maddubs_epi16(m32s, q8_10);
                __m256i q8s_11 = _mm256_maddubs_epi16(m32s, q8_11);
                __m256i q8s_12 = _mm256_maddubs_epi16(m32s, q8_12);
                __m256i q8s_13 = _mm256_maddubs_epi16(m32s, q8_13);

                __m256i p16_00 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_0, q8_00), q8s_00);
                __m256i p16_01 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_1, q8_01), q8s_01);
                __m256i p16_02 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_2, q8_02), q8s_02);
                __m256i p16_03 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_3, q8_03), q8s_03);
                __m256i p16_10 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_0, q8_10), q8s_10);
                __m256i p16_11 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_1, q8_11), q8s_11);
                __m256i p16_12 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_2, q8_12), q8s_12);
                __m256i p16_13 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_3, q8_13), q8s_13);

                p16_00 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_00);
                p16_01 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_01);
                p16_02 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_02);
                p16_03 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_03);
                p16_10 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_10);
                p16_11 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_11);
                p16_12 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_12);
                p16_13 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_13);

                sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(_mm256_add_epi32(p16_00, p16_01), _mm256_add_epi32(p16_02, p16_03)));
                sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(_mm256_add_epi32(p16_10, p16_11), _mm256_add_epi32(p16_12, p16_13)));
            }

            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d0), _mm256_cvtepi32_ps(sumi0), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d1), _mm256_cvtepi32_ps(sumi1), acc1);
        }

        out_flat[(size_t)p * out_stride + r] = hsum_float_8_qq(acc0);
        out_flat[(size_t)(p + 1) * out_stride + r] = hsum_float_8_qq(acc1);
    }

    for (; p < n_inputs; p++) {
        out_flat[(size_t)p * out_stride + r] = qq_dot_q6_K_q8_K(row, q8_flat + (size_t)p * q8_stride, cols);
    }
}

// Q3_K batch kernel: integer accumulation with 256-bit ops, 4 positions at a time.
static void qq_batch_row_q3_K(const uint8_t* restrict row, const uint8_t* restrict q8_flat,
                              int q8_stride, int n_inputs, int cols,
                              float* restrict out_flat, int out_stride, int r) {
    const int nb = cols / 256;
    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    const __m256i m3 = _mm256_set1_epi8(0x03);
    const __m256i zero = _mm256_setzero_si256();

    int p = 0;
    for (; p + 3 < n_inputs; p += 4) {
        const uint8_t* y0 = q8_flat + (size_t)p * q8_stride;
        const uint8_t* y1 = q8_flat + (size_t)(p+1) * q8_stride;
        const uint8_t* y2 = q8_flat + (size_t)(p+2) * q8_stride;
        const uint8_t* y3 = q8_flat + (size_t)(p+3) * q8_stride;
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

        for (int block = 0; block < nb; block++) {
            const uint8_t* xp = row + block * 110;
            const uint8_t* yp0 = y0 + block * 292;
            const uint8_t* yp1 = y1 + block * 292;
            const uint8_t* yp2 = y2 + block * 292;
            const uint8_t* yp3 = y3 + block * 292;
            float yd0, yd1, yd2, yd3;
            memcpy(&yd0, yp0, 4); memcpy(&yd1, yp1, 4);
            memcpy(&yd2, yp2, 4); memcpy(&yd3, yp3, 4);
            const float wd = f16_to_f32_qq(*(const uint16_t*)(xp + 108));
            const float d0 = yd0 * wd, d1 = yd1 * wd, d2 = yd2 * wd, d3 = yd3 * wd;

            uint32_t aux[4] = {0, 0, 0, 0};
            memcpy(aux, xp + 96, 12);
            { const uint32_t tmp = aux[2];
              aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
              aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
              aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
              aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4); }
            int8_t scales[16];
            for (int i = 0; i < 16; i++)
                scales[i] = (int8_t)((aux[i / 4] >> (8 * (i % 4))) & 0xff);

            const __m256i hbits = _mm256_loadu_si256((const __m256i*)xp);
            const uint8_t* q3 = xp + 32;
            const int8_t* q80 = (const int8_t*)(yp0 + 4);
            const int8_t* q81 = (const int8_t*)(yp1 + 4);
            const int8_t* q82 = (const int8_t*)(yp2 + 4);
            const int8_t* q83 = (const int8_t*)(yp3 + 4);
            __m256i si0 = zero, si1 = zero, si2 = zero, si3 = zero;
            int is = 0;
            uint8_t m = 1;

            for (int half = 0; half < 2; half++) {
                const __m256i qbits = _mm256_loadu_si256((const __m256i*)q3);
                for (int j = 0; j < 4; j++) {
                    const int shift = 2 * j;
                    __m256i q2 = shift == 0 ? _mm256_and_si256(qbits, m3)
                                            : _mm256_and_si256(_mm256_srli_epi16(qbits, shift), m3);
                    const __m256i hmask = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, _mm256_set1_epi8((char)m)), zero);
                    const __m256i qv = _mm256_sub_epi8(q2, _mm256_and_si256(hmask, _mm256_set1_epi8(4)));
                    const __m256i abs_qv = _mm256_sign_epi8(qv, qv);

                    const __m256i sc = MM256_SET_M128I_QQ(
                        _mm_set1_epi16((int16_t)(scales[is+1] - 32)),
                        _mm_set1_epi16((int16_t)(scales[is] - 32)));
                    is += 2;

                    #define Q3K_ACCUM(q8ptr, sumi) { \
                        const __m256i yv = _mm256_loadu_si256((const __m256i*)(q8ptr)); \
                        __m256i p16 = _mm256_maddubs_epi16(abs_qv, _mm256_sign_epi8(yv, qv)); \
                        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(sc, p16)); }
                    Q3K_ACCUM(q80, si0); Q3K_ACCUM(q81, si1);
                    Q3K_ACCUM(q82, si2); Q3K_ACCUM(q83, si3);
                    #undef Q3K_ACCUM
                    q80 += 32; q81 += 32; q82 += 32; q83 += 32;
                    m <<= 1;
                }
                q3 += 32;
            }
            sum0 += d0 * (float)hsum_i32_8_qq(si0);
            sum1 += d1 * (float)hsum_i32_8_qq(si1);
            sum2 += d2 * (float)hsum_i32_8_qq(si2);
            sum3 += d3 * (float)hsum_i32_8_qq(si3);
        }
        out_flat[(size_t)p * out_stride + r] = sum0;
        out_flat[(size_t)(p+1) * out_stride + r] = sum1;
        out_flat[(size_t)(p+2) * out_stride + r] = sum2;
        out_flat[(size_t)(p+3) * out_stride + r] = sum3;
    }

    for (; p + 1 < n_inputs; p += 2) {
        const uint8_t* y0 = q8_flat + (size_t)p * q8_stride;
        const uint8_t* y1 = q8_flat + (size_t)(p+1) * q8_stride;
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int block = 0; block < nb; block++) {
            const uint8_t* xp = row + block * 110;
            const uint8_t* yp0 = y0 + block * 292;
            const uint8_t* yp1 = y1 + block * 292;
            float yd0, yd1;
            memcpy(&yd0, yp0, 4); memcpy(&yd1, yp1, 4);
            const float wd = f16_to_f32_qq(*(const uint16_t*)(xp + 108));
            const float d_0 = yd0 * wd, d_1 = yd1 * wd;
            uint32_t aux[4] = {0,0,0,0};
            memcpy(aux, xp + 96, 12);
            { const uint32_t tmp = aux[2];
              aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
              aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
              aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
              aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4); }
            int8_t scales[16];
            for (int i = 0; i < 16; i++)
                scales[i] = (int8_t)((aux[i / 4] >> (8 * (i % 4))) & 0xff);
            const __m256i hbits = _mm256_loadu_si256((const __m256i*)xp);
            const uint8_t* q3 = xp + 32;
            const int8_t* q80 = (const int8_t*)(yp0 + 4);
            const int8_t* q81 = (const int8_t*)(yp1 + 4);
            __m256i si0 = _mm256_setzero_si256(), si1 = _mm256_setzero_si256();
            int is = 0; uint8_t m = 1;
            for (int half = 0; half < 2; half++) {
                const __m256i qbits = _mm256_loadu_si256((const __m256i*)q3);
                for (int j = 0; j < 4; j++) {
                    const int shift = 2*j;
                    __m256i q2 = shift == 0 ? _mm256_and_si256(qbits, m3)
                                            : _mm256_and_si256(_mm256_srli_epi16(qbits, shift), m3);
                    const __m256i hmask = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, _mm256_set1_epi8((char)m)), zero);
                    const __m256i qv = _mm256_sub_epi8(q2, _mm256_and_si256(hmask, _mm256_set1_epi8(4)));
                    const __m256i abs_qv = _mm256_sign_epi8(qv, qv);
                    const __m256i sc = MM256_SET_M128I_QQ(
                        _mm_set1_epi16((int16_t)(scales[is+1] - 32)),
                        _mm_set1_epi16((int16_t)(scales[is] - 32)));
                    is += 2;
                    { const __m256i yv = _mm256_loadu_si256((const __m256i*)q80);
                      __m256i p16 = _mm256_maddubs_epi16(abs_qv, _mm256_sign_epi8(yv, qv));
                      si0 = _mm256_add_epi32(si0, _mm256_madd_epi16(sc, p16)); }
                    { const __m256i yv = _mm256_loadu_si256((const __m256i*)q81);
                      __m256i p16 = _mm256_maddubs_epi16(abs_qv, _mm256_sign_epi8(yv, qv));
                      si1 = _mm256_add_epi32(si1, _mm256_madd_epi16(sc, p16)); }
                    q80 += 32; q81 += 32; m <<= 1;
                }
                q3 += 32;
            }
            sum0 += d_0 * (float)hsum_i32_8_qq(si0);
            sum1 += d_1 * (float)hsum_i32_8_qq(si1);
        }
        out_flat[(size_t)p * out_stride + r] = sum0;
        out_flat[(size_t)(p+1) * out_stride + r] = sum1;
    }

    for (; p < n_inputs; p++) {
        out_flat[(size_t)p * out_stride + r] = qq_dot_q3_K_q8_K(row, q8_flat + (size_t)p * q8_stride, cols);
    }
}


// Batch GEMM: row-first order for batch (n_inputs > 1).
// For each weight row, computes dot products against all n_inputs Q8 vectors.
// Weight row data loaded once, Q8 vectors cycle through from L1/L2.
void qq_batch_gemm(const uint8_t* w_data, uint32_t w_type,
                   const uint8_t* q8_flat, int q8_stride, int n_inputs,
                   int cols, float* out_flat, int n_rows,
                   int out_stride, int bpr) {
    if (n_inputs <= 1) {
        for (int p = 0; p < n_inputs; p++) {
            const uint8_t* q8 = q8_flat + (size_t)p * q8_stride;
            float* out = out_flat + (size_t)p * out_stride;
            qq_dot_batch(w_data, w_type, q8, cols, out, n_rows, bpr);
        }
        return;
    }
    for (int r = 0; r < n_rows; r++) {
        const uint8_t* row = w_data + (size_t)r * bpr;
        if (r + 1 < n_rows)
            _mm_prefetch((const char*)(w_data + (size_t)(r+1) * bpr), _MM_HINT_T1);
        switch (w_type) {
        case 2:
            qq_batch_row_q4_0(row, q8_flat, q8_stride, n_inputs, cols, out_flat, out_stride, r);
            break;
        case 6:
            qq_batch_row_q5_0(row, q8_flat, q8_stride, n_inputs, cols, out_flat, out_stride, r);
            break;
        case 8:
            qq_batch_row_q8_0(row, q8_flat, q8_stride, n_inputs, cols, out_flat, out_stride, r);
            break;
        case 11:
            qq_batch_row_q3_K(row, q8_flat, q8_stride, n_inputs, cols, out_flat, out_stride, r);
            break;
        case 12:
            qq_batch_row_q4_K(row, q8_flat, q8_stride, n_inputs, cols, out_flat, out_stride, r);
            break;
        case 14:
            qq_batch_row_q6_K(row, q8_flat, q8_stride, n_inputs, cols, out_flat, out_stride, r);
            break;
        default:
            for (int p = 0; p < n_inputs; p++) {
                const uint8_t* q8 = q8_flat + (size_t)p * q8_stride;
                switch (w_type) {
                case 10: out_flat[(size_t)p * out_stride + r] = qq_dot_q2_K_q8_K(row, q8, cols); break;
                case 11: out_flat[(size_t)p * out_stride + r] = qq_dot_q3_K_q8_K(row, q8, cols); break;
                case 12: out_flat[(size_t)p * out_stride + r] = qq_dot_q4_K_q8_K(row, q8, cols); break;
                case 13: out_flat[(size_t)p * out_stride + r] = qq_dot_q5_K_q8_K(row, q8, cols); break;
                case 14: out_flat[(size_t)p * out_stride + r] = qq_dot_q6_K_q8_K(row, q8, cols); break;
                default: out_flat[(size_t)p * out_stride + r] = 0; break;
                }
            }
            break;
        }
    }
}

// Returns the Q8 buffer size needed for a given quantization type and element count.
// K-quant types use Q8_K (292 bytes per 256 elements), others use Q8_0 (34 bytes per 32 elements).
int q8_buffer_size(uint32_t w_type, int n) {
    switch (w_type) {
    case 10: case 11: case 12: case 13: case 14: // K-quants
        return (n / 256) * 292;
    default: // Q4_0, Q8_0, etc.
        return (n / 32) * 34;
    }
}

// Quantize float to the appropriate Q8 format for the weight type.
void quantize_for_type(const float* x, uint8_t* out, uint32_t w_type, int n) {
    switch (w_type) {
    case 10: case 11: case 12: case 13: case 14:
        quantize_to_q8_K(x, out, n);
        break;
    default:
        quantize_to_q8_0(x, out, n);
        break;
    }
}

void batch_quantize_for_type(const float* x_flat, uint8_t* out_flat,
                             uint32_t w_type, int cols, int q8_stride, int n_pos) {
    for (int p = 0; p < n_pos; p++) {
        quantize_for_type(x_flat + (size_t)p * cols,
                          out_flat + (size_t)p * q8_stride,
                          w_type, cols);
    }
}
