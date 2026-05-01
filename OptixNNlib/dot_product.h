#pragma once
#include "OptixNNlib/vector_types.h"
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cassert>



// dot_product.h



#if defined(__AVX2__)
#   include <immintrin.h>
#endif

#if defined(__SSE__) || defined(__SSE2__)
#   include <xmmintrin.h>
#   include <emmintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#   include <arm_neon.h>
#endif

namespace OptixNN {

//Reference implementation

inline float DotRef(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
    double acc = 0.0;
    for (dim_t i = 0; i < padded_dim; ++i) {
        acc += double(a[i]) * double(b[i]);
    }
    return float(acc);
}

//AVX2 implementation 

#if defined(__AVX2__)

inline float DotAvx2(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    dim_t i = 0;
    const dim_t step = 16;

    for (; i + step - 1 < padded_dim; i += step) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    }

    for (; i + 7 < padded_dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
    }

    __m256 acc = _mm256_add_ps(acc0, acc1);

    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float partial = _mm_cvtss_f32(sum128);

    double tail = 0.0;
    for (; i < padded_dim; ++i) {
        tail += double(a[i]) * double(b[i]);
    }

    return float(partial + tail);
}
#else
inline float DotAvx2(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
    return DotRef(a, b, padded_dim);
}
#endif

//SSE2 implementation 

#if defined(__SSE__) || defined(__SSE2__)

inline float DotSse(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
    __m128 acc = _mm_setzero_ps();
    dim_t i = 0;

    for (; i + 3 < padded_dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }

    float tmp[4];
    _mm_storeu_ps(tmp, acc);
    double partial = double(tmp[0]) + tmp[1] + tmp[2] + tmp[3];

    double tail = 0.0;
    for (; i < padded_dim; ++i)
        tail += double(a[i]) * double(b[i]);

    return float(partial + tail);
}
#else
inline float DotSse(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
    return DotRef(a, b, padded_dim);
}
#endif


//NEON implementation 
#if defined(__ARM_NEON) || defined(__ARM_NEON__)

inline float DotNeon(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
    
    float32x4_t vacc = vdupq_n_f32(0.0f);
    dim_t i = 0;

    for (; i + 3 < padded_dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vacc = vmlaq_f32(vacc, va, vb);
    }

    float tmp[4];
    vst1q_f32(tmp, vacc);
    double partial = double(tmp[0]) + tmp[1] + tmp[2] + tmp[3];

    double tail = 0.0;
    for (; i < padded_dim; ++i)
        tail += double(a[i]) * double(b[i]);

    return float(partial + tail);
}
#else
inline float DotNeon(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
    return DotRef(a, b, padded_dim);
}
#endif

//Dispatcher implementation 

inline float Dot(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {

#if defined(__AVX2__)
    if (padded_dim % 8 == 0)
        return DotAvx2(a, b, padded_dim);
#endif

#if defined(__SSE__) || defined(__SSE2__)
    if (padded_dim % 4 == 0)
        return DotSse(a, b, padded_dim);
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (padded_dim % 4 == 0)
        return DotNeon(a, b, padded_dim);
#endif

    return DotRef(a, b, padded_dim);
}
} // namespace OptixNN

