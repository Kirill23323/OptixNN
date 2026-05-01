#pragma once

//distance_l2.h


#include "OptixNNlib/vector_types.h"
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cassert>

#if defined(__AVX2__)
  #include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
#endif

#if defined(__SSE__) || defined(__SSE2__)
  #include <xmmintrin.h>
  #include <emmintrin.h>
#endif

namespace OptixNN {

// utils

inline bool IsAlignedPtr(const void* ptr, std::size_t align) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr) & (align - 1)) == 0;
}

//Reference implementation

inline float L2SqrRef(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
    double acc = 0.0;
    for (dim_t i = 0; i < padded_dim; ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    return static_cast<float>(acc);
}

//SSE2 implementation 

#if defined(__SSE__) || defined(__SSE2__)


/* portable horizontal reduce for __m128 -> float */
inline float ReduceM128ToFloatPortable(__m128 v) noexcept {
    float tmp[4];
    _mm_storeu_ps(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

inline float L2SqrSse(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
#ifdef KBEST_ENABLE_CHECKS
    assert(a != nullptr && b != nullptr);
    assert(padded_dim >= 0);
#endif

    __m128 vacc = _mm_setzero_ps();
    dim_t i = 0;
    for (; i + 3 < padded_dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        __m128 sq = _mm_mul_ps(diff, diff);
        vacc = _mm_add_ps(vacc, sq);
    }

    float s = ReduceM128ToFloatPortable(vacc);

    double tail = 0.0;
    for (; i < padded_dim; ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        tail += d * d;
    }

    return static_cast<float>(static_cast<double>(s) + tail);
}

#else
inline float L2SqrSse(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
    return L2SqrRef(a, b, padded_dim);
}
#endif // SSE

//AVX2 implementation 

#if defined(__AVX2__)


inline float L2SqrAvx2(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
#ifdef KBEST_ENABLE_CHECKS
    assert(padded_dim % 8 == 0 && "padded_dim must be multiple of 8 for AVX2 kernel");
    assert(a != nullptr && b != nullptr);
    assert(IsAlignedPtr(a, 32) && IsAlignedPtr(b, 32) && "Pointers should be 32-byte aligned for aligned loads");
#endif

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    dim_t i = 0;
    const dim_t step = 16;
    for (; i + (step - 1) < padded_dim; i += step) {
        __m256 va0 = _mm256_load_ps(a + i);
        __m256 vb0 = _mm256_load_ps(b + i);
        __m256 diff0 = _mm256_sub_ps(va0, vb0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        __m256 va1 = _mm256_load_ps(a + i + 8);
        __m256 vb1 = _mm256_load_ps(b + i + 8);
        __m256 diff1 = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
    }

    for (; i + 7 < padded_dim; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc0 = _mm256_fmadd_ps(diff, diff, acc0);
    }

    __m256 acc = _mm256_add_ps(acc0, acc1);

    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    return result;
}

#else
inline float L2SqrAvx2(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
    return L2SqrRef(a, b, padded_dim);
}
#endif // __AVX2__

//NEON implementation 

#if defined(__ARM_NEON) || defined(__ARM_NEON__)


inline float L2SqrNeon(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
#ifdef KBEST_ENABLE_CHECKS
    assert(padded_dim % 4 == 0 && "padded_dim must be multiple of 4 for NEON kernel");
    assert(a != nullptr && b != nullptr);
#endif

    float32x4_t vacc = vdupq_n_f32(0.0f);
    dim_t i = 0;
    for (; i + 3 < padded_dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        float32x4_t sq = vmulq_f32(diff, diff);
        vacc = vaddq_f32(vacc, sq);
    }
    float tmp[4];
    vst1q_f32(tmp, vacc);
    double acc = static_cast<double>(tmp[0]) + tmp[1] + tmp[2] + tmp[3];

    for (; i < padded_dim; ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }

    return static_cast<float>(acc);
}

#else
inline float L2SqrNeon(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {
    return L2SqrRef(a, b, padded_dim);
}
#endif // NEON

//Dispatcher 


inline float L2Sqr(const float_t* a, const float_t* b, dim_t padded_dim) noexcept {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    // prefer SSE for tiny padded_dim == 4 (avoid AVX overkill)
    if (padded_dim == 4) {
  #if defined(__SSE__) || defined(__SSE2__)
        return L2SqrSse(a, b, padded_dim);
  #else
        return L2SqrRef(a, b, padded_dim);
  #endif
    }
#endif

#if defined(__AVX2__)
    if (padded_dim % 8 == 0) {
        return L2SqrAvx2(a, b, padded_dim);
    }
#endif

#if defined(__SSE__) || defined(__SSE2__)
    if (padded_dim % 4 == 0) {
        return L2SqrSse(a, b, padded_dim);
    }
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return L2SqrNeon(a, b, padded_dim);
#else
    return L2SqrRef(a, b, padded_dim);
#endif
}

} // namespace OptixNN
