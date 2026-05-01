#pragma once

//norm_backend.h


#include "OptixNNlib/vector_types.h"
#include <cstddef>
#include <cmath>
#include <cassert>

#if defined(__AVX2__)
    #include <immintrin.h>
#endif

#if defined(__SSE__) || defined(__SSE2__)
    #include <xmmintrin.h>
    #include <emmintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
#endif

namespace OptixNN {

static constexpr float kNormEps = 1e-12f;

//Reference implementation 

inline float ComputeNormRef(const float_t* a, dim_t dim) noexcept {
    double acc = 0.0;
    for (dim_t i = 0; i < dim; ++i) {
        double v = static_cast<double>(a[i]);
        acc += v * v;
    }
    return static_cast<float>(std::sqrt(acc));
}

inline float ComputeNormPaddedRef(const float_t* a, dim_t padded_dim) noexcept {
    double acc = 0.0;
    for (dim_t i = 0; i < padded_dim; ++i) {
        double v = static_cast<double>(a[i]);
        acc += v * v;
    }
    return static_cast<float>(std::sqrt(acc));
}

inline void NormalizeInplaceRef(float_t* a, dim_t dim) noexcept {
    float n = ComputeNormRef(a, dim);
    if (n <= kNormEps){
        return;
    }
    float inv = 1.0f / n;
    for (dim_t i = 0; i < dim; ++i) a[i] *= inv;
}

inline void NormalizeInplacePaddedRef(float_t* a, dim_t dim, dim_t padded_dim) noexcept {
    NormalizeInplaceRef(a, dim);
    for (dim_t i = dim; i < padded_dim; ++i) a[i] = 0.0f;
}

//AVX2 implementation 

#if defined(__AVX2__)


inline float ComputeNormPaddedAvx2(const float_t* a, dim_t padded_dim) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    dim_t i = 0;
    const dim_t block16 = 16;

    for (; i + (block16 - 1) < padded_dim; i += block16) {
        __m256 v0 = _mm256_load_ps(a + i);
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);

        __m256 v1 = _mm256_load_ps(a + i + 8);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
    }
    for (; i + 7 < padded_dim; i += 8) {
        __m256 v = _mm256_load_ps(a + i);
        acc0 = _mm256_fmadd_ps(v, v, acc0);
    }

    __m256 s = _mm256_add_ps(acc0, acc1);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 lo = _mm256_castps256_ps128(s);

    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    float v = _mm_cvtss_f32(sum128);
    return std::sqrt(v);
}

inline float ComputeNormAvx2(const float_t* a, dim_t dim) noexcept {
    dim_t i = 0;

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (; i + 15 < dim; i += 16) {
        __m256 v0 = _mm256_loadu_ps(a + i);
        __m256 v1 = _mm256_loadu_ps(a + i + 8);
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
    }
    for (; i + 7 < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        acc0 = _mm256_fmadd_ps(v, v, acc0);
    }

    __m256 s = _mm256_add_ps(acc0, acc1);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 lo = _mm256_castps256_ps128(s);

    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    float partial = _mm_cvtss_f32(sum128);

    double tail = 0.0;
    for (; i < dim; ++i) {
        double v = static_cast<double>(a[i]);
        tail += v * v;
    }

    return static_cast<float>(std::sqrt(partial + tail));
}

inline void NormalizeInplaceAvx2(float_t* a, dim_t dim) noexcept {
    float n = ComputeNormAvx2(a, dim);
    if (n <= kNormEps){
        return;
    }

    const __m256 invv = _mm256_set1_ps(1.0f / n);

    dim_t i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        v = _mm256_mul_ps(v, invv);
        _mm256_storeu_ps(a + i, v);
    }
    for (; i < dim; ++i) a[i] *= (1.0f / n);
}

inline void NormalizeInplacePaddedAvx2(float_t* a, dim_t dim, dim_t padded_dim) noexcept {
    NormalizeInplaceAvx2(a, dim);
    for (dim_t i = dim; i < padded_dim; ++i) a[i] = 0.0f;
}

#else // no AVX2

inline float ComputeNormPaddedAvx2(const float_t* a, dim_t p) noexcept { return ComputeNormPaddedRef(a, p); }
inline float ComputeNormAvx2(const float_t* a, dim_t d) noexcept { return ComputeNormRef(a, d); }
inline void NormalizeInplaceAvx2(float_t* a, dim_t d) noexcept { NormalizeInplaceRef(a, d); }
inline void NormalizeInplacePaddedAvx2(float_t* a, dim_t d, dim_t p) noexcept {
    NormalizeInplacePaddedRef(a, d, p);
}

#endif // AVX2


//SSE2 implementation 

#if defined(__SSE__) || defined(__SSE2__)

inline float ComputeNormPaddedSse(const float_t* a, dim_t padded_dim) noexcept {
    __m128 acc = _mm_setzero_ps();

    for (dim_t i = 0; i < padded_dim; i += 4) {
        __m128 v = _mm_loadu_ps(a + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(v, v));
    }

    float tmp[4];
    _mm_storeu_ps(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    return std::sqrt(s);
}

inline float ComputeNormSse(const float_t* a, dim_t dim) noexcept {
    dim_t i = 0;
    __m128 acc = _mm_setzero_ps();

    for (; i + 3 < dim; i += 4) {
        __m128 v = _mm_loadu_ps(a + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(v, v));
    }

    float tmp[4];
    _mm_storeu_ps(tmp, acc);
    double s = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < dim; ++i) {
        double v = static_cast<double>(a[i]);
        s += v * v;
    }

    return static_cast<float>(std::sqrt(s));
}

inline void NormalizeInplaceSse(float_t* a, dim_t dim) noexcept {
    float n = ComputeNormSse(a, dim);
    if (n <= kNormEps){
        return;
    }

    __m128 invv = _mm_set1_ps(1.0f / n);
    dim_t i = 0;

    for (; i + 3 < dim; i += 4) {
        __m128 v = _mm_loadu_ps(a + i);
        v = _mm_mul_ps(v, invv);
        _mm_storeu_ps(a + i, v);
    }

    for (; i < dim; ++i) a[i] *= (1.0f / n);
}

inline void NormalizeInplacePaddedSse(float_t* a, dim_t dim, dim_t padded_dim) noexcept {
    NormalizeInplaceSse(a, dim);
    for (dim_t i = dim; i < padded_dim; ++i) a[i] = 0.0f;
}

#else // no SSE

inline float ComputeNormPaddedSse(const float_t* a, dim_t p) noexcept { return ComputeNormPaddedRef(a, p); }
inline float ComputeNormSse(const float_t* a, dim_t d) noexcept { return ComputeNormRef(a, d); }
inline float NormalizeInplaceSse(float_t* a, dim_t d) noexcept { NormalizeInplaceRef(a, d); }
inline float NormalizeInplacePaddedSse(float_t* a, dim_t d, dim_t p) noexcept {
    NormalizeInplacePaddedRef(a, d, p);
}

#endif // SSE


//NEON implementation 

#if defined(__ARM_NEON) || defined(__ARM_NEON__)


inline float ComputeNormPaddedNeon(const float_t* a, dim_t padded_dim) noexcept {
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (dim_t i = 0; i + 3 < padded_dim; i += 4) {
        float32x4_t v = vld1q_f32(a + i);
        acc = vmlaq_f32(acc, v, v);
    }

    float tmp[4];
    vst1q_f32(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    return std::sqrt(s);
}

inline float ComputeNormNeon(const float_t* a, dim_t dim) noexcept {
    dim_t i = 0;
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (; i + 3 < dim; i += 4) {
        float32x4_t v = vld1q_f32(a + i);
        acc = vmlaq_f32(acc, v, v);
    }

    float tmp[4];
    vst1q_f32(tmp, acc);
    double s = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < dim; ++i) {
        double v = static_cast<double>(a[i]);
        s += v * v;
    }

    return static_cast<float>(std::sqrt(s));
}

inline void NormalizeInplaceNeon(float_t* a, dim_t dim) noexcept {
    float n = ComputeNormNeon(a, dim);
    if (n <= kNormEps){
        return;
    }

    float inv = 1.0f / n;
    float32x4_t invv = vdupq_n_f32(inv);

    dim_t i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t v = vld1q_f32(a + i);
        v = vmulq_f32(v, invv);
        vst1q_f32(a + i, v);
    }

    for (; i < dim; ++i) a[i] *= inv;

}

inline void NormalizeInplacePaddedNeon(float_t* a, dim_t dim, dim_t padded_dim) noexcept {
    NormalizeInplaceNeon(a, dim);
    for (dim_t i = dim; i < padded_dim; ++i) a[i] = 0.0f;
}

#else // no NEON

inline float ComputeNormPaddedNeon(const float_t* a, dim_t p) noexcept { return ComputeNormPaddedRef(a, p); }
inline float ComputeNormNeon(const float_t* a, dim_t d) noexcept { return ComputeNormRef(a, d); }
inline void NormalizeInplaceNeon(float_t* a, dim_t d) noexcept { NormalizeInplaceRef(a, d); }
inline void NormalizeInplacePaddedNeon(float_t* a, dim_t d, dim_t p) noexcept {
    NormalizeInplacePaddedRef(a, d, p);
}

#endif // NEON


//Dispatcher 

inline float ComputeNorm(const float_t* a, dim_t dim) noexcept {
#if defined(__AVX2__)
    return ComputeNormAvx2(a, dim);
#elif defined(__SSE__) || defined(__SSE2__)
    return ComputeNormSse(a, dim);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    return ComputeNormNeon(a, dim);
#else
    return ComputeNormRef(a, dim);
#endif
}

inline float ComputeNormPadded(const float_t* a, dim_t padded_dim) noexcept {
#if defined(__AVX2__)
    if (padded_dim % 8 == 0) return ComputeNormPaddedAvx2(a, padded_dim);
    return ComputeNormAvx2(a, padded_dim);

#elif defined(__SSE__) || defined(__SSE2__)
    if (padded_dim % 4 == 0) return ComputeNormPaddedSse(a, padded_dim);
    return ComputeNormSse(a, padded_dim);

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (padded_dim % 4 == 0) return ComputeNormPaddedNeon(a, padded_dim);
    return ComputeNormNeon(a, padded_dim);

#else
    return ComputeNormPaddedRef(a, padded_dim);
#endif
}

inline void NormalizeInplace(float_t* a, dim_t dim) noexcept {
#if defined(__AVX2__)
    NormalizeInplaceAvx2(a, dim);
#elif defined(__SSE__) || defined(__SSE2__)
    NormalizeInplaceSse(a, dim);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    NormalizeInplaceNeon(a, dim);
#else
    NormalizeInplaceRef(a, dim);
#endif
}

inline void NormalizeInplacePadded(float_t* a, dim_t dim, dim_t padded_dim) noexcept {
#if defined(__AVX2__)
    NormalizeInplacePaddedAvx2(a, dim, padded_dim);
#elif defined(__SSE__) || defined(__SSE2__)
    NormalizeInplacePaddedSse(a, dim, padded_dim);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    NormalizeInplacePaddedNeon(a, dim, padded_dim);
#else
    NormalizeInplacePaddedRef(a, dim, padded_dim);
#endif
}

} // namespace OptixNN
