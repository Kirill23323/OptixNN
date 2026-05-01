#pragma once


#include "OptixNNlib/vector_types.h"
#include "OptixNNlib/norm_backend.h"
#include "OptixNNlib/dot_product.h" 

#include <cassert>

namespace OptixNN {

//cosine for normalized vectors 

inline float CosineDistanceNormed(const float_t* a_norm,
                                    const float_t* b_norm,
                                    dim_t padded_dim) noexcept
{
    float dot = Dot(a_norm, b_norm, padded_dim);
    return 1.0f - dot;
}

// cosine for NOT normalized 

inline float CosineDistanceWithNorms(const float_t* a,
                                        const float_t* b,
                                        float norm_a,
                                        float norm_b,
                                        dim_t padded_dim) noexcept
{
#ifdef KBEST_ENABLE_CHECKS
    assert(norm_a > 0.0f);
    assert(norm_b > 0.0f);
#endif

    float dot = Dot(a, b, padded_dim);
    float denom = norm_a * norm_b;
    if (denom <= 1e-12f) denom = 1e-12f;

    return 1.0f - (dot / denom);
}

// convenience wrapper 

inline float cosine_distance_auto(const float_t* a,
                                  const float_t* b,
                                  dim_t dim,
                                  dim_t padded_dim) noexcept
{
    float norm_a = ComputeNorm(a, dim);
    float norm_b = ComputeNorm(b, dim);

    return CosineDistanceWithNorms(a, b, norm_a, norm_b, padded_dim);
}

} // namespace OptixNN
