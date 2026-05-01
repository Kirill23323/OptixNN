#pragma once

#include "OptixNNlib/vector_types.h"
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace OptixNN {


inline dim_t PaddedDimFor(dim_t dim, dim_t simd_width) {
    if (simd_width == 0) {
        return dim;
    }
    return ((dim + simd_width - 1) / simd_width) * simd_width;
}


inline void CopyAndPad(float_t* dst, const float_t* src, dim_t dim, dim_t padded_dim) {
    std::memcpy(dst, src, sizeof(float_t) * dim);
    if (padded_dim > dim) {
        std::memset(dst + dim, 0, sizeof(float_t) * (padded_dim - dim));
    }
}

} // namespace OptixNN
