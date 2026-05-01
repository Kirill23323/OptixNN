#pragma once

#include <cstddef>
#include <cstdint>

namespace OptixNN {

// Основные типы
using float_t = float;
using dim_t = std::size_t;
using internal_id = std::uint32_t;

// Тип метрики 
enum class MetricType : std::uint8_t {
    L2,
    COSINE
};

} // namespace OptixNN
