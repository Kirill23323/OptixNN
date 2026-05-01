#pragma once
#include <cstddef>
#include <cstdlib>

namespace OptixNN {

inline void* AlignedMalloc(std::size_t size, std::size_t alignment) {
#if defined(_MSC_VER) || defined(_WIN32)
    return _aligned_malloc(size, alignment);
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) && !defined(__APPLE__)
    // std::aligned_alloc (C11)
    if (size % alignment != 0) {
        size = ((size + alignment - 1) / alignment) * alignment;
    }
    return std::aligned_alloc(alignment, size);
#else
    void* p = nullptr;
    if (posix_memalign(&p, alignment, size) != 0) {
        return nullptr;
    }
    return p;
#endif
}

inline void AlignedFree(void* p) {
    if (!p) return;
#if defined(_MSC_VER) || defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

} // namespace OptixNN
