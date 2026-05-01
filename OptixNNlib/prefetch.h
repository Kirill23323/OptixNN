#pragma once

// prefetches

#if defined(__x86_64__) || defined(_M_X64)

// x86_64

#include <xmmintrin.h>

inline void PrefetchL1(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T0);
}
inline void PrefetchL2(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T1);
}
inline void PrefetchL3(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T2);
}

#elif defined(__aarch64__)

// ARM64

#ifdef _MSC_VER

// todo: arm on MSVC
inline void PrefetchL1(const void* address) {}
inline void PrefetchL2(const void* address) {}
inline void PrefetchL3(const void* address) {}

#else
// arm on non-MSVC

inline void PrefetchL1(const void* address) {
    __builtin_prefetch(address, 0, 3);
}
inline void PrefetchL2(const void* address) {
    __builtin_prefetch(address, 0, 2);
}
inline void PrefetchL3(const void* address) {
    __builtin_prefetch(address, 0, 1);
}
#endif

#else

// a generic platform

#ifdef _MSC_VER

inline void PrefetchL1(const void* address) {}
inline void PrefetchL2(const void* address) {}
inline void PrefetchL3(const void* address) {}

#else

inline void PrefetchL1(const void* address) {
    __builtin_prefetch(address, 0, 3);
}
inline void PrefetchL2(const void* address) {
    __builtin_prefetch(address, 0, 2);
}
inline void PrefetchL3(const void* address) {
    __builtin_prefetch(address, 0, 1);
}

#endif

#endif