cmake_minimum_required(VERSION 3.12)
project(OptixNN LANGUAGES CXX)

# ==========================================
#              COMPILER
# ==========================================
if(NOT CMAKE_C_COMPILER)
    find_program(GCC_14_PATH gcc-14)
    if(GCC_14_PATH)
        set(CMAKE_C_COMPILER "${GCC_14_PATH}")
    endif()
endif()

if(NOT CMAKE_CXX_COMPILER)
    find_program(GPP_14_PATH g++-14)
    if(GPP_14_PATH)
        set(CMAKE_CXX_COMPILER "${GPP_14_PATH}")
    endif()
endif()

message(STATUS "Using C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "Using CXX compiler: ${CMAKE_CXX_COMPILER}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
#               OPTIONS
# ==========================================
option(OPTIXNN_USE_ASAN "Enable AddressSanitizer" OFF)

if (OPTIXNN_USE_ASAN)
    message(STATUS "AddressSanitizer enabled")

    add_compile_options(
        -fsanitize=address
        -fno-omit-frame-pointer
        -O1
        -g
    )

    add_link_options(-fsanitize=address)
endif()

# SIMD выбор (ключевая часть)
option(OPTIXNN_ENABLE_AVX2 "Enable AVX2 backend" OFF)
option(OPTIXNN_ENABLE_SSE  "Enable SSE backend"  OFF)

# ==========================================
#          DETECT ARCHITECTURE
# ==========================================
message(STATUS "Detected architecture: ${CMAKE_SYSTEM_PROCESSOR}")

set(OPTIXNN_ARCH_X86 OFF)
set(OPTIXNN_ARCH_ARM OFF)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(OPTIXNN_ARCH_X86 ON)
    message(STATUS "Building for x86_64")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(OPTIXNN_ARCH_ARM ON)
    message(STATUS "Building for ARM64 (NEON)")
endif()

# ==========================================
#             COMMON FLAGS
# ==========================================
set(COMMON_FLAGS
    -O3
    -fno-omit-frame-pointer
)

if (OPTIXNN_USE_ASAN)
    message(STATUS "AddressSanitizer enabled")
    set(ASAN_FLAGS "-fsanitize=address")
    set(COMMON_FLAGS "${COMMON_FLAGS} ${ASAN_FLAGS}")
endif()

add_compile_options(${COMMON_FLAGS})

# ==========================================
#         SIMD CONFIGURATION
# ==========================================
if (OPTIXNN_ARCH_X86)

    if (OPTIXNN_ENABLE_AVX2)
        message(STATUS "SIMD: AVX2 enabled")
        add_compile_options(-mavx2 -mfma)

    elseif (OPTIXNN_ENABLE_SSE)
        message(STATUS "SIMD: SSE4.2 enabled")
        add_compile_options(-msse4.2 -mno-avx -mno-avx2)

    else()
        message(STATUS "SIMD: scalar (no SIMD flags)")
    endif()

endif()

if (OPTIXNN_ARCH_ARM)
    message(STATUS "SIMD: NEON (ARMv8 default)")
    add_compile_options(-march=armv8-a)
endif()

# ==========================================
#               OPENMP
# ==========================================
find_package(OpenMP REQUIRED)

if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP enabled")
else()
    message(FATAL_ERROR "OpenMP NOT FOUND")
endif()

# ==========================================
#     HEADER-ONLY LIBRARY: OPTIXNN
# ==========================================
add_library(OptixNN INTERFACE)

target_include_directories(OptixNN INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(OptixNN INTERFACE OpenMP::OpenMP_CXX)

# ==========================================
#               SUBDIRS
# ==========================================
add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)