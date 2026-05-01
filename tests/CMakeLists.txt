cmake_minimum_required(VERSION 3.14)

# ==============================
#       GOOGLE TEST
# ==============================
include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# ==============================
#       COLLECT FILES
# ==============================
file(GLOB_RECURSE TEST_SOURCES
    distance/*.cpp
    storage/*.cpp
    common/*.cpp
    simd/*.cpp
    asan/*.cpp
    omp/*.cpp
    rnndescent/*.cpp
)

# ==============================
#       TEST EXECUTABLE
# ==============================
add_executable(OptixNN_tests ${TEST_SOURCES})

target_link_libraries(OptixNN_tests
    PRIVATE
        OptixNN
        gtest
        gtest_main
        OpenMP::OpenMP_CXX
)

target_include_directories(OptixNN_tests PRIVATE
    ${PROJECT_SOURCE_DIR}
)

# ==============================
#       AUTO DISCOVER
# ==============================
include(GoogleTest)
gtest_discover_tests(OptixNN_tests)