# OptixNN — Build and Test Guide

## Approximate nearest neighbor search lib

## Requirements

Install dependencies:

```bash
sudo apt update
sudo apt install -y build-essential cmake git libomp-dev
```

Recommended compilers:

* GCC 10+
* Clang 12+

Check versions:

```bash
gcc --version
g++ --version
```

---

## Build Instructions

The project supports multiple SIMD backends via CMake options.

### AVX2 (x86_64, recommended)

```bash
cmake -B build/avx2 -DOPTIXNN_ENABLE_AVX2=ON
cmake --build build/avx2 -j

```

---

### SSE4.2

```bash

cmake -B build/avx2 -DOPTIXNN_ENABLE_SSE=ON
cmake --build build/avx2 -j

```

---

### Scalar (no SIMD)

```bash
cmake -B build/scalar
cmake --build build/scalar -j
```

---

### NEON (ARM)

On ARM (aarch64), NEON is enabled automatically:

```bash

cmake -B build/neon
cmake --build build/neon -j
```

---

### Build with AddressSanitizer (debugging)

```bash

cmake -B build/asan \
      -DCMAKE_BUILD_TYPE=Debug \
      -DOPTIXNN_USE_ASAN=ON

cmake --build build/asan -j

```

---


## Running tests from Project Root

```bash
./build/avx2/tests/OptixNN_tests
```

With filter:

```bash
./build/avx2/tests/OptixNN_tests --gtest_filter=VectorStorage.*
```

---

## Useful GoogleTest Options

```bash
--gtest_list_tests
--gtest_repeat=10
--gtest_shuffle
```

---

## Notes

* All vectors in `VectorStorage` are normalized during insertion.
* Use `EXPECT_NEAR` instead of `EXPECT_FLOAT_EQ` in tests.
* OpenMP is enabled. You can control threads:

```bash
export OMP_NUM_THREADS=4
```
* The project is still in its early stages of development and may contain bugs in the code, which will be fixed over time :D

---

## Deterministic Testing

For stable results:

* use deterministic versions of algorithms (without OpenMP)
* fix random seed in code:

```cpp
rng.seed(42);
```

---


