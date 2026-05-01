#include <gtest/gtest.h>
#include "OptixNNlib/dot_product.h"
#include <vector>
#include <cmath>

using namespace OptixNN;


inline void ExpectFloatNear(float a, float b, float eps = 1e-5f) {
    EXPECT_NEAR(a, b, eps);
}

float DotNaive(const float* a, const float* b, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i)
        acc += double(a[i]) * double(b[i]);
    return float(acc);
}

//Reference tests 

TEST(DotRefTest, Simple) {
    float a[4] = {1,2,3,4};
    float b[4] = {4,3,2,1};

    float expected = 1*4 + 2*3 + 3*2 + 4*1;
    ExpectFloatNear(DotRef(a,b,4), expected);
}

TEST(DotRefTest, WithZeros) {
    float a[6] = {1,2,3,0,0,0};
    float b[6] = {4,5,6,0,0,0};

    float expected = 1*4 + 2*5 + 3*6;
    ExpectFloatNear(DotRef(a,b,6), expected);
}

//SSE tests

#if defined(__SSE__) || defined(__SSE2__)

TEST(DotSseTest, MultipleOf4) {
    float a[8] = {1,2,3,4,5,6,7,8};
    float b[8] = {8,7,6,5,4,3,2,1};

    float expected = DotNaive(a,b,8);
    ExpectFloatNear(DotSse(a,b,8), expected);
}

TEST(DotSseTest, TailProcessing) {
    float a[6] = {1,2,3,4,5,6};
    float b[6] = {6,5,4,3,2,1};

    float expected = DotNaive(a,b,6);
    ExpectFloatNear(DotSse(a,b,6), expected);
}

#endif

//AVX2 tests 

#if defined(__AVX2__)

TEST(DotAvx2Test, MultipleOf8) {
    alignas(32) float a[16];
    alignas(32) float b[16];

    for (int i = 0; i < 16; ++i) {
        a[i] = float(i+1);
        b[i] = float(16-i);
    }

    float expected = DotNaive(a,b,16);
    ExpectFloatNear(DotAvx2(a,b,16), expected);
}

TEST(DotAvx2Test, WithTail) {
    alignas(32) float a[20];
    alignas(32) float b[20];

    for (int i = 0; i < 20; ++i) {
        a[i] = float(i+1);
        b[i] = float(20-i);
    }

    float expected = DotNaive(a,b,20);
    ExpectFloatNear(DotAvx2(a,b,20), expected);
}

#endif

//NEON tests 

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

TEST(DotNeonTest, MultipleOf4) {
    float a[8] = {1,2,3,4,5,6,7,8};
    float b[8] = {8,7,6,5,4,3,2,1};

    float expected = DotNaive(a,b,8);
    ExpectFloatNear(DotNeon(a,b,8), expected);
}

TEST(DotNeonTest, TailProcessing) {
    float a[6] = {1,2,3,4,5,6};
    float b[6] = {6,5,4,3,2,1};

    float expected = DotNaive(a,b,6);
    ExpectFloatNear(DotNeon(a,b,6), expected);
}

#endif

//Dispatcher tests 

TEST(DotDispatcherTest, MultipleOf8) {
    float a[16];
    float b[16];

    for (int i = 0; i < 16; ++i) {
        a[i] = float(i+1);
        b[i] = float(16-i);
    }

    float expected = DotNaive(a,b,16);
    ExpectFloatNear(Dot(a,b,16), expected);
}

TEST(DotDispatcherTest, MultipleOf4) {
    float a[8];
    float b[8];

    for (int i = 0; i < 8; ++i) {
        a[i] = float(i+1);
        b[i] = float(8-i);
    }

    float expected = DotNaive(a,b,8);
    ExpectFloatNear(Dot(a,b,8), expected);
}

TEST(DotDispatcherTest, NonMultiple) {
    float a[7];
    float b[7];

    for (int i = 0; i < 7; ++i) {
        a[i] = float(i+1);
        b[i] = float(7-i);
    }

    float expected = DotNaive(a,b,7);
    ExpectFloatNear(Dot(a,b,7), expected);
}

/* ============================================================
 *                  Numerical stability
 * ============================================================ */

TEST(DotTest, LargeValues) {
    float a[4] = {1e6f, 2e6f, 3e6f, 4e6f};
    float b[4] = {1e6f, 2e6f, 3e6f, 4e6f};

    float expected = DotNaive(a,b,4);
    ExpectFloatNear(Dot(a,b,4), expected, 1e-2f); // допускаем большую погрешность
}

TEST(DotTest, SmallValues) {
    float a[4] = {1e-6f, 2e-6f, 3e-6f, 4e-6f};
    float b[4] = {1e-6f, 2e-6f, 3e-6f, 4e-6f};

    float expected = DotNaive(a,b,4);
    ExpectFloatNear(Dot(a,b,4), expected, 1e-12f);
}