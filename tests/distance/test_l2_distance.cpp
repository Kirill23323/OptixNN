#include <gtest/gtest.h>
#include "OptixNNlib/distance_l2.h"
#include <vector>
#include <cmath>
#include <memory>

using namespace OptixNN;

// Вспомогательная функция для сравнения float с допуском
inline void ExpectFloatNear(float a, float b, float eps = 1e-5f) {
    EXPECT_NEAR(a, b, eps);
}

/* ----------------------- Reference ----------------------- */

TEST(L2SqrRefTest, SimpleVectors) {
    float a[4] = {1.f, 2.f, 3.f, 4.f};
    float b[4] = {4.f, 3.f, 2.f, 1.f};
    float expected = (1-4)*(1-4) + (2-3)*(2-3) + (3-2)*(3-2) + (4-1)*(4-1); // 20
    ExpectFloatNear(L2SqrRef(a, b, 4), expected);
}

TEST(L2SqrRefTest, ZeroVector) {
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    float b[4] = {1.f, 2.f, 3.f, 4.f};
    float expected = 1*1 + 2*2 + 3*3 + 4*4; // 30
    ExpectFloatNear(L2SqrRef(a, b, 4), expected);
}

/* ----------------------- SSE ----------------------- */
#if defined(__SSE__) || defined(__SSE2__)

TEST(L2SqrSseTest, SimpleVectors) {
    float a[8] = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    float b[8] = {8.f,7.f,6.f,5.f,4.f,3.f,2.f,1.f};
    float expected = 0;
    for (int i=0; i<8; ++i) expected += (a[i]-b[i])*(a[i]-b[i]);
    ExpectFloatNear(L2SqrSse(a, b, 8), expected);
}

TEST(L2SqrSseTest, NonMultipleOfFourTail) {
    float a[6] = {1,2,3,4,5,6};
    float b[6] = {6,5,4,3,2,1};
    float expected = 0;
    for (int i=0;i<6;++i) expected += (a[i]-b[i])*(a[i]-b[i]);
    ExpectFloatNear(L2SqrSse(a,b,6), expected);
}

#endif

/* ----------------------- AVX2 ----------------------- */
#if defined(__AVX2__)

TEST(L2SqrAvx2Test, AlignedVectors) {
    alignas(32) float a[16];
    alignas(32) float b[16];
    for (int i=0; i<16; ++i) { a[i]=i+1.f; b[i]=16.f-i; }
    float expected = 0;
    for (int i=0;i<16;++i) expected += (a[i]-b[i])*(a[i]-b[i]);
    ExpectFloatNear(L2SqrAvx2(a,b,16), expected);
}

TEST(L2SqrAvx2Test, PartialLoop) {
    alignas(32) float a[24];
    alignas(32) float b[24];
    for (int i=0; i<24; ++i) { a[i]=i+1.f; b[i]=24.f-i; }
    float expected = 0;
    for (int i=0;i<24;++i) expected += (a[i]-b[i])*(a[i]-b[i]);
    ExpectFloatNear(L2SqrAvx2(a,b,24), expected);
}

#endif

/* ----------------------- NEON ----------------------- */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)

TEST(L2SqrNeonTest, SimpleVectors) {
    float a[8] = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    float b[8] = {8.f,7.f,6.f,5.f,4.f,3.f,2.f,1.f};
    float expected = 0;
    for (int i=0;i<8;++i) expected += (a[i]-b[i])*(a[i]-b[i]);
    ExpectFloatNear(L2SqrNeon(a,b,8), expected);
}

#endif

/* ----------------------- Dispatcher ----------------------- */

TEST(L2SqrDispatcherTest, SmallDimUsesSSE) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    float a[4] = {1,2,3,4};
    float b[4] = {4,3,2,1};
    float expected = (1-4)*(1-4)+(2-3)*(2-3)+(3-2)*(3-2)+(4-1)*(4-1); // 20
    ExpectFloatNear(L2Sqr(a,b,4), expected);
#endif
}

TEST(L2SqrDispatcherTest, LargerDim) {
    float a[8] = {1,2,3,4,5,6,7,8};
    float b[8] = {8,7,6,5,4,3,2,1};
    float expected = 0;
    for (int i=0;i<8;++i) expected += (a[i]-b[i])*(a[i]-b[i]);
    ExpectFloatNear(L2Sqr(a,b,8), expected);
}

TEST(L2SqrDispatcherTest, NonAlignedDim) {
    float a[10] = {0,1,2,3,4,5,6,7,8,9};
    float b[10] = {9,8,7,6,5,4,3,2,1,0};
    float expected = 0;
    for (int i=0;i<10;++i) expected += (a[i]-b[i])*(a[i]-b[i]);
    ExpectFloatNear(L2Sqr(a,b,10), expected);
}