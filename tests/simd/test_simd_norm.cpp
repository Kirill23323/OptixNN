#include <gtest/gtest.h>
#include "OptixNNlib/norm_backend.h"
#include <vector>
#include <cmath>

using namespace OptixNN;

inline void ExpectFloatNear(float a, float b, float eps = 1e-5f) {
    EXPECT_NEAR(a, b, eps);
}

//Reference tests 

TEST(NormRefTest, ComputeNormSimple) {
    float a[3] = {3.f, 4.f, 12.f};
    float expected = std::sqrt(3*3 + 4*4 + 12*12);
    ExpectFloatNear(ComputeNormRef(a, 3), expected);
}

TEST(NormRefTest, ComputeNormPadded) {
    float a[5] = {1.f, 2.f, 2.f, 0.f, 0.f};
    float expected = std::sqrt(1+4+4);
    ExpectFloatNear(ComputeNormPaddedRef(a, 5), expected);
}

TEST(NormRefTest, NormalizeInplace) {
    float a[3] = {3.f, 4.f, 0.f};
    float original_norm = ComputeNormRef(a, 3);
    NormalizeInplaceRef(a, 3);
    ExpectFloatNear(a[0], 3.f/original_norm);
    ExpectFloatNear(a[1], 4.f/original_norm);
    ExpectFloatNear(a[2], 0.f);
}

TEST(NormRefTest, NormalizeInplacePadded) {
    float a[5] = {3.f, 4.f, 0.f, 9.f, 10.f};
    float original_norm = std::sqrt(3*3+4*4); // norm calculated using only first 2 elements
    NormalizeInplacePaddedRef(a, 2, 5);
    ExpectFloatNear(a[0], 3.f/original_norm);
    ExpectFloatNear(a[1], 4.f/original_norm);
    EXPECT_FLOAT_EQ(a[2], 0.f);
    EXPECT_FLOAT_EQ(a[3], 0.f);
    EXPECT_FLOAT_EQ(a[4], 0.f);
}

//SSE tests 
#if defined(__SSE__) || defined(__SSE2__)

TEST(NormSseTest, ComputeNorm) {
    float a[5] = {1.f, 2.f, 3.f, 4.f, 5.f};
    float expected = std::sqrt(1+4+9+16+25);
    ExpectFloatNear(ComputeNormSse(a,5), expected);
}

TEST(NormSseTest, NormalizeInplace) {
    float a[4] = {3.f, 4.f, 0.f, 0.f};
    float original_norm = ComputeNormSse(a, 4);
    NormalizeInplaceSse(a, 4);
    ExpectFloatNear(a[0], 3.f/original_norm);
    ExpectFloatNear(a[1], 4.f/original_norm);
    EXPECT_FLOAT_EQ(a[2], 0.f);
    EXPECT_FLOAT_EQ(a[3], 0.f);
}

#endif

//AVX2 tests 
#if defined(__AVX2__)

TEST(NormAvx2Test, ComputeNorm) {
    alignas(32) float a[16];
    for(int i=0;i<16;++i) a[i] = float(i+1);
    double sumsq = 0;
    for(int i=0;i<16;++i) sumsq += (i+1)*(i+1);
    float expected = std::sqrt(sumsq);
    ExpectFloatNear(ComputeNormAvx2(a,16), expected);
}

TEST(NormAvx2Test, NormalizeInplace) {
    alignas(32) float a[8] = {3,4,0,0,1,2,0,0};
    float original_norm = std::sqrt(3*3+4*4+1*1+2*2); // 3^2+4^2+1^2+2^2=30 => sqrt30
    NormalizeInplaceAvx2(a, 8);
    float scale = 1.f/original_norm;
    ExpectFloatNear(a[0],3*scale);
    ExpectFloatNear(a[1],4*scale);
    ExpectFloatNear(a[4],1*scale);
    ExpectFloatNear(a[5],2*scale);
}

#endif

//NEON tests 
#if defined(__ARM_NEON) || defined(__ARM_NEON__)

TEST(NormNeonTest, ComputeNorm) {
    float a[8] = {1,2,3,4,5,6,7,8};
    double sumsq = 0;
    for(int i=0;i<8;++i) sumsq += a[i]*a[i];
    float expected = std::sqrt(sumsq);
    ExpectFloatNear(ComputeNormNeon(a,8), expected);
}

TEST(NormNeonTest, NormalizeInplace) {
    float a[4] = {3,4,0,0};
    float original_norm = ComputeNormNeon(a, 4);
    NormalizeInplaceNeon(a, 4);
    float scale = 1.f/original_norm;
    ExpectFloatNear(a[0],3.f/original_norm);
    ExpectFloatNear(a[1],4.f/original_norm);
    EXPECT_FLOAT_EQ(a[2],0.f);
    EXPECT_FLOAT_EQ(a[3],0.f);
}

#endif

//Dispatcher tests 

TEST(NormDispatcherTest, ComputeNorm) {
    float a[8] = {1,2,3,4,5,0,0,0};
    double sumsq=0;
    for(int i=0;i<5;++i) sumsq+=a[i]*a[i];
    float expected = std::sqrt(sumsq);
    ExpectFloatNear(ComputeNorm(a,5),expected);
    ExpectFloatNear(ComputeNormPadded(a,8),std::sqrt(sumsq));
}

TEST(NormDispatcherTest, NormalizeInplace) {
    float a[6] = {3,4,0,0,1,2};
    float original_norm = std::sqrt(3*3+4*4+1+4); //3^2+4^2+1^2+2^2=30
    NormalizeInplace(a, 6);
    float scale = 1.f/original_norm;
    ExpectFloatNear(a[0],3*scale);
    ExpectFloatNear(a[1],4*scale);
    ExpectFloatNear(a[4],1*scale);
    ExpectFloatNear(a[5],2*scale);
}

TEST(NormDispatcherTest, NormalizeInplacePadded) {
    float a[8] = {3,4,0,0,1,2,0,0};
    float original_norm = std::sqrt(3*3+4*4+1*1+2*2);
    NormalizeInplacePadded(a, 6, 8);
    float scale = 1.f/original_norm;
    ExpectFloatNear(a[0],3*scale);
    ExpectFloatNear(a[1],4*scale);
    ExpectFloatNear(a[4],1*scale);
    ExpectFloatNear(a[5],2*scale);
    for(int i=6;i<8;++i) EXPECT_FLOAT_EQ(a[i],0.f);
}