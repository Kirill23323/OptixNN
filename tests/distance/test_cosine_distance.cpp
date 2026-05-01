#include <gtest/gtest.h>
#include "OptixNNlib/distance_cosine.h"

using namespace OptixNN;

TEST(CosineDistance, NormedVectorsSame) {
    float a[4] = {1, 0, 0, 0};
    float b[4] = {1, 0, 0, 0};

    float dist = CosineDistanceNormed(a, b, 4);

    EXPECT_NEAR(dist, 0.0f, 1e-6);
}

TEST(CosineDistance, OrthogonalVectors) {
    float a[4] = {1, 0, 0, 0};
    float b[4] = {0, 1, 0, 0};

    float dist = CosineDistanceNormed(a, b, 4);

    EXPECT_NEAR(dist, 1.0f, 1e-6);
}

TEST(CosineDistance, OppositeVectors) {
    float a[4] = {1, 0, 0, 0};
    float b[4] = {-1, 0, 0, 0};

    float dist = CosineDistanceNormed(a, b, 4);

    EXPECT_NEAR(dist, 2.0f, 1e-6);
}

TEST(CosineDistance, WithNorms) {
    float a[4] = {3, 0, 0, 0};
    float b[4] = {0, 4, 0, 0};

    float dist = CosineDistanceWithNorms(a, b, 3.0f, 4.0f, 4);

    EXPECT_NEAR(dist, 1.0f, 1e-6);
}

TEST(CosineDistance, AutoVersion) {
    float a[4] = {1, 0, 0, 0};
    float b[4] = {1, 0, 0, 0};

    float dist = cosine_distance_auto(a, b, 4, 4);

    EXPECT_NEAR(dist, 0.0f, 1e-6);
}

TEST(CosineDistance, ZeroVectorStability) {
    float a[4] = {0, 0, 0, 0};
    float b[4] = {0, 0, 0, 0};

    float dist = cosine_distance_auto(a, b, 4, 4);

    EXPECT_TRUE(std::isfinite(dist));
}