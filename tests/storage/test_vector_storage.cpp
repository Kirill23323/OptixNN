#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <chrono>

#include "OptixNNlib/vector_storage.h"

using namespace OptixNN;


static float Norm(const float* v, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += v[i] * v[i];
    }
    return std::sqrt(sum);
}


static void ExpectNormalized(const float* v, size_t dim, float eps = 1e-4f) {
    float norm = Norm(v, dim);
    EXPECT_NEAR(norm, 1.0f, eps);
}

//basic

TEST(VectorStorage, BasicAddAndGet_Normalized) {
    VectorStorage storage(3, 10);

    float vec[3] = {1.0f, 2.0f, 3.0f};
    storage.AddVector(0, vec);

    ASSERT_TRUE(storage.HasId(0));

    const float* stored = storage.GetVectorPtr(0);

    ExpectNormalized(stored, 3);

    // проверяем пропорции сохраняются
    EXPECT_GT(stored[1], stored[0]);
    EXPECT_GT(stored[2], stored[1]);
}

TEST(VectorStorage, PaddingWorks_Normalized) {
    VectorStorage storage(3, 1, 4);

    float vec[3] = {1, 2, 3};
    storage.AddVector(0, vec);

    const float* stored = storage.GetVectorPtr(0);

    EXPECT_EQ(storage.GetPaddedDim(), 4);
    ExpectNormalized(stored, 3);

    EXPECT_FLOAT_EQ(stored[3], 0.0f);
}

TEST(VectorStorage, OutOfRangeId) {
    VectorStorage storage(3, 2);

    float vec[3] = {1, 2, 3};

    EXPECT_THROW(storage.AddVector(5, vec), std::out_of_range);
}

TEST(VectorStorage, GetInvalidId) {
    VectorStorage storage(3, 2);

    EXPECT_THROW(storage.GetVectorPtr(0), std::out_of_range);
}

//batch

TEST(VectorStorage, BatchInsert_Normalized) {
    VectorStorage storage(3, 5);

    float data[6] = {
        1,2,3,
        4,5,6
    };

    storage.AddBatch(0, data, 2);

    EXPECT_TRUE(storage.HasId(0));
    EXPECT_TRUE(storage.HasId(1));

    const float* v0 = storage.GetVectorPtr(0);
    const float* v1 = storage.GetVectorPtr(1);

    ExpectNormalized(v0, 3);
    ExpectNormalized(v1, 3);
}

TEST(VectorStorage, BatchInvalidRange) {
    VectorStorage storage(3, 2);

    float data[6] = {1,2,3,4,5,6};

    EXPECT_THROW(storage.AddBatch(1, data, 5), std::out_of_range);
}


//large batch

TEST(VectorStorage, BatchInsertLargeNumberOfVectors_Normalized) {
    VectorStorage storage(4, 1000);

    const size_t n = 100;
    const size_t dim = 4;
    std::vector<float> data(n * dim);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            data[i * dim + j] = float(i * 10 + j);
        }
    }

    storage.AddBatch(0, data.data(), n);

    for (size_t i = 0; i < n; ++i) {
        ASSERT_TRUE(storage.HasId(i));
        const float* v = storage.GetVectorPtr(i);

        ExpectNormalized(v, dim);
    }
}

TEST(VectorStorage, BatchInsertConcurrentAccess) {
    VectorStorage storage(4, 100);

    const size_t n = 10;
    const size_t dim = 4;

    std::vector<float> b1(n * dim), b2(n * dim);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            b1[i * dim + j] = float(i * 10 + j);
            b2[i * dim + j] = float((i + n) * 10 + j);
        }
    }

    storage.AddBatch(0, b1.data(), n);
    storage.AddBatch(n, b2.data(), n);

    for (size_t i = 0; i < 2 * n; ++i) {
        ASSERT_TRUE(storage.HasId(i));
        const float* v = storage.GetVectorPtr(i);
        ExpectNormalized(v, dim);
    }
}

TEST(VectorStorage, BatchInsertWithPadding) {
    VectorStorage storage(3, 50, 4);

    const size_t n = 10;
    std::vector<float> data(n * 3);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            data[i * 3 + j] = float(i * 10 + j);
        }
    }

    storage.AddBatch(0, data.data(), n);

    for (size_t i = 0; i < n; ++i) {
        const float* v = storage.GetVectorPtr(i);

        ExpectNormalized(v, 3);

        for (size_t j = 3; j < storage.GetPaddedDim(); ++j) {
            EXPECT_FLOAT_EQ(v[j], 0.0f);
        }
    }
}

TEST(VectorStorage, BatchInsertSingleVector) {
    VectorStorage storage(4, 10);

    float v[4] = {1,2,3,4};

    storage.AddBatch(5, v, 1);

    const float* out = storage.GetVectorPtr(5);

    ASSERT_TRUE(storage.HasId(5));
    ExpectNormalized(out, 4);
}

TEST(VectorStorage, BatchInsertZeroSize) {
    VectorStorage storage(4, 10);

    float data[8] = {1,2,3,4,5,6,7,8};

    storage.AddBatch(0, data, 0);

    EXPECT_FALSE(storage.HasId(0));
}