#include <gtest/gtest.h>
#include "OptixNNlib/distance_computer.h"
#include "OptixNNlib/vector_types.h"
#include "OptixNNlib/vector_storage.h"

using namespace OptixNN;

TEST(DistanceComputer, L2DistanceComputer) {
    // Create a vector storage for testing
    dim_t dim = 4;
    VectorStorage storage(dim, 10); // storage for up to 10 vectors
    
    // Add test vectors
    float vec1_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vec2_data[] = {2.0f, 3.0f, 4.0f, 5.0f};
    
    storage.AddVector(0, vec1_data);
    storage.AddVector(1, vec2_data);

    // Test L2 distance using L2DistanceComputer
    L2DistanceComputer l2_comp(storage);
    
    float query[] = {1.0f, 2.0f, 3.0f, 4.0f};
    l2_comp.SetQuery(query);
    
    float dist = l2_comp.Compute(1); // distance to vec2: {2.0f, 3.0f, 4.0f, 5.0f};
    
    // Expected: (1-2)^2 + (2-3)^2 + (3-4)^2 + (4-5)^2 = 1 + 1 + 1 + 1 = 4
    EXPECT_FLOAT_EQ(dist, 4.0f);
}


TEST(DistanceComputer, CosineDistanceComputerSameDirection) {
    // Create a vector storage for testing
    dim_t dim = 4;
    VectorStorage storage(dim, 10); // storage for up to 10 vectors
    
    // Add test vectors
    float vec1_data[] = {1.0f, 0.0f, 0.0f, 0.0f};  // unit vector
    float vec2_data[] = {1.0f, 0.0f, 0.0f, 0.0f};  // same direction
    
    storage.AddVector(0, vec1_data);
    storage.AddVector(1, vec2_data);

    // Test Cosine distance using CosineDistanceComputer
    CosineDistanceComputer cos_comp(storage);
    
    float query[] = {1.0f, 0.0f, 0.0f, 0.0f};  // unit vector
    cos_comp.SetQuery(query);
    
    float dist = cos_comp.Compute(1); // distance to vec2: {1.0f, 0.0f, 0.0f, 0.0f}
    
    // Expected: 1 - cosine_similarity = 1 - 1 = 0 (for same direction vectors)
    EXPECT_NEAR(dist, 0.0f, 1e-6);
}

TEST(DistanceComputer, CosineDistanceComputerDifferentVectors) {
    // Create a vector storage for testing
    dim_t dim = 4;
    VectorStorage storage(dim, 10); // storage for up to 10 vectors
    
    // Add test vectors
    float vec1_data[] = {1.0f, 0.0f, 0.0f, 0.0f};  // x-axis unit vector
    float vec2_data[] = {1.0f, 2.0f, 3.0f, 4.0f};  // different vector
    
    storage.AddVector(0, vec1_data);
    storage.AddVector(1, vec2_data);

    // Test Cosine distance using CosineDistanceComputer
    CosineDistanceComputer cos_comp(storage);
    
    float query[] = {1.0f, 0.0f, 0.0f, 0.0f};  // x-axis unit vector
    cos_comp.SetQuery(query);
    
    float dist = cos_comp.Compute(1); // distance to vec2: {1.0f, 2.0f, 3.0f, 4.0f}
    
    // This will be some positive value depending on the angle between vectors
    EXPECT_GE(dist, 0.0f); // Cosine distance should be non-negative
}

TEST(DistanceComputer, MultipleComputations) {
    // Create a vector storage for testing
    dim_t dim = 4;
    VectorStorage storage(dim, 10); // storage for up to 10 vectors
    
    // Add test vectors
    float vec1_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float vec2_data[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float vec3_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    storage.AddVector(0, vec1_data);
    storage.AddVector(1, vec2_data);
    storage.AddVector(2, vec3_data);

    // Test L2 distance using L2DistanceComputer
    L2DistanceComputer l2_comp(storage);
    
    float query[] = {0.0f, 0.0f, 0.0f, 0.0f};
    l2_comp.SetQuery(query);
    
    // Compute distances to multiple vectors
    float dist1 = l2_comp.Compute(1); // distance to {1.0f, 0.0f, 0.0f, 0.0f}
    float dist2 = l2_comp.Compute(2); // distance to {1.0f, 2.0f, 3.0f, 4.0f}
    
    EXPECT_FLOAT_EQ(dist1, 1.0f); // distance squared from origin to (1,0,0,0) = 1^2 = 1
    EXPECT_GT(dist2, 1.0f);       // should be larger than 1
}