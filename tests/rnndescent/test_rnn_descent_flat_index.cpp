#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cmath>

#include "OptixNNlib/vector_storage.h"
#include "OptixNNlib/vector_types.h"
#include "OptixNNlib/distance_computer.h"
#include "OptixNNlib/visited_table.h"

#include "OptixNNlib/deterministic_rnn_descent_flat_index.h"

using namespace OptixNN;

static std::vector<std::pair<int,float>> BruteForceL2(
    const std::vector<float>& data,
    const std::vector<float>& query,
    int n,
    int dim)
{
    std::vector<std::pair<int,float>> res;

    auto dist = [&](int i) {
        float s = 0;
        for (int d = 0; d < dim; ++d) {
            float diff = data[i * dim + d] - query[d];
            s += diff * diff;
        }
        return s;
    };

    for (int i = 0; i < n; ++i) {
        res.emplace_back(i, dist(i));
    }

    std::sort(res.begin(), res.end(),
              [](auto& a, auto& b) {
                  return a.second < b.second;
              });

    return res;
}

TEST(DetFlatIndex, EmptyThrows) {
    DeterministicRNNDescentFlatIndex index(2, 10, MetricType::L2);

    std::vector<float> query = {0,0};

    std::vector<float> dist(1);
    std::vector<internal_id> idx(1);

    EXPECT_THROW(
        index.Search(1, query.data(), 1, dist.data(), idx.data()),
        std::runtime_error
    );
}

TEST(DetFlatIndex, BasicCorrectness) {
    int n = 5;
    int dim = 2;

    std::vector<float> data = {
        0,0,
        1,1,
        2,2,
        3,3,
        4,4
    };

    VectorStorage storage(dim, n);
    storage.AddBatch(0, data.data(), n);

    DeterministicRNNDescentFlatIndex index(dim, n, MetricType::L2);
    index.Add(n, data.data());

    std::vector<float> query = {0.1f, 0.1f};

    std::vector<float> dist(3);
    std::vector<internal_id> idx(3);

    index.Search(1, query.data(), 3, dist.data(), idx.data());

    EXPECT_EQ(idx[0], 0);
}

TEST(DetFlatIndex, DeterministicOutput) {
    int n = 4;
    int dim = 2;

    std::vector<float> data = {
        1,0,
        0,1,
        2,2,
        3,3
    };

    VectorStorage storage(dim, n);
    storage.AddBatch(0, data.data(), n);

    DeterministicRNNDescentFlatIndex index(dim, n, MetricType::L2);
    index.Add(n, data.data());

    std::vector<float> query = {0,0};

    std::vector<float> d1(2), d2(2);
    std::vector<internal_id> i1(2), i2(2);

    index.Search(1, query.data(), 2, d1.data(), i1.data());
    index.Search(1, query.data(), 2, d2.data(), i2.data());

    EXPECT_EQ(i1, i2);
    EXPECT_EQ(d1, d2);
}

TEST(DetFlatIndex, MatchesBruteForce) {
    int n = 6;
    int dim = 2;

    std::vector<float> data = {
        0,0,
        1,1,
        2,2,
        3,3,
        4,4,
        5,5
    };

    VectorStorage storage(dim, n);
    storage.AddBatch(0, data.data(), n);

    DeterministicRNNDescentFlatIndex index(dim, n, MetricType::L2);
    index.Add(n, data.data());

    std::vector<float> query = {1.1f, 1.1f};

    std::vector<float> dist(3);
    std::vector<internal_id> idx(3);

    index.Search(1, query.data(), 3, dist.data(), idx.data());

    auto gt = BruteForceL2(data, query, n, dim);

    EXPECT_EQ(idx[0], gt[0].first);
}