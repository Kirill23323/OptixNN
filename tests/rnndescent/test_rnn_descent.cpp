#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cmath>

#include "OptixNNlib/vector_storage.h"
#include "OptixNNlib/vector_types.h"
#include "OptixNNlib/distance_computer.h"
#include "OptixNNlib/visited_table.h"

#include "OptixNNlib/deterministic_rnn_descent.h"

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

TEST(DetRNNDescent, BuildCreatesValidGraph) {
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

    L2DistanceComputer qdis(storage);

    DeterministicRNNDescent index(dim);
    index.Build(qdis, n, false);

    EXPECT_TRUE(index.has_built);
    EXPECT_EQ(index.ntotal, n);
    EXPECT_FALSE(index.graph.empty());
}

TEST(DetRNNDescent, Top1Correctness) {
    int n = 5;
    int dim = 2;

    std::vector<float> data = {
        10,10,
        0,0,
        5,5,
        1,1,
        2,2
    };

    VectorStorage storage(dim, n);
    storage.AddBatch(0, data.data(), n);

    L2DistanceComputer qdis(storage);

    DeterministicRNNDescent index(dim);
    index.Build(qdis, n, false);

    std::vector<float> query = {0,0};

    VisitedTable vt(n);

    std::vector<internal_id> idx(1);
    std::vector<float> dist(1);

    qdis.SetQuery(query.data());

    index.Search(qdis, 1, idx.data(), dist.data(), vt);

    EXPECT_EQ(idx[0], 1);
}

TEST(DetRNNDescent, DeterministicOutput) {
    int n = 4;
    int dim = 2;

    std::vector<float> data = {
        1,1,
        2,2,
        3,3,
        4,4
    };

    VectorStorage storage(dim, n);
    storage.AddBatch(0, data.data(), n);

    L2DistanceComputer qdis(storage);

    DeterministicRNNDescent index(dim);
    index.Build(qdis, n, false);

    std::vector<float> query = {1.2f, 1.2f};

    std::vector<internal_id> idx1(2), idx2(2);
    std::vector<float> d1(2), d2(2);

    {
        VisitedTable vt(n);
        qdis.SetQuery(query.data());
        index.Search(qdis, 2, idx1.data(), d1.data(), vt);
    }

    {
        VisitedTable vt(n);
        qdis.SetQuery(query.data());
        index.Search(qdis, 2, idx2.data(), d2.data(), vt);
    }

    EXPECT_EQ(idx1, idx2);
    EXPECT_EQ(d1, d2);
}


TEST(DetRNNDescent, MatchesBruteforceTopK) {
    int n = 8;
    int dim = 3;

    std::vector<float> data = {
        0,0,0,
        1,0,0,
        0,1,0,
        0,0,1,
        1,1,1,
        2,2,2,
        3,3,3,
        4,4,4
    };

    VectorStorage storage(dim, n);
    storage.AddBatch(0, data.data(), n);

    L2DistanceComputer qdis(storage);

    DeterministicRNNDescent index(dim);
    index.Build(qdis, n, false);

    std::vector<float> query = {0,0,0};

    VisitedTable vt(n);

    std::vector<internal_id> idx(3);
    std::vector<float> dist(3);

    qdis.SetQuery(query.data());

    index.Search(qdis, 3, idx.data(), dist.data(), vt);

    auto gt = BruteForceL2(data, query, n, dim);

    EXPECT_EQ(idx[0], gt[0].first);
    EXPECT_EQ(idx[1], gt[1].first);
}