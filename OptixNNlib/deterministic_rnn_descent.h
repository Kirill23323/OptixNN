#pragma once

#include "OptixNNlib/neighbor.h"
#include "OptixNNlib/vector_types.h"
#include "OptixNNlib/visited_table.h"
#include "OptixNNlib/distance_computer.h"

#include <vector>
#include <algorithm>
#include <mutex>

namespace OptixNN {

struct DeterministicRNNDescent {

    using KNNGraph = std::vector<Nhood>;

    explicit DeterministicRNNDescent(int d) : d(d) {}

    ~DeterministicRNNDescent() = default;

  
    void Build(DistanceComputer& qdis, int n, bool /*verbose*/) {

        ntotal = n;
        graph.clear();
        graph.resize(n);

        // deterministic init: i -> (i+1)%n, (i+2)%n ...
        for (int i = 0; i < n; ++i) {
            for (int j = 1; j <= S; ++j) {
                int id = (i + j) % n;
                float dist = qdis.ComputeSymmetricDist(i, id);
                graph[i].pool.emplace_back(id, dist, true);
            }
        }

        has_built = true;
    }

 
    void Search(DistanceComputer& qdis,
                int topk,
                internal_id* indices,
                float* dists,
                VisitedTable& vt) const {

        std::vector<Neighbor> candidates;

        for (int i = 0; i < ntotal; ++i) {
            float d = qdis.Compute(i);
            candidates.emplace_back(i, d, true);
        }

        std::sort(candidates.begin(), candidates.end());

        for (int i = 0; i < topk; ++i) {
            indices[i] = candidates[i].id;
            dists[i] = candidates[i].distance;
        }

        vt.Advance();
    }

    void Reset() {
        has_built = false;
        ntotal = 0;
        graph.clear();
    }

    int InsertIntoPool(Neighbor* addr, int size, Neighbor n) const {
        addr[size] = n;
        return size;
    }

    bool has_built = false;

    int S = 8;
    int d = 0;
    int ntotal = 0;

    KNNGraph graph;
};

} // namespace OptixNN