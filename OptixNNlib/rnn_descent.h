#pragma once

#include "OptixNNlib/base_index.h"
#include "OptixNNlib/distance_computer.h"
#include "OptixNNlib/neighbor.h"
#include "OptixNNlib/vector_types.h"
#include "OptixNNlib/visited_table.h"

#include <vector>
#include <omp.h>
#include <algorithm>
#include <random>

namespace OptixNN {

struct RNNDescent {

  using KNNGraph = std::vector<Nhood>;

  explicit RNNDescent(const int d) : d(d) {}

  ~RNNDescent() = default;

  void Build(DistanceComputer &qdis, const int n, bool verbose) {
    ntotal = n;
    GenerateRandomGraph(qdis);
    for (int t1 = 0; t1 < T1; ++t1) {
      for (int t2 = 0; t2 < T2; ++t2) {
        UpdateNeighbors(qdis);
      }
      if (t1 != T1 - 1) {
        AddReverseEdges();
      }
    }
#pragma omp parallel for
    for (int u = 0; u < n; u++) {
      auto &pool = graph[u].pool;
      std::sort(pool.begin(), pool.end());
      pool.erase(
          std::unique(pool.begin(), pool.end(),
                      [](Neighbor &a, Neighbor &b) { return a.id == b.id; }),
          pool.end());
    }
    offsets.resize(ntotal + 1);
    offsets[0] = 0;
    for (int u = 0; u < ntotal; ++u) {
      offsets[u + 1] = offsets[u] + graph[u].pool.size();
    }
    final_graph.resize(offsets.back(), -1);
#pragma omp parallel for
    for (int u = 0; u < n; ++u) {
      auto &pool = graph[u].pool;
      int offset = offsets[u];
      for (int i = 0; i < pool.size(); ++i) {
        final_graph[offset + i] = pool[i].id;
      }
    }
    std::vector<Nhood>().swap(graph);
    has_built = true;
  }

  int InsertIntoPool(Neighbor *addr, int size,
                     Neighbor neartest_neighbor) const {
    int left = 0;
    int right = size - 1;
    if (addr[left].distance > neartest_neighbor.distance) {
      memmove(&addr[left + 1],
        &addr[left],
        (size - left) * sizeof(Neighbor));
      addr[left] = neartest_neighbor;
      return left;
    }
    if (addr[right].distance < neartest_neighbor.distance) {
      addr[size] = neartest_neighbor;
      return size;
    }
    while (left < right - 1) {
      int mid = (left + right) / 2;
      if (addr[mid].distance > neartest_neighbor.distance) {
        right = mid;
      } else {
        left = mid;
      }
    }
    while (left > 0) {
      if (addr[left].distance < neartest_neighbor.distance) {
        break;
      }
      if (addr[left].id == neartest_neighbor.id) {
        return size + 1;
      }
      left--;
    }
    if (addr[left].id == neartest_neighbor.id ||
        addr[right].id == neartest_neighbor.id) {
      memmove((char *)&addr[right + 1], &addr[right],
              (size - right) * sizeof(Neighbor));
    }
    addr[right] = neartest_neighbor;
    return right;
  }

  void Search(DistanceComputer &qdis, const int topk, internal_id *indices,
              float *dists, VisitedTable &vt) const {

    int L = std::max(search_L, topk);
    // candidate pool, the K best items is the result.
    std::vector<Neighbor> retset(L + 1);
    // Randomly choose L points to initialize the candidate pool
    std::vector<int> init_ids(L);
    std::mt19937 rng(random_seed);
    FillAnArrayWithRandomNums(rng, init_ids.data(), L, ntotal);
    for (int i = 0; i < L; i++) {
      int id = init_ids[i];
      float dist = qdis.Compute(id);
      retset[i] = Neighbor(id, dist, true);
    }
    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < L) {
      int nk = L;
      if (retset[k].flag) {
        retset[k].flag = false;
        int n = retset[k].id;
        int offset = offsets[n];
        int K = std::min(K0, offsets[n + 1] - offset);
        for (int m = 0; m < K; ++m) {
          int id = final_graph[offset + m];
          if (vt.Get(id)) {
            continue;
          }
          vt.Set(id);
          float dist = qdis.Compute(id);
          if (dist >= retset[L - 1].distance) {
            continue;
          }
          Neighbor nearest_neighbor(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nearest_neighbor);
          if (r < nk) {
            nk = r;
          }
        }
        if (nk <= k) {
          k = nk;
        } else {
          ++k;
        }
      }
    }
    for (size_t i = 0; i < topk; i++) {
        indices[i] = retset[i].id;
        dists[i] = retset[i].distance;
      }
    vt.Advance();
  }

  void Reset() {
    has_built = false;
    ntotal = 0;
    final_graph.resize(0);
    offsets.resize(0);
  }

  /// Initialize the KNN graph randomly
  void GenerateRandomGraph(DistanceComputer &qdis) {
    graph.reserve(ntotal);
    std::mt19937 rng(random_seed * 7777);
    for (int i = 0; i < ntotal; ++i) {
      graph.push_back(Nhood(S, rng, ntotal));
    }
#pragma omp parallel
    {
      std::mt19937 rng(random_seed * 7741 + omp_get_thread_num());
#pragma omp for
      for (int i = 0; i < ntotal; ++i) {
        std::vector<int> tmp(S);
        FillAnArrayWithRandomNums(rng, tmp.data(), S, ntotal);
        for (int j = 0; j < S; ++j) {
          int id = tmp[j];
          if (id == i) {
            continue;
          }
          float dist = qdis.ComputeSymmetricDist(i, id);
          graph[i].pool.push_back(Neighbor(id, dist, true));
        }
        std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
        graph[i].pool.reserve(L);
      }
    }
  }

  void UpdateNeighbors(DistanceComputer &qdis) {
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < ntotal; ++u) {
      auto &nhood = graph[u];
      auto &pool = nhood.pool;
      std::vector<Neighbor> new_pool;
      std::vector<Neighbor> old_pool;
      {
        std::lock_guard<std::mutex> guard(nhood.lock);
        old_pool = pool;
        pool.clear();
      }
      std::sort(old_pool.begin(), old_pool.end());
      old_pool.erase(
          std::unique(old_pool.begin(), old_pool.end(),
                      [](Neighbor &a, Neighbor &b) { return a.id == b.id; }),
          old_pool.end());
      for (auto &&nearest_neighbor : old_pool) {
        bool ok = true;
        for (auto &&other_nearest_neighbor : new_pool) {
          if (!nearest_neighbor.flag && !other_nearest_neighbor.flag) {
            continue;
          }
          if (nearest_neighbor.id == other_nearest_neighbor.id) {
            ok = false;
            break;
          }
          float distance = qdis.ComputeSymmetricDist(other_nearest_neighbor.id,
                                                     nearest_neighbor.id);
          if (distance < nearest_neighbor.distance) {
            ok = false;
            InsertNearestNeighbor(other_nearest_neighbor.id,
                                  nearest_neighbor.id, distance, true);
            break;
          }
        }
        if (ok) {
          new_pool.emplace_back(nearest_neighbor);
        }
      }
      for (auto &&nearest_neighbor : new_pool) {
        nearest_neighbor.flag = false;
      }
      {
        std::lock_guard<std::mutex> guard(nhood.lock);
        pool.insert(pool.end(), new_pool.begin(), new_pool.end());
      }
    }
  }

  void AddReverseEdges() {
    std::vector<std::vector<Neighbor>> reverse_pools(ntotal);
#pragma omp parallel for
    for (int u = 0; u < ntotal; ++u) {
      for (auto &&nearest_neighbor : graph[u].pool) {
        std::lock_guard<std::mutex> guard(graph[nearest_neighbor.id].lock);
        reverse_pools[nearest_neighbor.id].emplace_back(
            u, nearest_neighbor.distance, nearest_neighbor.flag);
      }
    }
#pragma omp parallel for
    for (int u = 0; u < ntotal; ++u) {
      auto &pool = graph[u].pool;
      for (auto &&nearest_neighbor : pool) {
        nearest_neighbor.flag = true;
      }
      auto &rpool = reverse_pools[u];
      rpool.insert(rpool.end(), pool.begin(), pool.end());
      pool.clear();
      std::sort(rpool.begin(), rpool.end());
      rpool.erase(
          std::unique(rpool.begin(), rpool.end(),
                      [](Neighbor &a, Neighbor &b) { return a.id == b.id; }),
          rpool.end());
      if (rpool.size() > R) {
        rpool.resize(R);
      }
    }
#pragma omp parallel for
    for (int u = 0; u < ntotal; ++u) {
      for (auto &&nearest_neighbor : reverse_pools[u]) {
        std::lock_guard<std::mutex> guard(graph[nearest_neighbor.id].lock);
        graph[nearest_neighbor.id].pool.emplace_back(
            u, nearest_neighbor.distance, nearest_neighbor.flag);
      }
    }
#pragma omp parallel for
    for (int u = 0; u < ntotal; ++u) {
      auto &pool = graph[u].pool;
      std::sort(pool.begin(), pool.end());
      if (pool.size() > R) {
        pool.resize(R);
      }
    }
  }

  void InsertNearestNeighbor(int id, int nn_id, float distance, bool flag) {
    auto &nhood = graph[id];
    {
      std::lock_guard<std::mutex> guard(nhood.lock);
      nhood.pool.emplace_back(nn_id, distance, flag);
    }
  }

  bool has_built = false;

  int T1 = 4;
  int T2 = 15;
  int S = 16;
  int R = 96;
  int K0 = 32; // maximum out-degree

  int search_L = 0;       // size of candidate pool in searching
  int random_seed = 2026; // random seed for generators

  int d;     // dimensions
  int L = 8; // initial size of memory allocation

  int ntotal = 0;

  KNNGraph graph;
  std::vector<int> final_graph;
  std::vector<int> offsets;
};
} // namespace OptixNN