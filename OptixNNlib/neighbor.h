#pragma once
#include "OptixNNlib/vector_storage.h"
#include "OptixNNlib/vector_types.h"

#include <mutex>
#include <random>
#include <algorithm>

namespace OptixNN{

    using LockGuard = std::lock_guard<std::mutex>;

    struct Neighbor {

        Neighbor() = default;
        Neighbor(int id,float distance,bool flag) :
        id(id),distance(distance),flag(flag){}

        inline bool operator < (const Neighbor& other){
            return distance < other.distance;
        }

        int id;
        float distance;
        bool flag;
    };

    inline void FillAnArrayWithRandomNums(std::mt19937& rng, int* addr, const int size, const int N){

        for (int i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);
        for (int i = 1; i < size; ++i) {
            if (addr[i] <= addr[i - 1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
        int off = rng() % N;
        for (int i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }
    struct Nhood{


        std::mutex lock;
        std::vector<Neighbor> pool; // candidate pool (a max heap)
        int M;                      // number of new neighbors to be operated

        std::vector<int> nn_old;  // old neighbors
        std::vector<int> nn_new;  // new neighbors
        std::vector<int> rnn_old; // reverse old neighbors
        std::vector<int> rnn_new; // reverse new neighbors

        Nhood() = default;

        Nhood(int s, std::mt19937& rng, int N){
            M = s;
            nn_new.resize(s * 2);
            FillAnArrayWithRandomNums(rng, nn_new.data(), (int)nn_new.size(), N);
        }

        Nhood& operator=(const Nhood& other){
            M = other.M;
            std::copy(
                    other.nn_new.begin(),
                    other.nn_new.end(),
                    std::back_inserter(nn_new));
            nn_new.reserve(other.nn_new.capacity());
            pool.reserve(other.pool.capacity());
            return *this;
        }

        Nhood(const Nhood& other){
            M = other.M;
            std::copy(
                    other.nn_new.begin(),
                    other.nn_new.end(),
                    std::back_inserter(nn_new));
            nn_new.reserve(other.nn_new.capacity());
            pool.reserve(other.pool.capacity());
            
        }

        void Insert(int id, float dist){
            LockGuard guard(lock);
            if (dist > pool.front().distance) {
                return;
            }
            for (size_t i = 0; i < pool.size(); i++) {
                if (id == pool[i].id) {
                    return;
                }
            }
            if (pool.size() < pool.capacity()) {
                pool.push_back(Neighbor(id, dist, true));
                std::push_heap(pool.begin(), pool.end());
            } else {
                std::pop_heap(pool.begin(), pool.end());
                pool[pool.size() - 1] = Neighbor(id, dist, true);
                std::push_heap(pool.begin(), pool.end());
            }
            
        }
    };
}