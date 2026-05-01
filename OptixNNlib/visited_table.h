#pragma once
#include <stdint.h>
#include <unordered_set>
#include <vector>
#include <cstring>

#include "OptixNNlib/prefetch.h"

namespace OptixNN {

    /// A fast, reusable Visited Set for graph search algorithms.
    struct VisitedTable {

        const size_t threshold_size_for_hashset_mode = 500000;
        std::vector<uint8_t> visited;
        std::unordered_set<size_t> visited_set;
        uint8_t visno; 

 
        explicit VisitedTable(size_t size) : visno((size >= threshold_size_for_hashset_mode) ? 0 : 1) {

            if(visno != 0) {
                visited.resize(size, 0);
            }

        }

        /// set flag #no to true, return whether this changed it.
        bool Set(size_t no) {
            if (visno == 0) {
                return visited_set.insert(no).second;
            } else if (visited[no] == visno) {
                return false;
            } else {
                visited[no] = visno;
                return true;
            }
        }

        /// get flag #no
        bool Get(size_t no) const {
            if (visno == 0) {
                return visited_set.count(no) != 0;
            } else {
                return visited[no] == visno;
            }
        }

        void Prefetch(size_t no) const {
            if (visno != 0) {
                PrefetchL2(&visited[no]);
            }
        }

        /// reset all flags to false
        void Advance() {

            if(visno == 0) {
                visited_set.clear();
            }
            else if (visno < 254) {
                ++visno;
            }
            else {
                memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
                visno = 1;
            }

        }
    };
} // namespace OptixNN