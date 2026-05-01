#pragma once

#include "OptixNNlib/vector_storage.h"
#include "OptixNNlib/vector_types.h"
#include "OptixNNlib/base_index.h"
#include "OptixNNlib/interrupt_callback.h"
#include "OptixNNlib/rnn_descent.h"
#include "OptixNNlib/visited_table.h"
#include "OptixNNlib/distance_computer.h"

#include <vector>
#include <memory>
#include <stdexcept>

namespace OptixNN {

class DeterministicRNNDescentFlatIndex : public BaseIndex {
public:

    DeterministicRNNDescentFlatIndex(dim_t dim,
                                     std::size_t max_elements,
                                     MetricType metric_type)
        : BaseIndex(dim, max_elements, metric_type),
          rnndescent_(dim) {}

    void Add(size_t n, const float_t* vecs_data) override {
        storage_.AddBatch(0, vecs_data, n);
        ntotal_ = n;
    }

    void Reset() override {
        ntotal_ = 0;
    }

    void Train(size_t n, const float* x) override {
        is_trained_ = true;
    }

    void Search(
        size_t nq,
        const float_t* queries,
        size_t k,
        float_t* distances,
        internal_id* labels
    ) const override {

        if (ntotal_ == 0) {
            throw std::runtime_error("Index is empty");
        }

        // фиксированный период = детерминированность
        size_t check_period = 1024;

        for (size_t i = 0; i < nq; i += check_period) {

            size_t i1 = std::min(i + check_period, nq);

            for (size_t q = i; q < i1; ++q) {

                std::unique_ptr<DistanceComputer> qdis;

                switch (metric_type_) {
                    case MetricType::L2:
                        qdis = std::make_unique<L2DistanceComputer>(storage_);
                        break;
                    case MetricType::COSINE:
                        qdis = std::make_unique<CosineDistanceComputer>(storage_);
                        break;
                    default:
                        throw std::runtime_error("Unsupported metric");
                }

                qdis->SetQuery(const_cast<float_t*>(queries + q * dim_));

                VisitedTable vt(ntotal_);

                rnndescent_.Search(
                    *qdis,
                    k,
                    labels + q * k,
                    distances + q * k,
                    vt
                );
            }

            InterruptCallback::Check();
        }
    }

protected:
    RNNDescent rnndescent_;
};

} // namespace OptixNN