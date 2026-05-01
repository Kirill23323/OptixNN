#pragma once
#include "OptixNNlib/vector_storage.h"
#include "OptixNNlib/vector_types.h"
#include "OptixNNlib/base_index.h"
#include "OptixNNlib/interrupt_callback.h"
#include "OptixNNlib/rnn_descent.h"
#include "OptixNNlib/visited_table.h"
#include "OptixNNlib/distance_computer.h"

#include <vector>
#include <unordered_map>
#include <stdexcept>


namespace OptixNN {

class RNNDescentFlatIndex : public BaseIndex {
public:

    RNNDescentFlatIndex(dim_t dim, std::size_t max_elements,MetricType metric_type)
        : BaseIndex(dim, max_elements, metric_type), rnndescent_(dim) {}

    void Add(size_t n, const float_t* vecs_data) override{
        storage_.AddBatch(0,vecs_data,n);
        ntotal_ = n;
    }
      
    void Reset() override {
        ntotal_ = 0;
    }

    void Train(size_t n, const float* x) override{
        // since it's a flat index we don't do anything
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

    size_t check_period =
        InterruptCallback::GetPeriodHint(k * dim_);

    for (size_t i0 = 0; i0 < nq; i0 += check_period) {

        size_t i1 = std::min(i0 + check_period, nq);

#pragma omp parallel
        {
            VisitedTable vt(ntotal_);
            std::unique_ptr<DistanceComputer> qdis;

            // создаём ОДИН раз на thread
#pragma omp for schedule(dynamic, 1)
            for (size_t i = i0; i < i1; ++i) {

                if (!qdis) {
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
                }

                qdis->SetQuery(const_cast<float_t*>(queries + i * dim_));

                rnndescent_.Search(*qdis, k,
                                   labels + i * k,
                                   distances + i * k,
                                   vt);
            }
        }

        InterruptCallback::Check();
    }
}

protected:
    RNNDescent rnndescent_;

};

} // namespace OptixNN
