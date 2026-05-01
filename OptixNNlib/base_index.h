#pragma once
#include "OptixNNlib/vector_storage.h"
#include "OptixNNlib/vector_types.h"

#include <vector>
#include <unordered_map>
#include <stdexcept>


namespace OptixNN {

class BaseIndex {
public:
    BaseIndex(dim_t dim, std::size_t max_elements,MetricType metric_type)
        : dim_(dim), storage_(dim, max_elements),metric_type_(metric_type),ntotal_(0),is_trained_(false) {}
    
    virtual ~BaseIndex() = default;

    virtual void Add(size_t n, const float_t* vecs_data) = 0;
      
    virtual void Reset() = 0;

    virtual void Train(size_t n, const float* x) = 0;

    virtual void Search(
        size_t nq,
        const float_t* queries,
        size_t k,
        float_t* distances,
        internal_id* labels
    ) const = 0;


protected:

    dim_t dim_;
    VectorStorage storage_;
    size_t ntotal_;
    bool is_trained_;
    MetricType metric_type_;
};

} // namespace OptixNN
