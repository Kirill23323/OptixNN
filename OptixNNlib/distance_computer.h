#pragma once

#include "OptixNNlib/vector_types.h"
#include "OptixNNlib/distance_l2.h"
#include "OptixNNlib/distance_cosine.h"
#include "OptixNNlib/vector_storage.h"

namespace OptixNN {

class DistanceComputer {
public:
    virtual ~DistanceComputer() = default;

    // установить query
    virtual void SetQuery(float_t* query) = 0;

    // вычислить расстояние до вектора по internal_id
    virtual float Compute(internal_id id) const = 0;

    virtual float ComputeSymmetricDist(internal_id vec_a_id,internal_id vec_b_id) const = 0;
};

class StorageDistanceComputer : public DistanceComputer {
public:
    StorageDistanceComputer(const VectorStorage& storage)
        : storage_(&storage), query_(nullptr) {}

    void SetQuery(float_t* query) override {
        query_ = query;
    }

protected:
    const VectorStorage* storage_;
    float_t* query_;
};

class L2DistanceComputer : public StorageDistanceComputer {
public:
    using StorageDistanceComputer::StorageDistanceComputer;

    float Compute(internal_id id) const override {
        const float_t* vec = storage_->GetVectorPtr(id);

        return L2Sqr(query_, vec, storage_->GetPaddedDim());
    }

    float ComputeSymmetricDist(internal_id vec_a_id,internal_id vec_b_id) const override {
        const float* vec_a = storage_->GetVectorPtr(vec_a_id);
        const float* vec_b = storage_->GetVectorPtr(vec_b_id);
        return L2Sqr(vec_a,vec_b,storage_->GetPaddedDim());

    }
};


class CosineDistanceComputer : public StorageDistanceComputer {
public:
    CosineDistanceComputer(const VectorStorage& storage)
        : StorageDistanceComputer(storage) {}

    void SetQuery(float_t* query) override {
        query_ = query;
        NormalizeInplace(query_,storage_->GetDim());
    }

    float Compute(internal_id id) const override {
        const float_t* vec = storage_->GetVectorPtr(id);
        return CosineDistanceNormed(query_,vec,storage_->GetPaddedDim());
    }

    float ComputeSymmetricDist(internal_id vec_a_id,internal_id vec_b_id) const override {
        const float* vec_a = storage_->GetVectorPtr(vec_a_id);
        const float* vec_b = storage_->GetVectorPtr(vec_b_id);
        return CosineDistanceNormed(vec_a,vec_b,storage_->GetPaddedDim());
    }
};



} // namespace OptixNN