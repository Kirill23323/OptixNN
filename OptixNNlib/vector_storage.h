#pragma once

#include "OptixNNlib/vector_types.h"
#include "OptixNNlib/aligned_alloc.h"
#include "OptixNNlib/vector_utils.h"
#include "OptixNNlib/norm_backend.h"


#include <iostream>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <omp.h>

namespace OptixNN {

    inline dim_t DetectSimdWidth() noexcept {
    #if defined(__AVX512F__)
        std::cout << "AVX512F enabled" << std::endl;
        return 16; // 512-bit → 16 float
    #elif defined(__AVX2__)
        std::cout << "AVX2 enabled" << std::endl;
        return 8;  // 256-bit → 8 float
    #elif defined(__SSE__) || defined(__SSE2__)
        std::cout << "SSE enabled" << std::endl;
        return 4;  // 128-bit → 4 float
    #elif defined(__ARM_NEON) || defined(__ARM_NEON__)
        std::cout << "NEON enabled" << std::endl;
        return 4;  // NEON  128-bit
    #else
        return 1;  // fallback
    #endif
    }


class VectorStorage {
public:

    VectorStorage(dim_t dim, std::size_t max_elements, dim_t simd_width = 0,bool store_norms = false) : dim_{dim},max_elements_{max_elements},
     simd_width_{simd_width ? simd_width : DetectSimdWidth()},padded_dim_{PaddedDimFor(dim, simd_width)},data_{nullptr}, occupied_(max_elements, 0) {

        if (max_elements > 0) {
            const std::size_t total_size = max_elements_ * padded_dim_ * sizeof(float_t);
            data_ = static_cast<float_t*>(AlignedMalloc(total_size, 64));
            if (!data_) {
                throw std::bad_alloc();
            }
        }
    }

    ~VectorStorage(){
        if(data_){
            AlignedFree(data_);
        }
    }

    VectorStorage(const VectorStorage&) = delete;
    VectorStorage& operator=(const VectorStorage&) = delete;
    VectorStorage(VectorStorage&&) = delete;
    VectorStorage& operator=(VectorStorage&&) = delete;

    // Вставить или обновить вектор
    void AddVector(internal_id id, const float_t* vector_data){

        if (id < 0 || static_cast<std::size_t>(id) >= max_elements_) {
            throw std::out_of_range("Vector id out of range");
        }
        WriteVectorInternal(id, vector_data);
        occupied_[id] = 1;
    }

    void AddBatch(internal_id start_id, const float_t* contiguous_data, std::size_t N){
        
        if (N == 0) {
            return; 
        }
        
        if (!contiguous_data) {
            throw std::invalid_argument("Null pointer for vector data");
        }
        
        if (start_id < 0 || static_cast<std::size_t>(start_id) + N > max_elements_) {
            throw std::out_of_range("Batch ids out of range");
        }
        
    #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < N; ++i) {
            internal_id id = start_id + static_cast<internal_id>(i);
            const float* src = contiguous_data + i * dim_;
            WriteVectorInternal(id, src);
        }
    }

    const float_t* GetVectorPtr(internal_id id) const {
        if (!HasId(id)) {
            throw std::out_of_range("Vector id not found");
        }
        return data_ + OffsetFor(id);
    }

    // Метаданные
    dim_t GetDim() const noexcept { 
        return dim_; 
    }
    dim_t GetPaddedDim() const noexcept { 
        return padded_dim_; 
    }
    std::size_t GetCapacity() const noexcept { 
        return max_elements_; 
    }

    bool HasId(internal_id id) const {
        return (id >= 0 && static_cast<std::size_t>(id) < max_elements_ && occupied_[id]);
    }


private:
    dim_t dim_;          
    dim_t padded_dim_;   
    dim_t simd_width_;   
    std::size_t max_elements_;
    
    float_t* data_;             
    std::vector<uint8_t> occupied_; 

    inline std::size_t OffsetFor(internal_id id) const noexcept {
        return static_cast<std::size_t>(id) * static_cast<std::size_t>(padded_dim_);
    }

    void WriteVectorInternal(internal_id id, const float_t* src){
        const std::size_t offset = OffsetFor(id);
        float_t* dst = data_ + offset;
        
        // Copy and pad the vector
        CopyAndPad(dst, src, dim_, padded_dim_);
        
        // normalize vector in storage
        NormalizeInplace(dst,padded_dim_);
    }
};

} // namespace OptixNN
