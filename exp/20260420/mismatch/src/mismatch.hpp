#pragma once

#include "data_types.hpp"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdint>
#include <cstddef>

namespace exp20260420::mismatch {

struct MismatchWorkspace {
    int32_t n_bus = 0;
    int32_t n_edges = 0;
    int32_t n_pv = 0;
    int32_t n_pq = 0;
    int32_t dim = 0;
    int32_t norm_blocks = 0;

    const int32_t* pv = nullptr;
    const int32_t* pq = nullptr;

    cusparseHandle_t cusparse = nullptr;
    cusparseSpMatDescr_t ybus_descr = nullptr;
    cusparseDnVecDescr_t v_descr = nullptr;
    cusparseDnVecDescr_t ibus_descr = nullptr;

    cuFloatComplex* ybus = nullptr;
    cuFloatComplex* v = nullptr;
    cuFloatComplex* ibus = nullptr;
    float* block_norm = nullptr;
    float* norm_value = nullptr;
    void* spmv_buffer = nullptr;
    std::size_t spmv_buffer_bytes = 0;
};

void mismatchAnalyze(MismatchWorkspace& ws,
                     const YbusGraph& ybus,
                     const int32_t* pv,
                     int32_t n_pv,
                     const int32_t* pq,
                     int32_t n_pq,
                     cudaStream_t stream = nullptr);

void mismatchUpdateYbus(MismatchWorkspace& ws,
                        const YbusGraph& ybus,
                        cudaStream_t stream = nullptr);

double mismatchCompute(MismatchWorkspace& ws,
                       const float* v_re,
                       const float* v_im,
                       const float* sbus_re,
                       const float* sbus_im,
                       float* F,
                       cudaStream_t stream = nullptr);

void mismatchDestroy(MismatchWorkspace& ws);

}  // namespace exp20260420::mismatch
