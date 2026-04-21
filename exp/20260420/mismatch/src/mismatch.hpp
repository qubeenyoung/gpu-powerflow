#pragma once

#include "data_types.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace exp20260420::mismatch {

struct MismatchWorkspace {
    int32_t n_bus = 0;
    int32_t n_edges = 0;
    int32_t n_pv = 0;
    int32_t n_pq = 0;
    int32_t dim = 0;
    int32_t batch_size = 1;
    int32_t norm_blocks = 0;

    YbusGraph ybus{};
    const int32_t* pv = nullptr;
    const int32_t* pq = nullptr;

    double* block_norm = nullptr;
    double* norm_value = nullptr;
};

void mismatchAnalyze(MismatchWorkspace& ws,
                     const YbusGraph& ybus,
                     const int32_t* pv,
                     int32_t n_pv,
                     const int32_t* pq,
                     int32_t n_pq,
                     int32_t batch_size,
                     cudaStream_t stream = nullptr);

double mismatchCompute(MismatchWorkspace& ws,
                       const double* v_re,
                       const double* v_im,
                       const float* sbus_re,
                       const float* sbus_im,
                       double* F,
                       cudaStream_t stream = nullptr);

void mismatchDestroy(MismatchWorkspace& ws);

}  // namespace exp20260420::mismatch
