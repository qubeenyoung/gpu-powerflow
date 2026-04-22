#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/core/solver_contexts.hpp"
#include "utils/cuda_utils.hpp"


// ---------------------------------------------------------------------------
// CudaFp64Buffers: end-to-end FP64 CUDA 경로의 device 버퍼.
// ---------------------------------------------------------------------------
struct CudaFp64Buffers {
    void prepare(const InitializeContext& ctx);
    void upload(const SolveContext& ctx);
    void download(NRResult& result) const;

    DeviceBuffer<double>  d_Ybus_re;
    DeviceBuffer<double>  d_Ybus_im;
    DeviceBuffer<int32_t> d_Ybus_indptr;
    DeviceBuffer<int32_t> d_Ybus_indices;
    DeviceBuffer<int32_t> d_Y_row;

    DeviceBuffer<double>  d_J_values;
    DeviceBuffer<int32_t> d_J_row_ptr;
    DeviceBuffer<int32_t> d_J_col_idx;

    DeviceBuffer<double>  d_F;
    DeviceBuffer<double>  d_normF;
    DeviceBuffer<double>  d_dx;

    DeviceBuffer<double>  d_Va;
    DeviceBuffer<double>  d_Vm;
    DeviceBuffer<double>  d_V_re;
    DeviceBuffer<double>  d_V_im;

    DeviceBuffer<double>  d_Sbus_re;
    DeviceBuffer<double>  d_Sbus_im;
    DeviceBuffer<double>  d_Ibus_re;
    DeviceBuffer<double>  d_Ibus_im;

    DeviceBuffer<int32_t> d_mapJ11, d_mapJ12, d_mapJ21, d_mapJ22;
    DeviceBuffer<int32_t> d_diagJ11, d_diagJ12, d_diagJ21, d_diagJ22;
    DeviceBuffer<int32_t> d_pvpq, d_pv, d_pq;

    int32_t n_bus  = 0;
    int32_t n_pvpq = 0;
    int32_t n_pq   = 0;
    int32_t dimF   = 0;
};

#endif  // CUPF_WITH_CUDA
