#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/core/solver_contexts.hpp"
#include "utils/cuda_utils.hpp"

#include <cstddef>
#include <vector>
#include <complex>


// ---------------------------------------------------------------------------
// CudaMixedBuffers: mixed precision CUDA 경로의 device 버퍼.
//
// Ybus/Va/Vm/F/Ibus/Sbus → FP64 (authoritative)
// d_J_values/d_dx → FP32 (Jacobian·solve 연산용)
// Batch-major layout: [batch_size * dim] contiguous.
// ---------------------------------------------------------------------------
struct CudaMixedBuffers {
    void prepare(const InitializeContext& ctx);
    void upload(const SolveContext& ctx);
    void download(NRResult& result) const;
    void download_batch(NRBatchResult& result) const;

    std::size_t bus_offset(int32_t batch, int32_t bus) const
    {
        return static_cast<std::size_t>(batch) * static_cast<std::size_t>(n_bus) +
               static_cast<std::size_t>(bus);
    }

    std::size_t residual_offset(int32_t batch, int32_t row) const
    {
        return static_cast<std::size_t>(batch) * static_cast<std::size_t>(dimF) +
               static_cast<std::size_t>(row);
    }

    std::size_t jacobian_offset(int32_t batch, int32_t pos) const
    {
        return static_cast<std::size_t>(batch) * static_cast<std::size_t>(nnz_J) +
               static_cast<std::size_t>(pos);
    }

    std::size_t ybus_value_offset(int32_t batch, int32_t pos) const
    {
        return ybus_values_batched
            ? static_cast<std::size_t>(batch) * static_cast<std::size_t>(nnz_ybus) +
                  static_cast<std::size_t>(pos)
            : static_cast<std::size_t>(pos);
    }

    DeviceBuffer<double>  d_Ybus_re;
    DeviceBuffer<double>  d_Ybus_im;
    DeviceBuffer<int32_t> d_Ybus_indptr;
    DeviceBuffer<int32_t> d_Ybus_indices;
    DeviceBuffer<int32_t> d_Y_row;

    DeviceBuffer<float>   d_J_values;
    DeviceBuffer<int32_t> d_J_row_ptr;
    DeviceBuffer<int32_t> d_J_col_idx;

    DeviceBuffer<double>  d_F;
    DeviceBuffer<double>  d_normF;
    DeviceBuffer<float>   d_dx;

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
    int32_t nnz_ybus = 0;
    int32_t nnz_J    = 0;
    int32_t batch_size = 1;
    bool    ybus_values_batched = false;
};

#endif  // CUPF_WITH_CUDA
