#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/core/solver_contexts.hpp"
#include "utils/cuda_utils.hpp"

#include <cstddef>
#include <complex>


// ---------------------------------------------------------------------------
// CudaFp32Storage: FP32 CUDA path device buffers.
//
// Public inputs are still the existing FP64 API, but upload casts all numeric
// device-side state to float. All GPU numeric buffers/operators for this path
// stay FP32.
// ---------------------------------------------------------------------------
struct CudaFp32Storage {
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

    DeviceBuffer<float>   d_Ybus_re;
    DeviceBuffer<float>   d_Ybus_im;
    DeviceBuffer<int32_t> d_Ybus_indptr;
    DeviceBuffer<int32_t> d_Ybus_indices;
    DeviceBuffer<int32_t> d_Ybus_row;

    DeviceBuffer<float>   d_J_values;
    DeviceBuffer<int32_t> d_J_row_ptr;
    DeviceBuffer<int32_t> d_J_col_idx;

    DeviceBuffer<float>   d_F;
    DeviceBuffer<float>   d_normF;
    DeviceBuffer<float>   d_dx;

    DeviceBuffer<float>   d_Va;
    DeviceBuffer<float>   d_Vm;
    DeviceBuffer<float>   d_V_re;
    DeviceBuffer<float>   d_V_im;

    DeviceBuffer<float>   d_Sbus_re;
    DeviceBuffer<float>   d_Sbus_im;
    DeviceBuffer<float>   d_Ibus_re;
    DeviceBuffer<float>   d_Ibus_im;

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
