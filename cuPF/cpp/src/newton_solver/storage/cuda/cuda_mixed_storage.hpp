#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/op_interfaces.hpp"
#include "newton_solver/core/contexts.hpp"
#include "utils/cuda_utils.hpp"

#include <cstddef>
#include <vector>
#include <complex>


// ---------------------------------------------------------------------------
// CudaMixedStorage: optimized mixed precision CUDA profile.
//
// Layout policy:
//   PublicScalar   = double
//   Va/Vm          = double  (authoritative voltage state)
//   V cache        = double  (derived V_re/V_im cache for mismatch)
//   Ybus           = float
//   Ibus           = double
//   Sbus           = double
//   F              = double
//   Jacobian       = FP64 inputs cast to FP32 inside kernels, FP32 values
//   dx             = float
// ---------------------------------------------------------------------------
class CudaMixedStorage final : public IStorage {
public:
    CudaMixedStorage();
    ~CudaMixedStorage();

    BackendKind   backend() const override { return BackendKind::CUDA; }
    ComputePolicy compute()  const override { return ComputePolicy::Mixed; }

    void prepare(const AnalyzeContext& ctx) override;
    void upload(const SolveContext&   ctx) override;
    void download_result(NRResultF64& result) const override;
    void download_batch_result(NRBatchResultF64& result) const override;

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

    // -----------------------------------------------------------------------
    // Device buffers — accessed by CUDA ops
    // -----------------------------------------------------------------------

    // Ybus (device, FP32 complex values, batch-common unless ybus_values_batched)
    DeviceBuffer<float>   d_Ybus_re;
    DeviceBuffer<float>   d_Ybus_im;
    DeviceBuffer<int32_t> d_Ybus_indptr;
    DeviceBuffer<int32_t> d_Ybus_indices;
    DeviceBuffer<int32_t> d_Y_row;

    // Jacobian (device, FP32 values, INT32 structure)
    DeviceBuffer<float>   d_J_values;   // FP32 Jacobian values
    DeviceBuffer<int32_t> d_J_row_ptr;
    DeviceBuffer<int32_t> d_J_col_idx;

    // Residual stays FP64. The FP32 solve path casts only when preparing RHS.
    DeviceBuffer<double>  d_F;
    DeviceBuffer<double>  d_normF;
    DeviceBuffer<float>   d_dx;

    // Voltage state.
    DeviceBuffer<double>  d_Va;         // [batch_size * n_bus], authoritative
    DeviceBuffer<double>  d_Vm;         // [batch_size * n_bus], authoritative
    DeviceBuffer<double>  d_V_re;       // [batch_size * n_bus], mismatch cache
    DeviceBuffer<double>  d_V_im;       // [batch_size * n_bus], mismatch cache

    // Sbus and reusable Ibus current cache stay FP64.
    DeviceBuffer<double>  d_Sbus_re;
    DeviceBuffer<double>  d_Sbus_im;
    DeviceBuffer<double>  d_Ibus_re;
    DeviceBuffer<double>  d_Ibus_im;

    // Jacobian maps (device, INT32)
    DeviceBuffer<int32_t> d_mapJ11, d_mapJ12, d_mapJ21, d_mapJ22;
    DeviceBuffer<int32_t> d_diagJ11, d_diagJ12, d_diagJ21, d_diagJ22;
    DeviceBuffer<int32_t> d_pvpq, d_pv, d_pq;

    // Topology (host)
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
