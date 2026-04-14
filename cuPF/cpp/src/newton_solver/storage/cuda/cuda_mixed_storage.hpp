#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/op_interfaces.hpp"
#include "newton_solver/core/contexts.hpp"
#include "utils/cuda_utils.hpp"

#include <vector>
#include <complex>


// ---------------------------------------------------------------------------
// CudaMixedStorage: Mixed precision (FP64 public API, FP32 Jacobian/solve,
//                   FP64 voltage state).
//
// Layout policy:
//   PublicScalar   = double
//   VoltageScalar  = double
//   MismatchScalar = double
//   JacobianScalar = float
//   SolveScalar    = float
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

    // -----------------------------------------------------------------------
    // Device buffers — accessed by CUDA Single ops
    // -----------------------------------------------------------------------

    // Ybus (device, FP64 complex)
    DeviceBuffer<double>  d_Ybus_re;
    DeviceBuffer<double>  d_Ybus_im;
    DeviceBuffer<int32_t> d_Ybus_indptr;
    DeviceBuffer<int32_t> d_Ybus_indices;
    DeviceBuffer<int32_t> d_Y_row;

    // Jacobian (device, FP32 values, INT32 structure)
    DeviceBuffer<float>   d_J_values;   // FP32 Jacobian values
    DeviceBuffer<int32_t> d_J_row_ptr;
    DeviceBuffer<int32_t> d_J_col_idx;

    // Mismatch stays FP64. The FP32 solve path casts only when preparing RHS.
    DeviceBuffer<double>  d_F;
    DeviceBuffer<float>   d_dx;

    // Voltage state (device, FP64)
    DeviceBuffer<double>  d_Va;         // [n_bus]
    DeviceBuffer<double>  d_Vm;         // [n_bus]
    DeviceBuffer<double>  d_V_re;       // [n_bus]
    DeviceBuffer<double>  d_V_im;       // [n_bus]

    // Sbus (device, FP64 complex)
    DeviceBuffer<double>  d_Sbus_re;
    DeviceBuffer<double>  d_Sbus_im;

    // Jacobian maps (device, INT32)
    DeviceBuffer<int32_t> d_mapJ11, d_mapJ12, d_mapJ21, d_mapJ22;
    DeviceBuffer<int32_t> d_diagJ11, d_diagJ12, d_diagJ21, d_diagJ22;
    DeviceBuffer<int32_t> d_pvpq, d_pv, d_pq;

    // Topology (host)
    int32_t n_bus  = 0;
    int32_t n_pvpq = 0;
    int32_t n_pq   = 0;
    int32_t dimF   = 0;

};

#endif  // CUPF_WITH_CUDA
