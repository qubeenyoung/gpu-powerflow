#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/op_interfaces.hpp"


// ---------------------------------------------------------------------------
// CudaJacobianOpEdgeFp32: edge-based FP32 Jacobian fill kernel.
//
// Used by the CUDA Mixed profile. Public voltage state is FP64; this op casts
// Ybus/V to FP32 inside the kernel before filling the FP32 Jacobian.
// ---------------------------------------------------------------------------
class CudaJacobianOpEdgeFp32 final : public IJacobianOp {
public:
    explicit CudaJacobianOpEdgeFp32(IStorage& storage);
    void run(IterationContext& ctx) override;

private:
    IStorage& storage_;
};

#endif  // CUPF_WITH_CUDA
