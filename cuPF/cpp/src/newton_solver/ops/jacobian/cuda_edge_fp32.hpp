#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/op_interfaces.hpp"


// ---------------------------------------------------------------------------
// CudaJacobianOpEdgeFp32: edge-based FP32 Jacobian fill kernel.
//
// Used by the CUDA Mixed profile. Ybus/J are FP32 while V cache and Ibus are
// FP64; final Jacobian entries are written as FP32.
// ---------------------------------------------------------------------------
class CudaJacobianOpEdgeFp32 final : public IJacobianOp {
public:
    explicit CudaJacobianOpEdgeFp32(IStorage& storage);
    void run(IterationContext& ctx) override;

private:
    IStorage& storage_;
};

#endif  // CUPF_WITH_CUDA
