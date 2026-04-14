#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/op_interfaces.hpp"


// ---------------------------------------------------------------------------
// CudaJacobianOpVertexFp32: vertex-based FP32 Jacobian fill kernel.
//
// One warp owns one active bus row and walks the Ybus CSR row. This avoids the
// edge kernel's atomicAdd hot spot for off-diagonal entries.
// ---------------------------------------------------------------------------
class CudaJacobianOpVertexFp32 final : public IJacobianOp {
public:
    explicit CudaJacobianOpVertexFp32(IStorage& storage);
    void run(IterationContext& ctx) override;

private:
    IStorage& storage_;
};

#endif  // CUPF_WITH_CUDA
