#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/op_interfaces.hpp"


// ---------------------------------------------------------------------------
// CudaJacobianOpEdgeFp64: edge-based FP64 Jacobian fill kernel.
//
// Used for the end-to-end FP64 CUDA path.
// ---------------------------------------------------------------------------
class CudaJacobianOpEdgeFp64 final : public IJacobianOp {
public:
    explicit CudaJacobianOpEdgeFp64(IStorage& storage);
    void run(IterationContext& ctx) override;

private:
    IStorage& storage_;
};

#endif  // CUPF_WITH_CUDA
