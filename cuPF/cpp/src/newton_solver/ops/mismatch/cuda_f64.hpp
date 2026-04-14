#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/op_interfaces.hpp"


// ---------------------------------------------------------------------------
// CudaMismatchOpF64: CUDA mismatch operator shared by CUDA FP64 and Mixed.
//
// Public I/O and mismatch residual are FP64 in both profiles. Mixed casts to
// FP32 only at the cuDSS32 RHS boundary, not inside mismatch.
// ---------------------------------------------------------------------------
class CudaMismatchOpF64 final : public IMismatchOp {
public:
    explicit CudaMismatchOpF64(IStorage& storage);
    void run(IterationContext& ctx) override;

private:
    IStorage& storage_;
};

#endif  // CUPF_WITH_CUDA
