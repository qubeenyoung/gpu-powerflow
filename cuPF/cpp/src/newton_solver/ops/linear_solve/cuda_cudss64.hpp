#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/op_interfaces.hpp"


// ---------------------------------------------------------------------------
// CudaLinearSolveCuDSS64: cuDSS-based sparse direct solver in FP64.
//
// Used for the end-to-end FP64 CUDA path.
// ---------------------------------------------------------------------------
class CudaLinearSolveCuDSS64 final : public ILinearSolveOp {
public:
    explicit CudaLinearSolveCuDSS64(IStorage& storage);
    ~CudaLinearSolveCuDSS64();

    void analyze(const AnalyzeContext& ctx) override;
    void factorize(IterationContext& ctx) override;
    void solve(IterationContext& ctx) override;

private:
    IStorage& storage_;

    struct CuDSS64State;
    CuDSS64State* state_ = nullptr;
};

#endif  // CUPF_WITH_CUDA
