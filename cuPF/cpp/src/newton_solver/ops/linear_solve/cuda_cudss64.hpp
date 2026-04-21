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
    explicit CudaLinearSolveCuDSS64(IStorage& storage,
                                    CuDSSOptions cudss_options = {});
    ~CudaLinearSolveCuDSS64();

    void analyze(const AnalyzeContext& ctx) override;
    void run(IterationContext& ctx) override;
    void factorize(IterationContext& ctx);
    void solve(IterationContext& ctx);
    void factorize_and_solve(IterationContext& ctx);

private:
    IStorage& storage_;
    CuDSSOptions cudss_options_;

    struct CuDSS64State;
    CuDSS64State* state_ = nullptr;
};

#endif  // CUPF_WITH_CUDA
