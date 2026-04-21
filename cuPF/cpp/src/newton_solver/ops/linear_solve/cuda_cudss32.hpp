#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/op_interfaces.hpp"


// ---------------------------------------------------------------------------
// CudaLinearSolveCuDSS32: cuDSS-based sparse direct solver in FP32.
//
// Used for Mixed and FP32 paths (both use FP32 Jacobian/solve internally).
// cuDSS FP32 mode: analysis done once, refactorization each iteration.
// ---------------------------------------------------------------------------
class CudaLinearSolveCuDSS32 final : public ILinearSolveOp {
public:
    explicit CudaLinearSolveCuDSS32(IStorage& storage,
                                    CuDSSOptions cudss_options = {});
    ~CudaLinearSolveCuDSS32();

    void analyze(const AnalyzeContext& ctx) override;
    void run(IterationContext& ctx) override;
    void factorize(IterationContext& ctx);
    void solve(IterationContext& ctx);
    void factorize_and_solve(IterationContext& ctx);

    struct CuDSS32State;

private:
    IStorage& storage_;
    CuDSSOptions cudss_options_;

    CuDSS32State* state_ = nullptr;
};

#endif  // CUPF_WITH_CUDA
