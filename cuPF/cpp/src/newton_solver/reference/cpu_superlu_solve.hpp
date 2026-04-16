#pragma once

#include "newton_solver/ops/op_interfaces.hpp"


class CpuFp64Storage;


// ---------------------------------------------------------------------------
// CpuLinearSolveSuperLU: one-shot SuperLU solve for CPU FP64.
//
// Mirrors the v2 naive CPU path and SciPy's spsolve() fallback:
// symbolic ordering + numeric factorization + triangular solve every call.
// ---------------------------------------------------------------------------
class CpuLinearSolveSuperLU final : public ILinearSolveOp {
public:
    explicit CpuLinearSolveSuperLU(IStorage& storage);
    void analyze(const AnalyzeContext& ctx) override;
    void factorize(IterationContext& ctx) override;
    void solve(IterationContext& ctx) override;

private:
    CpuFp64Storage& storage_;
};
