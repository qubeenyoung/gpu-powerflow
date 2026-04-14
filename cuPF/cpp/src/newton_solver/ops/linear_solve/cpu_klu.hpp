#pragma once

#include "newton_solver/ops/op_interfaces.hpp"


class CpuFp64Storage;


// ---------------------------------------------------------------------------
// CpuLinearSolveKLU: KLU-based sparse direct solver for the CPU FP64 path.
//
// Symbolic analysis runs explicitly in the analyze phase. Numeric
// refactorization still happens every NR iteration.
// ---------------------------------------------------------------------------
class CpuLinearSolveKLU final : public ILinearSolveOp {
public:
    explicit CpuLinearSolveKLU(IStorage& storage);
    void analyze(const AnalyzeContext& ctx) override;
    void run(IterationContext& ctx) override;

private:
    CpuFp64Storage& storage_;
};
