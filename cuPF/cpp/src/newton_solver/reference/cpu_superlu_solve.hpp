#pragma once

#include "newton_solver/core/solver_contexts.hpp"


struct CpuFp64Buffers;


// ---------------------------------------------------------------------------
// CpuLinearSolveSuperLU: one-shot SuperLU solve for CPU FP64.
//
// Mirrors the v2 naive CPU path and SciPy's spsolve() fallback:
// symbolic ordering + numeric factorization + triangular solve every call.
// ---------------------------------------------------------------------------
struct CpuLinearSolveSuperLU {
    void initialize(CpuFp64Buffers& buf, const InitializeContext& ctx);
    void prepare_rhs(CpuFp64Buffers& buf, IterationContext& ctx);
    void factorize(CpuFp64Buffers& buf, IterationContext& ctx);
    void solve(CpuFp64Buffers& buf, IterationContext& ctx);
};
