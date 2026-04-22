#pragma once

#include "newton_solver/core/solver_contexts.hpp"


struct CpuFp64Buffers;


// ---------------------------------------------------------------------------
// CpuNaiveJacobianOpF64: PyPower-like Jacobian assembly on CPU FP64.
//
// Mirrors v2 naive CPU backend / PYPOWER's dSbus_dV workflow:
//   1. build full dS_dVa and dS_dVm sparse matrices
//   2. slice J11/J12/J21/J22
//   3. rebuild J from triplets every iteration
// ---------------------------------------------------------------------------
struct CpuNaiveJacobianOpF64 {
    void run(CpuFp64Buffers& buf, IterationContext& ctx);
};
