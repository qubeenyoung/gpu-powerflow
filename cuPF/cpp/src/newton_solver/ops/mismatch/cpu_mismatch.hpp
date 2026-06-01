#pragma once

#include "newton_solver/core/solver_contexts.hpp"

struct CpuFp64Storage;


struct CpuMismatchOp {
    void run(CpuFp64Storage& buf, IterationContext& ctx);
};

struct CpuMismatchNormOp {
    void run(CpuFp64Storage& buf, IterationContext& ctx);
};
