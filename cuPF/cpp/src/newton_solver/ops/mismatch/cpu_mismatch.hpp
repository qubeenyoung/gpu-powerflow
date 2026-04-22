#pragma once

#include "newton_solver/core/solver_contexts.hpp"

struct CpuFp64Buffers;


struct CpuMismatchOp {
    void run(CpuFp64Buffers& buf, IterationContext& ctx);
};

struct CpuMismatchNormOp {
    void run(CpuFp64Buffers& buf, IterationContext& ctx);
};
