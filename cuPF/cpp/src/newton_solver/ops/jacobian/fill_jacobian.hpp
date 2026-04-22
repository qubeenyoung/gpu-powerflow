#pragma once

#include "newton_solver/core/solver_contexts.hpp"

struct CpuFp64Buffers;


struct CpuJacobianOpF64 {
    void run(CpuFp64Buffers& buf, IterationContext& ctx);
};
