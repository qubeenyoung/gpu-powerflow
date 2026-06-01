#pragma once

#include "newton_solver/core/solver_contexts.hpp"

struct CpuFp64Storage;


struct CpuJacobianOpF64 {
    void run(CpuFp64Storage& buf, IterationContext& ctx);
};


struct CpuPandapowerJacobianOpF64 {
    void run(CpuFp64Storage& buf, IterationContext& ctx);
};
