#pragma once

#include "newton_solver/core/solver_contexts.hpp"

struct CpuFp64Storage;


struct CpuVoltageUpdateOp {
    void run(CpuFp64Storage& buf, IterationContext& ctx);
};
