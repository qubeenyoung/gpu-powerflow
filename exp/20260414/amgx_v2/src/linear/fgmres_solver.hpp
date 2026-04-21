#pragma once

#include "solver/common.hpp"

#include <cstdint>

namespace exp_20260414::amgx_v2 {

class FgmresSolver {
public:
    using OperatorFn = void (*)(const double* input_device, double* output_device, void* user);

    SolveStats solve(int32_t dim,
                     const double* rhs_device,
                     double* dx_device,
                     const SolverOptions& options,
                     OperatorFn apply_operator,
                     OperatorFn apply_preconditioner,
                     void* user);
};

}  // namespace exp_20260414::amgx_v2
