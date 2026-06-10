#pragma once

#include "internal/runtime/state.hpp"

// Solve driver. issue_solve is the host wrapper called by internal/runtime/setup.cu for graph capture.

namespace custom_linear_solver {

void issue_solve(const plan::MultifrontalPlan& plan, State& state, void* stream);

bool solve(const plan::MultifrontalPlan& plan, State& state,
           const double* d_rhs_batch, double* d_solution_batch, const int* d_perm);
bool solve(const plan::MultifrontalPlan& plan, State& state,
           const double* d_rhs_batch, float* d_solution_batch, const int* d_perm);
bool solve(const plan::MultifrontalPlan& plan, State& state,
           const float* d_rhs_batch, float* d_solution_batch, const int* d_perm);

}  // namespace custom_linear_solver
