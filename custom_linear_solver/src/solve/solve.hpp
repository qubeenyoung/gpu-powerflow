#pragma once

#include "internal/runtime/state.hpp"

// Solve driver. Solve.cu lazily captures the full Solve graph (gather + levels
// + scatter).

namespace custom_linear_solver {

bool Solve(const plan::MultifrontalPlan& plan, State& state,
           const double* d_rhs_batch, double* d_solution_batch,
           const int* d_perm, const int* d_iperm);
bool Solve(const plan::MultifrontalPlan& plan, State& state,
           const double* d_rhs_batch, float* d_solution_batch,
           const int* d_perm, const int* d_iperm);
bool Solve(const plan::MultifrontalPlan& plan, State& state,
           const float* d_rhs_batch, float* d_solution_batch, const int* d_perm,
           const int* d_iperm);

}  // namespace custom_linear_solver
