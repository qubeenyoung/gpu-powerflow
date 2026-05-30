#pragma once

#include <vector>

#include "plan/multifrontal_plan.hpp"

namespace custom_linear_solver::solve {

void capture_multifrontal_solve_graph(custom_linear_solver::plan::MultifrontalPlan& plan,
                                      const std::vector<int>& front_ptr,
                                      const std::vector<int>& plcols, bool solve_f32,
                                      bool use_selected_inverse);

bool solve_multifrontal_device(custom_linear_solver::plan::MultifrontalPlan& plan,
                               const double* d_rhs, double* d_solution, const int* d_perm,
                               double* kernel_ms = nullptr);

}  // namespace custom_linear_solver::solve
