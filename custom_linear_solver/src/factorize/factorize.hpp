#pragma once

#include "internal/runtime/state.hpp"

// Factorize driver + factor-kernel registration. The host wrappers (register_factor_attributes,
// issue_factor) are called by internal/runtime/setup.cu so it needs no kernel includes.

namespace custom_linear_solver {

void register_factor_attributes(Precision precision);
void issue_factor(const plan::MultifrontalPlan& plan, State& state, void* stream);

bool factorize(const plan::MultifrontalPlan& plan, State& state,
               const double* d_values_batch, const int* d_ordered_value_to_csr);
bool factorize(const plan::MultifrontalPlan& plan, State& state,
               const float* d_values_batch, const int* d_ordered_value_to_csr);

}  // namespace custom_linear_solver
