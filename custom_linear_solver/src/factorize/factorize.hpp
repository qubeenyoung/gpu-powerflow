#pragma once

#include "internal/runtime/state.hpp"

// Factorize driver + factor-kernel registration. The host wrappers
// (RegisterFactorAttributes, IssueFactor) are called by
// internal/runtime/Setup.cu so it needs no kernel includes.

namespace custom_linear_solver {

void RegisterFactorAttributes(Precision precision);
void IssueFactor(const plan::MultifrontalPlan& plan, State& state,
                 void* stream);

bool Factorize(const plan::MultifrontalPlan& plan, State& state,
               const double* d_values_batch, const int* d_ordered_value_to_csr);
bool Factorize(const plan::MultifrontalPlan& plan, State& state,
               const float* d_values_batch, const int* d_ordered_value_to_csr);

}  // namespace custom_linear_solver
