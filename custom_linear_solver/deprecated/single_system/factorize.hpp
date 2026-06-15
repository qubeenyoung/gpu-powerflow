#pragma once

#include <vector>

#include "plan/multifrontal_plan.hpp"
#include "symbolic/supernode.hpp"

// GPU multifrontal factorization (PLAN §M3 dense-panel large-path). Ports the
// CPU-validated num::multifrontal_factor: relaxed dense panels, one CUDA block
// per front cooperating on the dense no-pivot LU, contribution blocks extend-added
// into parent fronts via a precomputed indexed scatter. This replaces the cy71
// right-looking per-op atomicAdd scatter (53.8M scattered atomics on SyntheticUSA)
// with dense block work + a 7-30x smaller extend-add scatter (cycle 78 gate).
// Kept SEPARATE from gpu_factor.cu (the proven cy71 kernel) so it can be benched
// and validated independently before becoming the large-path.
namespace custom_linear_solver::factorize {

// Build the multifrontal symbolic structure + device arena from the pattern.
// d_Ap/d_Ai are the ordered CSC pattern used to build the device-resident
// A-entry -> front map. panel_cap caps panel width. Returns an empty plan
// (num_panels==0) if the front arena would be too big.
custom_linear_solver::plan::MultifrontalPlan analyze_multifrontal(
    int n, int nnz_a, const int* d_Ap, const int* d_Ai,
    const std::vector<int>& Lp, const std::vector<int>& Li,
    const std::vector<int>& parent, int panel_cap = 8, bool fp32 = false,
    const custom_linear_solver::symbolic::PanelPartition* forced_panels = nullptr,
    bool pure_fp32 = false);

bool factorize_multifrontal_device(custom_linear_solver::plan::MultifrontalPlan& plan,
                                   const double* d_csr_values,
                                   const int* d_ordered_value_to_csr,
                                   double* kernel_ms = nullptr);

bool factorize_multifrontal_device(custom_linear_solver::plan::MultifrontalPlan& plan,
                                   const float* d_csr_values,
                                   const int* d_ordered_value_to_csr,
                                   double* kernel_ms = nullptr);

}  // namespace custom_linear_solver::factorize
