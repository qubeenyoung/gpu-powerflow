#pragma once

#include <vector>

#include "plan/multifrontal_plan.hpp"
#include "symbolic/supernode.hpp"

namespace custom_linear_solver::plan {

// Build the MultifrontalPlan from symbolic + L pattern. Inputs:
//   n, nnz_a, d_Ap, d_Ai  - input A (device CSR/CSC pattern, host shape)
//   Lp, Li                 - L factor pattern (host vectors)
//   parent                 - etree parent (host)
//   panel_cap              - panel cap hint (analyze may override for large n)
//   fp32, pure_fp32        - precision hints used to size optional FP32 working arenas
//   forced_panels          - optional caller-supplied panel partition (else built here)
MultifrontalPlan analyze_multifrontal(int n, int nnz_a, const int* d_Ap, const int* d_Ai,
                                      const std::vector<int>& Lp,
                                      const std::vector<int>& Li,
                                      const std::vector<int>& parent, int panel_cap, bool fp32,
                                      const custom_linear_solver::symbolic::PanelPartition* forced_panels,
                                      bool pure_fp32);

}  // namespace custom_linear_solver::plan
