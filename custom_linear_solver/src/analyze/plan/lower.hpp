#pragma once

#include <vector>

#include "analyze/symbolic/supernode.hpp"
#include "internal/plan/multifrontal_plan.hpp"

namespace custom_linear_solver::plan {

// Build the MultifrontalPlan from symbolic + L pattern. Inputs:
//   n, nnz_a, d_Ap, d_Ai  - input A (device CSR/CSC pattern, host shape)
//   Lp, Li                - L factor pattern (host vectors)
//   parent                - Etree parent (host)
//   max_panel_width       - panel cap hint (clamped to [1, 64])
//   float_front           - true if the factor/Solve front is float
//                           (FP32/TC/TF32); allocates float scratch arenas
//                           (d_front_f, d_y_f)
//   forced_panels         - optional caller-supplied panel partition (else
//                           built here)
//   emit_info             - if true, print front-size and subtree summary to
//                           stderr
//   dump_fronts_only      - if true, stop after host symbolic front/level
//                           metadata is built
MultifrontalPlan AnalyzeMultifrontal(
    int n, int nnz_a, const int* d_Ap, const int* d_Ai,
    const std::vector<int>& Lp, const std::vector<int>& Li,
    const std::vector<int>& parent, int max_panel_width,
    const custom_linear_solver::symbolic::PanelPartition* forced_panels,
    bool float_front, bool emit_info = false, bool dump_fronts_only = false);

}  // namespace custom_linear_solver::plan
