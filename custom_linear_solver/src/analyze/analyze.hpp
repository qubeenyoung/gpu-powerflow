#pragma once

// Plan-building pipeline: device CSR matrix + options → MultifrontalPlan + permutation buffers.
//
// Driven by `Solver::analyze()` (solver.cpp). The pipeline runs once per matrix sparsity pattern
// and is reused across factorize/solve calls (with the same sparsity but different numeric values):
//
//   1. build_csc_from_csr_device      — CSR → CSC on device
//   2. build_symmetric_graph_device   — A + A^T adjacency (METIS input)
//   3. metis_nd_from_graph            — METIS nested-dissection permutation (perm)
//   4. permute_csc_device             — apply perm; produces ordered_value_to_csr mapping
//   5. permute_symmetric_pattern      — host-side adjacency relabel for symbolic
//   6. symbolic::etree                — elimination tree
//   7. symbolic::fill_pattern         — L pattern
//   8. plan::analyze_multifrontal     — front layout + level packing + plan device buffers

#include <string>
#include <vector>

#include "internal/matrix_view.hpp"
#include "analyze/pattern/pattern_kernels.hpp"
#include "internal/plan/multifrontal_plan.hpp"

namespace custom_linear_solver::plan {

// Pipeline inputs (subset of SolverConfig — keeps plan/ independent of solver.hpp).
struct PlanBuildOptions {
    bool use_parallel_nested_dissection = true;
    int metis_seed = 42;
    int max_panel_width = 8;
    bool float_front = false;  // true if the factor/solve front is float (FP32 / TF32);
                               // controls whether float scratch arenas get allocated.
    // Debug dumps (off by default; surfaced through SolverConfig).
    // non-empty -> write q,p,fsz,nc,uc,level plus parent/extend metadata CSV here
    std::string dump_fronts_csv_path;
    bool emit_analyze_info = false;    // print front-size and subtree summary to stderr
};

// Pipeline outputs. All buffers needed by factorize/solve are bundled here so the caller
// (Solver::Impl) can `std::move` them into its state in one shot.
struct PlanBuildResult {
    MultifrontalPlan plan;
    std::vector<int> perm;       // host: METIS-ND permutation
    std::vector<int> iperm;      // host: inverse permutation
    matrix::IntDeviceBuffer d_perm;
    matrix::IntDeviceBuffer d_iperm;
    // Maps the analyze-time CSR ordering to the symbolic value slots (consumed by assemble_front_values
    // each factorize call).
    matrix::IntDeviceBuffer d_ordered_value_to_csr;
};

// Build the multifrontal plan from a device-resident CSR matrix. Returns true on success
// and populates `out`. On failure, `out` may be in a partially-constructed state and should
// be discarded by the caller.
bool build_plan_from_csr(const CsrMatrixView& matrix,
                         const PlanBuildOptions& options,
                         PlanBuildResult& out);

}  // namespace custom_linear_solver::plan
