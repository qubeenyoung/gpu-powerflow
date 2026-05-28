#pragma once

#include <vector>

#include "mysolver/factorize/sparse_lu.hpp"

// GPU triangular solve (PLAN §M4: forward.cu / backward.cu, level-set).
// Forward L y = b and backward U x = y over the symmetric-fill factor, with
// columns of an etree level processed in parallel; the scattered updates use
// atomicAdd (race-free). Matches numeric::solve semantics (unit-lower L, U with
// pivot diagonal).
namespace mysolver::gpu {

// Solve A x = b on the GPU given the factored SparseLU (from gpu_factor /
// factor_nopiv) and the elimination tree parent. Writes the solution to x_out.
void gpu_solve(const numeric::SparseLU& lu, const std::vector<int>& parent,
               const std::vector<double>& b, std::vector<double>& x_out);

// Device-resident solve with an analyze/apply split (like the factor path). The
// factor (Sp/Si/Sx), etree levels and pivot map are uploaded once and the whole
// fwd+bwd level schedule (2*num_levels launches) is captured in ONE CUDA graph, so
// repeated solves pay only the b upload + graph replay + x download -- no per-level
// launch overhead, no re-upload of the constant factor. This is the regime cuDSS's
// solve runs in (factor already resident). Non-copyable, movable; owns one arena.
struct GpuSolvePlan {
    int n = 0, nnz = 0, num_levels = 0;
    void* arena = nullptr;
    int *d_Sp = nullptr, *d_Si = nullptr, *d_lc = nullptr, *d_dp = nullptr;
    double *d_Sx = nullptr, *d_x = nullptr;
    void* stream = nullptr;
    void* graph_exec = nullptr;

    GpuSolvePlan() = default;
    ~GpuSolvePlan();
    GpuSolvePlan(const GpuSolvePlan&) = delete;
    GpuSolvePlan& operator=(const GpuSolvePlan&) = delete;
    GpuSolvePlan(GpuSolvePlan&&) noexcept;
    GpuSolvePlan& operator=(GpuSolvePlan&&) noexcept;
};

// Analyze: upload the factor + schedule, capture the fwd/bwd graph. Reusable across
// solves with the same factor (multiple right-hand sides / NR iterations).
GpuSolvePlan gpu_solve_analyze(const numeric::SparseLU& lu, const std::vector<int>& parent);

// Apply: solve A x = b (upload b, replay graph, download x). kernel_ms (optional)
// receives the device fwd+bwd time only.
void gpu_solve_apply(GpuSolvePlan& plan, const std::vector<double>& b,
                     std::vector<double>& x_out, double* kernel_ms = nullptr);

}  // namespace mysolver::gpu
