#pragma once

#include <vector>

#include "mysolver/factorize/sparse_lu.hpp"

// GPU multifrontal factorization (PLAN §M3 dense-panel large-path). Ports the
// CPU-validated num::multifrontal_factor: relaxed dense panels, one CUDA block
// per front cooperating on the dense no-pivot LU, contribution blocks extend-added
// into parent fronts via a precomputed indexed scatter. This replaces the cy71
// right-looking per-op atomicAdd scatter (53.8M scattered atomics on SyntheticUSA)
// with dense block work + a 7-30x smaller extend-add scatter (cycle 78 gate).
// Kept SEPARATE from gpu_factor.cu (the proven cy71 kernel) so it can be benched
// and validated independently before becoming the large-path.
namespace mysolver::gpu {

// Analyze result for the multifrontal path: panel/front structure + device arena.
// Value-independent; build once, factorize repeatedly.
struct GpuMfPlan {
    int n = 0, num_panels = 0, num_plevels = 0;
    int nnz_a = 0, nnz_s = 0, asm_total = 0, emit_total = 0;
    long front_total = 0;  // Σ fsz² doubles (front arena size)
    void* arena = nullptr;
    double *d_front = nullptr, *d_Ax = nullptr, *d_Sx = nullptr;
    float* d_frontf = nullptr;  // FP32 view of the front arena (mixed-precision factor/solve)
    bool fp32 = false;          // run factor+solve in FP32 (accuracy recovered by refinement)
    int *d_front_off = nullptr, *d_front_ptr = nullptr, *d_ncols = nullptr;
    int *d_plcols = nullptr, *d_panel_parent = nullptr;
    int *d_asm_ptr = nullptr, *d_asm_local = nullptr;
    int *d_a_pos = nullptr, *d_emit_front = nullptr;
    int* d_sing = nullptr;
    int* d_front_rows = nullptr;  // global row indices per front (solve gather/scatter)
    double* d_y = nullptr;        // solve working vector (b -> x, in place)
    int front_store = 0;          // Σ fsz (length of d_front_rows)
    std::vector<int> Sp, Si;   // symmetric fill pattern (host, for SparseLU output)
    std::vector<int> plptr;    // panel-level boundaries (host, launch loop)
    void* stream = nullptr;
    void* graph_exec = nullptr;
    void* solve_graph_exec = nullptr;  // multifrontal fwd+bwd schedule
    void* solve_graph = nullptr;       // cy172: kept (MF_SOLVE_CC) to instantiate concurrent execs

    GpuMfPlan() = default;
    ~GpuMfPlan();
    GpuMfPlan(const GpuMfPlan&) = delete;
    GpuMfPlan& operator=(const GpuMfPlan&) = delete;
    GpuMfPlan(GpuMfPlan&&) noexcept;
    GpuMfPlan& operator=(GpuMfPlan&&) noexcept;
};

// Build the multifrontal symbolic structure + device arena from the pattern
// (Ap/Ai for the A->front map, Lp/Li fill, etree parent). panel_cap caps panel
// width. Returns an empty plan (num_panels==0) if the front arena would be too big.
GpuMfPlan gpu_mf_analyze(int n, const int* Ap, const int* Ai, const std::vector<int>& Lp,
                         const std::vector<int>& Li, const std::vector<int>& parent,
                         int panel_cap = 8, bool fp32 = false);

// Factorize: scatter Ax into the fronts, run the level-scheduled dense-front LU +
// extend-add, emit into the SparseLU layout. False on a zero pivot. Optional
// kernel_ms gets device kernel time only.
bool gpu_mf_factorize(GpuMfPlan& plan, const double* Ax, numeric::SparseLU& out,
                      double* kernel_ms = nullptr);

// MULTIFRONTAL solve A x = b on the GPU, reusing the factored fronts (must be
// called after gpu_mf_factorize, before the next factorize overwrites the arena).
// Forward L y = b and backward U x = y over the PANEL etree levels (~72 vs the
// thousands of scalar column levels that made the level-set solve a dead end),
// using dense per-front block triangular solves. Writes the solution to x_out.
// kernel_ms (optional) gets the device fwd+bwd time only.
void gpu_mf_solve(GpuMfPlan& plan, const std::vector<double>& b, std::vector<double>& x_out,
                  double* kernel_ms = nullptr);

}  // namespace mysolver::gpu
