#pragma once

#include <vector>

#include "mysolver/factorize/sparse_lu.hpp"

// GPU level-set sparse LU (PLAN §M3 cuDSS blueprint: factorize_ker small-path +
// level scheduling). Left-looking, one thread per column, processing all
// independent columns of an etree level in parallel. Race-free: a thread writes
// only its own column and reads already-factored descendant columns. This is the
// fine-grained "small-path" that amortizes launch overhead across many tiny
// supernodes — the regime where dense-block BLAS failed (cycle 23).
namespace mysolver::gpu {

// Analyze result: all symbolic structure + device buffers that depend only on
// the sparsity pattern, not the numeric values. cuDSS-style analyze/factorize
// split — build this once, then factorize repeatedly (same pattern, new values)
// reusing the device arena and skipping the host symbolic precompute. Owns one
// device allocation (`arena`); non-copyable, movable.
struct GpuFactorPlan {
    int n = 0, nnz = 0, num_levels = 0;
    void* arena = nullptr;  // owned device allocation
    double* d_Sx = nullptr;
    int *d_Sp = nullptr, *d_Si = nullptr, *d_lc = nullptr, *d_optr = nullptr;
    int *d_ouk = nullptr, *d_osr = nullptr, *d_otg = nullptr, *d_dp = nullptr;
    int* d_sing = nullptr;
    std::vector<int> Sp, Si;       // symmetric fill pattern (host, for `out` layout)
    std::vector<int> lptr;         // etree level boundaries (host, launch loop)
    std::vector<int> op_rl_ptr;    // right-looking: op ranges by updater level
    std::vector<int> scatter;      // A-entry index -> position in Sx (-1 = dropped)
    void* stream = nullptr;        // cudaStream_t (CUDA-graph replay of the schedule)
    void* graph_exec = nullptr;    // cudaGraphExec_t: captured finalize+scatter sequence

    GpuFactorPlan() = default;
    ~GpuFactorPlan();
    GpuFactorPlan(const GpuFactorPlan&) = delete;
    GpuFactorPlan& operator=(const GpuFactorPlan&) = delete;
    GpuFactorPlan(GpuFactorPlan&&) noexcept;
    GpuFactorPlan& operator=(GpuFactorPlan&&) noexcept;
};

// Analyze: build the symbolic structure (fill pattern, dependency map, etree
// levels, A->Sx scatter map), allocate the device arena, and upload everything
// that is value-independent. Depends only on the pattern (Ap/Ai/Lp/Li/parent).
GpuFactorPlan gpu_analyze(int n, const int* Ap, const int* Ai,
                          const std::vector<int>& Lp, const std::vector<int>& Li,
                          const std::vector<int>& parent);

// Factorize: scatter the new values Ax into the reused device buffer, run the
// level-set kernel, download. No host symbolic work, no device (re)allocation.
// Result in `out`, solvable by numeric::solve. False on zero pivot. Optional
// kernel_ms receives device kernel time only.
bool gpu_factorize(GpuFactorPlan& plan, const double* Ax, numeric::SparseLU& out,
                   double* kernel_ms = nullptr);

// Convenience: one-shot analyze + factorize (back-compat). For repeated factoring
// of the same pattern, call gpu_analyze once + gpu_factorize per value set.
bool gpu_factor(int n, const int* Ap, const int* Ai, const double* Ax,
                const std::vector<int>& Lp, const std::vector<int>& Li,
                const std::vector<int>& parent, numeric::SparseLU& out,
                double* kernel_ms = nullptr);

}  // namespace mysolver::gpu
