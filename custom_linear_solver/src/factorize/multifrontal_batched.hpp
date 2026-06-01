#pragma once

#include "plan/multifrontal_plan.hpp"

// Uniform-batch multifrontal factorize + solve: B linear systems that share the SAME
// sparsity pattern (hence the same symbolic plan) but have different numeric values / RHS.
// The symbolic analysis (analyze_multifrontal) is done ONCE and shared; only the numeric
// factor/solve are batched. Layout is FRONT-MAJOR: front arena = B * front_total, batch b's
// front for panel p starts at b*front_total + front_off[p]. Each dense per-front kernel adds
// blockIdx.y = batch, so one kernel launch per etree level covers all B fronts -> the launch
// and per-level synchronization latency that bottlenecks the single-system path is amortized
// across B systems, and the B*level_size blocks fill the GPU on the otherwise occupancy-
// starved narrow levels. This is the regime where lower precision / tensor cores finally
// matter (the single-system path is latency-bound, not compute-bound).
namespace custom_linear_solver::factorize {

struct BatchedState {
    int B = 0;
    long front_total = 0;
    int n = 0;
    bool fp32 = false;
    bool selinv = false;
    double* d_frontB = nullptr;   // B * front_total (FP64 master)
    float* d_frontBf = nullptr;   // B * front_total (FP32 working, mixed factor)
    double* d_yB = nullptr;       // B * n (solve working vector)
    int* d_sing = nullptr;
    void* stream = nullptr;
    void* factor_graph_exec = nullptr;
    void* solve_graph_exec = nullptr;
    BatchedState() = default;
    ~BatchedState();
    BatchedState(const BatchedState&) = delete;
    BatchedState& operator=(const BatchedState&) = delete;
};

// Allocate batched arenas and capture the batched factor + solve CUDA graphs for B systems.
bool batched_setup(const custom_linear_solver::plan::MultifrontalPlan& plan, int B, bool fp32,
                   BatchedState& state);

// Factorize all B systems. d_valuesB is B*nnz_a FP64 CSR values (batch b at b*nnz_a).
bool batched_factorize(const custom_linear_solver::plan::MultifrontalPlan& plan,
                       BatchedState& state, const double* d_valuesB,
                       const int* d_ordered_value_to_csr, double* kernel_ms = nullptr);

// Solve all B systems. d_rhsB / d_solB are B*n (batch b at b*n). d_perm is shared.
bool batched_solve(const custom_linear_solver::plan::MultifrontalPlan& plan, BatchedState& state,
                   const double* d_rhsB, double* d_solB, const int* d_perm,
                   double* kernel_ms = nullptr);

}  // namespace custom_linear_solver::factorize
