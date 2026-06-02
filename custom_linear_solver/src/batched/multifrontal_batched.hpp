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
namespace custom_linear_solver::batched {

// The four factorization precision modes. The batched solve always reads the front the factor
// produced (FP64 front for FP64; FP32 front for FP32; the FP64 master for Mixed/TC):
//   FP64  - everything double. Reference accuracy (~1e-13), slowest factor.
//   FP32  - the whole front is float (no FP64 master). Fastest/least accurate (~1e-4..1e-2).
//   Mixed - FP64 master front (precise assembly/solve) + FP32 working LU. GA102 FP64 is 1/64
//           FP32, so the LU bulk is far cheaper while pivots/assembly stay precise (~1e-5..1e-3).
//   TC    - like Mixed but the dense trailing update is an FP16 tensor-core (WMMA) GEMM; needs
//           the deep-K amalgamation (Solver::analyze) to be worthwhile (~1e-3..1e-1).
enum class BatchPrecision { FP64, FP32, Mixed, TC };

struct BatchedState {
    int B = 0;
    long front_total = 0;
    int n = 0;
    BatchPrecision prec = BatchPrecision::FP64;
    bool selinv = false;
    double* d_frontB = nullptr;   // B * front_total FP64 front (FP64 mode) or FP64 master (Mixed/TC)
    float* d_frontBf = nullptr;   // B * front_total FP32 front (FP32 mode) or FP32 working (Mixed/TC)
    double* d_yB = nullptr;       // B * n FP64 solve working vector (FP64/Mixed/TC modes)
    float* d_yBf = nullptr;       // B * n FP32 solve working vector (pure-FP32 mode)
    int* d_sing = nullptr;
    void* stream = nullptr;       // internal-graph mode: solver-owned; external mode: caller's
    bool owns_stream = false;     // true only when the solver created `stream` (internal-graph mode)
    void* factor_graph_exec = nullptr;
    void* solve_graph_exec = nullptr;
    BatchedState() = default;
    ~BatchedState();
    BatchedState(const BatchedState&) = delete;
    BatchedState& operator=(const BatchedState&) = delete;
};

// Allocate batched arenas and (internal-graph mode only) capture the factor + solve CUDA graphs
// for B systems. In external mode (CLS_INTERNAL_GRAPH off) this only allocates; the caller must
// also call batched_set_stream() so factorize/solve issue onto its (capturable) stream.
bool batched_setup(const custom_linear_solver::plan::MultifrontalPlan& plan, int B,
                   BatchPrecision prec, BatchedState& state);

// External/capturable mode: adopt a caller-provided CUDA stream (cudaStream_t as void*) for the
// factor/solve kernel launches, so they can be recorded by an outer stream capture. Not owned by
// the state (the caller keeps lifetime). No-op effect on the internal-graph mode's private stream.
void batched_set_stream(BatchedState& state, void* stream);

// Factorize all B systems. d_valuesB is B*nnz_a FP64 CSR values (batch b at b*nnz_a).
bool batched_factorize(const custom_linear_solver::plan::MultifrontalPlan& plan,
                       BatchedState& state, const double* d_valuesB,
                       const int* d_ordered_value_to_csr, double* kernel_ms = nullptr);

// Solve all B systems. d_rhsB / d_solB are B*n (batch b at b*n). d_perm is shared.
bool batched_solve(const custom_linear_solver::plan::MultifrontalPlan& plan, BatchedState& state,
                   const double* d_rhsB, double* d_solB, const int* d_perm,
                   double* kernel_ms = nullptr);

// FP32-input overloads. The factor reads an FP32 CSR value array (b*nnz_a) straight into the
// front working buffers; the solve reads an FP64 RHS but writes an FP32 step (cuPF's Mixed
// profile: float Jacobian + step, double residual). The internal solve vector stays FP64.
bool batched_factorize(const custom_linear_solver::plan::MultifrontalPlan& plan,
                       BatchedState& state, const float* d_valuesB,
                       const int* d_ordered_value_to_csr, double* kernel_ms = nullptr);
bool batched_solve(const custom_linear_solver::plan::MultifrontalPlan& plan, BatchedState& state,
                   const double* d_rhsB, float* d_solB, const int* d_perm,
                   double* kernel_ms = nullptr);
bool batched_solve(const custom_linear_solver::plan::MultifrontalPlan& plan, BatchedState& state,
                   const float* d_rhsB, float* d_solB, const int* d_perm,
                   double* kernel_ms = nullptr);

}  // namespace custom_linear_solver::batched
