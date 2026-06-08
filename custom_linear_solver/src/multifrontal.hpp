#pragma once

#include "plan/multifrontal_plan.hpp"

// Uniform-batch multifrontal factorize + solve. B linear systems share a single sparsity
// pattern (and thus the same symbolic plan) but have independent numeric values / RHS. The
// symbolic analysis (analyze_multifrontal) runs once and is shared; only the numeric factor
// and solve are batched.
//
// Layout: FRONT-MAJOR. The front arena is B * front_total elements; for batch b and panel p,
// the front starts at b * front_total + front_off[p]. Each dense per-front kernel uses
// blockIdx.y = batch, so one launch per etree level covers all B fronts. The launch and
// per-level synchronization latency that dominates a B = 1 path is amortized across the
// systems, and the B * level_size blocks fill the GPU on otherwise occupancy-starved levels.

namespace custom_linear_solver {

// Numeric precision modes for the factor / solve. The solve reads the front the factor produced.
//
//   FP64       – everything double. Reference accuracy (~1e-13), slowest factor.
//   FP32       – the whole front is float. ~1e-4 accurate, ~2x faster than FP64 on RTX 3090.
//   FP16       – FP32 front + FP16 PTX mma.m16n8k8 trailing GEMM with FP32 accumulate
//                (per-lane register drain, no Csc readback; big fronts use
//                __launch_bounds__(512, 2)). Accuracy tracks FP32 except for the FP16 rounding
//                of trailing contributions.
//   TF32       – FP32 front + TF32 PTX mma.m16n8k8 trailing GEMM + per-level k4/k8 hybrid for
//                mid fronts + __launch_bounds__(512, 2) for big fronts (docs/15 V9h + docs/17
//                EXP-B). Recommended path on power-grid Jacobians.
//
// FP16 and TF32 require Ampere (sm80+).
enum class Precision { FP64, FP32, FP16, TF32 };

// Small utilities used by both the runtime wrapper (multifrontal.cu) and the dispatchers
// (factorize/dispatch.cuh, solve/dispatch.cuh). Defined here so there is one canonical copy.
inline bool is_fp32_front(Precision p)
{
    return p == Precision::FP32 || p == Precision::FP16 || p == Precision::TF32;
}

inline bool is_tf32_path(Precision p)
{
    return p == Precision::TF32;
}

// Per-batch runtime state. Owned by Solver; rebuilt by setup() each time batch_size or
// precision changes. The kernel launches in multifrontal.cu read State fields directly.
struct State {
    int B = 0;
    long front_total = 0;
    int n = 0;
    Precision prec = Precision::FP64;
    bool tier_split = true;  // occupancy-gated per-front tier split in factor/solve dispatch
    double* d_frontB  = nullptr;   // B * front_total FP64 front (FP64 mode only)
    float*  d_frontBf = nullptr;   // B * front_total FP32 front (FP32 / FP16 / TF32 modes)
    double* d_yB  = nullptr;       // B * n FP64 solve working vector (FP64 mode)
    float*  d_yBf = nullptr;       // B * n FP32 solve working vector (FP32 / FP16 / TF32 modes)
    int* d_sing = nullptr;
    void* stream = nullptr;        // internal-graph mode: solver-owned; external: caller-owned
    bool  owns_stream = false;     // true only when this State created `stream`
    void* factor_graph_exec = nullptr;
    void* solve_graph_exec  = nullptr;
    // Multi-stream subtree dispatch: each independent subtree gets its own stream (capped at 8),
    // joined before the spine levels which run on the main stream.
    int num_subtree_streams = 0;
    void* subtree_streams[8] = {nullptr};
    void* fork_event = nullptr;
    void* join_events[8] = {nullptr};
    State() = default;
    ~State();
    State(const State&) = delete;
    State& operator=(const State&) = delete;
};

// Allocate the runtime arenas for B systems and (in internal-graph mode) capture the factor +
// solve kernel sequences into replayable CUDA graphs. In external mode (CLS_INTERNAL_GRAPH
// off) this only allocates; the caller must also call set_stream() so factor / solve issue
// onto its capturable stream.
//
// `use_multistream_subtrees` (default true): when the plan has 2..8 independent subtrees,
// dispatch each on its own CUDA stream. Set false to force single-stream dispatch (debugging /
// reproducibility); when there is only one subtree the flag has no effect.
bool setup(const plan::MultifrontalPlan& plan, int B, Precision prec, State& state,
           bool use_multistream_subtrees = true, bool tier_split = true);

// Bind a caller-owned cudaStream_t (passed as void*). External / capturable mode only.
void set_stream(State& state, void* stream);

// Factor / solve. d_valuesB has B * nnz_a entries, d_rhsB / d_solB have B * n entries.
// d_ordered_value_to_csr maps the analyze-time CSR ordering to the symbolic value slots;
// d_perm is the symmetric permutation produced by analyze.
bool factorize(const plan::MultifrontalPlan& plan, State& state,
               const double* d_valuesB, const int* d_ordered_value_to_csr);
bool solve(const plan::MultifrontalPlan& plan, State& state,
           const double* d_rhsB, double* d_solB, const int* d_perm);

// FP32-input overloads. The factor reads an FP32 value array straight into the working buffers
// (no FP64 staging). The solve overloads cover combinations of (FP64 or FP32) RHS and
// (FP64 or FP32) solution buffer.
bool factorize(const plan::MultifrontalPlan& plan, State& state,
               const float* d_valuesB, const int* d_ordered_value_to_csr);
bool solve(const plan::MultifrontalPlan& plan, State& state,
           const double* d_rhsB, float* d_solB, const int* d_perm);
bool solve(const plan::MultifrontalPlan& plan, State& state,
           const float* d_rhsB, float* d_solB, const int* d_perm);

}  // namespace custom_linear_solver
