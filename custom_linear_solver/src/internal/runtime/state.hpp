#pragma once

#include <vector>

#include "internal/plan/multifrontal_plan.hpp"

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
//   TF32       – FP32 front + TF32 mma.m16n8k8 trailing GEMM. Small/big fronts use the shared-resident
//                blocked / panel-resident Tensor-Core kernels (factor_small / factor_big); large
//                fronts use the global-resident factor_large. Recommended path on power-grid Jacobians.
//
// TF32 requires Ampere (sm80+).
enum class Precision { FP64, FP32, TF32 };

// Small utilities used by both the runtime wrapper and the factor/solve dispatchers. Defined here
// so there is one canonical copy.
inline bool is_fp32_front(Precision p)
{
    return p == Precision::FP32 || p == Precision::TF32;
}

inline bool is_tf32_path(Precision p)
{
    return p == Precision::TF32;
}

// Per-batch runtime state. Owned by Solver; rebuilt by setup() each time batch_size or
// precision changes. The factor/solve kernel launches read State fields directly.
struct State {
    int batch_count = 0;
    long front_total = 0;
    int num_rows = 0;
    Precision precision = Precision::FP64;
    bool tier_split = true;  // occupancy-gated per-front tier split in factor/solve dispatch
    bool static_pivoting = false;
    double pivot_threshold = 0.0;
    double pivot_shift = 0.0;
    double* d_front_batch = nullptr;    // batch_count * front_total FP64 front (FP64 mode only)
    float*  d_front_batch_f = nullptr;  // batch_count * front_total FP32 front (FP32 / TF32)
    double* d_y_batch = nullptr;        // batch_count * num_rows FP64 solve vector (FP64 mode)
    float*  d_y_batch_f = nullptr;      // batch_count * num_rows FP32 solve vector (FP32/TF32)
    int* d_sing = nullptr;
    void* stream = nullptr;        // internal-graph mode: solver-owned; external: caller-owned
    bool  owns_stream = false;     // true only when this State created `stream`
    void* factor_graph_exec = nullptr;
    // Full solve graph (gather + solve levels + scatter), captured lazily by solve.cu and keyed by
    // the (rhs, solution, perm, iperm, type) it was recorded for; a key change recaptures it. This
    // supersedes the old solve-levels-only graph captured at setup.
    void* full_solve_graph_exec = nullptr;
    const void* full_solve_rhs = nullptr;
    void* full_solve_solution = nullptr;
    const int* full_solve_perm = nullptr;
    const int* full_solve_iperm = nullptr;
    int full_solve_type_tag = 0;
    // Optional cuBLAS TF32 grouped-batched trailing path. Pointer arrays are device arrays
    // indexed by the active panel order position q, then batch b: slot = q * B + b.
    void* cublas_handle = nullptr;
    float** d_cublas_Aptrs = nullptr;
    float** d_cublas_Bptrs = nullptr;
    float** d_cublas_Cptrs = nullptr;
    float** d_cublas_Aptrs_tier = nullptr;
    float** d_cublas_Bptrs_tier = nullptr;
    float** d_cublas_Cptrs_tier = nullptr;
    std::vector<int> cublas_m;
    std::vector<int> cublas_n;
    std::vector<int> cublas_k;
    std::vector<int> cublas_lda;
    std::vector<int> cublas_m_tier;
    std::vector<int> cublas_n_tier;
    std::vector<int> cublas_k_tier;
    std::vector<int> cublas_lda_tier;
    std::vector<int> cublas_group_size;
    std::vector<int> cublas_trans;
    std::vector<float> cublas_alpha;
    std::vector<float> cublas_beta;
    // Multi-stream subtree dispatch: each independent subtree gets its own stream, joined before
    // the spine levels which run on the main stream.
    int num_subtree_streams = 0;
    void* subtree_streams[kMaxSubtreeStreams] = {nullptr};
    void* fork_event = nullptr;
    void* join_events[kMaxSubtreeStreams] = {nullptr};
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
bool setup(const plan::MultifrontalPlan& plan, int B, Precision precision, State& state,
           bool use_multistream_subtrees = true, bool tier_split = true,
           bool static_pivoting = false, double pivot_threshold = 0.0,
           double pivot_shift = 0.0);

// Bind a caller-owned cudaStream_t (passed as void*). External / capturable mode only.
void set_stream(State& state, void* stream);
}  // namespace custom_linear_solver
