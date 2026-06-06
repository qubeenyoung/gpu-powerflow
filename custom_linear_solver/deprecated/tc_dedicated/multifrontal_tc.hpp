#pragma once

#include <vector>

#include "plan/multifrontal_plan.hpp"

// TC-dedicated multifrontal factorize + solve. Mirrors the batched API
// (multifrontal_batched.hpp) but with a hard-coded TC32 precision mode (float front + FP16 WMMA
// trailing) and TC-friendly dispatch thresholds.
//
// Differences from `batched::batched_setup(..., BatchPrecision::TC32)`:
//   Phase 1 (this revision): API + state isolation only. Dispatch reuses batched kernels with
//                            the same TC32 path. Measurement baseline.
//   Phase 2 (planned):       Lower the TC-trailing fsz threshold so fsz 32–48 fronts also go
//                            through WMMA (currently only fsz>=49 in mid_tc32_b<true>).
//   Phase 3 (planned):       Deep-tail dense LU absorption (L17–L29 collapse).
//
// Precision: FP32 front, FP16 WMMA trailing inputs, FP32 accumulate. Accuracy ~1e-3 (TC32 range,
// recover via IR if the app needs lower residual).

namespace custom_linear_solver::tc {

struct TCState {
    int B = 0;
    long front_total = 0;
    int n = 0;
    bool selinv = false;
    float* d_frontBf = nullptr;   // B * front_total FP32 front
    float* d_yBf = nullptr;       // B * n FP32 solve working vector
    int* d_sing = nullptr;
    void* stream = nullptr;
    bool owns_stream = false;
    void* factor_graph_exec = nullptr;
    void* solve_graph_exec = nullptr;
    // Phase 3 — K subtree streams + fork-join events. Allocated in tc_setup if K > 1 and
    // CLS_NO_MULTISTREAM is not set. Each subtree's factor work runs on its own stream so the
    // CUDA Graph captures parallel branches.
    int num_subtree_streams = 0;
    void* subtree_streams[8] = {nullptr};   // up to 8 subtrees (case8387/USA = 2)
    void* fork_event = nullptr;
    void* join_events[8] = {nullptr};
    // Phase Σ.6 / Σ.7 — cuBLAS handle + pre-built grouped-batched dispatch arrays. Handle and
    // arrays are created unconditionally during tc_setup so the CUDA Graph capture sees fully
    // initialized state (no runtime fast path branches). CLS_USE_CUBLAS=1 picks the cuBLAS
    // trailing instead of the in-kernel WMMA / staged-scalar trailing.
    //
    // Layout: per-panel arrays indexed by plcols position (not panel id), since the dispatch
    // walks plcols ranges by level. Per (panel-position, batch) pointer triple = (U, L, C)
    // base in the FP32 front arena. Per-level cublasSgemmGroupedBatched call passes slices
    // [plptr[L]..plptr[L+1]) of these arrays.
    //
    // Capture safety: scalar arrays live on host inside this struct (vectors held for the
    // lifetime of TCState); pointer arrays live on device. cublasSgemmGroupedBatched is
    // captured as part of the factor graph at tc_setup time.
    void* cublas_handle = nullptr;
    float** d_Aptrs = nullptr;   // P*B device pointers, U_base for (plcols_pos, batch)
    float** d_Bptrs = nullptr;   // P*B device pointers, L_base
    float** d_Cptrs = nullptr;   // P*B device pointers, C_base (trailing)
    std::vector<int> h_m;        // P entries: uc per plcols_pos
    std::vector<int> h_n;        // P entries: uc per plcols_pos
    std::vector<int> h_k;        // P entries: nc per plcols_pos
    std::vector<int> h_lda;      // P entries: fsz per plcols_pos
    std::vector<int> h_gsize;    // P entries: B per plcols_pos (group_size for grouped batched)
    std::vector<int> h_transa;   // P entries: CUBLAS_OP_N cast to int (header-only friendly)
    std::vector<float> h_alpha;  // P entries: -1.0f
    std::vector<float> h_beta;   // P entries: 1.0f

    // Phase Σ.8 — per-(system, panel-pivot-row) within-panel pivot storage. Layout:
    // d_pivotsB[bb * total_pivots + pivot_offset[p] + k] = swap target row at LU step k of
    // panel p in system bb. CLS_USE_PIVOTING=1 routes factor through *_pp kernels that fill
    // this; forward solve then reads it to swap RHS before triangular substitution.
    int* d_pivotsB = nullptr;
    bool use_pivoting = false;
    TCState() = default;
    ~TCState();
    TCState(const TCState&) = delete;
    TCState& operator=(const TCState&) = delete;
};

// Process-level warmup: pre-create the cuBLAS handle so the first analyze cycle does NOT
// pay the ~10 ms cublasCreate driver-init. Call once at program start (e.g. in the cuPF /
// app initializer); subsequent tc_setup calls reuse the cached handle for free.
bool tc_warmup();

// Allocate TC arenas and (internal-graph mode only) capture factor + solve CUDA graphs.
bool tc_setup(const custom_linear_solver::plan::MultifrontalPlan& plan, int B, TCState& state);

// External/capturable mode: adopt a caller-provided CUDA stream.
void tc_set_stream(TCState& state, void* stream);

// Factorize all B systems. d_valuesB is B*nnz_a FP32 CSR values.
bool tc_factorize(const custom_linear_solver::plan::MultifrontalPlan& plan, TCState& state,
                  const float* d_valuesB, const int* d_ordered_value_to_csr,
                  double* kernel_ms = nullptr);

// Solve all B systems. d_rhsB / d_solB are B*n FP32 (batch b at b*n). d_perm is shared.
bool tc_solve(const custom_linear_solver::plan::MultifrontalPlan& plan, TCState& state,
              const float* d_rhsB, float* d_solB, const int* d_perm,
              double* kernel_ms = nullptr);

}  // namespace custom_linear_solver::tc
