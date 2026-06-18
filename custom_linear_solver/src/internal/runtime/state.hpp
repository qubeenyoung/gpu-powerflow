#pragma once

#include <vector>

#include "internal/plan/multifrontal_plan.hpp"

// Uniform-batch multifrontal Factorize + Solve. B linear systems share a single
// sparsity pattern (and thus the same symbolic plan) but have independent
// numeric values / RHS. The symbolic analysis (AnalyzeMultifrontal) runs once
// and is shared; only the numeric factor and Solve are batched.
//
// Layout: FRONT-MAJOR. The front arena is B * front_total elements; for batch b
// and panel p, the front starts at b * front_total + front_off[p]. Each dense
// per-front kernel uses blockIdx.y = batch, so one launch per Etree level
// covers all B fronts. The launch and per-level synchronization latency that
// dominates a B = 1 path is amortized across the systems, and the B *
// level_size blocks fill the GPU on otherwise occupancy-starved levels.

namespace custom_linear_solver {

// Numeric precision modes for the factor / Solve. The Solve reads the front the
// factor produced.
//
//   FP64       – everything double. Reference accuracy (~1e-13), slowest
//   factor. FP32       – the whole front is float. ~1e-4 accurate, ~2x faster
//   than FP64 on RTX 3090. TF32       – FP32 front + TF32 mma.m16n8k8 trailing
//   GEMM. Small/mid fronts use the warp-packed / whole-front shared
//                kernels (FactorSmall / FactorMid); big
//                fronts use the global-resident multi-block FactorBig.
//                Recommended path on power-grid Jacobians.
//
// TF32 requires Ampere (sm80+).
enum class Precision { FP64, FP32, TF32 };

// Small utilities used by both the runtime wrapper and the factor/Solve
// dispatchers. Defined here so there is one canonical copy.
inline bool IsFp32Front(Precision p) {
  return p == Precision::FP32 || p == Precision::TF32;
}

inline bool IsTf32Path(Precision p) { return p == Precision::TF32; }

// Per-batch runtime state. Owned by Solver; rebuilt by Setup() each time
// batch_size or precision changes. The factor/Solve kernel launches read State
// fields directly.
struct State {
  int batch_count = 0;
  long front_total = 0;
  int num_rows = 0;
  Precision precision = Precision::FP64;
  bool static_pivoting = false;
  double pivot_threshold = 0.0;
  double pivot_shift = 0.0;
  double* d_front_batch =
      nullptr;  // batch_count * front_total FP64 front (FP64 mode only)
  float* d_front_batch_f =
      nullptr;  // batch_count * front_total FP32 front (FP32 / TF32)
  double* d_y_batch =
      nullptr;  // batch_count * num_rows FP64 Solve vector (FP64 mode)
  float* d_y_batch_f =
      nullptr;  // batch_count * num_rows FP32 Solve vector (FP32/TF32)
  int* d_sing = nullptr;
  void* stream =
      nullptr;  // internal-graph mode: solver-owned; external: caller-owned
  bool owns_stream = false;  // true only when this State created `stream`
  void* factor_graph_exec = nullptr;
  // Full Solve graph (gather + Solve levels + scatter), captured lazily by
  // Solve.cu and keyed by the (rhs, solution, perm, iperm, type) it was
  // recorded for; a key change recaptures it. This supersedes the old
  // Solve-levels-only graph captured at Setup.
  void* full_solve_graph_exec = nullptr;
  const void* full_solve_rhs = nullptr;
  void* full_solve_solution = nullptr;
  const int* full_solve_perm = nullptr;
  const int* full_solve_iperm = nullptr;
  int FullSolveTypeTag = 0;
  // Multi-stream subtree dispatch (batched): each independent subtree gets its
  // own stream, joined before the spine levels which run on the main stream.
  int num_subtree_streams = 0;
  void* subtree_streams[kMaxSubtreeStreams] = {nullptr};
  void* fork_event = nullptr;
  void* join_events[kMaxSubtreeStreams] = {nullptr};
  State() = default;
  ~State();
  State(const State&) = delete;
  State& operator=(const State&) = delete;
};

// THE single dispatch decision point, shared by Factorize (schedule.cuh) and
// Solve (dispatch.cuh) so the two paths cannot drift: B == 1 takes the
// single-system schedule (fused / tensor-core factor + partitioned-inverse
// pivots + GEMV Solve); B > 1 takes the batched front-major schedule.
inline bool UseSingleSystem(const State& st) { return st.batch_count == 1; }

// Allocate the runtime arenas for B systems and (in internal-graph mode)
// capture the factor + Solve kernel sequences into replayable CUDA graphs. In
// external mode (CLS_INTERNAL_GRAPH off) this only allocates; the caller must
// also call SetStream() so factor / Solve issue onto its capturable stream.
// For B > 1 with 2..8 independent subtrees the factor/Solve fork onto
// per-subtree streams.
bool Setup(const plan::MultifrontalPlan& plan, int B, Precision precision,
           State& state, bool static_pivoting = false,
           double pivot_threshold = 0.0, double pivot_shift = 0.0);

// Bind a caller-owned cudaStream_t (passed as void*). External / capturable
// mode only.
void SetStream(State& state, void* stream);
}  // namespace custom_linear_solver
