#pragma once

#include <memory>
#include <string>

#include "internal/matrix_view.hpp"
#include "internal/runtime/state.hpp"  // Precision { FP64, FP32, TF32 }

namespace custom_linear_solver {

enum class MatchingMode {
  None,
  Structural,
};

enum class PivotStrategy {
  None,
  StaticDiagonalShift,
  DynamicPartial,
};

enum class Status {
  kSuccess,
  kInvalidValue,
  kInvalidState,
  kAllocationFailed,
  kAnalysisFailed,
  kFactorizationFailed,
  kSolveFailed,
};

// User-configurable knobs. Set on the SolverConfig passed to Solver(). Tunables
// not listed here (the three front-size tier boundaries, the TF32 PTX trailing
// stack, the cp.async stage-in, the per-front kernel routing) are baked into
// the build because every measured off-default regressed at least one case.
struct SolverConfig {
  // ---- Symbolic analysis ----
  bool use_matching = false;  // compatibility alias for Structural matching
  MatchingMode matching =
      MatchingMode::None;  // optional row/column structural matching
  bool use_parallel_nested_dissection =
      true;                 // multi-threaded METIS-ND ordering
  int max_panel_width = 8;  // max columns amalgamated into one supernode
                            // panel (1..64). Width 8 is the fair-tuned
                            // optimum for all sizes (report §08, 2026-06-17).
  // ---- Numeric factorization ----
  bool enable_shift_retry =
      true;  // enables StaticDiagonalShift compatibility path
  double shift_retry_epsilon =
      1.0e-8;  // pivot threshold and replacement magnitude
  PivotStrategy pivot_strategy = PivotStrategy::StaticDiagonalShift;
  Precision precision =
      Precision::FP64;  // FP64 / FP32 / TF32 (TF32 Ozaki mma, recommended)
  // ---- Analyze-phase diagnostics (off by default) ----
  std::string analyze_dump_fronts_path;  // non-empty → write per-front CSV
                                         // after Analyze()
  bool analyze_fronts_only =
      false;  // build symbolic fronts/levels and dump CSV
              // without allocating numeric front arenas;
              // only valid for Analyze-only tooling.
  bool analyze_emit_info =
      false;  // print front-size and subtree summary to stderr
};

// Phase API. Typical sequence:
//
//   SetData(...)        // register A (sparsity + values pointer)
//   SetRhs(...)         // register b
//   SetSolution(...)    // register x
//   Analyze()            // one-time symbolic + plan build
//   Setup(B = 1)         // allocate per-system runtime state for B systems
//   Factorize()          // numeric factor using the registered values
//   Solve()              // y = A^{-1} * b using registered rhs / solution
//
// For B > 1 the registered buffers are batch-strided: values[b*nnz + ·],
// rhs[b*n + ·], solution[b*n + ·]. Caller measures kernel time externally
// (cudaEvent or wall clock).
class Solver {
 public:
  explicit Solver(const SolverConfig& config = {});
  ~Solver();

  Solver(Solver&&) noexcept;
  Solver& operator=(Solver&&) noexcept;
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  Status SetData(const CsrMatrixView& matrix);
  Status SetValues(const void* values,
                   ValueType value_type = ValueType::kFloat64);
  Status SetRhs(const DenseVectorView& rhs);
  Status SetSolution(const DenseVectorView& solution);

  Status get_data(CsrMatrixView* matrix) const;
  Status get_rhs(DenseVectorView* rhs) const;
  Status get_solution(DenseVectorView* solution) const;

  Status Analyze();
  Status Setup(int batch_size = 1);
  Status Factorize();
  Status Solve();

  // External / capturable mode (CLS_INTERNAL_GRAPH off in CMake): bind a
  // caller-owned cudaStream_t (passed as void*) so factor / Solve issue their
  // kernels onto it for an outer stream capture. Call after Setup(), before
  // Factorize() / Solve().
  Status SetStream(void* stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

const char* StatusString(Status status);

}  // namespace custom_linear_solver
