#pragma once

#include <memory>
#include <string>

#include "internal/matrix_view.hpp"
#include "internal/runtime/state.hpp"   // Precision { FP64, FP32, TF32 }

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

// User-configurable knobs. Set on the SolverConfig passed to Solver(). Tunables not listed
// here (the three front-size tier boundaries, the TF32 PTX trailing stack, the cp.async stage-in,
// the per-front kernel routing) are baked into the build because every measured off-default
// regressed at least one case.
struct SolverConfig {
    // ---- Symbolic analysis ----
    bool use_matching = false;                       // compatibility alias for Structural matching
    MatchingMode matching = MatchingMode::None;      // optional row/column structural matching
    bool use_parallel_nested_dissection = true;      // multi-threaded METIS-ND ordering
    int  metis_seed = 42;                            // METIS ordering seed (diagnostic / A-B)
    int  max_panel_width = 8;                        // max columns amalgamated into one supernode
                                                     // panel (1..64). Width 8 is the fair-tuned
                                                     // optimum for all sizes (report §08, 2026-06-17);
                                                     // analyzer uses this value directly unless
                                                     // CLS_RESPECT_PANEL_CAP changes clamp behavior.
    // ---- Numeric factorization ----
    bool enable_shift_retry = true;                  // enables StaticDiagonalShift compatibility path
    double shift_retry_epsilon = 1.0e-8;             // pivot threshold and replacement magnitude
    PivotStrategy pivot_strategy = PivotStrategy::StaticDiagonalShift;
    Precision precision = Precision::FP64;           // FP64 / FP32 / TF32 (TF32 PTX mma, recommended)
    // ---- Runtime dispatch ----
    bool tier_split = true;                          // deterministic per-tier dispatch in the
                                                     // factor/solve level walk (each front -> its
                                                     // tier's dedicated kernel). false = debug
                                                     // whole-level dispatch (every front promoted to
                                                     // the level's largest-tier kernel).
    bool one_block_per_front = false;                // STRUMPACK-style factor dispatch: ignore
                                                     // front-size tiers and factor each level with
                                                     // operation-separated global kernels using one
                                                     // CUDA block per (front, batch).
    bool use_multistream_subtrees = true;            // dispatch independent subtrees on
                                                     // separate streams (capped at 8). Set
                                                     // false for single-stream debugging.
    // ---- Analyze-phase diagnostics (off by default) ----
    std::string analyze_dump_fronts_path;            // non-empty → write per-front CSV here
                                                     // after analyze() (replaces CLS_DUMP_FRONTS)
    bool analyze_emit_info = false;                  // print front-size and subtree summary
                                                     // to stderr (replaces CLS_DUMP / CLS_TREE_INFO)
};

// Phase API. Typical sequence:
//
//   set_data(...)        // register A (sparsity + values pointer)
//   set_rhs(...)         // register b
//   set_solution(...)    // register x
//   analyze()            // one-time symbolic + plan build
//   setup(B = 1)         // allocate per-system runtime state for B systems
//   factorize()          // numeric factor using the registered values
//   solve()              // y = A^{-1} * b using registered rhs / solution
//
// For B > 1 the registered buffers are batch-strided: values[b*nnz + ·], rhs[b*n + ·],
// solution[b*n + ·]. Caller measures kernel time externally (cudaEvent or wall clock).
class Solver {
public:
    explicit Solver(const SolverConfig& config = {});
    ~Solver();

    Solver(Solver&&) noexcept;
    Solver& operator=(Solver&&) noexcept;
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    Status set_data(const CsrMatrixView& matrix);
    Status set_values(const void* values, ValueType value_type = ValueType::kFloat64);
    Status set_rhs(const DenseVectorView& rhs);
    Status set_solution(const DenseVectorView& solution);

    Status get_data(CsrMatrixView* matrix) const;
    Status get_rhs(DenseVectorView* rhs) const;
    Status get_solution(DenseVectorView* solution) const;

    Status analyze();
    Status setup(int batch_size = 1);
    Status factorize();
    Status solve();

    // External / capturable mode (CLS_INTERNAL_GRAPH off in CMake): bind a caller-owned
    // cudaStream_t (passed as void*) so factor / solve issue their kernels onto it for an
    // outer stream capture. Call after setup(), before factorize() / solve().
    Status set_stream(void* stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char* status_string(Status status);

}  // namespace custom_linear_solver
