#pragma once

#include <memory>

#include "matrix/view.hpp"

namespace custom_linear_solver {

enum class Precision;  // FP64 / FP32 / TC (defined in multifrontal.hpp)

enum class Status {
    Success,
    InvalidValue,
    InvalidState,
    AllocationFailed,
    AnalysisFailed,
    FactorizationFailed,
    SolveFailed,
};

struct SolverConfig {
    bool use_matching = false;
    bool use_parallel_nested_dissection = true;
    bool enable_shift_retry = true;
    double shift_retry_epsilon = 1.0e-8;
    int panel_cap = 8;
    Precision precision = static_cast<Precision>(0);  // FP64 by default (enum value 0)
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
    Status set_values(const void* values, ValueType value_type = ValueType::Float64);
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
