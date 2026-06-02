#pragma once

#include <memory>

#include "matrix/view.hpp"

namespace custom_linear_solver {

namespace factorize { enum class BatchPrecision; }  // FP64 / FP32 / Mixed / TC (defined in batched hdr)

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
};

class Solver {
public:
    explicit Solver(const SolverConfig& config = {});
    ~Solver();

    Solver(Solver&&) noexcept;
    Solver& operator=(Solver&&) noexcept;
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    Status set_data(const CsrMatrixView& matrix);
    Status set_rhs(const DenseVectorView& rhs);
    Status set_solution(const DenseVectorView& solution);

    Status get_data(CsrMatrixView* matrix) const;
    Status get_rhs(DenseVectorView* rhs) const;
    Status get_solution(DenseVectorView* solution) const;

    Status analyze();
    Status factorize(double* kernel_ms = nullptr);
    Status solve(double* kernel_ms = nullptr);

    // Uniform-batch path (research): B systems sharing this analyzed sparsity pattern.
    // setup once after analyze(); then factorize/solve all B at once. Device buffers are
    // batch-strided: valuesB[b*nnz + .], rhsB[b*n + .], solB[b*n + .].
    Status batched_setup(int batch, factorize::BatchPrecision prec);
    Status batched_factorize(const double* d_valuesB, double* kernel_ms = nullptr);
    Status batched_solve(const double* d_rhsB, double* d_solB, double* kernel_ms = nullptr);

    // FP32-input overloads for callers whose Jacobian/step buffers are single precision (e.g.
    // cuPF's Mixed profile: float Jacobian, double RHS, float step). Values are scattered
    // straight into the factor working buffers (no FP64 staging); the internal solve vector
    // stays FP64 for accuracy.
    Status batched_factorize(const float* d_valuesB, double* kernel_ms = nullptr);
    Status batched_solve(const double* d_rhsB, float* d_solB, double* kernel_ms = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char* status_string(Status status);

}  // namespace custom_linear_solver
