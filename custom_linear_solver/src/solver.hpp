#pragma once

#include <memory>

#include "matrix/view.hpp"

namespace custom_linear_solver {

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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char* status_string(Status status);

}  // namespace custom_linear_solver
