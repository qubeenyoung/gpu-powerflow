#pragma once

#include <string>
#include <vector>

#include "matrix/sparse_matrix.hpp"

namespace sparse_direct::solver {

struct SolverRun {
    bool success = false;
    std::string message;
    std::vector<matrix::Value> x;
    double analysis_ms = 0.0;
    double factor_ms = 0.0;
    double solve_ms = 0.0;
};

class LinearSolver {
public:
    virtual ~LinearSolver() = default;

    virtual std::string name() const = 0;
    virtual SolverRun solve(
        const matrix::CsrMatrix& csr,
        const matrix::CscMatrix& csc,
        const std::vector<matrix::Value>& b) = 0;
};

}  // namespace sparse_direct::solver
