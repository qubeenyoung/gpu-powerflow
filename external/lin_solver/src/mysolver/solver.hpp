#pragma once

#include <memory>
#include <vector>

#include "matrix/sparse_matrix.hpp"

// mysolver public API.
//
// The external call pattern is intentionally 1:1 with the cuDSS phase model
// (cf. cuDSS Getting Started "Workflow": analyze -> factorize -> solve).
// In a Newton-Raphson loop the Jacobian sparsity is fixed, so analyze() runs
// once and only factorize()/solve() repeat as values change.
//
// M0 milestone: the internal implementation delegates numeric work to
// SuiteSparse KLU (a faithful wrapper with the cuDSS I/O contract).  Later
// milestones replace each phase with native GPU implementations:
//   AnalyzeResult -> perm/iperm + elimination tree + supernode + schedule
//   FactorState   -> L/U numeric values + block metadata
namespace mysolver {

// ANALYSIS output. Reusable across many factorizations of the same pattern.
struct AnalyzeResult {
    std::shared_ptr<void> impl;
    bool valid() const { return static_cast<bool>(impl); }
};

// FACTORIZATION output. Recreated whenever the matrix values change.
struct FactorState {
    std::shared_ptr<void> impl;
    long lnz = 0;     // actual nonzeros in L (incl. diagonal), for fill-in tracking
    long unz = 0;     // actual nonzeros in U (incl. diagonal)
    int ordering = 0;  // ordering chosen by min-fill selection: 0 = AMD, 3 = METIS ND
    bool valid() const { return static_cast<bool>(impl); }
};

// Run once per sparsity pattern.
AnalyzeResult analyze(const sparse_direct::matrix::CscMatrix& A);

// Run once per Newton iteration (values of A change, pattern does not).
void factorize(const sparse_direct::matrix::CscMatrix& A,
               const AnalyzeResult& a,
               FactorState* out);

// Run once per right-hand side. Solves A x = rhs into x_out.
void solve(const FactorState& f,
           const AnalyzeResult& a,
           const std::vector<double>& rhs,
           std::vector<double>& x_out);

}  // namespace mysolver
