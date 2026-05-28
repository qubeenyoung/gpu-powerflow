#pragma once

#include <vector>

#include "matrix/sparse_matrix.hpp"

// Full mysolver-own numeric pipeline (no KLU), PLAN §M3.
// AMD ordering -> symbolic fill pattern -> no-pivot sparse LU -> permuted solve.
// Depends only on the symbolic stack + sparse_lu + SuiteSparse AMD (no KLU,
// METIS, or CXSparse), so it can be exercised in isolation while the production
// solver keeps its KLU fallback.
namespace mysolver {

// Per-phase timing of the own pipeline (cuDSS phase contract).
struct OwnSolveStats {
    double analysis_ms = 0.0;  // scaling + matching + AMD order + symbolic fill
    double factor_ms = 0.0;    // numeric factorization
    double solve_ms = 0.0;     // triangular solve + iterative refinement
};

// Reusable analysis (matching + fill-reducing order + symbolic fill pattern) for
// the repeated-factorization regime: in a Newton-Raphson power-flow loop the
// sparsity is fixed and only the values change, so this expensive pattern work
// (the large-matrix bottleneck, cycle 52) is computed once and reused.
struct OwnAnalysis {
    int n = 0;
    std::vector<int> match;   // column permutation: zero-free diagonal
    std::vector<int> perm;    // AMD fill-reducing order on the matched matrix
    std::vector<int> Lp, Li;  // symbolic fill pattern of L
};

// Solve A x = b with the own pipeline. Returns true on a finite solution; the
// caller is responsible for checking the backward error (no-pivot LU is only
// stable for SPD / diagonally-dominant systems until matching + scaling +
// iterative refinement land). Optional `stats` receives the phase timings.
// Optional `save` captures the reusable analysis for later own_refactor calls.
bool try_own_solve(const sparse_direct::matrix::CscMatrix& A,
                   const std::vector<double>& b, std::vector<double>& x_out,
                   OwnSolveStats* stats = nullptr, OwnAnalysis* save = nullptr);

// Re-solve a matrix with the SAME sparsity but new values, reusing `an` (skips
// matching + AMD + symbolic fill). For the NR refactorization loop. Recomputes
// equilibration scaling from the new values. False on a zero pivot.
bool own_refactor(const sparse_direct::matrix::CscMatrix& A, const OwnAnalysis& an,
                  const std::vector<double>& b, std::vector<double>& x_out,
                  OwnSolveStats* stats = nullptr);

}  // namespace mysolver
