#pragma once

#include <vector>

// Sparse LU on a static symmetric fill pattern (PLAN §M3 factorize_driver, CPU
// reference). Left-looking, no pivoting — correct for SPD / diagonally-dominant
// systems. General-matrix stability (matching + scaling + iterative refinement,
// the cuDSS-style path) is added in later cycles; this establishes mysolver's
// own end-to-end sparse factorize+solve without KLU.
namespace mysolver::numeric {

struct SparseLU {
    int n = 0;
    std::vector<int> Sp;     // symmetric filled pattern, column pointers (n+1)
    std::vector<int> Si;     // row indices (sorted per column)
    std::vector<double> x;   // values per position: L(i>j) multipliers, U(i<=j)
};

// Factor the structurally-symmetric matrix A (CSC: Ap/Ai/Ax, natural elimination
// order) into L/U on the fill pattern Lp/Li (lower incl. diagonal, from
// symbolic::fill_pattern). Returns false on a zero pivot.
bool factor_nopiv(int n, const int* Ap, const int* Ai, const double* Ax,
                  const std::vector<int>& Lp, const std::vector<int>& Li,
                  SparseLU& out);

// Multifrontal factorization (PLAN §M3 dense-panel large-path, CPU reference).
// Same no-pivot LU as factor_nopiv and same SparseLU output layout, but organised
// as relaxed dense panels: assemble each panel's front (A entries + children's
// contribution blocks), dense-factor its pivots, and extend-add its CB into the
// parent. Validates the dense-panel algorithm before the GPU port; the dense
// fronts are the BLAS/GPU-friendly unit of work the scattered scatter kernel lacks.
// `parent` is the elimination tree of the (already-ordered) matrix; panel_cap caps
// the amalgamated panel width. Returns false on a zero pivot.
bool multifrontal_factor(int n, const int* Ap, const int* Ai, const double* Ax,
                         const std::vector<int>& Lp, const std::vector<int>& Li,
                         const std::vector<int>& parent, SparseLU& out, int panel_cap = 16);

// Solve A y = b in natural order (no permutation), writing the solution to x_out.
void solve(const SparseLU& lu, const std::vector<double>& b, std::vector<double>& x_out);

}  // namespace mysolver::numeric
