#pragma once

#include <vector>

#include "mysolver/factorize/sparse_lu.hpp"
#include "mysolver/symbolic/supernode.hpp"

// Supernodal panel structure (PLAN §M3 factorize_dense_block / driver).
// Each supernode becomes one dense panel: its columns plus the below-diagonal
// rows they share. The numeric factorization (next cycle) factors each panel's
// diagonal block (dense LU) and applies TRSM/GEMM to the off-diagonal rows and
// the Schur updates — the dense BLAS / cuBLAS unit of work.
namespace mysolver::numeric {

struct SupernodalStruct {
    int n = 0;
    int num_supernodes = 0;
    std::vector<int> sn_first;     // first column of supernode s
    std::vector<int> sn_size;      // number of columns in supernode s
    std::vector<int> panel_ptr;    // CSR-like offsets into panel_rows (size S+1)
    std::vector<int> panel_rows;   // sorted rows of each panel (diagonal block first)
};

// Build the panel structure from the fill pattern (Lp/Li, lower incl diagonal)
// and the fundamental supernode partition. The panel rows of a supernode are the
// L-column structure of its first (largest-structure) column.
SupernodalStruct build_supernodal(int n, const std::vector<int>& Lp, const std::vector<int>& Li,
                                  const symbolic::SupernodePartition& sp);

// Right-looking supernodal no-pivot factorization producing the same L/U as
// factor_nopiv (in the symmetric fill `out`), but factoring each supernode as a
// dense panel (diagonal-block LU + TRSM panels + Schur GEMM). The matrix must be
// in postorder (supernodes contiguous), matching `ss`. Returns false on a zero
// pivot. Dense block ops are scalar here; a BLAS/cuBLAS swap is the speed step.
bool factor_supernodal(int n, const int* Ap, const int* Ai, const double* Ax,
                       const std::vector<int>& Lp, const std::vector<int>& Li,
                       const SupernodalStruct& ss, SparseLU& out);

}  // namespace mysolver::numeric
