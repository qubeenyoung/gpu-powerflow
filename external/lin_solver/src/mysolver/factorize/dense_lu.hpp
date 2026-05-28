#pragma once

#include <vector>

// Dense block LU with partial pivoting (PLAN §M3 factorize_dense_block).
// This is the per-supernode numeric kernel; the supernodal driver applies it to
// each supernode's diagonal block and propagates updates. CPU reference first
// (correctness), then a cuBLAS GEMM/TRSM GPU path in a later cycle.
namespace mysolver::numeric {

// In-place LU with partial (row) pivoting of an n x n column-major matrix `a`
// (a[i + j*n]). On success `a` holds the unit-lower L (below diagonal) and U
// (on/above diagonal); `piv[k]` is the row swapped to position k at step k.
// Returns false if the matrix is numerically singular.
bool dense_lu(int n, std::vector<double>& a, std::vector<int>& piv);

// Solve A x = b in place on `x` (= b on entry) using factors from dense_lu.
void dense_lu_solve(int n, const std::vector<double>& lu, const std::vector<int>& piv,
                    std::vector<double>& x);

}  // namespace mysolver::numeric
