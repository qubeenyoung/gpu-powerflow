#pragma once

#include <vector>

// First GPU building block of the cuDSS-chasing research track (PLAN §M3/§M5).
// A CSR sparse mat-vec on the device — the simplest sparse-GPU kernel, used to
// validate the CUDA build/data pipeline and as a primitive for residual/
// refinement and the upcoming level-set GPU factorization/solve kernels.
namespace mysolver::gpu {

// y = A x for an n x n CSR matrix, computed on the GPU. Returns y (size n).
std::vector<double> csr_spmv(int n, const std::vector<int>& row_ptr,
                             const std::vector<int>& col_idx,
                             const std::vector<double>& values,
                             const std::vector<double>& x);

}  // namespace mysolver::gpu
