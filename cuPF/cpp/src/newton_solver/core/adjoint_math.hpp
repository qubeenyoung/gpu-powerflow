#pragma once

#include "newton_solver/core/csr_transpose.hpp"
#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Pure numeric helpers for the implicit power-flow backward (adjoint) pass.
// Kept separate from the per-backend pipeline orchestration in
// newton_solver_adjoint.cpp so each file carries a single responsibility.

// Validate the public solve_adjoint() arguments against the stored forward
// state. Throws std::runtime_error / std::invalid_argument on mismatch.
void validate_adjoint_args(int32_t n_bus,
                           int32_t dimF,
                           int32_t stored_batch_size,
                           const double* grad_va,
                           int64_t grad_va_stride,
                           const double* grad_vm,
                           int64_t grad_vm_stride,
                           int32_t batch_size,
                           const int32_t* pv,
                           int32_t n_pv,
                           const int32_t* pq,
                           int32_t n_pq);

// Gather the dense per-bus voltage-angle/magnitude gradients into the packed
// [pv|pq angle | pq magnitude] state vector used as the adjoint RHS.
std::vector<double> build_grad_state(const double* grad_va,
                                     int64_t grad_va_stride,
                                     const double* grad_vm,
                                     int64_t grad_vm_stride,
                                     int32_t batch_size,
                                     const int32_t* pv,
                                     int32_t n_pv,
                                     const int32_t* pq,
                                     int32_t n_pq);

// Scatter the adjoint state (lambda) back into per-bus load gradients on
// result.grad_load_p / result.grad_load_q.
void project_load_gradients(const std::vector<double>& lambda,
                            int32_t n_bus,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            AdjointResult& result);

// ||J^T * lambda - rhs||_2 / max(||rhs||_2, eps) computed from a CSC-stored
// Jacobian (column-major), used as an adjoint solve residual diagnostic.
double relative_residual_norm_csc(const CpuJacobianMatrixF64& J,
                                  const std::vector<double>& lambda,
                                  const std::vector<double>& rhs);

// ||J^T * lambda - rhs||_2 / max(||rhs||_2, eps) computed from an explicit
// CSR transpose pattern with batched values.
template <typename T>
double relative_residual_norm_csr(const std::vector<int32_t>& row_ptr,
                                  const std::vector<int32_t>& col_idx,
                                  const std::vector<T>& values,
                                  const std::vector<double>& lambda,
                                  const std::vector<double>& rhs,
                                  int32_t batch_size,
                                  int32_t dim,
                                  int32_t nnz)
{
    long double residual_sq = 0.0L;
    long double rhs_sq = 0.0L;
    for (int32_t b = 0; b < batch_size; ++b) {
        const std::size_t dense_base = b * dim;
        const std::size_t sparse_base = b * nnz;
        for (int32_t row = 0; row < dim; ++row) {
            long double acc = 0.0L;
            for (int32_t k = row_ptr[row];
                 k < row_ptr[row + 1]; ++k) {
                const int32_t col = col_idx[k];
                acc += static_cast<long double>(values[sparse_base + k]) * static_cast<long double>(lambda[dense_base + col]);
            }
            const long double diff = acc - static_cast<long double>(rhs[dense_base + row]);
            residual_sq += diff * diff;
            const long double r = static_cast<long double>(rhs[dense_base + row]);
            rhs_sq += r * r;
        }
    }
    const long double denom = std::sqrt(std::max(rhs_sq, 1.0e-60L));
    return static_cast<double>(std::sqrt(residual_sq) / denom);
}
