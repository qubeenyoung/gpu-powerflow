// ---------------------------------------------------------------------------
// adjoint_math.cpp
//
// Backend-agnostic numeric helpers for the power-flow backward (adjoint) pass:
// input validation, packing the upstream voltage gradients into the dL/dx RHS,
// projecting the adjoint solution lambda onto load gradients, and a residual
// check. No CUDA / solver state here — pure host math, unit-test friendly.
// ---------------------------------------------------------------------------

#include "newton_solver/core/adjoint_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

// Validate caller-supplied adjoint arguments against the prepared solver state.
// Guards: positive/ matching batch size, non-null grads/index arrays, strides
// wide enough for n_bus, and dimF == n_pv + 2*n_pq (the PF unknown count).
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
                           int32_t n_pq)
{
    if (n_bus <= 0 || dimF <= 0) {
        throw std::runtime_error("NewtonSolver::solve_adjoint(): solver state is not prepared");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): batch_size must be positive");
    }
    if (batch_size != stored_batch_size) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): batch_size does not match the last forward solve");
    }
    if (grad_va == nullptr || grad_vm == nullptr) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): grad_va/grad_vm must not be null");
    }
    if (grad_va_stride < n_bus || grad_vm_stride < n_bus) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): grad strides must be at least n_bus");
    }
    if (n_pv > 0 && pv == nullptr) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): pv must not be null");
    }
    if (n_pq > 0 && pq == nullptr) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): pq must not be null");
    }
    if (dimF != n_pv + 2 * n_pq) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): pv/pq dimensions do not match solver dimF");
    }
}

// Pack the dense per-bus voltage-angle/magnitude gradients into the compact
// dL/dx RHS layout the solver uses: [ dVa over pv, dVa over pq, dVm over pq ]
// per batch case. Strides allow each case to be a row of a larger tensor.
std::vector<double> build_grad_state(const double* grad_va,
                                     int64_t grad_va_stride,
                                     const double* grad_vm,
                                     int64_t grad_vm_stride,
                                     int32_t batch_size,
                                     const int32_t* pv,
                                     int32_t n_pv,
                                     const int32_t* pq,
                                     int32_t n_pq)
{
    const int32_t n_pvpq = n_pv + n_pq;
    const int32_t dimF = n_pvpq + n_pq;
    // size_t widening keeps the batch_size * dimF allocation overflow-safe.
    std::vector<double> grad_state(static_cast<std::size_t>(batch_size) *
                                   static_cast<std::size_t>(dimF), 0.0);
    for (int32_t b = 0; b < batch_size; ++b) {
        // Offset into this case's source/destination rows.
        const double* va =
            grad_va + static_cast<std::size_t>(b) * static_cast<std::size_t>(grad_va_stride);
        const double* vm =
            grad_vm + static_cast<std::size_t>(b) * static_cast<std::size_t>(grad_vm_stride);
        double* dst = grad_state.data() +
            static_cast<std::size_t>(b) * static_cast<std::size_t>(dimF);
        // dVa at pv and pq buses, then dVm at pq buses (the dimF ordering).
        for (int32_t i = 0; i < n_pv; ++i) {
            dst[i] = va[pv[i]];
        }
        for (int32_t i = 0; i < n_pq; ++i) {
            dst[n_pv + i] = va[pq[i]];
            dst[n_pvpq + i] = vm[pq[i]];
        }
    }
    return grad_state;
}

// Scatter the adjoint solution lambda back onto per-bus load gradients. By the
// adjoint identity the load gradient is -lambda at the corresponding bus;
// non-pv/pq buses stay zero. Mirrors the build_grad_state() ordering.
void project_load_gradients(const std::vector<double>& lambda,
                            int32_t n_bus,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            AdjointResult& result)
{
    const int32_t n_pvpq = n_pv + n_pq;
    const int32_t dimF = n_pvpq + n_pq;
    result.grad_load_p.assign(static_cast<std::size_t>(batch_size) *
                              static_cast<std::size_t>(n_bus), 0.0);
    result.grad_load_q.assign(result.grad_load_p.size(), 0.0);

    for (int32_t b = 0; b < batch_size; ++b) {
        const double* lam = lambda.data() +
            static_cast<std::size_t>(b) * static_cast<std::size_t>(dimF);
        double* grad_p = result.grad_load_p.data() +
            static_cast<std::size_t>(b) * static_cast<std::size_t>(n_bus);
        double* grad_q = result.grad_load_q.data() +
            static_cast<std::size_t>(b) * static_cast<std::size_t>(n_bus);

        for (int32_t i = 0; i < n_pv; ++i) {
            grad_p[pv[i]] = -lam[i];
        }
        for (int32_t i = 0; i < n_pq; ++i) {
            grad_p[pq[i]] = -lam[n_pv + i];
            grad_q[pq[i]] = -lam[n_pvpq + i];
        }
    }
}

// Relative residual ||J^T lambda - rhs|| / ||rhs|| for the CSC-stored CPU
// Jacobian. Iterating columns of J yields rows of J^T, so this evaluates the
// adjoint product directly. Accumulation is done in long double to keep the
// accuracy check meaningful even when the residual is many orders below rhs.
double relative_residual_norm_csc(const CpuJacobianMatrixF64& J,
                                  const std::vector<double>& lambda,
                                  const std::vector<double>& rhs)
{
    const int32_t dim = J.cols();
    const int32_t* col_ptr = J.outerIndexPtr();
    const int32_t* row_idx = J.innerIndexPtr();
    const double*  vals    = J.valuePtr();

    long double residual_sq = 0.0L;
    long double rhs_sq      = 0.0L;
    for (int32_t c = 0; c < dim; ++c) {
        // (J^T lambda)[c] = sum over the nonzeros in column c of J.
        long double acc = 0.0L;
        for (int32_t k = col_ptr[c]; k < col_ptr[c + 1]; ++k) {
            acc += static_cast<long double>(vals[k]) *
                   static_cast<long double>(lambda[static_cast<std::size_t>(row_idx[k])]);
        }
        const long double r = static_cast<long double>(rhs[static_cast<std::size_t>(c)]);
        const long double d = acc - r;
        residual_sq += d * d;
        rhs_sq      += r * r;
    }
    // Floor the denominator so a near-zero rhs cannot blow up the ratio.
    const long double denom = std::sqrt(std::max(rhs_sq, static_cast<long double>(1.0e-60)));
    const long double denom_floored = std::max(denom, static_cast<long double>(1.0e-30));
    return static_cast<double>(std::sqrt(residual_sq) / denom_floored);
}
