#include "newton_solver/core/newton_solver_adjoint_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

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
    std::vector<double> grad_state(static_cast<std::size_t>(batch_size) *
                                   static_cast<std::size_t>(dimF), 0.0);
    for (int32_t b = 0; b < batch_size; ++b) {
        const double* va =
            grad_va + static_cast<std::size_t>(b) * static_cast<std::size_t>(grad_va_stride);
        const double* vm =
            grad_vm + static_cast<std::size_t>(b) * static_cast<std::size_t>(grad_vm_stride);
        double* dst = grad_state.data() +
            static_cast<std::size_t>(b) * static_cast<std::size_t>(dimF);
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
    const long double denom = std::sqrt(std::max(rhs_sq, static_cast<long double>(1.0e-60)));
    const long double denom_floored = std::max(denom, static_cast<long double>(1.0e-30));
    return static_cast<double>(std::sqrt(residual_sq) / denom_floored);
}
