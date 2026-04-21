#include "fgmres_solver.hpp"

#include "utils/cuda_utils.hpp"

#include <Eigen/Dense>

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace exp_20260414::amgx_v2 {
namespace {

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                                  \
    do {                                                                                    \
        cublasStatus_t status__ = (call);                                                   \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                            \
            throw std::runtime_error(                                                       \
                std::string("cuBLAS call failed at ") + __FILE__ + ":" +                  \
                std::to_string(__LINE__) + " status=" +                                    \
                std::to_string(static_cast<int>(status__)));                                \
        }                                                                                   \
    } while (0)
#endif

constexpr int32_t kBlockSize = 256;

class CublasHandle {
public:
    CublasHandle()
    {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }

    ~CublasHandle()
    {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

__global__ void scale_copy_kernel(int32_t n, const double* input, double scale, double* output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = scale * input[i];
    }
}

__global__ void subtract_kernel(int32_t n,
                                const double* lhs,
                                const double* rhs,
                                double* output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = lhs[i] - rhs[i];
    }
}

__global__ void add_linear_combination_kernel(int32_t n,
                                              int32_t k,
                                              const double* coeffs,
                                              const double* vectors,
                                              int32_t stride,
                                              double* x)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }

    double sum = 0.0;
    for (int32_t j = 0; j < k; ++j) {
        sum += coeffs[j] * vectors[static_cast<std::size_t>(j) * stride + row];
    }
    x[row] += sum;
}

void add_linear_combination(int32_t dim,
                            const Eigen::VectorXd& coeffs,
                            const double* vectors,
                            double* x,
                            DeviceBuffer<double>& d_coeffs)
{
    const int32_t k = static_cast<int32_t>(coeffs.size());
    if (k <= 0) {
        return;
    }
    d_coeffs.assign(coeffs.data(), static_cast<std::size_t>(k));
    const int32_t grid = (dim + kBlockSize - 1) / kBlockSize;
    add_linear_combination_kernel<<<grid, kBlockSize>>>(
        dim, k, d_coeffs.data(), vectors, dim, x);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

SolveStats FgmresSolver::solve(int32_t dim,
                               const double* rhs_device,
                               double* dx_device,
                               const SolverOptions& options,
                               OperatorFn apply_operator,
                               OperatorFn apply_preconditioner,
                               void* user)
{
    if (dim <= 0 || rhs_device == nullptr || dx_device == nullptr ||
        apply_operator == nullptr || apply_preconditioner == nullptr) {
        throw std::runtime_error("FgmresSolver::solve received invalid input");
    }
    if (options.linear_tolerance <= 0.0 ||
        options.max_inner_iterations <= 0 ||
        options.gmres_restart <= 0) {
        throw std::runtime_error("FgmresSolver::solve received invalid options");
    }

    SolveStats stats;
    CublasHandle blas;
    const int32_t restart = std::min(options.gmres_restart, options.max_inner_iterations);
    const int32_t grid = (dim + kBlockSize - 1) / kBlockSize;

    DeviceBuffer<double> d_x(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_r(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_ax(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_w(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_basis(static_cast<std::size_t>(dim) * (restart + 1));
    DeviceBuffer<double> d_z_basis(static_cast<std::size_t>(dim) * restart);
    DeviceBuffer<double> d_coeffs;

    d_x.memsetZero();
    CUBLAS_CHECK(cublasDcopy(blas.get(), dim, rhs_device, 1, d_r.data(), 1));

    double rhs_norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, rhs_device, 1, &rhs_norm));
    rhs_norm = std::max(rhs_norm, std::numeric_limits<double>::min());
    const double absolute_tolerance = options.linear_tolerance * rhs_norm;

    double residual_norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_r.data(), 1, &residual_norm));
    if (residual_norm <= absolute_tolerance) {
        stats.converged = true;
        stats.final_mismatch = residual_norm;
        CUBLAS_CHECK(cublasDcopy(blas.get(), dim, d_x.data(), 1, dx_device, 1));
        return stats;
    }

    while (stats.total_inner_iterations < options.max_inner_iterations) {
        double beta = 0.0;
        CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_r.data(), 1, &beta));
        if (!std::isfinite(beta)) {
            stats.failure_reason = "nonfinite_residual";
            break;
        }
        if (beta <= absolute_tolerance) {
            stats.converged = true;
            stats.final_mismatch = beta;
            break;
        }

        const int32_t basis_dim =
            std::min(restart,
                     static_cast<int32_t>(options.max_inner_iterations -
                                          stats.total_inner_iterations));
        Eigen::MatrixXd hessenberg = Eigen::MatrixXd::Zero(basis_dim + 1, basis_dim);
        Eigen::VectorXd g = Eigen::VectorXd::Zero(basis_dim + 1);
        g[0] = beta;

        scale_copy_kernel<<<grid, kBlockSize>>>(dim, d_r.data(), 1.0 / beta, d_basis.data());
        CUDA_CHECK(cudaGetLastError());

        Eigen::VectorXd best_y;
        int32_t best_k = 0;
        bool restart_completed = true;

        for (int32_t j = 0; j < basis_dim; ++j) {
            double* v_j = d_basis.data() + static_cast<std::size_t>(j) * dim;
            double* z_j = d_z_basis.data() + static_cast<std::size_t>(j) * dim;

            apply_preconditioner(v_j, z_j, user);
            apply_operator(z_j, d_w.data(), user);
            ++stats.total_jv_calls;

            for (int32_t i = 0; i <= j; ++i) {
                double* v_i = d_basis.data() + static_cast<std::size_t>(i) * dim;
                double dot = 0.0;
                CUBLAS_CHECK(cublasDdot(blas.get(), dim, v_i, 1, d_w.data(), 1, &dot));
                hessenberg(i, j) = dot;
                const double alpha = -dot;
                CUBLAS_CHECK(cublasDaxpy(blas.get(), dim, &alpha, v_i, 1, d_w.data(), 1));
            }

            double h_next = 0.0;
            CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_w.data(), 1, &h_next));
            hessenberg(j + 1, j) = h_next;
            const bool happy_breakdown = h_next <= std::numeric_limits<double>::epsilon();
            if (!happy_breakdown) {
                scale_copy_kernel<<<grid, kBlockSize>>>(
                    dim,
                    d_w.data(),
                    1.0 / h_next,
                    d_basis.data() + static_cast<std::size_t>(j + 1) * dim);
                CUDA_CHECK(cudaGetLastError());
            }

            ++stats.total_inner_iterations;
            const auto h = hessenberg.block(0, 0, j + 2, j + 1);
            const auto g_head = g.head(j + 2);
            const Eigen::VectorXd y = h.colPivHouseholderQr().solve(g_head);
            const double residual_estimate = (g_head - h * y).norm();

            best_y = y;
            best_k = j + 1;
            stats.final_mismatch = residual_estimate;

            if (residual_estimate <= absolute_tolerance || happy_breakdown) {
                add_linear_combination(dim, best_y, d_z_basis.data(), d_x.data(), d_coeffs);
                stats.converged = true;
                restart_completed = false;
                break;
            }
        }

        if (!restart_completed) {
            break;
        }

        if (best_k > 0) {
            add_linear_combination(dim, best_y, d_z_basis.data(), d_x.data(), d_coeffs);
        }

        apply_operator(d_x.data(), d_ax.data(), user);
        ++stats.total_jv_calls;
        subtract_kernel<<<grid, kBlockSize>>>(dim, rhs_device, d_ax.data(), d_r.data());
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_r.data(), 1, &residual_norm));
        stats.final_mismatch = residual_norm;
    }

    if (!stats.converged && stats.failure_reason.empty()) {
        stats.failure_reason = "max_inner_iterations";
        ++stats.linear_failures;
    }
    CUBLAS_CHECK(cublasDcopy(blas.get(), dim, d_x.data(), 1, dx_device, 1));
    return stats;
}

}  // namespace exp_20260414::amgx_v2
