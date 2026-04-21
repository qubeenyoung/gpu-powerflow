#include "linear/bicgstab_solver.hpp"

#include <cublas_v2.h>

#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace exp_20260415::block_ilu {
namespace {

constexpr int32_t kBlockSize = 256;

void cublas_check(cublasStatus_t status, const char* message)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(message) +
                                 " cublas_status=" +
                                 std::to_string(static_cast<int>(status)));
    }
}

__global__ void set_zero_kernel(int32_t n, double* x)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = 0.0;
    }
}

__global__ void copy_kernel(int32_t n, const double* src, double* dst)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

__global__ void update_p_kernel(int32_t n,
                                const double* r,
                                const double* v,
                                double beta,
                                double omega,
                                double* p)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = r[i] + beta * (p[i] - omega * v[i]);
    }
}

__global__ void s_update_kernel(int32_t n,
                                const double* r,
                                const double* v,
                                double alpha,
                                double* s)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        s[i] = r[i] - alpha * v[i];
    }
}

__global__ void x_r_update_kernel(int32_t n,
                                  const double* p_hat,
                                  const double* s_hat,
                                  const double* s,
                                  const double* t,
                                  double alpha,
                                  double omega,
                                  double* x,
                                  double* r)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += alpha * p_hat[i] + omega * s_hat[i];
        r[i] = s[i] - omega * t[i];
    }
}

__global__ void x_alpha_update_kernel(int32_t n,
                                      const double* p_hat,
                                      double alpha,
                                      double* x)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += alpha * p_hat[i];
    }
}

__global__ void residual_kernel(int32_t n,
                                const double* rhs,
                                const double* ax,
                                double* r)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        r[i] = rhs[i] - ax[i];
    }
}

int32_t grid_for(int32_t n)
{
    return (n + kBlockSize - 1) / kBlockSize;
}

double elapsed_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - start)
        .count();
}

}  // namespace

BicgstabSolver::BicgstabSolver()
{
    cublas_check(cublasCreate(&cublas_), "cublasCreate failed");
}

BicgstabSolver::~BicgstabSolver()
{
    if (cublas_ != nullptr) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
}

void BicgstabSolver::ensure_workspace(int32_t n)
{
    const std::size_t count = static_cast<std::size_t>(n);
    if (r_.size() != count) r_.resize(count);
    if (r_hat_.size() != count) r_hat_.resize(count);
    if (p_.size() != count) p_.resize(count);
    if (v_.size() != count) v_.resize(count);
    if (s_.size() != count) s_.resize(count);
    if (t_.size() != count) t_.resize(count);
    if (p_hat_.size() != count) p_hat_.resize(count);
    if (s_hat_.size() != count) s_hat_.resize(count);
    if (ax_.size() != count) ax_.resize(count);
}

double BicgstabSolver::dot(int32_t n, const double* x, const double* y)
{
    double result = 0.0;
    cublas_check(cublasDdot(cublas_, n, x, 1, y, 1, &result), "cublasDdot failed");
    return result;
}

double BicgstabSolver::norm2(int32_t n, const double* x)
{
    double result = 0.0;
    cublas_check(cublasDnrm2(cublas_, n, x, 1, &result), "cublasDnrm2 failed");
    return result;
}

BicgstabStats BicgstabSolver::solve(const CsrSpmv& matrix,
                                    BlockIluPreconditioner& preconditioner,
                                    int32_t n,
                                    const double* rhs_device,
                                    double* x_device,
                                    const BicgstabOptions& options)
{
    if (n <= 0 || rhs_device == nullptr || x_device == nullptr ||
        options.relative_tolerance <= 0.0 || options.max_iterations <= 0) {
        throw std::runtime_error("BicgstabSolver::solve received invalid input");
    }

    ensure_workspace(n);
    const int32_t grid = grid_for(n);
    set_zero_kernel<<<grid, kBlockSize>>>(n, x_device);
    set_zero_kernel<<<grid, kBlockSize>>>(n, p_.data());
    set_zero_kernel<<<grid, kBlockSize>>>(n, v_.data());
    CUDA_CHECK(cudaGetLastError());

    copy_kernel<<<grid, kBlockSize>>>(n, rhs_device, r_.data());
    copy_kernel<<<grid, kBlockSize>>>(n, rhs_device, r_hat_.data());
    CUDA_CHECK(cudaGetLastError());

    BicgstabStats stats;
    stats.initial_residual_norm = norm2(n, r_.data());
    ++stats.reduction_calls;
    const double rhs_norm = std::max(stats.initial_residual_norm,
                                     std::numeric_limits<double>::min());
    const double atol = options.relative_tolerance * rhs_norm;
    stats.final_residual_norm = stats.initial_residual_norm;
    stats.relative_residual_norm = stats.final_residual_norm / rhs_norm;
    if (stats.final_residual_norm <= atol) {
        stats.converged = true;
        return stats;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto solve_start = std::chrono::steady_clock::now();

    double rho_prev = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    auto time_stage = [&](double& bucket, auto&& work) {
        if (options.collect_timing_breakdown) {
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto start = std::chrono::steady_clock::now();
            work();
            CUDA_CHECK(cudaDeviceSynchronize());
            bucket += elapsed_since(start);
        } else {
            work();
        }
    };

    auto timed_dot = [&](const double* x, const double* y) {
        double value = 0.0;
        time_stage(stats.timing.reduction_sec, [&]() {
            value = dot(n, x, y);
        });
        ++stats.reduction_calls;
        return value;
    };

    auto timed_norm2 = [&](const double* x) {
        double value = 0.0;
        time_stage(stats.timing.reduction_sec, [&]() {
            value = norm2(n, x);
        });
        ++stats.reduction_calls;
        return value;
    };

    for (int32_t iter = 1; iter <= options.max_iterations; ++iter) {
        const double rho = timed_dot(r_hat_.data(), r_.data());
        if (!std::isfinite(rho) || std::abs(rho) <= std::numeric_limits<double>::min()) {
            stats.failure_reason = "rho_breakdown";
            break;
        }

        const double beta = (rho / rho_prev) * (alpha / omega);
        time_stage(stats.timing.vector_update_sec, [&]() {
            update_p_kernel<<<grid, kBlockSize>>>(
                n, r_.data(), v_.data(), beta, omega, p_.data());
            CUDA_CHECK(cudaGetLastError());
        });

        time_stage(stats.timing.preconditioner_sec, [&]() {
            preconditioner.apply(p_.data(), p_hat_.data());
        });
        ++stats.preconditioner_applies;
        time_stage(stats.timing.spmv_sec, [&]() {
            matrix.apply(p_hat_.data(), v_.data());
        });
        ++stats.spmv_calls;

        const double alpha_den = timed_dot(r_hat_.data(), v_.data());
        if (!std::isfinite(alpha_den) ||
            std::abs(alpha_den) <= std::numeric_limits<double>::min()) {
            stats.failure_reason = "alpha_breakdown";
            break;
        }
        alpha = rho / alpha_den;

        time_stage(stats.timing.vector_update_sec, [&]() {
            s_update_kernel<<<grid, kBlockSize>>>(n, r_.data(), v_.data(), alpha, s_.data());
            CUDA_CHECK(cudaGetLastError());
        });

        const double s_norm = timed_norm2(s_.data());
        if (s_norm <= atol) {
            time_stage(stats.timing.vector_update_sec, [&]() {
                x_alpha_update_kernel<<<grid, kBlockSize>>>(n, p_hat_.data(), alpha, x_device);
                CUDA_CHECK(cudaGetLastError());
            });
            stats.iterations = iter;
            stats.final_residual_norm = s_norm;
            stats.relative_residual_norm = s_norm / rhs_norm;
            stats.converged = true;
            break;
        }

        time_stage(stats.timing.preconditioner_sec, [&]() {
            preconditioner.apply(s_.data(), s_hat_.data());
        });
        ++stats.preconditioner_applies;
        time_stage(stats.timing.spmv_sec, [&]() {
            matrix.apply(s_hat_.data(), t_.data());
        });
        ++stats.spmv_calls;

        const double tt = timed_dot(t_.data(), t_.data());
        if (!std::isfinite(tt) || tt <= std::numeric_limits<double>::min()) {
            stats.failure_reason = "omega_denominator_breakdown";
            break;
        }
        omega = timed_dot(t_.data(), s_.data()) / tt;
        if (!std::isfinite(omega) || std::abs(omega) <= std::numeric_limits<double>::min()) {
            stats.failure_reason = "omega_breakdown";
            break;
        }

        time_stage(stats.timing.vector_update_sec, [&]() {
            x_r_update_kernel<<<grid, kBlockSize>>>(n,
                                                    p_hat_.data(),
                                                    s_hat_.data(),
                                                    s_.data(),
                                                    t_.data(),
                                                    alpha,
                                                    omega,
                                                    x_device,
                                                    r_.data());
            CUDA_CHECK(cudaGetLastError());
        });

        stats.iterations = iter;
        stats.final_residual_norm = timed_norm2(r_.data());
        stats.relative_residual_norm = stats.final_residual_norm / rhs_norm;
        if (stats.final_residual_norm <= atol) {
            stats.converged = true;
            break;
        }

        rho_prev = rho;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto iteration_end = std::chrono::steady_clock::now();

    time_stage(stats.timing.residual_refresh_sec, [&]() {
        matrix.apply(x_device, ax_.data());
        residual_kernel<<<grid, kBlockSize>>>(n, rhs_device, ax_.data(), r_.data());
        CUDA_CHECK(cudaGetLastError());
        stats.final_residual_norm = norm2(n, r_.data());
    });
    ++stats.spmv_calls;
    ++stats.reduction_calls;
    stats.relative_residual_norm = stats.final_residual_norm / rhs_norm;
    stats.converged = stats.relative_residual_norm <= options.relative_tolerance;
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto solve_end = std::chrono::steady_clock::now();
    stats.solve_sec = std::chrono::duration<double>(solve_end - solve_start).count();
    if (stats.iterations > 0) {
        const double iteration_loop_sec =
            std::chrono::duration<double>(iteration_end - solve_start).count();
        stats.avg_iteration_sec = iteration_loop_sec / static_cast<double>(stats.iterations);
    }
    if (!stats.converged && stats.failure_reason.empty()) {
        stats.failure_reason = "max_iterations";
    }

    return stats;
}

}  // namespace exp_20260415::block_ilu
