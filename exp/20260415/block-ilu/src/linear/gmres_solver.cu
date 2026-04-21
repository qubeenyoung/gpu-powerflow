#include "linear/gmres_solver.hpp"

#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

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

int32_t grid_for(int32_t n)
{
    return (n + kBlockSize - 1) / kBlockSize;
}

std::size_t offset_for(int32_t n, int32_t column)
{
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(column);
}

int h_index(int32_t restart, int32_t row, int32_t col)
{
    return col * (restart + 1) + row;
}

template <typename Fn>
double timed_seconds(bool collect, Fn&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    fn();
    if (collect) {
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }
    return 0.0;
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

__global__ void scale_copy_kernel(int32_t n,
                                  double alpha,
                                  const double* src,
                                  double* dst)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = alpha * src[i];
    }
}

__global__ void sub_scaled_device_scalar_kernel(int32_t n,
                                                const double* basis,
                                                const double* alpha,
                                                double* target)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        target[i] -= (*alpha) * basis[i];
    }
}

__global__ void combine_solution_kernel(int32_t n,
                                        int32_t basis_count,
                                        const double* z_basis,
                                        const double* y,
                                        double* x)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }

    double sum = 0.0;
    const std::size_t stride = static_cast<std::size_t>(n);
    for (int32_t col = 0; col < basis_count; ++col) {
        sum += z_basis[static_cast<std::size_t>(col) * stride + row] * y[col];
    }
    x[row] += sum;
}

}  // namespace

GmresSolver::GmresSolver()
{
    cublas_check(cublasCreate(&cublas_), "cublasCreate failed");
    cublas_check(cublasSetPointerMode(cublas_, CUBLAS_POINTER_MODE_DEVICE),
                 "cublasSetPointerMode failed");
}

GmresSolver::~GmresSolver()
{
    if (cublas_ != nullptr) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
}

void GmresSolver::ensure_workspace(int32_t n, int32_t restart)
{
    if (workspace_n_ == n && workspace_restart_ == restart) {
        return;
    }

    workspace_n_ = n;
    workspace_restart_ = restart;
    const std::size_t rows = static_cast<std::size_t>(n);
    r_.resize(rows);
    w_.resize(rows);
    ax_.resize(rows);
    v_basis_.resize(rows * static_cast<std::size_t>(restart + 1));
    z_basis_.resize(rows * static_cast<std::size_t>(restart));
    d_h_col_.resize(static_cast<std::size_t>(restart + 1));
    d_y_.resize(static_cast<std::size_t>(restart));

    h_col_host_.assign(static_cast<std::size_t>(restart + 1), 0.0);
    hessenberg_.assign(static_cast<std::size_t>(restart + 1) *
                           static_cast<std::size_t>(restart),
                       0.0);
    givens_c_.assign(static_cast<std::size_t>(restart), 0.0);
    givens_s_.assign(static_cast<std::size_t>(restart), 0.0);
    g_.assign(static_cast<std::size_t>(restart + 1), 0.0);
    y_host_.assign(static_cast<std::size_t>(restart), 0.0);
}

void GmresSolver::set_zero(int32_t n, double* x)
{
    set_zero_kernel<<<grid_for(n), kBlockSize>>>(n, x);
    CUDA_CHECK(cudaGetLastError());
}

void GmresSolver::copy_vector(int32_t n, const double* src, double* dst)
{
    copy_kernel<<<grid_for(n), kBlockSize>>>(n, src, dst);
    CUDA_CHECK(cudaGetLastError());
}

void GmresSolver::norm_to_device(int32_t n, const double* x, double* out_device)
{
    cublas_check(cublasDnrm2(cublas_, n, x, 1, out_device), "cublasDnrm2 failed");
}

void GmresSolver::dot_to_device(int32_t n,
                                const double* x,
                                const double* y,
                                double* out_device)
{
    cublas_check(cublasDdot(cublas_, n, x, 1, y, 1, out_device), "cublasDdot failed");
}

double GmresSolver::device_scalar_to_host(const double* scalar_device)
{
    double value = 0.0;
    CUDA_CHECK(cudaMemcpy(&value, scalar_device, sizeof(double), cudaMemcpyDeviceToHost));
    return value;
}

double GmresSolver::recompute_residual(const CsrSpmv& matrix,
                                       int32_t n,
                                       const double* rhs_device,
                                       const double* x_device,
                                       double rhs_norm,
                                       GmresStats& stats,
                                       const GmresOptions& options)
{
    stats.timing.residual_refresh_sec += timed_seconds(options.collect_timing_breakdown, [&] {
        matrix.apply(x_device, ax_.data());
        ++stats.spmv_calls;
        residual_kernel<<<grid_for(n), kBlockSize>>>(n, rhs_device, ax_.data(), r_.data());
        CUDA_CHECK(cudaGetLastError());
        norm_to_device(n, r_.data(), d_h_col_.data());
        ++stats.reduction_calls;
    });

    const double norm = device_scalar_to_host(d_h_col_.data());
    stats.final_residual_norm = norm;
    stats.relative_residual_norm = norm / rhs_norm;
    return norm;
}

void GmresSolver::update_solution(int32_t n, int32_t basis_count, double* x_device)
{
    d_y_.assign(y_host_.data(), static_cast<std::size_t>(basis_count));
    combine_solution_kernel<<<grid_for(n), kBlockSize>>>(
        n, basis_count, z_basis_.data(), d_y_.data(), x_device);
    CUDA_CHECK(cudaGetLastError());
}

bool GmresSolver::solve_small_upper(int32_t basis_count)
{
    std::fill(y_host_.begin(), y_host_.end(), 0.0);
    for (int32_t row = basis_count - 1; row >= 0; --row) {
        double rhs = g_[static_cast<std::size_t>(row)];
        for (int32_t col = row + 1; col < basis_count; ++col) {
            rhs -= hessenberg_[static_cast<std::size_t>(
                       h_index(workspace_restart_, row, col))] *
                   y_host_[static_cast<std::size_t>(col)];
        }

        const double diag =
            hessenberg_[static_cast<std::size_t>(
                h_index(workspace_restart_, row, row))];
        if (!std::isfinite(diag) ||
            std::abs(diag) <= std::numeric_limits<double>::min()) {
            return false;
        }
        y_host_[static_cast<std::size_t>(row)] = rhs / diag;
    }
    return true;
}

double GmresSolver::apply_givens_and_residual(int32_t column, double rhs_norm)
{
    for (int32_t row = 0; row < column; ++row) {
        const int idx0 = h_index(workspace_restart_, row, column);
        const int idx1 = h_index(workspace_restart_, row + 1, column);
        const double h0 = hessenberg_[static_cast<std::size_t>(idx0)];
        const double h1 = hessenberg_[static_cast<std::size_t>(idx1)];
        const double c = givens_c_[static_cast<std::size_t>(row)];
        const double s = givens_s_[static_cast<std::size_t>(row)];
        hessenberg_[static_cast<std::size_t>(idx0)] = c * h0 + s * h1;
        hessenberg_[static_cast<std::size_t>(idx1)] = -s * h0 + c * h1;
    }

    const int diag_idx = h_index(workspace_restart_, column, column);
    const int subdiag_idx = h_index(workspace_restart_, column + 1, column);
    const double diag = hessenberg_[static_cast<std::size_t>(diag_idx)];
    const double subdiag = hessenberg_[static_cast<std::size_t>(subdiag_idx)];
    const double radius = std::hypot(diag, subdiag);
    double c = 1.0;
    double s = 0.0;
    if (radius > std::numeric_limits<double>::min()) {
        c = diag / radius;
        s = subdiag / radius;
    }
    givens_c_[static_cast<std::size_t>(column)] = c;
    givens_s_[static_cast<std::size_t>(column)] = s;
    hessenberg_[static_cast<std::size_t>(diag_idx)] = c * diag + s * subdiag;
    hessenberg_[static_cast<std::size_t>(subdiag_idx)] = 0.0;

    const double old_g = g_[static_cast<std::size_t>(column)];
    g_[static_cast<std::size_t>(column)] = c * old_g;
    g_[static_cast<std::size_t>(column + 1)] = -s * old_g;
    return std::abs(g_[static_cast<std::size_t>(column + 1)]) / rhs_norm;
}

GmresStats GmresSolver::solve(const CsrSpmv& matrix,
                              BlockIluPreconditioner& preconditioner,
                              int32_t n,
                              const double* rhs_device,
                              double* x_device,
                              const GmresOptions& options)
{
    if (n <= 0 || rhs_device == nullptr || x_device == nullptr ||
        options.relative_tolerance <= 0.0 || options.max_iterations <= 0 ||
        options.restart <= 0 || options.residual_check_interval <= 0) {
        throw std::runtime_error("GmresSolver::solve received invalid input");
    }

    const int32_t restart = std::min(options.restart, options.max_iterations);
    ensure_workspace(n, restart);

    const auto solve_start = std::chrono::steady_clock::now();
    GmresStats stats;

    set_zero(n, x_device);
    copy_vector(n, rhs_device, r_.data());
    norm_to_device(n, r_.data(), d_h_col_.data());
    stats.initial_residual_norm = device_scalar_to_host(d_h_col_.data());
    const double rhs_norm =
        std::max(stats.initial_residual_norm, std::numeric_limits<double>::min());
    stats.final_residual_norm = stats.initial_residual_norm;
    stats.relative_residual_norm = stats.final_residual_norm / rhs_norm;
    if (stats.final_residual_norm <= options.relative_tolerance * rhs_norm) {
        stats.converged = true;
        stats.solve_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
        return stats;
    }

    double beta = stats.initial_residual_norm;
    const double atol = options.relative_tolerance * rhs_norm;

    while (stats.iterations < options.max_iterations) {
        ++stats.restart_cycles;
        const int32_t remaining = options.max_iterations - stats.iterations;
        const int32_t cycle_limit = std::min(restart, remaining);
        std::fill(hessenberg_.begin(), hessenberg_.end(), 0.0);
        std::fill(g_.begin(), g_.end(), 0.0);
        std::fill(givens_c_.begin(), givens_c_.end(), 0.0);
        std::fill(givens_s_.begin(), givens_s_.end(), 0.0);
        g_[0] = beta;

        stats.timing.vector_update_sec += timed_seconds(options.collect_timing_breakdown, [&] {
            scale_copy_kernel<<<grid_for(n), kBlockSize>>>(
                n, 1.0 / beta, r_.data(), v_basis_.data());
            CUDA_CHECK(cudaGetLastError());
        });

        bool solution_updated = false;
        int32_t steps_this_cycle = 0;
        for (int32_t j = 0; j < cycle_limit; ++j) {
            const double* v_j = v_basis_.data() + offset_for(n, j);
            double* z_j = z_basis_.data() + offset_for(n, j);
            double* v_next = v_basis_.data() + offset_for(n, j + 1);

            const auto iter_start = std::chrono::steady_clock::now();
            stats.timing.preconditioner_sec += timed_seconds(options.collect_timing_breakdown, [&] {
                preconditioner.apply(v_j, z_j);
            });
            ++stats.preconditioner_applies;

            stats.timing.spmv_sec += timed_seconds(options.collect_timing_breakdown, [&] {
                matrix.apply(z_j, w_.data());
            });
            ++stats.spmv_calls;

            stats.timing.reduction_sec += timed_seconds(options.collect_timing_breakdown, [&] {
                for (int32_t i = 0; i <= j; ++i) {
                    const double* v_i = v_basis_.data() + offset_for(n, i);
                    dot_to_device(n, w_.data(), v_i, d_h_col_.data() + i);
                    ++stats.reduction_calls;
                    sub_scaled_device_scalar_kernel<<<grid_for(n), kBlockSize>>>(
                        n, v_i, d_h_col_.data() + i, w_.data());
                    CUDA_CHECK(cudaGetLastError());
                }
                norm_to_device(n, w_.data(), d_h_col_.data() + j + 1);
                ++stats.reduction_calls;
                d_h_col_.copyTo(h_col_host_.data(), static_cast<std::size_t>(j + 2));
            });

            for (int32_t row = 0; row <= j + 1; ++row) {
                hessenberg_[static_cast<std::size_t>(
                    h_index(restart, row, j))] = h_col_host_[static_cast<std::size_t>(row)];
            }

            const double next_norm = h_col_host_[static_cast<std::size_t>(j + 1)];
            const bool happy_breakdown =
                std::isfinite(next_norm) &&
                next_norm <= std::numeric_limits<double>::epsilon() * rhs_norm;
            if (!std::isfinite(next_norm)) {
                stats.failure_reason = "arnoldi_norm_nan";
                break;
            }
            if (!happy_breakdown) {
                stats.timing.vector_update_sec +=
                    timed_seconds(options.collect_timing_breakdown, [&] {
                        scale_copy_kernel<<<grid_for(n), kBlockSize>>>(
                            n, 1.0 / next_norm, w_.data(), v_next);
                        CUDA_CHECK(cudaGetLastError());
                    });
            }

            const double estimated_relative_residual =
                apply_givens_and_residual(j, rhs_norm);
            ++stats.iterations;
            steps_this_cycle = j + 1;
            stats.iteration_sec +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - iter_start).count();
            stats.final_residual_norm = estimated_relative_residual * rhs_norm;
            stats.relative_residual_norm = estimated_relative_residual;

            const bool check_now =
                happy_breakdown ||
                stats.iterations == options.max_iterations ||
                ((stats.iterations % options.residual_check_interval) == 0);
            if (check_now && estimated_relative_residual <= options.relative_tolerance) {
                const int32_t basis_count = j + 1;
                const auto small_start = std::chrono::steady_clock::now();
                if (!solve_small_upper(basis_count)) {
                    stats.failure_reason = "small_upper_breakdown";
                    break;
                }
                stats.timing.small_solve_sec +=
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - small_start).count();
                stats.timing.vector_update_sec +=
                    timed_seconds(options.collect_timing_breakdown, [&] {
                        update_solution(n, basis_count, x_device);
                    });
                solution_updated = true;

                beta = recompute_residual(matrix, n, rhs_device, x_device, rhs_norm, stats, options);
                if (beta <= atol) {
                    stats.converged = true;
                }
                break;
            }

            if (happy_breakdown) {
                const int32_t basis_count = j + 1;
                const auto small_start = std::chrono::steady_clock::now();
                if (!solve_small_upper(basis_count)) {
                    stats.failure_reason = "small_upper_breakdown";
                    break;
                }
                stats.timing.small_solve_sec +=
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - small_start).count();
                stats.timing.vector_update_sec +=
                    timed_seconds(options.collect_timing_breakdown, [&] {
                        update_solution(n, basis_count, x_device);
                    });
                solution_updated = true;

                beta = recompute_residual(matrix, n, rhs_device, x_device, rhs_norm, stats, options);
                stats.converged = beta <= atol;
                break;
            }
        }

        if (stats.converged || !stats.failure_reason.empty()) {
            break;
        }

        if (!solution_updated) {
            const int32_t basis_count = steps_this_cycle;
            if (basis_count <= 0) {
                stats.failure_reason = "empty_restart_cycle";
                break;
            }
            const auto small_start = std::chrono::steady_clock::now();
            if (!solve_small_upper(basis_count)) {
                stats.failure_reason = "small_upper_breakdown";
                break;
            }
            stats.timing.small_solve_sec +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - small_start).count();
            stats.timing.vector_update_sec += timed_seconds(options.collect_timing_breakdown, [&] {
                update_solution(n, basis_count, x_device);
            });
            beta = recompute_residual(matrix, n, rhs_device, x_device, rhs_norm, stats, options);
            if (beta <= atol) {
                stats.converged = true;
                break;
            }
        }
    }

    if (!stats.converged && stats.failure_reason.empty()) {
        stats.failure_reason = "max_iterations";
    }
    if (stats.iterations > 0) {
        stats.avg_iteration_sec = stats.iteration_sec / static_cast<double>(stats.iterations);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    stats.solve_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
    return stats;
}

}  // namespace exp_20260415::block_ilu
