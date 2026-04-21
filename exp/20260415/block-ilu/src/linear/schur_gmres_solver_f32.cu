#include "linear/schur_gmres_solver_f32.hpp"

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

double elapsed_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

template <typename Work>
void time_stage(double& bucket, bool collect, Work&& work)
{
    if (collect) {
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto start = std::chrono::steady_clock::now();
        work();
        CUDA_CHECK(cudaDeviceSynchronize());
        bucket += elapsed_since(start);
    } else {
        work();
    }
}

__global__ void set_zero_kernel_f32(int32_t n, float* x)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

__global__ void copy_kernel_f32(int32_t n, const float* src, float* dst)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void residual_kernel_f32(int32_t n,
                                    const float* rhs,
                                    const float* ax,
                                    float* r)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = rhs[i] - ax[i];
}

__global__ void scale_copy_kernel_f32(int32_t n,
                                      float alpha,
                                      const float* src,
                                      float* dst)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = alpha * src[i];
}

__global__ void sub_scaled_device_scalar_kernel_f32(int32_t n,
                                                    const float* basis,
                                                    const float* alpha,
                                                    float* target)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) target[i] -= (*alpha) * basis[i];
}

__global__ void combine_solution_kernel_f32(int32_t n,
                                            int32_t basis_count,
                                            const float* z_basis,
                                            const float* y,
                                            float* x)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    float sum = 0.0f;
    const std::size_t stride = static_cast<std::size_t>(n);
    for (int32_t col = 0; col < basis_count; ++col) {
        sum += z_basis[static_cast<std::size_t>(col) * stride + row] * y[col];
    }
    x[row] += sum;
}

}  // namespace

SchurGmresSolverF32::SchurGmresSolverF32()
{
    cublas_check(cublasCreate(&cublas_), "cublasCreate failed");
    cublas_check(cublasSetPointerMode(cublas_, CUBLAS_POINTER_MODE_DEVICE),
                 "cublasSetPointerMode failed");
}

SchurGmresSolverF32::~SchurGmresSolverF32()
{
    if (cublas_ != nullptr) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
}

void SchurGmresSolverF32::ensure_workspace(int32_t n, int32_t restart)
{
    if (workspace_n_ == n && workspace_restart_ == restart) return;

    workspace_n_ = n;
    workspace_restart_ = restart;
    const std::size_t rows = static_cast<std::size_t>(n);
    rhs_s_.resize(rows);
    dvm_.resize(rows);
    r_.resize(rows);
    w_.resize(rows);
    ax_.resize(rows);
    v_basis_.resize(rows * static_cast<std::size_t>(restart + 1));
    z_basis_.resize(rows * static_cast<std::size_t>(restart));
    d_h_col_.resize(static_cast<std::size_t>(restart + 1));
    d_y_.resize(static_cast<std::size_t>(restart));

    h_col_host_.assign(static_cast<std::size_t>(restart + 1), 0.0f);
    hessenberg_.assign(static_cast<std::size_t>(restart + 1) *
                           static_cast<std::size_t>(restart),
                       0.0f);
    givens_c_.assign(static_cast<std::size_t>(restart), 0.0f);
    givens_s_.assign(static_cast<std::size_t>(restart), 0.0f);
    g_.assign(static_cast<std::size_t>(restart + 1), 0.0f);
    y_host_.assign(static_cast<std::size_t>(restart), 0.0f);
}

void SchurGmresSolverF32::set_zero(int32_t n, float* x)
{
    set_zero_kernel_f32<<<grid_for(n), kBlockSize>>>(n, x);
    CUDA_CHECK(cudaGetLastError());
}

void SchurGmresSolverF32::copy_vector(int32_t n, const float* src, float* dst)
{
    copy_kernel_f32<<<grid_for(n), kBlockSize>>>(n, src, dst);
    CUDA_CHECK(cudaGetLastError());
}

void SchurGmresSolverF32::norm_to_device(int32_t n, const float* x, float* out_device)
{
    cublas_check(cublasSnrm2(cublas_, n, x, 1, out_device), "cublasSnrm2 failed");
}

void SchurGmresSolverF32::dot_to_device(int32_t n,
                                        const float* x,
                                        const float* y,
                                        float* out_device)
{
    cublas_check(cublasSdot(cublas_, n, x, 1, y, 1, out_device), "cublasSdot failed");
}

float SchurGmresSolverF32::device_scalar_to_host(const float* scalar_device)
{
    float value = 0.0f;
    CUDA_CHECK(cudaMemcpy(&value, scalar_device, sizeof(float), cudaMemcpyDeviceToHost));
    return value;
}

float SchurGmresSolverF32::recompute_residual(ImplicitSchurOperatorF32& op,
                                              int32_t n,
                                              const float* rhs_device,
                                              const float* x_device,
                                              float rhs_norm,
                                              SchurBicgstabStats& stats,
                                              SchurOperatorStats& op_stats,
                                              const SchurBicgstabOptions& options)
{
    time_stage(stats.timing.residual_refresh_sec, options.collect_timing_breakdown, [&]() {
        op.apply(x_device, ax_.data(), op_stats, options.collect_timing_breakdown);
        residual_kernel_f32<<<grid_for(n), kBlockSize>>>(n, rhs_device, ax_.data(), r_.data());
        CUDA_CHECK(cudaGetLastError());
        norm_to_device(n, r_.data(), d_h_col_.data());
    });
    ++stats.reduction_calls;

    const float norm = device_scalar_to_host(d_h_col_.data());
    stats.final_residual_norm = static_cast<double>(norm);
    stats.relative_residual_norm = static_cast<double>(norm / rhs_norm);
    return norm;
}

void SchurGmresSolverF32::update_solution(int32_t n, int32_t basis_count, float* x_device)
{
    d_y_.assign(y_host_.data(), static_cast<std::size_t>(basis_count));
    combine_solution_kernel_f32<<<grid_for(n), kBlockSize>>>(
        n, basis_count, z_basis_.data(), d_y_.data(), x_device);
    CUDA_CHECK(cudaGetLastError());
}

bool SchurGmresSolverF32::solve_small_upper(int32_t basis_count)
{
    std::fill(y_host_.begin(), y_host_.end(), 0.0f);
    for (int32_t row = basis_count - 1; row >= 0; --row) {
        float rhs = g_[static_cast<std::size_t>(row)];
        for (int32_t col = row + 1; col < basis_count; ++col) {
            rhs -= hessenberg_[static_cast<std::size_t>(
                       h_index(workspace_restart_, row, col))] *
                   y_host_[static_cast<std::size_t>(col)];
        }

        const float diag =
            hessenberg_[static_cast<std::size_t>(
                h_index(workspace_restart_, row, row))];
        if (!std::isfinite(diag) ||
            std::abs(diag) <= std::numeric_limits<float>::min()) {
            return false;
        }
        y_host_[static_cast<std::size_t>(row)] = rhs / diag;
    }
    return true;
}

float SchurGmresSolverF32::apply_givens_and_residual(int32_t column, float rhs_norm)
{
    for (int32_t row = 0; row < column; ++row) {
        const int idx0 = h_index(workspace_restart_, row, column);
        const int idx1 = h_index(workspace_restart_, row + 1, column);
        const float h0 = hessenberg_[static_cast<std::size_t>(idx0)];
        const float h1 = hessenberg_[static_cast<std::size_t>(idx1)];
        const float c = givens_c_[static_cast<std::size_t>(row)];
        const float s = givens_s_[static_cast<std::size_t>(row)];
        hessenberg_[static_cast<std::size_t>(idx0)] = c * h0 + s * h1;
        hessenberg_[static_cast<std::size_t>(idx1)] = -s * h0 + c * h1;
    }

    const int diag_idx = h_index(workspace_restart_, column, column);
    const int subdiag_idx = h_index(workspace_restart_, column + 1, column);
    const float diag = hessenberg_[static_cast<std::size_t>(diag_idx)];
    const float subdiag = hessenberg_[static_cast<std::size_t>(subdiag_idx)];
    const float radius = std::hypot(diag, subdiag);
    float c = 1.0f;
    float s = 0.0f;
    if (radius > std::numeric_limits<float>::min()) {
        c = diag / radius;
        s = subdiag / radius;
    }
    givens_c_[static_cast<std::size_t>(column)] = c;
    givens_s_[static_cast<std::size_t>(column)] = s;
    hessenberg_[static_cast<std::size_t>(diag_idx)] = c * diag + s * subdiag;
    hessenberg_[static_cast<std::size_t>(subdiag_idx)] = 0.0f;

    const float old_g = g_[static_cast<std::size_t>(column)];
    g_[static_cast<std::size_t>(column)] = c * old_g;
    g_[static_cast<std::size_t>(column + 1)] = -s * old_g;
    return std::abs(g_[static_cast<std::size_t>(column + 1)]) / rhs_norm;
}

SchurBicgstabStats SchurGmresSolverF32::solve(ImplicitSchurOperatorF32& op,
                                              const double* rhs_full_device,
                                              double* dx_full_device,
                                              const SchurBicgstabOptions& options)
{
    if (rhs_full_device == nullptr || dx_full_device == nullptr ||
        options.relative_tolerance <= 0.0 || options.max_iterations <= 0 ||
        options.gmres_restart <= 0 ||
        options.gmres_residual_check_interval <= 0 ||
        op.n_pq() <= 0) {
        throw std::runtime_error("SchurGmresSolverF32::solve received invalid input");
    }

    const int32_t n = op.n_pq();
    const int32_t restart = std::min(options.gmres_restart, options.max_iterations);
    ensure_workspace(n, restart);

    SchurBicgstabStats stats;
    SchurOperatorStats op_stats;
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto solve_start = std::chrono::steady_clock::now();

    op.build_rhs(rhs_full_device, rhs_s_.data(), op_stats, options.collect_timing_breakdown);
    set_zero(n, dvm_.data());
    copy_vector(n, rhs_s_.data(), r_.data());
    time_stage(stats.timing.reduction_sec, options.collect_timing_breakdown, [&]() {
        norm_to_device(n, r_.data(), d_h_col_.data());
    });
    ++stats.reduction_calls;

    stats.initial_residual_norm = static_cast<double>(device_scalar_to_host(d_h_col_.data()));
    const float rhs_norm = std::max(static_cast<float>(stats.initial_residual_norm),
                                    std::numeric_limits<float>::min());
    const float atol = static_cast<float>(options.relative_tolerance) * rhs_norm;
    stats.final_residual_norm = stats.initial_residual_norm;
    stats.relative_residual_norm = static_cast<double>(
        static_cast<float>(stats.final_residual_norm) / rhs_norm);

    double iteration_sec = 0.0;
    if (static_cast<float>(stats.final_residual_norm) <= atol) {
        stats.converged = true;
    }

    float beta = static_cast<float>(stats.initial_residual_norm);
    while (!stats.converged && stats.iterations < options.max_iterations) {
        ++stats.restart_cycles;
        const int32_t remaining = options.max_iterations - stats.iterations;
        const int32_t cycle_limit = std::min(restart, remaining);
        std::fill(hessenberg_.begin(), hessenberg_.end(), 0.0f);
        std::fill(g_.begin(), g_.end(), 0.0f);
        std::fill(givens_c_.begin(), givens_c_.end(), 0.0f);
        std::fill(givens_s_.begin(), givens_s_.end(), 0.0f);
        g_[0] = beta;

        time_stage(stats.timing.vector_update_sec, options.collect_timing_breakdown, [&]() {
            scale_copy_kernel_f32<<<grid_for(n), kBlockSize>>>(
                n, 1.0f / beta, r_.data(), v_basis_.data());
            CUDA_CHECK(cudaGetLastError());
        });

        bool solution_updated = false;
        int32_t steps_this_cycle = 0;
        for (int32_t j = 0; j < cycle_limit; ++j) {
            const float* v_j = v_basis_.data() + offset_for(n, j);
            float* z_j = z_basis_.data() + offset_for(n, j);
            float* v_next = v_basis_.data() + offset_for(n, j + 1);

            const auto iter_start = std::chrono::steady_clock::now();
            op.apply_schur_preconditioner(
                v_j, z_j, op_stats, options.collect_timing_breakdown);
            op.apply(z_j, w_.data(), op_stats, options.collect_timing_breakdown);

            time_stage(stats.timing.reduction_sec, options.collect_timing_breakdown, [&]() {
                for (int32_t i = 0; i <= j; ++i) {
                    const float* v_i = v_basis_.data() + offset_for(n, i);
                    dot_to_device(n, w_.data(), v_i, d_h_col_.data() + i);
                    ++stats.reduction_calls;
                    sub_scaled_device_scalar_kernel_f32<<<grid_for(n), kBlockSize>>>(
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

            const float next_norm = h_col_host_[static_cast<std::size_t>(j + 1)];
            const bool happy_breakdown =
                std::isfinite(next_norm) &&
                next_norm <= std::numeric_limits<float>::epsilon() * rhs_norm;
            if (!std::isfinite(next_norm)) {
                stats.failure_reason = "arnoldi_norm_nan";
                break;
            }
            if (!happy_breakdown) {
                time_stage(stats.timing.vector_update_sec,
                           options.collect_timing_breakdown,
                           [&]() {
                               scale_copy_kernel_f32<<<grid_for(n), kBlockSize>>>(
                                   n, 1.0f / next_norm, w_.data(), v_next);
                               CUDA_CHECK(cudaGetLastError());
                           });
            }

            const float estimated_relative_residual =
                apply_givens_and_residual(j, rhs_norm);
            ++stats.iterations;
            steps_this_cycle = j + 1;
            iteration_sec += elapsed_since(iter_start);
            stats.final_residual_norm =
                static_cast<double>(estimated_relative_residual * rhs_norm);
            stats.relative_residual_norm = static_cast<double>(estimated_relative_residual);

            const bool check_now =
                happy_breakdown ||
                stats.iterations == options.max_iterations ||
                ((stats.iterations % options.gmres_residual_check_interval) == 0);
            if (check_now && estimated_relative_residual <= options.relative_tolerance) {
                const int32_t basis_count = j + 1;
                const auto small_start = std::chrono::steady_clock::now();
                if (!solve_small_upper(basis_count)) {
                    stats.failure_reason = "small_upper_breakdown";
                    break;
                }
                stats.timing.small_solve_sec += elapsed_since(small_start);
                time_stage(stats.timing.vector_update_sec,
                           options.collect_timing_breakdown,
                           [&]() {
                               update_solution(n, basis_count, dvm_.data());
                           });
                solution_updated = true;

                beta = recompute_residual(op,
                                          n,
                                          rhs_s_.data(),
                                          dvm_.data(),
                                          rhs_norm,
                                          stats,
                                          op_stats,
                                          options);
                stats.converged = beta <= atol;
                break;
            }

            if (happy_breakdown) {
                const int32_t basis_count = j + 1;
                const auto small_start = std::chrono::steady_clock::now();
                if (!solve_small_upper(basis_count)) {
                    stats.failure_reason = "small_upper_breakdown";
                    break;
                }
                stats.timing.small_solve_sec += elapsed_since(small_start);
                time_stage(stats.timing.vector_update_sec,
                           options.collect_timing_breakdown,
                           [&]() {
                               update_solution(n, basis_count, dvm_.data());
                           });
                solution_updated = true;

                beta = recompute_residual(op,
                                          n,
                                          rhs_s_.data(),
                                          dvm_.data(),
                                          rhs_norm,
                                          stats,
                                          op_stats,
                                          options);
                stats.converged = beta <= atol;
                break;
            }
        }

        if (stats.converged || !stats.failure_reason.empty()) break;

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
            stats.timing.small_solve_sec += elapsed_since(small_start);
            time_stage(stats.timing.vector_update_sec, options.collect_timing_breakdown, [&]() {
                update_solution(n, basis_count, dvm_.data());
            });
            beta = recompute_residual(op,
                                      n,
                                      rhs_s_.data(),
                                      dvm_.data(),
                                      rhs_norm,
                                      stats,
                                      op_stats,
                                      options);
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
        stats.avg_iteration_sec = iteration_sec / static_cast<double>(stats.iterations);
    }

    op.recover_solution(rhs_full_device, dvm_.data(), dx_full_device,
                        op_stats, options.collect_timing_breakdown);

    CUDA_CHECK(cudaDeviceSynchronize());
    stats.solve_sec = elapsed_since(solve_start);

    stats.schur_matvec_calls = op_stats.schur_matvec_calls;
    stats.schur_preconditioner_calls = op_stats.schur_preconditioner_calls;
    stats.spmv_calls = op_stats.spmv_calls;
    stats.j11_solve_calls = op_stats.j11_solve_calls;
    stats.timing.schur_rhs_sec = op_stats.rhs_sec;
    stats.timing.schur_matvec_sec = op_stats.matvec_sec;
    stats.timing.schur_recover_sec = op_stats.recover_sec;
    stats.timing.schur_spmv_sec = op_stats.spmv_sec;
    stats.timing.schur_j11_solve_sec = op_stats.j11_solve_sec;
    stats.timing.schur_preconditioner_sec = op_stats.schur_preconditioner_sec;
    stats.timing.vector_update_sec += op_stats.vector_update_sec;
    return stats;
}

}  // namespace exp_20260415::block_ilu
