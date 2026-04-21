#include "linear/schur_bicgstab_solver_f32.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace exp_20260415::block_ilu {
namespace {

constexpr int32_t kBlockSize = 256;
constexpr float kTiny = 1e-30f;

enum ScalarSlot : int32_t {
    kRho = 0,
    kRhoPrev,
    kAlpha,
    kOmega,
    kBeta,
    kAlphaDen,
    kSNorm,
    kTT,
    kTS,
    kRNorm,
    kRhsNorm,
    kAtol,
    kRelativeResidual,
    kScalarCount
};

enum StatusSlot : int32_t {
    kFailureStatus = 0,
    kConvergedStatus,
    kStatusCount
};

enum FailureCode : int32_t {
    kNoFailure = 0,
    kRhoBreakdown = 1,
    kAlphaBreakdown = 2,
    kOmegaDenominatorBreakdown = 3,
    kOmegaBreakdown = 4
};

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

__global__ void residual_kernel_f32(int32_t n, const float* rhs, const float* ax, float* r)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = rhs[i] - ax[i];
}

__global__ void norm_partial_kernel_f32(int32_t n, const float* x, float* partial)
{
    __shared__ float values[kBlockSize];
    const int32_t tid = threadIdx.x;
    const int32_t i = blockIdx.x * blockDim.x + tid;
    const float value = (i < n) ? x[i] : 0.0f;
    values[tid] = value * value;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) values[tid] += values[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = values[0];
}

__global__ void finalize_norm_kernel_f32(int32_t n_partials,
                                         const float* partial,
                                         float* scalars,
                                         int32_t scalar_slot)
{
    float sum = 0.0f;
    for (int32_t i = 0; i < n_partials; ++i) sum += partial[i];
    scalars[scalar_slot] = sqrtf(sum);
}

__global__ void init_scalar_state_kernel_f32(float relative_tolerance,
                                             float* scalars,
                                             int32_t* status)
{
    const float rhs_norm = fmaxf(scalars[kRNorm], kTiny);
    scalars[kRhsNorm] = rhs_norm;
    scalars[kAtol] = relative_tolerance * rhs_norm;
    scalars[kRelativeResidual] = scalars[kRNorm] / rhs_norm;
    scalars[kRhoPrev] = 1.0f;
    scalars[kAlpha] = 1.0f;
    scalars[kOmega] = 1.0f;
    scalars[kBeta] = 0.0f;
    status[kFailureStatus] = kNoFailure;
    status[kConvergedStatus] = (scalars[kRNorm] <= scalars[kAtol]) ? 1 : 0;
}

__device__ bool bad_scalar(float value)
{
    return !isfinite(value) || fabsf(value) <= kTiny;
}

__global__ void compute_beta_kernel_f32(float* scalars, int32_t* status)
{
    if (status[kFailureStatus] != kNoFailure) return;
    const float rho = scalars[kRho];
    const float rho_prev = scalars[kRhoPrev];
    const float alpha = scalars[kAlpha];
    const float omega = scalars[kOmega];
    if (bad_scalar(rho)) {
        status[kFailureStatus] = kRhoBreakdown;
        scalars[kBeta] = 0.0f;
        return;
    }
    scalars[kBeta] = (rho / rho_prev) * (alpha / omega);
}

__global__ void update_p_kernel_f32(int32_t n,
                                    const float* r,
                                    const float* v,
                                    const float* scalars,
                                    float* p)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float beta = scalars[kBeta];
        const float omega = scalars[kOmega];
        p[i] = r[i] + beta * (p[i] - omega * v[i]);
    }
}

__global__ void compute_alpha_kernel_f32(float* scalars, int32_t* status)
{
    if (status[kFailureStatus] != kNoFailure) return;
    const float alpha_den = scalars[kAlphaDen];
    if (bad_scalar(alpha_den)) {
        status[kFailureStatus] = kAlphaBreakdown;
        scalars[kAlpha] = 0.0f;
        return;
    }
    scalars[kAlpha] = scalars[kRho] / alpha_den;
}

__global__ void s_update_norm_partial_kernel_f32(int32_t n,
                                                 const float* r,
                                                 const float* v,
                                                 const float* scalars,
                                                 float* s,
                                                 float* partial)
{
    __shared__ float values[kBlockSize];
    const int32_t tid = threadIdx.x;
    const int32_t i = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    if (i < n) {
        const float value = r[i] - scalars[kAlpha] * v[i];
        s[i] = value;
        sum = value * value;
    }
    values[tid] = sum;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) values[tid] += values[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = values[0];
}

__global__ void finalize_s_norm_kernel_f32(int32_t n_partials,
                                           const float* partial,
                                           float* scalars,
                                           int32_t* status)
{
    float sum = 0.0f;
    for (int32_t i = 0; i < n_partials; ++i) sum += partial[i];
    const float norm = sqrtf(sum);
    scalars[kSNorm] = norm;
    scalars[kRelativeResidual] = norm / scalars[kRhsNorm];
    status[kConvergedStatus] =
        (status[kFailureStatus] == kNoFailure && norm <= scalars[kAtol]) ? 1 : 0;
}

__global__ void x_alpha_update_kernel_f32(int32_t n,
                                          const float* p,
                                          const float* scalars,
                                          float* x)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += scalars[kAlpha] * p[i];
}

__global__ void two_dot_partial_kernel_f32(int32_t n,
                                           const float* t,
                                           const float* s,
                                           float* partial_tt,
                                           float* partial_ts)
{
    __shared__ float tt_values[kBlockSize];
    __shared__ float ts_values[kBlockSize];
    const int32_t tid = threadIdx.x;
    const int32_t i = blockIdx.x * blockDim.x + tid;
    float tt = 0.0f;
    float ts = 0.0f;
    if (i < n) {
        const float tv = t[i];
        tt = tv * tv;
        ts = tv * s[i];
    }
    tt_values[tid] = tt;
    ts_values[tid] = ts;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tt_values[tid] += tt_values[tid + stride];
            ts_values[tid] += ts_values[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_tt[blockIdx.x] = tt_values[0];
        partial_ts[blockIdx.x] = ts_values[0];
    }
}

__global__ void finalize_omega_kernel_f32(int32_t n_partials,
                                          const float* partial_tt,
                                          const float* partial_ts,
                                          float* scalars,
                                          int32_t* status)
{
    if (status[kFailureStatus] != kNoFailure) return;
    float tt = 0.0f;
    float ts = 0.0f;
    for (int32_t i = 0; i < n_partials; ++i) {
        tt += partial_tt[i];
        ts += partial_ts[i];
    }
    scalars[kTT] = tt;
    scalars[kTS] = ts;
    if (!isfinite(tt) || tt <= kTiny) {
        status[kFailureStatus] = kOmegaDenominatorBreakdown;
        scalars[kOmega] = 0.0f;
        return;
    }
    const float omega = ts / tt;
    scalars[kOmega] = omega;
    if (bad_scalar(omega)) status[kFailureStatus] = kOmegaBreakdown;
}

__global__ void x_r_update_norm_partial_kernel_f32(int32_t n,
                                                   const float* p_direction,
                                                   const float* s_direction,
                                                   const float* s_residual,
                                                   const float* t,
                                                   const float* scalars,
                                                   float* x,
                                                   float* r,
                                                   float* partial)
{
    __shared__ float values[kBlockSize];
    const int32_t tid = threadIdx.x;
    const int32_t i = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    if (i < n) {
        const float alpha = scalars[kAlpha];
        const float omega = scalars[kOmega];
        x[i] += alpha * p_direction[i] + omega * s_direction[i];
        const float value = s_residual[i] - omega * t[i];
        r[i] = value;
        sum = value * value;
    }
    values[tid] = sum;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) values[tid] += values[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = values[0];
}

__global__ void finalize_r_norm_kernel_f32(int32_t n_partials,
                                           const float* partial,
                                           float* scalars,
                                           int32_t* status)
{
    float sum = 0.0f;
    for (int32_t i = 0; i < n_partials; ++i) sum += partial[i];
    const float norm = sqrtf(sum);
    scalars[kRNorm] = norm;
    scalars[kRelativeResidual] = norm / scalars[kRhsNorm];
    status[kConvergedStatus] =
        (status[kFailureStatus] == kNoFailure && norm <= scalars[kAtol]) ? 1 : 0;
}

__global__ void carry_rho_kernel_f32(float* scalars)
{
    scalars[kRhoPrev] = scalars[kRho];
}

}  // namespace

SchurBicgstabSolverF32::SchurBicgstabSolverF32()
{
    cublas_check(cublasCreate(&cublas_), "cublasCreate failed");
    cublas_check(cublasSetPointerMode(cublas_, CUBLAS_POINTER_MODE_DEVICE),
                 "cublasSetPointerMode failed");
}

SchurBicgstabSolverF32::~SchurBicgstabSolverF32()
{
    if (cublas_ != nullptr) cublasDestroy(cublas_);
}

void SchurBicgstabSolverF32::ensure_workspace(int32_t n)
{
    const std::size_t count = static_cast<std::size_t>(n);
    if (rhs_s_.size() != count) rhs_s_.resize(count);
    if (dvm_.size() != count) dvm_.resize(count);
    if (r_.size() != count) r_.resize(count);
    if (r_hat_.size() != count) r_hat_.resize(count);
    if (p_.size() != count) p_.resize(count);
    if (v_.size() != count) v_.resize(count);
    if (s_.size() != count) s_.resize(count);
    if (t_.size() != count) t_.resize(count);
    if (p_hat_.size() != count) p_hat_.resize(count);
    if (s_hat_.size() != count) s_hat_.resize(count);
    if (ax_.size() != count) ax_.resize(count);
    if (d_scalars_.size() != static_cast<std::size_t>(kScalarCount)) {
        d_scalars_.resize(static_cast<std::size_t>(kScalarCount));
    }
    const std::size_t partial_count = static_cast<std::size_t>(grid_for(n));
    if (d_reduce1_.size() != partial_count) d_reduce1_.resize(partial_count);
    if (d_reduce2_.size() != partial_count) d_reduce2_.resize(partial_count);
    if (d_status_.size() != static_cast<std::size_t>(kStatusCount)) {
        d_status_.resize(static_cast<std::size_t>(kStatusCount));
    }
    h_scalars_.resize(static_cast<std::size_t>(kScalarCount));
    h_status_.resize(static_cast<std::size_t>(kStatusCount));
}

void SchurBicgstabSolverF32::dot_to_device(int32_t n,
                                           const float* x,
                                           const float* y,
                                           float* out_device)
{
    cublas_check(cublasSdot(cublas_, n, x, 1, y, 1, out_device), "cublasSdot failed");
}

SchurBicgstabStats SchurBicgstabSolverF32::solve(ImplicitSchurOperatorF32& op,
                                                 const double* rhs_full_device,
                                                 double* dx_full_device,
                                                 const SchurBicgstabOptions& options)
{
    if (rhs_full_device == nullptr || dx_full_device == nullptr ||
        options.relative_tolerance <= 0.0 || options.max_iterations <= 0 ||
        op.n_pq() <= 0) {
        throw std::runtime_error("SchurBicgstabSolverF32::solve received invalid input");
    }

    const int32_t n = op.n_pq();
    const bool use_schur_preconditioner = op.schur_preconditioner_enabled();
    ensure_workspace(n);
    const int32_t grid = grid_for(n);
    const int32_t partial_count = grid;

    SchurBicgstabStats stats;
    SchurOperatorStats op_stats;
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto solve_start = std::chrono::steady_clock::now();

    op.build_rhs(rhs_full_device, rhs_s_.data(), op_stats, options.collect_timing_breakdown);

    set_zero_kernel_f32<<<grid, kBlockSize>>>(n, dvm_.data());
    set_zero_kernel_f32<<<grid, kBlockSize>>>(n, p_.data());
    set_zero_kernel_f32<<<grid, kBlockSize>>>(n, v_.data());
    copy_kernel_f32<<<grid, kBlockSize>>>(n, rhs_s_.data(), r_.data());
    copy_kernel_f32<<<grid, kBlockSize>>>(n, rhs_s_.data(), r_hat_.data());
    CUDA_CHECK(cudaGetLastError());

    auto copy_state_to_host = [&]() {
        d_status_.copyTo(h_status_.data(), static_cast<std::size_t>(kStatusCount));
        d_scalars_.copyTo(h_scalars_.data(), static_cast<std::size_t>(kScalarCount));
    };
    auto timed_copy_state_to_host = [&]() {
        time_stage(stats.timing.reduction_sec, options.collect_timing_breakdown, [&]() {
            copy_state_to_host();
        });
    };
    auto timed_dot_to_device = [&](const float* x, const float* y, int32_t scalar_slot) {
        time_stage(stats.timing.reduction_sec, options.collect_timing_breakdown, [&]() {
            dot_to_device(n, x, y, d_scalars_.data() + scalar_slot);
        });
        ++stats.reduction_calls;
    };
    auto timed_norm_to_device = [&](const float* x, int32_t scalar_slot) {
        time_stage(stats.timing.reduction_sec, options.collect_timing_breakdown, [&]() {
            norm_partial_kernel_f32<<<partial_count, kBlockSize>>>(n, x, d_reduce1_.data());
            finalize_norm_kernel_f32<<<1, 1>>>(
                partial_count, d_reduce1_.data(), d_scalars_.data(), scalar_slot);
            CUDA_CHECK(cudaGetLastError());
        });
        ++stats.reduction_calls;
    };

    timed_norm_to_device(r_.data(), kRNorm);
    init_scalar_state_kernel_f32<<<1, 1>>>(
        static_cast<float>(options.relative_tolerance), d_scalars_.data(), d_status_.data());
    CUDA_CHECK(cudaGetLastError());
    timed_copy_state_to_host();

    stats.initial_residual_norm = h_scalars_[kRNorm];
    const float rhs_norm = h_scalars_[kRhsNorm];
    stats.final_residual_norm = stats.initial_residual_norm;
    stats.relative_residual_norm = h_scalars_[kRelativeResidual];
    if (h_status_[kConvergedStatus] != 0) stats.converged = true;

    const auto iteration_start = std::chrono::steady_clock::now();
    for (int32_t iter = 1; !stats.converged && iter <= options.max_iterations; ++iter) {
        timed_dot_to_device(r_hat_.data(), r_.data(), kRho);
        compute_beta_kernel_f32<<<1, 1>>>(d_scalars_.data(), d_status_.data());
        CUDA_CHECK(cudaGetLastError());
        time_stage(stats.timing.vector_update_sec, options.collect_timing_breakdown, [&]() {
            update_p_kernel_f32<<<grid, kBlockSize>>>(
                n, r_.data(), v_.data(), d_scalars_.data(), p_.data());
            CUDA_CHECK(cudaGetLastError());
        });

        const float* p_for_matvec = p_.data();
        if (use_schur_preconditioner) {
            op.apply_schur_preconditioner(
                p_.data(), p_hat_.data(), op_stats, options.collect_timing_breakdown);
            p_for_matvec = p_hat_.data();
        }
        op.apply(p_for_matvec, v_.data(), op_stats, options.collect_timing_breakdown);
        timed_dot_to_device(r_hat_.data(), v_.data(), kAlphaDen);
        compute_alpha_kernel_f32<<<1, 1>>>(d_scalars_.data(), d_status_.data());
        CUDA_CHECK(cudaGetLastError());

        time_stage(stats.timing.reduction_sec, options.collect_timing_breakdown, [&]() {
            s_update_norm_partial_kernel_f32<<<grid, kBlockSize>>>(
                n, r_.data(), v_.data(), d_scalars_.data(), s_.data(), d_reduce1_.data());
            finalize_s_norm_kernel_f32<<<1, 1>>>(
                partial_count, d_reduce1_.data(), d_scalars_.data(), d_status_.data());
            CUDA_CHECK(cudaGetLastError());
        });
        ++stats.reduction_calls;
        timed_copy_state_to_host();
        if (h_status_[kFailureStatus] != kNoFailure) {
            stats.iterations = iter;
            break;
        }
        if (h_status_[kConvergedStatus] != 0) {
            time_stage(stats.timing.vector_update_sec, options.collect_timing_breakdown, [&]() {
                x_alpha_update_kernel_f32<<<grid, kBlockSize>>>(
                    n, p_for_matvec, d_scalars_.data(), dvm_.data());
                CUDA_CHECK(cudaGetLastError());
            });
            stats.iterations = iter;
            stats.final_residual_norm = h_scalars_[kSNorm];
            stats.relative_residual_norm = h_scalars_[kRelativeResidual];
            stats.converged = true;
            break;
        }

        const float* s_for_matvec = s_.data();
        if (use_schur_preconditioner) {
            op.apply_schur_preconditioner(
                s_.data(), s_hat_.data(), op_stats, options.collect_timing_breakdown);
            s_for_matvec = s_hat_.data();
        }
        op.apply(s_for_matvec, t_.data(), op_stats, options.collect_timing_breakdown);
        time_stage(stats.timing.reduction_sec, options.collect_timing_breakdown, [&]() {
            two_dot_partial_kernel_f32<<<grid, kBlockSize>>>(
                n, t_.data(), s_.data(), d_reduce1_.data(), d_reduce2_.data());
            finalize_omega_kernel_f32<<<1, 1>>>(
                partial_count, d_reduce1_.data(), d_reduce2_.data(),
                d_scalars_.data(), d_status_.data());
            CUDA_CHECK(cudaGetLastError());
        });
        stats.reduction_calls += 2;
        time_stage(stats.timing.reduction_sec, options.collect_timing_breakdown, [&]() {
            x_r_update_norm_partial_kernel_f32<<<grid, kBlockSize>>>(
                n, p_for_matvec, s_for_matvec, s_.data(), t_.data(), d_scalars_.data(),
                dvm_.data(), r_.data(), d_reduce1_.data());
            finalize_r_norm_kernel_f32<<<1, 1>>>(
                partial_count, d_reduce1_.data(), d_scalars_.data(), d_status_.data());
            CUDA_CHECK(cudaGetLastError());
        });
        ++stats.reduction_calls;
        timed_copy_state_to_host();

        stats.iterations = iter;
        stats.final_residual_norm = h_scalars_[kRNorm];
        stats.relative_residual_norm = h_scalars_[kRelativeResidual];
        if (h_status_[kFailureStatus] != kNoFailure) break;
        if (h_status_[kConvergedStatus] != 0) {
            stats.converged = true;
            break;
        }
        carry_rho_kernel_f32<<<1, 1>>>(d_scalars_.data());
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto iteration_end = std::chrono::steady_clock::now();

    time_stage(stats.timing.residual_refresh_sec, options.collect_timing_breakdown, [&]() {
        op.apply(dvm_.data(), ax_.data(), op_stats, options.collect_timing_breakdown);
        residual_kernel_f32<<<grid, kBlockSize>>>(n, rhs_s_.data(), ax_.data(), r_.data());
        norm_partial_kernel_f32<<<partial_count, kBlockSize>>>(n, r_.data(), d_reduce1_.data());
        finalize_norm_kernel_f32<<<1, 1>>>(
            partial_count, d_reduce1_.data(), d_scalars_.data(), kRNorm);
        CUDA_CHECK(cudaGetLastError());
    });
    ++stats.reduction_calls;
    timed_copy_state_to_host();
    stats.final_residual_norm = h_scalars_[kRNorm];
    stats.relative_residual_norm = stats.final_residual_norm / rhs_norm;
    stats.converged = stats.relative_residual_norm <= options.relative_tolerance;

    op.recover_solution(rhs_full_device, dvm_.data(), dx_full_device,
                        op_stats, options.collect_timing_breakdown);

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto solve_end = std::chrono::steady_clock::now();
    stats.solve_sec = std::chrono::duration<double>(solve_end - solve_start).count();
    if (stats.iterations > 0) {
        const double loop_sec =
            std::chrono::duration<double>(iteration_end - iteration_start).count();
        stats.avg_iteration_sec = loop_sec / static_cast<double>(stats.iterations);
    }
    if (!stats.converged && stats.failure_reason.empty()) {
        switch (h_status_[kFailureStatus]) {
        case kRhoBreakdown: stats.failure_reason = "rho_breakdown"; break;
        case kAlphaBreakdown: stats.failure_reason = "alpha_breakdown"; break;
        case kOmegaDenominatorBreakdown: stats.failure_reason = "omega_denominator_breakdown"; break;
        case kOmegaBreakdown: stats.failure_reason = "omega_breakdown"; break;
        default: stats.failure_reason = "max_iterations"; break;
        }
    }

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
    stats.timing.reduction_sec += 0.0;
    stats.timing.vector_update_sec += op_stats.vector_update_sec;
    return stats;
}

}  // namespace exp_20260415::block_ilu
