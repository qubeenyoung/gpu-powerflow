#include "cuiter/kernels/gmres_kernels.hpp"

#include "cuiter/common/cuda_utils.hpp"

#include <cstdint>

namespace cuiter::kernels {
namespace {

constexpr int32_t kBlockSize = 256;

int32_t grid_for(int32_t n)
{
    return (n + kBlockSize - 1) / kBlockSize;
}

__device__ double atomic_add_double(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__global__ void set_zero_kernel(int32_t n, double* x)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = 0.0;
    }
}

__global__ void set_constant_kernel(int32_t n, double value, double* x)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = value;
    }
}

__global__ void copy_kernel(int32_t n, const double* src, double* dst)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

__global__ void multiply_by_scale_kernel(int32_t n,
                                         const double* __restrict__ scale,
                                         const double* __restrict__ src,
                                         double* __restrict__ dst)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = scale[i] * src[i];
    }
}

__global__ void scale_copy_kernel(int32_t n, double alpha, const double* src, double* dst)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = alpha * src[i];
    }
}

__global__ void scale_copy_device_scalar_kernel(int32_t n,
                                                const double* __restrict__ alpha,
                                                const double* __restrict__ src,
                                                double* __restrict__ dst)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (*alpha) * src[i];
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

    const std::size_t stride = static_cast<std::size_t>(n);
    double sum = 0.0;
    for (int32_t col = 0; col < basis_count; ++col) {
        sum += z_basis[static_cast<std::size_t>(col) * stride + row] * y[col];
    }
    x[row] += sum;
}

__global__ void residual_kernel(int32_t n, const double* rhs, const double* ax, double* r)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        r[i] = rhs[i] - ax[i];
    }
}

__global__ void residual_scaled_kernel(int32_t n,
                                       const double* __restrict__ rhs,
                                       const double* __restrict__ ax,
                                       double alpha,
                                       double* __restrict__ r)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        r[i] = rhs[i] - alpha * ax[i];
    }
}

__global__ void residual_scaled_device_scalar_kernel(int32_t n,
                                                     const double* __restrict__ rhs,
                                                     const double* __restrict__ ax,
                                                     const double* __restrict__ alpha,
                                                     double* __restrict__ r)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        r[i] = rhs[i] - (*alpha) * ax[i];
    }
}

__global__ void residual_two_scaled_kernel(int32_t n,
                                           const double* __restrict__ rhs,
                                           const double* __restrict__ w0,
                                           double alpha0,
                                           const double* __restrict__ w1,
                                           double alpha1,
                                           double* __restrict__ r)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        r[i] = rhs[i] - alpha0 * w0[i] - alpha1 * w1[i];
    }
}

__global__ void linear_combination2_kernel(int32_t n,
                                           double alpha0,
                                           const double* __restrict__ x0,
                                           double alpha1,
                                           const double* __restrict__ x1,
                                           double* __restrict__ out)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = alpha0 * x0[i] + alpha1 * x1[i];
    }
}

__global__ void bicgstab_update_p_kernel(int32_t n,
                                         const double* __restrict__ r,
                                         double beta,
                                         double omega,
                                         const double* __restrict__ v,
                                         double* __restrict__ p)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = r[i] + beta * (p[i] - omega * v[i]);
    }
}

__global__ void bicgstab_update_p_device_scalar_kernel(int32_t n,
                                                       const double* __restrict__ r,
                                                       const double* __restrict__ beta,
                                                       const double* __restrict__ omega,
                                                       const double* __restrict__ v,
                                                       double* __restrict__ p)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = r[i] + (*beta) * (p[i] - (*omega) * v[i]);
    }
}

__global__ void bicgstab_update_x_r_kernel(int32_t n,
                                           double alpha,
                                           const double* __restrict__ p_hat,
                                           double omega,
                                           const double* __restrict__ s_hat,
                                           const double* __restrict__ s,
                                           const double* __restrict__ t,
                                           double* __restrict__ x,
                                           double* __restrict__ r)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += alpha * p_hat[i] + omega * s_hat[i];
        r[i] = s[i] - omega * t[i];
    }
}

__global__ void bicgstab_update_x_r_device_scalar_kernel(int32_t n,
                                                         const double* __restrict__ alpha,
                                                         const double* __restrict__ p_hat,
                                                         const double* __restrict__ omega,
                                                         const double* __restrict__ s_hat,
                                                         const double* __restrict__ s,
                                                         const double* __restrict__ t,
                                                         double* __restrict__ x,
                                                         double* __restrict__ r)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += (*alpha) * p_hat[i] + (*omega) * s_hat[i];
        r[i] = s[i] - (*omega) * t[i];
    }
}

__global__ void bicgstab_compute_beta_kernel(double* __restrict__ scalars)
{
    scalars[7] = (scalars[0] / scalars[4]) * (scalars[5] / scalars[6]);
}

__global__ void bicgstab_compute_alpha_kernel(double* __restrict__ scalars)
{
    scalars[5] = scalars[0] / scalars[1];
}

__global__ void bicgstab_compute_omega_and_advance_kernel(double* scalars)
{
    scalars[6] = scalars[2] / scalars[3];
    scalars[4] = scalars[0];
}

__global__ void mr1_two_dot_reduction_kernel(int32_t n,
                                             const double* __restrict__ w,
                                             const double* __restrict__ r,
                                             double* __restrict__ dot_wr_dot_ww)
{
    __shared__ double partial_wr[kBlockSize];
    __shared__ double partial_ww[kBlockSize];

    const int32_t tid = threadIdx.x;
    const int32_t i = blockIdx.x * blockDim.x + tid;
    double wr = 0.0;
    double ww = 0.0;
    if (i < n) {
        const double wi = w[i];
        wr = wi * r[i];
        ww = wi * wi;
    }
    partial_wr[tid] = wr;
    partial_ww[tid] = ww;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_wr[tid] += partial_wr[tid + stride];
            partial_ww[tid] += partial_ww[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomic_add_double(dot_wr_dot_ww, partial_wr[0]);
        atomic_add_double(dot_wr_dot_ww + 1, partial_ww[0]);
    }
}

__global__ void mr2_five_dot_reduction_kernel(int32_t n,
                                              const double* __restrict__ w0,
                                              const double* __restrict__ w1,
                                              const double* __restrict__ r,
                                              double* __restrict__ dots)
{
    __shared__ double partial_w0r[kBlockSize];
    __shared__ double partial_w1r[kBlockSize];
    __shared__ double partial_w0w0[kBlockSize];
    __shared__ double partial_w0w1[kBlockSize];
    __shared__ double partial_w1w1[kBlockSize];

    const int32_t tid = threadIdx.x;
    const int32_t i = blockIdx.x * blockDim.x + tid;
    double w0r = 0.0;
    double w1r = 0.0;
    double w0w0 = 0.0;
    double w0w1 = 0.0;
    double w1w1 = 0.0;
    if (i < n) {
        const double w0i = w0[i];
        const double w1i = w1[i];
        const double ri = r[i];
        w0r = w0i * ri;
        w1r = w1i * ri;
        w0w0 = w0i * w0i;
        w0w1 = w0i * w1i;
        w1w1 = w1i * w1i;
    }
    partial_w0r[tid] = w0r;
    partial_w1r[tid] = w1r;
    partial_w0w0[tid] = w0w0;
    partial_w0w1[tid] = w0w1;
    partial_w1w1[tid] = w1w1;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_w0r[tid] += partial_w0r[tid + stride];
            partial_w1r[tid] += partial_w1r[tid + stride];
            partial_w0w0[tid] += partial_w0w0[tid + stride];
            partial_w0w1[tid] += partial_w0w1[tid + stride];
            partial_w1w1[tid] += partial_w1w1[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomic_add_double(dots, partial_w0r[0]);
        atomic_add_double(dots + 1, partial_w1r[0]);
        atomic_add_double(dots + 2, partial_w0w0[0]);
        atomic_add_double(dots + 3, partial_w0w1[0]);
        atomic_add_double(dots + 4, partial_w1w1[0]);
    }
}

__global__ void csr_spmv_kernel(int32_t rows,
                                const int32_t* __restrict__ row_ptr,
                                const int32_t* __restrict__ col_idx,
                                const double* __restrict__ values,
                                const double* __restrict__ x,
                                double* __restrict__ y)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    for (int32_t pos = row_ptr[row]; pos < row_ptr[row + 1]; ++pos) {
        sum += values[pos] * x[col_idx[pos]];
    }
    y[row] = sum;
}

__global__ void permute_vector_kernel(int32_t n,
                                      const int32_t* __restrict__ new_to_old,
                                      const double* __restrict__ x_old,
                                      double* __restrict__ x_new)
{
    const int32_t new_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_index < n) {
        x_new[new_index] = x_old[new_to_old[new_index]];
    }
}

__global__ void unpermute_vector_kernel(int32_t n,
                                        const int32_t* __restrict__ new_to_old,
                                        const double* __restrict__ x_new,
                                        double* __restrict__ x_old)
{
    const int32_t new_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_index < n) {
        x_old[new_to_old[new_index]] = x_new[new_index];
    }
}

__global__ void scaled_row_l2_norms_kernel(int32_t rows,
                                           const int32_t* __restrict__ row_ptr,
                                           const int32_t* __restrict__ col_idx,
                                           const double* __restrict__ values,
                                           const double* __restrict__ row_scale,
                                           const double* __restrict__ col_scale,
                                           double* __restrict__ row_norms)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    const double rs = row_scale[row];
    for (int32_t pos = row_ptr[row]; pos < row_ptr[row + 1]; ++pos) {
        const double value = rs * values[pos] * col_scale[col_idx[pos]];
        sum += value * value;
    }
    row_norms[row] = sqrt(sum);
}

__global__ void scaled_col_l2_sums_kernel(int32_t rows,
                                          const int32_t* __restrict__ row_ptr,
                                          const int32_t* __restrict__ col_idx,
                                          const double* __restrict__ values,
                                          const double* __restrict__ row_scale,
                                          const double* __restrict__ col_scale,
                                          double* __restrict__ col_sums)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const double rs = row_scale[row];
    for (int32_t pos = row_ptr[row]; pos < row_ptr[row + 1]; ++pos) {
        const int32_t col = col_idx[pos];
        const double value = rs * values[pos] * col_scale[col];
        atomic_add_double(col_sums + col, value * value);
    }
}

__global__ void sqrt_norms_kernel(int32_t n, double* norms)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        norms[i] = sqrt(norms[i]);
    }
}

__global__ void update_ruiz_scale_kernel(int32_t n,
                                         const double* __restrict__ norms,
                                         double eps,
                                         double clamp,
                                         double* __restrict__ scale)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const double norm = fmax(norms[i], eps);
    const double factor = rsqrt(norm);
    const double lo = 1.0 / clamp;
    const double hi = clamp;
    scale[i] = fmin(hi, fmax(lo, scale[i] * factor));
}

__global__ void apply_scaled_csr_values_kernel(int32_t rows,
                                               const int32_t* __restrict__ row_ptr,
                                               const int32_t* __restrict__ col_idx,
                                               const double* __restrict__ values,
                                               const double* __restrict__ row_scale,
                                               const double* __restrict__ col_scale,
                                               double* __restrict__ scaled_values)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const double rs = row_scale[row];
    for (int32_t pos = row_ptr[row]; pos < row_ptr[row + 1]; ++pos) {
        scaled_values[pos] = rs * values[pos] * col_scale[col_idx[pos]];
    }
}

}  // namespace

void launch_set_zero(int32_t n, double* x)
{
    set_zero_kernel<<<grid_for(n), kBlockSize>>>(n, x);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_set_constant(int32_t n, double value, double* x)
{
    set_constant_kernel<<<grid_for(n), kBlockSize>>>(n, value, x);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_copy(int32_t n, const double* src, double* dst)
{
    copy_kernel<<<grid_for(n), kBlockSize>>>(n, src, dst);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_scale_copy(int32_t n, double alpha, const double* src, double* dst)
{
    scale_copy_kernel<<<grid_for(n), kBlockSize>>>(n, alpha, src, dst);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_multiply_by_scale(int32_t n, const double* scale, const double* src, double* dst)
{
    multiply_by_scale_kernel<<<grid_for(n), kBlockSize>>>(n, scale, src, dst);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_scale_copy_device_scalar(int32_t n, const double* alpha, const double* src, double* dst)
{
    scale_copy_device_scalar_kernel<<<grid_for(n), kBlockSize>>>(n, alpha, src, dst);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_sub_scaled_device_scalar(int32_t n, const double* basis, const double* alpha, double* target)
{
    sub_scaled_device_scalar_kernel<<<grid_for(n), kBlockSize>>>(n, basis, alpha, target);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_combine_solution(int32_t n,
                             int32_t basis_count,
                             const double* z_basis,
                             const double* y,
                             double* x)
{
    combine_solution_kernel<<<grid_for(n), kBlockSize>>>(n, basis_count, z_basis, y, x);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_residual(int32_t n, const double* rhs, const double* ax, double* r)
{
    residual_kernel<<<grid_for(n), kBlockSize>>>(n, rhs, ax, r);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_residual_scaled(int32_t n, const double* rhs, const double* ax, double alpha, double* r)
{
    residual_scaled_kernel<<<grid_for(n), kBlockSize>>>(n, rhs, ax, alpha, r);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_residual_scaled_device_scalar(int32_t n,
                                          const double* rhs,
                                          const double* ax,
                                          const double* alpha,
                                          double* r)
{
    residual_scaled_device_scalar_kernel<<<grid_for(n), kBlockSize>>>(n, rhs, ax, alpha, r);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_residual_two_scaled(int32_t n,
                                const double* rhs,
                                const double* w0,
                                double alpha0,
                                const double* w1,
                                double alpha1,
                                double* r)
{
    residual_two_scaled_kernel<<<grid_for(n), kBlockSize>>>(
        n, rhs, w0, alpha0, w1, alpha1, r);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_linear_combination2(int32_t n,
                                double alpha0,
                                const double* x0,
                                double alpha1,
                                const double* x1,
                                double* out)
{
    linear_combination2_kernel<<<grid_for(n), kBlockSize>>>(
        n, alpha0, x0, alpha1, x1, out);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_bicgstab_update_p(int32_t n,
                              const double* r,
                              double beta,
                              double omega,
                              const double* v,
                              double* p)
{
    bicgstab_update_p_kernel<<<grid_for(n), kBlockSize>>>(n, r, beta, omega, v, p);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_bicgstab_update_p_device_scalar(int32_t n,
                                            const double* r,
                                            const double* beta,
                                            const double* omega,
                                            const double* v,
                                            double* p)
{
    bicgstab_update_p_device_scalar_kernel<<<grid_for(n), kBlockSize>>>(
        n, r, beta, omega, v, p);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_bicgstab_update_x_r(int32_t n,
                                double alpha,
                                const double* p_hat,
                                double omega,
                                const double* s_hat,
                                const double* s,
                                const double* t,
                                double* x,
                                double* r)
{
    bicgstab_update_x_r_kernel<<<grid_for(n), kBlockSize>>>(
        n, alpha, p_hat, omega, s_hat, s, t, x, r);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_bicgstab_update_x_r_device_scalar(int32_t n,
                                              const double* alpha,
                                              const double* p_hat,
                                              const double* omega,
                                              const double* s_hat,
                                              const double* s,
                                              const double* t,
                                              double* x,
                                              double* r)
{
    bicgstab_update_x_r_device_scalar_kernel<<<grid_for(n), kBlockSize>>>(
        n, alpha, p_hat, omega, s_hat, s, t, x, r);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_bicgstab_compute_beta(double* scalars)
{
    bicgstab_compute_beta_kernel<<<1, 1>>>(scalars);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_bicgstab_compute_alpha(double* scalars)
{
    bicgstab_compute_alpha_kernel<<<1, 1>>>(scalars);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_bicgstab_compute_omega_and_advance(double* scalars)
{
    bicgstab_compute_omega_and_advance_kernel<<<1, 1>>>(scalars);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_mr1_two_dot_reduction(int32_t n,
                                  const double* w,
                                  const double* r,
                                  double* dot_wr_dot_ww)
{
    CUITER_CUDA_CHECK(cudaMemset(dot_wr_dot_ww, 0, 2 * sizeof(double)));
    mr1_two_dot_reduction_kernel<<<grid_for(n), kBlockSize>>>(n, w, r, dot_wr_dot_ww);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_mr2_five_dot_reduction(int32_t n,
                                   const double* w0,
                                   const double* w1,
                                   const double* r,
                                   double* dots)
{
    CUITER_CUDA_CHECK(cudaMemset(dots, 0, 5 * sizeof(double)));
    mr2_five_dot_reduction_kernel<<<grid_for(n), kBlockSize>>>(n, w0, w1, r, dots);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_csr_spmv(int32_t rows,
                     const int32_t* row_ptr,
                     const int32_t* col_idx,
                     const double* values,
                     const double* x,
                     double* y)
{
    csr_spmv_kernel<<<grid_for(rows), kBlockSize>>>(rows, row_ptr, col_idx, values, x, y);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_permute_vector(int32_t n, const int32_t* new_to_old, const double* x_old, double* x_new)
{
    permute_vector_kernel<<<grid_for(n), kBlockSize>>>(n, new_to_old, x_old, x_new);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_unpermute_vector(int32_t n, const int32_t* new_to_old, const double* x_new, double* x_old)
{
    unpermute_vector_kernel<<<grid_for(n), kBlockSize>>>(n, new_to_old, x_new, x_old);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_compute_scaled_row_l2_norms(int32_t rows,
                                        const int32_t* row_ptr,
                                        const int32_t* col_idx,
                                        const double* values,
                                        const double* row_scale,
                                        const double* col_scale,
                                        double* row_norms)
{
    scaled_row_l2_norms_kernel<<<grid_for(rows), kBlockSize>>>(
        rows, row_ptr, col_idx, values, row_scale, col_scale, row_norms);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_compute_scaled_col_l2_norms(int32_t rows,
                                        int32_t cols,
                                        const int32_t* row_ptr,
                                        const int32_t* col_idx,
                                        const double* values,
                                        const double* row_scale,
                                        const double* col_scale,
                                        double* col_norms)
{
    CUITER_CUDA_CHECK(cudaMemset(col_norms, 0, static_cast<std::size_t>(cols) * sizeof(double)));
    scaled_col_l2_sums_kernel<<<grid_for(rows), kBlockSize>>>(
        rows, row_ptr, col_idx, values, row_scale, col_scale, col_norms);
    sqrt_norms_kernel<<<grid_for(cols), kBlockSize>>>(cols, col_norms);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_update_ruiz_scale(int32_t n,
                              const double* norms,
                              double eps,
                              double clamp,
                              double* scale)
{
    update_ruiz_scale_kernel<<<grid_for(n), kBlockSize>>>(n, norms, eps, clamp, scale);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_apply_scaled_csr_values(int32_t rows,
                                    const int32_t* row_ptr,
                                    const int32_t* col_idx,
                                    const double* values,
                                    const double* row_scale,
                                    const double* col_scale,
                                    double* scaled_values)
{
    apply_scaled_csr_values_kernel<<<grid_for(rows), kBlockSize>>>(
        rows, row_ptr, col_idx, values, row_scale, col_scale, scaled_values);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuiter::kernels
