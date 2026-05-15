#include "cupf_minimal/nr_kernels.hpp"

// experimental minimal cuPF NR port

#include "cuiter/common/cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace cupf_minimal {
namespace {

constexpr int32_t kBlock = 256;

int32_t grid_for(int32_t n)
{
    return (n + kBlock - 1) / kBlock;
}

template <int32_t LANES>
__global__ void compute_ibus_kernel(int32_t n_bus,
                                    const int32_t* __restrict__ y_row_ptr,
                                    const int32_t* __restrict__ y_col,
                                    const double* __restrict__ y_re,
                                    const double* __restrict__ y_im,
                                    const double* __restrict__ v_re,
                                    const double* __restrict__ v_im,
                                    double* __restrict__ ibus_re,
                                    double* __restrict__ ibus_im)
{
    const int32_t row = blockIdx.x;
    const int32_t lane = threadIdx.x;
    if (row >= n_bus) {
        return;
    }

    double acc_re = 0.0;
    double acc_im = 0.0;
    for (int32_t k = y_row_ptr[row] + lane; k < y_row_ptr[row + 1]; k += LANES) {
        const int32_t col = y_col[k];
        const double yr = y_re[k];
        const double yi = y_im[k];
        const double vr = v_re[col];
        const double vi = v_im[col];
        acc_re += yr * vr - yi * vi;
        acc_im += yr * vi + yi * vr;
    }

    constexpr uint32_t mask = 0xffffffffu;
    for (int32_t offset = LANES / 2; offset > 0; offset >>= 1) {
        acc_re += __shfl_down_sync(mask, acc_re, offset, LANES);
        acc_im += __shfl_down_sync(mask, acc_im, offset, LANES);
    }
    if (lane == 0) {
        ibus_re[row] = acc_re;
        ibus_im[row] = acc_im;
    }
}

__global__ void compute_mismatch_from_ibus_kernel(int32_t dimF,
                                                  int32_t n_bus,
                                                  int32_t n_pv,
                                                  int32_t n_pq,
                                                  const double* __restrict__ v_re,
                                                  const double* __restrict__ v_im,
                                                  const double* __restrict__ ibus_re,
                                                  const double* __restrict__ ibus_im,
                                                  const double* __restrict__ sbus_re,
                                                  const double* __restrict__ sbus_im,
                                                  const int32_t* __restrict__ pv,
                                                  const int32_t* __restrict__ pq,
                                                  double* __restrict__ F)
{
    const int32_t local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= dimF) {
        return;
    }

    int32_t bus = 0;
    bool take_imag = false;
    if (local < n_pv) {
        bus = pv[local];
    } else if (local < n_pv + n_pq) {
        bus = pq[local - n_pv];
    } else {
        bus = pq[local - n_pv - n_pq];
        take_imag = true;
    }
    if (bus < 0 || bus >= n_bus) {
        F[local] = NAN;
        return;
    }

    const double vr = v_re[bus];
    const double vi = v_im[bus];
    const double ir = ibus_re[bus];
    const double ii = ibus_im[bus];
    const double mis_re = vr * ir + vi * ii - sbus_re[bus];
    const double mis_im = vi * ir - vr * ii - sbus_im[bus];
    F[local] = take_imag ? mis_im : mis_re;
}

__global__ void reduce_abs_max_kernel(int32_t n,
                                      const double* __restrict__ values,
                                      double* __restrict__ out)
{
    extern __shared__ double shared[];
    const int32_t lane = threadIdx.x;
    double local_max = 0.0;
    for (int32_t i = lane; i < n; i += blockDim.x) {
        local_max = fmax(local_max, fabs(values[i]));
    }
    shared[lane] = local_max;
    __syncthreads();

    for (int32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            shared[lane] = fmax(shared[lane], shared[lane + offset]);
        }
        __syncthreads();
    }
    if (lane == 0) {
        *out = shared[0];
    }
}

__global__ void fill_jacobian_kernel(int32_t nnz_ybus,
                                     int32_t nnz_J,
                                     int32_t n_bus,
                                     const double* __restrict__ y_re,
                                     const double* __restrict__ y_im,
                                     const int32_t* __restrict__ y_row,
                                     const int32_t* __restrict__ y_col,
                                     const int32_t* __restrict__ y_row_ptr,
                                     const double* __restrict__ v_re,
                                     const double* __restrict__ v_im,
                                     const double* __restrict__ vm,
                                     const double* __restrict__ ibus_re,
                                     const double* __restrict__ ibus_im,
                                     const int32_t* __restrict__ map11,
                                     const int32_t* __restrict__ map21,
                                     const int32_t* __restrict__ map12,
                                     const int32_t* __restrict__ map22,
                                     const int32_t* __restrict__ diag11,
                                     const int32_t* __restrict__ diag21,
                                     const int32_t* __restrict__ diag12,
                                     const int32_t* __restrict__ diag22,
                                     double* __restrict__ J_values)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz_ybus) {
        return;
    }

    const int32_t i = y_row[k];
    const int32_t j = y_col[k];
    if (i < 0 || i >= n_bus || j < 0 || j >= n_bus) {
        return;
    }

    const double yr = y_re[k];
    const double yi = y_im[k];
    const double vi_re = v_re[i];
    const double vi_im = v_im[i];
    const double vj_re = v_re[j];
    const double vj_im = v_im[j];

    const double curr_re = yr * vj_re - yi * vj_im;
    const double curr_im = yr * vj_im + yi * vj_re;

    const double neg_j_vi_re = vi_im;
    const double neg_j_vi_im = -vi_re;
    const double term_va_re = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;
    const double term_va_im = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;

    const double vj_abs = vm[j];
    double term_vm_re = 0.0;
    double term_vm_im = 0.0;
    if (vj_abs > 1.0e-12) {
        const double scaled_re = curr_re / vj_abs;
        const double scaled_im = curr_im / vj_abs;
        term_vm_re = vi_re * scaled_re + vi_im * scaled_im;
        term_vm_im = vi_im * scaled_re - vi_re * scaled_im;
    }

    if (map11[k] >= 0 && map11[k] < nnz_J) {
        J_values[map11[k]] = term_va_re;
    }
    if (map21[k] >= 0 && map21[k] < nnz_J) {
        J_values[map21[k]] = term_va_im;
    }
    if (map12[k] >= 0 && map12[k] < nnz_J) {
        J_values[map12[k]] = term_vm_re;
    }
    if (map22[k] >= 0 && map22[k] < nnz_J) {
        J_values[map22[k]] = term_vm_im;
    }

    if (i != j) {
        return;
    }

    const double ir = ibus_re[i];
    const double ii = ibus_im[i];
    const double vi_conj_i_re = vi_re * ir + vi_im * ii;
    const double vi_conj_i_im = vi_im * ir - vi_re * ii;
    const double corr_va_re = -vi_conj_i_im;
    const double corr_va_im = vi_conj_i_re;

    const double vi_abs = vm[i];
    double corr_vm_re = 0.0;
    double corr_vm_im = 0.0;
    if (vi_abs > 1.0e-12) {
        const double vnorm_re = vi_re / vi_abs;
        const double vnorm_im = vi_im / vi_abs;
        corr_vm_re = ir * vnorm_re + ii * vnorm_im;
        corr_vm_im = ir * vnorm_im - ii * vnorm_re;
    }

    if (diag11[i] >= 0 && diag11[i] < nnz_J) {
        J_values[diag11[i]] += corr_va_re;
    }
    if (diag21[i] >= 0 && diag21[i] < nnz_J) {
        J_values[diag21[i]] += corr_va_im;
    }
    if (diag12[i] >= 0 && diag12[i] < nnz_J) {
        J_values[diag12[i]] += corr_vm_re;
    }
    if (diag22[i] >= 0 && diag22[i] < nnz_J) {
        J_values[diag22[i]] += corr_vm_im;
    }
}

__device__ inline void sincos_double(double x, double* s, double* c)
{
    sincos(x, s, c);
}

__global__ void apply_voltage_update_kernel(int32_t dimF,
                                            int32_t n_pv,
                                            int32_t n_pq,
                                            double* __restrict__ va,
                                            double* __restrict__ vm,
                                            const double* __restrict__ dx,
                                            const int32_t* __restrict__ pv,
                                            const int32_t* __restrict__ pq,
                                            double damping_factor)
{
    const int32_t local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= dimF) {
        return;
    }
    const double dx_value = damping_factor * dx[local];
    if (local < n_pv) {
        va[pv[local]] -= dx_value;
    } else if (local < n_pv + n_pq) {
        va[pq[local - n_pv]] -= dx_value;
    } else {
        vm[pq[local - n_pv - n_pq]] -= dx_value;
    }
}

__global__ void reconstruct_voltage_kernel(int32_t n_bus,
                                           const double* __restrict__ va,
                                           const double* __restrict__ vm,
                                           double* __restrict__ v_re,
                                           double* __restrict__ v_im)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }
    double s = 0.0;
    double c = 0.0;
    sincos_double(va[bus], &s, &c);
    v_re[bus] = vm[bus] * c;
    v_im[bus] = vm[bus] * s;
}

__global__ void count_nonfinite_kernel(int32_t n,
                                       const double* __restrict__ values,
                                       int32_t* __restrict__ count)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !isfinite(values[i])) {
        atomicAdd(count, 1);
    }
}

__global__ void scatter_field_values_kernel(int32_t nnz,
                                            const int32_t* __restrict__ full_positions,
                                            const double* __restrict__ full_values,
                                            double* __restrict__ field_values)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nnz) {
        field_values[k] = full_values[full_positions[k]];
    }
}

__global__ void copy_field_rhs_kernel(int32_t n_p,
                                      int32_t n_q,
                                      const double* __restrict__ residual,
                                      double* __restrict__ rhs_p,
                                      double* __restrict__ rhs_q)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n_p) {
        rhs_p[k] = residual[k];
    }
    if (k < n_q) {
        rhs_q[k] = residual[n_p + k];
    }
}

__global__ void accumulate_field_dx_kernel(int32_t n_p,
                                           int32_t n_q,
                                           const double* __restrict__ dtheta0,
                                           const double* __restrict__ dvm0,
                                           const double* __restrict__ dtheta1,
                                           const double* __restrict__ dvm1,
                                           bool include_round1,
                                           double* __restrict__ dx)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n_p) {
        double value = dtheta0[k];
        if (include_round1) {
            value += dtheta1[k];
        }
        dx[k] += value;
    }
    if (k < n_q) {
        double value = dvm0[k];
        if (include_round1) {
            value += dvm1[k];
        }
        dx[n_p + k] += value;
    }
}

__global__ void build_fdlf_rhs_kernel(int32_t n_pv,
                                      int32_t n_pq,
                                      const double* __restrict__ residual_p,
                                      const double* __restrict__ residual_q,
                                      const double* __restrict__ vm,
                                      const int32_t* __restrict__ pv,
                                      const int32_t* __restrict__ pq,
                                      double p_sign,
                                      double q_sign,
                                      bool p_scale_by_v,
                                      bool q_scale_by_v,
                                      double* __restrict__ rhs_p,
                                      double* __restrict__ rhs_q)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t n_p = n_pv + n_pq;
    if (k < n_p) {
        const int32_t bus = k < n_pv ? pv[k] : pq[k - n_pv];
        double value = p_sign * residual_p[k];
        if (p_scale_by_v) {
            value /= fmax(fabs(vm[bus]), 1.0e-12);
        }
        rhs_p[k] = value;
    }
    if (k < n_pq) {
        const int32_t bus = pq[k];
        double value = q_sign * residual_q[k];
        if (q_scale_by_v) {
            value /= fmax(fabs(vm[bus]), 1.0e-12);
        }
        rhs_q[k] = value;
    }
}

__global__ void build_fdlf_rhs_p_kernel(int32_t n_pv,
                                        int32_t n_pq,
                                        const double* __restrict__ residual_p,
                                        const double* __restrict__ vm,
                                        const int32_t* __restrict__ pv,
                                        const int32_t* __restrict__ pq,
                                        double sign,
                                        bool scale_by_v,
                                        double* __restrict__ rhs_p)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t n_p = n_pv + n_pq;
    if (k >= n_p) {
        return;
    }
    const int32_t bus = k < n_pv ? pv[k] : pq[k - n_pv];
    double value = sign * residual_p[k];
    if (scale_by_v) {
        value /= fmax(fabs(vm[bus]), 1.0e-12);
    }
    rhs_p[k] = value;
}

__global__ void build_fdlf_rhs_q_kernel(int32_t n_pq,
                                        const double* __restrict__ residual_q,
                                        const double* __restrict__ vm,
                                        const int32_t* __restrict__ pq,
                                        double sign,
                                        bool scale_by_v,
                                        double* __restrict__ rhs_q)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_pq) {
        return;
    }
    const int32_t bus = pq[k];
    double value = sign * residual_q[k];
    if (scale_by_v) {
        value /= fmax(fabs(vm[bus]), 1.0e-12);
    }
    rhs_q[k] = value;
}

}  // namespace

void launch_compute_ibus(int32_t n_bus,
                         const int32_t* y_row_ptr,
                         const int32_t* y_col,
                         const double* y_re,
                         const double* y_im,
                         const double* v_re,
                         const double* v_im,
                         double* ibus_re,
                         double* ibus_im)
{
    if (n_bus <= 0) {
        throw std::runtime_error("launch_compute_ibus: invalid dimensions");
    }
    compute_ibus_kernel<32><<<n_bus, 32>>>(
        n_bus, y_row_ptr, y_col, y_re, y_im, v_re, v_im, ibus_re, ibus_im);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_compute_mismatch_from_ibus(int32_t dimF,
                                       int32_t n_bus,
                                       int32_t n_pv,
                                       int32_t n_pq,
                                       const double* v_re,
                                       const double* v_im,
                                       const double* ibus_re,
                                       const double* ibus_im,
                                       const double* sbus_re,
                                       const double* sbus_im,
                                       const int32_t* pv,
                                       const int32_t* pq,
                                       double* F)
{
    if (dimF <= 0 || n_bus <= 0 || n_pv < 0 || n_pq < 0 || dimF != n_pv + 2 * n_pq) {
        throw std::runtime_error("launch_compute_mismatch_from_ibus: invalid dimensions");
    }
    compute_mismatch_from_ibus_kernel<<<grid_for(dimF), kBlock>>>(dimF,
                                                                  n_bus,
                                                                  n_pv,
                                                                  n_pq,
                                                                  v_re,
                                                                  v_im,
                                                                  ibus_re,
                                                                  ibus_im,
                                                                  sbus_re,
                                                                  sbus_im,
                                                                  pv,
                                                                  pq,
                                                                  F);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_abs_max(int32_t n, const double* values, double* out)
{
    if (n <= 0) {
        throw std::runtime_error("launch_reduce_abs_max: invalid dimensions");
    }
    reduce_abs_max_kernel<<<1, kBlock, static_cast<std::size_t>(kBlock) * sizeof(double)>>>(
        n, values, out);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_fill_jacobian(int32_t nnz_ybus,
                          int32_t nnz_J,
                          int32_t n_bus,
                          const double* y_re,
                          const double* y_im,
                          const int32_t* y_row,
                          const int32_t* y_col,
                          const int32_t* y_row_ptr,
                          const double* v_re,
                          const double* v_im,
                          const double* vm,
                          const double* ibus_re,
                          const double* ibus_im,
                          const int32_t* map11,
                          const int32_t* map21,
                          const int32_t* map12,
                          const int32_t* map22,
                          const int32_t* diag11,
                          const int32_t* diag21,
                          const int32_t* diag12,
                          const int32_t* diag22,
                          double* J_values)
{
    if (nnz_ybus <= 0 || nnz_J <= 0 || n_bus <= 0) {
        throw std::runtime_error("launch_fill_jacobian: invalid dimensions");
    }
    CUITER_CUDA_CHECK(cudaMemset(J_values, 0, static_cast<std::size_t>(nnz_J) * sizeof(double)));
    fill_jacobian_kernel<<<grid_for(nnz_ybus), kBlock>>>(nnz_ybus,
                                                         nnz_J,
                                                         n_bus,
                                                         y_re,
                                                         y_im,
                                                         y_row,
                                                         y_col,
                                                         y_row_ptr,
                                                         v_re,
                                                         v_im,
                                                         vm,
                                                         ibus_re,
                                                         ibus_im,
                                                         map11,
                                                         map21,
                                                         map12,
                                                         map22,
                                                         diag11,
                                                         diag21,
                                                         diag12,
                                                         diag22,
                                                         J_values);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_voltage_update(int32_t n_bus,
                           int32_t dimF,
                           int32_t n_pv,
                           int32_t n_pq,
                           double* va,
                           double* vm,
                           double* v_re,
                           double* v_im,
                           const double* dx,
                           const int32_t* pv,
                           const int32_t* pq,
                           double damping_factor)
{
    if (n_bus <= 0 || dimF <= 0 || dimF != n_pv + 2 * n_pq) {
        throw std::runtime_error("launch_voltage_update: invalid dimensions");
    }
    apply_voltage_update_kernel<<<grid_for(dimF), kBlock>>>(
        dimF, n_pv, n_pq, va, vm, dx, pv, pq, damping_factor);
    CUITER_CUDA_CHECK(cudaGetLastError());
    reconstruct_voltage_kernel<<<grid_for(n_bus), kBlock>>>(n_bus, va, vm, v_re, v_im);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_count_nonfinite(int32_t n, const double* values, int32_t* count)
{
    if (n <= 0) {
        throw std::runtime_error("launch_count_nonfinite: invalid dimensions");
    }
    CUITER_CUDA_CHECK(cudaMemset(count, 0, sizeof(int32_t)));
    count_nonfinite_kernel<<<grid_for(n), kBlock>>>(n, values, count);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_scatter_field_values(int32_t nnz,
                                 const int32_t* full_positions,
                                 const double* full_values,
                                 double* field_values,
                                 cudaStream_t stream)
{
    if (nnz < 0 || full_positions == nullptr || full_values == nullptr || field_values == nullptr) {
        throw std::runtime_error("launch_scatter_field_values: invalid input");
    }
    if (nnz == 0) {
        return;
    }
    scatter_field_values_kernel<<<grid_for(nnz), kBlock, 0, stream>>>(
        nnz, full_positions, full_values, field_values);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_copy_field_rhs(int32_t n_p,
                           int32_t n_q,
                           const double* residual,
                           double* rhs_p,
                           double* rhs_q,
                           cudaStream_t stream)
{
    if (n_p <= 0 || n_q < 0 || residual == nullptr || rhs_p == nullptr || rhs_q == nullptr) {
        throw std::runtime_error("launch_copy_field_rhs: invalid input");
    }
    copy_field_rhs_kernel<<<grid_for(std::max(n_p, n_q)), kBlock, 0, stream>>>(
        n_p, n_q, residual, rhs_p, rhs_q);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_accumulate_field_dx(int32_t n_p,
                                int32_t n_q,
                                const double* dtheta0,
                                const double* dvm0,
                                const double* dtheta1,
                                const double* dvm1,
                                bool include_round1,
                                double* dx,
                                cudaStream_t stream)
{
    if (n_p <= 0 || n_q < 0 || dtheta0 == nullptr || dvm0 == nullptr || dx == nullptr ||
        (include_round1 && (dtheta1 == nullptr || dvm1 == nullptr))) {
        throw std::runtime_error("launch_accumulate_field_dx: invalid input");
    }
    accumulate_field_dx_kernel<<<grid_for(std::max(n_p, n_q)), kBlock, 0, stream>>>(
        n_p, n_q, dtheta0, dvm0, dtheta1, dvm1, include_round1, dx);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_build_fdlf_rhs(int32_t n_pv,
                           int32_t n_pq,
                           const double* residual_p,
                           const double* residual_q,
                           const double* vm,
                           const int32_t* pv,
                           const int32_t* pq,
                           double p_sign,
                           double q_sign,
                           bool p_scale_by_v,
                           bool q_scale_by_v,
                           double* rhs_p,
                           double* rhs_q,
                           cudaStream_t stream)
{
    if (n_pv < 0 || n_pq <= 0 || residual_p == nullptr || residual_q == nullptr ||
        vm == nullptr || pv == nullptr || pq == nullptr || rhs_p == nullptr || rhs_q == nullptr) {
        throw std::runtime_error("launch_build_fdlf_rhs: invalid input");
    }
    build_fdlf_rhs_kernel<<<grid_for(n_pv + n_pq), kBlock, 0, stream>>>(
        n_pv,
        n_pq,
        residual_p,
        residual_q,
        vm,
        pv,
        pq,
        p_sign,
        q_sign,
        p_scale_by_v,
        q_scale_by_v,
        rhs_p,
        rhs_q);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_build_fdlf_rhs_p(int32_t n_pv,
                             int32_t n_pq,
                             const double* residual_p,
                             const double* vm,
                             const int32_t* pv,
                             const int32_t* pq,
                             double sign,
                             bool scale_by_v,
                             double* rhs_p,
                             cudaStream_t stream)
{
    if (n_pv < 0 || n_pq < 0 || residual_p == nullptr || vm == nullptr || pv == nullptr ||
        pq == nullptr || rhs_p == nullptr) {
        throw std::runtime_error("launch_build_fdlf_rhs_p: invalid input");
    }
    build_fdlf_rhs_p_kernel<<<grid_for(n_pv + n_pq), kBlock, 0, stream>>>(
        n_pv, n_pq, residual_p, vm, pv, pq, sign, scale_by_v, rhs_p);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

void launch_build_fdlf_rhs_q(int32_t n_pq,
                             const double* residual_q,
                             const double* vm,
                             const int32_t* pq,
                             double sign,
                             bool scale_by_v,
                             double* rhs_q,
                             cudaStream_t stream)
{
    if (n_pq <= 0 || residual_q == nullptr || vm == nullptr || pq == nullptr || rhs_q == nullptr) {
        throw std::runtime_error("launch_build_fdlf_rhs_q: invalid input");
    }
    build_fdlf_rhs_q_kernel<<<grid_for(n_pq), kBlock, 0, stream>>>(
        n_pq, residual_q, vm, pq, sign, scale_by_v, rhs_q);
    CUITER_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cupf_minimal
