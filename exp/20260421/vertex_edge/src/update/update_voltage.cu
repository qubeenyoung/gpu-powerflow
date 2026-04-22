#include "update_voltage.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace exp20260421::vertex_edge::voltage_update {

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err = (call);                                                          \
        if (err != cudaSuccess) {                                                          \
            throw std::runtime_error(                                                       \
                std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - " + cudaGetErrorString(err));                                         \
        }                                                                                   \
    } while (0)

namespace {

constexpr int32_t kBlock = 256;

__global__ void init_voltage_kernel(const double* __restrict__ v0_re,
                                    const double* __restrict__ v0_im,
                                    double* __restrict__ v_re,
                                    double* __restrict__ v_im,
                                    int32_t total_bus)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_bus) {
        return;
    }

    v_re[idx] = v0_re[idx];
    v_im[idx] = v0_im[idx];
}

__global__ void update_voltage_kernel(double* __restrict__ v_re,
                                      double* __restrict__ v_im,
                                      const float* __restrict__ dx,
                                      const int32_t* __restrict__ pvpq,
                                      int32_t n_pvpq,
                                      int32_t n_pq,
                                      int32_t n_bus,
                                      int32_t dim,
                                      int32_t total_active)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_active) {
        return;
    }

    const int32_t batch = idx / n_pvpq;
    const int32_t slot = idx - batch * n_pvpq;
    const int32_t bus = pvpq[slot];
    const int32_t bus_index = batch * n_bus + bus;
    const int32_t dim_base = batch * dim;
    const int32_t n_pv = n_pvpq - n_pq;

    const double d_va = static_cast<double>(dx[dim_base + slot]);
    double re = v_re[bus_index];
    double im = v_im[bus_index];
    double mag_scale = 1.0;

    if (slot >= n_pv) {
        const int32_t pq_slot = slot - n_pv;
        const double old_mag = hypot(re, im);
        const double new_mag = old_mag + static_cast<double>(dx[dim_base + n_pvpq + pq_slot]);
        mag_scale = (old_mag > 1e-12) ? (new_mag / old_mag) : 0.0;
    }

    double sin_dva = 0.0;
    double cos_dva = 0.0;
    sincos(d_va, &sin_dva, &cos_dva);

    const double scaled_re = mag_scale * re;
    const double scaled_im = mag_scale * im;
    v_re[bus_index] = scaled_re * cos_dva - scaled_im * sin_dva;
    v_im[bus_index] = scaled_re * sin_dva + scaled_im * cos_dva;
}

}  // namespace

void init_voltage(const double* v0_re,
                  const double* v0_im,
                  double* v_re,
                  double* v_im,
                  int32_t n_bus,
                  int32_t batch_size,
                  cudaStream_t stream)
{
    if (n_bus <= 0 || batch_size <= 0) {
        throw std::invalid_argument("init_voltage: dimensions must be positive");
    }
    if (v0_re == nullptr || v0_im == nullptr || v_re == nullptr || v_im == nullptr) {
        throw std::invalid_argument("init_voltage: device pointer is null");
    }

    const int32_t total_bus = n_bus * batch_size;
    const int32_t grid = (total_bus + kBlock - 1) / kBlock;
    init_voltage_kernel<<<grid, kBlock, 0, stream>>>(v0_re, v0_im, v_re, v_im, total_bus);
    CUDA_CHECK(cudaGetLastError());
}

void update_voltage(double* v_re,
                    double* v_im,
                    const float* dx,
                    const int32_t* pvpq,
                    int32_t n_pvpq,
                    const int32_t* pq,
                    int32_t n_pq,
                    int32_t n_bus,
                    int32_t batch_size,
                    cudaStream_t stream)
{
    (void)pq;
    if (n_bus <= 0 || n_pvpq <= 0 || n_pq < 0 || batch_size <= 0) {
        throw std::invalid_argument("update_voltage: bad dimensions");
    }
    if (v_re == nullptr || v_im == nullptr || dx == nullptr || pvpq == nullptr) {
        throw std::invalid_argument("update_voltage: device pointer is null");
    }

    const int32_t dim = n_pvpq + n_pq;
    const int32_t total_active = n_pvpq * batch_size;
    const int32_t grid = (total_active + kBlock - 1) / kBlock;
    update_voltage_kernel<<<grid, kBlock, 0, stream>>>(
        v_re,
        v_im,
        dx,
        pvpq,
        n_pvpq,
        n_pq,
        n_bus,
        dim,
        total_active);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace exp20260421::vertex_edge::voltage_update
