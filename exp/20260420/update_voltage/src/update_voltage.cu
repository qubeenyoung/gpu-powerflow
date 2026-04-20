#include "update_voltage.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace exp20260420::voltage_update {

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

__global__ void init_voltage_kernel(const float* __restrict__ v0_re,
                                    const float* __restrict__ v0_im,
                                    float* __restrict__ v_re,
                                    float* __restrict__ v_im,
                                    float* __restrict__ va,
                                    float* __restrict__ vm,
                                    float* __restrict__ v_norm_re,
                                    float* __restrict__ v_norm_im,
                                    int32_t n_bus)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }

    const float re = v0_re[bus];
    const float im = v0_im[bus];
    const float mag = hypotf(re, im);

    v_re[bus] = re;
    v_im[bus] = im;
    va[bus] = atan2f(im, re);
    vm[bus] = mag;
    v_norm_re[bus] = mag > 1e-6f ? re / mag : 0.0f;
    v_norm_im[bus] = mag > 1e-6f ? im / mag : 0.0f;
}

__global__ void apply_dx_kernel(float* __restrict__ va,
                                float* __restrict__ vm,
                                const float* __restrict__ dx,
                                const int32_t* __restrict__ pvpq,
                                int32_t n_pvpq,
                                const int32_t* __restrict__ pq,
                                int32_t n_pq)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t dim = n_pvpq + n_pq;
    if (tid >= dim) {
        return;
    }

    if (tid < n_pvpq) {
        va[pvpq[tid]] += dx[tid];
    } else {
        vm[pq[tid - n_pvpq]] += dx[tid];
    }
}

__global__ void rebuild_voltage_kernel(const float* __restrict__ va,
                                       const float* __restrict__ vm,
                                       float* __restrict__ v_re,
                                       float* __restrict__ v_im,
                                       float* __restrict__ v_norm_re,
                                       float* __restrict__ v_norm_im,
                                       int32_t n_bus)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }

    float sin_va = 0.0f;
    float cos_va = 0.0f;
    sincosf(va[bus], &sin_va, &cos_va);

    const float mag = vm[bus];
    const float re = mag * cos_va;
    const float im = mag * sin_va;
    const float abs_mag = fabsf(mag);
    const float norm_scale = abs_mag > 1e-6f ? 1.0f / abs_mag : 0.0f;

    v_re[bus] = re;
    v_im[bus] = im;
    v_norm_re[bus] = re * norm_scale;
    v_norm_im[bus] = im * norm_scale;
}

}  // namespace

void init_voltage(const float* v0_re,
                  const float* v0_im,
                  float* v_re,
                  float* v_im,
                  float* va,
                  float* vm,
                  float* v_norm_re,
                  float* v_norm_im,
                  int32_t n_bus,
                  cudaStream_t stream)
{
    if (n_bus <= 0) {
        throw std::invalid_argument("init_voltage: n_bus must be positive");
    }
    if (v0_re == nullptr || v0_im == nullptr ||
        v_re == nullptr || v_im == nullptr ||
        va == nullptr || vm == nullptr ||
        v_norm_re == nullptr || v_norm_im == nullptr) {
        throw std::invalid_argument("init_voltage: device pointer is null");
    }

    const int32_t grid = (n_bus + kBlock - 1) / kBlock;
    init_voltage_kernel<<<grid, kBlock, 0, stream>>>(
        v0_re,
        v0_im,
        v_re,
        v_im,
        va,
        vm,
        v_norm_re,
        v_norm_im,
        n_bus);
    CUDA_CHECK(cudaGetLastError());
}

void update_voltage(float* v_re,
                    float* v_im,
                    float* va,
                    float* vm,
                    float* v_norm_re,
                    float* v_norm_im,
                    const float* dx,
                    const int32_t* pvpq,
                    int32_t n_pvpq,
                    const int32_t* pq,
                    int32_t n_pq,
                    int32_t n_bus,
                    cudaStream_t stream)
{
    if (n_bus <= 0 || n_pvpq <= 0 || n_pq < 0) {
        throw std::invalid_argument("update_voltage: bad dimensions");
    }
    if (v_re == nullptr || v_im == nullptr ||
        va == nullptr || vm == nullptr ||
        v_norm_re == nullptr || v_norm_im == nullptr ||
        dx == nullptr || pvpq == nullptr ||
        (n_pq > 0 && pq == nullptr)) {
        throw std::invalid_argument("update_voltage: device pointer is null");
    }

    const int32_t dim = n_pvpq + n_pq;
    const int32_t dx_grid = (dim + kBlock - 1) / kBlock;
    apply_dx_kernel<<<dx_grid, kBlock, 0, stream>>>(
        va,
        vm,
        dx,
        pvpq,
        n_pvpq,
        pq,
        n_pq);
    CUDA_CHECK(cudaGetLastError());

    const int32_t bus_grid = (n_bus + kBlock - 1) / kBlock;
    rebuild_voltage_kernel<<<bus_grid, kBlock, 0, stream>>>(
        va,
        vm,
        v_re,
        v_im,
        v_norm_re,
        v_norm_im,
        n_bus);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace exp20260420::voltage_update
