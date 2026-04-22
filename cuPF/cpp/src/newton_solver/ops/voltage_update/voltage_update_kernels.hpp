#pragma once

#ifdef CUPF_WITH_CUDA

#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <stdexcept>


namespace {

constexpr int32_t kVoltageUpdateBlock = 256;

template <typename DxScalar>
__global__ void apply_voltage_update_kernel(
    int32_t total_entries,
    int32_t dimF,
    int32_t n_bus,
    double* __restrict__ va,
    double* __restrict__ vm,
    const DxScalar* __restrict__ dx,
    const int32_t* __restrict__ pv,
    const int32_t* __restrict__ pq,
    int32_t n_pv,
    int32_t n_pq)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_entries) {
        return;
    }

    const int32_t batch = tid / dimF;
    const int32_t local = tid - batch * dimF;
    const int32_t bus_base = batch * n_bus;
    const int32_t dx_base = batch * dimF;
    const double dx_value = static_cast<double>(dx[dx_base + local]);

    if (local < n_pv) {
        va[bus_base + pv[local]] -= dx_value;
    } else if (local < n_pv + n_pq) {
        va[bus_base + pq[local - n_pv]] -= dx_value;
    } else {
        vm[bus_base + pq[local - n_pv - n_pq]] -= dx_value;
    }
}

__global__ void reconstruct_voltage_kernel(
    int32_t total_buses,
    const double* __restrict__ va,
    const double* __restrict__ vm,
    double* __restrict__ v_re,
    double* __restrict__ v_im)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= total_buses) {
        return;
    }

    double s = 0.0;
    double c = 0.0;
    sincos(va[bus], &s, &c);
    const double vm_value = vm[bus];
    const double re = vm_value * c;
    const double im = vm_value * s;

    v_re[bus] = re;
    v_im[bus] = im;
}

template <typename DxScalar>
void launch_voltage_update_state(
    int32_t batch_size,
    int32_t n_bus,
    int32_t dimF,
    int32_t n_pv,
    int32_t n_pq,
    double* va,
    double* vm,
    const DxScalar* dx,
    const int32_t* pv,
    const int32_t* pq)
{
    if (batch_size <= 0 || n_bus <= 0 || dimF <= 0 || n_pv < 0 || n_pq < 0 ||
        dimF != n_pv + 2 * n_pq) {
        throw std::runtime_error("launch_voltage_update_state: invalid dimensions");
    }

    const int32_t total_dx = batch_size * dimF;
    const int32_t grid_dx = (total_dx + kVoltageUpdateBlock - 1) / kVoltageUpdateBlock;
    apply_voltage_update_kernel<DxScalar><<<grid_dx, kVoltageUpdateBlock>>>(
        total_dx,
        dimF,
        n_bus,
        va,
        vm,
        dx,
        pv,
        pq,
        n_pv,
        n_pq);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reconstruct_voltage(
    int32_t total_buses,
    const double* va,
    const double* vm,
    double* v_re,
    double* v_im)
{
    if (total_buses <= 0) {
        throw std::runtime_error("launch_reconstruct_voltage: invalid dimensions");
    }

    const int32_t grid_bus = (total_buses + kVoltageUpdateBlock - 1) / kVoltageUpdateBlock;
    reconstruct_voltage_kernel<<<grid_bus, kVoltageUpdateBlock>>>(
        total_buses,
        va,
        vm,
        v_re,
        v_im);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

#endif  // CUPF_WITH_CUDA
