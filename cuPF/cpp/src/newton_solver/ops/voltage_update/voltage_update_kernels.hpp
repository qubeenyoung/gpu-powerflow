// ---------------------------------------------------------------------------
// voltage_update_kernels.hpp
//
// Device kernels + launchers for the CUDA voltage-update stage, the GPU
// counterpart of cpu_voltage_update.cpp: apply the Newton step dx to (Va, Vm),
// then rebuild the rectangular voltage. Header-only so the precision-specific
// dispatch in cuda_voltage_update.cu can instantiate them per storage type.
// ---------------------------------------------------------------------------

#pragma once

#ifdef CUPF_WITH_CUDA

#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <stdexcept>


namespace {

constexpr int32_t kVoltageUpdateBlock = 256;

// Precision-matched sincos so float state uses sincosf, double uses sincos.
__device__ inline void sincos_scalar(double x, double* s, double* c)
{
    sincos(x, s, c);
}

__device__ inline void sincos_scalar(float x, float* s, float* c)
{
    sincosf(x, s, c);
}

// Apply the Newton step to the polar voltage state (one thread per dx entry).
//
// The step solved J*dx = F (mismatch), so the Newton update is x <- x - dx.
// dx is segmented per case by the dimF layout, mirroring the residual order:
//   local in [0,         n_pv)        -> angle Va at pv bus  pv[local]
//   local in [n_pv,      n_pv+n_pq)   -> angle Va at pq bus  pq[local-n_pv]
//   local in [n_pv+n_pq, dimF)        -> magnitude Vm at pq bus pq[...]
// (hence dimF == n_pv + 2*n_pq.) dx may be lower precision than the state
// (Mixed: float dx into double state); static_cast bridges it before subtract.
template <typename StateScalar, typename DxScalar>
__global__ void apply_voltage_update_kernel(
    int32_t total_entries,
    int32_t dimF,
    int32_t n_bus,
    StateScalar* __restrict__ va,
    StateScalar* __restrict__ vm,
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

    // Map the flat id to (batch case, position within that case's dx). Per-bus
    // state is batch-major from bus_base; this case's dx starts at dx_base.
    const int32_t batch = tid / dimF;
    const int32_t local = tid - batch * dimF;
    const int32_t bus_base = batch * n_bus;
    const int32_t dx_base = batch * dimF;
    const StateScalar dx_value = static_cast<StateScalar>(dx[dx_base + local]);

    // Route this dx component to its unknown per the segmentation above.
    if (local < n_pv) {
        va[bus_base + pv[local]] -= dx_value;                 // pv angle
    } else if (local < n_pv + n_pq) {
        va[bus_base + pq[local - n_pv]] -= dx_value;          // pq angle
    } else {
        vm[bus_base + pq[local - n_pv - n_pq]] -= dx_value;   // pq magnitude
    }
}

// Rebuild rectangular voltage from polar (one thread per bus):
// V = Vm * (cos Va + i sin Va).
template <typename StateScalar>
__global__ void reconstruct_voltage_kernel(
    int32_t total_buses,
    const StateScalar* __restrict__ va,
    const StateScalar* __restrict__ vm,
    StateScalar* __restrict__ v_re,
    StateScalar* __restrict__ v_im)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= total_buses) {
        return;
    }

    StateScalar s = StateScalar(0);
    StateScalar c = StateScalar(0);
    sincos_scalar(va[bus], &s, &c);
    const StateScalar vm_value = vm[bus];
    const StateScalar re = vm_value * c;
    const StateScalar im = vm_value * s;

    v_re[bus] = re;
    v_im[bus] = im;
}

template <typename StateScalar, typename DxScalar>
void launch_apply_voltage_update(
    int32_t batch_size,
    int32_t n_bus,
    int32_t dimF,
    int32_t n_pv,
    int32_t n_pq,
    StateScalar* va,
    StateScalar* vm,
    const DxScalar* dx,
    const int32_t* pv,
    const int32_t* pq)
{
    if (batch_size <= 0 || n_bus <= 0 || dimF <= 0 || n_pv < 0 || n_pq < 0 ||
        dimF != n_pv + 2 * n_pq) {
        throw std::runtime_error("launch_apply_voltage_update: invalid dimensions");
    }

    const int32_t total_dx = batch_size * dimF;
    const int32_t grid_dx = (total_dx + kVoltageUpdateBlock - 1) / kVoltageUpdateBlock;
    apply_voltage_update_kernel<StateScalar, DxScalar><<<grid_dx, kVoltageUpdateBlock, 0, cupf_current_cuda_stream()>>>(
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

template <typename StateScalar>
void launch_reconstruct_voltage(
    int32_t total_buses,
    const StateScalar* va,
    const StateScalar* vm,
    StateScalar* v_re,
    StateScalar* v_im)
{
    if (total_buses <= 0) {
        throw std::runtime_error("launch_reconstruct_voltage: invalid dimensions");
    }

    const int32_t grid_bus = (total_buses + kVoltageUpdateBlock - 1) / kVoltageUpdateBlock;
    reconstruct_voltage_kernel<StateScalar><<<grid_bus, kVoltageUpdateBlock, 0, cupf_current_cuda_stream()>>>(
        total_buses,
        va,
        vm,
        v_re,
        v_im);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

#endif  // CUPF_WITH_CUDA
