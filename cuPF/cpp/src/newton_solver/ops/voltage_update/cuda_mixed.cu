// ---------------------------------------------------------------------------
// cuda_mixed.cu — Mixed 정밀도 전압 갱신 커널
//
// cuda_fp64.cu 와 동일한 3단계 파이프라인이며, 단계 2에서 dx 가 float* 이다.
//
//   1. decompose_voltage_kernel  — FP64 V → FP64 Va, Vm  (cuda_fp64.cu 와 동일)
//   2. update_voltage_mixed_kernel — FP32 dx → FP64 Va/Vm 누산
//        dx_value = static_cast<double>(dx[tid])  ← FP32 → FP64 승격
//        Va/Vm 에 double로 더하므로 전압 상태는 FP64 정밀도 유지.
//   3. reconstruct_voltage_kernel — FP64 Va, Vm → FP64 V  (cuda_fp64.cu 와 동일)
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_mixed.hpp"

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"

#include <cmath>
#include <stdexcept>


namespace {

void sync_cuda_for_timing()
{
#ifdef CUPF_ENABLE_TIMING
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

// 단계 1: V(복소 FP64) → Va(각도 FP64), Vm(크기 FP64)
__global__ void decompose_voltage_kernel(
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    double* __restrict__ va,
    double* __restrict__ vm,
    int32_t n_bus)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }

    const double re = v_re[bus];
    const double im = v_im[bus];
    va[bus] = atan2(im, re);
    vm[bus] = hypot(re, im);
}

// 단계 2: FP32 dx → FP64 Va/Vm 갱신
// dx 는 cuDSS FP32 solver 의 출력 (float*).
// static_cast<double>(dx[tid]) 로 FP64 에 더함 → 전압 상태 FP64 정밀도 유지.
__global__ void update_voltage_mixed_kernel(
    double* __restrict__ va,
    double* __restrict__ vm,
    const float* __restrict__ dx,
    const int32_t* __restrict__ pv,
    const int32_t* __restrict__ pq,
    int32_t n_pv,
    int32_t n_pq)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pv + 2 * n_pq) {
        return;
    }

    // FP32 dx → FP64 승격 후 FP64 Va/Vm 에 누산
    const double dx_value = static_cast<double>(dx[tid]);
    if (tid < n_pv) {
        va[pv[tid]] += dx_value;
    } else if (tid < n_pv + n_pq) {
        va[pq[tid - n_pv]] += dx_value;
    } else {
        vm[pq[tid - n_pv - n_pq]] += dx_value;
    }
}

// 단계 3: FP64 Va, Vm → FP64 V(복소)
__global__ void reconstruct_voltage_kernel(
    const double* __restrict__ va,
    const double* __restrict__ vm,
    double* __restrict__ v_re,
    double* __restrict__ v_im,
    int32_t n_bus)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }

    v_re[bus] = vm[bus] * cos(va[bus]);
    v_im[bus] = vm[bus] * sin(va[bus]);
}

}  // namespace


CudaVoltageUpdateMixed::CudaVoltageUpdateMixed(IStorage& storage)
    : storage_(storage) {}


void CudaVoltageUpdateMixed::run(IterationContext& ctx)
{
    (void)ctx;
    auto& storage = static_cast<CudaMixedStorage&>(storage_);

    if (storage.n_bus <= 0 || storage.dimF <= 0) {
        throw std::runtime_error("CudaVoltageUpdateMixed::run: storage is not prepared");
    }

    constexpr int32_t block = 256;
    const int32_t n_pv = storage.n_pvpq - storage.n_pq;
    const int32_t grid_bus = (storage.n_bus + block - 1) / block;
    const int32_t grid_dx = (storage.dimF + block - 1) / block;

    decompose_voltage_kernel<<<grid_bus, block>>>(
        storage.d_V_re.data(),
        storage.d_V_im.data(),
        storage.d_Va.data(),
        storage.d_Vm.data(),
        storage.n_bus);
    CUDA_CHECK(cudaGetLastError());

    update_voltage_mixed_kernel<<<grid_dx, block>>>(
        storage.d_Va.data(),
        storage.d_Vm.data(),
        storage.d_dx.data(),
        storage.d_pv.data(),
        storage.d_pq.data(),
        n_pv,
        storage.n_pq);
    CUDA_CHECK(cudaGetLastError());

    reconstruct_voltage_kernel<<<grid_bus, block>>>(
        storage.d_Va.data(),
        storage.d_Vm.data(),
        storage.d_V_re.data(),
        storage.d_V_im.data(),
        storage.n_bus);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

#endif  // CUPF_WITH_CUDA
