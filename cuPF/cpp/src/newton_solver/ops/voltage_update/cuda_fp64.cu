// ---------------------------------------------------------------------------
// cuda_fp64.cu — CUDA FP64 전압 갱신 커널 (3단계 파이프라인)
//
// NR 반복 종료 후 dx 보정을 적용해 전압 벡터를 갱신한다.
// 3개의 커널을 순차 실행한다:
//
//   1. decompose_voltage_kernel (bus 단위)
//      V(복소) → Va(각도), Vm(크기)
//        Va = atan2(V_im, V_re)
//        Vm = hypot(V_re, V_im)
//
//   2. update_voltage_fp64_kernel (dimF 단위)
//      dx 보정 적용 (dx 레이아웃 = [Δθ_pv, Δθ_pq, Δ|V|_pq]):
//        Va[pv[tid]]         += dx[tid]           (tid < n_pv)
//        Va[pq[tid-n_pv]]    += dx[tid]           (n_pv ≤ tid < n_pv+n_pq)
//        Vm[pq[tid-n_pv-n_pq]] += dx[tid]         (n_pv+n_pq ≤ tid < dimF)
//      dx 는 FP64; Va/Vm 도 FP64 → 정밀도 손실 없음.
//
//   3. reconstruct_voltage_kernel (bus 단위)
//      Va, Vm → V(복소):
//        V_re = Vm · cos(Va)
//        V_im = Vm · sin(Va)
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_fp64.hpp"

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"

#include <cmath>
#include <stdexcept>


namespace {

void sync_cuda_for_timing()
{
#ifdef CUPF_ENABLE_TIMING
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

// 단계 1: V(복소) → Va(각도), Vm(크기)
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

// 단계 2: dx 보정 적용 (FP64 dx → FP64 Va/Vm)
__global__ void update_voltage_fp64_kernel(
    double* __restrict__ va,
    double* __restrict__ vm,
    const double* __restrict__ dx,
    const int32_t* __restrict__ pv,
    const int32_t* __restrict__ pq,
    int32_t n_pv,
    int32_t n_pq)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pv + 2 * n_pq) {
        return;
    }

    if (tid < n_pv) {
        va[pv[tid]] += dx[tid];
    } else if (tid < n_pv + n_pq) {
        va[pq[tid - n_pv]] += dx[tid];
    } else {
        vm[pq[tid - n_pv - n_pq]] += dx[tid];
    }
}

// 단계 3: Va, Vm → V(복소)  V = Vm·e^{jVa}
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


CudaVoltageUpdateFp64::CudaVoltageUpdateFp64(IStorage& storage)
    : storage_(storage) {}


void CudaVoltageUpdateFp64::run(IterationContext& ctx)
{
    (void)ctx;
    auto& storage = static_cast<CudaFp64Storage&>(storage_);

    if (storage.n_bus <= 0 || storage.dimF <= 0) {
        throw std::runtime_error("CudaVoltageUpdateFp64::run: storage is not prepared");
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

    update_voltage_fp64_kernel<<<grid_dx, block>>>(
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
