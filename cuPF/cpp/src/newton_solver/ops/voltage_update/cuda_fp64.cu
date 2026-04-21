// ---------------------------------------------------------------------------
// cuda_fp64.cu — CUDA FP64 전압 갱신 커널
//
// Va/Vm을 authoritative state로 유지하고, V_re/V_im은 mismatch/Jacobian용
// derived cache로 재구성한다.
//
// Sign convention:
//   mismatch computes F = S_calc - S_spec.
//   linear solve returns dx from J * dx = F.
//   update applies state -= dx.
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

// dx 보정 적용 (FP64 dx → FP64 Va/Vm)
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
        va[pv[tid]] -= dx[tid];
    } else if (tid < n_pv + n_pq) {
        va[pq[tid - n_pv]] -= dx[tid];
    } else {
        vm[pq[tid - n_pv - n_pq]] -= dx[tid];
    }
}

// Va, Vm → V(복소)  V = Vm·e^{jVa}
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

    double s = 0.0;
    double c = 0.0;
    sincos(va[bus], &s, &c);
    v_re[bus] = vm[bus] * c;
    v_im[bus] = vm[bus] * s;
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
