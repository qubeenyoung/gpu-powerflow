// ---------------------------------------------------------------------------
// cuda_mixed.cu — Mixed 정밀도 전압 갱신 커널
//
// Va/Vm(FP64)을 authoritative state로 유지하고, V_re/V_im(FP64)은
// mismatch/Jacobian 입력용 derived cache로 재구성한다.
//
// Sign convention:
//   mismatch computes F = S_calc - S_spec.
//   linear solve returns dx from J * dx = F.
//   update applies state -= dx.
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

// FP32 dx → FP64 Va/Vm 갱신.
// dx 는 cuDSS FP32 solver 의 출력 (float*).
// static_cast<double>(dx[tid]) 로 FP64 에서 뺌 → 전압 상태 FP64 정밀도 유지.
__global__ void update_voltage_mixed_kernel(
    int32_t total_entries,
    int32_t dimF,
    int32_t n_bus,
    double* __restrict__ va,
    double* __restrict__ vm,
    const float* __restrict__ dx,
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

    // FP32 dx → FP64 승격 후 FP64 Va/Vm 에 적용.
    const double dx_value = static_cast<double>(dx[dx_base + local]);
    if (local < n_pv) {
        va[bus_base + pv[local]] -= dx_value;
    } else if (local < n_pv + n_pq) {
        va[bus_base + pq[local - n_pv]] -= dx_value;
    } else {
        vm[bus_base + pq[local - n_pv - n_pq]] -= dx_value;
    }
}

// FP64 Va, Vm → FP64 V cache(복소)
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
    const int32_t total_buses = storage.batch_size * storage.n_bus;
    const int32_t total_dx = storage.batch_size * storage.dimF;
    const int32_t grid_bus = (total_buses + block - 1) / block;
    const int32_t grid_dx = (total_dx + block - 1) / block;

    update_voltage_mixed_kernel<<<grid_dx, block>>>(
        total_dx,
        storage.dimF,
        storage.n_bus,
        storage.d_Va.data(),
        storage.d_Vm.data(),
        storage.d_dx.data(),
        storage.d_pv.data(),
        storage.d_pq.data(),
        n_pv,
        storage.n_pq);
    CUDA_CHECK(cudaGetLastError());

    reconstruct_voltage_kernel<<<grid_bus, block>>>(
        total_buses,
        storage.d_Va.data(),
        storage.d_Vm.data(),
        storage.d_V_re.data(),
        storage.d_V_im.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

#endif  // CUPF_WITH_CUDA
