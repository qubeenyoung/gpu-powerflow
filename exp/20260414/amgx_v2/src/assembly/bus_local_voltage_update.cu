#include "bus_local_voltage_update.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace exp_20260414::amgx_v2 {
namespace {

constexpr int32_t kBlockSize = 256;

__global__ void decompose_voltage_kernel(int32_t n_bus,
                                         const double* __restrict__ v_re,
                                         const double* __restrict__ v_im,
                                         double* __restrict__ va,
                                         double* __restrict__ vm)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }
    va[bus] = atan2(v_im[bus], v_re[bus]);
    vm[bus] = hypot(v_re[bus], v_im[bus]);
}

__global__ void apply_bus_local_dx_kernel(int32_t n_bus,
                                          const double* __restrict__ dx,
                                          const int32_t* __restrict__ theta_slot,
                                          const int32_t* __restrict__ vm_slot,
                                          const int32_t* __restrict__ theta_active,
                                          const int32_t* __restrict__ vm_active,
                                          double* __restrict__ va,
                                          double* __restrict__ vm)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }
    if (theta_active[bus]) {
        va[bus] += dx[theta_slot[bus]];
    }
    if (vm_active[bus]) {
        vm[bus] += dx[vm_slot[bus]];
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
    v_re[bus] = vm[bus] * cos(va[bus]);
    v_im[bus] = vm[bus] * sin(va[bus]);
}

}  // namespace

void BusLocalVoltageUpdate::analyze(const BusLocalIndex& index)
{
    if (index.n_bus <= 0 || index.dim != 2 * index.n_bus) {
        throw std::runtime_error("BusLocalVoltageUpdate::analyze received invalid index");
    }

    n_bus_ = index.n_bus;
    std::vector<int32_t> theta_active(static_cast<std::size_t>(n_bus_), 0);
    std::vector<int32_t> vm_active(static_cast<std::size_t>(n_bus_), 0);
    for (int32_t bus = 0; bus < n_bus_; ++bus) {
        theta_active[static_cast<std::size_t>(bus)] = index.is_p_active(bus) ? 1 : 0;
        vm_active[static_cast<std::size_t>(bus)] = index.is_q_active(bus) ? 1 : 0;
    }

    d_theta_slot_.assign(index.theta_slot.data(), index.theta_slot.size());
    d_vm_slot_.assign(index.vm_slot.data(), index.vm_slot.size());
    d_theta_active_.assign(theta_active.data(), theta_active.size());
    d_vm_active_.assign(vm_active.data(), vm_active.size());
    d_va_.resize(static_cast<std::size_t>(n_bus_));
    d_vm_.resize(static_cast<std::size_t>(n_bus_));
}

void BusLocalVoltageUpdate::apply(const double* dx_device,
                                  double* voltage_re_device,
                                  double* voltage_im_device)
{
    if (n_bus_ <= 0 || d_va_.empty() || d_vm_.empty()) {
        throw std::runtime_error("BusLocalVoltageUpdate::apply called before analyze");
    }
    if (dx_device == nullptr || voltage_re_device == nullptr || voltage_im_device == nullptr) {
        throw std::runtime_error("BusLocalVoltageUpdate::apply received null device input");
    }

    const int32_t grid = (n_bus_ + kBlockSize - 1) / kBlockSize;
    decompose_voltage_kernel<<<grid, kBlockSize>>>(
        n_bus_, voltage_re_device, voltage_im_device, d_va_.data(), d_vm_.data());
    CUDA_CHECK(cudaGetLastError());

    apply_bus_local_dx_kernel<<<grid, kBlockSize>>>(
        n_bus_,
        dx_device,
        d_theta_slot_.data(),
        d_vm_slot_.data(),
        d_theta_active_.data(),
        d_vm_active_.data(),
        d_va_.data(),
        d_vm_.data());
    CUDA_CHECK(cudaGetLastError());

    reconstruct_voltage_kernel<<<grid, kBlockSize>>>(
        n_bus_, d_va_.data(), d_vm_.data(), voltage_re_device, voltage_im_device);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace exp_20260414::amgx_v2
