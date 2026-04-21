#include "bus_local_residual_assembler.hpp"

#include <stdexcept>

namespace exp_20260414::amgx_v2 {
namespace {

constexpr int32_t kBlockSize = 256;

__global__ void assemble_bus_local_residual_kernel(
    int32_t n_bus,
    const int32_t* __restrict__ ybus_row_ptr,
    const int32_t* __restrict__ ybus_col_idx,
    const double* __restrict__ ybus_re,
    const double* __restrict__ ybus_im,
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    const double* __restrict__ sbus_re,
    const double* __restrict__ sbus_im,
    const int32_t* __restrict__ theta_slot,
    const int32_t* __restrict__ vm_slot,
    const int32_t* __restrict__ p_active,
    const int32_t* __restrict__ q_active,
    double* __restrict__ residual)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }

    double i_re = 0.0;
    double i_im = 0.0;
    for (int32_t pos = ybus_row_ptr[bus]; pos < ybus_row_ptr[bus + 1]; ++pos) {
        const int32_t col = ybus_col_idx[pos];
        const double yr = ybus_re[pos];
        const double yi = ybus_im[pos];
        const double vr = v_re[col];
        const double vi = v_im[col];
        i_re += yr * vr - yi * vi;
        i_im += yr * vi + yi * vr;
    }

    const double vr = v_re[bus];
    const double vi = v_im[bus];
    const double mis_p = vr * i_re + vi * i_im - sbus_re[bus];
    const double mis_q = vi * i_re - vr * i_im - sbus_im[bus];

    residual[theta_slot[bus]] = p_active[bus] ? mis_p : 0.0;
    residual[vm_slot[bus]] = q_active[bus] ? mis_q : 0.0;
}

}  // namespace

void BusLocalResidualAssembler::analyze(const BusLocalIndex& index)
{
    if (index.n_bus <= 0 || index.dim != 2 * index.n_bus) {
        throw std::runtime_error("BusLocalResidualAssembler::analyze received invalid index");
    }

    n_bus_ = index.n_bus;
    dim_ = index.dim;

    std::vector<int32_t> p_active(static_cast<std::size_t>(n_bus_), 0);
    std::vector<int32_t> q_active(static_cast<std::size_t>(n_bus_), 0);
    for (int32_t bus = 0; bus < n_bus_; ++bus) {
        p_active[static_cast<std::size_t>(bus)] = index.is_p_active(bus) ? 1 : 0;
        q_active[static_cast<std::size_t>(bus)] = index.is_q_active(bus) ? 1 : 0;
    }

    d_theta_slot_.assign(index.theta_slot.data(), index.theta_slot.size());
    d_vm_slot_.assign(index.vm_slot.data(), index.vm_slot.size());
    d_p_active_.assign(p_active.data(), p_active.size());
    d_q_active_.assign(q_active.data(), q_active.size());
    d_values_.resize(static_cast<std::size_t>(dim_));
}

void BusLocalResidualAssembler::assemble(const int32_t* ybus_row_ptr_device,
                                         const int32_t* ybus_col_idx_device,
                                         const double* ybus_re_device,
                                         const double* ybus_im_device,
                                         const double* voltage_re_device,
                                         const double* voltage_im_device,
                                         const double* sbus_re_device,
                                         const double* sbus_im_device)
{
    if (n_bus_ <= 0 || dim_ <= 0 || d_values_.empty()) {
        throw std::runtime_error("BusLocalResidualAssembler::assemble called before analyze");
    }
    if (ybus_row_ptr_device == nullptr || ybus_col_idx_device == nullptr ||
        ybus_re_device == nullptr || ybus_im_device == nullptr ||
        voltage_re_device == nullptr || voltage_im_device == nullptr ||
        sbus_re_device == nullptr || sbus_im_device == nullptr) {
        throw std::runtime_error("BusLocalResidualAssembler::assemble received null device input");
    }

    const int32_t grid = (n_bus_ + kBlockSize - 1) / kBlockSize;
    assemble_bus_local_residual_kernel<<<grid, kBlockSize>>>(
        n_bus_,
        ybus_row_ptr_device,
        ybus_col_idx_device,
        ybus_re_device,
        ybus_im_device,
        voltage_re_device,
        voltage_im_device,
        sbus_re_device,
        sbus_im_device,
        d_theta_slot_.data(),
        d_vm_slot_.data(),
        d_p_active_.data(),
        d_q_active_.data(),
        d_values_.data());
    CUDA_CHECK(cudaGetLastError());
}

void BusLocalResidualAssembler::download_values(std::vector<double>& values) const
{
    values.resize(static_cast<std::size_t>(dim_));
    d_values_.copyTo(values.data(), values.size());
}

}  // namespace exp_20260414::amgx_v2
