#include "bus_local_jacobian_assembler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace exp_20260414::amgx_v2 {
namespace {

constexpr int32_t kBlockSize = 256;

__device__ double atomic_add_f64(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

int32_t find_csr_position(const BusLocalJacobianPattern& pattern, int32_t row, int32_t col)
{
    if (row < 0 || row >= pattern.dim || col < 0 || col >= pattern.dim) {
        return -1;
    }
    const auto begin = pattern.col_idx.begin() + pattern.row_ptr[static_cast<std::size_t>(row)];
    const auto end = pattern.col_idx.begin() + pattern.row_ptr[static_cast<std::size_t>(row + 1)];
    const auto it = std::lower_bound(begin, end, col);
    if (it == end || *it != col) {
        return -1;
    }
    return static_cast<int32_t>(it - pattern.col_idx.begin());
}

void set_position_if_active(std::vector<int32_t>& map, std::size_t k, int32_t position)
{
    map[k] = position >= 0 ? position : -1;
}

__global__ void set_fixed_identity_kernel(int32_t count,
                                          const int32_t* __restrict__ positions,
                                          double* __restrict__ values)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        values[positions[i]] = 1.0;
    }
}

// Edge-based analytic Jacobian fill for the augmented bus-local layout.
//
// The algebra is the same as cuPF's FP64 edge Jacobian. The difference is that
// row and column positions already refer to [bus.theta, bus.Vm] slots, including
// fixed identity slots for slack and PV Vm entries.
__global__ void assemble_bus_local_jacobian_kernel(
    int32_t n_elements,
    const double* __restrict__ y_re,
    const double* __restrict__ y_im,
    const int32_t* __restrict__ y_row,
    const int32_t* __restrict__ y_col,
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    const int32_t* __restrict__ map11,
    const int32_t* __restrict__ map21,
    const int32_t* __restrict__ map12,
    const int32_t* __restrict__ map22,
    const int32_t* __restrict__ diag11,
    const int32_t* __restrict__ diag21,
    const int32_t* __restrict__ diag12,
    const int32_t* __restrict__ diag22,
    double* __restrict__ values)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_elements) {
        return;
    }

    const int32_t i = y_row[k];
    const int32_t j = y_col[k];

    const double yr = y_re[k];
    const double yi = y_im[k];
    const double vi_re = v_re[i];
    const double vi_im = v_im[i];
    const double vj_re = v_re[j];
    const double vj_im = v_im[j];

    const double curr_re = yr * vj_re - yi * vj_im;
    const double curr_im = yr * vj_im + yi * vj_re;

    const double neg_j_vi_re = vi_im;
    const double neg_j_vi_im = -vi_re;
    const double term_va_re = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;
    const double term_va_im = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;

    const double vj_abs = hypot(vj_re, vj_im);
    double term_vm_re = 0.0;
    double term_vm_im = 0.0;
    if (vj_abs > 1e-12) {
        const double scaled_re = curr_re / vj_abs;
        const double scaled_im = curr_im / vj_abs;
        term_vm_re = vi_re * scaled_re + vi_im * scaled_im;
        term_vm_im = vi_im * scaled_re - vi_re * scaled_im;
    }

    if (map11[k] >= 0) atomic_add_f64(&values[map11[k]], term_va_re);
    if (map21[k] >= 0) atomic_add_f64(&values[map21[k]], term_va_im);
    if (map12[k] >= 0) atomic_add_f64(&values[map12[k]], term_vm_re);
    if (map22[k] >= 0) atomic_add_f64(&values[map22[k]], term_vm_im);

    if (diag11[i] >= 0) atomic_add_f64(&values[diag11[i]], -term_va_re);
    if (diag21[i] >= 0) atomic_add_f64(&values[diag21[i]], -term_va_im);

    const double vi_abs = hypot(vi_re, vi_im);
    if (vi_abs > 1e-12) {
        const double vi_norm_re = vi_re / vi_abs;
        const double vi_norm_im = vi_im / vi_abs;
        const double term_vm2_re = vi_norm_re * curr_re + vi_norm_im * curr_im;
        const double term_vm2_im = vi_norm_im * curr_re - vi_norm_re * curr_im;
        if (diag12[i] >= 0) atomic_add_f64(&values[diag12[i]], term_vm2_re);
        if (diag22[i] >= 0) atomic_add_f64(&values[diag22[i]], term_vm2_im);
    }
}

}  // namespace

void BusLocalJacobianAssembler::analyze(const BusLocalIndex& index,
                                        const BusLocalJacobianPattern& pattern,
                                        const std::vector<int32_t>& ybus_row_ptr,
                                        const std::vector<int32_t>& ybus_col_idx)
{
    if (pattern.dim != index.dim) {
        throw std::runtime_error("bus-local Jacobian pattern dimension does not match index");
    }
    if (ybus_row_ptr.size() != static_cast<std::size_t>(index.n_bus + 1)) {
        throw std::runtime_error("Ybus row_ptr size does not match bus-local index");
    }
    if (ybus_row_ptr.front() != 0 ||
        ybus_row_ptr.back() != static_cast<int32_t>(ybus_col_idx.size())) {
        throw std::runtime_error("Ybus CSR pointers are inconsistent");
    }

    dim_ = pattern.dim;
    nnz_ = pattern.nnz;
    ybus_nnz_ = static_cast<int32_t>(ybus_col_idx.size());

    std::vector<int32_t> y_row(static_cast<std::size_t>(ybus_nnz_), 0);
    std::vector<int32_t> map11(static_cast<std::size_t>(ybus_nnz_), -1);
    std::vector<int32_t> map21(static_cast<std::size_t>(ybus_nnz_), -1);
    std::vector<int32_t> map12(static_cast<std::size_t>(ybus_nnz_), -1);
    std::vector<int32_t> map22(static_cast<std::size_t>(ybus_nnz_), -1);
    std::vector<int32_t> diag11(static_cast<std::size_t>(index.n_bus), -1);
    std::vector<int32_t> diag21(static_cast<std::size_t>(index.n_bus), -1);
    std::vector<int32_t> diag12(static_cast<std::size_t>(index.n_bus), -1);
    std::vector<int32_t> diag22(static_cast<std::size_t>(index.n_bus), -1);
    std::vector<int32_t> fixed_identity_pos;

    for (int32_t bus = 0; bus < index.n_bus; ++bus) {
        const int32_t p_row = index.theta(bus);
        const int32_t q_row = index.vm(bus);
        if (index.is_p_active(bus)) {
            diag11[static_cast<std::size_t>(bus)] =
                find_csr_position(pattern, p_row, index.theta(bus));
            diag12[static_cast<std::size_t>(bus)] =
                find_csr_position(pattern, p_row, index.vm(bus));
        } else {
            fixed_identity_pos.push_back(find_csr_position(pattern, p_row, p_row));
        }
        if (index.is_q_active(bus)) {
            diag21[static_cast<std::size_t>(bus)] =
                find_csr_position(pattern, q_row, index.theta(bus));
            diag22[static_cast<std::size_t>(bus)] =
                find_csr_position(pattern, q_row, index.vm(bus));
        } else {
            fixed_identity_pos.push_back(find_csr_position(pattern, q_row, q_row));
        }
    }
    for (int32_t pos : fixed_identity_pos) {
        if (pos < 0) {
            throw std::runtime_error("fixed identity slot is missing from bus-local pattern");
        }
    }

    for (int32_t row_bus = 0; row_bus < index.n_bus; ++row_bus) {
        for (int32_t y_pos = ybus_row_ptr[static_cast<std::size_t>(row_bus)];
             y_pos < ybus_row_ptr[static_cast<std::size_t>(row_bus + 1)];
             ++y_pos) {
            const int32_t col_bus = ybus_col_idx[static_cast<std::size_t>(y_pos)];
            y_row[static_cast<std::size_t>(y_pos)] = row_bus;

            if (index.is_p_active(row_bus)) {
                const int32_t p_row = index.theta(row_bus);
                set_position_if_active(map11,
                                       static_cast<std::size_t>(y_pos),
                                       find_csr_position(pattern, p_row, index.theta(col_bus)));
                set_position_if_active(map12,
                                       static_cast<std::size_t>(y_pos),
                                       find_csr_position(pattern, p_row, index.vm(col_bus)));
            }
            if (index.is_q_active(row_bus)) {
                const int32_t q_row = index.vm(row_bus);
                set_position_if_active(map21,
                                       static_cast<std::size_t>(y_pos),
                                       find_csr_position(pattern, q_row, index.theta(col_bus)));
                set_position_if_active(map22,
                                       static_cast<std::size_t>(y_pos),
                                       find_csr_position(pattern, q_row, index.vm(col_bus)));
            }
        }
    }

    d_row_ptr_.assign(pattern.row_ptr.data(), pattern.row_ptr.size());
    d_col_idx_.assign(pattern.col_idx.data(), pattern.col_idx.size());
    d_values_.resize(static_cast<std::size_t>(nnz_));
    d_y_row_.assign(y_row.data(), y_row.size());
    d_y_col_.assign(ybus_col_idx.data(), ybus_col_idx.size());
    d_map11_.assign(map11.data(), map11.size());
    d_map21_.assign(map21.data(), map21.size());
    d_map12_.assign(map12.data(), map12.size());
    d_map22_.assign(map22.data(), map22.size());
    d_diag11_.assign(diag11.data(), diag11.size());
    d_diag21_.assign(diag21.data(), diag21.size());
    d_diag12_.assign(diag12.data(), diag12.size());
    d_diag22_.assign(diag22.data(), diag22.size());
    d_fixed_identity_pos_.assign(fixed_identity_pos.data(), fixed_identity_pos.size());
}

void BusLocalJacobianAssembler::assemble(const double* ybus_re_device,
                                         const double* ybus_im_device,
                                         const double* voltage_re_device,
                                         const double* voltage_im_device)
{
    if (dim_ <= 0 || nnz_ <= 0 || ybus_nnz_ <= 0) {
        throw std::runtime_error("BusLocalJacobianAssembler::assemble called before analyze");
    }
    if (ybus_re_device == nullptr || ybus_im_device == nullptr ||
        voltage_re_device == nullptr || voltage_im_device == nullptr) {
        throw std::runtime_error("BusLocalJacobianAssembler::assemble received null device input");
    }

    d_values_.memsetZero();

    const int32_t y_grid = (ybus_nnz_ + kBlockSize - 1) / kBlockSize;
    assemble_bus_local_jacobian_kernel<<<y_grid, kBlockSize>>>(
        ybus_nnz_,
        ybus_re_device,
        ybus_im_device,
        d_y_row_.data(),
        d_y_col_.data(),
        voltage_re_device,
        voltage_im_device,
        d_map11_.data(),
        d_map21_.data(),
        d_map12_.data(),
        d_map22_.data(),
        d_diag11_.data(),
        d_diag21_.data(),
        d_diag12_.data(),
        d_diag22_.data(),
        d_values_.data());
    CUDA_CHECK(cudaGetLastError());

    const int32_t fixed_count = static_cast<int32_t>(d_fixed_identity_pos_.size());
    if (fixed_count > 0) {
        const int32_t fixed_grid = (fixed_count + kBlockSize - 1) / kBlockSize;
        set_fixed_identity_kernel<<<fixed_grid, kBlockSize>>>(
            fixed_count, d_fixed_identity_pos_.data(), d_values_.data());
        CUDA_CHECK(cudaGetLastError());
    }
}

CsrMatrixView BusLocalJacobianAssembler::device_matrix_view() const
{
    if (dim_ <= 0 || nnz_ <= 0) {
        return {};
    }
    return CsrMatrixView{
        .rows = dim_,
        .nnz = nnz_,
        .row_ptr = d_row_ptr_.data(),
        .col_idx = d_col_idx_.data(),
        .values = d_values_.data(),
    };
}

void BusLocalJacobianAssembler::download_values(std::vector<double>& values) const
{
    values.resize(static_cast<std::size_t>(nnz_));
    d_values_.copyTo(values.data(), values.size());
}

}  // namespace exp_20260414::amgx_v2
