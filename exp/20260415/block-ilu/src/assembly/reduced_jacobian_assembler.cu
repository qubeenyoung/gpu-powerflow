#include "assembly/reduced_jacobian_assembler.hpp"

#include <cmath>
#include <stdexcept>

namespace exp_20260415::block_ilu {
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

__device__ void add_if_valid(double* values, int32_t pos, double value)
{
    if (pos >= 0) {
        atomic_add_f64(&values[pos], value);
    }
}

__global__ void assemble_reduced_jacobian_kernel(
    int32_t n_elements,
    const double* __restrict__ y_re,
    const double* __restrict__ y_im,
    const int32_t* __restrict__ y_row,
    const int32_t* __restrict__ y_col,
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    const int32_t* __restrict__ map11,
    const int32_t* __restrict__ map12,
    const int32_t* __restrict__ map21,
    const int32_t* __restrict__ map22,
    const int32_t* __restrict__ diag11,
    const int32_t* __restrict__ diag12,
    const int32_t* __restrict__ diag21,
    const int32_t* __restrict__ diag22,
    const int32_t* __restrict__ map_b11,
    const int32_t* __restrict__ diag_b11,
    const int32_t* __restrict__ map_b12,
    const int32_t* __restrict__ diag_b12,
    const int32_t* __restrict__ map_b21,
    const int32_t* __restrict__ diag_b21,
    const int32_t* __restrict__ map_b22,
    const int32_t* __restrict__ diag_b22,
    double* __restrict__ full_values,
    double* __restrict__ j11_values,
    double* __restrict__ j12_values,
    double* __restrict__ j21_values,
    double* __restrict__ j22_values)
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

    add_if_valid(full_values, map11[k], term_va_re);
    add_if_valid(full_values, map12[k], term_vm_re);
    add_if_valid(full_values, map21[k], term_va_im);
    add_if_valid(full_values, map22[k], term_vm_im);

    add_if_valid(j11_values, map_b11[k], term_va_re);
    add_if_valid(j12_values, map_b12[k], term_vm_re);
    add_if_valid(j21_values, map_b21[k], term_va_im);
    add_if_valid(j22_values, map_b22[k], term_vm_im);

    add_if_valid(full_values, diag11[i], -term_va_re);
    add_if_valid(full_values, diag21[i], -term_va_im);
    add_if_valid(j11_values, diag_b11[i], -term_va_re);
    add_if_valid(j21_values, diag_b21[i], -term_va_im);

    const double vi_abs = hypot(vi_re, vi_im);
    if (vi_abs > 1e-12) {
        const double vi_norm_re = vi_re / vi_abs;
        const double vi_norm_im = vi_im / vi_abs;
        const double term_vm2_re = vi_norm_re * curr_re + vi_norm_im * curr_im;
        const double term_vm2_im = vi_norm_im * curr_re - vi_norm_re * curr_im;
        add_if_valid(full_values, diag12[i], term_vm2_re);
        add_if_valid(full_values, diag22[i], term_vm2_im);
        add_if_valid(j12_values, diag_b12[i], term_vm2_re);
        add_if_valid(j22_values, diag_b22[i], term_vm2_im);
    }
}

void upload_pattern(const HostCsrPattern& pattern,
                    DeviceBuffer<int32_t>& row_ptr,
                    DeviceBuffer<int32_t>& col_idx,
                    DeviceBuffer<double>& values)
{
    row_ptr.assign(pattern.row_ptr.data(), pattern.row_ptr.size());
    col_idx.assign(pattern.col_idx.data(), pattern.col_idx.size());
    values.resize(static_cast<std::size_t>(pattern.nnz()));
}

}  // namespace

void ReducedJacobianAssembler::analyze(const ReducedJacobianPatterns& patterns)
{
    if (patterns.full.rows <= 0 || patterns.full.nnz() <= 0 ||
        patterns.j11.rows <= 0 || patterns.j11.nnz() <= 0 ||
        patterns.j12.rows <= 0 || patterns.j12.nnz() <= 0 ||
        patterns.j21.rows <= 0 || patterns.j21.nnz() <= 0 ||
        patterns.j22.rows <= 0 || patterns.j22.nnz() <= 0 ||
        patterns.ybus_row.size() != patterns.ybus_col.size()) {
        throw std::runtime_error("ReducedJacobianAssembler::analyze received invalid patterns");
    }

    patterns_ = patterns;
    ybus_nnz_ = static_cast<int32_t>(patterns_.ybus_row.size());

    d_ybus_row_.assign(patterns_.ybus_row.data(), patterns_.ybus_row.size());
    d_ybus_col_.assign(patterns_.ybus_col.data(), patterns_.ybus_col.size());

    upload_pattern(patterns_.full, d_full_row_ptr_, d_full_col_idx_, d_full_values_);
    upload_pattern(patterns_.j11, d_j11_row_ptr_, d_j11_col_idx_, d_j11_values_);
    upload_pattern(patterns_.j12, d_j12_row_ptr_, d_j12_col_idx_, d_j12_values_);
    upload_pattern(patterns_.j21, d_j21_row_ptr_, d_j21_col_idx_, d_j21_values_);
    upload_pattern(patterns_.j22, d_j22_row_ptr_, d_j22_col_idx_, d_j22_values_);

    d_map11_.assign(patterns_.full_maps.map11.data(), patterns_.full_maps.map11.size());
    d_map12_.assign(patterns_.full_maps.map12.data(), patterns_.full_maps.map12.size());
    d_map21_.assign(patterns_.full_maps.map21.data(), patterns_.full_maps.map21.size());
    d_map22_.assign(patterns_.full_maps.map22.data(), patterns_.full_maps.map22.size());
    d_diag11_.assign(patterns_.full_maps.diag11.data(), patterns_.full_maps.diag11.size());
    d_diag12_.assign(patterns_.full_maps.diag12.data(), patterns_.full_maps.diag12.size());
    d_diag21_.assign(patterns_.full_maps.diag21.data(), patterns_.full_maps.diag21.size());
    d_diag22_.assign(patterns_.full_maps.diag22.data(), patterns_.full_maps.diag22.size());

    d_map_b11_.assign(patterns_.j11_maps.map.data(), patterns_.j11_maps.map.size());
    d_diag_b11_.assign(patterns_.j11_maps.diag.data(), patterns_.j11_maps.diag.size());
    d_map_b12_.assign(patterns_.j12_maps.map.data(), patterns_.j12_maps.map.size());
    d_diag_b12_.assign(patterns_.j12_maps.diag.data(), patterns_.j12_maps.diag.size());
    d_map_b21_.assign(patterns_.j21_maps.map.data(), patterns_.j21_maps.map.size());
    d_diag_b21_.assign(patterns_.j21_maps.diag.data(), patterns_.j21_maps.diag.size());
    d_map_b22_.assign(patterns_.j22_maps.map.data(), patterns_.j22_maps.map.size());
    d_diag_b22_.assign(patterns_.j22_maps.diag.data(), patterns_.j22_maps.diag.size());
}

void ReducedJacobianAssembler::assemble(const double* ybus_re_device,
                                        const double* ybus_im_device,
                                        const double* voltage_re_device,
                                        const double* voltage_im_device)
{
    if (ybus_nnz_ <= 0 || d_full_values_.empty() ||
        ybus_re_device == nullptr || ybus_im_device == nullptr ||
        voltage_re_device == nullptr || voltage_im_device == nullptr) {
        throw std::runtime_error("ReducedJacobianAssembler::assemble received invalid input");
    }

    d_full_values_.memsetZero();
    d_j11_values_.memsetZero();
    d_j12_values_.memsetZero();
    d_j21_values_.memsetZero();
    d_j22_values_.memsetZero();

    const int32_t grid = (ybus_nnz_ + kBlockSize - 1) / kBlockSize;
    assemble_reduced_jacobian_kernel<<<grid, kBlockSize>>>(
        ybus_nnz_,
        ybus_re_device,
        ybus_im_device,
        d_ybus_row_.data(),
        d_ybus_col_.data(),
        voltage_re_device,
        voltage_im_device,
        d_map11_.data(),
        d_map12_.data(),
        d_map21_.data(),
        d_map22_.data(),
        d_diag11_.data(),
        d_diag12_.data(),
        d_diag21_.data(),
        d_diag22_.data(),
        d_map_b11_.data(),
        d_diag_b11_.data(),
        d_map_b12_.data(),
        d_diag_b12_.data(),
        d_map_b21_.data(),
        d_diag_b21_.data(),
        d_map_b22_.data(),
        d_diag_b22_.data(),
        d_full_values_.data(),
        d_j11_values_.data(),
        d_j12_values_.data(),
        d_j21_values_.data(),
        d_j22_values_.data());
    CUDA_CHECK(cudaGetLastError());
}

DeviceCsrMatrixView ReducedJacobianAssembler::full_view() const
{
    return DeviceCsrMatrixView{
        .rows = patterns_.full.rows,
        .cols = patterns_.full.cols,
        .nnz = patterns_.full.nnz(),
        .row_ptr = d_full_row_ptr_.data(),
        .col_idx = d_full_col_idx_.data(),
        .values = d_full_values_.data(),
    };
}

DeviceCsrMatrixView ReducedJacobianAssembler::j11_view() const
{
    return DeviceCsrMatrixView{
        .rows = patterns_.j11.rows,
        .cols = patterns_.j11.cols,
        .nnz = patterns_.j11.nnz(),
        .row_ptr = d_j11_row_ptr_.data(),
        .col_idx = d_j11_col_idx_.data(),
        .values = d_j11_values_.data(),
    };
}

DeviceCsrMatrixView ReducedJacobianAssembler::j22_view() const
{
    return DeviceCsrMatrixView{
        .rows = patterns_.j22.rows,
        .cols = patterns_.j22.cols,
        .nnz = patterns_.j22.nnz(),
        .row_ptr = d_j22_row_ptr_.data(),
        .col_idx = d_j22_col_idx_.data(),
        .values = d_j22_values_.data(),
    };
}

DeviceCsrMatrixView ReducedJacobianAssembler::j12_view() const
{
    return DeviceCsrMatrixView{
        .rows = patterns_.j12.rows,
        .cols = patterns_.j12.cols,
        .nnz = patterns_.j12.nnz(),
        .row_ptr = d_j12_row_ptr_.data(),
        .col_idx = d_j12_col_idx_.data(),
        .values = d_j12_values_.data(),
    };
}

DeviceCsrMatrixView ReducedJacobianAssembler::j21_view() const
{
    return DeviceCsrMatrixView{
        .rows = patterns_.j21.rows,
        .cols = patterns_.j21.cols,
        .nnz = patterns_.j21.nnz(),
        .row_ptr = d_j21_row_ptr_.data(),
        .col_idx = d_j21_col_idx_.data(),
        .values = d_j21_values_.data(),
    };
}

const ReducedJacobianPatterns& ReducedJacobianAssembler::host_patterns() const
{
    return patterns_;
}

void ReducedJacobianAssembler::download_full_values(std::vector<double>& values) const
{
    values.resize(static_cast<std::size_t>(patterns_.full.nnz()));
    d_full_values_.copyTo(values.data(), values.size());
}

void ReducedJacobianAssembler::download_j11_values(std::vector<double>& values) const
{
    values.resize(static_cast<std::size_t>(patterns_.j11.nnz()));
    d_j11_values_.copyTo(values.data(), values.size());
}

void ReducedJacobianAssembler::download_j22_values(std::vector<double>& values) const
{
    values.resize(static_cast<std::size_t>(patterns_.j22.nnz()));
    d_j22_values_.copyTo(values.data(), values.size());
}

void ReducedJacobianAssembler::download_j12_values(std::vector<double>& values) const
{
    values.resize(static_cast<std::size_t>(patterns_.j12.nnz()));
    d_j12_values_.copyTo(values.data(), values.size());
}

void ReducedJacobianAssembler::download_j21_values(std::vector<double>& values) const
{
    values.resize(static_cast<std::size_t>(patterns_.j21.nnz()));
    d_j21_values_.copyTo(values.data(), values.size());
}

}  // namespace exp_20260415::block_ilu
