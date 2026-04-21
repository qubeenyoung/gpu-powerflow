#include "bus_block_jacobian_assembler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace exp_20260414::amgx_v2 {
namespace {

constexpr int32_t kBlockDim = 2;
constexpr int32_t kBlockValues = kBlockDim * kBlockDim;
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

void add_unique_block_col(std::vector<int32_t>& cols, int32_t col)
{
    if (col >= 0) {
        cols.push_back(col);
    }
}

void finalize_block_cols(std::vector<int32_t>& cols)
{
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
}

int32_t find_block_position(const std::vector<int32_t>& row_ptr,
                            const std::vector<int32_t>& col_idx,
                            int32_t block_row,
                            int32_t block_col)
{
    const auto begin = col_idx.begin() + row_ptr[static_cast<std::size_t>(block_row)];
    const auto end = col_idx.begin() + row_ptr[static_cast<std::size_t>(block_row + 1)];
    const auto it = std::lower_bound(begin, end, block_col);
    if (it == end || *it != block_col) {
        return -1;
    }
    return static_cast<int32_t>(it - col_idx.begin());
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

// Direct edge-based analytic fill for the bus-local 2x2 block Jacobian. The
// formulas mirror BusLocalJacobianAssembler; only the target layout differs.
__global__ void assemble_bus_block_jacobian_kernel(
    int32_t n_elements,
    const double* __restrict__ y_re,
    const double* __restrict__ y_im,
    const int32_t* __restrict__ y_row,
    const int32_t* __restrict__ y_col,
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    const int32_t* __restrict__ edge_block_value_base,
    const int32_t* __restrict__ diag_block_value_base_by_bus,
    const int32_t* __restrict__ p_active,
    const int32_t* __restrict__ q_active,
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

    if (p_active[i]) {
        const int32_t edge_base = edge_block_value_base[k];
        atomic_add_f64(&values[edge_base + 0], term_va_re);
        atomic_add_f64(&values[edge_base + 1], term_vm_re);
    }
    if (q_active[i]) {
        const int32_t edge_base = edge_block_value_base[k];
        atomic_add_f64(&values[edge_base + 2], term_va_im);
        atomic_add_f64(&values[edge_base + 3], term_vm_im);
    }

    const int32_t diag_base = diag_block_value_base_by_bus[i];
    if (p_active[i]) {
        atomic_add_f64(&values[diag_base + 0], -term_va_re);
    }
    if (q_active[i]) {
        atomic_add_f64(&values[diag_base + 2], -term_va_im);
    }

    const double vi_abs = hypot(vi_re, vi_im);
    if (vi_abs > 1e-12) {
        const double vi_norm_re = vi_re / vi_abs;
        const double vi_norm_im = vi_im / vi_abs;
        const double term_vm2_re = vi_norm_re * curr_re + vi_norm_im * curr_im;
        const double term_vm2_im = vi_norm_im * curr_re - vi_norm_re * curr_im;
        if (p_active[i]) {
            atomic_add_f64(&values[diag_base + 1], term_vm2_re);
        }
        if (q_active[i]) {
            atomic_add_f64(&values[diag_base + 3], term_vm2_im);
        }
    }
}

}  // namespace

void BusBlockJacobianAssembler::analyze(const BusLocalIndex& index,
                                        const std::vector<int32_t>& ybus_row_ptr,
                                        const std::vector<int32_t>& ybus_col_idx)
{
    if (index.n_bus <= 0 || index.dim != 2 * index.n_bus) {
        throw std::runtime_error("BusBlockJacobianAssembler::analyze received invalid index");
    }
    if (ybus_row_ptr.size() != static_cast<std::size_t>(index.n_bus + 1)) {
        throw std::runtime_error("Ybus row_ptr size does not match bus-local index");
    }
    if (ybus_row_ptr.front() != 0 ||
        ybus_row_ptr.back() != static_cast<int32_t>(ybus_col_idx.size())) {
        throw std::runtime_error("Ybus CSR pointers are inconsistent");
    }

    n_bus_ = index.n_bus;
    ybus_nnz_ = static_cast<int32_t>(ybus_col_idx.size());
    block_rows_ = index.n_bus;

    std::vector<std::vector<int32_t>> block_cols(static_cast<std::size_t>(block_rows_));
    for (int32_t position = 0; position < block_rows_; ++position) {
        const int32_t row_bus = index.ordered_bus[static_cast<std::size_t>(position)];
        if (index.is_p_active(row_bus) || index.is_q_active(row_bus)) {
            for (int32_t y_pos = ybus_row_ptr[static_cast<std::size_t>(row_bus)];
                 y_pos < ybus_row_ptr[static_cast<std::size_t>(row_bus + 1)];
                 ++y_pos) {
                const int32_t col_bus = ybus_col_idx[static_cast<std::size_t>(y_pos)];
                if (col_bus < 0 || col_bus >= index.n_bus) {
                    throw std::runtime_error("Ybus column is out of range");
                }
                add_unique_block_col(block_cols[static_cast<std::size_t>(position)],
                                     index.bus_to_position[static_cast<std::size_t>(col_bus)]);
            }
        }
        add_unique_block_col(block_cols[static_cast<std::size_t>(position)], position);
    }

    std::vector<int32_t> row_ptr(static_cast<std::size_t>(block_rows_ + 1), 0);
    for (auto& cols : block_cols) {
        finalize_block_cols(cols);
    }
    for (int32_t row = 0; row < block_rows_; ++row) {
        row_ptr[static_cast<std::size_t>(row + 1)] =
            row_ptr[static_cast<std::size_t>(row)] +
            static_cast<int32_t>(block_cols[static_cast<std::size_t>(row)].size());
    }

    block_nnz_ = row_ptr.back();
    std::vector<int32_t> col_idx;
    col_idx.reserve(static_cast<std::size_t>(block_nnz_));
    for (const auto& cols : block_cols) {
        col_idx.insert(col_idx.end(), cols.begin(), cols.end());
    }

    std::vector<int32_t> y_row(static_cast<std::size_t>(ybus_nnz_), 0);
    std::vector<int32_t> edge_block_value_base(static_cast<std::size_t>(ybus_nnz_), -1);
    std::vector<int32_t> diag_base_by_bus(static_cast<std::size_t>(index.n_bus), -1);
    std::vector<int32_t> diag_base_by_position(static_cast<std::size_t>(index.n_bus), -1);
    std::vector<int32_t> p_active(static_cast<std::size_t>(index.n_bus), 0);
    std::vector<int32_t> q_active(static_cast<std::size_t>(index.n_bus), 0);
    std::vector<int32_t> fixed_identity_value_pos;

    for (int32_t position = 0; position < block_rows_; ++position) {
        const int32_t bus = index.ordered_bus[static_cast<std::size_t>(position)];
        const int32_t block_pos = find_block_position(row_ptr, col_idx, position, position);
        if (block_pos < 0) {
            throw std::runtime_error("diagonal 2x2 bus block is missing");
        }
        const int32_t value_base = block_pos * kBlockValues;
        diag_base_by_bus[static_cast<std::size_t>(bus)] = value_base;
        diag_base_by_position[static_cast<std::size_t>(position)] = value_base;

        p_active[static_cast<std::size_t>(bus)] = index.is_p_active(bus) ? 1 : 0;
        q_active[static_cast<std::size_t>(bus)] = index.is_q_active(bus) ? 1 : 0;
        if (!index.is_p_active(bus)) {
            fixed_identity_value_pos.push_back(value_base + 0);
        }
        if (!index.is_q_active(bus)) {
            fixed_identity_value_pos.push_back(value_base + 3);
        }
    }

    for (int32_t row_bus = 0; row_bus < index.n_bus; ++row_bus) {
        const int32_t block_row = index.bus_to_position[static_cast<std::size_t>(row_bus)];
        for (int32_t y_pos = ybus_row_ptr[static_cast<std::size_t>(row_bus)];
             y_pos < ybus_row_ptr[static_cast<std::size_t>(row_bus + 1)];
             ++y_pos) {
            y_row[static_cast<std::size_t>(y_pos)] = row_bus;
            if (!index.is_p_active(row_bus) && !index.is_q_active(row_bus)) {
                continue;
            }
            const int32_t col_bus = ybus_col_idx[static_cast<std::size_t>(y_pos)];
            const int32_t block_col = index.bus_to_position[static_cast<std::size_t>(col_bus)];
            const int32_t block_pos =
                find_block_position(row_ptr, col_idx, block_row, block_col);
            if (block_pos < 0) {
                throw std::runtime_error("missing Ybus edge in direct 2x2 block pattern");
            }
            edge_block_value_base[static_cast<std::size_t>(y_pos)] =
                block_pos * kBlockValues;
        }
    }

    d_row_ptr_.assign(row_ptr.data(), row_ptr.size());
    d_col_idx_.assign(col_idx.data(), col_idx.size());
    d_values_.resize(static_cast<std::size_t>(block_nnz_) * kBlockValues);
    d_y_row_.assign(y_row.data(), y_row.size());
    d_y_col_.assign(ybus_col_idx.data(), ybus_col_idx.size());
    d_edge_block_value_base_.assign(edge_block_value_base.data(), edge_block_value_base.size());
    d_diag_block_value_base_by_bus_.assign(diag_base_by_bus.data(), diag_base_by_bus.size());
    d_diag_block_value_base_by_position_.assign(diag_base_by_position.data(),
                                                diag_base_by_position.size());
    d_p_active_.assign(p_active.data(), p_active.size());
    d_q_active_.assign(q_active.data(), q_active.size());
    d_fixed_identity_value_pos_.assign(fixed_identity_value_pos.data(),
                                       fixed_identity_value_pos.size());
}

void BusBlockJacobianAssembler::assemble(const double* ybus_re_device,
                                         const double* ybus_im_device,
                                         const double* voltage_re_device,
                                         const double* voltage_im_device)
{
    if (block_rows_ <= 0 || block_nnz_ <= 0 || ybus_nnz_ <= 0 || d_values_.empty()) {
        throw std::runtime_error("BusBlockJacobianAssembler::assemble called before analyze");
    }
    if (ybus_re_device == nullptr || ybus_im_device == nullptr ||
        voltage_re_device == nullptr || voltage_im_device == nullptr) {
        throw std::runtime_error("BusBlockJacobianAssembler::assemble received null device input");
    }

    d_values_.memsetZero();

    const int32_t y_grid = (ybus_nnz_ + kBlockSize - 1) / kBlockSize;
    assemble_bus_block_jacobian_kernel<<<y_grid, kBlockSize>>>(
        ybus_nnz_,
        ybus_re_device,
        ybus_im_device,
        d_y_row_.data(),
        d_y_col_.data(),
        voltage_re_device,
        voltage_im_device,
        d_edge_block_value_base_.data(),
        d_diag_block_value_base_by_bus_.data(),
        d_p_active_.data(),
        d_q_active_.data(),
        d_values_.data());
    CUDA_CHECK(cudaGetLastError());

    const int32_t fixed_count = static_cast<int32_t>(d_fixed_identity_value_pos_.size());
    if (fixed_count > 0) {
        const int32_t fixed_grid = (fixed_count + kBlockSize - 1) / kBlockSize;
        set_fixed_identity_kernel<<<fixed_grid, kBlockSize>>>(
            fixed_count, d_fixed_identity_value_pos_.data(), d_values_.data());
        CUDA_CHECK(cudaGetLastError());
    }
}

BlockCsrMatrixView BusBlockJacobianAssembler::device_matrix_view() const
{
    if (block_rows_ <= 0 || block_nnz_ <= 0) {
        return {};
    }
    return BlockCsrMatrixView{
        .rows = block_rows_,
        .nnz = block_nnz_,
        .block_dim = kBlockDim,
        .row_ptr = d_row_ptr_.data(),
        .col_idx = d_col_idx_.data(),
        .values = d_values_.data(),
    };
}

BusBlockJacobiView BusBlockJacobianAssembler::jacobi_view() const
{
    if (block_rows_ <= 0 || d_diag_block_value_base_by_position_.empty()) {
        return {};
    }
    return BusBlockJacobiView{
        .block_rows = block_rows_,
        .diagonal_value_base = d_diag_block_value_base_by_position_.data(),
        .values = d_values_.data(),
    };
}

}  // namespace exp_20260414::amgx_v2
