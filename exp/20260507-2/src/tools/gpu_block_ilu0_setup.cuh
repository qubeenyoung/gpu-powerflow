#pragma once

// File responsibility:
//   - Symbolic block ILU(0) setup
//   - Dense scatter metadata
//   - Device buffer allocation
//   - GpuBlockILU0::setup

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_block_ilu0 {
namespace detail {

inline std::vector<int32_t> build_block_id_by_row(int32_t n,
                                                  const std::vector<int32_t>& starts,
                                                  const std::vector<int32_t>& dims)
{
    std::vector<int32_t> block_of_row(static_cast<std::size_t>(n), -1);
    for (int32_t block = 0; block < static_cast<int32_t>(dims.size()); ++block) {
        const int32_t begin = starts[static_cast<std::size_t>(block)];
        const int32_t end = begin + dims[static_cast<std::size_t>(block)];
        for (int32_t row = begin; row < end; ++row) {
            block_of_row[static_cast<std::size_t>(row)] = block;
        }
    }
    return block_of_row;
}

inline void build_sorted_block_rows(GpuPattern& gpu)
{
    gpu.row_block_ptr.assign(static_cast<std::size_t>(gpu.pattern.num_blocks + 1), 0);
    gpu.row_block_indices_sorted.clear();

    for (int32_t row = 0; row < gpu.pattern.num_blocks; ++row) {
        std::vector<int32_t> row_blocks =
            gpu.pattern.row_block_indices[static_cast<std::size_t>(row)];
        std::sort(row_blocks.begin(), row_blocks.end(), [&](int32_t lhs, int32_t rhs) {
            return gpu.pattern.blocks[static_cast<std::size_t>(lhs)].col <
                   gpu.pattern.blocks[static_cast<std::size_t>(rhs)].col;
        });

        gpu.row_block_ptr[static_cast<std::size_t>(row)] =
            static_cast<int32_t>(gpu.row_block_indices_sorted.size());
        gpu.row_block_indices_sorted.insert(gpu.row_block_indices_sorted.end(),
                                            row_blocks.begin(),
                                            row_blocks.end());
    }

    gpu.row_block_ptr[static_cast<std::size_t>(gpu.pattern.num_blocks)] =
        static_cast<int32_t>(gpu.row_block_indices_sorted.size());
}

inline void build_ilu_update_lists(GpuPattern& gpu)
{
    gpu.update_begin.assign(gpu.pattern.blocks.size(), 0);
    gpu.update_count.assign(gpu.pattern.blocks.size(), 0);

    for (int32_t row = 0; row < gpu.pattern.num_blocks; ++row) {
        for (int32_t ptr = gpu.row_block_ptr[static_cast<std::size_t>(row)];
             ptr < gpu.row_block_ptr[static_cast<std::size_t>(row + 1)];
             ++ptr) {
            const int32_t lower_index =
                gpu.row_block_indices_sorted[static_cast<std::size_t>(ptr)];
            const auto& lower_block = gpu.pattern.blocks[static_cast<std::size_t>(lower_index)];
            const int32_t pivot_col = lower_block.col;
            if (pivot_col >= row) {
                continue;
            }

            gpu.lower_blocks.push_back(lower_index);
            gpu.update_begin[static_cast<std::size_t>(lower_index)] =
                static_cast<int32_t>(gpu.update_target.size());

            for (int32_t upper_index :
                 gpu.pattern.row_block_indices[static_cast<std::size_t>(pivot_col)]) {
                const auto& upper_block =
                    gpu.pattern.blocks[static_cast<std::size_t>(upper_index)];
                const int32_t target_col = upper_block.col;
                if (target_col <= pivot_col) {
                    continue;
                }

                const int32_t target =
                    cuiter::cpu_pilot::detail::find_block(gpu.pattern, row, target_col);
                if (target < 0) {
                    continue;
                }

                gpu.update_target.push_back(target);
                gpu.update_rhs.push_back(upper_index);
                gpu.work.offdiag_update_work +=
                    static_cast<double>(gpu.pattern.block_dims[static_cast<std::size_t>(row)]) *
                    static_cast<double>(
                        gpu.pattern.block_dims[static_cast<std::size_t>(pivot_col)]) *
                    static_cast<double>(
                        gpu.pattern.block_dims[static_cast<std::size_t>(target_col)]);
            }

            gpu.update_count[static_cast<std::size_t>(lower_index)] =
                static_cast<int32_t>(gpu.update_target.size()) -
                gpu.update_begin[static_cast<std::size_t>(lower_index)];
        }
    }
}

inline void compute_work_estimates(GpuPattern& gpu)
{
    for (int32_t block_index = 0; block_index < static_cast<int32_t>(gpu.pattern.blocks.size());
         ++block_index) {
        const auto& block = gpu.pattern.blocks[static_cast<std::size_t>(block_index)];
        const int32_t rows = gpu.pattern.block_dims[static_cast<std::size_t>(block.row)];
        const int32_t cols = gpu.pattern.block_dims[static_cast<std::size_t>(block.col)];

        if (block.row == block.col) {
            gpu.work.diag_factor_work += static_cast<double>(rows) * rows * rows;
            gpu.work.diag_apply_work += static_cast<double>(rows) * rows;
        } else if (block.row > block.col) {
            gpu.work.offdiag_apply_work += static_cast<double>(rows) * cols;
            gpu.work.lower_edges++;
        } else {
            gpu.work.offdiag_apply_work += static_cast<double>(rows) * cols;
            gpu.work.upper_edges++;
            gpu.upper_blocks.push_back(block_index);
        }
    }

    gpu.work.total_factor_work = gpu.work.diag_factor_work + gpu.work.offdiag_update_work;
    gpu.work.total_apply_work = gpu.work.diag_apply_work + gpu.work.offdiag_apply_work;
    gpu.work.factor_offdiag_work_share =
        gpu.work.total_factor_work > 0.0
            ? gpu.work.offdiag_update_work / gpu.work.total_factor_work
            : 0.0;
    gpu.work.apply_offdiag_work_share =
        gpu.work.total_apply_work > 0.0
            ? gpu.work.offdiag_apply_work / gpu.work.total_apply_work
            : 0.0;
    gpu.work.factor_work_relative_to_bj =
        gpu.work.diag_factor_work > 0.0
            ? gpu.work.total_factor_work / gpu.work.diag_factor_work
            : 0.0;
    gpu.work.apply_work_relative_to_bj =
        gpu.work.diag_apply_work > 0.0
            ? gpu.work.total_apply_work / gpu.work.diag_apply_work
            : 0.0;
}

inline void build_dense_scatter_maps(GpuPattern& gpu)
{
    const std::vector<int32_t> block_of_row =
        build_block_id_by_row(gpu.a_perm.rows, gpu.pattern.block_starts, gpu.pattern.block_dims);

    gpu.nnz_block.resize(gpu.a_perm.values.size());
    gpu.nnz_local_row.resize(gpu.a_perm.values.size());
    gpu.nnz_local_col.resize(gpu.a_perm.values.size());

    for (int32_t row = 0; row < gpu.a_perm.rows; ++row) {
        const int32_t row_block = block_of_row[static_cast<std::size_t>(row)];
        const int32_t local_row =
            row - gpu.pattern.block_starts[static_cast<std::size_t>(row_block)];

        for (int32_t pos = gpu.a_perm.row_ptr[static_cast<std::size_t>(row)];
             pos < gpu.a_perm.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = gpu.a_perm.col_idx[static_cast<std::size_t>(pos)];
            const int32_t col_block = block_of_row[static_cast<std::size_t>(col)];
            const int32_t local_col =
                col - gpu.pattern.block_starts[static_cast<std::size_t>(col_block)];
            const int32_t block_index =
                cuiter::cpu_pilot::detail::find_block(gpu.pattern, row_block, col_block);

            gpu.nnz_block[static_cast<std::size_t>(pos)] = block_index;
            gpu.nnz_local_row[static_cast<std::size_t>(pos)] = local_row;
            gpu.nnz_local_col[static_cast<std::size_t>(pos)] = local_col;
        }
    }
}

}  // namespace detail

inline GpuPattern build_gpu_pattern(const cuiter::CsrMatrix& matrix,
                                    const std::vector<double>& rhs,
                                    int32_t block_size,
                                    int32_t requested_pad,
                                    const std::string& case_name)
{
    GpuPattern gpu;

    const cuiter::MetisPermutation permutation =
        cuiter::cpu_pilot::detail::build_colored_block_permutation(matrix, block_size);
    gpu.a_perm = cuiter::cpu_pilot::detail::build_permuted_matrix(matrix, permutation);
    gpu.b_perm = cuiter::cpu_pilot::detail::permute_vector(rhs, permutation);
    gpu.pattern = cuiter::cpu_pilot::detail::make_block_pattern(
        gpu.a_perm, permutation.block_starts, permutation.block_sizes, false);

    const int32_t max_dim =
        *std::max_element(gpu.pattern.block_dims.begin(), gpu.pattern.block_dims.end());
    gpu.pad = std::max(requested_pad, max_dim);
    if (gpu.pad > kMaxBlockDim) {
        throw std::runtime_error("gpu_block_ilu0 supports max block dim 32");
    }

    gpu.block_rows.reserve(gpu.pattern.blocks.size());
    gpu.block_cols.reserve(gpu.pattern.blocks.size());
    for (const auto& block : gpu.pattern.blocks) {
        gpu.block_rows.push_back(block.row);
        gpu.block_cols.push_back(block.col);
    }

    detail::build_sorted_block_rows(gpu);
    detail::build_ilu_update_lists(gpu);
    detail::compute_work_estimates(gpu);
    detail::build_dense_scatter_maps(gpu);

    gpu.work.case_name = case_name;
    gpu.work.block_size = block_size;
    gpu.work.n = gpu.a_perm.rows;
    gpu.work.nnz = static_cast<int32_t>(gpu.a_perm.values.size());
    gpu.work.num_blocks = gpu.pattern.num_blocks;
    gpu.work.block_nnz = static_cast<int32_t>(gpu.pattern.blocks.size());
    gpu.work.max_block_dim = gpu.pad;
    gpu.work.update_ops = static_cast<int32_t>(gpu.update_target.size());
    return gpu;
}

inline DevicePatternBuffers make_device_buffers(const GpuPattern& gpu)
{
    DevicePatternBuffers d;
    d.values.assign(gpu.a_perm.values.data(), gpu.a_perm.values.size());
    d.rhs.assign(gpu.b_perm.data(), gpu.b_perm.size());
    d.blocks.resize(static_cast<std::size_t>(gpu.pattern.blocks.size() * gpu.pad * gpu.pad));
    d.inv_diag.resize(static_cast<std::size_t>(gpu.pattern.num_blocks * gpu.pad * gpu.pad));
    d.diag_work.resize(static_cast<std::size_t>(gpu.pattern.num_blocks * gpu.pad * gpu.pad));
    d.y.resize(static_cast<std::size_t>(gpu.a_perm.rows));
    d.z.resize(static_cast<std::size_t>(gpu.a_perm.rows));
    d.nnz_block.assign(gpu.nnz_block.data(), gpu.nnz_block.size());
    d.nnz_local_row.assign(gpu.nnz_local_row.data(), gpu.nnz_local_row.size());
    d.nnz_local_col.assign(gpu.nnz_local_col.data(), gpu.nnz_local_col.size());

    if (!gpu.update_target.empty()) {
        d.update_target.assign(gpu.update_target.data(), gpu.update_target.size());
        d.update_rhs.assign(gpu.update_rhs.data(), gpu.update_rhs.size());
    }

    d.pivots.resize(static_cast<std::size_t>(gpu.pattern.num_blocks * gpu.pad));
    d.getrf_info.resize(static_cast<std::size_t>(gpu.pattern.num_blocks));
    d.getri_info.resize(static_cast<std::size_t>(gpu.pattern.num_blocks));

    std::vector<float*> diag_ptrs(static_cast<std::size_t>(gpu.pattern.num_blocks), nullptr);
    std::vector<float*> inv_ptrs(static_cast<std::size_t>(gpu.pattern.num_blocks), nullptr);
    for (int32_t block = 0; block < gpu.pattern.num_blocks; ++block) {
        diag_ptrs[static_cast<std::size_t>(block)] =
            d.diag_work.data() + static_cast<std::ptrdiff_t>(block * gpu.pad * gpu.pad);
        inv_ptrs[static_cast<std::size_t>(block)] =
            d.inv_diag.data() + static_cast<std::ptrdiff_t>(block * gpu.pad * gpu.pad);
    }
    d.diag_ptrs.assign(diag_ptrs.data(), diag_ptrs.size());
    d.inv_ptrs.assign(inv_ptrs.data(), inv_ptrs.size());
    return d;
}

inline void GpuBlockILU0::setup(const cuiter::CsrMatrix& matrix,
                                const std::vector<double>& rhs,
                                const std::string& case_name,
                                const Options& options)
{
    options_ = options;

    cuiter::CudaEventTimer timer;
    timer.start(options_.stream);

    gpu_ = build_gpu_pattern(matrix, rhs, options.block_size, options.pad, case_name);
    d_ = make_device_buffers(gpu_);

    if (cublas_ == nullptr) {
        CUITER_CUBLAS_CHECK(cublasCreate(&cublas_));
    }
    CUITER_CUBLAS_CHECK(cublasSetStream(cublas_, options_.stream));

    setup_ms_ = 1000.0 * timer.stop(options_.stream);
}

}  // namespace gpu_block_ilu0
