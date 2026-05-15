#pragma once

// File responsibility:
//   - GpuBlockILU0::factorize
//   - Numeric dense scatter
//   - Block ILU(0) row updates
//   - Diagonal LU/inverse construction
//   - Factorization error collection

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace gpu_block_ilu0 {

inline GpuBlockILU0::Stats GpuBlockILU0::factorize(double shift_scale)
{
    return factorize_device(d_.values.data(), d_.values.size(), shift_scale);
}

inline GpuBlockILU0::Stats GpuBlockILU0::factorize_device(const double* d_values,
                                                          std::size_t count,
                                                          double shift_scale)
{
    Stats stats;
    stats.setup_ms = setup_ms_;

    const int32_t nnz = static_cast<int32_t>(gpu_.a_perm.values.size());
    if (count != 0 && count != static_cast<std::size_t>(nnz)) {
        throw std::runtime_error("GpuBlockILU0::factorize_device value count mismatch");
    }
    if (d_values != nullptr && d_values != d_.values.data()) {
        CUITER_CUDA_CHECK(cudaMemcpyAsync(d_.values.data(),
                                          d_values,
                                          static_cast<std::size_t>(nnz) * sizeof(double),
                                          cudaMemcpyDeviceToDevice,
                                          options_.stream));
    }

    reset_numeric_buffers();

    cuiter::CudaEventTimer total_timer;
    total_timer.start(options_.stream);

    scatter_current_values(nnz, stats);

    const float shift = static_cast<float>(shift_scale);
    for (int32_t row = 0; row < gpu_.pattern.num_blocks; ++row) {
        eliminate_lower_blocks(row, stats);
        invert_diagonal_block(row, shift, stats);
    }

    stats.factor_total_ms = 1000.0 * total_timer.stop(options_.stream);
    copy_factor_error(stats);
    return stats;
}

inline void GpuBlockILU0::reset_numeric_buffers()
{
    CUITER_CUDA_CHECK(cudaMemsetAsync(d_.blocks.data(),
                                      0,
                                      d_.blocks.size() * sizeof(float),
                                      options_.stream));
    CUITER_CUDA_CHECK(cudaMemsetAsync(d_.inv_diag.data(),
                                      0,
                                      d_.inv_diag.size() * sizeof(float),
                                      options_.stream));
    CUITER_CUDA_CHECK(cudaMemsetAsync(d_.diag_work.data(),
                                      0,
                                      d_.diag_work.size() * sizeof(float),
                                      options_.stream));
    CUITER_CUDA_CHECK(cudaMemsetAsync(d_.getrf_info.data(),
                                      0,
                                      d_.getrf_info.size() * sizeof(int32_t),
                                      options_.stream));
    CUITER_CUDA_CHECK(cudaMemsetAsync(d_.getri_info.data(),
                                      0,
                                      d_.getri_info.size() * sizeof(int32_t),
                                      options_.stream));
}

inline void GpuBlockILU0::scatter_current_values(int32_t nnz, Stats& stats)
{
    launch_or_profile(stats.dense_scatter_ms, [&] {
        const int threads = 256;
        const int grid = (nnz + threads - 1) / threads;
        scatter_dense_kernel<<<grid, threads, 0, options_.stream>>>(
            nnz,
            gpu_.pad,
            d_.values.data(),
            d_.nnz_block.data(),
            d_.nnz_local_row.data(),
            d_.nnz_local_col.data(),
            d_.blocks.data());
        CUITER_CUDA_CHECK(cudaGetLastError());
    });
}

inline void GpuBlockILU0::eliminate_lower_blocks(int32_t row, Stats& stats)
{
    for (int32_t ptr = gpu_.row_block_ptr[static_cast<std::size_t>(row)];
         ptr < gpu_.row_block_ptr[static_cast<std::size_t>(row + 1)];
         ++ptr) {
        const int32_t lower_index =
            gpu_.row_block_indices_sorted[static_cast<std::size_t>(ptr)];
        const auto& lower_block = gpu_.pattern.blocks[static_cast<std::size_t>(lower_index)];
        const int32_t pivot_col = lower_block.col;
        if (pivot_col >= row) {
            continue;
        }

        const int32_t rows = gpu_.pattern.block_dims[static_cast<std::size_t>(row)];
        const int32_t inner = gpu_.pattern.block_dims[static_cast<std::size_t>(pivot_col)];
        launch_or_profile(stats.factor_right_ms, [&] {
            right_multiply_kernel<<<1, dim3(gpu_.pad, gpu_.pad), 0, options_.stream>>>(
                d_.blocks.data(),
                d_.inv_diag.data(),
                lower_index,
                pivot_col,
                rows,
                inner,
                gpu_.pad);
            CUITER_CUDA_CHECK(cudaGetLastError());
        });

        const int32_t begin = gpu_.update_begin[static_cast<std::size_t>(lower_index)];
        const int32_t count = gpu_.update_count[static_cast<std::size_t>(lower_index)];
        if (count == 0) {
            continue;
        }

        launch_or_profile(stats.factor_update_ms, [&] {
            subtract_product_batch_kernel<<<count,
                                            dim3(gpu_.pad, gpu_.pad),
                                            0,
                                            options_.stream>>>(d_.blocks.data(),
                                                               d_.update_target.data(),
                                                               d_.update_rhs.data(),
                                                               begin,
                                                               count,
                                                               lower_index,
                                                               inner,
                                                               gpu_.pad);
            CUITER_CUDA_CHECK(cudaGetLastError());
        });
    }
}

inline void GpuBlockILU0::invert_diagonal_block(int32_t block, float shift, Stats& stats)
{
    const int32_t diag_index = gpu_.pattern.diagonal_index[static_cast<std::size_t>(block)];
    const int32_t dim = gpu_.pattern.block_dims[static_cast<std::size_t>(block)];

    launch_or_profile(stats.factor_diag_inv_ms, [&] {
        prepare_diag_for_cublas_kernel<<<1,
                                         dim3(gpu_.pad, gpu_.pad),
                                         0,
                                         options_.stream>>>(d_.blocks.data(),
                                                            d_.diag_work.data(),
                                                            block,
                                                            diag_index,
                                                            dim,
                                                            gpu_.pad,
                                                            shift);
        CUITER_CUDA_CHECK(cudaGetLastError());

        CUITER_CUBLAS_CHECK(cublasSgetrfBatched(cublas_,
                                                dim,
                                                d_.diag_ptrs.data() + block,
                                                gpu_.pad,
                                                d_.pivots.data() + block * gpu_.pad,
                                                d_.getrf_info.data() + block,
                                                1));
        CUITER_CUBLAS_CHECK(cublasSgetriBatched(cublas_,
                                                dim,
                                                d_.diag_ptrs.data() + block,
                                                gpu_.pad,
                                                d_.pivots.data() + block * gpu_.pad,
                                                d_.inv_ptrs.data() + block,
                                                gpu_.pad,
                                                d_.getri_info.data() + block,
                                                1));
    });
}

inline void GpuBlockILU0::copy_factor_error(Stats& stats)
{
    std::vector<int32_t> getrf_info(static_cast<std::size_t>(gpu_.pattern.num_blocks), 0);
    std::vector<int32_t> getri_info(static_cast<std::size_t>(gpu_.pattern.num_blocks), 0);
    d_.getrf_info.copy_to(getrf_info.data(), getrf_info.size());
    d_.getri_info.copy_to(getri_info.data(), getri_info.size());

    for (int32_t block = 0; block < static_cast<int32_t>(getrf_info.size()); ++block) {
        if (getrf_info[static_cast<std::size_t>(block)] != 0 ||
            getri_info[static_cast<std::size_t>(block)] != 0) {
            stats.factor_failed = true;
            stats.failed_block = block;
            return;
        }
    }
}

}  // namespace gpu_block_ilu0
