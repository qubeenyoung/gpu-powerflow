#pragma once

// File responsibility:
//   - GpuBlockILU0::apply
//   - Forward triangular apply
//   - Backward triangular apply
//   - Optional output norm copy for benchmark diagnostics

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpu_block_ilu0 {
namespace detail {

inline double host_norm2(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values) {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
}

}  // namespace detail

inline GpuBlockILU0::Stats GpuBlockILU0::apply()
{
    return apply_device(d_.rhs.data(), d_.z.data());
}

inline GpuBlockILU0::Stats GpuBlockILU0::apply_device(const double* d_rhs, double* d_out)
{
    Stats stats;
    const int32_t n = gpu_.a_perm.rows;
    const int threads = 256;
    const int grid = (n + threads - 1) / threads;

    cuiter::CudaEventTimer total_timer;
    total_timer.start(options_.stream);

    // Forward solve: y = L^{-1} rhs.
    launch_or_profile(stats.forward_ms, [&] {
        copy_vector_kernel<<<grid, threads, 0, options_.stream>>>(n, d_rhs, d_.y.data());
        CUITER_CUDA_CHECK(cudaGetLastError());
    });
    forward_solve(stats);

    // Backward solve: z = U^{-1} y.
    launch_or_profile(stats.backward_ms, [&] {
        copy_vector_kernel<<<grid, threads, 0, options_.stream>>>(n, d_.y.data(), d_out);
        CUITER_CUDA_CHECK(cudaGetLastError());
    });
    backward_solve(d_out, stats);

    stats.apply_total_ms = 1000.0 * total_timer.stop(options_.stream);
    maybe_copy_output_norm(d_out, stats);
    return stats;
}

inline void GpuBlockILU0::forward_solve(Stats& stats)
{
    for (int32_t block = 0; block < gpu_.pattern.num_blocks; ++block) {
        const int32_t row_begin = gpu_.pattern.block_starts[static_cast<std::size_t>(block)];
        const int32_t rows = gpu_.pattern.block_dims[static_cast<std::size_t>(block)];

        for (int32_t ptr = gpu_.row_block_ptr[static_cast<std::size_t>(block)];
             ptr < gpu_.row_block_ptr[static_cast<std::size_t>(block + 1)];
             ++ptr) {
            const int32_t index = gpu_.row_block_indices_sorted[static_cast<std::size_t>(ptr)];
            const auto& offdiag = gpu_.pattern.blocks[static_cast<std::size_t>(index)];
            if (offdiag.col >= block) {
                continue;
            }

            const int32_t col_begin =
                gpu_.pattern.block_starts[static_cast<std::size_t>(offdiag.col)];
            const int32_t cols = gpu_.pattern.block_dims[static_cast<std::size_t>(offdiag.col)];
            const int grid = (rows + 127) / 128;
            double ms = 0.0;
            launch_or_profile(ms, [&] {
                gemv_sub_kernel<<<grid, 128, 0, options_.stream>>>(d_.blocks.data(),
                                                                   index,
                                                                   row_begin,
                                                                   col_begin,
                                                                   rows,
                                                                   cols,
                                                                   gpu_.pad,
                                                                   d_.y.data(),
                                                                   d_.y.data());
                CUITER_CUDA_CHECK(cudaGetLastError());
            });
            stats.forward_ms += ms;
            stats.apply_offdiag_ms += ms;
        }
    }
}

inline void GpuBlockILU0::backward_solve(double* d_out, Stats& stats)
{
    for (int32_t block = gpu_.pattern.num_blocks - 1; block >= 0; --block) {
        const int32_t row_begin = gpu_.pattern.block_starts[static_cast<std::size_t>(block)];
        const int32_t rows = gpu_.pattern.block_dims[static_cast<std::size_t>(block)];

        for (int32_t ptr = gpu_.row_block_ptr[static_cast<std::size_t>(block)];
             ptr < gpu_.row_block_ptr[static_cast<std::size_t>(block + 1)];
             ++ptr) {
            const int32_t index = gpu_.row_block_indices_sorted[static_cast<std::size_t>(ptr)];
            const auto& offdiag = gpu_.pattern.blocks[static_cast<std::size_t>(index)];
            if (offdiag.col <= block) {
                continue;
            }

            const int32_t col_begin =
                gpu_.pattern.block_starts[static_cast<std::size_t>(offdiag.col)];
            const int32_t cols = gpu_.pattern.block_dims[static_cast<std::size_t>(offdiag.col)];
            const int grid = (rows + 127) / 128;
            double ms = 0.0;
            launch_or_profile(ms, [&] {
                gemv_sub_kernel<<<grid, 128, 0, options_.stream>>>(d_.blocks.data(),
                                                                   index,
                                                                   row_begin,
                                                                   col_begin,
                                                                   rows,
                                                                   cols,
                                                                   gpu_.pad,
                                                                   d_out,
                                                                   d_out);
                CUITER_CUDA_CHECK(cudaGetLastError());
            });
            stats.backward_ms += ms;
            stats.apply_offdiag_ms += ms;
        }

        const int32_t dim = gpu_.pattern.block_dims[static_cast<std::size_t>(block)];
        double ms = 0.0;
        launch_or_profile(ms, [&] {
            diag_apply_kernel<<<1, gpu_.pad, 0, options_.stream>>>(d_.inv_diag.data(),
                                                                   block,
                                                                   row_begin,
                                                                   dim,
                                                                   gpu_.pad,
                                                                   d_out);
            CUITER_CUDA_CHECK(cudaGetLastError());
        });
        stats.backward_ms += ms;
        stats.apply_diag_ms += ms;
    }
}

inline void GpuBlockILU0::maybe_copy_output_norm(const double* d_out, Stats& stats)
{
    if (!options_.compute_output_norm) {
        return;
    }

    std::vector<double> output(static_cast<std::size_t>(gpu_.a_perm.rows), 0.0);
    CUITER_CUDA_CHECK(cudaMemcpyAsync(output.data(),
                                      d_out,
                                      output.size() * sizeof(double),
                                      cudaMemcpyDeviceToHost,
                                      options_.stream));
    CUITER_CUDA_CHECK(cudaStreamSynchronize(options_.stream));
    stats.output_norm2 = detail::host_norm2(output);
}

}  // namespace gpu_block_ilu0
