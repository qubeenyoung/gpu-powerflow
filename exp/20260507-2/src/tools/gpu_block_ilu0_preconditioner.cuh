#pragma once

// File responsibility:
//   - Shared data types for the GPU block ILU(0) pilot
//   - GpuBlockILU0 class declaration
//   - Inclusion point for setup/factorize/apply implementations
//
// Implementation files:
//   - gpu_block_ilu0_setup.cuh
//   - gpu_block_ilu0_factorize.cuh
//   - gpu_block_ilu0_apply.cuh

#include "cuiter/common/cuda_utils.hpp"
#include "cuiter/core/csr_matrix.hpp"
#include "cuiter/solver/cpu_block_ilu0_pilot.hpp"
#include "gpu_block_ilu0_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_block_ilu0 {

// One CSV/report row for this standalone benchmark.
// It intentionally contains both structural work estimates and measured timings.
struct BenchRow {
    std::string case_name;
    int32_t block_size = 0;
    int32_t n = 0;
    int32_t nnz = 0;
    int32_t num_blocks = 0;
    int32_t block_nnz = 0;
    int32_t max_block_dim = 0;
    int32_t lower_edges = 0;
    int32_t upper_edges = 0;
    int32_t update_ops = 0;
    bool factor_failed = false;
    int32_t failed_block = -1;

    double setup_ms = 0.0;
    double dense_scatter_ms = 0.0;
    double factor_total_ms = 0.0;
    double factor_right_ms = 0.0;
    double factor_update_ms = 0.0;
    double factor_diag_inv_ms = 0.0;
    double apply_total_ms = 0.0;
    double forward_ms = 0.0;
    double backward_ms = 0.0;
    double apply_offdiag_ms = 0.0;
    double apply_diag_ms = 0.0;

    double diag_factor_work = 0.0;
    double offdiag_update_work = 0.0;
    double total_factor_work = 0.0;
    double factor_offdiag_work_share = 0.0;
    double diag_apply_work = 0.0;
    double offdiag_apply_work = 0.0;
    double total_apply_work = 0.0;
    double apply_offdiag_work_share = 0.0;
    double factor_work_relative_to_bj = 0.0;
    double apply_work_relative_to_bj = 0.0;

    double output_norm2 = 0.0;
};

// Symbolic data derived once from the CSR matrix and block ordering.
struct GpuPattern {
    using CpuBlockPattern = cuiter::cpu_pilot::detail::BlockPattern;

    CpuBlockPattern pattern;
    cuiter::CsrMatrix a_perm;
    std::vector<double> b_perm;
    int32_t pad = 0;

    std::vector<int32_t> block_rows;
    std::vector<int32_t> block_cols;
    std::vector<int32_t> row_block_ptr;
    std::vector<int32_t> row_block_indices_sorted;
    std::vector<int32_t> lower_blocks;
    std::vector<int32_t> upper_blocks;
    std::vector<int32_t> update_begin;
    std::vector<int32_t> update_count;
    std::vector<int32_t> update_target;
    std::vector<int32_t> update_rhs;
    std::vector<int32_t> nnz_block;
    std::vector<int32_t> nnz_local_row;
    std::vector<int32_t> nnz_local_col;

    BenchRow work;
};

struct DevicePatternBuffers {
    cuiter::DeviceBuffer<double> values;
    cuiter::DeviceBuffer<double> rhs;
    cuiter::DeviceBuffer<float> blocks;
    cuiter::DeviceBuffer<float> inv_diag;
    cuiter::DeviceBuffer<float> diag_work;
    cuiter::DeviceBuffer<double> y;
    cuiter::DeviceBuffer<double> z;
    cuiter::DeviceBuffer<int32_t> nnz_block;
    cuiter::DeviceBuffer<int32_t> nnz_local_row;
    cuiter::DeviceBuffer<int32_t> nnz_local_col;
    cuiter::DeviceBuffer<int32_t> update_target;
    cuiter::DeviceBuffer<int32_t> update_rhs;
    cuiter::DeviceBuffer<int32_t> pivots;
    cuiter::DeviceBuffer<int32_t> getrf_info;
    cuiter::DeviceBuffer<int32_t> getri_info;
    cuiter::DeviceBuffer<float*> diag_ptrs;
    cuiter::DeviceBuffer<float*> inv_ptrs;
};

class GpuBlockILU0 {
public:
    struct Options {
        int32_t block_size = 32;
        int32_t pad = 32;
        bool enable_profile = false;
        bool compute_output_norm = false;
        double default_shift_scale = 0.0;
        cudaStream_t stream = nullptr;
    };

    struct Stats {
        double setup_ms = 0.0;
        double dense_scatter_ms = 0.0;
        double factor_total_ms = 0.0;
        double factor_right_ms = 0.0;
        double factor_update_ms = 0.0;
        double factor_diag_inv_ms = 0.0;
        double apply_total_ms = 0.0;
        double forward_ms = 0.0;
        double backward_ms = 0.0;
        double apply_offdiag_ms = 0.0;
        double apply_diag_ms = 0.0;
        double output_norm2 = 0.0;
        bool factor_failed = false;
        int32_t failed_block = -1;
    };

    GpuBlockILU0() = default;

    ~GpuBlockILU0()
    {
        if (cublas_ != nullptr) {
            cublasDestroy(cublas_);
            cublas_ = nullptr;
        }
    }

    GpuBlockILU0(const GpuBlockILU0&) = delete;
    GpuBlockILU0& operator=(const GpuBlockILU0&) = delete;

    void setup(const cuiter::CsrMatrix& matrix,
               const std::vector<double>& rhs,
               const std::string& case_name,
               const Options& options);

    Stats factorize(double shift_scale);
    Stats factorize_device(const double* d_values, std::size_t count, double shift_scale);

    Stats apply();
    Stats apply_device(const double* d_rhs, double* d_out);

    const GpuPattern& pattern() const
    {
        return gpu_;
    }

private:
    template <typename Fn>
    void launch_or_profile(double& ms, Fn&& fn)
    {
        if (!options_.enable_profile) {
            fn();
            return;
        }

        cuiter::CudaEventTimer timer;
        timer.start(options_.stream);
        fn();
        ms += 1000.0 * timer.stop(options_.stream);
    }

    void reset_numeric_buffers();
    void scatter_current_values(int32_t nnz, Stats& stats);
    void eliminate_lower_blocks(int32_t row, Stats& stats);
    void invert_diagonal_block(int32_t block, float shift, Stats& stats);
    void copy_factor_error(Stats& stats);

    void forward_solve(Stats& stats);
    void backward_solve(double* d_out, Stats& stats);
    void maybe_copy_output_norm(const double* d_out, Stats& stats);

    Options options_;
    GpuPattern gpu_;
    DevicePatternBuffers d_;
    cublasHandle_t cublas_ = nullptr;
    double setup_ms_ = 0.0;
};

}  // namespace gpu_block_ilu0

#include "gpu_block_ilu0_setup.cuh"
#include "gpu_block_ilu0_factorize.cuh"
#include "gpu_block_ilu0_apply.cuh"
