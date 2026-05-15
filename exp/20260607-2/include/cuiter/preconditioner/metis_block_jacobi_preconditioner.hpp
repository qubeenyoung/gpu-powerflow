#pragma once

#include "cuiter/common/cuda_utils.hpp"
#include "cuiter/core/csr_matrix.hpp"
#include "cuiter/reorder/metis_partition.hpp"

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cstdint>
#include <string>
#include <vector>

namespace cuiter {

enum class BlockJacobiApplyMode {
    InverseGemv,
    LuSolve,
};

BlockJacobiApplyMode parse_block_jacobi_apply_mode(const std::string& value);
std::string to_string(BlockJacobiApplyMode mode);

struct MetisBlockJacobiOptions {
    int32_t block_size = 32;
    bool use_fp32_preconditioner = true;
    BlockJacobiApplyMode apply_mode = BlockJacobiApplyMode::InverseGemv;
    double diagonal_shift = 1.0e-8;
    bool enable_coarse = false;
    int32_t coarse_vars_per_block = 1;
    std::string coarse_refresh = "bootstrap_only";
    std::string coarse_precision = "fp32";
    double coarse_diag_shift_scale = 1.0e-6;
    std::string partition_mode = "unknown_metis";
    int32_t ras_overlap = 0;
    std::string bus_edge_weight = "jacobian_frobenius";
    double bus_edge_weight_scale = 1000.0;
    int32_t bus_edge_weight_clamp = 1000000;
    int32_t target_block_unknowns = 64;
    int32_t n_bus = 0;
    std::vector<int32_t> index_to_bus;
    std::vector<int32_t> index_field;
};

struct MetisBlockJacobiTimings {
    double metis_partition_seconds = 0.0;
    double permutation_build_seconds = 0.0;
    double weighted_graph_build_seconds = 0.0;
    double block_extract_seconds = 0.0;
    double block_lu_seconds = 0.0;
    double ras_symbolic_seconds = 0.0;
    double ras_setup_seconds = 0.0;
    double setup_total_seconds = 0.0;
};

struct MetisBlockJacobiApplyTimings {
    double block_jacobi_apply_seconds = 0.0;
    double coarse_az0_spmv_seconds = 0.0;
    double coarse_compress_seconds = 0.0;
    double coarse_solve_seconds = 0.0;
    double coarse_expand_seconds = 0.0;
    double coarse_total_seconds = 0.0;
    double ras_apply_seconds = 0.0;
    double ras_gather_seconds = 0.0;
    double ras_local_gemv_seconds = 0.0;
    double ras_scatter_seconds = 0.0;
    double preconditioner_total_seconds = 0.0;
    bool coarse_failed = false;
};

struct RasOverlapStats {
    int32_t overlap = 0;
    int32_t num_blocks = 0;
    int32_t min_owned_dim = 0;
    int32_t max_owned_dim = 0;
    double avg_owned_dim = 0.0;
    int32_t min_overlap_dim = 0;
    int32_t max_overlap_dim = 0;
    double avg_overlap_dim = 0.0;
    int32_t min_neighbor_count = 0;
    int32_t max_neighbor_count = 0;
    double avg_neighbor_count = 0.0;
    double overlap_dim_growth = 0.0;
    double estimated_dense_storage_mb = 0.0;
    double estimated_setup_work = 0.0;
    double estimated_apply_work = 0.0;
    int32_t local_nnz_total = 0;
    bool risk = false;
};

class MetisBlockJacobiPreconditioner {
public:
    MetisBlockJacobiPreconditioner();
    ~MetisBlockJacobiPreconditioner();

    MetisBlockJacobiPreconditioner(const MetisBlockJacobiPreconditioner&) = delete;
    MetisBlockJacobiPreconditioner& operator=(const MetisBlockJacobiPreconditioner&) = delete;

    void analyze(const CsrMatrix& matrix, const MetisBlockJacobiOptions& options);
    void setup(const double* d_values);
    void setup_permuted_values(const double* d_permuted_values);
    void refresh_permuted_values(const double* d_values);
    void apply(const double* d_rhs, double* d_out);
    void apply_local(const double* d_rhs, double* d_out);
    void apply_coarse_correction(const double* d_residual, double* d_out);

    DeviceCsrMatrixView permuted_matrix_view() const;

    int32_t rows() const { return rows_; }
    int32_t nnz() const { return nnz_; }
    bool analyzed() const { return analyzed_; }
    bool setup_ready() const { return setup_ready_; }
    const MetisPermutation& permutation() const { return permutation_; }
    const BlockStructureStats& block_stats() const { return permutation_.stats; }
    const MetisBlockJacobiTimings& timings() const { return timings_; }
    const MetisBlockJacobiApplyTimings& last_apply_timings() const { return last_apply_timings_; }
    const int32_t* d_new_to_old() const { return d_new_to_old_.data(); }
    bool coarse_enabled() const { return options_.enable_coarse; }
    bool coarse_ready() const { return coarse_ready_; }
    bool coarse_failed() const { return coarse_failed_; }
    int32_t coarse_dim() const { return coarse_dim_; }
    const RasOverlapStats& ras_stats() const { return ras_stats_; }

private:
    void build_dense_extract_maps();
    void build_ras_metadata();
    void build_coarse_metadata();
    void initialize_from_permutation();
    void build_bus_weighted_partition_if_needed(const double* d_values);
    void update_numeric_partition_stats(const std::vector<double>& host_values);
    void allocate_dense_buffers();
    void allocate_ras_buffers();
    void allocate_coarse_buffers();
    void rebuild_pointer_arrays();
    void rebuild_ras_pointer_arrays();
    void rebuild_coarse_pointer_arrays();
    void factorize_or_invert();
    void factorize_or_invert_ras();
    void setup_coarse_if_needed();
    void assemble_and_factorize_coarse();
    void apply_block_jacobi(const double* d_rhs, double* d_out);
    void apply_ras(const double* d_rhs, double* d_out);

    MetisBlockJacobiOptions options_;
    MetisPermutation permutation_;
    PermutedCsrPattern permuted_pattern_;
    MetisBlockJacobiTimings timings_;
    MetisBlockJacobiApplyTimings last_apply_timings_;

    int32_t rows_ = 0;
    int32_t nnz_ = 0;
    int32_t leading_dim_ = 0;
    int32_t ras_leading_dim_ = 0;
    int32_t ras_extract_nnz_ = 0;
    int32_t num_blocks_ = 0;
    int32_t coarse_dim_ = 0;
    bool analyzed_ = false;
    bool setup_ready_ = false;
    bool partition_ready_ = false;
    bool coarse_ready_ = false;
    bool coarse_failed_ = false;
    bool ras_enabled_ = false;
    CsrMatrix matrix_;
    RasOverlapStats ras_stats_;

    DeviceBuffer<int32_t> d_row_ptr_;
    DeviceBuffer<int32_t> d_col_idx_;
    DeviceBuffer<int32_t> d_value_source_index_;
    DeviceBuffer<int32_t> d_new_to_old_;
    DeviceBuffer<int32_t> d_block_starts_;
    DeviceBuffer<int32_t> d_block_sizes_;
    DeviceBuffer<int32_t> d_dense_block_offsets_;
    DeviceBuffer<int32_t> d_dense_local_rows_;
    DeviceBuffer<int32_t> d_dense_local_cols_;
    DeviceBuffer<int32_t> d_ras_local_offsets_;
    DeviceBuffer<int32_t> d_ras_local_to_global_;
    DeviceBuffer<int32_t> d_ras_owned_sizes_;
    DeviceBuffer<int32_t> d_ras_local_sizes_;
    DeviceBuffer<int32_t> d_ras_extract_source_pos_;
    DeviceBuffer<int32_t> d_ras_dense_block_offsets_;
    DeviceBuffer<int32_t> d_ras_dense_local_rows_;
    DeviceBuffer<int32_t> d_ras_dense_local_cols_;
    DeviceBuffer<int32_t> d_block_ids_;
    DeviceBuffer<int32_t> d_coarse_ids_;
    DeviceBuffer<float> d_coarse_weights_f32_;
    DeviceBuffer<double> d_coarse_weights_f64_;
    DeviceBuffer<double> d_permuted_values_;

    DeviceBuffer<float> d_dense_f32_;
    DeviceBuffer<float> d_inverse_f32_;
    DeviceBuffer<double> d_dense_f64_;
    DeviceBuffer<double> d_inverse_f64_;
    DeviceBuffer<float> d_ras_dense_f32_;
    DeviceBuffer<float> d_ras_inverse_f32_;
    DeviceBuffer<double> d_ras_dense_f64_;
    DeviceBuffer<double> d_ras_inverse_f64_;
    DeviceBuffer<int32_t> d_pivots_;
    DeviceBuffer<int32_t> d_info_;
    DeviceBuffer<int32_t> d_ras_pivots_;
    DeviceBuffer<int32_t> d_ras_info_;
    DeviceBuffer<float*> d_dense_ptrs_f32_;
    DeviceBuffer<const float*> d_dense_const_ptrs_f32_;
    DeviceBuffer<float*> d_inverse_ptrs_f32_;
    DeviceBuffer<double*> d_dense_ptrs_f64_;
    DeviceBuffer<const double*> d_dense_const_ptrs_f64_;
    DeviceBuffer<double*> d_inverse_ptrs_f64_;
    DeviceBuffer<float*> d_ras_dense_ptrs_f32_;
    DeviceBuffer<const float*> d_ras_dense_const_ptrs_f32_;
    DeviceBuffer<float*> d_ras_inverse_ptrs_f32_;
    DeviceBuffer<double*> d_ras_dense_ptrs_f64_;
    DeviceBuffer<const double*> d_ras_dense_const_ptrs_f64_;
    DeviceBuffer<double*> d_ras_inverse_ptrs_f64_;

    DeviceBuffer<double> d_coarse_az0_;
    DeviceBuffer<double> d_coarse_q_;
    DeviceBuffer<float> d_coarse_lu_f32_;
    DeviceBuffer<float> d_coarse_rhs_f32_;
    DeviceBuffer<double> d_coarse_lu_f64_;
    DeviceBuffer<double> d_coarse_rhs_f64_;
    DeviceBuffer<int32_t> d_coarse_pivots_;
    DeviceBuffer<int32_t> d_coarse_info_;
    DeviceBuffer<float> d_coarse_work_f32_;
    DeviceBuffer<double> d_coarse_work_f64_;
    DeviceBuffer<float*> d_coarse_lu_ptrs_f32_;
    DeviceBuffer<const float*> d_coarse_lu_const_ptrs_f32_;
    DeviceBuffer<float*> d_coarse_rhs_ptrs_f32_;
    DeviceBuffer<double*> d_coarse_lu_ptrs_f64_;
    DeviceBuffer<const double*> d_coarse_lu_const_ptrs_f64_;
    DeviceBuffer<double*> d_coarse_rhs_ptrs_f64_;

    cublasHandle_t cublas_ = nullptr;
    cusolverDnHandle_t cusolver_ = nullptr;
    int32_t coarse_lwork_ = 0;
};

}  // namespace cuiter
