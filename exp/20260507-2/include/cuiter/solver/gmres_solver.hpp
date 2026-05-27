#pragma once

#include "cuiter/common/cuda_utils.hpp"
#include "cuiter/core/csr_matrix.hpp"
#include "cuiter/preconditioner/metis_block_jacobi_preconditioner.hpp"

#include <cublas_v2.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace cuiter {

struct GmresSolverOptions {
    int32_t max_iters = 32;
    int32_t restart = 16;
    double rel_tolerance = 1.0e-3;
    double abs_tolerance = 0.0;
    std::string preconditioner = "metis_block_jacobi";
    int32_t block_size = 32;
    bool use_fp32_preconditioner = true;
    bool use_right_preconditioning = true;
    bool compute_true_residual = true;
    bool minimize_host_sync = true;
    BlockJacobiApplyMode block_jacobi_apply = BlockJacobiApplyMode::InverseGemv;
    double block_jacobi_diagonal_shift = 1.0e-8;
    bool use_mr1_fast_path = false;
    bool use_mr2_fast_path = false;
    bool use_bicgstab_fixed_path = false;
    bool use_bicgstab_fused_fixed2 = false;
    int32_t coarse_vars_per_block = 1;
    std::string coarse_refresh = "bootstrap_only";
    std::string coarse_precision = "fp32";
    double coarse_diag_shift_scale = 1.0e-6;
    std::string linear_scaling = "none";
    int32_t scaling_iters = 3;
    std::string scaling_norm = "l2";
    double scaling_clamp = 1.0e6;
    double scaling_eps = 1.0e-30;
    bool log_scaling_stats = false;
    bool use_initial_guess = false;
    std::string partition_mode = "unknown_metis";
    std::string bus_edge_weight = "jacobian_frobenius";
    double bus_edge_weight_scale = 1000.0;
    int32_t bus_edge_weight_clamp = 1000000;
    int32_t target_block_unknowns = 64;
    int32_t n_bus = 0;
    std::vector<int32_t> index_to_bus;
    std::vector<int32_t> index_field;
};

struct GmresTimings {
    double metis_partition_seconds = 0.0;
    double permutation_build_seconds = 0.0;
    double block_extract_seconds = 0.0;
    double block_lu_seconds = 0.0;
    double ras_symbolic_seconds = 0.0;
    double ras_setup_seconds = 0.0;
    double setup_total_seconds = 0.0;

    double rhs_permute_seconds = 0.0;
    double gmres_loop_seconds = 0.0;
    double preconditioner_apply_seconds = 0.0;
    double spmv_seconds = 0.0;
    double orthogonalization_seconds = 0.0;
    double dot_reduction_seconds = 0.0;
    double solution_update_seconds = 0.0;
    double final_residual_seconds = 0.0;
    double unpermute_seconds = 0.0;
    double solve_total_seconds = 0.0;

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
    double mr1_spmv_seconds = 0.0;
    double mr1_fused_dot_seconds = 0.0;
    double mr1_update_seconds = 0.0;
    double mr2_w1_spmv_seconds = 0.0;
    double bicgstab_total_seconds = 0.0;
    double bicgstab_spmv_seconds = 0.0;
    double bicgstab_dot_reduction_seconds = 0.0;
    double bicgstab_update_seconds = 0.0;
    double bicgstab_scalar_sync_seconds = 0.0;
    double middle_solver_total_seconds = 0.0;
    bool coarse_failed = false;

    double scaling_row_norm_seconds = 0.0;
    double scaling_col_norm_seconds = 0.0;
    double scaling_apply_values_seconds = 0.0;
    double scaling_apply_rhs_seconds = 0.0;
    double scaling_total_seconds = 0.0;
    double weighted_graph_build_seconds = 0.0;
};

struct LinearSolveResult {
    bool converged = false;
    int32_t iterations = 0;
    double residual_norm2 = 0.0;
    double relative_residual_norm2 = 0.0;
    double scaled_residual_norm2 = 0.0;
    double scaled_relative_residual_norm2 = 0.0;
    double unscaled_residual_norm2 = 0.0;
    double unscaled_relative_residual_norm2 = 0.0;
    std::string stop_reason;
    std::vector<double> residual_estimates;
    std::vector<double> solution;
    GmresTimings timings;
    BlockStructureStats block_stats;

    double dr_min = 1.0;
    double dr_max = 1.0;
    double dr_geomean = 1.0;
    double dc_min = 1.0;
    double dc_max = 1.0;
    double dc_geomean = 1.0;
    double row_norm_cv_before = 0.0;
    double row_norm_cv_after = 0.0;
    double col_norm_cv_before = 0.0;
    double col_norm_cv_after = 0.0;
};

class GmresSolver {
public:
    explicit GmresSolver(GmresSolverOptions options = {});
    ~GmresSolver();

    GmresSolver(const GmresSolver&) = delete;
    GmresSolver& operator=(const GmresSolver&) = delete;

    void set_options(const GmresSolverOptions& options);
    void analyze(const CsrMatrix& matrix);
    void setup(const double* d_values);
    void refresh_matrix_values(const double* d_values);

    LinearSolveResult solve(const std::vector<double>& values,
                            const std::vector<double>& rhs);
    LinearSolveResult solve_device(const double* d_values,
                                   const double* d_rhs,
                                   double* d_x);

    const GmresSolverOptions& options() const { return options_; }
    const MetisBlockJacobiPreconditioner& preconditioner() const { return preconditioner_; }

private:
    void ensure_workspace(int32_t n, int32_t restart);
    void ensure_scaling_workspace(int32_t n, int32_t nnz);
    DeviceCsrMatrixView active_matrix_view() const;
    DeviceCsrMatrixView unscaled_permuted_matrix_view() const;
    bool using_metis_block_jacobi() const;
    bool using_ruiz_scaling() const;
    bool using_field_scaling() const;
    bool using_linear_scaling() const;
    void compute_ruiz_scaling(const DeviceCsrMatrixView& raw_matrix);
    void compute_field_scaling(const DeviceCsrMatrixView& raw_matrix);
    void attach_scaling_result_metadata(LinearSolveResult& result) const;
    LinearSolveResult solve_mr1_device(const DeviceCsrMatrixView& matrix,
                                       const double* d_rhs,
                                       double* d_x,
                                       std::chrono::steady_clock::time_point solve_start);
    LinearSolveResult solve_mr2_device(const DeviceCsrMatrixView& matrix,
                                       const double* d_rhs,
                                       double* d_x,
                                       std::chrono::steady_clock::time_point solve_start);
    LinearSolveResult solve_bicgstab_device(const DeviceCsrMatrixView& matrix,
                                            const double* d_rhs,
                                            double* d_x,
                                            std::chrono::steady_clock::time_point solve_start);
    LinearSolveResult solve_bicgstab_fused_fixed2_device(
        const DeviceCsrMatrixView& matrix,
        const double* d_rhs,
        double* d_x,
        std::chrono::steady_clock::time_point solve_start);
    double norm_device(int32_t n, const double* d_x);
    double compute_residual(const DeviceCsrMatrixView& matrix,
                            int32_t n,
                            const double* d_rhs,
                            const double* d_x,
                            double rhs_norm,
                            LinearSolveResult& result);
    void update_solution(int32_t n, int32_t basis_count, double* d_x);
    bool solve_small_upper(int32_t basis_count);
    double apply_givens_and_residual(int32_t column, double rhs_norm);
    int32_t h_index(int32_t row, int32_t col) const;

    GmresSolverOptions options_;
    CsrMatrix matrix_;
    bool analyzed_ = false;
    bool setup_ready_ = false;

    DeviceBuffer<int32_t> d_row_ptr_;
    DeviceBuffer<int32_t> d_col_idx_;
    DeviceBuffer<double> d_original_values_;

    MetisBlockJacobiPreconditioner preconditioner_;

    cublasHandle_t cublas_ = nullptr;
    int32_t workspace_n_ = 0;
    int32_t workspace_restart_ = 0;
    DeviceBuffer<double> d_rhs_work_;
    DeviceBuffer<double> d_x_work_;
    DeviceBuffer<double> d_r_;
    DeviceBuffer<double> d_w_;
    DeviceBuffer<double> d_ax_;
    DeviceBuffer<double> d_v_basis_;
    DeviceBuffer<double> d_z_basis_;
    DeviceBuffer<double> d_h_col_;
    DeviceBuffer<double> d_y_;
    DeviceBuffer<double> d_mr1_dots_;
    DeviceBuffer<double> d_bicgstab_r_hat_;
    DeviceBuffer<double> d_bicgstab_p_;
    DeviceBuffer<double> d_bicgstab_p_hat_;
    DeviceBuffer<double> d_bicgstab_s_;
    DeviceBuffer<double> d_bicgstab_s_hat_;
    DeviceBuffer<double> d_bicgstab_v_;
    DeviceBuffer<double> d_bicgstab_t_;
    DeviceBuffer<double> d_rhs_unscaled_;
    DeviceBuffer<double> d_unscaled_permuted_values_;
    DeviceBuffer<double> d_scaled_values_;
    DeviceBuffer<double> d_row_scale_;
    DeviceBuffer<double> d_col_scale_;
    DeviceBuffer<double> d_row_norms_;
    DeviceBuffer<double> d_col_norms_;
    DeviceBuffer<double> d_initial_guess_permuted_;

    std::vector<double> h_col_host_;
    std::vector<double> hessenberg_;
    std::vector<double> givens_c_;
    std::vector<double> givens_s_;
    std::vector<double> g_;
    std::vector<double> y_host_;
    GmresTimings last_scaling_timings_;
    double dr_min_ = 1.0;
    double dr_max_ = 1.0;
    double dr_geomean_ = 1.0;
    double dc_min_ = 1.0;
    double dc_max_ = 1.0;
    double dc_geomean_ = 1.0;
    double row_norm_cv_before_ = 0.0;
    double row_norm_cv_after_ = 0.0;
    double col_norm_cv_before_ = 0.0;
    double col_norm_cv_after_ = 0.0;
};

}  // namespace cuiter
