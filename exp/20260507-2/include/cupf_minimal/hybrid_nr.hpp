#pragma once

// experimental minimal cuPF NR port

#include "cupf_minimal/case_data.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cupf_minimal {

struct HybridNrOptions {
    int32_t max_nr_iters = 20;
    double nr_mismatch_inf_tol = 1.0e-8;
    double nr_mismatch_2_tol = 0.0;

    int32_t cudss_bootstrap_iters = 1;
    double cudss_polish_threshold = 1.0e-4;
    double iterative_start_mismatch_threshold = -1.0;
    int32_t force_gmres_min_steps = 0;
    int32_t max_consecutive_iterative_failures = 2;

    std::string solver = "hybrid";       // "pure_cudss", "hybrid"
    std::string middle_solver = "gmres_block_jacobi";
    int32_t gmres_restart = 16;
    int32_t gmres_max_iters = 8;
    int32_t bicgstab_iters = 1;
    int32_t ginkgo_parilut_iters = 5;
    double ginkgo_parilut_fill = 2.0;
    double gmres_rtol = 2.0e-1;
    double gmres_atol = 0.0;
    bool gmres_fixed_iter_mode = true;
    bool bicgstab_fused_fixed2 = false;
    bool a1_event_dag = false;
    bool full_cudss_analyze_before_loop = false;
    std::string fdlf_p_rhs = "auto";
    std::string fdlf_q_rhs = "auto";

    std::string preconditioner = "metis_block_jacobi";
    std::string bj_setup = "every_middle";
    int32_t block_size = 64;
    std::string block_precision = "fp32";
    std::string block_apply = "inverse_gemv";
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
    bool previous_dx_warm_start = false;
    std::string partition_mode = "unknown_metis";
    std::string bus_edge_weight = "jacobian_frobenius";
    double bus_edge_weight_scale = 1000.0;
    int32_t bus_edge_weight_clamp = 1000000;
    int32_t target_block_unknowns = 64;

    bool enable_cudss_fallback = true;
    std::string fallback_policy = "immediate";  // "off", "immediate", "after_two_failures"
    bool accept_iterative_by_mismatch = true;
    double accept_mismatch_ratio = 0.95;
    double reject_mismatch_ratio = 1.05;
    int32_t max_middle_accepts = -1;
    int32_t max_a1_middle_accepts = -1;
    bool enable_damped_iterative_step = false;
    std::vector<double> damping_factors = {1.0, 0.5, 0.25};
    bool enable_shadow_dx_diagnostic = false;
    bool skip_middle_backup = false;
    std::string dx_safety_check = "full"; // "full", "nonfinite", "off"
    bool enable_scaled_mr1_step = false;
    std::vector<double> scaled_mr1_gamma_candidates = {4.0, 2.0, 1.0};
    std::string global_correction = "none";       // "none", "post"
    std::string global_basis_source = "fallback"; // "fallback", "diagnostic"
    int32_t global_rank = 0;
    bool global_diagnostic_full_step = false;
    std::string global_correction_acceptance = "residual"; // "always", "residual"
    bool global_reset_on_event = false;
    double global_orth_tol = 1.0e-6;
    std::string field_gain_correction = "none"; // "none", "ls2"
    double field_gain_theta_max = 8.0;
    double field_gain_vmax = 4.0;
    bool field_gain_nonnegative = true;
    double field_gain_trust_ratio = 4.0;
    std::string theta_j11_correction = "none"; // "none", "scalar", "gmres"
    int32_t theta_j11_gmres_maxit = 5;
    double theta_j11_correction_trust_ratio = 1.0;

    bool dump_each_iteration_jf = false;
    std::string iteration_f_dump_dir;
    bool log_linear_residual = true;
    bool log_mismatch_2 = true;
    bool log_mismatch_inf = true;
};

struct HybridNrIterationLog {
    int32_t case_id = 0;
    std::string case_name;
    int32_t nr_iter = 0;
    std::string solver_used;
    int32_t linear_iters = 0;

    double mismatch_inf_before = 0.0;
    double mismatch_inf_after = 0.0;
    double mismatch_2_before = 0.0;
    double mismatch_2_after = 0.0;

    double linear_abs_res = 0.0;
    double linear_rel_res = 0.0;
    int32_t factor_age = 0;
    int32_t stale_solve_calls = 0;
    int32_t current_j_spmv_calls = 0;
    int32_t bicgstab_refinement_iters = 0;
    int32_t gmres_refinement_iters = 0;
    int32_t stale_prec_solve_count = 0;
    double stale_solve_f_seconds = 0.0;
    double current_j_spmv_r0_seconds = 0.0;
    double stale_solve_r0_seconds = 0.0;
    double current_j_spmv_r1_seconds = 0.0;
    double stale_solve_r1_seconds = 0.0;
    double stale_prec_solve_seconds = 0.0;
    double linear_residual_after_r0 = 0.0;
    double linear_residual_after_r1_richardson = 0.0;
    double linear_residual_after_r2_richardson = 0.0;
    double linear_residual_after_bicgstab = 0.0;
    double linear_residual_after_gmres = 0.0;

    double jacobian_seconds = 0.0;
    double rhs_seconds = 0.0;
    double linear_setup_seconds = 0.0;
    double linear_solve_seconds = 0.0;
    double gmres_trial_setup_seconds = 0.0;
    double gmres_trial_solve_seconds = 0.0;
    double fallback_cudss_setup_seconds = 0.0;
    double fallback_cudss_solve_seconds = 0.0;
    double block_jacobi_apply_seconds = 0.0;
    double ras_setup_seconds = 0.0;
    double ras_apply_seconds = 0.0;
    double ras_gather_seconds = 0.0;
    double ras_local_gemv_seconds = 0.0;
    double ras_scatter_seconds = 0.0;
    double coarse_az0_spmv_seconds = 0.0;
    double coarse_compress_seconds = 0.0;
    double coarse_solve_seconds = 0.0;
    double coarse_expand_seconds = 0.0;
    double coarse_total_seconds = 0.0;
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
    double gmres_total_seconds = 0.0;
    double gmres_spmv_seconds = 0.0;
    double gmres_dot_seconds = 0.0;
    double gmres_orthogonalization_seconds = 0.0;
    double gmres_update_seconds = 0.0;
    double gmres_scalar_sync_seconds = 0.0;
    double host_sync_seconds = 0.0;
    double unaccounted_seconds = 0.0;
    double bj_metadata_setup_seconds = 0.0;
    double bj_value_update_seconds = 0.0;
    double bj_inverse_build_seconds = 0.0;
    double bj_setup_total_seconds = 0.0;
    bool bj_cache_reused = false;
    double middle_solver_total_seconds = 0.0;
    double block_ilu_factor_seconds = 0.0;
    double block_ilu_apply_seconds = 0.0;
    double block_ilu_forward_seconds = 0.0;
    double block_ilu_backward_seconds = 0.0;
    int32_t block_ilu_l_levels = 0;
    int32_t block_ilu_u_levels = 0;
    double block_ilu_avg_level_width = 0.0;
    int32_t block_ilu_max_level_width = 0;
    int32_t block_ilu_block_nnz = 0;
    bool block_ilu_failed = false;
    double j11_factor_seconds = 0.0;
    double j11_solve_seconds = 0.0;
    double j22_factor_seconds = 0.0;
    double j22_solve_seconds = 0.0;
    double fdlf_bp_solve_seconds = 0.0;
    double fdlf_bpp_solve_seconds = 0.0;
    double fdlf_cross_spmv_seconds = 0.0;
    double fdlf_round0_wall_seconds = 0.0;
    double fdlf_round1_wall_seconds = 0.0;
    double fdlf_2round_wall_seconds = 0.0;
    double field_correction_wall_seconds = 0.0;
    double field_correction_serial_sum_seconds = 0.0;
    double full_residual_spmv_seconds = 0.0;
    double residual_axpy_seconds = 0.0;
    double rhs_gather_seconds = 0.0;
    double j11_value_update_seconds = 0.0;
    double j22_value_update_seconds = 0.0;
    double j12_value_update_seconds = 0.0;
    double j21_value_update_seconds = 0.0;
    double j12_spmv_seconds = 0.0;
    double j21_spmv_seconds = 0.0;
    double j11_solve_round1_seconds = 0.0;
    double j22_solve_round1_seconds = 0.0;
    double dx_accum_seconds = 0.0;
    double non_cudss_overhead_seconds = 0.0;
    double a1_event_wait_seconds = 0.0;
    double a1_unaccounted_seconds = 0.0;
    bool coarse_failed = false;
    double chosen_gamma = 1.0;
    double mismatch_ratio_gamma_4 = 0.0;
    double mismatch_ratio_gamma_2 = 0.0;
    double mismatch_ratio_gamma_1 = 0.0;
    double extra_mismatch_eval_seconds = 0.0;
    double scaled_total_middle_seconds = 0.0;
    double scaling_row_norm_seconds = 0.0;
    double scaling_col_norm_seconds = 0.0;
    double scaling_apply_values_seconds = 0.0;
    double scaling_apply_rhs_seconds = 0.0;
    double scaling_total_seconds = 0.0;
    double scaled_linear_abs_res = 0.0;
    double scaled_linear_rel_res = 0.0;
    double unscaled_linear_abs_res = 0.0;
    double unscaled_linear_rel_res = 0.0;
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
    std::string partition_mode = "unknown_metis";
    double weighted_graph_build_seconds = 0.0;
    double metis_partition_seconds = 0.0;
    double permutation_build_seconds = 0.0;
    double block_extract_seconds = 0.0;
    double block_inverse_seconds = 0.0;
    int32_t num_bus_partitions = 0;
    int32_t min_block_unknowns = 0;
    int32_t max_block_unknowns = 0;
    double avg_block_unknowns = 0.0;
    double std_block_unknowns = 0.0;
    double diagonal_block_nnz_ratio = 0.0;
    double offblock_nnz_ratio = 0.0;
    double total_weighted_coupling = 0.0;
    double diagonal_weighted_coupling_ratio = 0.0;
    double offblock_weighted_coupling_ratio = 0.0;
    double j11_diagonal_weighted_ratio = 0.0;
    double j12_diagonal_weighted_ratio = 0.0;
    double j21_diagonal_weighted_ratio = 0.0;
    double j22_diagonal_weighted_ratio = 0.0;
    int32_t theta_vmag_split_count = 0;
    int32_t pq_split_count = 0;
    double voltage_update_seconds = 0.0;
    double mismatch_recompute_seconds = 0.0;
    double nr_iter_total_seconds = 0.0;

    bool shadow_dx_diagnostic = false;
    double shadow_dx_diagnostic_seconds = 0.0;
    double shadow_mismatch_eval_seconds = 0.0;
    double shadow_gmres_setup_seconds = 0.0;
    double shadow_gmres_solve_seconds = 0.0;
    double shadow_cudss_analyze_seconds = 0.0;
    double shadow_cudss_factorize_seconds = 0.0;
    double shadow_cudss_solve_seconds = 0.0;
    double shadow_linear_rel_res_gmres = 0.0;
    double shadow_linear_abs_res_gmres = 0.0;
    double shadow_dx_gmres_norm2 = 0.0;
    double shadow_mismatch_after_gmres_inf = 0.0;
    double shadow_mismatch_after_gmres_2 = 0.0;
    double shadow_mismatch_after_gmres_p_inf = 0.0;
    double shadow_mismatch_after_gmres_p_2 = 0.0;
    double shadow_mismatch_after_gmres_q_inf = 0.0;
    double shadow_mismatch_after_gmres_q_2 = 0.0;
    double shadow_linear_rel_res_cudss = 0.0;
    double shadow_linear_abs_res_cudss = 0.0;
    double shadow_dx_cudss_norm2 = 0.0;
    double shadow_mismatch_after_cudss_inf = 0.0;
    double shadow_mismatch_after_cudss_2 = 0.0;
    double shadow_mismatch_after_cudss_p_inf = 0.0;
    double shadow_mismatch_after_cudss_p_2 = 0.0;
    double shadow_mismatch_after_cudss_q_inf = 0.0;
    double shadow_mismatch_after_cudss_q_2 = 0.0;
    double shadow_dx_norm_ratio = 0.0;
    double shadow_dx_cosine = 0.0;
    double shadow_dx_projection = 0.0;
    double shadow_dx_orth_error = 0.0;
    double shadow_theta_norm_ratio = 0.0;
    double shadow_theta_cosine = 0.0;
    double shadow_vmag_norm_ratio = 0.0;
    double shadow_vmag_cosine = 0.0;
    double shadow_max_abs_dx_gmres = 0.0;
    double shadow_max_abs_dx_cudss = 0.0;
    double shadow_max_abs_dx_diff = 0.0;

    bool step_accepted = false;
    bool fallback_used = false;
    std::string stop_reason;

    bool global_correction_attempted = false;
    bool global_correction_used = false;
    std::string global_correction_skipped_reason;
    int32_t global_basis_rank_before = 0;
    int32_t global_basis_rank_after = 0;
    double global_linear_res_before = 0.0;
    double global_linear_res_after = 0.0;
    double global_correction_gain = 0.0;
    double global_correction_norm_ratio = 0.0;
    double global_correction_seconds = 0.0;
    double global_az_seconds = 0.0;
    double global_dense_ls_seconds = 0.0;
    bool global_required_basis_added = false;
    bool global_diagnostic_basis_added = false;
    bool used_diagnostic_full_cudss = false;
    double cos_theta_gmres = 0.0;
    double cos_theta_corr = 0.0;
    double norm_ratio_gmres = 0.0;
    double norm_ratio_corr = 0.0;

    bool field_gain_attempted = false;
    bool field_gain_accepted = false;
    std::string field_gain_skipped_reason;
    double gamma_theta = 1.0;
    double gamma_v = 1.0;
    double lin_res_before_gain = 0.0;
    double lin_res_after_gain = 0.0;
    double nonlinear_res_iter_trial = 0.0;
    double nonlinear_res_gain_trial = 0.0;
    double gain_step_norm_ratio = 0.0;
    double field_gain_seconds = 0.0;

    bool theta_corr_attempted = false;
    bool theta_corr_accepted = false;
    std::string theta_corr_skipped_reason;
    double theta_p_scalar_beta = 0.0;
    double j11_corr_norm = 0.0;
    double j11_corr_res_before = 0.0;
    double j11_corr_res_after = 0.0;
    double p_res_before = 0.0;
    double p_res_after = 0.0;
    double q_res_before = 0.0;
    double q_res_after = 0.0;
    double nonlinear_res_after_theta_corr = 0.0;
    double theta_corr_seconds = 0.0;

    double alpha_theta_oracle = 0.0;
    double alpha_v_oracle = 0.0;
    double norm_m11 = 0.0;
    double norm_m12 = 0.0;
    double norm_m21 = 0.0;
    double norm_m22 = 0.0;
    double norm_p_missing = 0.0;
    double norm_q_missing = 0.0;
    double norm_ad = 0.0;
    double frac_m11 = 0.0;
    double frac_m12 = 0.0;
    double frac_m21 = 0.0;
    double frac_m22 = 0.0;
};

struct HybridNrResult {
    std::string case_name;
    int32_t buses = 0;
    int32_t n = 0;
    int32_t nnz = 0;
    bool converged = false;
    int32_t nr_iters = 0;
    int32_t cudss_calls = 0;
    int32_t gmres_calls = 0;
    int32_t accepted_gmres_steps = 0;
    int32_t rejected_gmres_steps = 0;
    int32_t fallback_calls = 0;
    int32_t polish_calls = 0;
    int32_t pure_full_cudss_calls = 0;
    int32_t diagnostic_full_cudss_calls = 0;
    int32_t corrections_attempted = 0;
    int32_t corrections_accepted = 0;
    int32_t corrections_skipped = 0;
    int32_t final_basis_rank = 0;
    double global_correction_seconds = 0.0;
    double global_az_seconds = 0.0;
    double global_dense_ls_seconds = 0.0;
    double total_seconds = 0.0;
    double middle_solver_analyze_setup_seconds = 0.0;
    double field_correction_analyze_setup_seconds = 0.0;
    double full_cudss_analyze_setup_seconds = 0.0;
    double fdlf_bp_analyze_seconds = 0.0;
    double fdlf_bp_factor_seconds = 0.0;
    double fdlf_bpp_analyze_seconds = 0.0;
    double fdlf_bpp_factor_seconds = 0.0;
    std::string fdlf_p_rhs;
    std::string fdlf_q_rhs;
    double pure_cudss_total_seconds = 0.0;
    double speedup_vs_pure_cudss = 0.0;
    double final_mismatch_inf = 0.0;
    double final_mismatch_2 = 0.0;
    double max_linear_rel_res_accepted = 0.0;
    double avg_linear_rel_res_accepted = 0.0;
    double accepted_gmres_mismatch_reduction_ratio_mean = 0.0;
    double shadow_dx_diagnostic_seconds = 0.0;
    int32_t gmres_block_size = 0;
    int32_t gmres_restart = 0;
    int32_t gmres_max_iters = 0;
    int32_t bicgstab_iters = 0;
    double gmres_rtol = 0.0;
    bool gmres_fixed_iter_mode = false;
    std::string stop_reason;
    std::vector<HybridNrIterationLog> iteration_logs;
};

HybridNrResult run_hybrid_nr_case(const DumpCaseData& data,
                                  const HybridNrOptions& options,
                                  double pure_cudss_total_seconds = 0.0,
                                  int32_t pure_full_cudss_calls = 0,
                                  int32_t case_id = 0);

}  // namespace cupf_minimal
