#include "cupf_minimal/case_data.hpp"
#include "cupf_minimal/hybrid_nr.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path case_root = "/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps";
    std::vector<std::string> cases;
    std::filesystem::path output = "results/hybrid_nr_gmres_block_jacobi.csv";
    std::filesystem::path iter_output = "results/hybrid_nr_gmres_block_jacobi_iters.csv";
    std::filesystem::path shadow_output;
    std::filesystem::path timing_output;
    std::filesystem::path partition_stats_output;
    bool shadow_output_explicit = false;
    bool timing_output_explicit = false;
    bool partition_stats_output_explicit = false;
    bool run_pure_cudss_baseline = true;
    int32_t warmup = 0;
    cupf_minimal::HybridNrOptions nr;
};

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " --case case_ACTIVSg10k [options]\n\n"
        << "Options:\n"
        << "  --case-root PATH\n"
        << "  --case NAME[,NAME...]\n"
        << "  --output PATH\n"
        << "  --iter-output PATH\n"
        << "  --shadow-output PATH\n"
        << "  --warmup INT\n"
        << "  --solver pure_cudss|hybrid\n"
              << "  --middle-solver gmres_block_jacobi|mr1_block_jacobi|mr1_block_jacobi_coarse|mr2_block_jacobi_coarse|bicgstab_block_jacobi|bicgstab_block_jacobi_a0|bicgstab_block_jacobi_a1|bicgstab_block_jacobi_a0_device|bicgstab_block_jacobi_a1_device|bicgstab_block_jacobi_j11_device|bicgstab_block_jacobi_bpbpp_refine|bicgstab_block_ilu0|gmres_block_ilu0|ginkgo_parilut_bicgstab|fdlf_bpbpp_2round|stale_R0|stale_R1_Richardson|stale_R2_Richardson|stale_R1_BiCGSTAB1_no_prec|stale_R1_BiCGSTAB2_no_prec|stale_R1_BiCGSTAB1_stale_prec|stale_R1_BiCGSTAB2_stale_prec|stale_R1_GMRES1_stale_prec|stale_R1_GMRES2_stale_prec|stale_GMRES1|stale_BJ1|stale_GMRES1_refresh\n"
        << "  --max-nr-iters INT\n"
        << "  --nr-mismatch-inf-tol FLOAT\n"
        << "  --cudss-bootstrap-iters INT\n"
        << "  --cudss-polish-threshold FLOAT|disabled\n"
        << "  --iterative-start-mismatch-threshold FLOAT|disabled\n"
        << "  --force-gmres-min-steps INT\n"
        << "  --gmres-restart INT\n"
        << "  --gmres-max-iters INT\n"
        << "  --bicgstab-iters INT\n"
        << "  --ginkgo-parilut-iters INT\n"
        << "  --ginkgo-parilut-fill FLOAT\n"
              << "  --bicgstab-fused-fixed2 true|false\n"
              << "  --full-cudss-analyze-before-loop true|false\n"
              << "  --fdlf-p-rhs r|-r|r_over_v|-r_over_v|auto\n"
              << "  --fdlf-q-rhs r|-r|r_over_v|-r_over_v|auto\n"
        << "  --gmres-rtol FLOAT\n"
        << "  --gmres-fixed-iter-mode true|false\n"
        << "  --preconditioner metis_block_jacobi|metis_block_jacobi_coarse|ras_overlap1|block_ilu0\n"
        << "  --bj-setup every_middle|value_update_only|numeric_reuse_after_full_cudss|reuse_after_full_cudss|reuse_for_2_middle_steps\n"
        << "  --block-size 4|8|16|32|64\n"
        << "  --block-jacobi-precision fp32|fp64\n"
        << "  --block-jacobi-apply inverse_gemv|lu_solve\n"
        << "  --coarse-vars-per-block 1|2\n"
        << "  --coarse-refresh bootstrap_only|after_cudss_fallback|every_iter\n"
        << "  --coarse-precision fp32|fp64\n"
        << "  --coarse-diag-shift-scale FLOAT\n"
        << "  --partition-mode unknown_metis|bus_weighted_metis\n"
        << "  --bus-edge-weight jacobian_frobenius\n"
        << "  --bus-edge-weight-scale FLOAT\n"
        << "  --bus-edge-weight-clamp INT\n"
        << "  --target-block-unknowns INT\n"
        << "  --linear-scaling none|ruiz|field\n"
        << "  --scaling-iters INT\n"
        << "  --scaling-norm l2\n"
        << "  --scaling-clamp FLOAT\n"
        << "  --scaling-eps FLOAT\n"
        << "  --log-scaling-stats true|false\n"
        << "  --previous-dx-warm-start true|false\n"
        << "  --enable-cudss-fallback true|false\n"
        << "  --fallback-policy off|immediate|after_two_failures\n"
        << "  --accept-iterative-by-mismatch true|false\n"
        << "  --accept-mismatch-ratio FLOAT\n"
        << "  --reject-mismatch-ratio FLOAT\n"
        << "  --max-middle-accepts INT\n"
        << "  --max-a1-middle-accepts INT\n"
        << "  --enable-damped-iterative-step true|false\n"
        << "  --damping-factors 1.0,0.5,0.25\n"
        << "  --enable-scaled-mr1-step true|false\n"
        << "  --scaled-mr1-gammas 4.0,2.0,1.0\n"
        << "  --shadow-dx-diagnostic true|false\n"
        << "  --skip-middle-backup true|false\n"
        << "  --dx-safety-check full|nonfinite|off\n"
        << "  --global-correction none|post\n"
        << "  --global-basis-source fallback|diagnostic\n"
        << "  --global-rank INT\n"
        << "  --global-diagnostic-full-step true|false\n"
        << "  --global-correction-acceptance always|residual\n"
        << "  --global-reset-on-event true|false\n"
        << "  --field-gain-correction none|ls2\n"
        << "  --field-gain-theta-max FLOAT\n"
        << "  --field-gain-vmax FLOAT\n"
        << "  --field-gain-nonnegative true|false\n"
        << "  --field-gain-trust-ratio FLOAT\n"
        << "  --theta-j11-correction none|scalar|gmres\n"
        << "  --theta-j11-gmres-maxit INT\n"
        << "  --theta-j11-correction-trust-ratio FLOAT\n"
        << "  --dump-iteration-f-dir PATH\n"
        << "  --timing-output PATH\n"
        << "  --partition-stats-output PATH\n"
        << "  --no-pure-cudss-baseline\n";
}

std::vector<std::string> split_list(const std::string& value)
{
    std::vector<std::string> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) {
            return std::isspace(ch);
        }), item.end());
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

bool parse_bool(const std::string& value)
{
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (lowered == "true" || lowered == "1" || lowered == "yes" || lowered == "on") {
        return true;
    }
    if (lowered == "false" || lowered == "0" || lowered == "no" || lowered == "off") {
        return false;
    }
    throw std::runtime_error("expected boolean value, got: " + value);
}

double parse_threshold(const std::string& value)
{
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (lowered == "disabled" || lowered == "off" || lowered == "none") {
        return -1.0;
    }
    return std::stod(value);
}

std::vector<double> parse_double_list(const std::string& value)
{
    std::vector<double> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::stod(item));
        }
    }
    if (out.empty()) {
        throw std::runtime_error("expected at least one numeric list item");
    }
    return out;
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-root" && i + 1 < argc) {
            options.case_root = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            const auto parsed = split_list(argv[++i]);
            options.cases.insert(options.cases.end(), parsed.begin(), parsed.end());
        } else if (arg == "--output" && i + 1 < argc) {
            options.output = argv[++i];
        } else if (arg == "--iter-output" && i + 1 < argc) {
            options.iter_output = argv[++i];
        } else if (arg == "--shadow-output" && i + 1 < argc) {
            options.shadow_output = argv[++i];
            options.shadow_output_explicit = true;
        } else if (arg == "--timing-output" && i + 1 < argc) {
            options.timing_output = argv[++i];
            options.timing_output_explicit = true;
        } else if (arg == "--partition-stats-output" && i + 1 < argc) {
            options.partition_stats_output = argv[++i];
            options.partition_stats_output_explicit = true;
        } else if (arg == "--warmup" && i + 1 < argc) {
            options.warmup = std::stoi(argv[++i]);
        } else if (arg == "--solver" && i + 1 < argc) {
            options.nr.solver = argv[++i];
        } else if (arg == "--max-nr-iters" && i + 1 < argc) {
            options.nr.max_nr_iters = std::stoi(argv[++i]);
        } else if (arg == "--nr-mismatch-inf-tol" && i + 1 < argc) {
            options.nr.nr_mismatch_inf_tol = std::stod(argv[++i]);
        } else if (arg == "--nr-mismatch-2-tol" && i + 1 < argc) {
            options.nr.nr_mismatch_2_tol = std::stod(argv[++i]);
        } else if (arg == "--cudss-bootstrap-iters" && i + 1 < argc) {
            options.nr.cudss_bootstrap_iters = std::stoi(argv[++i]);
        } else if (arg == "--cudss-polish-threshold" && i + 1 < argc) {
            options.nr.cudss_polish_threshold = parse_threshold(argv[++i]);
        } else if (arg == "--iterative-start-mismatch-threshold" && i + 1 < argc) {
            options.nr.iterative_start_mismatch_threshold = parse_threshold(argv[++i]);
        } else if (arg == "--force-gmres-min-steps" && i + 1 < argc) {
            options.nr.force_gmres_min_steps = std::stoi(argv[++i]);
        } else if (arg == "--max-consecutive-iterative-failures" && i + 1 < argc) {
            options.nr.max_consecutive_iterative_failures = std::stoi(argv[++i]);
        } else if (arg == "--middle-solver" && i + 1 < argc) {
            options.nr.middle_solver = argv[++i];
        } else if (arg == "--gmres-restart" && i + 1 < argc) {
            options.nr.gmres_restart = std::stoi(argv[++i]);
        } else if (arg == "--gmres-max-iters" && i + 1 < argc) {
            options.nr.gmres_max_iters = std::stoi(argv[++i]);
        } else if (arg == "--bicgstab-iters" && i + 1 < argc) {
            options.nr.bicgstab_iters = std::stoi(argv[++i]);
        } else if (arg == "--ginkgo-parilut-iters" && i + 1 < argc) {
            options.nr.ginkgo_parilut_iters = std::stoi(argv[++i]);
        } else if (arg == "--ginkgo-parilut-fill" && i + 1 < argc) {
            options.nr.ginkgo_parilut_fill = std::stod(argv[++i]);
        } else if ((arg == "--bicgstab-fused-fixed2" ||
                    arg == "--bicgstab-fixed2-fused") && i + 1 < argc) {
            options.nr.bicgstab_fused_fixed2 = parse_bool(argv[++i]);
        } else if (arg == "--a1-event-dag" && i + 1 < argc) {
            options.nr.a1_event_dag = parse_bool(argv[++i]);
        } else if (arg == "--full-cudss-analyze-before-loop" && i + 1 < argc) {
            options.nr.full_cudss_analyze_before_loop = parse_bool(argv[++i]);
        } else if (arg == "--fdlf-p-rhs" && i + 1 < argc) {
            options.nr.fdlf_p_rhs = argv[++i];
        } else if (arg == "--fdlf-q-rhs" && i + 1 < argc) {
            options.nr.fdlf_q_rhs = argv[++i];
        } else if (arg == "--gmres-rtol" && i + 1 < argc) {
            options.nr.gmres_rtol = std::stod(argv[++i]);
        } else if (arg == "--gmres-atol" && i + 1 < argc) {
            options.nr.gmres_atol = std::stod(argv[++i]);
        } else if (arg == "--gmres-fixed-iter-mode" && i + 1 < argc) {
            options.nr.gmres_fixed_iter_mode = parse_bool(argv[++i]);
        } else if (arg == "--preconditioner" && i + 1 < argc) {
            options.nr.preconditioner = argv[++i];
        } else if ((arg == "--bj-setup" || arg == "--bj-setup-mode") && i + 1 < argc) {
            options.nr.bj_setup = argv[++i];
        } else if (arg == "--block-size" && i + 1 < argc) {
            options.nr.block_size = std::stoi(argv[++i]);
        } else if (arg == "--block-jacobi-precision" && i + 1 < argc) {
            options.nr.block_precision = argv[++i];
        } else if (arg == "--block-jacobi-apply" && i + 1 < argc) {
            options.nr.block_apply = argv[++i];
        } else if (arg == "--coarse-vars-per-block" && i + 1 < argc) {
            options.nr.coarse_vars_per_block = std::stoi(argv[++i]);
        } else if (arg == "--coarse-refresh" && i + 1 < argc) {
            options.nr.coarse_refresh = argv[++i];
        } else if (arg == "--coarse-precision" && i + 1 < argc) {
            options.nr.coarse_precision = argv[++i];
        } else if (arg == "--coarse-diag-shift-scale" && i + 1 < argc) {
            options.nr.coarse_diag_shift_scale = std::stod(argv[++i]);
        } else if (arg == "--partition-mode" && i + 1 < argc) {
            options.nr.partition_mode = argv[++i];
        } else if (arg == "--bus-edge-weight" && i + 1 < argc) {
            options.nr.bus_edge_weight = argv[++i];
        } else if (arg == "--bus-edge-weight-scale" && i + 1 < argc) {
            options.nr.bus_edge_weight_scale = std::stod(argv[++i]);
        } else if (arg == "--bus-edge-weight-clamp" && i + 1 < argc) {
            options.nr.bus_edge_weight_clamp = std::stoi(argv[++i]);
        } else if (arg == "--target-block-unknowns" && i + 1 < argc) {
            options.nr.target_block_unknowns = std::stoi(argv[++i]);
        } else if (arg == "--linear-scaling" && i + 1 < argc) {
            options.nr.linear_scaling = argv[++i];
        } else if (arg == "--scaling-iters" && i + 1 < argc) {
            options.nr.scaling_iters = std::stoi(argv[++i]);
        } else if (arg == "--scaling-norm" && i + 1 < argc) {
            options.nr.scaling_norm = argv[++i];
        } else if (arg == "--scaling-clamp" && i + 1 < argc) {
            options.nr.scaling_clamp = std::stod(argv[++i]);
        } else if (arg == "--scaling-eps" && i + 1 < argc) {
            options.nr.scaling_eps = std::stod(argv[++i]);
        } else if (arg == "--log-scaling-stats" && i + 1 < argc) {
            options.nr.log_scaling_stats = parse_bool(argv[++i]);
        } else if (arg == "--previous-dx-warm-start" && i + 1 < argc) {
            options.nr.previous_dx_warm_start = parse_bool(argv[++i]);
        } else if (arg == "--enable-cudss-fallback" && i + 1 < argc) {
            options.nr.enable_cudss_fallback = parse_bool(argv[++i]);
            if (!options.nr.enable_cudss_fallback) {
                options.nr.fallback_policy = "off";
            }
        } else if (arg == "--fallback-policy" && i + 1 < argc) {
            options.nr.fallback_policy = argv[++i];
            options.nr.enable_cudss_fallback = options.nr.fallback_policy != "off";
        } else if (arg == "--accept-iterative-by-mismatch" && i + 1 < argc) {
            options.nr.accept_iterative_by_mismatch = parse_bool(argv[++i]);
        } else if (arg == "--accept-mismatch-ratio" && i + 1 < argc) {
            options.nr.accept_mismatch_ratio = std::stod(argv[++i]);
        } else if (arg == "--reject-mismatch-ratio" && i + 1 < argc) {
            options.nr.reject_mismatch_ratio = std::stod(argv[++i]);
        } else if (arg == "--max-middle-accepts" && i + 1 < argc) {
            options.nr.max_middle_accepts = std::stoi(argv[++i]);
        } else if (arg == "--max-a1-middle-accepts" && i + 1 < argc) {
            options.nr.max_a1_middle_accepts = std::stoi(argv[++i]);
        } else if (arg == "--enable-damped-iterative-step" && i + 1 < argc) {
            options.nr.enable_damped_iterative_step = parse_bool(argv[++i]);
        } else if (arg == "--damping-factors" && i + 1 < argc) {
            options.nr.damping_factors = parse_double_list(argv[++i]);
        } else if (arg == "--enable-scaled-mr1-step" && i + 1 < argc) {
            options.nr.enable_scaled_mr1_step = parse_bool(argv[++i]);
        } else if (arg == "--scaled-mr1-gammas" && i + 1 < argc) {
            options.nr.scaled_mr1_gamma_candidates = parse_double_list(argv[++i]);
        } else if ((arg == "--shadow-dx-diagnostic" ||
                    arg == "--enable-shadow-dx-diagnostic") &&
                   i + 1 < argc) {
            options.nr.enable_shadow_dx_diagnostic = parse_bool(argv[++i]);
        } else if (arg == "--skip-middle-backup" && i + 1 < argc) {
            options.nr.skip_middle_backup = parse_bool(argv[++i]);
        } else if (arg == "--dx-safety-check" && i + 1 < argc) {
            options.nr.dx_safety_check = argv[++i];
        } else if (arg == "--global-correction" && i + 1 < argc) {
            options.nr.global_correction = argv[++i];
        } else if (arg == "--global-basis-source" && i + 1 < argc) {
            options.nr.global_basis_source = argv[++i];
        } else if (arg == "--global-rank" && i + 1 < argc) {
            options.nr.global_rank = std::stoi(argv[++i]);
        } else if (arg == "--global-diagnostic-full-step" && i + 1 < argc) {
            options.nr.global_diagnostic_full_step = parse_bool(argv[++i]);
        } else if (arg == "--global-correction-acceptance" && i + 1 < argc) {
            options.nr.global_correction_acceptance = argv[++i];
        } else if (arg == "--global-reset-on-event" && i + 1 < argc) {
            options.nr.global_reset_on_event = parse_bool(argv[++i]);
        } else if (arg == "--field-gain-correction" && i + 1 < argc) {
            options.nr.field_gain_correction = argv[++i];
        } else if (arg == "--field-gain-theta-max" && i + 1 < argc) {
            options.nr.field_gain_theta_max = std::stod(argv[++i]);
        } else if (arg == "--field-gain-vmax" && i + 1 < argc) {
            options.nr.field_gain_vmax = std::stod(argv[++i]);
        } else if (arg == "--field-gain-nonnegative" && i + 1 < argc) {
            options.nr.field_gain_nonnegative = parse_bool(argv[++i]);
        } else if (arg == "--field-gain-trust-ratio" && i + 1 < argc) {
            options.nr.field_gain_trust_ratio = std::stod(argv[++i]);
        } else if (arg == "--theta-j11-correction" && i + 1 < argc) {
            options.nr.theta_j11_correction = argv[++i];
        } else if (arg == "--theta-j11-gmres-maxit" && i + 1 < argc) {
            options.nr.theta_j11_gmres_maxit = std::stoi(argv[++i]);
        } else if (arg == "--theta-j11-correction-trust-ratio" && i + 1 < argc) {
            options.nr.theta_j11_correction_trust_ratio = std::stod(argv[++i]);
        } else if (arg == "--dump-iteration-f-dir" && i + 1 < argc) {
            options.nr.iteration_f_dump_dir = argv[++i];
            options.nr.dump_each_iteration_jf = true;
        } else if (arg == "--no-pure-cudss-baseline") {
            options.run_pure_cudss_baseline = false;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.cases.empty()) {
        throw std::runtime_error("--case is required");
    }
    if (options.warmup < 0) {
        throw std::runtime_error("--warmup must be nonnegative");
    }
    return options;
}

std::filesystem::path default_shadow_output_path(const std::filesystem::path& iter_output)
{
    const auto parent = iter_output.parent_path();
    const std::string stem = iter_output.stem().string();
    const std::string ext = iter_output.extension().empty() ? ".csv" : iter_output.extension().string();
    return parent / (stem + "_shadow_dx" + ext);
}

void ensure_parent_dir(const std::filesystem::path& path)
{
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void write_summary_header(std::ostream& out)
{
    out << "case_name,middle_solver,preconditioner,buses,n,nnz,converged,nr_iters,cudss_calls,gmres_calls,"
           "accepted_gmres_steps,rejected_gmres_steps,fallback_calls,polish_calls,"
           "total_seconds,middle_solver_analyze_setup_seconds,"
           "field_correction_analyze_setup_seconds,full_cudss_analyze_setup_seconds,"
           "fdlf_bp_analyze_seconds,fdlf_bp_factor_seconds,"
           "fdlf_bpp_analyze_seconds,fdlf_bpp_factor_seconds,fdlf_p_rhs,fdlf_q_rhs,"
           "pure_cudss_total_seconds,speedup_vs_pure_cudss,"
           "final_mismatch_inf,final_mismatch_2,max_linear_rel_res_accepted,"
           "avg_linear_rel_res_accepted,accepted_gmres_linear_rel_res_mean,"
           "accepted_gmres_linear_rel_res_max,accepted_gmres_mismatch_reduction_ratio_mean,"
           "shadow_dx_diagnostic_seconds,"
           "gmres_block_size,gmres_restart,gmres_max_iters,bicgstab_iters,gmres_rtol,"
           "gmres_fixed_iter_mode,"
           "pure_full_cudss_calls,hybrid_required_full_cudss_calls,"
           "diagnostic_full_cudss_calls,total_full_cudss_calls,"
           "corrections_attempted,corrections_accepted,corrections_skipped,"
           "final_basis_rank,global_correction_seconds,global_az_seconds,"
           "global_dense_ls_seconds,stop_reason\n";
}

void write_summary_row(std::ostream& out,
                       const cupf_minimal::HybridNrResult& result,
                       const cupf_minimal::HybridNrOptions& options)
{
    out << std::boolalpha << std::setprecision(12)
        << result.case_name << ','
        << options.middle_solver << ','
        << options.preconditioner << ','
        << result.buses << ','
        << result.n << ','
        << result.nnz << ','
        << result.converged << ','
        << result.nr_iters << ','
        << result.cudss_calls << ','
        << result.gmres_calls << ','
        << result.accepted_gmres_steps << ','
        << result.rejected_gmres_steps << ','
        << result.fallback_calls << ','
        << result.polish_calls << ','
        << result.total_seconds << ','
        << result.middle_solver_analyze_setup_seconds << ','
        << result.field_correction_analyze_setup_seconds << ','
        << result.full_cudss_analyze_setup_seconds << ','
        << result.fdlf_bp_analyze_seconds << ','
        << result.fdlf_bp_factor_seconds << ','
        << result.fdlf_bpp_analyze_seconds << ','
        << result.fdlf_bpp_factor_seconds << ','
        << result.fdlf_p_rhs << ','
        << result.fdlf_q_rhs << ','
        << result.pure_cudss_total_seconds << ','
        << result.speedup_vs_pure_cudss << ','
        << result.final_mismatch_inf << ','
        << result.final_mismatch_2 << ','
        << result.max_linear_rel_res_accepted << ','
        << result.avg_linear_rel_res_accepted << ','
        << result.avg_linear_rel_res_accepted << ','
        << result.max_linear_rel_res_accepted << ','
        << result.accepted_gmres_mismatch_reduction_ratio_mean << ','
        << result.shadow_dx_diagnostic_seconds << ','
        << result.gmres_block_size << ','
        << result.gmres_restart << ','
        << result.gmres_max_iters << ','
        << result.bicgstab_iters << ','
        << result.gmres_rtol << ','
        << result.gmres_fixed_iter_mode << ','
        << result.pure_full_cudss_calls << ','
        << result.cudss_calls << ','
        << result.diagnostic_full_cudss_calls << ','
        << result.cudss_calls + result.diagnostic_full_cudss_calls << ','
        << result.corrections_attempted << ','
        << result.corrections_accepted << ','
        << result.corrections_skipped << ','
        << result.final_basis_rank << ','
        << result.global_correction_seconds << ','
        << result.global_az_seconds << ','
        << result.global_dense_ls_seconds << ','
        << result.stop_reason << '\n';
}

void write_iter_header(std::ostream& out)
{
    out << "case_name,nr_iter,solver_used,linear_iters,mismatch_inf_before,"
           "mismatch_inf_after,mismatch_2_before,mismatch_2_after,linear_rel_res,"
           "linear_abs_res,factor_age,stale_solve_calls,current_j_spmv_calls,"
           "bicgstab_refinement_iters,gmres_refinement_iters,stale_prec_solve_count,"
           "stale_solve_f_seconds,"
           "current_j_spmv_r0_seconds,stale_solve_r0_seconds,"
           "current_j_spmv_r1_seconds,stale_solve_r1_seconds,"
           "stale_prec_solve_seconds,"
           "linear_residual_after_r0,linear_residual_after_r1_richardson,"
           "linear_residual_after_r2_richardson,linear_residual_after_bicgstab,"
           "linear_residual_after_gmres,"
           "jacobian_seconds,linear_setup_seconds,linear_solve_seconds,"
           "gmres_trial_setup_seconds,gmres_trial_solve_seconds,"
           "fallback_cudss_setup_seconds,fallback_cudss_solve_seconds,"
           "block_jacobi_apply_seconds,ras_setup_seconds,ras_apply_seconds,"
           "ras_gather_seconds,ras_local_gemv_seconds,ras_scatter_seconds,"
           "coarse_az0_spmv_seconds,"
           "coarse_compress_seconds,coarse_solve_seconds,coarse_expand_seconds,"
           "coarse_total_seconds,preconditioner_total_seconds,mr1_spmv_seconds,"
           "mr1_fused_dot_seconds,mr1_update_seconds,mr2_w1_spmv_seconds,"
           "bicgstab_total_seconds,bicgstab_spmv_seconds,"
           "bicgstab_dot_reduction_seconds,bicgstab_update_seconds,"
           "gmres_total_seconds,gmres_spmv_seconds,gmres_dot_seconds,"
           "gmres_orthogonalization_seconds,gmres_update_seconds,"
           "gmres_scalar_sync_seconds,host_sync_seconds,unaccounted_seconds,"
           "middle_solver_total_seconds,"
           "block_ilu_factor_seconds,block_ilu_apply_seconds,"
           "block_ilu_forward_seconds,block_ilu_backward_seconds,"
           "block_ilu_l_levels,block_ilu_u_levels,block_ilu_avg_level_width,"
           "block_ilu_max_level_width,block_ilu_block_nnz,block_ilu_failed,"
           "j11_factor_seconds,j11_solve_seconds,j22_factor_seconds,j22_solve_seconds,"
           "fdlf_bp_solve_seconds,fdlf_bpp_solve_seconds,fdlf_cross_spmv_seconds,"
           "fdlf_round0_wall_seconds,fdlf_round1_wall_seconds,fdlf_2round_wall_seconds,"
           "field_correction_wall_seconds,field_correction_serial_sum_seconds,"
           "full_residual_spmv_seconds,residual_axpy_seconds,rhs_gather_seconds,"
           "j11_value_update_seconds,j22_value_update_seconds,"
           "j12_value_update_seconds,j21_value_update_seconds,"
           "j12_spmv_seconds,j21_spmv_seconds,"
           "j11_solve_round1_seconds,j22_solve_round1_seconds,"
           "dx_accum_seconds,non_cudss_overhead_seconds,"
           "a1_event_wait_seconds,a1_unaccounted_seconds,"
           "coarse_failed,chosen_gamma,mismatch_ratio_gamma_4,"
           "mismatch_ratio_gamma_2,mismatch_ratio_gamma_1,"
           "extra_mismatch_eval_seconds,scaled_total_middle_seconds,"
           "scaling_row_norm_seconds,scaling_col_norm_seconds,"
           "scaling_apply_values_seconds,scaling_apply_rhs_seconds,"
           "scaling_total_seconds,scaled_linear_rel_res,unscaled_linear_rel_res,"
           "scaled_linear_abs_res,unscaled_linear_abs_res,"
           "dr_min,dr_max,dr_geomean,dc_min,dc_max,dc_geomean,"
           "row_norm_cv_before,row_norm_cv_after,col_norm_cv_before,col_norm_cv_after,"
           "partition_mode,weighted_graph_build_seconds,metis_partition_seconds,"
           "permutation_build_seconds,num_bus_partitions,min_block_unknowns,"
           "max_block_unknowns,avg_block_unknowns,std_block_unknowns,"
           "diagonal_block_nnz_ratio,offblock_nnz_ratio,total_weighted_coupling,"
           "diagonal_weighted_coupling_ratio,offblock_weighted_coupling_ratio,"
           "j11_diagonal_weighted_ratio,j12_diagonal_weighted_ratio,"
           "j21_diagonal_weighted_ratio,j22_diagonal_weighted_ratio,"
           "theta_vmag_split_count,pq_split_count,"
           "shadow_dx_diagnostic,shadow_dx_diagnostic_seconds,"
           "shadow_mismatch_eval_seconds,"
           "shadow_gmres_setup_seconds,shadow_gmres_solve_seconds,"
           "shadow_cudss_analyze_seconds,shadow_cudss_factorize_seconds,"
           "shadow_cudss_solve_seconds,shadow_linear_rel_res_gmres,"
           "shadow_linear_abs_res_gmres,shadow_dx_gmres_norm2,"
           "shadow_mismatch_after_gmres_inf,shadow_mismatch_after_gmres_2,"
           "shadow_mismatch_after_gmres_p_inf,shadow_mismatch_after_gmres_p_2,"
           "shadow_mismatch_after_gmres_q_inf,shadow_mismatch_after_gmres_q_2,"
           "shadow_linear_rel_res_cudss,shadow_linear_abs_res_cudss,"
           "shadow_dx_cudss_norm2,shadow_mismatch_after_cudss_inf,"
           "shadow_mismatch_after_cudss_2,shadow_mismatch_after_cudss_p_inf,"
           "shadow_mismatch_after_cudss_p_2,shadow_mismatch_after_cudss_q_inf,"
           "shadow_mismatch_after_cudss_q_2,shadow_dx_norm_ratio,"
           "shadow_dx_cosine,shadow_dx_projection,shadow_dx_orth_error,"
           "shadow_theta_norm_ratio,shadow_theta_cosine,shadow_vmag_norm_ratio,"
           "shadow_vmag_cosine,shadow_max_abs_dx_gmres,shadow_max_abs_dx_cudss,"
           "shadow_max_abs_dx_diff,"
           "voltage_update_seconds,mismatch_recompute_seconds,step_accepted,"
           "fallback_used,nr_iter_total_seconds,"
           "global_correction_attempted,global_correction_used,"
           "global_correction_skipped_reason,global_basis_rank_before,"
           "global_basis_rank_after,global_linear_res_before,"
           "global_linear_res_after,global_correction_gain,"
           "global_correction_norm_ratio,global_correction_seconds,"
           "global_az_seconds,global_dense_ls_seconds,"
           "global_required_basis_added,global_diagnostic_basis_added,"
           "used_diagnostic_full_cudss,cos_theta_gmres,cos_theta_corr,"
           "norm_ratio_gmres,norm_ratio_corr,"
           "field_gain_attempted,field_gain_accepted,field_gain_skipped_reason,"
           "gamma_theta,gamma_v,lin_res_before_gain,lin_res_after_gain,"
           "nonlinear_res_iter_trial,nonlinear_res_gain_trial,gain_step_norm_ratio,"
           "field_gain_seconds,theta_corr_attempted,theta_corr_accepted,"
           "theta_corr_skipped_reason,theta_p_scalar_beta,j11_corr_norm,"
           "j11_corr_res_before,j11_corr_res_after,p_res_before,p_res_after,"
           "q_res_before,q_res_after,nonlinear_res_after_theta_corr,"
           "theta_corr_seconds,alpha_theta_oracle,alpha_v_oracle,"
           "norm_m11,norm_m12,norm_m21,norm_m22,norm_p_missing,norm_q_missing,"
           "norm_ad,frac_m11,frac_m12,frac_m21,frac_m22,stop_reason\n";
}

void write_iter_rows(std::ostream& out, const cupf_minimal::HybridNrResult& result)
{
    out << std::boolalpha << std::setprecision(12);
    for (const auto& log : result.iteration_logs) {
        out << log.case_name << ','
            << log.nr_iter << ','
            << log.solver_used << ','
            << log.linear_iters << ','
            << log.mismatch_inf_before << ','
            << log.mismatch_inf_after << ','
            << log.mismatch_2_before << ','
            << log.mismatch_2_after << ','
            << log.linear_rel_res << ','
            << log.linear_abs_res << ','
            << log.factor_age << ','
            << log.stale_solve_calls << ','
            << log.current_j_spmv_calls << ','
            << log.bicgstab_refinement_iters << ','
            << log.gmres_refinement_iters << ','
            << log.stale_prec_solve_count << ','
            << log.stale_solve_f_seconds << ','
            << log.current_j_spmv_r0_seconds << ','
            << log.stale_solve_r0_seconds << ','
            << log.current_j_spmv_r1_seconds << ','
            << log.stale_solve_r1_seconds << ','
            << log.stale_prec_solve_seconds << ','
            << log.linear_residual_after_r0 << ','
            << log.linear_residual_after_r1_richardson << ','
            << log.linear_residual_after_r2_richardson << ','
            << log.linear_residual_after_bicgstab << ','
            << log.linear_residual_after_gmres << ','
            << log.jacobian_seconds << ','
            << log.linear_setup_seconds << ','
            << log.linear_solve_seconds << ','
            << log.gmres_trial_setup_seconds << ','
            << log.gmres_trial_solve_seconds << ','
            << log.fallback_cudss_setup_seconds << ','
            << log.fallback_cudss_solve_seconds << ','
            << log.block_jacobi_apply_seconds << ','
            << log.ras_setup_seconds << ','
            << log.ras_apply_seconds << ','
            << log.ras_gather_seconds << ','
            << log.ras_local_gemv_seconds << ','
            << log.ras_scatter_seconds << ','
            << log.coarse_az0_spmv_seconds << ','
            << log.coarse_compress_seconds << ','
            << log.coarse_solve_seconds << ','
            << log.coarse_expand_seconds << ','
            << log.coarse_total_seconds << ','
            << log.preconditioner_total_seconds << ','
            << log.mr1_spmv_seconds << ','
            << log.mr1_fused_dot_seconds << ','
            << log.mr1_update_seconds << ','
            << log.mr2_w1_spmv_seconds << ','
            << log.bicgstab_total_seconds << ','
            << log.bicgstab_spmv_seconds << ','
            << log.bicgstab_dot_reduction_seconds << ','
            << log.bicgstab_update_seconds << ','
            << log.gmres_total_seconds << ','
            << log.gmres_spmv_seconds << ','
            << log.gmres_dot_seconds << ','
            << log.gmres_orthogonalization_seconds << ','
            << log.gmres_update_seconds << ','
            << log.gmres_scalar_sync_seconds << ','
            << log.host_sync_seconds << ','
            << log.unaccounted_seconds << ','
            << log.middle_solver_total_seconds << ','
            << log.block_ilu_factor_seconds << ','
            << log.block_ilu_apply_seconds << ','
            << log.block_ilu_forward_seconds << ','
            << log.block_ilu_backward_seconds << ','
            << log.block_ilu_l_levels << ','
            << log.block_ilu_u_levels << ','
            << log.block_ilu_avg_level_width << ','
            << log.block_ilu_max_level_width << ','
            << log.block_ilu_block_nnz << ','
            << log.block_ilu_failed << ','
            << log.j11_factor_seconds << ','
            << log.j11_solve_seconds << ','
            << log.j22_factor_seconds << ','
            << log.j22_solve_seconds << ','
            << log.fdlf_bp_solve_seconds << ','
            << log.fdlf_bpp_solve_seconds << ','
            << log.fdlf_cross_spmv_seconds << ','
            << log.fdlf_round0_wall_seconds << ','
            << log.fdlf_round1_wall_seconds << ','
            << log.fdlf_2round_wall_seconds << ','
            << log.field_correction_wall_seconds << ','
            << log.field_correction_serial_sum_seconds << ','
            << log.full_residual_spmv_seconds << ','
            << log.residual_axpy_seconds << ','
            << log.rhs_gather_seconds << ','
            << log.j11_value_update_seconds << ','
            << log.j22_value_update_seconds << ','
            << log.j12_value_update_seconds << ','
            << log.j21_value_update_seconds << ','
            << log.j12_spmv_seconds << ','
            << log.j21_spmv_seconds << ','
            << log.j11_solve_round1_seconds << ','
            << log.j22_solve_round1_seconds << ','
            << log.dx_accum_seconds << ','
            << log.non_cudss_overhead_seconds << ','
            << log.a1_event_wait_seconds << ','
            << log.a1_unaccounted_seconds << ','
            << log.coarse_failed << ','
            << log.chosen_gamma << ','
            << log.mismatch_ratio_gamma_4 << ','
            << log.mismatch_ratio_gamma_2 << ','
            << log.mismatch_ratio_gamma_1 << ','
            << log.extra_mismatch_eval_seconds << ','
            << log.scaled_total_middle_seconds << ','
            << log.scaling_row_norm_seconds << ','
            << log.scaling_col_norm_seconds << ','
            << log.scaling_apply_values_seconds << ','
            << log.scaling_apply_rhs_seconds << ','
            << log.scaling_total_seconds << ','
            << log.scaled_linear_rel_res << ','
            << log.unscaled_linear_rel_res << ','
            << log.scaled_linear_abs_res << ','
            << log.unscaled_linear_abs_res << ','
            << log.dr_min << ','
            << log.dr_max << ','
            << log.dr_geomean << ','
            << log.dc_min << ','
            << log.dc_max << ','
            << log.dc_geomean << ','
            << log.row_norm_cv_before << ','
            << log.row_norm_cv_after << ','
            << log.col_norm_cv_before << ','
            << log.col_norm_cv_after << ','
            << log.partition_mode << ','
            << log.weighted_graph_build_seconds << ','
            << log.metis_partition_seconds << ','
            << log.permutation_build_seconds << ','
            << log.num_bus_partitions << ','
            << log.min_block_unknowns << ','
            << log.max_block_unknowns << ','
            << log.avg_block_unknowns << ','
            << log.std_block_unknowns << ','
            << log.diagonal_block_nnz_ratio << ','
            << log.offblock_nnz_ratio << ','
            << log.total_weighted_coupling << ','
            << log.diagonal_weighted_coupling_ratio << ','
            << log.offblock_weighted_coupling_ratio << ','
            << log.j11_diagonal_weighted_ratio << ','
            << log.j12_diagonal_weighted_ratio << ','
            << log.j21_diagonal_weighted_ratio << ','
            << log.j22_diagonal_weighted_ratio << ','
            << log.theta_vmag_split_count << ','
            << log.pq_split_count << ','
            << log.shadow_dx_diagnostic << ','
            << log.shadow_dx_diagnostic_seconds << ','
            << log.shadow_mismatch_eval_seconds << ','
            << log.shadow_gmres_setup_seconds << ','
            << log.shadow_gmres_solve_seconds << ','
            << log.shadow_cudss_analyze_seconds << ','
            << log.shadow_cudss_factorize_seconds << ','
            << log.shadow_cudss_solve_seconds << ','
            << log.shadow_linear_rel_res_gmres << ','
            << log.shadow_linear_abs_res_gmres << ','
            << log.shadow_dx_gmres_norm2 << ','
            << log.shadow_mismatch_after_gmres_inf << ','
            << log.shadow_mismatch_after_gmres_2 << ','
            << log.shadow_mismatch_after_gmres_p_inf << ','
            << log.shadow_mismatch_after_gmres_p_2 << ','
            << log.shadow_mismatch_after_gmres_q_inf << ','
            << log.shadow_mismatch_after_gmres_q_2 << ','
            << log.shadow_linear_rel_res_cudss << ','
            << log.shadow_linear_abs_res_cudss << ','
            << log.shadow_dx_cudss_norm2 << ','
            << log.shadow_mismatch_after_cudss_inf << ','
            << log.shadow_mismatch_after_cudss_2 << ','
            << log.shadow_mismatch_after_cudss_p_inf << ','
            << log.shadow_mismatch_after_cudss_p_2 << ','
            << log.shadow_mismatch_after_cudss_q_inf << ','
            << log.shadow_mismatch_after_cudss_q_2 << ','
            << log.shadow_dx_norm_ratio << ','
            << log.shadow_dx_cosine << ','
            << log.shadow_dx_projection << ','
            << log.shadow_dx_orth_error << ','
            << log.shadow_theta_norm_ratio << ','
            << log.shadow_theta_cosine << ','
            << log.shadow_vmag_norm_ratio << ','
            << log.shadow_vmag_cosine << ','
            << log.shadow_max_abs_dx_gmres << ','
            << log.shadow_max_abs_dx_cudss << ','
            << log.shadow_max_abs_dx_diff << ','
            << log.voltage_update_seconds << ','
            << log.mismatch_recompute_seconds << ','
            << log.step_accepted << ','
            << log.fallback_used << ','
            << log.nr_iter_total_seconds << ','
            << log.global_correction_attempted << ','
            << log.global_correction_used << ','
            << log.global_correction_skipped_reason << ','
            << log.global_basis_rank_before << ','
            << log.global_basis_rank_after << ','
            << log.global_linear_res_before << ','
            << log.global_linear_res_after << ','
            << log.global_correction_gain << ','
            << log.global_correction_norm_ratio << ','
            << log.global_correction_seconds << ','
            << log.global_az_seconds << ','
            << log.global_dense_ls_seconds << ','
            << log.global_required_basis_added << ','
            << log.global_diagnostic_basis_added << ','
            << log.used_diagnostic_full_cudss << ','
            << log.cos_theta_gmres << ','
            << log.cos_theta_corr << ','
            << log.norm_ratio_gmres << ','
            << log.norm_ratio_corr << ','
            << log.field_gain_attempted << ','
            << log.field_gain_accepted << ','
            << log.field_gain_skipped_reason << ','
            << log.gamma_theta << ','
            << log.gamma_v << ','
            << log.lin_res_before_gain << ','
            << log.lin_res_after_gain << ','
            << log.nonlinear_res_iter_trial << ','
            << log.nonlinear_res_gain_trial << ','
            << log.gain_step_norm_ratio << ','
            << log.field_gain_seconds << ','
            << log.theta_corr_attempted << ','
            << log.theta_corr_accepted << ','
            << log.theta_corr_skipped_reason << ','
            << log.theta_p_scalar_beta << ','
            << log.j11_corr_norm << ','
            << log.j11_corr_res_before << ','
            << log.j11_corr_res_after << ','
            << log.p_res_before << ','
            << log.p_res_after << ','
            << log.q_res_before << ','
            << log.q_res_after << ','
            << log.nonlinear_res_after_theta_corr << ','
            << log.theta_corr_seconds << ','
            << log.alpha_theta_oracle << ','
            << log.alpha_v_oracle << ','
            << log.norm_m11 << ','
            << log.norm_m12 << ','
            << log.norm_m21 << ','
            << log.norm_m22 << ','
            << log.norm_p_missing << ','
            << log.norm_q_missing << ','
            << log.norm_ad << ','
            << log.frac_m11 << ','
            << log.frac_m12 << ','
            << log.frac_m21 << ','
            << log.frac_m22 << ','
            << log.stop_reason << '\n';
    }
}

double ratio_or_zero(double numerator, double denominator)
{
    return std::abs(denominator) > 0.0 ? numerator / denominator : 0.0;
}

double nonlinear_ratio(double after, double before)
{
    return before > 0.0 ? after / before : 0.0;
}

double step_efficiency(double before, double after_gmres, double after_cudss)
{
    return ratio_or_zero(before - after_gmres, before - after_cudss);
}

std::string setting_label(const cupf_minimal::HybridNrOptions& options)
{
    std::ostringstream ss;
    ss << options.middle_solver
       << "_" << options.preconditioner
       << "_bs" << options.block_size
       << "_r" << options.gmres_restart
       << "_i" << options.gmres_max_iters
       << "_bicg" << options.bicgstab_iters
       << "_accept" << options.accept_mismatch_ratio
       << "_reject" << options.reject_mismatch_ratio
       << "_polish" << options.cudss_polish_threshold
       << "_iterstart" << options.iterative_start_mismatch_threshold
       << "_fallback_" << options.fallback_policy
       << "_scaled" << options.enable_scaled_mr1_step
       << "_linscale_" << options.linear_scaling
       << "_prevdx_" << options.previous_dx_warm_start
       << "_part_" << options.partition_mode
       << "_bjsetup_" << options.bj_setup
       << "_fused2_" << options.bicgstab_fused_fixed2
       << "_a1dag_" << options.a1_event_dag;
    return ss.str();
}

void write_shadow_header(std::ostream& out)
{
    out << "case,nr_iter,setting,mismatch_before_inf,mismatch_before_2,"
           "linear_rel_res_gmres,linear_rel_res_cudss,"
           "dx_norm_gmres,dx_norm_cudss,dx_norm_ratio,dx_cosine,dx_projection,"
           "dx_orth_error,theta_norm_ratio,theta_cosine,vmag_norm_ratio,vmag_cosine,"
           "alpha_theta_oracle,alpha_v_oracle,"
           "mismatch_after_gmres_inf,mismatch_after_gmres_2,"
           "mismatch_after_cudss_inf,mismatch_after_cudss_2,"
           "mismatch_after_gmres_p_inf,mismatch_after_gmres_q_inf,"
           "mismatch_after_cudss_p_inf,mismatch_after_cudss_q_inf,"
           "gmres_p_ratio_inf,gmres_q_ratio_inf,cudss_p_ratio_inf,cudss_q_ratio_inf,"
           "gmres_nonlinear_ratio_inf,gmres_nonlinear_ratio_2,"
           "cudss_nonlinear_ratio_inf,cudss_nonlinear_ratio_2,"
           "step_efficiency_inf,step_efficiency_2,"
           "gmres_setup_ms,gmres_solve_ms,gmres_total_ms,shadow_cudss_ms,"
           "mismatch_eval_ms,accepted,fallback,actual_solver_used\n";
}

void write_shadow_rows(std::ostream& out,
                       const cupf_minimal::HybridNrResult& result,
                       const cupf_minimal::HybridNrOptions& options)
{
    out << std::boolalpha << std::setprecision(12);
    const std::string setting = setting_label(options);
    for (const auto& log : result.iteration_logs) {
        if (!log.shadow_dx_diagnostic) {
            continue;
        }
        const double gmres_ratio_inf =
            nonlinear_ratio(log.shadow_mismatch_after_gmres_inf, log.mismatch_inf_before);
        const double gmres_ratio_2 =
            nonlinear_ratio(log.shadow_mismatch_after_gmres_2, log.mismatch_2_before);
        const double cudss_ratio_inf =
            nonlinear_ratio(log.shadow_mismatch_after_cudss_inf, log.mismatch_inf_before);
        const double cudss_ratio_2 =
            nonlinear_ratio(log.shadow_mismatch_after_cudss_2, log.mismatch_2_before);
        const double gmres_setup_ms = 1000.0 * log.shadow_gmres_setup_seconds;
        const double gmres_solve_ms = 1000.0 * log.shadow_gmres_solve_seconds;
        const double shadow_cudss_ms =
            1000.0 * (log.shadow_cudss_analyze_seconds +
                      log.shadow_cudss_factorize_seconds +
                      log.shadow_cudss_solve_seconds);

        out << log.case_name << ','
            << log.nr_iter << ','
            << setting << ','
            << log.mismatch_inf_before << ','
            << log.mismatch_2_before << ','
            << log.shadow_linear_rel_res_gmres << ','
            << log.shadow_linear_rel_res_cudss << ','
            << log.shadow_dx_gmres_norm2 << ','
            << log.shadow_dx_cudss_norm2 << ','
            << log.shadow_dx_norm_ratio << ','
            << log.shadow_dx_cosine << ','
            << log.shadow_dx_projection << ','
            << log.shadow_dx_orth_error << ','
            << log.shadow_theta_norm_ratio << ','
            << log.shadow_theta_cosine << ','
            << log.shadow_vmag_norm_ratio << ','
            << log.shadow_vmag_cosine << ','
            << log.alpha_theta_oracle << ','
            << log.alpha_v_oracle << ','
            << log.shadow_mismatch_after_gmres_inf << ','
            << log.shadow_mismatch_after_gmres_2 << ','
            << log.shadow_mismatch_after_cudss_inf << ','
            << log.shadow_mismatch_after_cudss_2 << ','
            << log.shadow_mismatch_after_gmres_p_inf << ','
            << log.shadow_mismatch_after_gmres_q_inf << ','
            << log.shadow_mismatch_after_cudss_p_inf << ','
            << log.shadow_mismatch_after_cudss_q_inf << ','
            << nonlinear_ratio(log.shadow_mismatch_after_gmres_p_inf,
                               log.mismatch_inf_before) << ','
            << nonlinear_ratio(log.shadow_mismatch_after_gmres_q_inf,
                               log.mismatch_inf_before) << ','
            << nonlinear_ratio(log.shadow_mismatch_after_cudss_p_inf,
                               log.mismatch_inf_before) << ','
            << nonlinear_ratio(log.shadow_mismatch_after_cudss_q_inf,
                               log.mismatch_inf_before) << ','
            << gmres_ratio_inf << ','
            << gmres_ratio_2 << ','
            << cudss_ratio_inf << ','
            << cudss_ratio_2 << ','
            << step_efficiency(log.mismatch_inf_before,
                               log.shadow_mismatch_after_gmres_inf,
                               log.shadow_mismatch_after_cudss_inf) << ','
            << step_efficiency(log.mismatch_2_before,
                               log.shadow_mismatch_after_gmres_2,
                               log.shadow_mismatch_after_cudss_2) << ','
            << gmres_setup_ms << ','
            << gmres_solve_ms << ','
            << gmres_setup_ms + gmres_solve_ms << ','
            << shadow_cudss_ms << ','
            << 1000.0 * log.shadow_mismatch_eval_seconds << ','
            << log.step_accepted << ','
            << log.fallback_used << ','
            << log.solver_used << '\n';
    }
}

void write_timing_header(std::ostream& out)
{
    out << "case_name,nr_iter,setting,solver_used,block_jacobi_apply_ms,"
           "ras_setup_ms,ras_apply_ms,ras_gather_ms,ras_local_gemv_ms,ras_scatter_ms,"
           "coarse_az0_spmv_ms,coarse_compress_ms,coarse_solve_ms,"
           "coarse_expand_ms,coarse_total_ms,preconditioner_total_ms,"
           "mr1_spmv_ms,mr1_fused_dot_ms,mr1_update_ms,mr2_w1_spmv_ms,"
           "bicgstab_spmv_ms,bicgstab_dot_reduction_ms,bicgstab_update_ms,"
           "bicgstab_scalar_sync_ms,bj_metadata_setup_ms,bj_value_update_ms,"
           "bj_inverse_build_ms,bj_setup_total_ms,bj_cache_reused,"
           "middle_solver_total_ms,block_ilu_factor_ms,block_ilu_apply_ms,"
           "block_ilu_forward_ms,block_ilu_backward_ms,block_ilu_l_levels,"
           "block_ilu_u_levels,block_ilu_avg_level_width,block_ilu_max_level_width,"
           "block_ilu_block_nnz,block_ilu_failed,j11_factor_ms,j11_solve_ms,"
           "j22_factor_ms,j22_solve_ms,field_correction_wall_ms,"
           "fdlf_bp_solve_ms,fdlf_bpp_solve_ms,fdlf_cross_spmv_ms,"
           "fdlf_round0_wall_ms,fdlf_round1_wall_ms,fdlf_2round_wall_ms,"
           "field_correction_serial_sum_ms,full_residual_spmv_ms,residual_axpy_ms,"
           "rhs_gather_ms,j11_value_update_ms,j22_value_update_ms,"
           "j12_value_update_ms,j21_value_update_ms,j12_spmv_ms,j21_spmv_ms,"
           "j11_solve_round1_ms,j22_solve_round1_ms,dx_accum_ms,"
           "a1_event_wait_ms,a1_unaccounted_ms,non_cudss_overhead_ms,"
           "chosen_gamma,mismatch_ratio_gamma_4,"
           "mismatch_ratio_gamma_2,mismatch_ratio_gamma_1,"
           "extra_mismatch_eval_ms,total_middle_time_ms,"
           "scaling_row_norm_ms,scaling_col_norm_ms,scaling_apply_values_ms,"
           "scaling_apply_rhs_ms,scaling_total_ms,"
           "scaled_linear_rel_res,unscaled_linear_rel_res,"
           "dr_min,dr_max,dr_geomean,dc_min,dc_max,dc_geomean,"
           "row_norm_cv_before,row_norm_cv_after,col_norm_cv_before,col_norm_cv_after,"
           "partition_build_ms,weighted_graph_build_ms,"
           "linear_setup_ms,linear_solve_ms,linear_rel_res,accepted,fallback,"
           "coarse_failed,stop_reason\n";
}

void write_timing_rows(std::ostream& out,
                       const cupf_minimal::HybridNrResult& result,
                       const cupf_minimal::HybridNrOptions& options)
{
    out << std::boolalpha << std::setprecision(12);
    const std::string setting = setting_label(options);
    for (const auto& log : result.iteration_logs) {
        if (log.solver_used != "gmres_middle" && log.solver_used != "mr1_middle" &&
            log.solver_used != "mr2_middle" && log.solver_used != "bicgstab_middle" &&
            log.solver_used != "bicgstab_a0_middle" &&
            log.solver_used != "bicgstab_a1_middle" &&
            log.solver_used != "bicgstab_a0_device_middle" &&
            log.solver_used != "bicgstab_a1_device_middle" &&
            log.solver_used != "bicgstab_j11_device_middle" &&
            log.solver_used != "bicgstab_bpbpp_refine_middle" &&
            log.solver_used != "fdlf_bpbpp_2round" &&
            log.solver_used != "cudss_fallback") {
            continue;
        }
        double total_middle_seconds = 0.0;
        if (log.field_correction_wall_seconds > 0.0) {
            total_middle_seconds =
                log.gmres_trial_setup_seconds +
                log.gmres_trial_solve_seconds +
                log.field_correction_wall_seconds;
        } else if (log.scaled_total_middle_seconds > 0.0) {
            total_middle_seconds = log.scaled_total_middle_seconds;
        } else if (log.gmres_trial_setup_seconds > 0.0 || log.gmres_trial_solve_seconds > 0.0) {
            total_middle_seconds = log.gmres_trial_setup_seconds + log.gmres_trial_solve_seconds;
        }
        if (total_middle_seconds <= 0.0) {
            total_middle_seconds = log.linear_setup_seconds + log.linear_solve_seconds;
        }
        out << log.case_name << ','
            << log.nr_iter << ','
            << setting << ','
            << log.solver_used << ','
            << 1000.0 * log.block_jacobi_apply_seconds << ','
            << 1000.0 * log.ras_setup_seconds << ','
            << 1000.0 * log.ras_apply_seconds << ','
            << 1000.0 * log.ras_gather_seconds << ','
            << 1000.0 * log.ras_local_gemv_seconds << ','
            << 1000.0 * log.ras_scatter_seconds << ','
            << 1000.0 * log.coarse_az0_spmv_seconds << ','
            << 1000.0 * log.coarse_compress_seconds << ','
            << 1000.0 * log.coarse_solve_seconds << ','
            << 1000.0 * log.coarse_expand_seconds << ','
            << 1000.0 * log.coarse_total_seconds << ','
            << 1000.0 * log.preconditioner_total_seconds << ','
            << 1000.0 * log.mr1_spmv_seconds << ','
            << 1000.0 * log.mr1_fused_dot_seconds << ','
            << 1000.0 * log.mr1_update_seconds << ','
            << 1000.0 * log.mr2_w1_spmv_seconds << ','
            << 1000.0 * log.bicgstab_spmv_seconds << ','
            << 1000.0 * log.bicgstab_dot_reduction_seconds << ','
            << 1000.0 * log.bicgstab_update_seconds << ','
            << 1000.0 * log.bicgstab_scalar_sync_seconds << ','
            << 1000.0 * log.bj_metadata_setup_seconds << ','
            << 1000.0 * log.bj_value_update_seconds << ','
            << 1000.0 * log.bj_inverse_build_seconds << ','
            << 1000.0 * log.bj_setup_total_seconds << ','
            << log.bj_cache_reused << ','
            << 1000.0 * log.middle_solver_total_seconds << ','
            << 1000.0 * log.block_ilu_factor_seconds << ','
            << 1000.0 * log.block_ilu_apply_seconds << ','
            << 1000.0 * log.block_ilu_forward_seconds << ','
            << 1000.0 * log.block_ilu_backward_seconds << ','
            << log.block_ilu_l_levels << ','
            << log.block_ilu_u_levels << ','
            << log.block_ilu_avg_level_width << ','
            << log.block_ilu_max_level_width << ','
            << log.block_ilu_block_nnz << ','
            << log.block_ilu_failed << ','
            << 1000.0 * log.j11_factor_seconds << ','
            << 1000.0 * log.j11_solve_seconds << ','
            << 1000.0 * log.j22_factor_seconds << ','
            << 1000.0 * log.j22_solve_seconds << ','
            << 1000.0 * log.field_correction_wall_seconds << ','
            << 1000.0 * log.fdlf_bp_solve_seconds << ','
            << 1000.0 * log.fdlf_bpp_solve_seconds << ','
            << 1000.0 * log.fdlf_cross_spmv_seconds << ','
            << 1000.0 * log.fdlf_round0_wall_seconds << ','
            << 1000.0 * log.fdlf_round1_wall_seconds << ','
            << 1000.0 * log.fdlf_2round_wall_seconds << ','
            << 1000.0 * log.field_correction_serial_sum_seconds << ','
            << 1000.0 * log.full_residual_spmv_seconds << ','
            << 1000.0 * log.residual_axpy_seconds << ','
            << 1000.0 * log.rhs_gather_seconds << ','
            << 1000.0 * log.j11_value_update_seconds << ','
            << 1000.0 * log.j22_value_update_seconds << ','
            << 1000.0 * log.j12_value_update_seconds << ','
            << 1000.0 * log.j21_value_update_seconds << ','
            << 1000.0 * log.j12_spmv_seconds << ','
            << 1000.0 * log.j21_spmv_seconds << ','
            << 1000.0 * log.j11_solve_round1_seconds << ','
            << 1000.0 * log.j22_solve_round1_seconds << ','
            << 1000.0 * log.dx_accum_seconds << ','
            << 1000.0 * log.a1_event_wait_seconds << ','
            << 1000.0 * log.a1_unaccounted_seconds << ','
            << 1000.0 * log.non_cudss_overhead_seconds << ','
            << log.chosen_gamma << ','
            << log.mismatch_ratio_gamma_4 << ','
            << log.mismatch_ratio_gamma_2 << ','
            << log.mismatch_ratio_gamma_1 << ','
            << 1000.0 * log.extra_mismatch_eval_seconds << ','
            << 1000.0 * total_middle_seconds << ','
            << 1000.0 * log.scaling_row_norm_seconds << ','
            << 1000.0 * log.scaling_col_norm_seconds << ','
            << 1000.0 * log.scaling_apply_values_seconds << ','
            << 1000.0 * log.scaling_apply_rhs_seconds << ','
            << 1000.0 * log.scaling_total_seconds << ','
            << log.scaled_linear_rel_res << ','
            << log.unscaled_linear_rel_res << ','
            << log.dr_min << ','
            << log.dr_max << ','
            << log.dr_geomean << ','
            << log.dc_min << ','
            << log.dc_max << ','
            << log.dc_geomean << ','
            << log.row_norm_cv_before << ','
            << log.row_norm_cv_after << ','
            << log.col_norm_cv_before << ','
            << log.col_norm_cv_after << ','
            << 1000.0 * (log.metis_partition_seconds + log.permutation_build_seconds) << ','
            << 1000.0 * log.weighted_graph_build_seconds << ','
            << 1000.0 * log.linear_setup_seconds << ','
            << 1000.0 * log.linear_solve_seconds << ','
            << log.linear_rel_res << ','
            << log.step_accepted << ','
            << log.fallback_used << ','
            << log.coarse_failed << ','
            << log.stop_reason << '\n';
    }
}

void write_partition_stats_header(std::ostream& out)
{
    out << "case_name,nr_iter,setting,partition_mode,num_bus_partitions,"
           "min_block_unknowns,max_block_unknowns,avg_block_unknowns,std_block_unknowns,"
           "total_nnz,diagonal_block_nnz_ratio,offblock_nnz_ratio,"
           "total_weighted_coupling,diagonal_weighted_coupling_ratio,"
           "offblock_weighted_coupling_ratio,j11_diagonal_weighted_ratio,"
           "j11_offblock_weighted_ratio,j12_diagonal_weighted_ratio,"
           "j12_offblock_weighted_ratio,j21_diagonal_weighted_ratio,"
           "j21_offblock_weighted_ratio,j22_diagonal_weighted_ratio,"
           "j22_offblock_weighted_ratio,theta_vmag_split_count,pq_split_count,"
           "partition_build_ms,weighted_graph_build_ms,block_extract_ms,block_inverse_ms\n";
}

void write_partition_stats_rows(std::ostream& out,
                                const cupf_minimal::HybridNrResult& result,
                                const cupf_minimal::HybridNrOptions& options)
{
    out << std::boolalpha << std::setprecision(12);
    const std::string setting = setting_label(options);
    for (const auto& log : result.iteration_logs) {
        if (log.solver_used != "gmres_middle" && log.solver_used != "mr1_middle" &&
            log.solver_used != "bicgstab_middle" &&
            log.solver_used != "bicgstab_a0_middle" &&
            log.solver_used != "bicgstab_a1_middle" &&
            log.solver_used != "bicgstab_a0_device_middle" &&
            log.solver_used != "bicgstab_a1_device_middle" &&
            log.solver_used != "bicgstab_j11_device_middle" &&
            log.solver_used != "cudss_fallback") {
            continue;
        }
        out << log.case_name << ','
            << log.nr_iter << ','
            << setting << ','
            << log.partition_mode << ','
            << log.num_bus_partitions << ','
            << log.min_block_unknowns << ','
            << log.max_block_unknowns << ','
            << log.avg_block_unknowns << ','
            << log.std_block_unknowns << ','
            << result.nnz << ','
            << log.diagonal_block_nnz_ratio << ','
            << log.offblock_nnz_ratio << ','
            << log.total_weighted_coupling << ','
            << log.diagonal_weighted_coupling_ratio << ','
            << log.offblock_weighted_coupling_ratio << ','
            << log.j11_diagonal_weighted_ratio << ','
            << 1.0 - log.j11_diagonal_weighted_ratio << ','
            << log.j12_diagonal_weighted_ratio << ','
            << 1.0 - log.j12_diagonal_weighted_ratio << ','
            << log.j21_diagonal_weighted_ratio << ','
            << 1.0 - log.j21_diagonal_weighted_ratio << ','
            << log.j22_diagonal_weighted_ratio << ','
            << 1.0 - log.j22_diagonal_weighted_ratio << ','
            << log.theta_vmag_split_count << ','
            << log.pq_split_count << ','
            << 1000.0 * (log.metis_partition_seconds + log.permutation_build_seconds) << ','
            << 1000.0 * log.weighted_graph_build_seconds << ','
            << 1000.0 * log.block_extract_seconds << ','
            << 1000.0 * log.block_inverse_seconds << '\n';
    }
}

void print_iteration_trace(const cupf_minimal::HybridNrResult& result)
{
    std::cout << std::scientific << std::setprecision(6);
    for (const auto& log : result.iteration_logs) {
        std::cout << "iter=" << log.nr_iter
                  << " solver=" << log.solver_used
                  << " mismatch_inf=" << log.mismatch_inf_before
                  << " -> " << log.mismatch_inf_after
                  << " mismatch_2=" << log.mismatch_2_before
                  << " -> " << log.mismatch_2_after
                  << " linear_rel=" << log.linear_rel_res
                  << " linear_iters=" << log.linear_iters
                  << " gamma=" << log.chosen_gamma
                  << " accepted=" << std::boolalpha << log.step_accepted
                  << " fallback=" << log.fallback_used
                  << "\n";
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        const bool write_shadow = options.nr.enable_shadow_dx_diagnostic ||
                                  options.shadow_output_explicit;
        const std::filesystem::path shadow_output =
            options.shadow_output_explicit ? options.shadow_output :
                                             default_shadow_output_path(options.iter_output);
        ensure_parent_dir(options.output);
        ensure_parent_dir(options.iter_output);
        if (write_shadow) {
            ensure_parent_dir(shadow_output);
        }
        if (options.timing_output_explicit) {
            ensure_parent_dir(options.timing_output);
        }
        if (options.partition_stats_output_explicit) {
            ensure_parent_dir(options.partition_stats_output);
        }

        std::ofstream summary_out(options.output);
        std::ofstream iter_out(options.iter_output);
        std::ofstream shadow_out;
        std::ofstream timing_out;
        std::ofstream partition_stats_out;
        if (write_shadow) {
            shadow_out.open(shadow_output);
        }
        if (options.timing_output_explicit) {
            timing_out.open(options.timing_output);
        }
        if (options.partition_stats_output_explicit) {
            partition_stats_out.open(options.partition_stats_output);
        }
        if (!summary_out || !iter_out) {
            throw std::runtime_error("failed to open output CSV files");
        }
        if (write_shadow && !shadow_out) {
            throw std::runtime_error("failed to open shadow diagnostic CSV file");
        }
        if (options.timing_output_explicit && !timing_out) {
            throw std::runtime_error("failed to open timing CSV file");
        }
        if (options.partition_stats_output_explicit && !partition_stats_out) {
            throw std::runtime_error("failed to open partition stats CSV file");
        }
        write_summary_header(summary_out);
        write_iter_header(iter_out);
        if (write_shadow) {
            write_shadow_header(shadow_out);
        }
        if (options.timing_output_explicit) {
            write_timing_header(timing_out);
        }
        if (options.partition_stats_output_explicit) {
            write_partition_stats_header(partition_stats_out);
        }

        for (int32_t case_id = 0; case_id < static_cast<int32_t>(options.cases.size()); ++case_id) {
            const std::string& case_name = options.cases[static_cast<std::size_t>(case_id)];
            const auto data = cupf_minimal::load_dump_case(options.case_root / case_name);

            double pure_total = 0.0;
            int32_t pure_full_cudss_calls = 0;
            if (options.run_pure_cudss_baseline && options.nr.solver != "pure_cudss") {
                auto pure_options = options.nr;
                pure_options.solver = "pure_cudss";
                for (int32_t w = 0; w < options.warmup; ++w) {
                    (void)cupf_minimal::run_hybrid_nr_case(data, pure_options, 0.0, 0, case_id);
                }
                const auto pure = cupf_minimal::run_hybrid_nr_case(data, pure_options, 0.0, 0, case_id);
                pure_total = pure.total_seconds;
                pure_full_cudss_calls = pure.cudss_calls;
                std::cout << "[pure_cudss] case=" << case_name
                          << " converged=" << std::boolalpha << pure.converged
                          << " nr_iters=" << pure.nr_iters
                          << " total=" << std::scientific << std::setprecision(6)
                          << pure.total_seconds
                          << " final_inf=" << pure.final_mismatch_inf << "\n";
            }

            for (int32_t w = 0; w < options.warmup; ++w) {
                (void)cupf_minimal::run_hybrid_nr_case(data, options.nr, pure_total, pure_full_cudss_calls, case_id);
            }
            const auto result = cupf_minimal::run_hybrid_nr_case(data, options.nr, pure_total, pure_full_cudss_calls, case_id);
            write_summary_row(summary_out, result, options.nr);
            write_iter_rows(iter_out, result);
            if (write_shadow) {
                write_shadow_rows(shadow_out, result, options.nr);
            }
            if (options.timing_output_explicit) {
                write_timing_rows(timing_out, result, options.nr);
            }
            if (options.partition_stats_output_explicit) {
                write_partition_stats_rows(partition_stats_out, result, options.nr);
            }
            print_iteration_trace(result);
            std::cout << "[result] case=" << result.case_name
                      << " converged=" << std::boolalpha << result.converged
                      << " nr_iters=" << result.nr_iters
                      << " cudss_calls=" << result.cudss_calls
                      << " gmres_calls=" << result.gmres_calls
                      << " fallback_calls=" << result.fallback_calls
                      << " total=" << std::scientific << std::setprecision(6)
                      << result.total_seconds
                      << " speedup_vs_pure=" << result.speedup_vs_pure_cudss
                      << " final_inf=" << result.final_mismatch_inf
                      << " stop=" << result.stop_reason << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "hybrid_nr_bench failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
