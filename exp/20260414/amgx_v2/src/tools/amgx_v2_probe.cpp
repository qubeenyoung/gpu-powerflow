#include "solver/jfnk_amgx_solver.hpp"

#include "dump_case_loader.hpp"
#include "utils/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef AMGX_V2_DEFAULT_DATASET_ROOT
#define AMGX_V2_DEFAULT_DATASET_ROOT "/workspace/gpu-powerflow/exp/20260414/amgx/cupf_dumps"
#endif

namespace {

using Clock = std::chrono::steady_clock;
using exp_20260414::amgx_v2::DevicePowerFlowState;
using exp_20260414::amgx_v2::GraphOrderingMethod;
using exp_20260414::amgx_v2::HostPowerFlowStructure;
using exp_20260414::amgx_v2::JfnkAmgxSolver;
using exp_20260414::amgx_v2::SolveStats;
using exp_20260414::amgx_v2::SolverOptions;
using exp_20260414::amgx_v2::parse_preconditioner_kind;
using exp_20260414::amgx_v2::preconditioner_kind_name;
using exp_20260414::amgx_v2::parse_graph_ordering_method;

struct CliOptions {
    std::filesystem::path dataset_root = AMGX_V2_DEFAULT_DATASET_ROOT;
    std::vector<std::string> cases = {"case_ACTIVSg200"};
    std::vector<std::filesystem::path> case_dirs;
    std::filesystem::path output_csv;
    std::filesystem::path step_trace_csv;
    GraphOrderingMethod ordering = GraphOrderingMethod::Natural;
    SolverOptions solver;
    bool list_cases = false;
};

struct CaseResult {
    std::string case_name;
    SolveStats stats;
    double total_sec = 0.0;
};

double elapsed_sec(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

std::string ordering_label(GraphOrderingMethod method)
{
    switch (method) {
    case GraphOrderingMethod::Natural:
        return "natural";
    case GraphOrderingMethod::ReverseCuthillMcKee:
        return "rcm";
    }
    return "unknown";
}

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0
        << " [--dataset-root PATH]"
        << " [--case NAME] [--case-dir PATH]"
        << " [--ordering natural|rcm]"
        << " [--preconditioner scalar_amgx|block2x2_amgx|block2x2_amgx_jacobi|bus_block_jacobi]"
        << " [--nonlinear-tol FLOAT] [--linear-tol FLOAT]"
        << " [--max-outer INT] [--inner-max-iter INT] [--gmres-restart INT]"
        << " [--preconditioner-rebuild-interval INT]"
        << " [--continue-on-linear-failure]"
        << " [--dx-residual-check] [--step-trace-csv PATH]"
        << " [--output-csv PATH] [--list-cases]\n";
}

void add_case(CliOptions& options, const std::string& case_name, bool& custom_cases)
{
    if (!custom_cases) {
        options.cases.clear();
        custom_cases = true;
    }
    options.cases.push_back(case_name);
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    bool custom_cases = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset-root" && i + 1 < argc) {
            options.dataset_root = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            add_case(options, argv[++i], custom_cases);
        } else if (arg == "--case-dir" && i + 1 < argc) {
            options.case_dirs.push_back(argv[++i]);
        } else if (arg == "--ordering" && i + 1 < argc) {
            options.ordering = parse_graph_ordering_method(argv[++i]);
        } else if (arg == "--preconditioner" && i + 1 < argc) {
            options.solver.preconditioner = parse_preconditioner_kind(argv[++i]);
        } else if (arg == "--nonlinear-tol" && i + 1 < argc) {
            options.solver.nonlinear_tolerance = std::stod(argv[++i]);
        } else if (arg == "--linear-tol" && i + 1 < argc) {
            options.solver.linear_tolerance = std::stod(argv[++i]);
        } else if (arg == "--max-outer" && i + 1 < argc) {
            options.solver.max_outer_iterations = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--inner-max-iter" && i + 1 < argc) {
            options.solver.max_inner_iterations = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--gmres-restart" && i + 1 < argc) {
            options.solver.gmres_restart = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--preconditioner-rebuild-interval" && i + 1 < argc) {
            options.solver.preconditioner_rebuild_interval =
                static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--continue-on-linear-failure") {
            options.solver.continue_on_linear_failure = true;
        } else if (arg == "--dx-residual-check") {
            options.solver.track_dx_residual = true;
        } else if (arg == "--step-trace-csv" && i + 1 < argc) {
            options.step_trace_csv = argv[++i];
            options.solver.track_dx_residual = true;
        } else if (arg == "--output-csv" && i + 1 < argc) {
            options.output_csv = argv[++i];
        } else if (arg == "--list-cases") {
            options.list_cases = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.solver.nonlinear_tolerance <= 0.0) {
        throw std::runtime_error("--nonlinear-tol must be positive");
    }
    if (options.solver.linear_tolerance <= 0.0) {
        throw std::runtime_error("--linear-tol must be positive");
    }
    if (options.solver.max_outer_iterations <= 0) {
        throw std::runtime_error("--max-outer must be positive");
    }
    if (options.solver.max_inner_iterations <= 0) {
        throw std::runtime_error("--inner-max-iter must be positive");
    }
    if (options.solver.gmres_restart <= 0) {
        throw std::runtime_error("--gmres-restart must be positive");
    }
    if (options.solver.preconditioner_rebuild_interval < 0) {
        throw std::runtime_error("--preconditioner-rebuild-interval must be nonnegative");
    }
    return options;
}

std::vector<std::filesystem::path> resolve_case_dirs(const CliOptions& options)
{
    std::vector<std::filesystem::path> paths = options.case_dirs;
    for (const std::string& case_name : options.cases) {
        paths.push_back(options.dataset_root / case_name);
    }
    if (paths.empty() && !options.list_cases) {
        throw std::runtime_error("No cases were selected");
    }
    return paths;
}

void list_cases(const std::filesystem::path& dataset_root)
{
    if (!std::filesystem::exists(dataset_root)) {
        throw std::runtime_error("Dataset root does not exist: " + dataset_root.string());
    }

    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(dataset_root)) {
        if (entry.is_directory() && std::filesystem::exists(entry.path() / "dump_Ybus.mtx")) {
            names.push_back(entry.path().filename().string());
        }
    }
    std::sort(names.begin(), names.end());
    for (const std::string& name : names) {
        std::cout << name << '\n';
    }
}

void split_complex_vector(const std::vector<std::complex<double>>& input,
                          std::vector<double>& re,
                          std::vector<double>& im)
{
    re.resize(input.size());
    im.resize(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        re[i] = input[i].real();
        im[i] = input[i].imag();
    }
}

CaseResult run_case(const CliOptions& options, const std::filesystem::path& case_dir)
{
    const auto start = Clock::now();
    const cupf::tests::DumpCaseData case_data = cupf::tests::load_dump_case(case_dir);

    std::vector<double> ybus_re;
    std::vector<double> ybus_im;
    std::vector<double> sbus_re;
    std::vector<double> sbus_im;
    std::vector<double> voltage_re;
    std::vector<double> voltage_im;
    split_complex_vector(case_data.ybus_data, ybus_re, ybus_im);
    split_complex_vector(case_data.sbus, sbus_re, sbus_im);
    split_complex_vector(case_data.v0, voltage_re, voltage_im);

    DeviceBuffer<int32_t> d_ybus_row_ptr;
    DeviceBuffer<int32_t> d_ybus_col_idx;
    DeviceBuffer<double> d_ybus_re;
    DeviceBuffer<double> d_ybus_im;
    DeviceBuffer<double> d_sbus_re;
    DeviceBuffer<double> d_sbus_im;
    DeviceBuffer<double> d_voltage_re;
    DeviceBuffer<double> d_voltage_im;

    d_ybus_row_ptr.assign(case_data.indptr.data(), case_data.indptr.size());
    d_ybus_col_idx.assign(case_data.indices.data(), case_data.indices.size());
    d_ybus_re.assign(ybus_re.data(), ybus_re.size());
    d_ybus_im.assign(ybus_im.data(), ybus_im.size());
    d_sbus_re.assign(sbus_re.data(), sbus_re.size());
    d_sbus_im.assign(sbus_im.data(), sbus_im.size());
    d_voltage_re.assign(voltage_re.data(), voltage_re.size());
    d_voltage_im.assign(voltage_im.data(), voltage_im.size());

    HostPowerFlowStructure structure{
        .n_bus = case_data.rows,
        .ybus_row_ptr = case_data.indptr,
        .ybus_col_idx = case_data.indices,
        .pv = case_data.pv,
        .pq = case_data.pq,
        .ordering = options.ordering,
    };
    DevicePowerFlowState state{
        .ybus_row_ptr = d_ybus_row_ptr.data(),
        .ybus_col_idx = d_ybus_col_idx.data(),
        .ybus_re = d_ybus_re.data(),
        .ybus_im = d_ybus_im.data(),
        .sbus_re = d_sbus_re.data(),
        .sbus_im = d_sbus_im.data(),
        .voltage_re = d_voltage_re.data(),
        .voltage_im = d_voltage_im.data(),
    };

    JfnkAmgxSolver solver(options.solver);
    solver.analyze(structure);
    SolveStats stats = solver.solve(state);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto end = Clock::now();
    return CaseResult{
        .case_name = case_data.case_name,
        .stats = std::move(stats),
        .total_sec = elapsed_sec(start, end),
    };
}

void print_result(const CliOptions& options, const CaseResult& result)
{
    const SolveStats& stats = result.stats;
    std::cout << std::boolalpha << std::scientific << std::setprecision(12)
              << "AMGX_V2 "
              << "case=" << result.case_name << ' '
              << "ordering=" << ordering_label(options.ordering) << ' '
              << "preconditioner=" << preconditioner_kind_name(options.solver.preconditioner) << ' '
              << "converged=" << stats.converged << ' '
              << "outer_iterations=" << stats.outer_iterations << ' '
              << "final_nonlinear_mismatch=" << stats.final_mismatch << ' '
              << "linear_tol=" << options.solver.linear_tolerance << ' '
              << "nonlinear_tol=" << options.solver.nonlinear_tolerance << ' '
              << "total_inner_iterations=" << stats.total_inner_iterations << ' '
              << "total_jv_calls=" << stats.total_jv_calls << ' '
              << "linear_failures=" << stats.linear_failures << ' '
              << "preconditioner_rebuilds=" << stats.preconditioner_rebuilds << ' '
              << "preconditioner_rebuild_interval="
              << options.solver.preconditioner_rebuild_interval << ' '
              << "failure_reason="
              << (stats.failure_reason.empty() ? "none" : stats.failure_reason) << ' '
              << "total_sec=" << result.total_sec
              << '\n';

    if (options.solver.track_dx_residual) {
        for (const auto& step : stats.step_trace) {
            std::cout << std::boolalpha << std::scientific << std::setprecision(12)
                      << "DX_RESIDUAL_CHECK "
                      << "case=" << result.case_name << ' '
                      << "outer=" << step.outer_iteration << ' '
                      << "linear_converged=" << step.linear_converged << ' '
                      << "dx_was_applied=" << step.dx_was_applied << ' '
                      << "preconditioner_rebuilt=" << step.preconditioner_rebuilt << ' '
                      << "preconditioner_age=" << step.preconditioner_age << ' '
                      << "before=" << step.before_mismatch << ' '
                      << "after=" << step.after_mismatch << ' '
                      << "ratio=" << step.after_before_ratio << ' '
                      << "linear_residual=" << step.linear_residual << ' '
                      << "inner_iterations=" << step.inner_iterations << ' '
                      << "jv_calls=" << step.jv_calls << ' '
                      << "linear_failures=" << step.linear_failures
                      << '\n';
        }
    }
}

void write_csv(const std::filesystem::path& output_path,
               const CliOptions& options,
               const std::vector<CaseResult>& results)
{
    const auto parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("Failed to open output CSV: " + output_path.string());
    }

    out << "case,ordering,preconditioner,converged,outer_iterations,final_nonlinear_mismatch,"
           "nonlinear_tol,linear_tol,inner_max_iter,gmres_restart,"
           "preconditioner_rebuild_interval,total_inner_iterations,total_jv_calls,"
           "linear_failures,preconditioner_rebuilds,failure_reason,total_sec\n";
    out << std::boolalpha << std::scientific << std::setprecision(17);
    for (const CaseResult& result : results) {
        const SolveStats& stats = result.stats;
        out << result.case_name << ','
            << ordering_label(options.ordering) << ','
            << preconditioner_kind_name(options.solver.preconditioner) << ','
            << stats.converged << ','
            << stats.outer_iterations << ','
            << stats.final_mismatch << ','
            << options.solver.nonlinear_tolerance << ','
            << options.solver.linear_tolerance << ','
            << options.solver.max_inner_iterations << ','
            << options.solver.gmres_restart << ','
            << options.solver.preconditioner_rebuild_interval << ','
            << stats.total_inner_iterations << ','
            << stats.total_jv_calls << ','
            << stats.linear_failures << ','
            << stats.preconditioner_rebuilds << ','
            << (stats.failure_reason.empty() ? "none" : stats.failure_reason) << ','
            << result.total_sec << '\n';
    }
}

void write_step_trace_csv(const std::filesystem::path& output_path,
                          const CliOptions& options,
                          const std::vector<CaseResult>& results)
{
    const auto parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("Failed to open step trace CSV: " + output_path.string());
    }

    out << "case,ordering,preconditioner,outer_iteration,linear_converged,dx_was_applied,"
           "preconditioner_rebuilt,preconditioner_age,"
           "before_mismatch,after_mismatch,after_before_ratio,linear_residual,"
           "inner_iterations,jv_calls,linear_failures\n";
    out << std::boolalpha << std::scientific << std::setprecision(17);
    for (const CaseResult& result : results) {
        for (const auto& step : result.stats.step_trace) {
            out << result.case_name << ','
                << ordering_label(options.ordering) << ','
                << preconditioner_kind_name(options.solver.preconditioner) << ','
                << step.outer_iteration << ','
                << step.linear_converged << ','
                << step.dx_was_applied << ','
                << step.preconditioner_rebuilt << ','
                << step.preconditioner_age << ','
                << step.before_mismatch << ','
                << step.after_mismatch << ','
                << step.after_before_ratio << ','
                << step.linear_residual << ','
                << step.inner_iterations << ','
                << step.jv_calls << ','
                << step.linear_failures << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        if (options.list_cases) {
            list_cases(options.dataset_root);
            return 0;
        }

        const std::vector<std::filesystem::path> case_dirs = resolve_case_dirs(options);
        std::vector<CaseResult> results;
        results.reserve(case_dirs.size());
        for (const std::filesystem::path& case_dir : case_dirs) {
            CaseResult result = run_case(options, case_dir);
            print_result(options, result);
            results.push_back(std::move(result));
        }

        if (!options.output_csv.empty()) {
            write_csv(options.output_csv, options, results);
        }
        if (!options.step_trace_csv.empty()) {
            write_step_trace_csv(options.step_trace_csv, options, results);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "amgx_v2_probe failed: " << ex.what() << '\n';
        return 1;
    }
}
