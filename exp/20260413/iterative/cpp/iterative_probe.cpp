#include "iterative_probe_common.hpp"
#include "iterative_solvers.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using exp_20260413::iterative::find_snapshot_dirs;
using exp_20260413::iterative::read_snapshot;
using exp_20260413::iterative::probe::ProbeResult;
using exp_20260413::iterative::probe::SolverOptions;

struct CliOptions {
    std::vector<std::filesystem::path> snapshot_dirs;
    std::filesystem::path snapshot_root = "/workspace/exp/20260413/iterative/dumps";
    std::filesystem::path output_csv;
    SolverOptions solver_options;
};

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0
        << " [--snapshot-root PATH] [--snapshot-dir PATH]"
        << " [--solver " << exp_20260413::iterative::probe::supported_solver_names() << ']'
        << " [--tolerance FLOAT] [--max-iter INT]"
        << " [--ilut-drop-tol FLOAT] [--ilut-fill-factor INT]"
        << " [--block-size INT] [--ilu-pivot-tol FLOAT]"
        << " [--output-csv PATH]\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--snapshot-root" && i + 1 < argc) {
            options.snapshot_root = argv[++i];
        } else if (arg == "--snapshot-dir" && i + 1 < argc) {
            options.snapshot_dirs.push_back(argv[++i]);
        } else if (arg == "--solver" && i + 1 < argc) {
            options.solver_options.solver = argv[++i];
        } else if (arg == "--tolerance" && i + 1 < argc) {
            options.solver_options.tolerance = std::stod(argv[++i]);
        } else if (arg == "--max-iter" && i + 1 < argc) {
            options.solver_options.max_iter = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--ilut-drop-tol" && i + 1 < argc) {
            options.solver_options.ilut_drop_tol = std::stod(argv[++i]);
        } else if (arg == "--ilut-fill-factor" && i + 1 < argc) {
            options.solver_options.ilut_fill_factor = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--block-size" && i + 1 < argc) {
            options.solver_options.block_size = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--ilu-pivot-tol" && i + 1 < argc) {
            options.solver_options.ilu_pivot_tol = std::stod(argv[++i]);
        } else if (arg == "--output-csv" && i + 1 < argc) {
            options.output_csv = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    const SolverOptions& solver = options.solver_options;
    if (solver.tolerance <= 0.0) {
        throw std::runtime_error("--tolerance must be positive");
    }
    if (solver.max_iter <= 0) {
        throw std::runtime_error("--max-iter must be positive");
    }
    if (solver.ilut_fill_factor <= 0) {
        throw std::runtime_error("--ilut-fill-factor must be positive");
    }
    if (solver.block_size <= 0) {
        throw std::runtime_error("--block-size must be positive");
    }
    if (solver.ilu_pivot_tol <= 0.0) {
        throw std::runtime_error("--ilu-pivot-tol must be positive");
    }
    if (!exp_20260413::iterative::probe::is_supported_solver(solver.solver)) {
        throw std::runtime_error("Unknown solver: " + solver.solver);
    }

    return options;
}

ProbeResult probe_snapshot(const std::filesystem::path& snapshot_dir,
                           const SolverOptions& solver_options)
{
    const auto snapshot = read_snapshot(snapshot_dir);
    const auto matrix = exp_20260413::iterative::probe::make_sparse_matrix(snapshot);
    const auto rhs = exp_20260413::iterative::probe::make_rhs(snapshot);
    return exp_20260413::iterative::probe::solve_snapshot(
        matrix, rhs, snapshot, solver_options, snapshot_dir);
}

std::vector<std::filesystem::path> selected_snapshot_dirs(const CliOptions& options)
{
    if (!options.snapshot_dirs.empty()) {
        return options.snapshot_dirs;
    }
    return find_snapshot_dirs(options.snapshot_root);
}

void print_result(const ProbeResult& result)
{
    std::cout << std::boolalpha << std::scientific << std::setprecision(12)
              << "PROBE "
              << "snapshot=" << result.snapshot_dir << ' '
              << "solver=" << result.solver << ' '
              << "success=" << result.success << ' '
              << "iterations=" << result.iterations << ' '
              << "estimated_error=" << result.estimated_error << ' '
              << "setup_sec=" << result.setup_sec << ' '
              << "solve_sec=" << result.solve_sec << ' '
              << "rhs_inf=" << result.rhs_inf << ' '
              << "residual_inf=" << result.residual_inf << ' '
              << "relative_residual_inf=" << result.relative_residual_inf << ' '
              << "x_inf=" << result.x_inf << ' '
              << "direct_residual_inf=" << result.direct_residual_inf << ' '
              << "x_delta_direct_inf=" << result.x_delta_direct_inf
              << "\n";
}

void write_csv(const std::filesystem::path& output_path,
               const std::vector<ProbeResult>& results)
{
    const auto parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("Failed to open CSV: " + output_path.string());
    }

    out << "snapshot,solver,success,iterations,estimated_error,setup_sec,solve_sec,"
           "rhs_inf,residual_inf,relative_residual_inf,x_inf,direct_residual_inf,x_delta_direct_inf\n";
    out << std::boolalpha << std::scientific << std::setprecision(17);
    for (const ProbeResult& result : results) {
        out << result.snapshot_dir.string() << ','
            << result.solver << ','
            << result.success << ','
            << result.iterations << ','
            << result.estimated_error << ','
            << result.setup_sec << ','
            << result.solve_sec << ','
            << result.rhs_inf << ','
            << result.residual_inf << ','
            << result.relative_residual_inf << ','
            << result.x_inf << ','
            << result.direct_residual_inf << ','
            << result.x_delta_direct_inf
            << '\n';
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        const auto snapshot_dirs = selected_snapshot_dirs(options);
        if (snapshot_dirs.empty()) {
            throw std::runtime_error("No snapshots found");
        }

        std::vector<ProbeResult> results;
        results.reserve(snapshot_dirs.size());
        for (const auto& snapshot_dir : snapshot_dirs) {
            ProbeResult result = probe_snapshot(snapshot_dir, options.solver_options);
            print_result(result);
            results.push_back(std::move(result));
        }

        if (!options.output_csv.empty()) {
            write_csv(options.output_csv, results);
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "iterative_probe failed: " << ex.what() << "\n";
        return 1;
    }
}
