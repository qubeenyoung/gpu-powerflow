#include "dump_case_loader.hpp"

#include "newton_solver/core/newton_solver.hpp"
#include "utils/timer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path case_dir = "/workspace/v1/core/dumps/case30_ieee";
    std::string backend   = "cpu";
    std::string jacobian  = "edge_based";
    std::string algorithm = "optimized";
    double  tolerance = 1e-8;
    int32_t max_iter  = 50;
    int32_t warmup    = 0;
    int32_t repeats   = 1;
};

struct BenchRun {
    bool    success = false;
    int32_t iterations = 0;
    double  final_mismatch = 0.0;
    double  analyze_sec = 0.0;
    double  solve_sec = 0.0;
    double  total_sec = 0.0;
    double  max_v_delta_from_v0 = 0.0;
    std::vector<newton_solver::utils::TimingEntry> timing_entries;
};

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0
        << " [--case-dir PATH] [--backend cpu|cuda]"
        << " [--jacobian edge_based|vertex_based]"
        << " [--algorithm optimized|pypower_like]"
        << " [--tolerance FLOAT] [--max-iter INT]"
        << " [--warmup INT] [--repeats INT]\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-dir" && i + 1 < argc) {
            options.case_dir = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            options.backend = argv[++i];
        } else if (arg == "--jacobian" && i + 1 < argc) {
            options.jacobian = argv[++i];
        } else if (arg == "--algorithm" && i + 1 < argc) {
            options.algorithm = argv[++i];
        } else if (arg == "--tolerance" && i + 1 < argc) {
            options.tolerance = std::stod(argv[++i]);
        } else if (arg == "--max-iter" && i + 1 < argc) {
            options.max_iter = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--warmup" && i + 1 < argc) {
            options.warmup = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--repeats" && i + 1 < argc) {
            options.repeats = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.warmup < 0) {
        throw std::runtime_error("warmup must be >= 0");
    }
    if (options.repeats <= 0) {
        throw std::runtime_error("repeats must be > 0");
    }

    return options;
}

BackendKind parse_backend(const std::string& backend)
{
    if (backend == "cpu")  return BackendKind::CPU;
    if (backend == "cuda") return BackendKind::CUDA;
    throw std::runtime_error("backend must be 'cpu' or 'cuda'");
}

CpuAlgorithm parse_algorithm(const std::string& algorithm)
{
    if (algorithm == "optimized")    return CpuAlgorithm::Optimized;
    if (algorithm == "pypower_like") return CpuAlgorithm::PyPowerLike;
    throw std::runtime_error("algorithm must be 'optimized' or 'pypower_like'");
}

JacobianBuilderType parse_jacobian(const std::string& jacobian)
{
    if (jacobian == "edge_based") {
        return JacobianBuilderType::EdgeBased;
    }
    if (jacobian == "vertex_based") {
        return JacobianBuilderType::VertexBased;
    }
    throw std::runtime_error("jacobian must be 'edge_based' or 'vertex_based'");
}

double max_voltage_delta(const std::vector<std::complex<double>>& lhs,
                         const std::vector<std::complex<double>>& rhs)
{
    double max_delta = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        max_delta = std::max(max_delta, std::abs(lhs[i] - rhs[i]));
    }
    return max_delta;
}

void validate_result(const cupf::tests::DumpCaseData& data,
                     const NRResult& result,
                     double tolerance)
{
    if (static_cast<int32_t>(result.V.size()) != data.rows) {
        throw std::runtime_error("Result voltage size does not match case size");
    }
    if (!std::isfinite(result.final_mismatch)) {
        throw std::runtime_error("Final mismatch is not finite");
    }
    if (!result.converged) {
        throw std::runtime_error("Solver did not converge");
    }
    if (result.final_mismatch > tolerance) {
        throw std::runtime_error("Final mismatch exceeds requested tolerance");
    }
}

BenchRun run_once(const cupf::tests::DumpCaseData& case_data,
                  BackendKind backend,
                  JacobianBuilderType jacobian,
                  CpuAlgorithm algorithm,
                  const NRConfig& config)
{
    const YbusView ybus = case_data.ybus();

    NewtonOptions opts;
    opts.backend       = backend;
    opts.jacobian      = jacobian;
    opts.cpu_algorithm = algorithm;

    NewtonSolver solver(opts);
    newton_solver::utils::resetTimingCollector();

    const auto t0 = std::chrono::steady_clock::now();
    solver.analyze(
        ybus,
        case_data.pv.data(), static_cast<int32_t>(case_data.pv.size()),
        case_data.pq.data(), static_cast<int32_t>(case_data.pq.size()));
    const auto t1 = std::chrono::steady_clock::now();

    NRResult result;
    solver.solve(
        ybus,
        case_data.sbus.data(),
        case_data.v0.data(),
        case_data.pv.data(), static_cast<int32_t>(case_data.pv.size()),
        case_data.pq.data(), static_cast<int32_t>(case_data.pq.size()),
        config,
        result);
    const auto t2 = std::chrono::steady_clock::now();

    validate_result(case_data, result, config.tolerance);

    const auto analyze_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto solve_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto total_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();

    BenchRun run;
    run.success = result.converged;
    run.iterations = result.iterations;
    run.final_mismatch = result.final_mismatch;
    run.analyze_sec = static_cast<double>(analyze_us) / 1000000.0;
    run.solve_sec = static_cast<double>(solve_us) / 1000000.0;
    run.total_sec = static_cast<double>(total_us) / 1000000.0;
    run.max_v_delta_from_v0 = max_voltage_delta(case_data.v0, result.V);
    run.timing_entries = newton_solver::utils::timingSnapshot();
    std::sort(run.timing_entries.begin(), run.timing_entries.end(),
              [](const newton_solver::utils::TimingEntry& lhs,
                 const newton_solver::utils::TimingEntry& rhs) {
                  return std::string_view(lhs.name) < std::string_view(rhs.name);
              });
    return run;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions cli = parse_args(argc, argv);
        const auto case_data = cupf::tests::load_dump_case(cli.case_dir);
        const BackendKind backend = parse_backend(cli.backend);
        const JacobianBuilderType jacobian = parse_jacobian(cli.jacobian);
        const CpuAlgorithm algorithm = parse_algorithm(cli.algorithm);
        const NRConfig cfg{cli.tolerance, cli.max_iter};

        for (int32_t i = 0; i < cli.warmup; ++i) {
            (void)run_once(case_data, backend, jacobian, algorithm, cfg);
        }

        std::cout << std::boolalpha << std::scientific << std::setprecision(12);
        for (int32_t repeat_idx = 0; repeat_idx < cli.repeats; ++repeat_idx) {
            const BenchRun run = run_once(case_data, backend, jacobian, algorithm, cfg);

            std::cout
                << "RUN "
                << "case=" << case_data.case_name << ' '
                << "backend=" << cli.backend << ' '
                << "jacobian=" << cli.jacobian << ' '
                << "algorithm=" << cli.algorithm << ' '
                << "repeat=" << repeat_idx << ' '
                << "success=" << run.success << ' '
                << "iterations=" << run.iterations << ' '
                << "final_mismatch=" << run.final_mismatch << ' '
                << "analyze_sec=" << run.analyze_sec << ' '
                << "solve_sec=" << run.solve_sec << ' '
                << "total_sec=" << run.total_sec << ' '
                << "max_abs_v_delta_from_v0=" << run.max_v_delta_from_v0 << ' '
                << "buses=" << case_data.rows << ' '
                << "pv=" << case_data.pv.size() << ' '
                << "pq=" << case_data.pq.size()
                << "\n";

            for (const newton_solver::utils::TimingEntry& entry : run.timing_entries) {
                const double total_sec = static_cast<double>(entry.total_us) / 1000000.0;
                const double avg_sec =
                    (entry.count > 0) ? (total_sec / static_cast<double>(entry.count)) : 0.0;

                std::cout
                    << "METRIC "
                    << "repeat=" << repeat_idx << ' '
                    << "name=" << entry.name << ' '
                    << "count=" << entry.count << ' '
                    << "total_sec=" << total_sec << ' '
                    << "avg_sec=" << avg_sec
                    << "\n";
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "cupf_case_benchmark failed: " << ex.what() << "\n";
        return 1;
    }
}
