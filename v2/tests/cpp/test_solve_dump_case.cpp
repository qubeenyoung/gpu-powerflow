#include "dump_case_loader.hpp"

#include "newton_solver/core/newton_solver.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>


namespace {

struct CliOptions {
    std::filesystem::path case_dir = "/workspace/v1/core/dumps/case30_ieee";
    std::string backend   = "cpu";
    std::string jacobian  = "edge_based";
    std::string algorithm = "optimized";   // "optimized" | "pypower_like"
    double  tolerance = 1e-8;
    int32_t max_iter  = 15;
    bool    compare   = false;   // if true, run both and compare V
    double  compare_tol = 1e-10; // max|V_naive - V_opt| threshold
};

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [--case-dir PATH] [--backend cpu|cuda]"
        << " [--jacobian edge_based|vertex_based]"
        << " [--algorithm optimized|pypower_like]"
        << " [--tolerance FLOAT] [--max-iter INT]"
        << " [--compare] [--compare-tol FLOAT]\n";
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
        } else if (arg == "--tolerance" && i + 1 < argc) {
            options.tolerance = std::stod(argv[++i]);
        } else if (arg == "--algorithm" && i + 1 < argc) {
            options.algorithm = argv[++i];
        } else if (arg == "--max-iter" && i + 1 < argc) {
            options.max_iter = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--compare") {
            options.compare = true;
        } else if (arg == "--compare-tol" && i + 1 < argc) {
            options.compare_tol = std::stod(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
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

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions cli = parse_args(argc, argv);
        const auto case_data = cupf::tests::load_dump_case(cli.case_dir);
        const YbusView ybus  = case_data.ybus();
        const NRConfig cfg   = {cli.tolerance, cli.max_iter};

        auto run_solver = [&](CpuAlgorithm algo) -> NRResult {
            NewtonOptions opts;
            opts.backend       = parse_backend(cli.backend);
            opts.jacobian      = parse_jacobian(cli.jacobian);
            opts.cpu_algorithm = algo;

            NewtonSolver solver(opts);
            solver.analyze(ybus,
                           case_data.pv.data(), static_cast<int32_t>(case_data.pv.size()),
                           case_data.pq.data(), static_cast<int32_t>(case_data.pq.size()));

            NRResult result;
            solver.solve(ybus,
                         case_data.sbus.data(), case_data.v0.data(),
                         case_data.pv.data(), static_cast<int32_t>(case_data.pv.size()),
                         case_data.pq.data(), static_cast<int32_t>(case_data.pq.size()),
                         cfg, result);
            validate_result(case_data, result, cli.tolerance);
            return result;
        };

        if (cli.compare) {
            // --compare: run Optimized and PyPowerLike, report max|V_naive - V_opt|
            const NRResult r_opt   = run_solver(CpuAlgorithm::Optimized);
            const NRResult r_naive = run_solver(CpuAlgorithm::PyPowerLike);

            const double delta = max_voltage_delta(r_opt.V, r_naive.V);

            std::cout << std::scientific << std::setprecision(6)
                      << "case="     << case_data.case_name
                      << " buses="   << case_data.rows
                      << " pv="      << case_data.pv.size()
                      << " pq="      << case_data.pq.size()
                      << " iter_opt="   << r_opt.iterations
                      << " iter_naive=" << r_naive.iterations
                      << " max|V_naive-V_opt|=" << delta
                      << "\n";

            if (delta > cli.compare_tol) {
                std::cerr << "FAIL: max|V_naive-V_opt|=" << delta
                          << " exceeds threshold " << cli.compare_tol << "\n";
                return 1;
            }
            return 0;
        }

        // Single-algorithm run (default)
        const NRResult result = run_solver(parse_algorithm(cli.algorithm));

        std::cout << "case="      << case_data.case_name
                  << " backend="  << cli.backend
                  << " jacobian=" << cli.jacobian
                  << " algorithm="<< cli.algorithm
                  << " buses="    << case_data.rows
                  << " pv="       << case_data.pv.size()
                  << " pq="       << case_data.pq.size()
                  << " iterations=" << result.iterations
                  << " mismatch=" << std::scientific << std::setprecision(6) << result.final_mismatch
                  << " max|V-V0|=" << max_voltage_delta(case_data.v0, result.V)
                  << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "test_solve_dump_case failed: " << ex.what() << "\n";
        return 1;
    }
}
