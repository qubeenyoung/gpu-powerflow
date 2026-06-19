// cupf_bench: the single cuPF C++ benchmark. Replicates one dump case into a
// B-case batch, runs solve_batch, and reports one of three measurement modes:
//
//   solve      (default) optimized initialize() + solve() wall time, NO per-stage
//              timing collector (which injects a cudaDeviceSynchronize per operator
//              and so does not reflect the optimized path).
//   operators  per-operator breakdown (ibus / mismatch / jacobian / factor / solve
//              / voltage-update) via the timing collector — for analysis.
//   debug      convergence + residual detail per system — for correctness checks.
#include "dump_case_loader.hpp"
#include "newton_solver/core/newton_solver.hpp"
#include "utils/timer.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class Mode { Solve, Operators, Debug };

Mode parse_mode(const std::string& s)
{
    if (s == "solve") return Mode::Solve;
    if (s == "operators") return Mode::Operators;
    if (s == "debug") return Mode::Debug;
    throw std::invalid_argument("mode must be solve, operators, or debug");
}

// Parsed "<backend>-<precision>" spec. Precision is the two coupled axes the user
// sees as one token: cuPF ComputePolicy (state/Jacobian) + the backend's factor
// precision. fp64 -> FP64/FP64; fp32 -> Mixed/FP32; tf32 -> Mixed/TF32 (custom only);
// mixed -> Mixed/FP32. (cuDSS ignores the custom factor precision.)
struct BenchSpec {
    bool                 cpu         = false;               // CPU backend (KLU/UMFPACK)?
    CpuLinearSolverKind  cpu_solver  = CpuLinearSolverKind::KLU;
    CudaLinearSolverKind backend     = CudaLinearSolverKind::CuDSS;
    ComputePolicy        compute     = ComputePolicy::Mixed;
    CustomPrecision      custom_prec = CustomPrecision::FP32;
};

BenchSpec parse_spec(const std::string& s)
{
    const size_t dash = s.find('-');
    const std::string backend = (dash == std::string::npos) ? "cudss" : s.substr(0, dash);
    const std::string precision = (dash == std::string::npos) ? s : s.substr(dash + 1);

    BenchSpec spec;
    // CPU backend: "cpu-klu" / "cpu-umfpack" (FP64; per-operator timing incl. Jacobian).
    if (backend == "cpu") {
        spec.cpu = true;
        spec.compute = ComputePolicy::FP64;
        if (precision == "klu")           spec.cpu_solver = CpuLinearSolverKind::KLU;
        else if (precision == "umfpack")  spec.cpu_solver = CpuLinearSolverKind::UMFPACK;
        else throw std::invalid_argument("cpu solver must be klu or umfpack (cpu-klu | cpu-umfpack)");
        return spec;
    }
    if (backend == "custom")      spec.backend = CudaLinearSolverKind::Custom;
    else if (backend == "cudss")  spec.backend = CudaLinearSolverKind::CuDSS;
    else throw std::invalid_argument("backend must be cpu, cudss, or custom");

    if (precision == "fp64") {
        spec.compute = ComputePolicy::FP64; spec.custom_prec = CustomPrecision::FP64;
    } else if (precision == "fp32") {
        spec.compute = ComputePolicy::Mixed; spec.custom_prec = CustomPrecision::FP32;
    } else if (precision == "tf32") {
        spec.compute = ComputePolicy::Mixed; spec.custom_prec = CustomPrecision::TF32;
    } else if (precision == "mixed") {
        spec.compute = ComputePolicy::Mixed; spec.custom_prec = CustomPrecision::FP32;
    } else {
        throw std::invalid_argument("precision must be fp64, fp32, tf32, or mixed");
    }
    return spec;
}

double ms_since(const std::chrono::steady_clock::time_point& t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "usage: cupf_bench <case_dir> <backend-precision> <B1,B2,...> "
                     "[mode] [repeats] [max_iter] [scale_step] [matching] [pivot] [pivot_eps] [jacobian]\n"
                     "  backend-precision: cpu-klu|cpu-umfpack|cudss-fp64|cudss-fp32|custom-fp64|custom-fp32|custom-tf32\n"
                     "  mode: solve (default) | operators | debug\n";
        return 2;
    }
    const std::string case_dir = argv[1];
    const BenchSpec spec = parse_spec(argv[2]);
    std::vector<int32_t> batches;
    {
        std::string s = argv[3];
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            if (comma == std::string::npos) comma = s.size();
            batches.push_back(std::stoi(s.substr(pos, comma - pos)));
            pos = comma + 1;
        }
    }
    const Mode mode                = (argc > 4) ? parse_mode(argv[4]) : Mode::Solve;
    const int repeats              = (argc > 5) ? std::stoi(argv[5]) : 5;
    const int max_iter             = (argc > 6) ? std::stoi(argv[6]) : 30;
    const double load_scale_step   = (argc > 7) ? std::stod(argv[7]) : 0.001;
    const std::string matching_arg = (argc > 8) ? argv[8] : "none";
    const std::string pivot_arg    = (argc > 9) ? argv[9] : "shift";
    const double pivot_epsilon     = (argc > 10) ? std::stod(argv[10]) : 1.0e-8;
    std::string jacobian_arg       = (argc > 11) ? argv[11] : "edge";
    if (const char* j = std::getenv("CUPF_BENCH_JACOBIAN")) jacobian_arg = j;

    const auto data = cupf::tests::load_dump_case(case_dir);
    const int32_t n = data.rows;

    NewtonOptions opts;
    opts.compute = spec.compute;
    if (spec.cpu) {
        // CPU backend: per-operator timing (incl. Jacobian) still flows through the
        // ENABLE_TIMING collector; cuda_jacobian / custom.* are ignored here.
        opts.backend            = BackendKind::CPU;
        opts.cpu_linear_solver  = spec.cpu_solver;
    } else {
        opts.backend            = BackendKind::CUDA;
        opts.cuda_linear_solver = spec.backend;
    }
    if (jacobian_arg == "edge")              opts.cuda_jacobian = CudaJacobianKind::Edge;
    else if (jacobian_arg == "edge_atomic")  opts.cuda_jacobian = CudaJacobianKind::EdgeAtomic;
    else if (jacobian_arg == "vertex_warp" || jacobian_arg == "vertex")
                                             opts.cuda_jacobian = CudaJacobianKind::VertexWarp;
    else throw std::invalid_argument("jacobian must be edge, edge_atomic, vertex_warp, or vertex");

    opts.custom.precision = spec.custom_prec;   // ignored unless backend = Custom
    opts.custom.serial_nd = true;               // deterministic ordering for reproducible benchmarking
    if (matching_arg == "none" || matching_arg == "0")            opts.custom.matching = CustomMatchingMode::None;
    else if (matching_arg == "structural" || matching_arg == "1") opts.custom.matching = CustomMatchingMode::Structural;
    else throw std::invalid_argument("matching must be none|0 or structural|1");
    if (pivot_arg == "none" || pivot_arg == "0")        opts.custom.pivot_strategy = CustomPivotStrategy::None;
    else if (pivot_arg == "shift" || pivot_arg == "1")  opts.custom.pivot_strategy = CustomPivotStrategy::StaticDiagonalShift;
    else throw std::invalid_argument("pivot must be none|0 or shift|1");
    opts.custom.shift_retry_epsilon = pivot_epsilon;

    NRConfig config;
    config.tolerance = 1e-6;
    if (const char* t = std::getenv("CUPF_BENCH_TOL")) config.tolerance = std::stod(t);
    config.max_iter  = max_iter;
    SolveOptions solve_options;

    const char* mode_name = (mode == Mode::Solve) ? "solve"
                          : (mode == Mode::Operators) ? "operators" : "debug";
    std::cout << "case=" << data.case_name << " n_bus=" << n
              << " nnz=" << data.ybus_data.size()
              << " spec=" << argv[2] << " jacobian=" << jacobian_arg
              << " mode=" << mode_name << "\n";
    if (mode == Mode::Solve) {
        std::cout << "B,init_ms,solve_ms,solve_per_sys_ms,converged_count,max_mismatch,max_iterations\n";
    } else if (mode == Mode::Operators) {
        std::cout << "B,solve_total_us,ibus_us,mismatch_us,mnorm_us,jac_us,prep_us,fac_us,"
                     "sol_us,vupd_us,upload_us,download_us,converged_count,max_mismatch,max_iterations\n";
    } else {
        std::cout << "B,system,iterations,converged,final_mismatch\n";
    }

    for (int32_t B : batches) {
        // Replicate sbus/V0 into B contiguous cases (stride = n); a per-case load
        // step makes the systems distinct (step=0 repeats the same load).
        std::vector<std::complex<double>> sbus(static_cast<size_t>(B) * n);
        std::vector<std::complex<double>> v0(static_cast<size_t>(B) * n);
        for (int32_t b = 0; b < B; ++b) {
            const double scale = 1.0 + load_scale_step * b;
            for (int32_t i = 0; i < n; ++i) {
                sbus[static_cast<size_t>(b) * n + i] = data.sbus[i] * scale;
                v0[static_cast<size_t>(b) * n + i]   = data.v0[i];
            }
        }

        // --- initialize() (one-time symbolic setup), timed for solve mode ---
        cudaDeviceSynchronize();
        const auto t_init = std::chrono::steady_clock::now();
        NewtonSolver solver(opts);
        solver.initialize(data.ybus(), data.pv.data(), static_cast<int32_t>(data.pv.size()),
                          data.pq.data(), static_cast<int32_t>(data.pq.size()));
        cudaDeviceSynchronize();
        const double init_ms = ms_since(t_init);

        NRBatchResult result;
        auto run = [&]() {
            solver.solve_batch(data.ybus(), sbus.data(), n, v0.data(), n, B,
                               data.pv.data(), static_cast<int32_t>(data.pv.size()),
                               data.pq.data(), static_cast<int32_t>(data.pq.size()),
                               config, solve_options, result);
        };

        run();  // warmup (also primes deferred first-solve work)

        double solve_ms = 0.0;
        if (mode == Mode::Operators) {
            newton_solver::utils::resetTimingCollector();
            for (int r = 0; r < repeats; ++r) run();
        } else {
            // Clean wall-clock solve time: sync-bracketed, no per-stage collector.
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < repeats; ++r) run();
            cudaDeviceSynchronize();
            solve_ms = ms_since(t0) / repeats;
        }

        // --- convergence summary (common to all modes) ---
        int32_t converged_count = 0, max_iterations = 0;
        double max_mismatch = 0.0;
        for (uint8_t c : result.converged)       converged_count += (c != 0);
        for (double m : result.final_mismatch)   max_mismatch = std::max(max_mismatch, m);
        for (int32_t it : result.iterations)     max_iterations = std::max(max_iterations, it);

        if (mode == Mode::Solve) {
            std::cout << B << "," << init_ms << "," << solve_ms << ","
                      << (solve_ms / B) << "," << converged_count << ","
                      << max_mismatch << "," << max_iterations << "\n";
        } else if (mode == Mode::Operators) {
            const auto snap = newton_solver::utils::timingSnapshot();
            auto us = [&](const char* name) -> double {
                for (const auto& e : snap)
                    if (e.name == name) return static_cast<double>(e.total_us) / repeats;
                return 0.0;
            };
            std::cout << B << "," << us("NR.solve.total") << "," << us("NR.iteration.ibus") << ","
                      << us("NR.iteration.mismatch") << "," << us("NR.iteration.mismatch_norm") << ","
                      << us("NR.iteration.jacobian") << "," << us("NR.iteration.prepare_rhs") << ","
                      << us("NR.iteration.factorize") << "," << us("NR.iteration.solve") << ","
                      << us("NR.iteration.voltage_update") << "," << us("NR.solve.upload") << ","
                      << us("NR.solve.download") << "," << converged_count << ","
                      << max_mismatch << "," << max_iterations << "\n";
        } else {  // Mode::Debug — per-system convergence/residual
            for (int32_t b = 0; b < B; ++b) {
                std::cout << B << "," << b << ","
                          << (b < static_cast<int32_t>(result.iterations.size()) ? result.iterations[b] : -1) << ","
                          << (b < static_cast<int32_t>(result.converged.size()) ? int(result.converged[b]) : -1) << ","
                          << (b < static_cast<int32_t>(result.final_mismatch.size()) ? result.final_mismatch[b] : -1.0)
                          << "\n";
            }
        }
    }
    return 0;
}
