// lifecycle_bench: Mixed-profile first-solve benchmark for solver lifecycle timing.
//
//   usage: lifecycle_bench <case_dir> <cudss|custom> <B1,B2,...> [trials]
//
// Each trial creates a fresh NewtonSolver, measures initialize(), then measures the
// first solve_batch(). This intentionally includes one-time per-batch setup in the
// solve time, including cuDSS's deferred work on the first cudssExecute path.
#include "dump_case_loader.hpp"
#include "newton_solver/core/newton_solver.hpp"
#include "utils/timer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int32_t> parse_batches(const std::string& text)
{
    std::vector<int32_t> batches;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t comma = text.find(',', pos);
        if (comma == std::string::npos) comma = text.size();
        batches.push_back(std::stoi(text.substr(pos, comma - pos)));
        pos = comma + 1;
    }
    return batches;
}

void cuda_check(cudaError_t err, const char* where)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "usage: lifecycle_bench <case_dir> <cudss|custom> <B1,B2,...> [trials]\n";
        return 2;
    }

    const std::string case_dir = argv[1];
    const std::string backend = argv[2];
    const std::vector<int32_t> batches = parse_batches(argv[3]);
    const int trials = (argc > 4) ? std::stoi(argv[4]) : 5;

    NewtonOptions opts;
    opts.backend = BackendKind::CUDA;
    opts.compute = ComputePolicy::Mixed;
    if (backend == "cudss") {
        opts.cuda_linear_solver = CudaLinearSolverKind::CuDSS;
    } else if (backend == "custom") {
        opts.cuda_linear_solver = CudaLinearSolverKind::Custom;
    } else {
        std::cerr << "unknown backend: " << backend << "\n";
        return 2;
    }

    const auto data = cupf::tests::load_dump_case(case_dir);
    const int32_t n = data.rows;

    NRConfig config;
    config.tolerance = 1e-8;
    if (const char* t = std::getenv("CUPF_BENCH_TOL")) config.tolerance = std::stod(t);
    config.max_iter = 30;
    SolveOptions solve_options;

    // Keep CUDA context creation out of the measured solver lifecycle.
    cuda_check(cudaFree(nullptr), "cudaFree(nullptr)");

    std::cout << "case,backend,n_bus,ybus_nnz,B,trial,init_ms,init_analyze_ms,solve_ms,solve_analyze_ms,total_ms,ms_per_sys,iters,relmis,converged\n";

    for (int32_t B : batches) {
        std::vector<std::complex<double>> sbus(static_cast<size_t>(B) * n);
        std::vector<std::complex<double>> v0(static_cast<size_t>(B) * n);
        for (int32_t b = 0; b < B; ++b) {
            const double scale = 1.0 + 0.001 * b;
            for (int32_t i = 0; i < n; ++i) {
                sbus[static_cast<size_t>(b) * n + i] = data.sbus[i] * scale;
                v0[static_cast<size_t>(b) * n + i] = data.v0[i];
            }
        }

        for (int trial = 0; trial < trials; ++trial) {
            NewtonSolver solver(opts);

            newton_solver::utils::resetTimingCollector();
            const auto ti0 = std::chrono::steady_clock::now();
            solver.initialize(data.ybus(), data.pv.data(), static_cast<int32_t>(data.pv.size()),
                              data.pq.data(), static_cast<int32_t>(data.pq.size()));
            const auto ti1 = std::chrono::steady_clock::now();
            const auto init_timing = newton_solver::utils::timingSnapshot();

            NRBatchResult result;
            newton_solver::utils::resetTimingCollector();
            const auto ts0 = std::chrono::steady_clock::now();
            solver.solve_batch(data.ybus(), sbus.data(), n, v0.data(), n, B,
                               data.pv.data(), static_cast<int32_t>(data.pv.size()),
                               data.pq.data(), static_cast<int32_t>(data.pq.size()),
                               config, solve_options, result);
            const auto ts1 = std::chrono::steady_clock::now();
            const auto solve_timing = newton_solver::utils::timingSnapshot();

            const double init_ms = std::chrono::duration<double, std::milli>(ti1 - ti0).count();
            const double solve_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
            const double total_ms = init_ms + solve_ms;
            auto timing_ms = [](const std::vector<newton_solver::utils::TimingEntry>& entries,
                                const char* name) {
                for (const auto& entry : entries) {
                    if (std::string(entry.name ? entry.name : "") == name) {
                        return static_cast<double>(entry.total_us) / 1000.0;
                    }
                }
                return 0.0;
            };
            const double init_analyze_ms =
                timing_ms(init_timing, "NR.initialize.cudss_analyze") +
                timing_ms(init_timing, "NR.initialize.custom_analyze");
            const double solve_analyze_ms = timing_ms(solve_timing, "NR.iteration.cudss_analyze");

            double relmis = 0.0;
            for (double m : result.final_mismatch) relmis = std::max(relmis, m);

            int iters = -1;
            for (int value : result.iterations) iters = std::max(iters, value);
            const bool all_converged =
                std::all_of(result.converged.begin(), result.converged.end(),
                            [](uint8_t value) { return value != 0; });

            std::cout << data.case_name << "," << backend << "," << n << ","
                      << data.ybus_data.size() << "," << B << "," << trial << ","
                      << init_ms << "," << init_analyze_ms << ","
                      << solve_ms << "," << solve_analyze_ms << "," << total_ms << ","
                      << (solve_ms / B) << "," << iters << "," << relmis << ","
                      << (all_converged ? 1 : 0) << "\n";
        }
    }

    return 0;
}
