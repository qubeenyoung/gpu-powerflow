// graph_bench: end-to-end wall-clock benchmark of solve_batch for the cuPF Mixed profile across
// linear-solver backends, including the CUDA-graph custom path. Unlike batch_bench it does NOT use
// the per-stage timing collector (that injects cudaDeviceSynchronize per stage, which is incompatible
// with CUDA-graph capture); it just times the whole solve_batch with warmup + averaging. solve_batch
// is synchronous on return (it downloads results), so chrono captures the full GPU time.
//
//   usage: graph_bench <case_dir> <cudss|custom|custom_graph> <B1,B2,...> [repeats]
//
// backend:
//   cudss        - cuda_linear_solver=CuDSS
//   custom       - cuda_linear_solver=Custom, use_cuda_graph=false
//   custom_graph - cuda_linear_solver=Custom, use_cuda_graph=true (needs a CUPF_ENABLE_CUDA_GRAPH build)
#include "dump_case_loader.hpp"
#include "newton_solver/core/newton_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "usage: graph_bench <case_dir> <cudss|custom|custom_graph> <B1,B2,...> [repeats]\n";
        return 2;
    }
    const std::string case_dir = argv[1];
    const std::string backend = argv[2];

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
    const int repeats = (argc > 4) ? std::stoi(argv[4]) : 10;

    NewtonOptions opts;
    opts.backend = BackendKind::CUDA;
    opts.compute = ComputePolicy::Mixed;   // user request: Mixed profile
    if (backend == "cudss") {
        opts.cuda_linear_solver = CudaLinearSolverKind::CuDSS;
    } else if (backend == "custom") {
        opts.cuda_linear_solver = CudaLinearSolverKind::Custom;
    } else if (backend == "custom_graph") {
        opts.cuda_linear_solver = CudaLinearSolverKind::Custom;
        opts.use_cuda_graph = true;
    } else {
        std::cerr << "unknown backend: " << backend << "\n";
        return 2;
    }

    const auto data = cupf::tests::load_dump_case(case_dir);
    const int32_t n = data.rows;

    NRConfig config;
    config.tolerance = 1e-8;  // project default; override with CUPF_BENCH_TOL
    if (const char* t = std::getenv("CUPF_BENCH_TOL")) config.tolerance = std::stod(t);
    config.max_iter = 30;
    SolveOptions solve_options;

    std::cout << "case=" << data.case_name << " n_bus=" << n
              << " nnz=" << data.ybus_data.size() << " backend=" << backend
              << " tol=" << config.tolerance << "\n";
    std::cout << "B,init_ms,warmup_ms,solve_ms,ms_per_sys,iters,relmis\n";

    for (int32_t B : batches) {
        std::vector<std::complex<double>> sbus(static_cast<size_t>(B) * n);
        std::vector<std::complex<double>> v0(static_cast<size_t>(B) * n);
        for (int32_t b = 0; b < B; ++b) {
            const double scale = 1.0 + 0.001 * b;  // distinct-yet-convergent per-case load
            for (int32_t i = 0; i < n; ++i) {
                sbus[static_cast<size_t>(b) * n + i] = data.sbus[i] * scale;
                v0[static_cast<size_t>(b) * n + i] = data.v0[i];
            }
        }

        NewtonSolver solver(opts);
        // initialize(): one-time symbolic analysis (Jacobian analysis + the linear solver's
        // analyze()). Measured separately from solve.
        const auto ti0 = std::chrono::steady_clock::now();
        solver.initialize(data.ybus(), data.pv.data(), static_cast<int32_t>(data.pv.size()),
                          data.pq.data(), static_cast<int32_t>(data.pq.size()));
        const auto ti1 = std::chrono::steady_clock::now();
        const double init_ms = std::chrono::duration<double, std::milli>(ti1 - ti0).count();

        NRBatchResult result;
        auto run = [&]() {
            solver.solve_batch(data.ybus(), sbus.data(), n, v0.data(), n, B,
                               data.pv.data(), static_cast<int32_t>(data.pv.size()),
                               data.pq.data(), static_cast<int32_t>(data.pq.size()),
                               config, solve_options, result);
        };

        // First solve carries the one-time per-batch setup: custom's batched_setup (arena alloc +
        // internal-graph capture) or, for custom_graph, graph_prepare + the iteration-graph capture.
        const auto tw0 = std::chrono::steady_clock::now();
        run();
        const auto tw1 = std::chrono::steady_clock::now();
        const double warmup_ms = std::chrono::duration<double, std::milli>(tw1 - tw0).count();

        double best_ms = 1e300, sum_ms = 0.0;
        for (int r = 0; r < repeats; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            run();
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            best_ms = std::min(best_ms, ms);
            sum_ms += ms;
        }
        const double avg_ms = sum_ms / repeats;

        double relmis = 0.0;
        for (double m : result.final_mismatch) relmis = std::max(relmis, m);
        const int iters = result.iterations.empty() ? -1 : result.iterations[0];

        // init_ms = one-time analyze; warmup_ms = first solve (incl. one-time setup/capture);
        // solve_ms = steady-state solve (avg); setup_ms ~= warmup_ms - solve_ms (one-time per-batch).
        std::cout << B << "," << init_ms << "," << warmup_ms << "," << avg_ms << ","
                  << (avg_ms / B) << "," << iters << "," << relmis
                  << "   (best_solve_ms=" << best_ms << ", setup~=" << (warmup_ms - avg_ms) << ")\n";
    }
    return 0;
}
