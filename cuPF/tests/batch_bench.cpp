// Batch scaling benchmark: replicate one dump case into a B-case batch and
// run solve_batch, printing per-stage timing so the ibus(spmm) vs linear-solve
// balance can be observed as B grows. Built only with BUILD_EVALUATORS + timing.
#include "dump_case_loader.hpp"
#include "newton_solver/core/newton_solver.hpp"
#include "utils/timer.hpp"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

ComputePolicy parse_compute(const std::string& s)
{
    if (s == "fp32") return ComputePolicy::FP32;
    if (s == "mixed") return ComputePolicy::Mixed;
    return ComputePolicy::FP64;
}

}  // namespace

int main(int argc, char** argv)
{
    // args: <case_dir> <compute=mixed|fp32> <B1,B2,...> [repeats]
    if (argc < 4) {
        std::cerr << "usage: batch_bench <case_dir> <compute> <B1,B2,...> [repeats]\n";
        return 2;
    }
    const std::string case_dir = argv[1];
    const ComputePolicy compute = parse_compute(argv[2]);
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
    const int repeats = (argc > 4) ? std::stoi(argv[4]) : 5;

    const auto data = cupf::tests::load_dump_case(case_dir);
    const int32_t n = data.rows;

    NewtonOptions opts;
    opts.backend = BackendKind::CUDA;
    opts.compute = compute;
    // CUPF_BENCH_CUSTOM=1 benches the custom direct solver (FP64 compute only; its internal factor
    // precision is set by CUPF_CUSTOM_PRECISION). Default is cuDSS.
    opts.cuda_linear_solver = (std::getenv("CUPF_BENCH_CUSTOM") != nullptr)
                                  ? CudaLinearSolverKind::Custom
                                  : CudaLinearSolverKind::CuDSS;

    NRConfig config;
    config.tolerance = 1e-6;
    config.max_iter = 30;
    SolveOptions solve_options;

    std::cout << "case=" << data.case_name << " n_bus=" << n
              << " nnz=" << data.ybus_data.size()
              << " compute=" << argv[2] << "\n";
    std::cout << "B,solve_total_us,ibus_us,mismatch_us,mnorm_us,jac_us,prep_us,fac_us,sol_us,vupd_us,upload_us,download_us\n";

    for (int32_t B : batches) {
        // Replicate sbus/V0 into B contiguous cases (stride = n), with a tiny
        // per-case load scaling so the systems are distinct yet converge.
        std::vector<std::complex<double>> sbus(static_cast<size_t>(B) * n);
        std::vector<std::complex<double>> v0(static_cast<size_t>(B) * n);
        for (int32_t b = 0; b < B; ++b) {
            const double scale = 1.0 + 0.001 * b;
            for (int32_t i = 0; i < n; ++i) {
                sbus[static_cast<size_t>(b) * n + i] = data.sbus[i] * scale;
                v0[static_cast<size_t>(b) * n + i] = data.v0[i];
            }
        }

        NewtonSolver solver(opts);
        solver.initialize(data.ybus(), data.pv.data(), static_cast<int32_t>(data.pv.size()),
                          data.pq.data(), static_cast<int32_t>(data.pq.size()));

        NRBatchResult result;
        auto run = [&]() {
            solver.solve_batch(data.ybus(), sbus.data(), n, v0.data(), n, B,
                               data.pv.data(), static_cast<int32_t>(data.pv.size()),
                               data.pq.data(), static_cast<int32_t>(data.pq.size()),
                               config, solve_options, result);
        };

        run();  // warmup
        newton_solver::utils::resetTimingCollector();
        for (int r = 0; r < repeats; ++r) run();
        const auto snap = newton_solver::utils::timingSnapshot();

        auto us = [&](const char* name) -> double {
            for (const auto& e : snap) {
                if (e.name == name) return static_cast<double>(e.total_us) / repeats;
            }
            return 0.0;
        };

        std::cout << B << ","
                  << us("NR.solve.total") << ","
                  << us("NR.iteration.ibus") << ","
                  << us("NR.iteration.mismatch") << ","
                  << us("NR.iteration.mismatch_norm") << ","
                  << us("NR.iteration.jacobian") << ","
                  << us("NR.iteration.prepare_rhs") << ","
                  << us("NR.iteration.factorize") << ","
                  << us("NR.iteration.solve") << ","
                  << us("NR.iteration.voltage_update") << ","
                  << us("NR.solve.upload") << ","
                  << us("NR.solve.download")
                  << "   (iters=" << (result.iterations.empty() ? -1 : result.iterations[0]) << ")\n";
    }
    return 0;
}
