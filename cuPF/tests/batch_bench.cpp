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

// Parsed "<backend>-<precision>" spec, e.g. "custom-tf32" / "cudss-fp64". A bare precision
// ("fp64") defaults to the cuDSS backend (back-compat with the old <compute> arg).
struct BenchSpec {
    CudaLinearSolverKind backend     = CudaLinearSolverKind::CuDSS;
    ComputePolicy        compute     = ComputePolicy::Mixed;
    CustomPrecision      custom_prec = CustomPrecision::FP32;
};

// Grammar: [<backend>-]<compute>[-<custom_precision>]
//   backend          : cudss | custom            (default cudss)
//   compute          : fp64 | fp32 | mixed        (cuPF ComputePolicy)
//   custom_precision : fp64 | fp32 | tf32         (custom factor precision; default fp32)
// e.g. cudss-fp64, custom-fp64, cudss-mixed, custom-mixed-fp32, custom-mixed-tf32.
BenchSpec parse_spec(const std::string& s)
{
    std::vector<std::string> t;
    for (size_t pos = 0; pos <= s.size();) {
        size_t d = s.find('-', pos);
        if (d == std::string::npos) d = s.size();
        t.push_back(s.substr(pos, d - pos));
        pos = d + 1;
    }
    BenchSpec spec;
    size_t idx = 0;
    if (!t.empty() && (t[0] == "cudss" || t[0] == "custom")) {
        spec.backend = (t[0] == "custom") ? CudaLinearSolverKind::Custom : CudaLinearSolverKind::CuDSS;
        idx = 1;
    }
    const std::string compute = (idx < t.size()) ? t[idx] : "mixed";
    const std::string cprec   = (idx + 1 < t.size()) ? t[idx + 1] : "";
    if      (compute == "fp64") { spec.compute = ComputePolicy::FP64;  spec.custom_prec = CustomPrecision::FP64; }
    else if (compute == "fp32") { spec.compute = ComputePolicy::FP32;  spec.custom_prec = CustomPrecision::FP32; }
    else if (compute == "tf32") { spec.compute = ComputePolicy::FP32;  spec.custom_prec = CustomPrecision::TF32; }
    else                        { spec.compute = ComputePolicy::Mixed; spec.custom_prec = CustomPrecision::FP32; }
    if      (cprec == "fp64") spec.custom_prec = CustomPrecision::FP64;
    else if (cprec == "fp32") spec.custom_prec = CustomPrecision::FP32;
    else if (cprec == "tf32") spec.custom_prec = CustomPrecision::TF32;
    return spec;
}

}  // namespace

int main(int argc, char** argv)
{
    // args: <case_dir> <backend-precision> <B1,B2,...> [repeats] [max_iter] [scale_step]
    if (argc < 4) {
        std::cerr << "usage: batch_bench <case_dir> <backend-precision> <B1,B2,...> [repeats] [max_iter] [scale_step]\n"
                  << "  backend-precision: cudss-fp64|cudss-fp32|cudss-mixed|custom-fp64|custom-fp32|custom-tf32\n";
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
    const int repeats              = (argc > 4) ? std::stoi(argv[4]) : 5;
    const int max_iter             = (argc > 5) ? std::stoi(argv[5]) : 30;
    const double load_scale_step   = (argc > 6) ? std::stod(argv[6]) : 0.001;

    const auto data = cupf::tests::load_dump_case(case_dir);
    const int32_t n = data.rows;

    NewtonOptions opts;
    opts.backend            = BackendKind::CUDA;
    opts.compute            = spec.compute;
    opts.cuda_linear_solver = spec.backend;
    opts.custom.precision   = spec.custom_prec;   // ignored unless backend = Custom
    opts.custom.serial_nd   = true;               // deterministic ordering for reproducible benchmarking
    opts.custom.metis_seed  = 1588;               // (matches the standalone sweep)

    NRConfig config;
    config.tolerance = 1e-6;
    if (const char* t = std::getenv("CUPF_BENCH_TOL")) config.tolerance = std::stod(t);
    config.max_iter  = max_iter;
    SolveOptions solve_options;

    std::cout << "case=" << data.case_name << " n_bus=" << n
              << " nnz=" << data.ybus_data.size()
              << " spec=" << argv[2] << "\n";
    std::cout << "B,solve_total_us,ibus_us,mismatch_us,mnorm_us,jac_us,prep_us,fac_us,sol_us,vupd_us,upload_us,download_us\n";

    for (int32_t B : batches) {
        // Replicate sbus/V0 into B contiguous cases (stride = n). A configurable
        // per-case load step can make cases distinct; step=0 repeats the same load.
        std::vector<std::complex<double>> sbus(static_cast<size_t>(B) * n);
        std::vector<std::complex<double>> v0(static_cast<size_t>(B) * n);
        for (int32_t b = 0; b < B; ++b) {
            const double scale = 1.0 + load_scale_step * b;
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
