#include "dump_case_loader.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/ops/jacobian/jacobian_analysis.hpp"
#include "newton_solver/core/newton_solver.hpp"
#include "newton_solver/ops/jacobian/fill_jacobian.hpp"
#include "newton_solver/ops/linear_solve/cpu_klu.hpp"
#include "newton_solver/ops/mismatch/cpu_mismatch.hpp"
#include "newton_solver/ops/voltage_update/cpu_voltage_update.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "utils/dump.hpp"
#include "utils/logger.hpp"
#include "utils/timer.hpp"

#ifdef CUPF_WITH_SUPERLU
  #include "newton_solver/reference/cpu_naive_jacobian_f64.hpp"
  #include "newton_solver/reference/cpu_superlu_solve.hpp"
#endif

#ifdef CUPF_WITH_CUDA
  #include "newton_solver/ops/jacobian/fill_jacobian_gpu.hpp"
  #include "newton_solver/ops/linear_solve/cuda_cudss.hpp"
  #include "newton_solver/ops/mismatch/cuda_mismatch.hpp"
  #include "newton_solver/ops/voltage_update/cuda_voltage_update.hpp"
  #include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
  #include <cuda_runtime.h>
#endif

#include <Eigen/Sparse>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path case_dir = "/workspace/datasets/cuPF_benchmark_dumps/case30_ieee";
    std::string profile = "cpu_fp64_edge";
    double tolerance = 1e-8;
    int32_t max_iter = 50;
    int32_t warmup = 0;
    int32_t repeats = 1;
    int32_t batch_size = 1;
    bool dump_residuals = false;
    std::filesystem::path dump_dir = "residuals";
    CuDSSOptions cudss;
};

struct ProfileConfig {
    std::string profile;
    std::string implementation;
    std::string backend;
    std::string compute;
    std::string jacobian;
    bool reference_pypowerlike = false;
    bool hybrid_cuda_mixed_edge = false;
    bool modified_schedule = false;
    bool use_cpu_naive_jacobian = false;
    bool use_cpu_superlu = false;
    NewtonOptions options;
};

struct BenchRun {
    bool success = false;
    int32_t iterations = 0;
    double final_mismatch = 0.0;
    double initialize_sec = 0.0;
    double solve_sec = 0.0;
    double total_sec = 0.0;
    double max_v_delta_from_v0 = 0.0;
    std::vector<newton_solver::utils::TimingEntry> timing_entries;
};

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0
        << " [--case-dir PATH]"
        << " [--profile cpp_pypowerlike|cpu_fp64_edge|cuda_mixed_edge|cuda_mixed_edge_modified|cuda_fp64_edge"
        << "|cuda_mixed_edge_cpu_naive_jacobian|cuda_mixed_edge_cpu_superlu]"
        << " [--tolerance FLOAT] [--max-iter INT] [--warmup INT] [--repeats INT] [--batch-size INT]"
        << " [--cudss-use-matching] [--cudss-matching-alg DEFAULT|ALG_1|ALG_2|ALG_3|ALG_4|ALG_5]"
        << " [--cudss-pivot-epsilon AUTO|FLOAT]"
        << " [--dump-residuals|--dump-newton-diagnostics] [--dump-dir PATH]\n";
}

std::string uppercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    return value;
}

CuDSSAlgorithm parse_cudss_algorithm(const std::string& value)
{
    const std::string normalized = uppercase(value);
    if (normalized == "DEFAULT") {
        return CuDSSAlgorithm::Default;
    }
    if (normalized == "ALG_1") {
        return CuDSSAlgorithm::Alg1;
    }
    if (normalized == "ALG_2") {
        return CuDSSAlgorithm::Alg2;
    }
    if (normalized == "ALG_3") {
        return CuDSSAlgorithm::Alg3;
    }
    if (normalized == "ALG_4") {
        return CuDSSAlgorithm::Alg4;
    }
    if (normalized == "ALG_5") {
        return CuDSSAlgorithm::Alg5;
    }
    throw std::runtime_error("Unknown cuDSS algorithm: " + value);
}

void set_cudss_pivot_epsilon(CuDSSOptions& options, const std::string& value)
{
    if (uppercase(value) == "AUTO") {
        options.auto_pivot_epsilon = true;
        options.pivot_epsilon = 0.0;
        return;
    }

    const double parsed = std::stod(value);
    if (!std::isfinite(parsed) || parsed < 0.0) {
        throw std::runtime_error("--cudss-pivot-epsilon must be AUTO or a non-negative finite value");
    }
    options.auto_pivot_epsilon = false;
    options.pivot_epsilon = parsed;
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-dir" && i + 1 < argc) {
            options.case_dir = argv[++i];
        } else if (arg == "--profile" && i + 1 < argc) {
            options.profile = argv[++i];
        } else if (arg == "--tolerance" && i + 1 < argc) {
            options.tolerance = std::stod(argv[++i]);
        } else if (arg == "--max-iter" && i + 1 < argc) {
            options.max_iter = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--warmup" && i + 1 < argc) {
            options.warmup = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--repeats" && i + 1 < argc) {
            options.repeats = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--batch-size" && i + 1 < argc) {
            options.batch_size = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--dump-residuals" || arg == "--dump-newton-diagnostics") {
            options.dump_residuals = true;
        } else if (arg == "--dump-dir" && i + 1 < argc) {
            options.dump_dir = argv[++i];
        } else if (arg == "--cudss-use-matching" || arg == "--cudss-matching") {
            options.cudss.use_matching = true;
        } else if (arg == "--cudss-matching-alg" && i + 1 < argc) {
            options.cudss.matching_alg = parse_cudss_algorithm(argv[++i]);
        } else if ((arg == "--cudss-pivot-epsilon" || arg == "--cudss-epsilon") && i + 1 < argc) {
            set_cudss_pivot_epsilon(options.cudss, argv[++i]);
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
    if (options.batch_size <= 0) {
        throw std::runtime_error("batch-size must be > 0");
    }

    return options;
}

std::filesystem::path repeat_dump_dir(const std::filesystem::path& root, int32_t repeat)
{
    std::ostringstream name;
    name << "repeat_" << std::setw(2) << std::setfill('0') << repeat;
    return root / name.str();
}

ProfileConfig parse_profile(const std::string& profile)
{
    ProfileConfig cfg;
    cfg.profile = profile;

    if (profile == "cpp_pypowerlike") {
#ifndef CUPF_WITH_SUPERLU
        throw std::runtime_error("cpp_pypowerlike requires SuperLU support");
#else
        cfg.reference_pypowerlike = true;
        cfg.implementation = "cpp_pypowerlike";
        cfg.backend = "cpu";
        cfg.compute = "fp64";
        cfg.jacobian = "pypower_like";
        cfg.options.backend = BackendKind::CPU;
        cfg.options.compute = ComputePolicy::FP64;
        return cfg;
#endif
    }

    cfg.implementation = profile;
    if (profile == "cpu_fp64_edge") {
        cfg.backend = "cpu";
        cfg.compute = "fp64";
        cfg.jacobian = "edge_based";
        cfg.options.backend = BackendKind::CPU;
        cfg.options.compute = ComputePolicy::FP64;
        return cfg;
    }
    if (profile == "cuda_mixed_edge") {
        cfg.backend = "cuda";
        cfg.compute = "mixed";
        cfg.jacobian = "edge_based";
        cfg.options.backend = BackendKind::CUDA;
        cfg.options.compute = ComputePolicy::Mixed;
        return cfg;
    }
    if (profile == "cuda_edge_modified" || profile == "cuda_mixed_edge_modified") {
        cfg.profile = "cuda_mixed_edge_modified";
        cfg.implementation = "cuda_edge_modified";
        cfg.backend = "cuda";
        cfg.compute = "mixed";
        cfg.jacobian = "edge_based";
        cfg.modified_schedule = true;
        cfg.options.backend = BackendKind::CUDA;
        cfg.options.compute = ComputePolicy::Mixed;
        return cfg;
    }
    if (profile == "cuda_mixed_edge_cpu_naive_jacobian") {
#if !defined(CUPF_WITH_CUDA) || !defined(CUPF_WITH_SUPERLU)
        throw std::runtime_error("cuda_mixed_edge_cpu_naive_jacobian requires CUDA and SuperLU support");
#else
        cfg.hybrid_cuda_mixed_edge = true;
        cfg.use_cpu_naive_jacobian = true;
        cfg.implementation = "cuda_wo_jacobian";
        cfg.backend = "cuda";
        cfg.compute = "mixed";
        cfg.jacobian = "cpu_naive_pypower_like";
        cfg.options.backend = BackendKind::CUDA;
        cfg.options.compute = ComputePolicy::Mixed;
        return cfg;
#endif
    }
    if (profile == "cuda_mixed_edge_cpu_superlu") {
#if !defined(CUPF_WITH_CUDA) || !defined(CUPF_WITH_SUPERLU)
        throw std::runtime_error("cuda_mixed_edge_cpu_superlu requires CUDA and SuperLU support");
#else
        cfg.hybrid_cuda_mixed_edge = true;
        cfg.use_cpu_superlu = true;
        cfg.implementation = "cuda_wo_cudss";
        cfg.backend = "cuda";
        cfg.compute = "mixed";
        cfg.jacobian = "edge_based";
        cfg.options.backend = BackendKind::CUDA;
        cfg.options.compute = ComputePolicy::Mixed;
        return cfg;
#endif
    }
    if (profile == "cuda_fp64_edge") {
        cfg.backend = "cuda";
        cfg.compute = "fp64";
        cfg.jacobian = "edge_based";
        cfg.options.backend = BackendKind::CUDA;
        cfg.options.compute = ComputePolicy::FP64;
        return cfg;
    }
    throw std::runtime_error("Unknown profile: " + profile);
}

double max_voltage_delta(const std::vector<std::complex<double>>& lhs,
                         const std::vector<std::complex<double>>& rhs)
{
    double max_delta = 0.0;
    const size_t n = std::min(lhs.size(), rhs.size());
    for (size_t i = 0; i < n; ++i) {
        max_delta = std::max(max_delta, std::abs(lhs[i] - rhs[i]));
    }
    return max_delta;
}

std::vector<std::complex<double>> repeat_complex_vector(const std::vector<std::complex<double>>& src,
                                                        int32_t batch_size)
{
    std::vector<std::complex<double>> dst(
        static_cast<std::size_t>(batch_size) * src.size());
    for (int32_t b = 0; b < batch_size; ++b) {
        std::copy(src.begin(),
                  src.end(),
                  dst.begin() + static_cast<std::ptrdiff_t>(b) *
                                  static_cast<std::ptrdiff_t>(src.size()));
    }
    return dst;
}

void validate_basic_result(const cupf::tests::DumpCaseData& data, const NRResult& result)
{
    if (static_cast<int32_t>(result.V.size()) != data.rows) {
        throw std::runtime_error("Result voltage size does not match case size");
    }
    if (!std::isfinite(result.final_mismatch)) {
        throw std::runtime_error("Final mismatch is not finite");
    }
}

void validate_basic_batch_result(const cupf::tests::DumpCaseData& data,
                                 const NRBatchResult& result,
                                 int32_t batch_size)
{
    if (result.n_bus != data.rows || result.batch_size != batch_size) {
        throw std::runtime_error("Batch result metadata does not match benchmark input");
    }
    if (result.V.size() != static_cast<std::size_t>(batch_size) *
                           static_cast<std::size_t>(data.rows)) {
        throw std::runtime_error("Batch result voltage size does not match benchmark input");
    }
    if (result.final_mismatch.size() != static_cast<std::size_t>(batch_size)) {
        throw std::runtime_error("Batch result mismatch vector size does not match batch size");
    }
    for (double value : result.final_mismatch) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Batch result contains a non-finite mismatch");
        }
    }
}

bool all_batch_items_converged(const NRBatchResult& result)
{
    if (result.converged.size() != static_cast<std::size_t>(result.batch_size)) {
        return false;
    }
    return std::all_of(result.converged.begin(), result.converged.end(),
                       [](uint8_t value) { return value != 0; });
}

int32_t max_batch_iterations(const NRBatchResult& result)
{
    if (result.iterations.empty()) {
        return 0;
    }
    return *std::max_element(result.iterations.begin(), result.iterations.end());
}

double max_batch_mismatch(const NRBatchResult& result)
{
    double max_value = 0.0;
    for (double value : result.final_mismatch) {
        max_value = std::max(max_value, std::abs(value));
    }
    return max_value;
}

double max_batched_voltage_delta_from_v0(const cupf::tests::DumpCaseData& data,
                                         const NRBatchResult& result)
{
    double max_delta = 0.0;
    for (int32_t b = 0; b < result.batch_size; ++b) {
        const std::size_t batch_base =
            static_cast<std::size_t>(b) * static_cast<std::size_t>(result.n_bus);
        for (int32_t bus = 0; bus < result.n_bus; ++bus) {
            const std::size_t idx = batch_base + static_cast<std::size_t>(bus);
            max_delta = std::max(max_delta,
                                 std::abs(result.V[idx] -
                                          data.v0[static_cast<std::size_t>(bus)]));
        }
    }
    return max_delta;
}

struct JacobianAnalysisResult {
    JacobianScatterMap maps;
    JacobianPattern pattern;
};

JacobianAnalysisResult run_jacobian_analysis(const YbusView& ybus,
                                             const int32_t* pv,
                                             int32_t n_pv,
                                             const int32_t* pq,
                                             int32_t n_pq)
{
    const JacobianIndexing indexing =
        make_jacobian_indexing(ybus.rows, pv, n_pv, pq, n_pq);
    JacobianAnalysisResult result;
    result.pattern = JacobianPatternGenerator().generate(ybus, indexing);
    result.maps = JacobianMapBuilder().build(ybus, indexing, result.pattern);
    return result;
}

#ifdef CUPF_WITH_CUDA
void synchronize_if_cuda(const ProfileConfig& profile)
{
    if (profile.backend == "cuda") {
        cudaDeviceSynchronize();
    }
}
#else
void synchronize_if_cuda(const ProfileConfig&) {}
#endif

BenchRun run_core_once(const cupf::tests::DumpCaseData& case_data,
                       const ProfileConfig& profile,
                       const NRConfig& config,
                       int32_t batch_size)
{
    if (batch_size != 1 && !(profile.backend == "cuda" && profile.compute == "mixed")) {
        throw std::runtime_error("--batch-size > 1 is currently supported only by CUDA mixed profiles");
    }

    const YbusView ybus = case_data.ybus();
    NewtonSolver solver(profile.options);

    newton_solver::utils::resetTimingCollector();

    const auto t0 = std::chrono::steady_clock::now();
    solver.initialize(
        ybus,
        case_data.pv.data(), static_cast<int32_t>(case_data.pv.size()),
        case_data.pq.data(), static_cast<int32_t>(case_data.pq.size()));
    synchronize_if_cuda(profile);
    const auto t1 = std::chrono::steady_clock::now();

    std::vector<std::complex<double>> batched_sbus;
    std::vector<std::complex<double>> batched_v0;
    const std::complex<double>* sbus = case_data.sbus.data();
    const std::complex<double>* v0 = case_data.v0.data();
    if (batch_size > 1) {
        batched_sbus = repeat_complex_vector(case_data.sbus, batch_size);
        batched_v0 = repeat_complex_vector(case_data.v0, batch_size);
        sbus = batched_sbus.data();
        v0 = batched_v0.data();
    }

    NRBatchResult result;
    solver.solve_batch(
        ybus,
        sbus,
        ybus.rows,
        v0,
        ybus.rows,
        batch_size,
        case_data.pv.data(), static_cast<int32_t>(case_data.pv.size()),
        case_data.pq.data(), static_cast<int32_t>(case_data.pq.size()),
        config,
        result);
    synchronize_if_cuda(profile);
    const auto t2 = std::chrono::steady_clock::now();

    validate_basic_batch_result(case_data, result, batch_size);

    BenchRun run;
    run.final_mismatch = max_batch_mismatch(result);
    run.success = all_batch_items_converged(result) && run.final_mismatch <= config.tolerance;
    run.iterations = max_batch_iterations(result);
    run.initialize_sec =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / 1000000.0;
    run.solve_sec =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000000.0;
    run.total_sec =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count()) / 1000000.0;
    run.max_v_delta_from_v0 = max_batched_voltage_delta_from_v0(case_data, result);
    run.timing_entries = newton_solver::utils::timingSnapshot();
    std::sort(run.timing_entries.begin(), run.timing_entries.end(),
              [](const auto& lhs, const auto& rhs) {
                  return std::string_view(lhs.name) < std::string_view(rhs.name);
              });
    return run;
}

#ifdef CUPF_WITH_CUDA
BenchRun run_cuda_mixed_modified_once(const cupf::tests::DumpCaseData& case_data,
                                      const ProfileConfig& profile,
                                      const NRConfig& config)
{
    if (profile.backend != "cuda" || profile.compute != "mixed") {
        throw std::runtime_error("modified benchmark path currently supports CUDA mixed profiles only");
    }

    const YbusView ybus = case_data.ybus();
    const int32_t n_pv = static_cast<int32_t>(case_data.pv.size());
    const int32_t n_pq = static_cast<int32_t>(case_data.pq.size());

    CudaMixedStorage storage;
    CudaMismatchOp mismatch(storage);
    CudaJacobianOp<float> jacobian(storage);
    CudaLinearSolveCuDSS<float, CudaMixedStorage> linear_solve(storage, profile.options.cudss);
    CudaVoltageUpdateOp<float> voltage_update(storage);

    newton_solver::utils::resetTimingCollector();

    JacobianAnalysisResult analysis;

    const auto t0 = std::chrono::steady_clock::now();
    {
        newton_solver::utils::ScopedTimer total("NR.initialize.total");

        {
            newton_solver::utils::ScopedTimer stage("NR.initialize.jacobian_analysis");
            analysis = run_jacobian_analysis(
                ybus,
                case_data.pv.data(), n_pv,
                case_data.pq.data(), n_pq);
        }

        InitializeContext initialize_ctx{
            .ybus = ybus,
            .maps = analysis.maps,
            .J = analysis.pattern,
            .n_bus = ybus.rows,
            .pv = case_data.pv.data(),
            .n_pv = n_pv,
            .pq = case_data.pq.data(),
            .n_pq = n_pq,
        };

        {
            newton_solver::utils::ScopedTimer stage("NR.initialize.storage_prepare");
            storage.prepare(initialize_ctx);
        }
        {
            newton_solver::utils::ScopedTimer stage("NR.initialize.linear_solve_initialize");
            linear_solve.initialize(initialize_ctx);
        }
    }
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();

    SolveContext solve_ctx{
        .ybus = &ybus,
        .sbus = case_data.sbus.data(),
        .V0 = case_data.v0.data(),
        .config = &config,
        .batch_size = 1,
        .sbus_stride = ybus.rows,
        .V0_stride = ybus.rows,
        .ybus_values_batched = false,
        .ybus_value_stride = ybus.nnz,
    };

    NRResult result;
    int32_t completed = 0;

    {
        newton_solver::utils::ScopedTimer total("NR.solve.total");

        {
            newton_solver::utils::ScopedTimer stage("NR.solve.upload");
            storage.upload(solve_ctx);
        }

        IterationContext iter_ctx{
            .storage = storage,
            .config = config,
            .pv = case_data.pv.data(),
            .n_pv = n_pv,
            .pq = case_data.pq.data(),
            .n_pq = n_pq,
        };

        int32_t last_jacobian_iter = -1;
        int32_t solves_since_factorize = 2;

        for (int32_t iter = 0; iter < config.max_iter; ++iter) {
            newton_solver::utils::ScopedTimer iter_total("NR.iteration.total");
            iter_ctx.iter = iter;
            iter_ctx.jacobian_updated_this_iter = false;
            iter_ctx.jacobian_age =
                (last_jacobian_iter >= 0) ? (iter - last_jacobian_iter) : 0;
            completed = iter + 1;

            {
                newton_solver::utils::ScopedTimer stage("NR.iteration.mismatch");
                mismatch.run(iter_ctx);
            }
            if (iter_ctx.converged) {
                break;
            }

            const bool refresh_jacobian =
                (last_jacobian_iter < 0) || (solves_since_factorize >= 2);
            if (refresh_jacobian) {
                {
                    newton_solver::utils::ScopedTimer stage("NR.iteration.jacobian");
                    jacobian.run(iter_ctx);
                }
                iter_ctx.jacobian_updated_this_iter = true;
                iter_ctx.jacobian_age = 0;
                last_jacobian_iter = iter;
                solves_since_factorize = 0;

                {
                    newton_solver::utils::ScopedTimer stage("NR.iteration.linear_factorize");
                    linear_solve.factorize(iter_ctx);
                }
            } else {
                iter_ctx.jacobian_age = iter - last_jacobian_iter;
            }

            {
                newton_solver::utils::ScopedTimer stage("NR.iteration.linear_solve");
                linear_solve.solve(iter_ctx);
            }
            {
                newton_solver::utils::ScopedTimer stage("NR.iteration.voltage_update");
                voltage_update.run(iter_ctx);
            }
            ++solves_since_factorize;
        }

        result.iterations = completed;
        result.converged = iter_ctx.converged;
        result.final_mismatch = iter_ctx.normF;

        {
            newton_solver::utils::ScopedTimer stage("NR.solve.download");
            storage.download_result(result);
        }
    }
    cudaDeviceSynchronize();
    const auto t2 = std::chrono::steady_clock::now();

    validate_basic_result(case_data, result);

    BenchRun run;
    run.success = result.converged && result.final_mismatch <= config.tolerance;
    run.iterations = result.iterations;
    run.final_mismatch = result.final_mismatch;
    run.initialize_sec =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / 1000000.0;
    run.solve_sec =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000000.0;
    run.total_sec =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count()) / 1000000.0;
    run.max_v_delta_from_v0 = max_voltage_delta(case_data.v0, result.V);
    run.timing_entries = newton_solver::utils::timingSnapshot();
    std::sort(run.timing_entries.begin(), run.timing_entries.end(),
              [](const auto& lhs, const auto& rhs) {
                  return std::string_view(lhs.name) < std::string_view(rhs.name);
              });
    return run;
}
#endif

BenchRun run_reference_pypowerlike_once(const cupf::tests::DumpCaseData& case_data,
                                        const ProfileConfig& profile,
                                        const NRConfig& config)
{
    (void)profile;

#ifndef CUPF_WITH_SUPERLU
    throw std::runtime_error("cpp_pypowerlike requires SuperLU support");
#else
    const YbusView ybus = case_data.ybus();

    CpuFp64Storage storage;
    CpuMismatchOp mismatch(storage);
    CpuNaiveJacobianOpF64 jacobian(storage);
    CpuLinearSolveSuperLU linear_solve(storage);
    CpuVoltageUpdateOp voltage_update(storage);

    JacobianScatterMap maps;
    JacobianPattern J;
    InitializeContext initialize_ctx{
        .ybus = ybus,
        .maps = maps,
        .J = J,
        .n_bus = ybus.rows,
        .pv = case_data.pv.data(),
        .n_pv = static_cast<int32_t>(case_data.pv.size()),
        .pq = case_data.pq.data(),
        .n_pq = static_cast<int32_t>(case_data.pq.size()),
    };

    newton_solver::utils::resetTimingCollector();

    const auto t0 = std::chrono::steady_clock::now();
    storage.prepare(initialize_ctx);
    linear_solve.initialize(initialize_ctx);
    const auto t1 = std::chrono::steady_clock::now();

    SolveContext solve_ctx{
        .ybus = &ybus,
        .sbus = case_data.sbus.data(),
        .V0 = case_data.v0.data(),
        .config = &config,
        .batch_size = 1,
        .sbus_stride = ybus.rows,
        .V0_stride = ybus.rows,
        .ybus_values_batched = false,
        .ybus_value_stride = ybus.nnz,
    };
    storage.upload(solve_ctx);

    IterationContext iter_ctx{
        .storage = storage,
        .config = config,
        .pv = case_data.pv.data(),
        .n_pv = static_cast<int32_t>(case_data.pv.size()),
        .pq = case_data.pq.data(),
        .n_pq = static_cast<int32_t>(case_data.pq.size()),
    };

    int32_t completed = 0;
    for (int32_t iter = 0; iter < config.max_iter; ++iter) {
        iter_ctx.iter = iter;
        completed = iter + 1;

        mismatch.run(iter_ctx);
        if (iter_ctx.converged) {
            break;
        }
        jacobian.run(iter_ctx);
        linear_solve.factorize(iter_ctx);
        linear_solve.solve(iter_ctx);
        voltage_update.run(iter_ctx);
    }

    NRResult result;
    result.iterations = completed;
    result.final_mismatch = iter_ctx.normF;
    result.converged = iter_ctx.converged;
    storage.download_result(result);
    const auto t2 = std::chrono::steady_clock::now();

    validate_basic_result(case_data, result);

    BenchRun run;
    run.success = result.converged && result.final_mismatch <= config.tolerance;
    run.iterations = result.iterations;
    run.final_mismatch = result.final_mismatch;
    run.initialize_sec =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / 1000000.0;
    run.solve_sec =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000000.0;
    run.total_sec =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count()) / 1000000.0;
    run.max_v_delta_from_v0 = max_voltage_delta(case_data.v0, result.V);
    run.timing_entries = newton_solver::utils::timingSnapshot();
    std::sort(run.timing_entries.begin(), run.timing_entries.end(),
              [](const auto& lhs, const auto& rhs) {
                  return std::string_view(lhs.name) < std::string_view(rhs.name);
              });
    return run;
#endif
}

#if defined(CUPF_WITH_CUDA) && defined(CUPF_WITH_SUPERLU)
void copy_cuda_voltage_to_cpu(const CudaMixedStorage& cuda_storage,
                              CpuFp64Storage& cpu_storage)
{
    newton_solver::utils::ScopedTimer timer("ABLATION.cuda_to_cpu_voltage");

    const int32_t n_bus = cuda_storage.n_bus;
    if (n_bus != cpu_storage.n_bus) {
        throw std::runtime_error("Hybrid ablation voltage copy: CUDA/CPU bus counts differ");
    }

    std::vector<double> h_va(static_cast<std::size_t>(n_bus));
    std::vector<double> h_vm(static_cast<std::size_t>(n_bus));
    cuda_storage.d_Va.copyTo(h_va.data(), h_va.size());
    cuda_storage.d_Vm.copyTo(h_vm.data(), h_vm.size());

    for (int32_t bus = 0; bus < n_bus; ++bus) {
        const double va = h_va[static_cast<std::size_t>(bus)];
        const double vm = h_vm[static_cast<std::size_t>(bus)];
        const std::complex<double> value(vm * std::cos(va), vm * std::sin(va));
        cpu_storage.V[static_cast<std::size_t>(bus)] = value;
        cpu_storage.Va[static_cast<std::size_t>(bus)] = va;
        cpu_storage.Vm[static_cast<std::size_t>(bus)] = vm;
        cpu_storage.Ibus[static_cast<std::size_t>(bus)] = std::complex<double>(0.0, 0.0);
    }
    cpu_storage.has_cached_Ibus = false;
}

void copy_cpu_jacobian_to_cuda_mixed(const CpuFp64Storage& cpu_storage,
                                     CudaMixedStorage& cuda_storage,
                                     const JacobianPattern& J)
{
    newton_solver::utils::ScopedTimer timer("ABLATION.cpu_jacobian_to_cuda_csr");

    if (J.dim != cpu_storage.dimF || J.nnz != cuda_storage.nnz_J) {
        throw std::runtime_error("Hybrid ablation Jacobian copy: CPU/CUDA Jacobian dimensions differ");
    }

    std::vector<float> h_values(static_cast<std::size_t>(J.nnz), 0.0f);
    for (int32_t row = 0; row < J.dim; ++row) {
        for (int32_t pos = J.row_ptr[static_cast<std::size_t>(row)];
             pos < J.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = J.col_idx[static_cast<std::size_t>(pos)];
            h_values[static_cast<std::size_t>(pos)] =
                static_cast<float>(cpu_storage.J.coeff(row, col));
        }
    }
    cuda_storage.d_J_values.assign(h_values.data(), h_values.size());
}

void copy_cuda_mismatch_to_cpu(const CudaMixedStorage& cuda_storage,
                               CpuFp64Storage& cpu_storage)
{
    newton_solver::utils::ScopedTimer timer("ABLATION.cuda_F_to_cpu");

    if (cuda_storage.dimF != cpu_storage.dimF) {
        throw std::runtime_error("Hybrid ablation mismatch copy: CUDA/CPU dimensions differ");
    }
    cuda_storage.d_F.copyTo(cpu_storage.F.data(), cpu_storage.F.size());
}

void copy_cuda_jacobian_to_cpu_csc(const CudaMixedStorage& cuda_storage,
                                   CpuFp64Storage& cpu_storage,
                                   const JacobianPattern& J)
{
    newton_solver::utils::ScopedTimer timer("ABLATION.cuda_J_to_cpu_csc");

    if (J.dim != cpu_storage.dimF || J.nnz != cuda_storage.nnz_J) {
        throw std::runtime_error("Hybrid ablation Jacobian copy: CUDA/CPU Jacobian dimensions differ");
    }

    std::vector<float> h_values(static_cast<std::size_t>(J.nnz));
    cuda_storage.d_J_values.copyTo(h_values.data(), h_values.size());

    using RealTriplet = Eigen::Triplet<double, int32_t>;
    std::vector<RealTriplet> trips;
    trips.reserve(static_cast<std::size_t>(J.nnz));
    for (int32_t row = 0; row < J.dim; ++row) {
        for (int32_t pos = J.row_ptr[static_cast<std::size_t>(row)];
             pos < J.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            trips.emplace_back(
                row,
                J.col_idx[static_cast<std::size_t>(pos)],
                static_cast<double>(h_values[static_cast<std::size_t>(pos)]));
        }
    }

    cpu_storage.J.resize(J.dim, J.dim);
    cpu_storage.J.setFromTriplets(trips.begin(), trips.end());
    cpu_storage.J.makeCompressed();
}

void copy_cpu_dx_to_cuda_mixed(const CpuFp64Storage& cpu_storage,
                               CudaMixedStorage& cuda_storage)
{
    newton_solver::utils::ScopedTimer timer("ABLATION.cpu_dx_to_cuda");

    if (cuda_storage.dimF != cpu_storage.dimF) {
        throw std::runtime_error("Hybrid ablation dx copy: CUDA/CPU dimensions differ");
    }

    std::vector<float> h_dx(cpu_storage.dx.size());
    for (std::size_t i = 0; i < cpu_storage.dx.size(); ++i) {
        h_dx[i] = static_cast<float>(cpu_storage.dx[i]);
    }
    cuda_storage.d_dx.assign(h_dx.data(), h_dx.size());
}

BenchRun run_hybrid_cuda_mixed_edge_once(const cupf::tests::DumpCaseData& case_data,
                                         const ProfileConfig& profile,
                                         const NRConfig& config)
{
    if (!profile.use_cpu_naive_jacobian && !profile.use_cpu_superlu) {
        throw std::runtime_error("Hybrid CUDA mixed edge profile has no CPU ablation operator selected");
    }

    const YbusView ybus = case_data.ybus();
    const int32_t n_pv = static_cast<int32_t>(case_data.pv.size());
    const int32_t n_pq = static_cast<int32_t>(case_data.pq.size());

    CudaMixedStorage cuda_storage;
    CudaMismatchOp mismatch(cuda_storage);
    CudaJacobianOp<float> cuda_jacobian(cuda_storage);
    CudaLinearSolveCuDSS<float, CudaMixedStorage> cudss(cuda_storage, profile.options.cudss);
    CudaVoltageUpdateOp<float> voltage_update(cuda_storage);

    CpuFp64Storage cpu_storage;
    CpuNaiveJacobianOpF64 cpu_naive_jacobian(cpu_storage);
    CpuLinearSolveSuperLU cpu_superlu(cpu_storage);

    newton_solver::utils::resetTimingCollector();

    JacobianAnalysisResult analysis;
    JacobianScatterMap cpu_empty_maps;
    JacobianPattern cpu_empty_jacobian;

    const auto t0 = std::chrono::steady_clock::now();
    {
        newton_solver::utils::ScopedTimer total("NR.initialize.total");

        {
            newton_solver::utils::ScopedTimer stage("NR.initialize.jacobian_analysis");
            analysis = run_jacobian_analysis(
                ybus,
                case_data.pv.data(), n_pv,
                case_data.pq.data(), n_pq);
        }

        InitializeContext cuda_initialize_ctx{
            .ybus = ybus,
            .maps = analysis.maps,
            .J = analysis.pattern,
            .n_bus = ybus.rows,
            .pv = case_data.pv.data(),
            .n_pv = n_pv,
            .pq = case_data.pq.data(),
            .n_pq = n_pq,
        };
        InitializeContext cpu_initialize_ctx{
            .ybus = ybus,
            .maps = cpu_empty_maps,
            .J = cpu_empty_jacobian,
            .n_bus = ybus.rows,
            .pv = case_data.pv.data(),
            .n_pv = n_pv,
            .pq = case_data.pq.data(),
            .n_pq = n_pq,
        };

        {
            newton_solver::utils::ScopedTimer stage("NR.initialize.storage_prepare");
            cuda_storage.prepare(cuda_initialize_ctx);
            cpu_storage.prepare(cpu_initialize_ctx);
        }
        {
            newton_solver::utils::ScopedTimer stage("NR.initialize.linear_solve_initialize");
            if (profile.use_cpu_superlu) {
                cpu_superlu.initialize(cpu_initialize_ctx);
            } else {
                cudss.initialize(cuda_initialize_ctx);
            }
        }
    }
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();

    SolveContext solve_ctx{
        .ybus = &ybus,
        .sbus = case_data.sbus.data(),
        .V0 = case_data.v0.data(),
        .config = &config,
        .batch_size = 1,
        .sbus_stride = ybus.rows,
        .V0_stride = ybus.rows,
        .ybus_values_batched = false,
        .ybus_value_stride = ybus.nnz,
    };

    NRResult result;
    int32_t completed = 0;

    {
        newton_solver::utils::ScopedTimer total("NR.solve.total");

        {
            newton_solver::utils::ScopedTimer stage("NR.solve.upload");
            cuda_storage.upload(solve_ctx);
            if (profile.use_cpu_naive_jacobian) {
                cpu_storage.upload(solve_ctx);
            }
        }

        IterationContext cuda_ctx{
            .storage = cuda_storage,
            .config = config,
            .pv = case_data.pv.data(),
            .n_pv = n_pv,
            .pq = case_data.pq.data(),
            .n_pq = n_pq,
        };
        IterationContext cpu_ctx{
            .storage = cpu_storage,
            .config = config,
            .pv = case_data.pv.data(),
            .n_pv = n_pv,
            .pq = case_data.pq.data(),
            .n_pq = n_pq,
        };

        for (int32_t iter = 0; iter < config.max_iter; ++iter) {
            newton_solver::utils::ScopedTimer iter_total("NR.iteration.total");
            cuda_ctx.iter = iter;
            cpu_ctx.iter = iter;
            cuda_ctx.jacobian_updated_this_iter = false;
            cpu_ctx.jacobian_updated_this_iter = false;
            completed = iter + 1;

            {
                newton_solver::utils::ScopedTimer stage("NR.iteration.mismatch");
                mismatch.run(cuda_ctx);
            }
            cpu_ctx.normF = cuda_ctx.normF;
            cpu_ctx.converged = cuda_ctx.converged;
            if (cuda_ctx.converged) {
                break;
            }

            {
                newton_solver::utils::ScopedTimer stage("NR.iteration.jacobian");
                if (profile.use_cpu_naive_jacobian) {
                    copy_cuda_voltage_to_cpu(cuda_storage, cpu_storage);
                    {
                        newton_solver::utils::ScopedTimer cpu_stage("ABLATION.cpu_naive_jacobian");
                        cpu_naive_jacobian.run(cpu_ctx);
                    }
                    copy_cpu_jacobian_to_cuda_mixed(cpu_storage, cuda_storage, analysis.pattern);
                } else {
                    cuda_jacobian.run(cuda_ctx);
                }
                cuda_ctx.jacobian_updated_this_iter = true;
                cuda_ctx.jacobian_age = 0;
                cpu_ctx.jacobian_updated_this_iter = true;
                cpu_ctx.jacobian_age = 0;
            }
            {
                newton_solver::utils::ScopedTimer stage("NR.iteration.linear_solve");
                if (profile.use_cpu_superlu) {
                    copy_cuda_mismatch_to_cpu(cuda_storage, cpu_storage);
                    copy_cuda_jacobian_to_cpu_csc(cuda_storage, cpu_storage, analysis.pattern);
                    cpu_superlu.factorize(cpu_ctx);
                    cpu_superlu.solve(cpu_ctx);
                    copy_cpu_dx_to_cuda_mixed(cpu_storage, cuda_storage);
                } else {
                    cudss.factorize(cuda_ctx);
                    cudss.solve(cuda_ctx);
                }
            }
            {
                newton_solver::utils::ScopedTimer stage("NR.iteration.voltage_update");
                voltage_update.run(cuda_ctx);
            }
        }

        result.iterations = completed;
        result.converged = cuda_ctx.converged;
        result.final_mismatch = cuda_ctx.normF;

        {
            newton_solver::utils::ScopedTimer stage("NR.solve.download");
            cuda_storage.download_result(result);
        }
    }
    cudaDeviceSynchronize();
    const auto t2 = std::chrono::steady_clock::now();

    validate_basic_result(case_data, result);

    BenchRun run;
    run.success = result.converged && result.final_mismatch <= config.tolerance;
    run.iterations = result.iterations;
    run.final_mismatch = result.final_mismatch;
    run.initialize_sec =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / 1000000.0;
    run.solve_sec =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000000.0;
    run.total_sec =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count()) / 1000000.0;
    run.max_v_delta_from_v0 = max_voltage_delta(case_data.v0, result.V);
    run.timing_entries = newton_solver::utils::timingSnapshot();
    std::sort(run.timing_entries.begin(), run.timing_entries.end(),
              [](const auto& lhs, const auto& rhs) {
                  return std::string_view(lhs.name) < std::string_view(rhs.name);
              });
    return run;
}
#endif

BenchRun run_once(const cupf::tests::DumpCaseData& case_data,
                  const ProfileConfig& profile,
                  const NRConfig& config,
                  int32_t batch_size)
{
    if (profile.reference_pypowerlike) {
        if (batch_size != 1) {
            throw std::runtime_error("cpp_pypowerlike benchmark path supports only --batch-size 1");
        }
        return run_reference_pypowerlike_once(case_data, profile, config);
    }
    if (profile.modified_schedule) {
        if (batch_size != 1) {
            throw std::runtime_error("modified benchmark paths support only --batch-size 1");
        }
#ifdef CUPF_WITH_CUDA
        return run_cuda_mixed_modified_once(case_data, profile, config);
#else
        throw std::runtime_error("modified CUDA profiles require CUDA support");
#endif
    }
    if (profile.hybrid_cuda_mixed_edge) {
        if (batch_size != 1) {
            throw std::runtime_error("hybrid ablation benchmark paths support only --batch-size 1");
        }
#if defined(CUPF_WITH_CUDA) && defined(CUPF_WITH_SUPERLU)
        return run_hybrid_cuda_mixed_edge_once(case_data, profile, config);
#else
        throw std::runtime_error("Hybrid CUDA mixed edge profiles require CUDA and SuperLU support");
#endif
    }
    return run_core_once(case_data, profile, config, batch_size);
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        newton_solver::utils::initLogger(newton_solver::utils::LogLevel::Off, false);

        const CliOptions cli = parse_args(argc, argv);
        const auto case_data = cupf::tests::load_dump_case(cli.case_dir);
        ProfileConfig profile = parse_profile(cli.profile);
        profile.options.cudss = cli.cudss;
        const NRConfig config{cli.tolerance, cli.max_iter};

        newton_solver::utils::setDumpEnabled(false);
        for (int32_t i = 0; i < cli.warmup; ++i) {
            (void)run_once(case_data, profile, config, cli.batch_size);
        }

        std::cout << std::boolalpha << std::scientific << std::setprecision(12);
        for (int32_t repeat = 0; repeat < cli.repeats; ++repeat) {
            if (cli.dump_residuals) {
                newton_solver::utils::setDumpDirectory(repeat_dump_dir(cli.dump_dir, repeat).string());
                newton_solver::utils::setDumpEnabled(true);
            }
            const BenchRun run = run_once(case_data, profile, config, cli.batch_size);
            newton_solver::utils::setDumpEnabled(false);

            std::cout
                << "RUN "
                << "case=" << case_data.case_name << ' '
                << "profile=" << profile.profile << ' '
                << "implementation=" << profile.implementation << ' '
                << "backend=" << profile.backend << ' '
                << "compute=" << profile.compute << ' '
                << "jacobian=" << profile.jacobian << ' '
                << "repeat=" << repeat << ' '
                << "batch_size=" << cli.batch_size << ' '
                << "success=" << run.success << ' '
                << "iterations=" << run.iterations << ' '
                << "final_mismatch=" << run.final_mismatch << ' '
                << "initialize_sec=" << run.initialize_sec << ' '
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
                    (entry.count > 0) ? total_sec / static_cast<double>(entry.count) : 0.0;
                std::cout
                    << "METRIC "
                    << "repeat=" << repeat << ' '
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
