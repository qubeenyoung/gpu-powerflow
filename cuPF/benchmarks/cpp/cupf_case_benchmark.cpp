#include "dump_case_loader.hpp"

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/core/jacobian_builder.hpp"
#include "newton_solver/core/newton_solver.hpp"
#include "newton_solver/ops/jacobian/cpu_f64.hpp"
#include "newton_solver/ops/linear_solve/cpu_klu.hpp"
#include "newton_solver/ops/mismatch/cpu_f64.hpp"
#include "newton_solver/ops/voltage_update/cpu_f64.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "utils/dump.hpp"
#include "utils/logger.hpp"
#include "utils/timer.hpp"

#ifdef CUPF_WITH_SUPERLU
  #include "newton_solver/reference/cpu_naive_jacobian_f64.hpp"
  #include "newton_solver/reference/cpu_superlu_solve.hpp"
#endif

#ifdef CUPF_WITH_CUDA
  #include "newton_solver/ops/jacobian/cuda_edge_fp32.hpp"
  #include "newton_solver/ops/linear_solve/cuda_cudss32.hpp"
  #include "newton_solver/ops/mismatch/cuda_f64.hpp"
  #include "newton_solver/ops/voltage_update/cuda_mixed.hpp"
  #include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
  #include <cuda_runtime.h>
#endif

#include <Eigen/Sparse>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
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
    bool dump_residuals = false;
    std::filesystem::path dump_dir = "residuals";
};

struct ProfileConfig {
    std::string profile;
    std::string implementation;
    std::string backend;
    std::string compute;
    std::string jacobian;
    std::string algorithm = "standard";
    bool reference_pypowerlike = false;
    bool hybrid_cuda_mixed_edge = false;
    bool use_cpu_naive_jacobian = false;
    bool use_cpu_superlu = false;
    NewtonOptions options;
};

struct BenchRun {
    bool success = false;
    int32_t iterations = 0;
    double final_mismatch = 0.0;
    double analyze_sec = 0.0;
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
        << " [--profile cpp_pypowerlike|cpu_fp64_edge|cuda_mixed_edge|cuda_mixed_vertex|cuda_fp64_edge|cuda_fp64_vertex"
        << "|cuda_mixed_edge_cpu_naive_jacobian|cuda_mixed_edge_cpu_superlu|<core_profile>_modified]"
        << " [--tolerance FLOAT] [--max-iter INT] [--warmup INT] [--repeats INT]"
        << " [--dump-residuals] [--dump-dir PATH]\n";
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
        } else if (arg == "--dump-residuals") {
            options.dump_residuals = true;
        } else if (arg == "--dump-dir" && i + 1 < argc) {
            options.dump_dir = argv[++i];
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

std::filesystem::path repeat_dump_dir(const std::filesystem::path& root, int32_t repeat)
{
    std::ostringstream name;
    name << "repeat_" << std::setw(2) << std::setfill('0') << repeat;
    return root / name.str();
}

bool consume_suffix(std::string& value, std::string_view suffix)
{
    if (value.size() < suffix.size()) {
        return false;
    }
    const auto offset = value.size() - suffix.size();
    if (value.compare(offset, suffix.size(), suffix.data(), suffix.size()) != 0) {
        return false;
    }
    value.erase(offset);
    return true;
}

ProfileConfig parse_profile(const std::string& profile)
{
    ProfileConfig cfg;
    cfg.profile = profile;
    cfg.options.algorithm = NewtonAlgorithm::Standard;

    std::string base_profile = profile;
    const bool modified = consume_suffix(base_profile, "_modified");
    if (modified) {
        cfg.algorithm = "modified";
        cfg.options.algorithm = NewtonAlgorithm::Modified;
    }

    if (base_profile == "cpp_pypowerlike") {
        if (modified) {
            throw std::runtime_error("cpp_pypowerlike_modified is not supported by this reference path");
        }
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
        cfg.options.jacobian_builder = JacobianBuilderType::EdgeBased;
        return cfg;
#endif
    }

    cfg.implementation = profile;
    if (base_profile == "cpu_fp64_edge") {
        cfg.backend = "cpu";
        cfg.compute = "fp64";
        cfg.jacobian = "edge_based";
        cfg.options.backend = BackendKind::CPU;
        cfg.options.compute = ComputePolicy::FP64;
        cfg.options.jacobian_builder = JacobianBuilderType::EdgeBased;
        return cfg;
    }
    if (base_profile == "cuda_mixed_edge") {
        cfg.backend = "cuda";
        cfg.compute = "mixed";
        cfg.jacobian = "edge_based";
        cfg.options.backend = BackendKind::CUDA;
        cfg.options.compute = ComputePolicy::Mixed;
        cfg.options.jacobian_builder = JacobianBuilderType::EdgeBased;
        return cfg;
    }
    if (base_profile == "cuda_mixed_edge_cpu_naive_jacobian") {
        if (modified) {
            throw std::runtime_error("cuda_mixed_edge_cpu_naive_jacobian_modified is not supported");
        }
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
        cfg.options.jacobian_builder = JacobianBuilderType::EdgeBased;
        return cfg;
#endif
    }
    if (base_profile == "cuda_mixed_edge_cpu_superlu") {
        if (modified) {
            throw std::runtime_error("cuda_mixed_edge_cpu_superlu_modified is not supported");
        }
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
        cfg.options.jacobian_builder = JacobianBuilderType::EdgeBased;
        return cfg;
#endif
    }
    if (base_profile == "cuda_mixed_vertex") {
        cfg.backend = "cuda";
        cfg.compute = "mixed";
        cfg.jacobian = "vertex_based";
        cfg.options.backend = BackendKind::CUDA;
        cfg.options.compute = ComputePolicy::Mixed;
        cfg.options.jacobian_builder = JacobianBuilderType::VertexBased;
        return cfg;
    }
    if (base_profile == "cuda_fp64_edge") {
        cfg.backend = "cuda";
        cfg.compute = "fp64";
        cfg.jacobian = "edge_based";
        cfg.options.backend = BackendKind::CUDA;
        cfg.options.compute = ComputePolicy::FP64;
        cfg.options.jacobian_builder = JacobianBuilderType::EdgeBased;
        return cfg;
    }
    if (base_profile == "cuda_fp64_vertex") {
        cfg.backend = "cuda";
        cfg.compute = "fp64";
        cfg.jacobian = "vertex_based";
        cfg.options.backend = BackendKind::CUDA;
        cfg.options.compute = ComputePolicy::FP64;
        cfg.options.jacobian_builder = JacobianBuilderType::VertexBased;
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

void validate_basic_result(const cupf::tests::DumpCaseData& data, const NRResultF64& result)
{
    if (static_cast<int32_t>(result.V.size()) != data.rows) {
        throw std::runtime_error("Result voltage size does not match case size");
    }
    if (!std::isfinite(result.final_mismatch)) {
        throw std::runtime_error("Final mismatch is not finite");
    }
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
                       const NRConfig& config)
{
    const YbusViewF64 ybus = case_data.ybus();
    NewtonSolver solver(profile.options);

    newton_solver::utils::resetTimingCollector();

    const auto t0 = std::chrono::steady_clock::now();
    solver.analyze(
        ybus,
        case_data.pv.data(), static_cast<int32_t>(case_data.pv.size()),
        case_data.pq.data(), static_cast<int32_t>(case_data.pq.size()));
    synchronize_if_cuda(profile);
    const auto t1 = std::chrono::steady_clock::now();

    NRResultF64 result;
    solver.solve(
        ybus,
        case_data.sbus.data(),
        case_data.v0.data(),
        case_data.pv.data(), static_cast<int32_t>(case_data.pv.size()),
        case_data.pq.data(), static_cast<int32_t>(case_data.pq.size()),
        config,
        result);
    synchronize_if_cuda(profile);
    const auto t2 = std::chrono::steady_clock::now();

    validate_basic_result(case_data, result);

    BenchRun run;
    run.success = result.converged && result.final_mismatch <= config.tolerance;
    run.iterations = result.iterations;
    run.final_mismatch = result.final_mismatch;
    run.analyze_sec =
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

BenchRun run_reference_pypowerlike_once(const cupf::tests::DumpCaseData& case_data,
                                        const ProfileConfig& profile,
                                        const NRConfig& config)
{
    (void)profile;

#ifndef CUPF_WITH_SUPERLU
    throw std::runtime_error("cpp_pypowerlike requires SuperLU support");
#else
    const YbusViewF64 ybus = case_data.ybus();

    CpuFp64Storage storage;
    CpuMismatchOpF64 mismatch(storage);
    CpuNaiveJacobianOpF64 jacobian(storage);
    CpuLinearSolveSuperLU linear_solve(storage);
    CpuVoltageUpdateF64 voltage_update(storage);

    JacobianMaps maps;
    JacobianStructure J;
    AnalyzeContext analyze_ctx{
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
    storage.prepare(analyze_ctx);
    linear_solve.analyze(analyze_ctx);
    const auto t1 = std::chrono::steady_clock::now();

    SolveContext solve_ctx{
        .ybus = &ybus,
        .sbus = case_data.sbus.data(),
        .V0 = case_data.v0.data(),
        .config = &config,
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
        linear_solve.run(iter_ctx);
        voltage_update.run(iter_ctx);
    }

    NRResultF64 result;
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
    run.analyze_sec =
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

    std::vector<double> h_re(static_cast<std::size_t>(n_bus));
    std::vector<double> h_im(static_cast<std::size_t>(n_bus));
    cuda_storage.d_V_re.copyTo(h_re.data(), h_re.size());
    cuda_storage.d_V_im.copyTo(h_im.data(), h_im.size());

    for (int32_t bus = 0; bus < n_bus; ++bus) {
        const std::complex<double> value(h_re[static_cast<std::size_t>(bus)],
                                         h_im[static_cast<std::size_t>(bus)]);
        cpu_storage.V[static_cast<std::size_t>(bus)] = value;
        cpu_storage.Va[static_cast<std::size_t>(bus)] = std::arg(value);
        cpu_storage.Vm[static_cast<std::size_t>(bus)] = std::abs(value);
        cpu_storage.Ibus[static_cast<std::size_t>(bus)] = std::complex<double>(0.0, 0.0);
    }
    cpu_storage.has_cached_Ibus = false;
}

void copy_cpu_jacobian_to_cuda_mixed(const CpuFp64Storage& cpu_storage,
                                     CudaMixedStorage& cuda_storage,
                                     const JacobianStructure& J)
{
    newton_solver::utils::ScopedTimer timer("ABLATION.cpu_jacobian_to_cuda_csr");

    if (J.dim != cpu_storage.dimF || J.nnz != static_cast<int32_t>(cuda_storage.d_J_values.size())) {
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
                                   const JacobianStructure& J)
{
    newton_solver::utils::ScopedTimer timer("ABLATION.cuda_J_to_cpu_csc");

    if (J.dim != cpu_storage.dimF || J.nnz != static_cast<int32_t>(cuda_storage.d_J_values.size())) {
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

    const YbusViewF64 ybus = case_data.ybus();
    const int32_t n_pv = static_cast<int32_t>(case_data.pv.size());
    const int32_t n_pq = static_cast<int32_t>(case_data.pq.size());

    CudaMixedStorage cuda_storage;
    CudaMismatchOpF64 mismatch(cuda_storage);
    CudaJacobianOpEdgeFp32 cuda_jacobian(cuda_storage);
    CudaLinearSolveCuDSS32 cudss(cuda_storage);
    CudaVoltageUpdateMixed voltage_update(cuda_storage);

    CpuFp64Storage cpu_storage;
    CpuNaiveJacobianOpF64 cpu_naive_jacobian(cpu_storage);
    CpuLinearSolveSuperLU cpu_superlu(cpu_storage);

    newton_solver::utils::resetTimingCollector();

    JacobianBuilder::Result analysis;
    JacobianMaps cpu_empty_maps;
    JacobianStructure cpu_empty_jacobian;

    const auto t0 = std::chrono::steady_clock::now();
    {
        newton_solver::utils::ScopedTimer total("NR.analyze.total");

        {
            newton_solver::utils::ScopedTimer stage("NR.analyze.jacobian_builder");
            JacobianBuilder builder(JacobianBuilderType::EdgeBased);
            analysis = builder.analyze(
                ybus,
                case_data.pv.data(), n_pv,
                case_data.pq.data(), n_pq);
        }

        AnalyzeContext cuda_analyze_ctx{
            .ybus = ybus,
            .maps = analysis.maps,
            .J = analysis.J,
            .n_bus = ybus.rows,
            .pv = case_data.pv.data(),
            .n_pv = n_pv,
            .pq = case_data.pq.data(),
            .n_pq = n_pq,
        };
        AnalyzeContext cpu_analyze_ctx{
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
            newton_solver::utils::ScopedTimer stage("NR.analyze.storage_prepare");
            cuda_storage.prepare(cuda_analyze_ctx);
            cpu_storage.prepare(cpu_analyze_ctx);
        }
        {
            newton_solver::utils::ScopedTimer stage("NR.analyze.linear_solve");
            if (profile.use_cpu_superlu) {
                cpu_superlu.analyze(cpu_analyze_ctx);
            } else {
                cudss.analyze(cuda_analyze_ctx);
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
    };

    NRResultF64 result;
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
                    copy_cpu_jacobian_to_cuda_mixed(cpu_storage, cuda_storage, analysis.J);
                } else {
                    cuda_jacobian.run(cuda_ctx);
                }
            }
            {
                newton_solver::utils::ScopedTimer stage("NR.iteration.linear_solve");
                if (profile.use_cpu_superlu) {
                    copy_cuda_mismatch_to_cpu(cuda_storage, cpu_storage);
                    copy_cuda_jacobian_to_cpu_csc(cuda_storage, cpu_storage, analysis.J);
                    cpu_superlu.run(cpu_ctx);
                    copy_cpu_dx_to_cuda_mixed(cpu_storage, cuda_storage);
                } else {
                    cudss.run(cuda_ctx);
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
    run.analyze_sec =
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
                  const NRConfig& config)
{
    if (profile.reference_pypowerlike) {
        return run_reference_pypowerlike_once(case_data, profile, config);
    }
    if (profile.hybrid_cuda_mixed_edge) {
#if defined(CUPF_WITH_CUDA) && defined(CUPF_WITH_SUPERLU)
        return run_hybrid_cuda_mixed_edge_once(case_data, profile, config);
#else
        throw std::runtime_error("Hybrid CUDA mixed edge profiles require CUDA and SuperLU support");
#endif
    }
    return run_core_once(case_data, profile, config);
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        newton_solver::utils::initLogger(newton_solver::utils::LogLevel::Off, false);

        const CliOptions cli = parse_args(argc, argv);
        const auto case_data = cupf::tests::load_dump_case(cli.case_dir);
        const ProfileConfig profile = parse_profile(cli.profile);
        const NRConfig config{cli.tolerance, cli.max_iter};

        newton_solver::utils::setDumpEnabled(false);
        for (int32_t i = 0; i < cli.warmup; ++i) {
            (void)run_once(case_data, profile, config);
        }

        std::cout << std::boolalpha << std::scientific << std::setprecision(12);
        for (int32_t repeat = 0; repeat < cli.repeats; ++repeat) {
            if (cli.dump_residuals) {
                newton_solver::utils::setDumpDirectory(repeat_dump_dir(cli.dump_dir, repeat).string());
                newton_solver::utils::setDumpEnabled(true);
            }
            const BenchRun run = run_once(case_data, profile, config);
            newton_solver::utils::setDumpEnabled(false);

            std::cout
                << "RUN "
                << "case=" << case_data.case_name << ' '
                << "profile=" << profile.profile << ' '
                << "implementation=" << profile.implementation << ' '
                << "backend=" << profile.backend << ' '
                << "compute=" << profile.compute << ' '
                << "jacobian=" << profile.jacobian << ' '
                << "algorithm=" << profile.algorithm << ' '
                << "repeat=" << repeat << ' '
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
