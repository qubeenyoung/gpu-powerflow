#include "dump_case_loader.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/ops/jacobian/jacobian_analysis.hpp"
#include "newton_solver/core/newton_solver.hpp"
#include "newton_solver/ops/ibus/compute_ibus.hpp"
#include "newton_solver/ops/jacobian/fill_jacobian.hpp"
#include "newton_solver/ops/linear_solve/cpu_klu.hpp"
#include "newton_solver/ops/mismatch/cpu_mismatch.hpp"
#include "newton_solver/ops/voltage_update/cpu_voltage_update.hpp"
#include "newton_solver/reference/cpu_naive_jacobian_f64.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "utils/dump.hpp"
#include "utils/logger.hpp"
#include "utils/timer.hpp"

#ifdef CUPF_WITH_CUDA
  #include <cuda_runtime.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
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
#include <thread>
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
    bool sample_gpu_memory = false;
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
        << " [--profile cpp_pypowerlike|cpu_fp64_edge|cuda_fp32_edge|cuda_mixed_edge|cuda_fp64_edge]"
        << " [--tolerance FLOAT] [--max-iter INT] [--warmup INT] [--repeats INT] [--batch-size INT]"
        << " [--cudss-use-matching] [--cudss-matching-alg DEFAULT|ALG_1|ALG_2|ALG_3|ALG_4|ALG_5]"
        << " [--cudss-pivot-epsilon AUTO|FLOAT]"
        << " [--dump-residuals|--dump-newton-diagnostics] [--dump-dir PATH]"
        << " [--sample-gpu-memory]\n";
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
        } else if (arg == "--sample-gpu-memory") {
            options.sample_gpu_memory = true;
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

#ifdef CUPF_WITH_CUDA
class GpuMemorySampler {
public:
    struct Snapshot {
        std::size_t total_bytes = 0;
        std::size_t initial_free_bytes = 0;
        std::size_t final_free_bytes = 0;
        std::size_t min_free_bytes = 0;
        std::uint64_t samples = 0;
        std::uint64_t errors = 0;
    };

    explicit GpuMemorySampler(std::chrono::microseconds period)
        : period_(period)
    {
        cudaError_t set_device_status = cudaSetDevice(0);
        if (set_device_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaSetDevice failed for memory sampler: ") +
                                     cudaGetErrorString(set_device_status));
        }

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        cudaError_t mem_status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (mem_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemGetInfo failed for memory sampler: ") +
                                     cudaGetErrorString(mem_status));
        }

        total_bytes_.store(total_bytes, std::memory_order_relaxed);
        initial_free_bytes_ = free_bytes;
        final_free_bytes_.store(free_bytes, std::memory_order_relaxed);
        min_free_bytes_.store(free_bytes, std::memory_order_relaxed);
        worker_ = std::thread([this]() { sample_loop(); });
    }

    ~GpuMemorySampler()
    {
        stop();
    }

    void stop()
    {
        const bool already_stopped = stopped_.exchange(true, std::memory_order_acq_rel);
        if (!already_stopped && worker_.joinable()) {
            worker_.join();
        }
    }

    Snapshot snapshot() const
    {
        Snapshot s;
        s.total_bytes = total_bytes_.load(std::memory_order_relaxed);
        s.initial_free_bytes = initial_free_bytes_;
        s.final_free_bytes = final_free_bytes_.load(std::memory_order_relaxed);
        s.min_free_bytes = min_free_bytes_.load(std::memory_order_relaxed);
        s.samples = samples_.load(std::memory_order_relaxed);
        s.errors = errors_.load(std::memory_order_relaxed);
        return s;
    }

private:
    void sample_loop()
    {
        while (!stopped_.load(std::memory_order_acquire)) {
            sample_once();
            std::this_thread::sleep_for(period_);
        }
        sample_once();
    }

    void sample_once()
    {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (status != cudaSuccess) {
            errors_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        total_bytes_.store(total_bytes, std::memory_order_relaxed);
        final_free_bytes_.store(free_bytes, std::memory_order_relaxed);
        std::size_t current = min_free_bytes_.load(std::memory_order_relaxed);
        while (free_bytes < current &&
               !min_free_bytes_.compare_exchange_weak(current,
                                                       free_bytes,
                                                       std::memory_order_relaxed,
                                                       std::memory_order_relaxed)) {
        }
        samples_.fetch_add(1, std::memory_order_relaxed);
    }

    std::chrono::microseconds period_;
    std::thread worker_;
    std::atomic<bool> stopped_{false};
    std::atomic<std::size_t> total_bytes_{0};
    std::size_t initial_free_bytes_ = 0;
    std::atomic<std::size_t> final_free_bytes_{0};
    std::atomic<std::size_t> min_free_bytes_{0};
    std::atomic<std::uint64_t> samples_{0};
    std::atomic<std::uint64_t> errors_{0};
};
#endif

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
        cfg.reference_pypowerlike = true;
        cfg.implementation = "cpp_pypowerlike";
        cfg.backend = "cpu";
        cfg.compute = "fp64";
        cfg.jacobian = "pypower_like";
        cfg.options.backend = BackendKind::CPU;
        cfg.options.compute = ComputePolicy::FP64;
        return cfg;
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
    if (profile == "cuda_fp32_edge") {
        cfg.backend = "cuda";
        cfg.compute = "fp32";
        cfg.jacobian = "edge_based";
        cfg.options.backend = BackendKind::CUDA;
        cfg.options.compute = ComputePolicy::FP32;
        return cfg;
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
    if (batch_size != 1 &&
        !(profile.backend == "cuda" && (profile.compute == "mixed" || profile.compute == "fp32"))) {
        throw std::runtime_error("--batch-size > 1 is currently supported only by CUDA mixed/fp32 profiles");
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

BenchRun run_reference_pypowerlike_once(const cupf::tests::DumpCaseData& case_data,
                                        const ProfileConfig& profile,
                                        const NRConfig& config)
{
    (void)profile;
    const YbusView ybus = case_data.ybus();

    CpuFp64Buffers storage;
    CpuIbusOp ibus;
    CpuMismatchOp mismatch;
    CpuMismatchNormOp mismatch_norm;
    CpuNaiveJacobianOpF64 jacobian;
    CpuLinearSolveKLU linear_solve;
    CpuVoltageUpdateOp voltage_update;

    JacobianScatterMap maps;
    const JacobianIndexing indexing =
        make_jacobian_indexing(ybus.rows,
                               case_data.pv.data(),
                               static_cast<int32_t>(case_data.pv.size()),
                               case_data.pq.data(),
                               static_cast<int32_t>(case_data.pq.size()));
    const JacobianPattern J = JacobianPatternGenerator().generate(ybus, indexing);
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
    linear_solve.initialize(storage, initialize_ctx);
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
        .config = config,
        .pv = case_data.pv.data(),
        .n_pv = static_cast<int32_t>(case_data.pv.size()),
        .pq = case_data.pq.data(),
        .n_pq = static_cast<int32_t>(case_data.pq.size()),
    };

    int32_t completed = 0;
    for (int32_t iter = 0; iter < config.max_iter; ++iter) {
        newton_solver::utils::ScopedTimer iteration_timer("NR.iteration.total");
        iter_ctx.iter = iter;
        completed = iter + 1;

        { newton_solver::utils::ScopedTimer timer("NR.iteration.ibus"); ibus.run(storage, iter_ctx); }
        { newton_solver::utils::ScopedTimer timer("NR.iteration.mismatch"); mismatch.run(storage, iter_ctx); }
        { newton_solver::utils::ScopedTimer timer("NR.iteration.mismatch_norm"); mismatch_norm.run(storage, iter_ctx); }
        if (iter_ctx.converged) {
            break;
        }
        { newton_solver::utils::ScopedTimer timer("NR.iteration.jacobian"); jacobian.run(storage, iter_ctx); }
        { newton_solver::utils::ScopedTimer timer("NR.iteration.prepare_rhs"); linear_solve.prepare_rhs(storage, iter_ctx); }
        { newton_solver::utils::ScopedTimer timer("NR.iteration.factorize"); linear_solve.factorize(storage, iter_ctx); }
        { newton_solver::utils::ScopedTimer timer("NR.iteration.solve"); linear_solve.solve(storage, iter_ctx); }
        { newton_solver::utils::ScopedTimer timer("NR.iteration.voltage_update"); voltage_update.run(storage, iter_ctx); }
    }

    NRResult result;
    result.iterations = completed;
    result.final_mismatch = iter_ctx.normF;
    result.converged = iter_ctx.converged;
    storage.download(result);
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

#ifdef CUPF_WITH_CUDA
        std::unique_ptr<GpuMemorySampler> gpu_memory_sampler;
        if (cli.sample_gpu_memory) {
            if (profile.backend != "cuda") {
                throw std::runtime_error("--sample-gpu-memory requires a CUDA profile");
            }
            gpu_memory_sampler = std::make_unique<GpuMemorySampler>(std::chrono::microseconds(500));
        }
#else
        if (cli.sample_gpu_memory) {
            throw std::runtime_error("--sample-gpu-memory requires a CUDA build");
        }
#endif

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
                << "analyze_sec=" << run.initialize_sec << ' '
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

#ifdef CUPF_WITH_CUDA
        if (gpu_memory_sampler) {
            gpu_memory_sampler->stop();
            const GpuMemorySampler::Snapshot memory = gpu_memory_sampler->snapshot();
            const double mib = 1024.0 * 1024.0;
            const double peak_device_used_mib =
                static_cast<double>(memory.total_bytes - memory.min_free_bytes) / mib;
            const double peak_delta_mib =
                static_cast<double>(memory.initial_free_bytes - memory.min_free_bytes) / mib;
            std::cout
                << "GPU_MEMORY "
                << "batch_size=" << cli.batch_size << ' '
                << "total_mib=" << static_cast<double>(memory.total_bytes) / mib << ' '
                << "initial_free_mib=" << static_cast<double>(memory.initial_free_bytes) / mib << ' '
                << "final_free_mib=" << static_cast<double>(memory.final_free_bytes) / mib << ' '
                << "min_free_mib=" << static_cast<double>(memory.min_free_bytes) / mib << ' '
                << "peak_device_used_mib=" << peak_device_used_mib << ' '
                << "peak_delta_mib=" << peak_delta_mib << ' '
                << "samples=" << memory.samples << ' '
                << "errors=" << memory.errors
                << "\n";
        }
#endif

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "cupf_case_benchmark failed: " << ex.what() << "\n";
        return 1;
    }
}
