#include "linear_system_io.hpp"

#include "dump_case_loader.hpp"

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/core/jacobian_builder.hpp"
#include "newton_solver/ops/jacobian/cpu_f64.hpp"
#include "newton_solver/ops/linear_solve/cpu_klu.hpp"
#include "newton_solver/ops/mismatch/cpu_f64.hpp"
#include "newton_solver/ops/voltage_update/cpu_f64.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "utils/logger.hpp"

#ifdef CUPF_WITH_CUDA
  #include "newton_solver/ops/jacobian/cuda_edge_fp32.hpp"
  #include "newton_solver/ops/jacobian/cuda_edge_fp64.hpp"
  #include "newton_solver/ops/jacobian/cuda_vertex_fp32.hpp"
  #include "newton_solver/ops/jacobian/cuda_vertex_fp64.hpp"
  #include "newton_solver/ops/linear_solve/cuda_cudss32.hpp"
  #include "newton_solver/ops/linear_solve/cuda_cudss64.hpp"
  #include "newton_solver/ops/mismatch/cuda_f64.hpp"
  #include "newton_solver/ops/voltage_update/cuda_fp64.hpp"
  #include "newton_solver/ops/voltage_update/cuda_mixed.hpp"
  #include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
  #include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#endif

#include <Eigen/Sparse>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using exp_20260413::iterative::LinearSystemSnapshot;
using exp_20260413::iterative::iter_dir_name;
using exp_20260413::iterative::write_csr;
using exp_20260413::iterative::write_metadata;
using exp_20260413::iterative::write_vector;

struct CliOptions {
    std::filesystem::path dataset_root = "/workspace/datasets/cuPF_benchmark_dumps";
    std::vector<std::string> cases = {
        "case118_ieee",
        "case2746wop_k",
        "case8387_pegase",
    };
    std::vector<std::filesystem::path> case_dirs;
    std::filesystem::path output_root = "/workspace/exp/20260413/iterative/dumps";
    std::string profile = "cuda_mixed_edge";
    double tolerance = 1e-8;
    int32_t max_iter = 50;
    int32_t max_dump_iters = -1;
    bool list_cases = false;
};

struct ProfileSpec {
    std::string name;
    std::string backend;
    std::string compute;
    std::string jacobian_name;
    std::string matrix_precision;
    std::string rhs_precision;
    bool cuda = false;
    bool mixed = false;
    JacobianBuilderType jacobian = JacobianBuilderType::EdgeBased;
};

struct RunSummary {
    std::string case_name;
    std::string profile;
    bool converged = false;
    int32_t iterations = 0;
    double final_mismatch = 0.0;
    int32_t snapshots = 0;
    double total_sec = 0.0;
};

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0
        << " [--dataset-root PATH]"
        << " [--case NAME] [--cases NAME...] [--case-dir PATH]"
        << " [--profile cpu_fp64_edge|cuda_mixed_edge|cuda_mixed_vertex|cuda_fp64_edge|cuda_fp64_vertex]"
        << " [--output-root PATH] [--tolerance FLOAT] [--max-iter INT]"
        << " [--max-dump-iters INT] [--list-cases]\n";
}

void add_case(CliOptions& options, const std::string& case_name, bool& custom_cases)
{
    if (!custom_cases) {
        options.cases.clear();
        custom_cases = true;
    }
    options.cases.push_back(case_name);
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    bool custom_cases = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset-root" && i + 1 < argc) {
            options.dataset_root = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            add_case(options, argv[++i], custom_cases);
        } else if (arg == "--cases") {
            bool consumed = false;
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("-", 0) != 0) {
                add_case(options, argv[++i], custom_cases);
                consumed = true;
            }
            if (!consumed) {
                throw std::runtime_error("--cases requires at least one case name");
            }
        } else if (arg == "--case-dir" && i + 1 < argc) {
            if (!custom_cases) {
                options.cases.clear();
                custom_cases = true;
            }
            options.case_dirs.push_back(argv[++i]);
        } else if (arg == "--profile" && i + 1 < argc) {
            options.profile = argv[++i];
        } else if (arg == "--output-root" && i + 1 < argc) {
            options.output_root = argv[++i];
        } else if (arg == "--tolerance" && i + 1 < argc) {
            options.tolerance = std::stod(argv[++i]);
        } else if (arg == "--max-iter" && i + 1 < argc) {
            options.max_iter = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--max-dump-iters" && i + 1 < argc) {
            options.max_dump_iters = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--list-cases") {
            options.list_cases = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.max_iter <= 0) {
        throw std::runtime_error("--max-iter must be positive");
    }
    if (options.tolerance <= 0.0) {
        throw std::runtime_error("--tolerance must be positive");
    }
    if (options.cases.empty() && options.case_dirs.empty() && !options.list_cases) {
        throw std::runtime_error("No cases were selected");
    }
    return options;
}

ProfileSpec parse_profile(const std::string& name)
{
    if (name == "cpu_fp64_edge") {
        return ProfileSpec{
            name, "cpu", "fp64", "edge_based", "float64", "float64",
            false, false, JacobianBuilderType::EdgeBased,
        };
    }
    if (name == "cuda_mixed_edge") {
        return ProfileSpec{
            name, "cuda", "mixed", "edge_based", "float32", "float32",
            true, true, JacobianBuilderType::EdgeBased,
        };
    }
    if (name == "cuda_mixed_vertex") {
        return ProfileSpec{
            name, "cuda", "mixed", "vertex_based", "float32", "float32",
            true, true, JacobianBuilderType::VertexBased,
        };
    }
    if (name == "cuda_fp64_edge") {
        return ProfileSpec{
            name, "cuda", "fp64", "edge_based", "float64", "float64",
            true, false, JacobianBuilderType::EdgeBased,
        };
    }
    if (name == "cuda_fp64_vertex") {
        return ProfileSpec{
            name, "cuda", "fp64", "vertex_based", "float64", "float64",
            true, false, JacobianBuilderType::VertexBased,
        };
    }
    throw std::runtime_error("Unknown profile: " + name);
}

std::string to_string(double value)
{
    std::ostringstream out;
    out << std::scientific << std::setprecision(17) << value;
    return out.str();
}

std::vector<std::pair<std::string, std::string>> metadata_entries(
    const cupf::tests::DumpCaseData& case_data,
    const std::filesystem::path& case_dir,
    const ProfileSpec& profile,
    const LinearSystemSnapshot& snapshot,
    int32_t iter,
    double normF,
    const NRConfig& config)
{
    return {
        {"case", case_data.case_name},
        {"case_dir", case_dir.string()},
        {"profile", profile.name},
        {"backend", profile.backend},
        {"compute", profile.compute},
        {"jacobian", profile.jacobian_name},
        {"iteration", std::to_string(iter)},
        {"normF", to_string(normF)},
        {"tolerance", to_string(config.tolerance)},
        {"max_iter", std::to_string(config.max_iter)},
        {"buses", std::to_string(case_data.rows)},
        {"pv", std::to_string(case_data.pv.size())},
        {"pq", std::to_string(case_data.pq.size())},
        {"dim", std::to_string(snapshot.rows)},
        {"nnz", std::to_string(snapshot.values.size())},
        {"matrix_precision", profile.matrix_precision},
        {"rhs_precision", profile.rhs_precision},
        {"rhs_definition", "b=-F"},
        {"matrix_definition", "J_at_Newton_iteration"},
        {"direct_solution_file", snapshot.x_direct.empty() ? "none" : "x_direct.txt"},
    };
}

std::filesystem::path write_snapshot(const CliOptions& cli,
                                     const cupf::tests::DumpCaseData& case_data,
                                     const std::filesystem::path& case_dir,
                                     const ProfileSpec& profile,
                                     const LinearSystemSnapshot& snapshot,
                                     int32_t iter,
                                     double normF,
                                     const NRConfig& config)
{
    const auto snapshot_dir = cli.output_root / case_data.case_name / profile.name / iter_dir_name(iter);
    std::filesystem::create_directories(snapshot_dir);

    write_csr(snapshot_dir / "J.csr",
              snapshot.rows,
              snapshot.cols,
              snapshot.row_ptr,
              snapshot.col_idx,
              snapshot.values);
    write_vector(snapshot_dir / "rhs.txt", snapshot.rhs);
    if (!snapshot.x_direct.empty()) {
        write_vector(snapshot_dir / "x_direct.txt", snapshot.x_direct);
    }
    write_metadata(snapshot_dir / "meta.txt",
                   metadata_entries(case_data, case_dir, profile, snapshot, iter, normF, config));
    return snapshot_dir;
}

bool should_dump_iteration(const CliOptions& cli, int32_t iter)
{
    return cli.max_dump_iters < 0 || iter < cli.max_dump_iters;
}

AnalyzeContext make_analyze_context(const cupf::tests::DumpCaseData& case_data,
                                    const YbusViewF64& ybus,
                                    const JacobianBuilder::Result& analysis)
{
    return AnalyzeContext{
        .ybus = ybus,
        .maps = analysis.maps,
        .J = analysis.J,
        .n_bus = case_data.rows,
        .pv = case_data.pv.data(),
        .n_pv = static_cast<int32_t>(case_data.pv.size()),
        .pq = case_data.pq.data(),
        .n_pq = static_cast<int32_t>(case_data.pq.size()),
    };
}

SolveContext make_solve_context(const cupf::tests::DumpCaseData& case_data,
                                const YbusViewF64& ybus,
                                const NRConfig& config)
{
    return SolveContext{
        .ybus = &ybus,
        .sbus = case_data.sbus.data(),
        .V0 = case_data.v0.data(),
        .config = &config,
    };
}

LinearSystemSnapshot snapshot_cpu_storage(const CpuFp64Storage& storage)
{
    using RowMajorMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>;

    RowMajorMatrix row_major = storage.J;
    row_major.makeCompressed();

    LinearSystemSnapshot snapshot;
    snapshot.rows = static_cast<int32_t>(row_major.rows());
    snapshot.cols = static_cast<int32_t>(row_major.cols());
    snapshot.row_ptr.assign(row_major.outerIndexPtr(),
                            row_major.outerIndexPtr() + row_major.rows() + 1);
    snapshot.col_idx.assign(row_major.innerIndexPtr(),
                            row_major.innerIndexPtr() + row_major.nonZeros());
    snapshot.values.assign(row_major.valuePtr(),
                           row_major.valuePtr() + row_major.nonZeros());
    snapshot.rhs.resize(storage.F.size());
    for (std::size_t i = 0; i < storage.F.size(); ++i) {
        snapshot.rhs[i] = -storage.F[i];
    }
    snapshot.x_direct = storage.dx;
    return snapshot;
}

RunSummary run_cpu_case(const CliOptions& cli,
                        const std::filesystem::path& case_dir,
                        const cupf::tests::DumpCaseData& case_data,
                        const ProfileSpec& profile,
                        const NRConfig& config)
{
    const auto t0 = std::chrono::steady_clock::now();
    const YbusViewF64 ybus = case_data.ybus();

    JacobianBuilder builder(profile.jacobian);
    const auto analysis = builder.analyze(ybus,
                                          case_data.pv.data(), static_cast<int32_t>(case_data.pv.size()),
                                          case_data.pq.data(), static_cast<int32_t>(case_data.pq.size()));
    const AnalyzeContext analyze_ctx = make_analyze_context(case_data, ybus, analysis);

    CpuFp64Storage storage;
    CpuMismatchOpF64 mismatch(storage);
    CpuJacobianOpF64 jacobian(storage);
    CpuLinearSolveKLU linear_solve(storage);
    CpuVoltageUpdateF64 voltage_update(storage);

    storage.prepare(analyze_ctx);
    linear_solve.analyze(analyze_ctx);
    storage.upload(make_solve_context(case_data, ybus, config));

    IterationContext iter_ctx{
        .storage = storage,
        .config = config,
        .pv = case_data.pv.data(),
        .n_pv = static_cast<int32_t>(case_data.pv.size()),
        .pq = case_data.pq.data(),
        .n_pq = static_cast<int32_t>(case_data.pq.size()),
    };

    RunSummary summary;
    summary.case_name = case_data.case_name;
    summary.profile = profile.name;

    for (int32_t iter = 0; iter < config.max_iter; ++iter) {
        iter_ctx.iter = iter;
        summary.iterations = iter + 1;

        mismatch.run(iter_ctx);
        summary.final_mismatch = iter_ctx.normF;
        if (iter_ctx.converged) {
            summary.converged = true;
            break;
        }

        jacobian.run(iter_ctx);
        linear_solve.run(iter_ctx);

        if (should_dump_iteration(cli, iter)) {
            const auto snapshot_dir = write_snapshot(cli,
                                                     case_data,
                                                     case_dir,
                                                     profile,
                                                     snapshot_cpu_storage(storage),
                                                     iter,
                                                     iter_ctx.normF,
                                                     config);
            ++summary.snapshots;
            std::cout << "DUMP case=" << case_data.case_name
                      << " profile=" << profile.name
                      << " iter=" << iter
                      << " path=" << snapshot_dir << "\n";
        }

        voltage_update.run(iter_ctx);
    }

    const auto t1 = std::chrono::steady_clock::now();
    summary.total_sec = std::chrono::duration<double>(t1 - t0).count();
    return summary;
}

#ifdef CUPF_WITH_CUDA

template <typename Float>
std::vector<double> copy_device_as_double(const DeviceBuffer<Float>& buffer)
{
    std::vector<Float> host(buffer.size());
    buffer.copyTo(host.data(), host.size());

    std::vector<double> out(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
        out[i] = static_cast<double>(host[i]);
    }
    return out;
}

LinearSystemSnapshot snapshot_cuda_mixed_storage(const CudaMixedStorage& storage)
{
    LinearSystemSnapshot snapshot;
    snapshot.rows = storage.dimF;
    snapshot.cols = storage.dimF;

    snapshot.row_ptr.resize(storage.d_J_row_ptr.size());
    snapshot.col_idx.resize(storage.d_J_col_idx.size());
    storage.d_J_row_ptr.copyTo(snapshot.row_ptr.data(), snapshot.row_ptr.size());
    storage.d_J_col_idx.copyTo(snapshot.col_idx.data(), snapshot.col_idx.size());
    snapshot.values = copy_device_as_double(storage.d_J_values);

    std::vector<double> F(storage.d_F.size());
    storage.d_F.copyTo(F.data(), F.size());
    snapshot.rhs.resize(F.size());
    for (std::size_t i = 0; i < F.size(); ++i) {
        snapshot.rhs[i] = static_cast<double>(-static_cast<float>(F[i]));
    }

    snapshot.x_direct = copy_device_as_double(storage.d_dx);
    return snapshot;
}

LinearSystemSnapshot snapshot_cuda_fp64_storage(const CudaFp64Storage& storage)
{
    LinearSystemSnapshot snapshot;
    snapshot.rows = storage.dimF;
    snapshot.cols = storage.dimF;

    snapshot.row_ptr.resize(storage.d_J_row_ptr.size());
    snapshot.col_idx.resize(storage.d_J_col_idx.size());
    storage.d_J_row_ptr.copyTo(snapshot.row_ptr.data(), snapshot.row_ptr.size());
    storage.d_J_col_idx.copyTo(snapshot.col_idx.data(), snapshot.col_idx.size());
    snapshot.values = copy_device_as_double(storage.d_J_values);

    std::vector<double> F(storage.d_F.size());
    storage.d_F.copyTo(F.data(), F.size());
    snapshot.rhs.resize(F.size());
    for (std::size_t i = 0; i < F.size(); ++i) {
        snapshot.rhs[i] = -F[i];
    }

    snapshot.x_direct = copy_device_as_double(storage.d_dx);
    return snapshot;
}

RunSummary run_cuda_mixed_case(const CliOptions& cli,
                               const std::filesystem::path& case_dir,
                               const cupf::tests::DumpCaseData& case_data,
                               const ProfileSpec& profile,
                               const NRConfig& config)
{
    const auto t0 = std::chrono::steady_clock::now();
    const YbusViewF64 ybus = case_data.ybus();

    JacobianBuilder builder(profile.jacobian);
    const auto analysis = builder.analyze(ybus,
                                          case_data.pv.data(), static_cast<int32_t>(case_data.pv.size()),
                                          case_data.pq.data(), static_cast<int32_t>(case_data.pq.size()));
    const AnalyzeContext analyze_ctx = make_analyze_context(case_data, ybus, analysis);

    CudaMixedStorage storage;
    CudaMismatchOpF64 mismatch(storage);
    std::unique_ptr<IJacobianOp> jacobian;
    if (profile.jacobian == JacobianBuilderType::VertexBased) {
        jacobian = std::make_unique<CudaJacobianOpVertexFp32>(storage);
    } else {
        jacobian = std::make_unique<CudaJacobianOpEdgeFp32>(storage);
    }
    CudaLinearSolveCuDSS32 linear_solve(storage);
    CudaVoltageUpdateMixed voltage_update(storage);

    storage.prepare(analyze_ctx);
    linear_solve.analyze(analyze_ctx);
    storage.upload(make_solve_context(case_data, ybus, config));

    IterationContext iter_ctx{
        .storage = storage,
        .config = config,
        .pv = case_data.pv.data(),
        .n_pv = static_cast<int32_t>(case_data.pv.size()),
        .pq = case_data.pq.data(),
        .n_pq = static_cast<int32_t>(case_data.pq.size()),
    };

    RunSummary summary;
    summary.case_name = case_data.case_name;
    summary.profile = profile.name;

    for (int32_t iter = 0; iter < config.max_iter; ++iter) {
        iter_ctx.iter = iter;
        summary.iterations = iter + 1;

        mismatch.run(iter_ctx);
        summary.final_mismatch = iter_ctx.normF;
        if (iter_ctx.converged) {
            summary.converged = true;
            break;
        }

        jacobian->run(iter_ctx);
        linear_solve.run(iter_ctx);

        if (should_dump_iteration(cli, iter)) {
            const auto snapshot_dir = write_snapshot(cli,
                                                     case_data,
                                                     case_dir,
                                                     profile,
                                                     snapshot_cuda_mixed_storage(storage),
                                                     iter,
                                                     iter_ctx.normF,
                                                     config);
            ++summary.snapshots;
            std::cout << "DUMP case=" << case_data.case_name
                      << " profile=" << profile.name
                      << " iter=" << iter
                      << " path=" << snapshot_dir << "\n";
        }

        voltage_update.run(iter_ctx);
    }

    const auto t1 = std::chrono::steady_clock::now();
    summary.total_sec = std::chrono::duration<double>(t1 - t0).count();
    return summary;
}

RunSummary run_cuda_fp64_case(const CliOptions& cli,
                              const std::filesystem::path& case_dir,
                              const cupf::tests::DumpCaseData& case_data,
                              const ProfileSpec& profile,
                              const NRConfig& config)
{
    const auto t0 = std::chrono::steady_clock::now();
    const YbusViewF64 ybus = case_data.ybus();

    JacobianBuilder builder(profile.jacobian);
    const auto analysis = builder.analyze(ybus,
                                          case_data.pv.data(), static_cast<int32_t>(case_data.pv.size()),
                                          case_data.pq.data(), static_cast<int32_t>(case_data.pq.size()));
    const AnalyzeContext analyze_ctx = make_analyze_context(case_data, ybus, analysis);

    CudaFp64Storage storage;
    CudaMismatchOpF64 mismatch(storage);
    std::unique_ptr<IJacobianOp> jacobian;
    if (profile.jacobian == JacobianBuilderType::VertexBased) {
        jacobian = std::make_unique<CudaJacobianOpVertexFp64>(storage);
    } else {
        jacobian = std::make_unique<CudaJacobianOpEdgeFp64>(storage);
    }
    CudaLinearSolveCuDSS64 linear_solve(storage);
    CudaVoltageUpdateFp64 voltage_update(storage);

    storage.prepare(analyze_ctx);
    linear_solve.analyze(analyze_ctx);
    storage.upload(make_solve_context(case_data, ybus, config));

    IterationContext iter_ctx{
        .storage = storage,
        .config = config,
        .pv = case_data.pv.data(),
        .n_pv = static_cast<int32_t>(case_data.pv.size()),
        .pq = case_data.pq.data(),
        .n_pq = static_cast<int32_t>(case_data.pq.size()),
    };

    RunSummary summary;
    summary.case_name = case_data.case_name;
    summary.profile = profile.name;

    for (int32_t iter = 0; iter < config.max_iter; ++iter) {
        iter_ctx.iter = iter;
        summary.iterations = iter + 1;

        mismatch.run(iter_ctx);
        summary.final_mismatch = iter_ctx.normF;
        if (iter_ctx.converged) {
            summary.converged = true;
            break;
        }

        jacobian->run(iter_ctx);
        linear_solve.run(iter_ctx);

        if (should_dump_iteration(cli, iter)) {
            const auto snapshot_dir = write_snapshot(cli,
                                                     case_data,
                                                     case_dir,
                                                     profile,
                                                     snapshot_cuda_fp64_storage(storage),
                                                     iter,
                                                     iter_ctx.normF,
                                                     config);
            ++summary.snapshots;
            std::cout << "DUMP case=" << case_data.case_name
                      << " profile=" << profile.name
                      << " iter=" << iter
                      << " path=" << snapshot_dir << "\n";
        }

        voltage_update.run(iter_ctx);
    }

    const auto t1 = std::chrono::steady_clock::now();
    summary.total_sec = std::chrono::duration<double>(t1 - t0).count();
    return summary;
}
#endif

RunSummary run_case(const CliOptions& cli,
                    const std::filesystem::path& case_dir,
                    const ProfileSpec& profile,
                    const NRConfig& config)
{
    const auto case_data = cupf::tests::load_dump_case(case_dir);

    std::cout << "CASE case=" << case_data.case_name
              << " profile=" << profile.name
              << " buses=" << case_data.rows
              << " pv=" << case_data.pv.size()
              << " pq=" << case_data.pq.size()
              << "\n";

    if (!profile.cuda) {
        return run_cpu_case(cli, case_dir, case_data, profile, config);
    }

#ifdef CUPF_WITH_CUDA
    if (profile.mixed) {
        return run_cuda_mixed_case(cli, case_dir, case_data, profile, config);
    }
    return run_cuda_fp64_case(cli, case_dir, case_data, profile, config);
#else
    throw std::runtime_error("CUDA profile requested, but this binary was built without CUPF_WITH_CUDA");
#endif
}

std::vector<std::filesystem::path> resolve_case_dirs(const CliOptions& cli)
{
    if (!cli.case_dirs.empty()) {
        return cli.case_dirs;
    }

    std::vector<std::filesystem::path> dirs;
    dirs.reserve(cli.cases.size());
    for (const std::string& case_name : cli.cases) {
        dirs.push_back(cli.dataset_root / case_name);
    }
    return dirs;
}

void list_cases(const std::filesystem::path& dataset_root)
{
    std::vector<std::string> cases;
    for (const auto& entry : std::filesystem::directory_iterator(dataset_root)) {
        if (entry.is_directory() && std::filesystem::exists(entry.path() / "dump_Ybus.mtx")) {
            cases.push_back(entry.path().filename().string());
        }
    }
    std::sort(cases.begin(), cases.end());
    for (const std::string& case_name : cases) {
        std::cout << case_name << '\n';
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        newton_solver::utils::initLogger(newton_solver::utils::LogLevel::Off, false);

        const CliOptions cli = parse_args(argc, argv);
        if (cli.list_cases) {
            list_cases(cli.dataset_root);
            return 0;
        }

        const ProfileSpec profile = parse_profile(cli.profile);
        const NRConfig config{cli.tolerance, cli.max_iter};
        const std::vector<std::filesystem::path> case_dirs = resolve_case_dirs(cli);

        std::cout << std::boolalpha << std::scientific << std::setprecision(12);
        for (const auto& case_dir : case_dirs) {
            const RunSummary summary = run_case(cli, case_dir, profile, config);
            std::cout << "SUMMARY case=" << summary.case_name
                      << " profile=" << summary.profile
                      << " converged=" << summary.converged
                      << " iterations=" << summary.iterations
                      << " final_mismatch=" << summary.final_mismatch
                      << " snapshots=" << summary.snapshots
                      << " total_sec=" << summary.total_sec
                      << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "dump_linear_systems failed: " << ex.what() << "\n";
        return 1;
    }
}
