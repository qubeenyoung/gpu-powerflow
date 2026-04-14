#include "jfnk_amgx_gpu.hpp"

#include "dump_case_loader.hpp"
#include "newton_solver/core/contexts.hpp"
#include "newton_solver/core/jacobian_builder.hpp"
#include "newton_solver/ops/mismatch/cuda_f64.hpp"
#include "newton_solver/ops/voltage_update/cuda_fp64.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "utils/logger.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using exp_20260414::newton_krylov::JfnkLinearSolveAmgx;
using exp_20260414::newton_krylov::JfnkOptions;
using exp_20260414::newton_krylov::JfnkStats;

struct CliOptions {
    std::filesystem::path dataset_root = "/workspace/datasets/pglib-opf/cuPF_benchmark_dumps";
    std::vector<std::string> cases = {"case30_ieee", "case118_ieee"};
    std::vector<std::filesystem::path> case_dirs;
    std::filesystem::path output_csv;
    double tolerance = 1e-8;
    int32_t max_iter = 50;
    JfnkOptions jfnk;
    bool continue_on_linear_failure = false;
    bool list_cases = false;
};

struct CaseResult {
    std::string case_name;
    bool converged = false;
    int32_t outer_iterations = 0;
    double final_mismatch = 0.0;
    std::string failure_reason;
    int64_t total_jv_calls = 0;
    int64_t total_inner_iterations = 0;
    int32_t max_inner_iterations = 0;
    int32_t linear_failures = 0;
    double total_sec = 0.0;
    double mismatch_sec = 0.0;
    double linear_solve_sec = 0.0;
    double preconditioner_setup_sec = 0.0;
    double voltage_update_sec = 0.0;
    double jv_sec = 0.0;
    double jv_mismatch_sec = 0.0;
    double jv_update_sec = 0.0;
};

double elapsed_sec(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

std::string format_double(double value)
{
    std::ostringstream out;
    out << std::scientific << std::setprecision(12) << value;
    return out.str();
}

std::string fd_eps_label(const JfnkOptions& options)
{
    return options.auto_epsilon ? std::string("auto") : format_double(options.fixed_epsilon);
}

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0
        << " [--dataset-root PATH]"
        << " [--case NAME] [--cases NAME...] [--case-dir PATH]"
        << " [--tolerance FLOAT] [--max-iter INT]"
        << " [--solver fgmres|gmres_none]"
        << " [--preconditioner none|amg_fd]"
        << " [--linear-tol FLOAT] [--inner-max-iter INT] [--gmres-restart INT]"
        << " [--ilut-drop-tol FLOAT] [--ilut-fill-factor INT] [--ilu-pivot-tol FLOAT]"
        << " [--fd-eps auto|FLOAT] [--continue-on-linear-failure]"
        << " [--output-csv PATH] [--list-cases]\n";
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
        } else if (arg == "--tolerance" && i + 1 < argc) {
            options.tolerance = std::stod(argv[++i]);
        } else if (arg == "--max-iter" && i + 1 < argc) {
            options.max_iter = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--solver" && i + 1 < argc) {
            options.jfnk.solver = argv[++i];
        } else if (arg == "--preconditioner" && i + 1 < argc) {
            options.jfnk.preconditioner = argv[++i];
        } else if (arg == "--linear-tol" && i + 1 < argc) {
            options.jfnk.linear_tolerance = std::stod(argv[++i]);
        } else if (arg == "--inner-max-iter" && i + 1 < argc) {
            options.jfnk.max_inner_iterations = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--gmres-restart" && i + 1 < argc) {
            options.jfnk.gmres_restart = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--ilut-drop-tol" && i + 1 < argc) {
            options.jfnk.ilut_drop_tol = std::stod(argv[++i]);
        } else if (arg == "--ilut-fill-factor" && i + 1 < argc) {
            options.jfnk.ilut_fill_factor = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--ilu-pivot-tol" && i + 1 < argc) {
            options.jfnk.ilu_pivot_tol = std::stod(argv[++i]);
        } else if (arg == "--fd-eps" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "auto") {
                options.jfnk.auto_epsilon = true;
            } else {
                options.jfnk.auto_epsilon = false;
                options.jfnk.fixed_epsilon = std::stod(value);
            }
        } else if (arg == "--continue-on-linear-failure") {
            options.continue_on_linear_failure = true;
        } else if (arg == "--output-csv" && i + 1 < argc) {
            options.output_csv = argv[++i];
        } else if (arg == "--list-cases") {
            options.list_cases = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.tolerance <= 0.0) {
        throw std::runtime_error("--tolerance must be positive");
    }
    if (options.max_iter <= 0) {
        throw std::runtime_error("--max-iter must be positive");
    }
    if (options.jfnk.solver != "fgmres" &&
        options.jfnk.solver != "gmres_none") {
        throw std::runtime_error("--solver must be fgmres or gmres_none");
    }
    if (options.jfnk.preconditioner != "none" &&
        options.jfnk.preconditioner != "amg_fd") {
        throw std::runtime_error("--preconditioner must be none or amg_fd");
    }
    if (options.jfnk.linear_tolerance <= 0.0) {
        throw std::runtime_error("--linear-tol must be positive");
    }
    if (options.jfnk.max_inner_iterations <= 0) {
        throw std::runtime_error("--inner-max-iter must be positive");
    }
    if (options.jfnk.gmres_restart <= 0) {
        throw std::runtime_error("--gmres-restart must be positive");
    }
    if (options.jfnk.ilut_drop_tol < 0.0) {
        throw std::runtime_error("--ilut-drop-tol must be nonnegative");
    }
    if (options.jfnk.ilut_fill_factor <= 0) {
        throw std::runtime_error("--ilut-fill-factor must be positive");
    }
    if (options.jfnk.ilu_pivot_tol <= 0.0) {
        throw std::runtime_error("--ilu-pivot-tol must be positive");
    }
    if (!options.jfnk.auto_epsilon && options.jfnk.fixed_epsilon <= 0.0) {
        throw std::runtime_error("--fd-eps must be auto or a positive float");
    }
    if (options.cases.empty() && options.case_dirs.empty() && !options.list_cases) {
        throw std::runtime_error("No cases were selected");
    }

    return options;
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

AnalyzeContext make_analyze_context(const cupf::tests::DumpCaseData& case_data,
                                    const YbusViewF64& ybus,
                                    const JacobianMaps& maps,
                                    const JacobianStructure& structure)
{
    return AnalyzeContext{
        .ybus = ybus,
        .maps = maps,
        .J = structure,
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

CaseResult run_case(const CliOptions& cli, const std::filesystem::path& case_dir)
{
    const auto total_start = Clock::now();
    const auto case_data = cupf::tests::load_dump_case(case_dir);
    const YbusViewF64 ybus = case_data.ybus();
    const NRConfig config{cli.tolerance, cli.max_iter};

    JacobianBuilder builder(JacobianBuilderType::EdgeBased);
    const JacobianBuilder::Result analysis =
        builder.analyze(ybus,
                        case_data.pv.data(),
                        static_cast<int32_t>(case_data.pv.size()),
                        case_data.pq.data(),
                        static_cast<int32_t>(case_data.pq.size()));
    const AnalyzeContext analyze_ctx =
        make_analyze_context(case_data, ybus, analysis.maps, analysis.J);

    CudaFp64Storage storage;
    CudaMismatchOpF64 mismatch(storage);
    JfnkLinearSolveAmgx linear_solve(storage, cli.jfnk);
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

    CaseResult result;
    result.case_name = case_data.case_name;

    for (int32_t iter = 0; iter < config.max_iter; ++iter) {
        iter_ctx.iter = iter;
        result.outer_iterations = iter + 1;

        const auto mismatch_start = Clock::now();
        mismatch.run(iter_ctx);
        const auto mismatch_end = Clock::now();
        result.mismatch_sec += elapsed_sec(mismatch_start, mismatch_end);
        result.final_mismatch = iter_ctx.normF;

        if (iter_ctx.converged) {
            result.converged = true;
            break;
        }

        const auto solve_start = Clock::now();
        linear_solve.run(iter_ctx);
        const auto solve_end = Clock::now();
        result.linear_solve_sec += elapsed_sec(solve_start, solve_end);

        const JfnkStats& stats = linear_solve.stats();
        if (!stats.last_success && !cli.continue_on_linear_failure) {
            result.failure_reason = "linear_" + stats.last_failure_reason;
            break;
        }

        const auto update_start = Clock::now();
        voltage_update.run(iter_ctx);
        const auto update_end = Clock::now();
        result.voltage_update_sec += elapsed_sec(update_start, update_end);
    }

    if (!result.converged && result.failure_reason.empty()) {
        result.failure_reason = "max_outer_iterations";
    }

    const JfnkStats& stats = linear_solve.stats();
    result.total_jv_calls = stats.total_jv_calls;
    result.total_inner_iterations = stats.total_inner_iterations;
    result.max_inner_iterations = stats.max_inner_iterations;
    result.linear_failures = stats.linear_failures;
    result.preconditioner_setup_sec = stats.total_preconditioner_setup_sec;
    result.jv_sec = stats.total_jv_sec;
    result.jv_mismatch_sec = stats.total_jv_mismatch_sec;
    result.jv_update_sec = stats.total_jv_update_sec;

    const auto total_end = Clock::now();
    result.total_sec = elapsed_sec(total_start, total_end);
    return result;
}

void print_result(const CliOptions& cli, const CaseResult& result)
{
    std::cout << std::boolalpha << std::scientific << std::setprecision(12)
              << "JFNK "
              << "case=" << result.case_name << ' '
              << "solver=" << cli.jfnk.solver << ' '
              << "preconditioner=" << cli.jfnk.preconditioner << ' '
              << "converged=" << result.converged << ' '
              << "outer_iterations=" << result.outer_iterations << ' '
              << "final_mismatch=" << result.final_mismatch << ' '
              << "linear_tol=" << cli.jfnk.linear_tolerance << ' '
              << "fd_eps=" << fd_eps_label(cli.jfnk) << ' '
              << "total_jv_calls=" << result.total_jv_calls << ' '
              << "total_inner_iterations=" << result.total_inner_iterations << ' '
              << "max_inner_iterations=" << result.max_inner_iterations << ' '
              << "linear_failures=" << result.linear_failures << ' '
              << "failure_reason=" << (result.failure_reason.empty() ? "none" : result.failure_reason) << ' '
              << "total_sec=" << result.total_sec
              << "\n";
}

void write_csv(const std::filesystem::path& output_path,
               const CliOptions& cli,
               const std::vector<CaseResult>& results)
{
    const auto parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("Failed to open CSV: " + output_path.string());
    }

    out << "case,solver,converged,outer_iterations,final_mismatch,"
           "linear_tol,inner_max_iter,gmres_restart,fd_eps,preconditioner,total_jv_calls,total_inner_iterations,"
           "max_inner_iterations,linear_failures,failure_reason,total_sec,mismatch_sec,"
           "linear_solve_sec,preconditioner_setup_sec,voltage_update_sec,jv_sec,jv_mismatch_sec,jv_update_sec\n";
    out << std::boolalpha << std::scientific << std::setprecision(17);
    for (const CaseResult& result : results) {
        out << result.case_name << ','
            << cli.jfnk.solver << ','
            << result.converged << ','
            << result.outer_iterations << ','
            << result.final_mismatch << ','
            << cli.jfnk.linear_tolerance << ','
            << cli.jfnk.max_inner_iterations << ','
            << cli.jfnk.gmres_restart << ','
            << fd_eps_label(cli.jfnk) << ','
            << cli.jfnk.preconditioner << ','
            << result.total_jv_calls << ','
            << result.total_inner_iterations << ','
            << result.max_inner_iterations << ','
            << result.linear_failures << ','
            << (result.failure_reason.empty() ? "none" : result.failure_reason) << ','
            << result.total_sec << ','
            << result.mismatch_sec << ','
            << result.linear_solve_sec << ','
            << result.preconditioner_setup_sec << ','
            << result.voltage_update_sec << ','
            << result.jv_sec << ','
            << result.jv_mismatch_sec << ','
            << result.jv_update_sec
            << '\n';
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

        const auto case_dirs = resolve_case_dirs(cli);
        std::vector<CaseResult> results;
        results.reserve(case_dirs.size());

        for (const auto& case_dir : case_dirs) {
            CaseResult result = run_case(cli, case_dir);
            print_result(cli, result);
            results.push_back(std::move(result));
        }

        if (!cli.output_csv.empty()) {
            write_csv(cli.output_csv, cli, results);
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "newton_krylov_probe failed: " << ex.what() << "\n";
        return 1;
    }
}
