#include "glu_lu.hpp"
#include "pf_case_loader.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path case_dir;
    std::string case_name;
    std::string rhs_mode = "synthetic";
    bool perturb = false;
    bool csv = false;
    bool print_glu_log = false;
};

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " --case-dir PATH [options]\n\n"
        << "Options:\n"
        << "  --case NAME\n"
        << "  --rhs-mode synthetic|mismatch\n"
        << "  --perturb\n"
        << "  --csv\n"
        << "  --print-glu-log\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--case-dir") {
            options.case_dir = need_value("--case-dir");
        } else if (arg == "--case") {
            options.case_name = need_value("--case");
        } else if (arg == "--rhs-mode") {
            options.rhs_mode = need_value("--rhs-mode");
        } else if (arg == "--perturb") {
            options.perturb = true;
        } else if (arg == "--csv") {
            options.csv = true;
        } else if (arg == "--print-glu-log") {
            options.print_glu_log = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.case_dir.empty()) {
        throw std::runtime_error("--case-dir is required");
    }
    if (options.case_name.empty()) {
        options.case_name = options.case_dir.filename().string();
    }
    if (options.rhs_mode != "synthetic" && options.rhs_mode != "mismatch") {
        throw std::runtime_error("--rhs-mode must be synthetic or mismatch");
    }
    return options;
}

std::vector<double> residual(const exp_20260520::LinearSystem& system,
                             const std::vector<double>& x)
{
    std::vector<double> ax = exp_20260520::matvec(system.matrix, x);
    for (std::size_t i = 0; i < ax.size(); ++i) {
        ax[i] -= system.rhs[i];
    }
    return ax;
}

void print_csv_header()
{
    std::cout
        << "case_name,rhs_mode,perturb,n_bus,n_pv,n_pq,linear_dim,linear_nnz,"
        << "glu_analyzed_nnz,glu_symbolic_nnz,glu_num_levels,"
        << "analyze_ms,symbolic_fill_ms,symbolic_csr_ms,symbolic_predict_ms,"
        << "symbolic_level_ms,numeric_ms,numeric_gpu_event_ms,solve_ms,total_ms,"
        << "residual_norm,relative_residual,relative_error\n";
}

void print_csv_row(const exp_20260520::CaseData& data,
                   const exp_20260520::LinearSystem& system,
                   const CliOptions& options,
                   const exp_20260520::GluResult& result,
                   double residual_norm,
                   double relative_residual,
                   double rel_error)
{
    const auto& t = result.timings;
    const auto& s = result.stats;
    const double total_ms = t.analyze_ms + t.symbolic_fill_ms + t.symbolic_csr_ms +
                            t.symbolic_predict_ms + t.symbolic_level_ms +
                            t.numeric_ms + t.solve_ms;

    std::cout << std::setprecision(12)
              << data.case_name << ','
              << system.rhs_mode << ','
              << (options.perturb ? 1 : 0) << ','
              << data.ybus.rows << ','
              << data.pv.size() << ','
              << data.pq.size() << ','
              << system.matrix.rows << ','
              << system.matrix.nnz() << ','
              << s.analyzed_nnz << ','
              << s.symbolic_nnz << ','
              << s.num_levels << ','
              << t.analyze_ms << ','
              << t.symbolic_fill_ms << ','
              << t.symbolic_csr_ms << ','
              << t.symbolic_predict_ms << ','
              << t.symbolic_level_ms << ','
              << t.numeric_ms << ','
              << t.numeric_gpu_event_ms << ','
              << t.solve_ms << ','
              << total_ms << ','
              << residual_norm << ','
              << relative_residual << ','
              << rel_error << '\n';
}

void print_markdown(const exp_20260520::CaseData& data,
                    const exp_20260520::LinearSystem& system,
                    const CliOptions& options,
                    const exp_20260520::GluResult& result,
                    double residual_norm,
                    double relative_residual,
                    double rel_error)
{
    const auto& t = result.timings;
    const auto& s = result.stats;
    const double total_ms = t.analyze_ms + t.symbolic_fill_ms + t.symbolic_csr_ms +
                            t.symbolic_predict_ms + t.symbolic_level_ms +
                            t.numeric_ms + t.solve_ms;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "# GLU Power Flow Jacobian Benchmark\n\n";
    std::cout << "- case: " << data.case_name << "\n";
    std::cout << "- rhs_mode: " << system.rhs_mode << "\n";
    std::cout << "- perturb: " << (options.perturb ? "on" : "off") << "\n";
    std::cout << "- n_bus: " << data.ybus.rows << "\n";
    std::cout << "- n_pv/n_pq: " << data.pv.size() << "/" << data.pq.size() << "\n";
    std::cout << "- linear_dim: " << system.matrix.rows << "\n";
    std::cout << "- linear_nnz: " << system.matrix.nnz() << "\n";
    std::cout << "- glu_analyzed_nnz: " << s.analyzed_nnz << "\n";
    std::cout << "- glu_symbolic_nnz: " << s.symbolic_nnz << "\n";
    std::cout << "- glu_num_levels: " << s.num_levels << "\n\n";

    std::cout << "| phase | ms |\n|---|---:|\n";
    std::cout << "| analyze/NICSLU preprocess | " << t.analyze_ms << " |\n";
    std::cout << "| symbolic fill | " << t.symbolic_fill_ms << " |\n";
    std::cout << "| symbolic CSR transpose | " << t.symbolic_csr_ms << " |\n";
    std::cout << "| symbolic predict LU values | " << t.symbolic_predict_ms << " |\n";
    std::cout << "| symbolic leveling | " << t.symbolic_level_ms << " |\n";
    std::cout << "| numeric GPU factorization | " << t.numeric_ms << " |\n";
    std::cout << "| numeric GPU event | " << t.numeric_gpu_event_ms << " |\n";
    std::cout << "| solve | " << t.solve_ms << " |\n";
    std::cout << "| total | " << total_ms << " |\n\n";

    std::cout << "| residual | value |\n|---|---:|\n";
    std::cout << std::scientific;
    std::cout << "| norm2 | " << residual_norm << " |\n";
    std::cout << "| relative_norm2 | " << relative_residual << " |\n";
    std::cout << "| relative_error | " << rel_error << " |\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        const exp_20260520::CaseData data =
            exp_20260520::load_case(options.case_dir, options.case_name);
        const exp_20260520::LinearSystem system =
            exp_20260520::build_linear_system(data, options.rhs_mode);

        exp_20260520::GluOptions glu_options;
        glu_options.perturb = options.perturb;
        glu_options.keep_glu_log = options.print_glu_log;

        const exp_20260520::GluResult result =
            exp_20260520::solve_with_glu(system.matrix, system.rhs, glu_options);

        const std::vector<double> r = residual(system, result.x);
        const double residual_norm = exp_20260520::norm2(r);
        const double relative_residual =
            residual_norm / std::max(exp_20260520::norm2(system.rhs),
                                     std::numeric_limits<double>::min());
        const double rel_error = options.rhs_mode == "synthetic"
            ? exp_20260520::relative_error(result.x, system.x_ref)
            : std::numeric_limits<double>::quiet_NaN();

        if (options.csv) {
            print_csv_header();
            print_csv_row(data, system, options, result,
                          residual_norm, relative_residual, rel_error);
        } else {
            print_markdown(data, system, options, result,
                           residual_norm, relative_residual, rel_error);
        }

        if (options.print_glu_log && (!result.glu_stdout.empty() || !result.glu_stderr.empty())) {
            std::cerr << "\n[GLU stdout]\n" << result.glu_stdout;
            if (!result.glu_stderr.empty()) {
                std::cerr << "\n[GLU stderr]\n" << result.glu_stderr;
            }
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "glu_pf_benchmark failed: " << ex.what() << '\n';
        return 1;
    }
}
