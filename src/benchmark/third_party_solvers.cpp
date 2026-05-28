#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "benchmark/solver_registry.hpp"
#include "third_party_solvers/cudss_solver.hpp"
#include "third_party_solvers/glu_solver.hpp"
#include "third_party_solvers/mumps_solver.hpp"
#include "third_party_solvers/pangulu_solver.hpp"
#include "third_party_solvers/pardiso_solver.hpp"
#include "third_party_solvers/pastix_solver.hpp"
#include "third_party_solvers/superlu_solver.hpp"
#include "third_party_solvers/strumpack_solver.hpp"
#include "third_party_solvers/suitesparse_solvers.hpp"
#include "tools/compute_error.hpp"
#include "tools/matrix_io.hpp"
#include "tools/profiling.hpp"

namespace {

struct BenchmarkCase {
    std::string group;
    std::string name;
    std::filesystem::path matrix_path;
};

struct CompanionVectors {
    std::vector<double> rhs;
    std::vector<double> x_true;
};

struct Options {
    std::string matrix_set = "smoke";
    std::string matrix_name;
    std::vector<std::string> matrix_names;  // explicit set; overrides matrix_set
    std::vector<std::string> solver_names = {"klu"};
    bool append_output = false;
    bool warmup_gpu = false;
    std::filesystem::path output_path = "report/benchmark/third_party_solvers.csv";
};

std::vector<std::string> split_csv(const std::string& text)
{
    std::vector<std::string> values;
    std::stringstream stream(text);
    std::string value;

    while (std::getline(stream, value, ',')) {
        if (!value.empty()) {
            values.push_back(value);
        }
    }

    return values;
}

void append_unique(std::vector<std::string>& values, std::string value)
{
    for (const std::string& existing : values) {
        if (existing == value) {
            return;
        }
    }
    values.push_back(std::move(value));
}

std::vector<std::string> expand_solver_groups(const std::vector<std::string>& requested_names)
{
    std::vector<std::string> expanded;

    for (const std::string& name : requested_names) {
        if (name == "cpu") {
            append_unique(expanded, "klu");
            append_unique(expanded, "umfpack");
            append_unique(expanded, "superlu");
            append_unique(expanded, "pardiso");
            append_unique(expanded, "mumps-cpu");
            append_unique(expanded, "pastix-cpu");
            append_unique(expanded, "strumpack-cpu");
        } else if (name == "gpu") {
            append_unique(expanded, "mumps-gpu");
            append_unique(expanded, "pastix-gpu");
            append_unique(expanded, "strumpack-gpu");
            append_unique(expanded, "cudss-gpu");
            append_unique(expanded, "glu3-gpu");
        } else if (name == "pangulu") {
            append_unique(expanded, "pangulu-gpu");
        } else {
            append_unique(expanded, name);
        }
    }

    return expanded;
}

bool is_gpu_solver_for_warmup(const std::string& solver_name)
{
    return solver_name == "mumps-gpu" ||
           solver_name == "pastix-gpu" ||
           solver_name == "strumpack-gpu" ||
           solver_name == "cudss-gpu" ||
           solver_name == "glu3-gpu" ||
           solver_name == "mysolver-gpu";  // cy342: warm it too (primes the MYSOLVER_GPU_REUSE plan cache)
}

void print_usage()
{
    std::cout
        << "Usage: benchmark [options]\n\n"
        << "Options:\n"
        << "  --matrix-set smoke|suitesparse|matpower_nr|all\n"
        << "  --matrix NAME\n"
        << "  --matrices NAME[,NAME..]   (explicit set; overrides --matrix-set)\n"
        << "  --solver cpu|gpu|pangulu|klu|umfpack|superlu|pardiso|mumps-cpu|mumps-gpu|pastix-cpu|pastix-gpu|strumpack-cpu|strumpack-gpu|cudss-gpu|glu3-gpu|pangulu-gpu|mysolver|all[,..]\n"
        << "  --warmup-gpu\n"
        << "  --append\n"
        << "  --output PATH\n";
}

Options parse_args(int argc, char** argv)
{
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&](const std::string& option_name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + option_name);
            }
            return argv[++i];
        };

        if (arg == "--matrix-set") {
            options.matrix_set = next_value(arg);
        } else if (arg == "--matrix") {
            options.matrix_name = next_value(arg);
        } else if (arg == "--matrices") {
            options.matrix_names = split_csv(next_value(arg));
        } else if (arg == "--solver") {
            options.solver_names = split_csv(next_value(arg));
            options.solver_names = expand_solver_groups(options.solver_names);
        } else if (arg == "--warmup-gpu") {
            options.warmup_gpu = true;
        } else if (arg == "--append") {
            options.append_output = true;
        } else if (arg == "--output") {
            options.output_path = next_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (options.solver_names.empty()) {
        throw std::runtime_error("at least one solver must be requested");
    }

    return options;
}

std::vector<BenchmarkCase> all_cases()
{
    const std::filesystem::path suite_root = "/datasets/benchmark_matrices/matrices";
    const std::filesystem::path nr_root = "/datasets/power_system/nr_linear_systems";

    return {
        {"suitesparse", "memplus", suite_root / "memplus/memplus.mtx"},
        {"suitesparse", "rajat27", suite_root / "rajat27/rajat27.mtx"},
        {"suitesparse", "wang3", suite_root / "wang3/wang3.mtx"},
        {"suitesparse", "onetone2", suite_root / "onetone2/onetone2.mtx"},
        {"suitesparse", "rajat15", suite_root / "rajat15/rajat15.mtx"},

        {"matpower_nr", "case30", nr_root / "case30/J.mtx"},
        {"matpower_nr", "case118", nr_root / "case118/J.mtx"},
        {"matpower_nr", "case1197", nr_root / "case1197/J.mtx"},
        {"matpower_nr", "case_ACTIVSg2000", nr_root / "case_ACTIVSg2000/J.mtx"},
        {"matpower_nr", "case3012wp", nr_root / "case3012wp/J.mtx"},
        {"matpower_nr", "case6468rte", nr_root / "case6468rte/J.mtx"},
        {"matpower_nr", "case8387pegase", nr_root / "case8387pegase/J.mtx"},
        {"matpower_nr", "case_ACTIVSg25k", nr_root / "case_ACTIVSg25k/J.mtx"},
        {"matpower_nr", "case_SyntheticUSA", nr_root / "case_SyntheticUSA/J.mtx"},
    };
}

bool selected_by_options(const BenchmarkCase& benchmark_case, const Options& options)
{
    if (!options.matrix_names.empty()) {
        // Explicit list selects exactly those matrices, ignoring matrix_set.
        for (const std::string& name : options.matrix_names) {
            if (benchmark_case.name == name) {
                return true;
            }
        }
        return false;
    }

    if (!options.matrix_name.empty() && benchmark_case.name != options.matrix_name) {
        return false;
    }

    if (options.matrix_set == "all") {
        return true;
    }
    if (options.matrix_set == "smoke") {
        // PLAN §2/§4 smoke set: 3 SuiteSparse + 3 MATPOWER NR cases.
        return benchmark_case.name == "memplus" ||
               benchmark_case.name == "rajat27" ||
               benchmark_case.name == "wang3" ||
               benchmark_case.name == "case30" ||
               benchmark_case.name == "case118" ||
               benchmark_case.name == "case1197";
    }

    return benchmark_case.group == options.matrix_set;
}

std::vector<double> read_expected_vector(
    const std::filesystem::path& path,
    int expected_size,
    const std::string& label)
{
    const sparse_direct::io::DenseVector vector = sparse_direct::io::read_matrix_market_vector(path);
    if (vector.cols != 1) {
        throw std::runtime_error(label + " must be a column vector: " + path.string());
    }
    if (vector.size() != expected_size) {
        throw std::runtime_error(label + " size does not match matrix dimension: " + path.string());
    }
    return vector.values;
}

CompanionVectors read_companion_vectors(
    const std::filesystem::path& matrix_path,
    int rows,
    int cols)
{
    const std::filesystem::path rhs_path = matrix_path.parent_path() / "rhs.mtx";
    const std::filesystem::path x_true_path = matrix_path.parent_path() / "x_true.mtx";
    if (!std::filesystem::exists(rhs_path) || !std::filesystem::exists(x_true_path)) {
        throw std::runtime_error(
            "missing companion vectors next to " + matrix_path.string() +
            ": expected rhs.mtx and x_true.mtx");
    }

    return {
        read_expected_vector(rhs_path, rows, "rhs"),
        read_expected_vector(x_true_path, cols, "x_true"),
    };
}

std::string csv_escape(std::string_view text)
{
    std::string escaped;
    bool must_quote = false;
    for (char ch : text) {
        if (ch == '"' || ch == ',' || ch == '\n' || ch == '\r') {
            must_quote = true;
            break;
        }
    }

    if (!must_quote) {
        return std::string(text);
    }

    escaped.push_back('"');
    for (char ch : text) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

void write_csv_header(std::ostream& output)
{
    output
        << "matrix,group,solver,rows,cols,nnz,density,"
        << "analysis_ms,factor_ms,solve_ms,"
        << "berr,absolute_error,success,message\n";
}

void write_csv_result(
    std::ostream& output,
    const BenchmarkCase& benchmark_case,
    const std::string& solver_name,
    int rows,
    int cols,
    int nnz,
    double analysis_ms,
    double factor_ms,
    double solve_ms,
    double berr,
    double absolute_error,
    bool success,
    const std::string& message)
{
    const double density = static_cast<double>(nnz) / (static_cast<double>(rows) * static_cast<double>(cols));

    output
        << std::setprecision(17)
        << csv_escape(benchmark_case.name) << ","
        << csv_escape(benchmark_case.group) << ","
        << csv_escape(solver_name) << ","
        << rows << ","
        << cols << ","
        << nnz << ","
        << density << ","
        << analysis_ms << ","
        << factor_ms << ","
        << solve_ms << ","
        << berr << ","
        << absolute_error << ","
        << (success ? "true" : "false") << ","
        << csv_escape(message)
        << "\n";
}

bool wants_solver(
    const std::vector<std::string>& requested_names,
    std::initializer_list<std::string_view> aliases)
{
    for (const std::string& requested_name : requested_names) {
        if (requested_name == "all") {
            return true;
        }
        for (std::string_view alias : aliases) {
            if (requested_name == alias) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        PROFILE_SCOPE("benchmark.total");
        PROFILE_MARK("benchmark.start");

        const Options options = parse_args(argc, argv);
        if (!options.output_path.parent_path().empty()) {
            std::filesystem::create_directories(options.output_path.parent_path());
        }

        const std::ios::openmode output_mode =
            options.append_output ? std::ios::app : std::ios::trunc;
        std::ofstream output(options.output_path, output_mode);
        if (!output) {
            throw std::runtime_error("failed to open output file: " + options.output_path.string());
        }
        if (!options.append_output || std::filesystem::file_size(options.output_path) == 0) {
            write_csv_header(output);
        }

        std::vector<std::unique_ptr<sparse_direct::solver::LinearSolver>> solvers =
            sparse_direct::solver::make_suitesparse_solvers(options.solver_names);
        if (wants_solver(options.solver_names, {"superlu"})) {
            solvers.push_back(sparse_direct::solver::make_superlu_solver());
        }
        if (wants_solver(options.solver_names, {"pardiso"})) {
            solvers.push_back(sparse_direct::solver::make_pardiso_solver());
        }
        if (wants_solver(options.solver_names, {"mumps-cpu", "mumps"})) {
            solvers.push_back(sparse_direct::solver::make_mumps_cpu_solver());
        }
        if (wants_solver(options.solver_names, {"mumps-gpu", "mumps"})) {
            solvers.push_back(sparse_direct::solver::make_mumps_gpu_solver());
        }
        if (wants_solver(options.solver_names, {"pastix-cpu"})) {
            solvers.push_back(sparse_direct::solver::make_pastix_cpu_solver());
        }
        if (wants_solver(options.solver_names, {"pastix-gpu", "pastix"})) {
            solvers.push_back(sparse_direct::solver::make_pastix_gpu_solver());
        }
        if (wants_solver(options.solver_names, {"strumpack-cpu"})) {
            solvers.push_back(sparse_direct::solver::make_strumpack_cpu_solver());
        }
        if (wants_solver(options.solver_names, {"strumpack-gpu", "strumpack"})) {
            solvers.push_back(sparse_direct::solver::make_strumpack_gpu_solver());
        }
        if (wants_solver(options.solver_names, {"cudss-gpu", "cudss"})) {
            solvers.push_back(sparse_direct::solver::make_cudss_gpu_solver());
        }
        if (wants_solver(options.solver_names, {"glu3-gpu", "glu3", "glu"})) {
            solvers.push_back(sparse_direct::solver::make_glu3_gpu_solver());
        }
        if (wants_solver(options.solver_names, {"pangulu-gpu", "pangulu"})) {
            solvers.push_back(sparse_direct::solver::make_pangulu_gpu_solver());
        }
        if (wants_solver(options.solver_names, {"mysolver"})) {
            solvers.push_back(sparse_direct::solver::make_mysolver_solver());
        }
        if (wants_solver(options.solver_names, {"mysolver-gpu"})) {
            solvers.push_back(sparse_direct::solver::make_mysolver_gpu_solver());
        }
        if (solvers.empty()) {
            throw std::runtime_error("no supported solvers were selected");
        }

        for (const BenchmarkCase& benchmark_case : all_cases()) {
            if (!selected_by_options(benchmark_case, options)) {
                continue;
            }

            PROFILE_SCOPE("case." + benchmark_case.group + "." + benchmark_case.name);
            std::cout << "[load] " << benchmark_case.name << " " << benchmark_case.matrix_path << "\n";

            sparse_direct::matrix::CsrMatrix csr =
                sparse_direct::io::read_matrix_market_csr(benchmark_case.matrix_path);
            sparse_direct::matrix::CscMatrix csc = sparse_direct::matrix::to_csc(csr);
            CompanionVectors companions =
                read_companion_vectors(benchmark_case.matrix_path, csr.rows, csr.cols);

            for (const std::unique_ptr<sparse_direct::solver::LinearSolver>& solver : solvers) {
                if (options.warmup_gpu && is_gpu_solver_for_warmup(solver->name())) {
                    std::cout << "[warmup] " << benchmark_case.name << " " << solver->name() << "\n";
                    const sparse_direct::solver::SolverRun warmup_run =
                        solver->solve(csr, csc, companions.rhs);
                    if (!warmup_run.success) {
                        std::cout
                            << "[warmup-failed] " << benchmark_case.name << " " << solver->name()
                            << " message=" << warmup_run.message
                            << "\n";
                    }
                }

                std::cout << "[solve] " << benchmark_case.name << " " << solver->name() << "\n";

                sparse_direct::solver::SolverRun run;
                {
                    PROFILE_CUDA_SCOPE("solve." + benchmark_case.name + "." + solver->name());
                    run = solver->solve(csr, csc, companions.rhs);
                }

                double berr = std::numeric_limits<double>::quiet_NaN();
                double absolute_error = std::numeric_limits<double>::quiet_NaN();
                std::string message = run.message;
                bool success = run.success;
                if (run.success) {
                    try {
                        const sparse_direct::error::ErrorMetrics metrics =
                            sparse_direct::error::compute_error(
                                csr,
                                companions.rhs,
                                run.x,
                                companions.x_true);
                        berr = metrics.berr;
                        absolute_error = metrics.absolute_error;
                    } catch (const std::exception& error) {
                        success = false;
                        message += "; error metric failed: ";
                        message += error.what();
                    }
                }

                write_csv_result(
                    output,
                    benchmark_case,
                    solver->name(),
                    csr.rows,
                    csr.cols,
                    csr.nnz(),
                    run.analysis_ms,
                    run.factor_ms,
                    run.solve_ms,
                    berr,
                    absolute_error,
                    success,
                    message);
                output.flush();

                std::cout
                    << "[done] " << benchmark_case.name << " " << solver->name()
                    << " success=" << (success ? "true" : "false")
                    << "\n";
            }
        }

        PROFILE_MARK("benchmark.finish");
    } catch (const std::exception& error) {
        std::cerr << "benchmark: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
