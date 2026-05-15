#include "cuiter/core/csr_matrix.hpp"

#include <ginkgo/ginkgo.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ValueType = double;
using IndexType = int32_t;
using Dense = gko::matrix::Dense<ValueType>;
using Csr = gko::matrix::Csr<ValueType, IndexType>;
using Bicgstab = gko::solver::Bicgstab<ValueType>;
using Jacobi = gko::preconditioner::Jacobi<ValueType, IndexType>;
using ParIlu = gko::factorization::ParIlu<ValueType, IndexType>;
using Ilu = gko::preconditioner::Ilu<ValueType, false, IndexType>;

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

struct Options {
    std::filesystem::path jf_root = "raw/cupf_jf_dumps";
    std::filesystem::path output_dir = "results/ginkgo_bicgstab_preconditioner_compare";
    std::vector<std::string> cases = {
        "case2383wp",
        "case3120sp",
        "case6468rte",
        "case9241pegase",
        "case13659pegase",
    };
    std::string executor = "cuda";
    int32_t iteration = 1;
    int32_t max_iters = 2;
    int32_t block_size = 16;
    int32_t parilu_iters = 10;
};

struct RunRow {
    std::string case_name;
    std::string preconditioner;
    std::string executor;
    int32_t iteration = 0;
    int32_t n = 0;
    int32_t nnz = 0;
    int32_t bicgstab_iters = 0;
    int32_t block_size = 0;
    int32_t parilu_iters = 0;
    double preconditioner_setup_ms = kNan;
    double parilu_factor_ms = 0.0;
    double ilu_preconditioner_ms = 0.0;
    double solver_generate_ms = kNan;
    double solve_ms = kNan;
    double total_ms = kNan;
    double true_abs_residual = kNan;
    double true_rel_residual = kNan;
    std::string status = "ok";
    std::string error_message;
};

std::vector<std::string> split_csv(const std::string& text)
{
    std::vector<std::string> out;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            out.push_back(token);
        }
    }
    return out;
}

void print_usage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "  --jf-root PATH\n"
        << "  --cases caseA,caseB\n"
        << "  --iter 1\n"
        << "  --executor cuda|omp|reference\n"
        << "  --bicgstab-iters 2\n"
        << "  --block-size 16\n"
        << "  --parilu-iters 10\n"
        << "  --output-dir PATH\n";
}

Options parse_args(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--jf-root") {
            options.jf_root = require_value("--jf-root");
        } else if (arg == "--cases") {
            options.cases = split_csv(require_value("--cases"));
        } else if (arg == "--iter") {
            options.iteration = std::stoi(require_value("--iter"));
        } else if (arg == "--executor") {
            options.executor = require_value("--executor");
        } else if (arg == "--bicgstab-iters") {
            options.max_iters = std::stoi(require_value("--bicgstab-iters"));
        } else if (arg == "--block-size") {
            options.block_size = std::stoi(require_value("--block-size"));
        } else if (arg == "--parilu-iters") {
            options.parilu_iters = std::stoi(require_value("--parilu-iters"));
        } else if (arg == "--output-dir") {
            options.output_dir = require_value("--output-dir");
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.cases.empty()) {
        throw std::runtime_error("at least one case is required");
    }
    if (options.max_iters <= 0) {
        throw std::runtime_error("--bicgstab-iters must be positive");
    }
    if (options.block_size <= 0 || options.block_size > 32) {
        throw std::runtime_error("--block-size must be in [1, 32] for Ginkgo Jacobi");
    }
    return options;
}

void expect_token(std::istream& in, const char* expected, const std::filesystem::path& path)
{
    std::string token;
    in >> token;
    if (token != expected) {
        throw std::runtime_error("expected token '" + std::string(expected) + "' in " +
                                 path.string());
    }
}

cuiter::CsrMatrix load_cupf_csr_dump(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open matrix file: " + path.string());
    }
    expect_token(in, "type", path);
    std::string type;
    in >> type;
    if (type != "csr_matrix") {
        throw std::runtime_error("not a cuPF CSR dump: " + path.string());
    }
    cuiter::CsrMatrix matrix;
    int32_t nnz = 0;
    expect_token(in, "rows", path);
    in >> matrix.rows;
    expect_token(in, "cols", path);
    in >> matrix.cols;
    expect_token(in, "nnz", path);
    in >> nnz;
    matrix.row_ptr.resize(static_cast<std::size_t>(matrix.rows + 1));
    matrix.col_idx.resize(static_cast<std::size_t>(nnz));
    matrix.values.resize(static_cast<std::size_t>(nnz));
    expect_token(in, "row_ptr", path);
    for (int32_t i = 0; i <= matrix.rows; ++i) {
        in >> matrix.row_ptr[static_cast<std::size_t>(i)];
    }
    expect_token(in, "col_idx", path);
    for (int32_t i = 0; i < nnz; ++i) {
        in >> matrix.col_idx[static_cast<std::size_t>(i)];
    }
    expect_token(in, "values", path);
    for (int32_t i = 0; i < nnz; ++i) {
        in >> matrix.values[static_cast<std::size_t>(i)];
    }
    if (!in || matrix.row_ptr.front() != 0 || matrix.row_ptr.back() != nnz) {
        throw std::runtime_error("malformed CSR dump: " + path.string());
    }
    return matrix;
}

std::vector<double> load_cupf_vector_dump(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open vector file: " + path.string());
    }
    expect_token(in, "type", path);
    std::string type;
    in >> type;
    if (type != "vector") {
        throw std::runtime_error("not a cuPF vector dump: " + path.string());
    }
    expect_token(in, "size", path);
    int32_t n = 0;
    in >> n;
    expect_token(in, "values", path);
    std::vector<double> values(static_cast<std::size_t>(n), 0.0);
    for (int32_t i = 0; i < n; ++i) {
        int32_t index = 0;
        double value = 0.0;
        in >> index >> value;
        if (index < 0 || index >= n) {
            throw std::runtime_error("vector index out of range in " + path.string());
        }
        values[static_cast<std::size_t>(index)] = value;
    }
    if (!in) {
        throw std::runtime_error("malformed vector dump: " + path.string());
    }
    return values;
}

std::filesystem::path jacobian_path(const std::filesystem::path& jf_root,
                                    const std::string& case_name,
                                    int32_t iteration)
{
    const auto case_dir = jf_root / case_name;
    const std::vector<std::filesystem::path> candidates = {
        case_dir / ("J" + std::to_string(iteration) + ".txt"),
        case_dir / "repeat_00" / ("jacobian_iter" + std::to_string(iteration) + ".txt"),
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return candidates.front();
}

std::filesystem::path rhs_path(const std::filesystem::path& jf_root,
                               const std::string& case_name,
                               int32_t iteration)
{
    const auto case_dir = jf_root / case_name;
    const std::vector<std::filesystem::path> candidates = {
        case_dir / ("F" + std::to_string(iteration) + ".txt"),
        case_dir / "repeat_00" / ("residual_iter" + std::to_string(iteration) + ".txt"),
        case_dir / "repeat_00" /
            ("residual_before_update_iter" + std::to_string(iteration) + ".txt"),
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return candidates.front();
}

double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                  const std::chrono::steady_clock::time_point& stop)
{
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

template <typename Func>
double timed_ms(const std::shared_ptr<gko::Executor>& exec, Func&& func)
{
    exec->synchronize();
    const auto start = std::chrono::steady_clock::now();
    func();
    exec->synchronize();
    const auto stop = std::chrono::steady_clock::now();
    return elapsed_ms(start, stop);
}

double norm2(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values) {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
}

std::vector<double> spmv(const cuiter::CsrMatrix& matrix, const std::vector<double>& x)
{
    std::vector<double> y(static_cast<std::size_t>(matrix.rows), 0.0);
    for (int32_t row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            sum += matrix.values[static_cast<std::size_t>(pos)] *
                   x[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(pos)])];
        }
        y[static_cast<std::size_t>(row)] = sum;
    }
    return y;
}

double true_relative_residual(const cuiter::CsrMatrix& matrix,
                              const std::vector<double>& rhs,
                              const std::vector<double>& x,
                              double* abs_residual)
{
    std::vector<double> residual = rhs;
    const std::vector<double> ax = spmv(matrix, x);
    for (std::size_t i = 0; i < residual.size(); ++i) {
        residual[i] -= ax[i];
    }
    const double abs_value = norm2(residual);
    if (abs_residual != nullptr) {
        *abs_residual = abs_value;
    }
    return abs_value / std::max(norm2(rhs), std::numeric_limits<double>::min());
}

std::shared_ptr<gko::Executor> make_executor(const std::string& name)
{
    if (name == "cuda") {
        return gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    }
    if (name == "omp") {
        return gko::OmpExecutor::create();
    }
    if (name == "reference") {
        return gko::ReferenceExecutor::create();
    }
    throw std::runtime_error("unsupported Ginkgo executor: " + name);
}

std::shared_ptr<const Csr> make_ginkgo_matrix(const cuiter::CsrMatrix& matrix,
                                              const std::shared_ptr<gko::Executor>& exec)
{
    gko::matrix_data<ValueType, IndexType> data(
        gko::dim<2>{static_cast<gko::size_type>(matrix.rows),
                    static_cast<gko::size_type>(matrix.cols)});
    data.nonzeros.reserve(matrix.values.size());
    for (int32_t row = 0; row < matrix.rows; ++row) {
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            data.nonzeros.emplace_back(row,
                                       matrix.col_idx[static_cast<std::size_t>(pos)],
                                       matrix.values[static_cast<std::size_t>(pos)]);
        }
    }
    auto out = Csr::create(exec);
    out->read(data);
    exec->synchronize();
    return gko::share(std::move(out));
}

std::shared_ptr<Dense> make_ginkgo_vector(const std::vector<double>& values,
                                          const std::shared_ptr<gko::Executor>& exec)
{
    const auto master = exec->get_master();
    auto host = Dense::create(master,
                              gko::dim<2>{static_cast<gko::size_type>(values.size()), 1});
    for (gko::size_type i = 0; i < values.size(); ++i) {
        host->at(i, 0) = values[static_cast<std::size_t>(i)];
    }
    return gko::clone(exec, host);
}

std::shared_ptr<Dense> make_zero_vector(gko::size_type n,
                                        const std::shared_ptr<gko::Executor>& exec)
{
    std::vector<double> zeros(static_cast<std::size_t>(n), 0.0);
    return make_ginkgo_vector(zeros, exec);
}

void warmup_ginkgo(const std::shared_ptr<gko::Executor>& exec)
{
    // Keep CUDA context creation and first-use Ginkgo kernels out of the first
    // measured case. The matrix is nonsymmetric so both BiCGSTAB and the
    // preconditioner paths exercise the same broad code path as the benchmark.
    cuiter::CsrMatrix host;
    host.rows = 2;
    host.cols = 2;
    host.row_ptr = {0, 2, 4};
    host.col_idx = {0, 1, 0, 1};
    host.values = {4.0, 1.0, -2.0, 3.0};
    const std::vector<double> rhs = {1.0, 2.0};
    const auto matrix = make_ginkgo_matrix(host, exec);
    const auto b = make_ginkgo_vector(rhs, exec);

    try {
        auto jacobi_preconditioner = gko::share(
            Jacobi::build().with_max_block_size(2u).on(exec)->generate(matrix));
        auto jacobi_solver =
            Bicgstab::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .with_generated_preconditioner(jacobi_preconditioner)
                .on(exec)
                ->generate(matrix);
        auto x = make_zero_vector(2, exec);
        jacobi_solver->apply(b, x);
        exec->synchronize();
    } catch (const std::exception&) {
        // Warm-up is best effort; the measured benchmark still reports errors.
    }

    try {
        auto parilu = gko::share(ParIlu::build().with_iterations(1u).on(exec)->generate(matrix));
        auto ilu_preconditioner = gko::share(Ilu::build().on(exec)->generate(parilu));
        auto ilu_solver =
            Bicgstab::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .with_generated_preconditioner(ilu_preconditioner)
                .on(exec)
                ->generate(matrix);
        auto x = make_zero_vector(2, exec);
        ilu_solver->apply(b, x);
        exec->synchronize();
    } catch (const std::exception&) {
        // Warm-up is best effort; the measured benchmark still reports errors.
    }
}

std::vector<double> copy_vector_to_host(const std::shared_ptr<Dense>& vector)
{
    auto host = gko::clone(vector->get_executor()->get_master(), vector);
    std::vector<double> out(static_cast<std::size_t>(vector->get_size()[0]), 0.0);
    for (gko::size_type i = 0; i < vector->get_size()[0]; ++i) {
        out[static_cast<std::size_t>(i)] = host->at(i, 0);
    }
    return out;
}

RunRow run_block_jacobi(const Options& options,
                        const std::string& case_name,
                        const cuiter::CsrMatrix& host_matrix,
                        const std::vector<double>& rhs,
                        const std::shared_ptr<gko::Executor>& exec,
                        const std::shared_ptr<const Csr>& matrix,
                        const std::shared_ptr<Dense>& b)
{
    RunRow row;
    row.case_name = case_name;
    row.preconditioner = "ginkgo_block_jacobi";
    row.executor = options.executor;
    row.iteration = options.iteration;
    row.n = host_matrix.rows;
    row.nnz = static_cast<int32_t>(host_matrix.values.size());
    row.bicgstab_iters = options.max_iters;
    row.block_size = options.block_size;

    std::shared_ptr<const gko::LinOp> preconditioner;
    row.preconditioner_setup_ms = timed_ms(exec, [&] {
        auto factory = Jacobi::build()
                           .with_max_block_size(static_cast<gko::size_type>(options.block_size))
                           .on(exec);
        preconditioner = gko::share(factory->generate(matrix));
    });

    std::unique_ptr<Bicgstab> solver;
    row.solver_generate_ms = timed_ms(exec, [&] {
        auto solver_factory =
            Bicgstab::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(
                    static_cast<gko::size_type>(options.max_iters)))
                .with_generated_preconditioner(preconditioner)
                .on(exec);
        solver = solver_factory->generate(matrix);
    });

    auto x = make_zero_vector(static_cast<gko::size_type>(host_matrix.cols), exec);
    row.solve_ms = timed_ms(exec, [&] { solver->apply(b, x); });

    const std::vector<double> x_host = copy_vector_to_host(x);
    row.true_rel_residual = true_relative_residual(host_matrix, rhs, x_host,
                                                   &row.true_abs_residual);
    if (!std::isfinite(row.true_rel_residual)) {
        row.status = "nonfinite_residual";
    }
    row.total_ms = row.preconditioner_setup_ms + row.solver_generate_ms + row.solve_ms;
    return row;
}

RunRow run_parilu(const Options& options,
                  const std::string& case_name,
                  const cuiter::CsrMatrix& host_matrix,
                  const std::vector<double>& rhs,
                  const std::shared_ptr<gko::Executor>& exec,
                  const std::shared_ptr<const Csr>& matrix,
                  const std::shared_ptr<Dense>& b)
{
    RunRow row;
    row.case_name = case_name;
    row.preconditioner = "ginkgo_parilu0_ilu";
    row.executor = options.executor;
    row.iteration = options.iteration;
    row.n = host_matrix.rows;
    row.nnz = static_cast<int32_t>(host_matrix.values.size());
    row.bicgstab_iters = options.max_iters;
    row.block_size = 0;
    row.parilu_iters = options.parilu_iters;

    std::shared_ptr<const gko::LinOp> parilu;
    row.parilu_factor_ms = timed_ms(exec, [&] {
        auto factory = ParIlu::build()
                           .with_iterations(static_cast<gko::size_type>(options.parilu_iters))
                           .on(exec);
        parilu = gko::share(factory->generate(matrix));
    });

    std::shared_ptr<const gko::LinOp> preconditioner;
    row.ilu_preconditioner_ms = timed_ms(exec, [&] {
        auto factory = Ilu::build().on(exec);
        preconditioner = gko::share(factory->generate(parilu));
    });
    row.preconditioner_setup_ms = row.parilu_factor_ms + row.ilu_preconditioner_ms;

    std::unique_ptr<Bicgstab> solver;
    row.solver_generate_ms = timed_ms(exec, [&] {
        auto solver_factory =
            Bicgstab::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(
                    static_cast<gko::size_type>(options.max_iters)))
                .with_generated_preconditioner(preconditioner)
                .on(exec);
        solver = solver_factory->generate(matrix);
    });

    auto x = make_zero_vector(static_cast<gko::size_type>(host_matrix.cols), exec);
    row.solve_ms = timed_ms(exec, [&] { solver->apply(b, x); });

    const std::vector<double> x_host = copy_vector_to_host(x);
    row.true_rel_residual = true_relative_residual(host_matrix, rhs, x_host,
                                                   &row.true_abs_residual);
    if (!std::isfinite(row.true_rel_residual)) {
        row.status = "nonfinite_residual";
    }
    row.total_ms = row.preconditioner_setup_ms + row.solver_generate_ms + row.solve_ms;
    return row;
}

std::string csv_escape(const std::string& text)
{
    if (text.find_first_of(",\"\n") == std::string::npos) {
        return text;
    }
    std::string out = "\"";
    for (char ch : text) {
        if (ch == '"') {
            out += "\"\"";
        } else {
            out += ch;
        }
    }
    out += '"';
    return out;
}

void write_summary_csv(const std::filesystem::path& path, const std::vector<RunRow>& rows)
{
    std::ofstream out(path);
    out << "case,preconditioner,executor,iteration,n,nnz,bicgstab_iters,block_size,"
           "parilu_iters,preconditioner_setup_ms,parilu_factor_ms,ilu_preconditioner_ms,"
           "solver_generate_ms,solve_ms,total_ms,true_abs_residual,true_rel_residual,"
           "status,error_message\n";
    out << std::setprecision(17);
    for (const auto& row : rows) {
        out << csv_escape(row.case_name) << ','
            << csv_escape(row.preconditioner) << ','
            << csv_escape(row.executor) << ','
            << row.iteration << ','
            << row.n << ','
            << row.nnz << ','
            << row.bicgstab_iters << ','
            << row.block_size << ','
            << row.parilu_iters << ','
            << row.preconditioner_setup_ms << ','
            << row.parilu_factor_ms << ','
            << row.ilu_preconditioner_ms << ','
            << row.solver_generate_ms << ','
            << row.solve_ms << ','
            << row.total_ms << ','
            << row.true_abs_residual << ','
            << row.true_rel_residual << ','
            << csv_escape(row.status) << ','
            << csv_escape(row.error_message) << '\n';
    }
}

std::string fmt(double value)
{
    if (!std::isfinite(value)) {
        return "nan";
    }
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(3) << value;
    return ss.str();
}

std::string fmt_ms(double value)
{
    if (!std::isfinite(value)) {
        return "nan";
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3) << value;
    return ss.str();
}

void write_report(const std::filesystem::path& path,
                  const Options& options,
                  const std::vector<RunRow>& rows)
{
    std::ofstream out(path);
    out << "# Ginkgo BiCGSTAB Preconditioner Comparison\n\n";
    out << "- Input: `J" << options.iteration << "/F" << options.iteration
        << "` from `" << options.jf_root.string() << "`.\n";
    out << "- Fixed outer solver: Ginkgo `BiCGSTAB`, max iterations `"
        << options.max_iters << "`.\n";
    out << "- `ginkgo_block_jacobi` is Ginkgo's block-Jacobi preconditioner "
           "with max block size `"
        << options.block_size << "`.\n";
    out << "- `ginkgo_parilu0_ilu` is Ginkgo ParILU(0) plus triangular ILU "
           "preconditioner. This is scalar-pattern ILU(0), not our dense block "
           "graph ILU(0).\n\n";

    out << "| case | preconditioner | setup ms | solve ms | total ms | true rel residual |\n";
    out << "|---|---:|---:|---:|---:|---:|\n";
    for (const auto& row : rows) {
        out << "| " << row.case_name
            << " | " << row.preconditioner
            << " | " << fmt_ms(row.preconditioner_setup_ms)
            << " | " << fmt_ms(row.solve_ms)
            << " | " << fmt_ms(row.total_ms)
            << " | " << fmt(row.true_rel_residual) << " |\n";
    }

    int better_residual = 0;
    int better_solve = 0;
    for (const auto& case_name : options.cases) {
        const RunRow* bj = nullptr;
        const RunRow* ilu = nullptr;
        for (const auto& row : rows) {
            if (row.case_name == case_name && row.status == "ok") {
                if (row.preconditioner == "ginkgo_block_jacobi") {
                    bj = &row;
                } else if (row.preconditioner == "ginkgo_parilu0_ilu") {
                    ilu = &row;
                }
            }
        }
        if (bj != nullptr && ilu != nullptr) {
            better_residual += ilu->true_rel_residual < bj->true_rel_residual ? 1 : 0;
            better_solve += ilu->solve_ms < bj->solve_ms ? 1 : 0;
        }
    }

    out << "\n## Short Read\n\n";
    out << "- ParILU/ILU residual better than block-Jacobi on `"
        << better_residual << "/" << options.cases.size() << "` cases.\n";
    out << "- ParILU/ILU solve phase faster than block-Jacobi on `"
        << better_solve << "/" << options.cases.size() << "` cases. Setup is usually "
        << "not comparable to block-Jacobi because it includes ParILU factor sweeps.\n";
    out << "- This benchmark answers whether Ginkgo's built-in preconditioners "
           "change BiCGSTAB quality/cost. It does not replace the custom dense "
           "block ILU pilot comparison.\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_args(argc, argv);
        std::filesystem::create_directories(options.output_dir);

        const auto exec = make_executor(options.executor);
        warmup_ginkgo(exec);
        std::vector<RunRow> rows;

        for (const auto& case_name : options.cases) {
            std::cout << "[case] " << case_name << std::endl;
            try {
                const cuiter::CsrMatrix matrix =
                    load_cupf_csr_dump(jacobian_path(options.jf_root, case_name,
                                                     options.iteration));
                const std::vector<double> rhs =
                    load_cupf_vector_dump(rhs_path(options.jf_root, case_name,
                                                   options.iteration));
                if (matrix.rows != matrix.cols ||
                    matrix.rows != static_cast<int32_t>(rhs.size())) {
                    throw std::runtime_error("matrix/vector dimension mismatch");
                }

                const auto gko_matrix = make_ginkgo_matrix(matrix, exec);
                const auto b = make_ginkgo_vector(rhs, exec);

                rows.push_back(run_block_jacobi(options, case_name, matrix, rhs,
                                                exec, gko_matrix, b));
                rows.push_back(run_parilu(options, case_name, matrix, rhs,
                                          exec, gko_matrix, b));
            } catch (const std::exception& e) {
                for (const std::string& preconditioner :
                     {"ginkgo_block_jacobi", "ginkgo_parilu0_ilu"}) {
                    RunRow row;
                    row.case_name = case_name;
                    row.preconditioner = preconditioner;
                    row.executor = options.executor;
                    row.iteration = options.iteration;
                    row.bicgstab_iters = options.max_iters;
                    row.block_size = options.block_size;
                    row.parilu_iters = options.parilu_iters;
                    row.status = "error";
                    row.error_message = e.what();
                    rows.push_back(row);
                }
                std::cerr << "[error] " << case_name << ": " << e.what() << '\n';
            }
        }

        write_summary_csv(options.output_dir / "ginkgo_bicgstab_preconditioner_summary.csv",
                          rows);
        write_report(options.output_dir / "ginkgo_bicgstab_preconditioner_report.md",
                     options, rows);
        std::cout << "[done] wrote Ginkgo comparison results to "
                  << options.output_dir << '\n';
    } catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << '\n';
        return 1;
    }
    return 0;
}
