#include "cuiter/solver/gmres_solver.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path matrix_path;
    std::filesystem::path rhs_path;
    std::filesystem::path dataset_root;
    std::string case_name;
    int32_t synthetic_n = 512;
    std::string solver = "gmres";
    int32_t gmres_restart = 16;
    int32_t gmres_max_iters = 32;
    double gmres_rtol = 1.0e-3;
    double gmres_atol = 0.0;
    std::string preconditioner = "metis_block_jacobi";
    int32_t block_size = 32;
    std::string block_jacobi_precision = "fp32";
    std::string block_jacobi_apply = "inverse_gemv";
    double block_jacobi_shift = 1.0e-8;
    bool csv = false;
    std::filesystem::path residual_history_path;
};

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [--matrix PATH] [--rhs PATH]\n"
        << "       " << argv0 << " [--dataset-root PATH --case case_ACTIVSg10k]\n"
        << "       " << argv0 << " [--synthetic-n N]\n\n"
        << "GMRES options:\n"
        << "  --solver gmres\n"
        << "  --gmres-restart 8|16|32\n"
        << "  --gmres-max-iters INT\n"
        << "  --gmres-rtol FLOAT\n"
        << "  --preconditioner none|metis_block_jacobi\n"
        << "  --block-size 4|8|16|32|64\n"
        << "  --block-jacobi-precision fp32|fp64\n"
        << "  --block-jacobi-apply inverse_gemv|lu_solve\n"
        << "  --residual-history PATH\n"
        << "  --csv\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--matrix" && i + 1 < argc) {
            options.matrix_path = argv[++i];
        } else if (arg == "--rhs" && i + 1 < argc) {
            options.rhs_path = argv[++i];
        } else if (arg == "--dataset-root" && i + 1 < argc) {
            options.dataset_root = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            options.case_name = argv[++i];
        } else if (arg == "--synthetic-n" && i + 1 < argc) {
            options.synthetic_n = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--solver" && i + 1 < argc) {
            options.solver = argv[++i];
        } else if (arg == "--gmres-restart" && i + 1 < argc) {
            options.gmres_restart = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--gmres-max-iters" && i + 1 < argc) {
            options.gmres_max_iters = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--gmres-rtol" && i + 1 < argc) {
            options.gmres_rtol = std::stod(argv[++i]);
        } else if (arg == "--gmres-atol" && i + 1 < argc) {
            options.gmres_atol = std::stod(argv[++i]);
        } else if (arg == "--preconditioner" && i + 1 < argc) {
            options.preconditioner = argv[++i];
        } else if (arg == "--block-size" && i + 1 < argc) {
            options.block_size = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--block-jacobi-precision" && i + 1 < argc) {
            options.block_jacobi_precision = argv[++i];
        } else if (arg == "--block-jacobi-apply" && i + 1 < argc) {
            options.block_jacobi_apply = argv[++i];
        } else if (arg == "--block-jacobi-shift" && i + 1 < argc) {
            options.block_jacobi_shift = std::stod(argv[++i]);
        } else if (arg == "--csv") {
            options.csv = true;
        } else if (arg == "--residual-history" && i + 1 < argc) {
            options.residual_history_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.solver != "gmres") {
        throw std::runtime_error("only --solver gmres is implemented in this experiment");
    }
    return options;
}

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

void expect_token(std::istream& in,
                  const std::string& expected,
                  const std::filesystem::path& path)
{
    std::string token;
    if (!(in >> token) || token != expected) {
        throw std::runtime_error("expected token '" + expected + "' in " + path.string());
    }
}

cuiter::CsrMatrix build_synthetic_matrix(int32_t n)
{
    if (n <= 0) {
        throw std::runtime_error("--synthetic-n must be positive");
    }

    cuiter::CsrMatrix matrix;
    matrix.rows = n;
    matrix.cols = n;
    matrix.row_ptr.assign(static_cast<std::size_t>(n + 1), 0);
    for (int32_t row = 0; row < n; ++row) {
        matrix.row_ptr[static_cast<std::size_t>(row)] = static_cast<int32_t>(matrix.col_idx.size());
        if (row - 17 >= 0) {
            matrix.col_idx.push_back(row - 17);
            matrix.values.push_back(0.02);
        }
        if (row - 1 >= 0) {
            matrix.col_idx.push_back(row - 1);
            matrix.values.push_back(-0.95);
        }
        matrix.col_idx.push_back(row);
        matrix.values.push_back(4.0 + 0.001 * static_cast<double>(row % 29));
        if (row + 1 < n) {
            matrix.col_idx.push_back(row + 1);
            matrix.values.push_back(-1.05);
        }
        if (row + 31 < n) {
            matrix.col_idx.push_back(row + 31);
            matrix.values.push_back(0.03);
        }
    }
    matrix.row_ptr[static_cast<std::size_t>(n)] = static_cast<int32_t>(matrix.col_idx.size());
    return matrix;
}

cuiter::CsrMatrix load_cupf_csr_dump(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open cuPF CSR dump: " + path.string());
    }

    expect_token(in, "type", path);
    std::string type;
    in >> type;
    if (type != "csr_matrix") {
        throw std::runtime_error("cuPF dump is not a csr_matrix: " + path.string());
    }

    cuiter::CsrMatrix matrix;
    int32_t nnz = 0;
    expect_token(in, "rows", path);
    in >> matrix.rows;
    expect_token(in, "cols", path);
    in >> matrix.cols;
    expect_token(in, "nnz", path);
    in >> nnz;
    if (matrix.rows <= 0 || matrix.cols <= 0 || matrix.rows != matrix.cols || nnz <= 0) {
        throw std::runtime_error("invalid cuPF CSR dimensions in " + path.string());
    }

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
        throw std::runtime_error("malformed cuPF CSR dump: " + path.string());
    }
    return matrix;
}

cuiter::CsrMatrix load_matrix_market(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open matrix file: " + path.string());
    }

    std::string line;
    std::getline(in, line);
    const std::string header = lower_copy(line);
    const bool symmetric = header.find("symmetric") != std::string::npos;
    if (header.find("matrixmarket") == std::string::npos ||
        header.find("coordinate") == std::string::npos) {
        throw std::runtime_error("only MatrixMarket coordinate files are supported");
    }

    do {
        if (!std::getline(in, line)) {
            throw std::runtime_error("missing MatrixMarket size line");
        }
    } while (!line.empty() && line[0] == '%');

    std::istringstream size_stream(line);
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t nnz = 0;
    size_stream >> rows >> cols >> nnz;
    if (rows <= 0 || cols <= 0 || rows != cols || nnz <= 0) {
        throw std::runtime_error("invalid MatrixMarket size line");
    }

    std::vector<std::tuple<int32_t, int32_t, double>> entries;
    entries.reserve(static_cast<std::size_t>(nnz) * (symmetric ? 2U : 1U));
    for (int32_t k = 0; k < nnz; ++k) {
        int32_t row = 0;
        int32_t col = 0;
        double value = 0.0;
        in >> row >> col >> value;
        --row;
        --col;
        entries.emplace_back(row, col, value);
        if (symmetric && row != col) {
            entries.emplace_back(col, row, value);
        }
    }
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        if (std::get<0>(lhs) != std::get<0>(rhs)) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        }
        return std::get<1>(lhs) < std::get<1>(rhs);
    });

    cuiter::CsrMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.row_ptr.assign(static_cast<std::size_t>(rows + 1), 0);
    int32_t current_row = 0;
    for (std::size_t i = 0; i < entries.size();) {
        const int32_t row = std::get<0>(entries[i]);
        const int32_t col = std::get<1>(entries[i]);
        double value = 0.0;
        while (i < entries.size() && std::get<0>(entries[i]) == row &&
               std::get<1>(entries[i]) == col) {
            value += std::get<2>(entries[i]);
            ++i;
        }
        while (current_row <= row) {
            matrix.row_ptr[static_cast<std::size_t>(current_row)] =
                static_cast<int32_t>(matrix.col_idx.size());
            ++current_row;
        }
        matrix.col_idx.push_back(col);
        matrix.values.push_back(value);
    }
    while (current_row <= rows) {
        matrix.row_ptr[static_cast<std::size_t>(current_row)] =
            static_cast<int32_t>(matrix.col_idx.size());
        ++current_row;
    }
    return matrix;
}

std::string first_token_in_file(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path.string());
    }
    std::string token;
    in >> token;
    return token;
}

cuiter::CsrMatrix load_matrix_auto(const std::filesystem::path& path)
{
    const std::string first_token = lower_copy(first_token_in_file(path));
    if (first_token == "type") {
        return load_cupf_csr_dump(path);
    }
    if (first_token.find("matrixmarket") != std::string::npos) {
        return load_matrix_market(path);
    }
    throw std::runtime_error("unsupported matrix format: " + path.string());
}

std::vector<double> load_cupf_vector_dump(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open cuPF vector dump: " + path.string());
    }

    expect_token(in, "type", path);
    std::string type;
    in >> type;
    if (type != "vector") {
        throw std::runtime_error("cuPF dump is not a vector: " + path.string());
    }
    expect_token(in, "size", path);
    int32_t n = 0;
    in >> n;
    if (n <= 0) {
        throw std::runtime_error("invalid cuPF vector length in " + path.string());
    }
    expect_token(in, "values", path);

    std::vector<double> values(static_cast<std::size_t>(n), 0.0);
    for (int32_t k = 0; k < n; ++k) {
        int32_t index = 0;
        double value = 0.0;
        in >> index >> value;
        if (!in || index < 0 || index >= n) {
            throw std::runtime_error("malformed cuPF vector dump: " + path.string());
        }
        values[static_cast<std::size_t>(index)] = value;
    }
    return values;
}

std::vector<double> load_vector_text(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open vector file: " + path.string());
    }
    std::vector<double> values;
    double value = 0.0;
    while (in >> value) {
        values.push_back(value);
    }
    return values;
}

std::vector<double> load_vector_auto(const std::filesystem::path& path)
{
    const std::string first_token = lower_copy(first_token_in_file(path));
    if (first_token == "type") {
        return load_cupf_vector_dump(path);
    }
    return load_vector_text(path);
}

std::filesystem::path first_existing(const std::vector<std::filesystem::path>& candidates)
{
    for (const auto& path : candidates) {
        if (!path.empty() && std::filesystem::exists(path)) {
            return path;
        }
    }
    return {};
}

void resolve_case_paths(CliOptions& options)
{
    if (options.dataset_root.empty() || options.case_name.empty() || !options.matrix_path.empty()) {
        return;
    }

    const std::filesystem::path case_dir = options.dataset_root / options.case_name;
    options.matrix_path = first_existing({
        case_dir / "J1.txt",
        case_dir / "jacobian_iter1.txt",
        case_dir / "repeat_00" / "jacobian_iter1.txt",
        case_dir / "jacobian_iter1.mtx",
        case_dir / "J_iter1.mtx",
        case_dir / "jacobian_iter1" / "matrix.mtx",
        options.dataset_root / (options.case_name + "_jacobian_iter1.mtx"),
    });
    if (options.rhs_path.empty()) {
        options.rhs_path = first_existing({
            case_dir / "F1.txt",
            case_dir / "residual_iter1.txt",
            case_dir / "repeat_00" / "residual_iter1.txt",
            case_dir / "F_iter1.txt",
            case_dir / "rhs_iter1.txt",
            case_dir / "F_iter1" / "vector.txt",
            options.dataset_root / (options.case_name + "_F_iter1.txt"),
        });
    }
}

std::vector<double> default_rhs(int32_t n)
{
    std::vector<double> rhs(static_cast<std::size_t>(n), 0.0);
    for (int32_t i = 0; i < n; ++i) {
        rhs[static_cast<std::size_t>(i)] =
            std::sin(0.013 * static_cast<double>(i + 1)) +
            0.1 * std::cos(0.071 * static_cast<double>(i + 3));
    }
    return rhs;
}

double host_norm2(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values) {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
}

void print_result_table(const cuiter::LinearSolveResult& result,
                        const CliOptions& cli,
                        const cuiter::CsrMatrix& matrix)
{
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "# GMRES + METIS Block-Jacobi Benchmark\n\n";
    std::cout << "- matrix_rows: " << matrix.rows << "\n";
    std::cout << "- matrix_nnz: " << matrix.nnz() << "\n";
    std::cout << "- solver: " << cli.solver << "\n";
    std::cout << "- preconditioner: " << cli.preconditioner << "\n";
    std::cout << "- block_jacobi_apply: " << cli.block_jacobi_apply << "\n";
    std::cout << "- block_jacobi_precision: " << cli.block_jacobi_precision << "\n\n";

    std::cout << "## 1. block structure\n\n";
    std::cout << "| metric | value |\n|---|---:|\n";
    std::cout << "| block_size_target | " << result.block_stats.block_size_target << " |\n";
    std::cout << "| num_blocks | " << result.block_stats.num_blocks << " |\n";
    std::cout << "| min_block_size | " << result.block_stats.min_block_size << " |\n";
    std::cout << "| max_block_size | " << result.block_stats.max_block_size << " |\n";
    std::cout << "| avg_block_size | " << result.block_stats.avg_block_size << " |\n";
    std::cout << "| diagonal_block_density_avg | "
              << result.block_stats.diagonal_block_density_avg << " |\n";
    std::cout << "| offblock_nnz_ratio | " << result.block_stats.offblock_nnz_ratio << " |\n\n";

    std::cout << "## 2. setup\n\n";
    std::cout << "| metric | seconds |\n|---|---:|\n";
    std::cout << "| metis_partition | " << result.timings.metis_partition_seconds << " |\n";
    std::cout << "| permutation_build | " << result.timings.permutation_build_seconds << " |\n";
    std::cout << "| setup_total | " << result.timings.setup_total_seconds << " |\n";
    std::cout << "| block_extract | " << result.timings.block_extract_seconds << " |\n";
    std::cout << "| block_lu_or_inverse | " << result.timings.block_lu_seconds << " |\n\n";

    std::cout << "## 3. solve\n\n";
    std::cout << "| metric | value |\n|---|---:|\n";
    std::cout << "| converged | " << (result.converged ? 1 : 0) << " |\n";
    std::cout << "| iterations | " << result.iterations << " |\n";
    std::cout << "| final_relative_residual | " << result.relative_residual_norm2 << " |\n";
    std::cout << "| final_absolute_residual | " << result.residual_norm2 << " |\n";
    std::cout << "| solve_total | " << result.timings.solve_total_seconds << " |\n";
    std::cout << "| gmres_loop | " << result.timings.gmres_loop_seconds << " |\n";
    std::cout << "| preconditioner_apply_total | "
              << result.timings.preconditioner_apply_seconds << " |\n";
    std::cout << "| spmv_total | " << result.timings.spmv_seconds << " |\n";
    std::cout << "| orthogonalization_total | "
              << result.timings.orthogonalization_seconds << " |\n";
    std::cout << "| dot_reduction_total | " << result.timings.dot_reduction_seconds << " |\n";
    std::cout << "| rhs_permute | " << result.timings.rhs_permute_seconds << " |\n";
    std::cout << "| solution_update | " << result.timings.solution_update_seconds << " |\n";
    std::cout << "| final_residual | " << result.timings.final_residual_seconds << " |\n";
    std::cout << "| unpermute | " << result.timings.unpermute_seconds << " |\n\n";

    std::cout << "## 4. comparison\n\n";
    std::cout << "| solver | time_sec |\n|---|---:|\n";
    std::cout << "| cuDSS factorize+solve time | n/a |\n";
    std::cout << "| BiCGSTAB+ILUT time | n/a |\n";
    std::cout << "| BiCGSTAB+ILU(k) time | n/a |\n";
    std::cout << "| GMRES+blockJacobi time | " << result.timings.solve_total_seconds << " |\n\n";
    std::cout << "stop_reason: " << result.stop_reason << "\n";
}

void print_result_csv(const cuiter::LinearSolveResult& result,
                      const CliOptions& cli,
                      const cuiter::CsrMatrix& matrix,
                      double rhs_norm)
{
    const double setup_total =
        result.timings.metis_partition_seconds +
        result.timings.permutation_build_seconds +
        result.timings.setup_total_seconds;
    std::cout << std::setprecision(12);
    std::cout << "case_name,n,nnz,restart,max_iters,rtol,preconditioner,block_size,"
                 "block_jacobi_apply,block_jacobi_precision,converged,iterations,"
                 "stop_reason,relative_residual,residual_norm,rhs_norm,metis_partition_sec,"
                 "permutation_build_sec,setup_total_sec,setup_including_analyze_sec,"
                 "block_extract_sec,block_lu_or_inverse_sec,solve_total_sec,gmres_loop_sec,"
                 "preconditioner_apply_sec,spmv_sec,orthogonalization_sec,dot_reduction_sec,"
                 "solution_update_sec,final_residual_sec,num_blocks,min_block_size,max_block_size,"
                 "avg_block_size,diagonal_block_density_avg,offblock_nnz_ratio\n";
    std::cout << cli.case_name << ","
              << matrix.rows << ","
              << matrix.nnz() << ","
              << cli.gmres_restart << ","
              << cli.gmres_max_iters << ","
              << cli.gmres_rtol << ","
              << cli.preconditioner << ","
              << cli.block_size << ","
              << cli.block_jacobi_apply << ","
              << cli.block_jacobi_precision << ","
              << (result.converged ? 1 : 0) << ","
              << result.iterations << ","
              << result.stop_reason << ","
              << result.relative_residual_norm2 << ","
              << result.residual_norm2 << ","
              << rhs_norm << ","
              << result.timings.metis_partition_seconds << ","
              << result.timings.permutation_build_seconds << ","
              << result.timings.setup_total_seconds << ","
              << setup_total << ","
              << result.timings.block_extract_seconds << ","
              << result.timings.block_lu_seconds << ","
              << result.timings.solve_total_seconds << ","
              << result.timings.gmres_loop_seconds << ","
              << result.timings.preconditioner_apply_seconds << ","
              << result.timings.spmv_seconds << ","
              << result.timings.orthogonalization_seconds << ","
              << result.timings.dot_reduction_seconds << ","
              << result.timings.solution_update_seconds << ","
              << result.timings.final_residual_seconds << ","
              << result.block_stats.num_blocks << ","
              << result.block_stats.min_block_size << ","
              << result.block_stats.max_block_size << ","
              << result.block_stats.avg_block_size << ","
              << result.block_stats.diagonal_block_density_avg << ","
              << result.block_stats.offblock_nnz_ratio << "\n";
}

void write_residual_history(const cuiter::LinearSolveResult& result,
                            const CliOptions& cli,
                            const cuiter::CsrMatrix& matrix,
                            double rhs_norm)
{
    if (cli.residual_history_path.empty()) {
        return;
    }
    if (cli.residual_history_path.has_parent_path()) {
        std::filesystem::create_directories(cli.residual_history_path.parent_path());
    }

    std::ofstream out(cli.residual_history_path);
    if (!out) {
        throw std::runtime_error("failed to open residual history file: " +
                                 cli.residual_history_path.string());
    }
    out << "case_name,n,nnz,block_size,restart,max_iters,rhs_norm,iteration,"
           "estimated_residual_norm,estimated_relative_residual\n";
    out << std::setprecision(12);
    for (std::size_t i = 0; i < result.residual_estimates.size(); ++i) {
        out << cli.case_name << ","
            << matrix.rows << ","
            << matrix.nnz() << ","
            << cli.block_size << ","
            << cli.gmres_restart << ","
            << cli.gmres_max_iters << ","
            << rhs_norm << ","
            << (i + 1) << ","
            << result.residual_estimates[i] * rhs_norm << ","
            << result.residual_estimates[i] << "\n";
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        CliOptions cli = parse_args(argc, argv);
        resolve_case_paths(cli);

        cuiter::CsrMatrix matrix;
        if (!cli.matrix_path.empty()) {
            matrix = load_matrix_auto(cli.matrix_path);
        } else {
            matrix = build_synthetic_matrix(cli.synthetic_n);
        }

        std::vector<double> rhs =
            !cli.rhs_path.empty() ? load_vector_auto(cli.rhs_path) : default_rhs(matrix.rows);
        if (static_cast<int32_t>(rhs.size()) != matrix.rows) {
            throw std::runtime_error("rhs length does not match matrix rows");
        }
        const double rhs_norm = host_norm2(rhs);

        cuiter::GmresSolverOptions solver_options;
        solver_options.max_iters = cli.gmres_max_iters;
        solver_options.restart = cli.gmres_restart;
        solver_options.rel_tolerance = cli.gmres_rtol;
        solver_options.abs_tolerance = cli.gmres_atol;
        solver_options.preconditioner = cli.preconditioner;
        solver_options.block_size = cli.block_size;
        solver_options.use_fp32_preconditioner = cli.block_jacobi_precision == "fp32";
        solver_options.block_jacobi_apply =
            cuiter::parse_block_jacobi_apply_mode(cli.block_jacobi_apply);
        solver_options.block_jacobi_diagonal_shift = cli.block_jacobi_shift;

        cuiter::GmresSolver solver(solver_options);
        solver.analyze(matrix);
        cuiter::LinearSolveResult result = solver.solve(matrix.values, rhs);
        write_residual_history(result, cli, matrix, rhs_norm);
        if (cli.csv) {
            print_result_csv(result, cli, matrix, rhs_norm);
        } else {
            print_result_table(result, cli, matrix);
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
