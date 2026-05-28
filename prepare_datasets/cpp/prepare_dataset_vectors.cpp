#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <suitesparse/klu.h>

#include "tools/matrix_io.hpp"

namespace {

struct Options {
    std::string mode;
    std::filesystem::path matrix_path;
    std::filesystem::path rhs_input_path;
    std::filesystem::path rhs_output_path;
    std::filesystem::path x_true_output_path;
    unsigned int seed = 20260521;
};

void print_usage()
{
    std::cout
        << "Usage: prepare_dataset_vectors --mode random-rhs|solve-x [options]\n\n"
        << "Options:\n"
        << "  --matrix PATH\n"
        << "  --rhs-in PATH       Required for --mode solve-x\n"
        << "  --rhs-out PATH\n"
        << "  --x-true-out PATH\n"
        << "  --seed N            Used by --mode random-rhs\n";
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

        if (arg == "--mode") {
            options.mode = next_value(arg);
        } else if (arg == "--matrix") {
            options.matrix_path = next_value(arg);
        } else if (arg == "--rhs-in") {
            options.rhs_input_path = next_value(arg);
        } else if (arg == "--rhs-out") {
            options.rhs_output_path = next_value(arg);
        } else if (arg == "--x-true-out") {
            options.x_true_output_path = next_value(arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<unsigned int>(std::stoul(next_value(arg)));
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (options.mode != "random-rhs" && options.mode != "solve-x") {
        throw std::runtime_error("--mode must be random-rhs or solve-x");
    }
    if (options.matrix_path.empty()) {
        throw std::runtime_error("--matrix is required");
    }
    if (options.rhs_output_path.empty()) {
        throw std::runtime_error("--rhs-out is required");
    }
    if (options.x_true_output_path.empty()) {
        throw std::runtime_error("--x-true-out is required");
    }
    if (options.mode == "solve-x" && options.rhs_input_path.empty()) {
        throw std::runtime_error("--rhs-in is required for --mode solve-x");
    }

    return options;
}

std::vector<sparse_direct::matrix::Value> make_random_vector(sparse_direct::matrix::Index size, unsigned int seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<sparse_direct::matrix::Value> distribution(-1.0, 1.0);

    std::vector<sparse_direct::matrix::Value> x(static_cast<std::size_t>(size));
    for (sparse_direct::matrix::Value& value : x) {
        value = distribution(generator);
    }

    return x;
}

std::vector<sparse_direct::matrix::Value> multiply(
    const sparse_direct::matrix::CsrMatrix& matrix,
    const std::vector<sparse_direct::matrix::Value>& x)
{
    matrix.validate();
    if (x.size() != static_cast<std::size_t>(matrix.cols)) {
        throw std::runtime_error("SpMV input vector size does not match matrix columns");
    }

    std::vector<sparse_direct::matrix::Value> y(static_cast<std::size_t>(matrix.rows), 0.0);
    for (sparse_direct::matrix::Index row = 0; row < matrix.rows; ++row) {
        sparse_direct::matrix::Value sum = 0.0;
        for (sparse_direct::matrix::Index pos = matrix.row_ptr[row]; pos < matrix.row_ptr[row + 1]; ++pos) {
            sum += matrix.values[pos] * x[static_cast<std::size_t>(matrix.col_idx[pos])];
        }
        y[static_cast<std::size_t>(row)] = sum;
    }

    return y;
}

std::vector<sparse_direct::matrix::Value> solve_with_klu(
    const sparse_direct::matrix::CscMatrix& csc,
    const std::vector<sparse_direct::matrix::Value>& rhs)
{
    csc.validate();
    if (csc.rows != csc.cols) {
        throw std::runtime_error("KLU solve requires a square matrix");
    }
    if (rhs.size() != static_cast<std::size_t>(csc.rows)) {
        throw std::runtime_error("RHS size does not match matrix rows");
    }

    klu_common common;
    klu_defaults(&common);

    klu_symbolic* symbolic = klu_analyze(
        csc.cols,
        const_cast<int*>(csc.col_ptr.data()),
        const_cast<int*>(csc.row_idx.data()),
        &common);
    if (!symbolic) {
        throw std::runtime_error("klu_analyze failed");
    }

    klu_numeric* numeric = klu_factor(
        const_cast<int*>(csc.col_ptr.data()),
        const_cast<int*>(csc.row_idx.data()),
        const_cast<double*>(csc.values.data()),
        symbolic,
        &common);
    if (!numeric) {
        klu_free_symbolic(&symbolic, &common);
        throw std::runtime_error("klu_factor failed");
    }

    std::vector<sparse_direct::matrix::Value> x = rhs;
    const int ok = klu_solve(symbolic, numeric, csc.cols, 1, x.data(), &common);

    klu_free_numeric(&numeric, &common);
    klu_free_symbolic(&symbolic, &common);

    if (!ok) {
        throw std::runtime_error("klu_solve failed");
    }

    return x;
}

void run_random_rhs(const Options& options)
{
    const sparse_direct::matrix::CsrMatrix matrix =
        sparse_direct::io::read_matrix_market_csr(options.matrix_path);
    const std::vector<sparse_direct::matrix::Value> x_true = make_random_vector(matrix.cols, options.seed);
    const std::vector<sparse_direct::matrix::Value> rhs = multiply(matrix, x_true);

    sparse_direct::io::write_matrix_market_vector(options.x_true_output_path, x_true);
    sparse_direct::io::write_matrix_market_vector(options.rhs_output_path, rhs);
}

void run_solve_x(const Options& options)
{
    const sparse_direct::matrix::CscMatrix matrix =
        sparse_direct::io::read_matrix_market_csc(options.matrix_path);
    const sparse_direct::io::DenseVector rhs =
        sparse_direct::io::read_matrix_market_vector(options.rhs_input_path);
    if (rhs.cols != 1) {
        throw std::runtime_error("RHS input must be a column vector");
    }

    const std::vector<sparse_direct::matrix::Value> x_true = solve_with_klu(matrix, rhs.values);

    sparse_direct::io::write_matrix_market_vector(options.rhs_output_path, rhs.values);
    sparse_direct::io::write_matrix_market_vector(options.x_true_output_path, x_true);
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_args(argc, argv);
        if (options.mode == "random-rhs") {
            run_random_rhs(options);
        } else {
            run_solve_x(options);
        }
    } catch (const std::exception& error) {
        std::cerr << "prepare_dataset_vectors: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
