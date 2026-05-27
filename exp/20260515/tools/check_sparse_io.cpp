#include "io/matrix_market_io.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path matrix_path;
    double tolerance = 1.0e-7;
};

/// Print command-line usage for the sparse IO checker.
void print_usage(const char* program)
{
    std::cerr << "Usage: " << program << " <matrix.mtx> [--tol VALUE]\n";
}

/// Parse command-line options for the sparse IO checker.
CliOptions parse_args(int argc, char** argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        throw std::runtime_error("missing Matrix Market path");
    }

    CliOptions options;
    options.matrix_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tol" && i + 1 < argc) {
            options.tolerance = std::stod(argv[++i]);
        } else {
            print_usage(argv[0]);
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        const benchio::CsrMatrix<float, int> A =
            benchio::load_matrix_market_csr_fp32(options.matrix_path);
        if (A.rows != A.cols) {
            throw std::runtime_error("sparse benchmark matrix is not square: " +
                                     options.matrix_path.string());
        }

        const std::vector<float> x_true(static_cast<std::size_t>(A.cols), 1.0f);
        const std::vector<float> b = benchio::make_rhs_from_x_true(A, x_true);
        const double residual = benchio::relative_residual(A, x_true, b);

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "matrix=" << options.matrix_path << "\n";
        std::cout << "rows=" << A.rows << " cols=" << A.cols << " nnz=" << A.nnz << "\n";
        std::cout << "relative_residual=" << residual << "\n";

        if (residual > options.tolerance) {
            std::cerr << "sparse IO check failed for " << options.matrix_path
                      << ": residual exceeded tolerance " << options.tolerance << "\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "check_sparse_io error: " << error.what() << "\n";
        return 1;
    }
}
