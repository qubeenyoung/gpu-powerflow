#include "io/dense_io.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliOptions {
    std::filesystem::path case_dir;
    double tolerance = 1.0e-4;
};

/// Print command-line usage for the dense IO checker.
void print_usage(const char* program)
{
    std::cerr << "Usage: " << program << " <dense_case_dir> [--tol VALUE]\n";
}

/// Parse command-line options for the dense IO checker.
CliOptions parse_args(int argc, char** argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        throw std::runtime_error("missing dense case directory");
    }

    CliOptions options;
    options.case_dir = argv[1];
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
        const std::filesystem::path A_path = options.case_dir / "A.npy";
        const std::filesystem::path b_path = options.case_dir / "b.npy";
        const std::filesystem::path x_path = options.case_dir / "x_true.npy";

        const benchio::DenseMatrix<float> A = benchio::load_dense_matrix_fp32(A_path);
        const std::vector<float> b = benchio::load_dense_vector_fp32(b_path);
        const std::vector<float> x_true = benchio::load_dense_vector_fp32(x_path);

        const double row_major_residual = benchio::relative_residual(A, x_true, b);
        const benchio::DenseMatrix<float> A_col = benchio::to_column_major_fp32(A);
        const double col_major_residual = benchio::relative_residual(A_col, x_true, b);

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "dense_case=" << options.case_dir << "\n";
        std::cout << "rows=" << A.rows << " cols=" << A.cols
                  << " layout=" << benchio::layout_name(A.layout) << "\n";
        std::cout << "relative_residual=" << row_major_residual << "\n";
        std::cout << "column_major_relative_residual=" << col_major_residual << "\n";

        if (row_major_residual > options.tolerance || col_major_residual > options.tolerance) {
            std::cerr << "dense IO check failed for " << options.case_dir
                      << ": residual exceeded tolerance " << options.tolerance << "\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "check_dense_io error: " << error.what() << "\n";
        return 1;
    }
}
