#pragma once

#include <filesystem>
#include <vector>

#include "matrix/sparse_matrix.hpp"

namespace sparse_direct::error {

struct ErrorMetrics {
    double berr = 0.0;
    double absolute_error = 0.0;
};

ErrorMetrics compute_error(
    const matrix::CsrMatrix& matrix,
    const std::vector<double>& rhs,
    const std::vector<double>& x,
    const std::vector<double>& x_true);

ErrorMetrics compute_error(
    const std::filesystem::path& matrix_path,
    const std::filesystem::path& rhs_path,
    const std::filesystem::path& x_path,
    const std::filesystem::path& x_true_path);

}  // namespace sparse_direct::error
