#include "tools/compute_error.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "tools/matrix_io.hpp"

namespace sparse_direct::error {
namespace {

std::vector<double> read_column_vector(
    const std::filesystem::path& path,
    int expected_size,
    const std::string& label)
{
    const io::DenseVector vector = io::read_matrix_market_vector(path);
    if (vector.cols != 1) {
        throw std::runtime_error(label + " must be a column vector: " + path.string());
    }
    if (vector.size() != expected_size) {
        throw std::runtime_error(label + " size does not match expected dimension: " + path.string());
    }
    return vector.values;
}

std::vector<double> difference(std::vector<double> lhs, const std::vector<double>& rhs)
{
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("vector sizes do not match");
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] -= rhs[i];
    }
    return lhs;
}

void ensure_finite_vector(const std::vector<double>& values, const std::string& label)
{
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            throw std::runtime_error(label + " contains a non-finite value at index " + std::to_string(i));
        }
    }
}

std::vector<double> multiply(const matrix::CsrMatrix& matrix, const std::vector<double>& x)
{
    matrix.validate();
    if (x.size() != static_cast<std::size_t>(matrix.cols)) {
        throw std::runtime_error("SpMV input vector size does not match matrix columns");
    }

    std::vector<double> y(static_cast<std::size_t>(matrix.rows), 0.0);
    for (matrix::Index row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (matrix::Index pos = matrix.row_ptr[row]; pos < matrix.row_ptr[row + 1]; ++pos) {
            sum += matrix.values[pos] * x[static_cast<std::size_t>(matrix.col_idx[pos])];
        }
        y[static_cast<std::size_t>(row)] = sum;
    }

    return y;
}

double norm2(const std::vector<double>& x)
{
    long double sum = 0.0;
    for (double value : x) {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
}

double componentwise_backward_error(
    const matrix::CsrMatrix& matrix,
    const std::vector<double>& x,
    const std::vector<double>& rhs,
    const std::vector<double>& residual)
{
    double berr = 0.0;
    for (matrix::Index row = 0; row < matrix.rows; ++row) {
        double denominator = std::abs(rhs[static_cast<std::size_t>(row)]);
        for (matrix::Index pos = matrix.row_ptr[row]; pos < matrix.row_ptr[row + 1]; ++pos) {
            denominator += std::abs(matrix.values[pos]) * std::abs(x[static_cast<std::size_t>(matrix.col_idx[pos])]);
        }

        const double numerator = std::abs(residual[static_cast<std::size_t>(row)]);
        if (denominator > 0.0) {
            berr = std::max(berr, numerator / denominator);
        } else if (numerator > 0.0) {
            berr = std::numeric_limits<double>::infinity();
        }
    }
    return berr;
}

}  // namespace

ErrorMetrics compute_error(
    const matrix::CsrMatrix& matrix,
    const std::vector<double>& rhs,
    const std::vector<double>& x,
    const std::vector<double>& x_true)
{
    if (rhs.size() != static_cast<std::size_t>(matrix.rows)) {
        throw std::runtime_error("rhs size does not match matrix rows");
    }
    if (x.size() != static_cast<std::size_t>(matrix.cols)) {
        throw std::runtime_error("x size does not match matrix columns");
    }
    if (x_true.size() != static_cast<std::size_t>(matrix.cols)) {
        throw std::runtime_error("x_true size does not match matrix columns");
    }

    ensure_finite_vector(rhs, "rhs");
    ensure_finite_vector(x, "x");
    ensure_finite_vector(x_true, "x_true");

    const std::vector<double> ax = multiply(matrix, x);
    ensure_finite_vector(ax, "A*x");
    const std::vector<double> residual = difference(ax, rhs);
    ensure_finite_vector(residual, "residual");
    const std::vector<double> error = difference(x, x_true);
    ensure_finite_vector(error, "solution error");

    ErrorMetrics metrics;
    metrics.berr = componentwise_backward_error(matrix, x, rhs, residual);
    metrics.absolute_error = norm2(error);
    return metrics;
}

ErrorMetrics compute_error(
    const std::filesystem::path& matrix_path,
    const std::filesystem::path& rhs_path,
    const std::filesystem::path& x_path,
    const std::filesystem::path& x_true_path)
{
    const matrix::CsrMatrix matrix = io::read_matrix_market_csr(matrix_path);
    const std::vector<double> rhs = read_column_vector(rhs_path, matrix.rows, "rhs");
    const std::vector<double> x = read_column_vector(x_path, matrix.cols, "x");
    const std::vector<double> x_true = read_column_vector(x_true_path, matrix.cols, "x_true");

    return compute_error(matrix, rhs, x, x_true);
}

}  // namespace sparse_direct::error
