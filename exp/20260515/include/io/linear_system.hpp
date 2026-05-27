#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace benchio {

enum class MatrixLayout {
    RowMajor,
    ColMajor
};

/// Return a stable printable name for a dense matrix storage layout.
inline const char* layout_name(MatrixLayout layout) noexcept
{
    return layout == MatrixLayout::RowMajor ? "row-major" : "column-major";
}

template <typename T>
struct DenseMatrix {
    using value_type = T;

    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<T> values;
    MatrixLayout layout = MatrixLayout::RowMajor;

    /// Return the number of scalar entries expected in the dense matrix.
    std::size_t size() const
    {
        return rows * cols;
    }

    /// Read one dense matrix entry using the recorded storage layout.
    const T& at(std::size_t row, std::size_t col) const
    {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("DenseMatrix index is out of range");
        }
        const std::size_t offset =
            layout == MatrixLayout::RowMajor ? row * cols + col : col * rows + row;
        return values[offset];
    }

    /// Write one dense matrix entry using the recorded storage layout.
    T& at(std::size_t row, std::size_t col)
    {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("DenseMatrix index is out of range");
        }
        const std::size_t offset =
            layout == MatrixLayout::RowMajor ? row * cols + col : col * rows + row;
        return values[offset];
    }
};

template <typename T, typename IndexT = int>
struct CsrMatrix {
    using value_type = T;
    using index_type = IndexT;

    IndexT rows = 0;
    IndexT cols = 0;
    IndexT nnz = 0;
    std::vector<IndexT> row_ptr;
    std::vector<IndexT> col_ind;
    std::vector<T> values;
};

template <typename T, typename MatrixT = DenseMatrix<T>>
struct LinearSystem {
    using value_type = T;
    using matrix_type = MatrixT;

    MatrixT A;
    std::vector<T> b;
    std::optional<std::vector<T>> x_true;
};

/// Validate that a dense matrix has a complete value array.
template <typename T>
inline void validate_dense_matrix(const DenseMatrix<T>& matrix, const std::string& name)
{
    if (matrix.rows == 0 || matrix.cols == 0) {
        throw std::runtime_error(name + " has zero rows or columns");
    }
    if (matrix.values.size() != matrix.size()) {
        throw std::runtime_error(name + " value count does not match rows * cols");
    }
}

/// Validate CSR structural arrays and their dimensions.
template <typename T, typename IndexT>
inline void validate_csr_matrix(const CsrMatrix<T, IndexT>& matrix, const std::string& name)
{
    if (matrix.rows <= 0 || matrix.cols <= 0) {
        throw std::runtime_error(name + " has non-positive rows or columns");
    }
    if (matrix.nnz < 0) {
        throw std::runtime_error(name + " has a negative nnz field");
    }
    if (matrix.row_ptr.size() != static_cast<std::size_t>(matrix.rows + 1)) {
        throw std::runtime_error(name + " row_ptr size does not equal rows + 1");
    }
    if (matrix.col_ind.size() != static_cast<std::size_t>(matrix.nnz) ||
        matrix.values.size() != static_cast<std::size_t>(matrix.nnz)) {
        throw std::runtime_error(name + " CSR nnz does not match col_ind/value sizes");
    }
    if (matrix.row_ptr.empty() || matrix.row_ptr.front() != 0 ||
        matrix.row_ptr.back() != matrix.nnz) {
        throw std::runtime_error(name + " CSR row_ptr endpoints are invalid");
    }
    for (IndexT row = 0; row < matrix.rows; ++row) {
        const IndexT begin = matrix.row_ptr[static_cast<std::size_t>(row)];
        const IndexT end = matrix.row_ptr[static_cast<std::size_t>(row + 1)];
        if (begin > end || begin < 0 || end > matrix.nnz) {
            throw std::runtime_error(name + " CSR row_ptr is not monotone");
        }
    }
    for (IndexT col : matrix.col_ind) {
        if (col < 0 || col >= matrix.cols) {
            throw std::runtime_error(name + " CSR column index is out of range");
        }
    }
}

/// Convert any dense matrix layout to column-major storage for cuSolverDN.
template <typename T>
inline DenseMatrix<T> to_column_major(const DenseMatrix<T>& matrix)
{
    validate_dense_matrix(matrix, "to_column_major input");
    if (matrix.layout == MatrixLayout::ColMajor) {
        return matrix;
    }

    DenseMatrix<T> out;
    out.rows = matrix.rows;
    out.cols = matrix.cols;
    out.layout = MatrixLayout::ColMajor;
    out.values.resize(matrix.size());

    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            out.values[col * matrix.rows + row] = matrix.values[row * matrix.cols + col];
        }
    }
    return out;
}

/// Compute y = A * x for a dense matrix, returning fp32/fp64 values in T.
template <typename T>
inline std::vector<T> dense_matvec(const DenseMatrix<T>& matrix, const std::vector<T>& x)
{
    validate_dense_matrix(matrix, "dense_matvec matrix");
    if (x.size() != matrix.cols) {
        throw std::runtime_error("dense_matvec x length does not match matrix columns");
    }

    std::vector<double> accum(matrix.rows, 0.0);
    if (matrix.layout == MatrixLayout::RowMajor) {
        for (std::size_t row = 0; row < matrix.rows; ++row) {
            double sum = 0.0;
            const std::size_t row_offset = row * matrix.cols;
            for (std::size_t col = 0; col < matrix.cols; ++col) {
                sum += static_cast<double>(matrix.values[row_offset + col]) *
                       static_cast<double>(x[col]);
            }
            accum[row] = sum;
        }
    } else {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            const double x_value = static_cast<double>(x[col]);
            const std::size_t col_offset = col * matrix.rows;
            for (std::size_t row = 0; row < matrix.rows; ++row) {
                accum[row] += static_cast<double>(matrix.values[col_offset + row]) * x_value;
            }
        }
    }

    std::vector<T> y(matrix.rows);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        y[row] = static_cast<T>(accum[row]);
    }
    return y;
}

/// Compute y = A * x for a CSR matrix, returning fp32/fp64 values in T.
template <typename T, typename IndexT>
inline std::vector<T> csr_matvec(const CsrMatrix<T, IndexT>& matrix, const std::vector<T>& x)
{
    validate_csr_matrix(matrix, "csr_matvec matrix");
    if (x.size() != static_cast<std::size_t>(matrix.cols)) {
        throw std::runtime_error("csr_matvec x length does not match matrix columns");
    }

    std::vector<T> y(static_cast<std::size_t>(matrix.rows), T{});
    for (IndexT row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        const IndexT begin = matrix.row_ptr[static_cast<std::size_t>(row)];
        const IndexT end = matrix.row_ptr[static_cast<std::size_t>(row + 1)];
        for (IndexT pos = begin; pos < end; ++pos) {
            const std::size_t idx = static_cast<std::size_t>(pos);
            sum += static_cast<double>(matrix.values[idx]) *
                   static_cast<double>(x[static_cast<std::size_t>(matrix.col_ind[idx])]);
        }
        y[static_cast<std::size_t>(row)] = static_cast<T>(sum);
    }
    return y;
}

/// Compute the Euclidean norm of a vector using double accumulation.
template <typename T>
inline double norm2(const std::vector<T>& values)
{
    double sum = 0.0;
    for (const T value : values) {
        const double v = static_cast<double>(value);
        sum += v * v;
    }
    return std::sqrt(sum);
}

/// Compute ||ax - b||_2 / ||b||_2 from an already evaluated product.
template <typename T>
inline double relative_residual_from_product(const std::vector<T>& ax,
                                             const std::vector<T>& b)
{
    if (ax.size() != b.size()) {
        throw std::runtime_error("relative_residual vector sizes do not match");
    }
    std::vector<double> residual(ax.size(), 0.0);
    for (std::size_t i = 0; i < ax.size(); ++i) {
        residual[i] = static_cast<double>(ax[i]) - static_cast<double>(b[i]);
    }
    const double b_norm = norm2(b);
    const double denom = std::max(b_norm, 1.0e-30);
    return norm2(residual) / denom;
}

/// Compute ||A*x - b||_2 / ||b||_2 for a dense matrix.
template <typename T>
inline double relative_residual(const DenseMatrix<T>& matrix,
                                const std::vector<T>& x,
                                const std::vector<T>& b)
{
    return relative_residual_from_product(dense_matvec(matrix, x), b);
}

/// Compute ||A*x - b||_2 / ||b||_2 for a CSR matrix.
template <typename T, typename IndexT>
inline double relative_residual(const CsrMatrix<T, IndexT>& matrix,
                                const std::vector<T>& x,
                                const std::vector<T>& b)
{
    return relative_residual_from_product(csr_matvec(matrix, x), b);
}

/// Build b = A*x_true for a dense matrix.
template <typename T>
inline std::vector<T> make_rhs_from_x_true(const DenseMatrix<T>& matrix,
                                           const std::vector<T>& x_true)
{
    return dense_matvec(matrix, x_true);
}

/// Build b = A*x_true for a CSR matrix.
template <typename T, typename IndexT>
inline std::vector<T> make_rhs_from_x_true(const CsrMatrix<T, IndexT>& matrix,
                                           const std::vector<T>& x_true)
{
    return csr_matvec(matrix, x_true);
}

}  // namespace benchio
