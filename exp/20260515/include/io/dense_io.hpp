#pragma once

#include "io/linear_system.hpp"

#include <filesystem>
#include <vector>

namespace benchio {

/// Load a 2-D little-endian fp32 NumPy .npy file as a dense matrix.
DenseMatrix<float> load_dense_matrix_fp32(const std::filesystem::path& path);

/// Load a 1-D or single-column/single-row little-endian fp32 NumPy .npy file.
std::vector<float> load_dense_vector_fp32(const std::filesystem::path& path);

/// Convert an fp32 dense matrix to column-major storage for cuSolverDN calls.
DenseMatrix<float> to_column_major_fp32(const DenseMatrix<float>& matrix);

}  // namespace benchio
