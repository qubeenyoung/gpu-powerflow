#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "matrix/sparse_matrix.hpp"

namespace sparse_direct::io {

struct MatrixMarketInfo {
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;
};

struct DenseVector {
    matrix::Index rows = 0;
    matrix::Index cols = 0;
    std::vector<matrix::Value> values;

    matrix::Index size() const
    {
        return static_cast<matrix::Index>(values.size());
    }
};

matrix::CsrMatrix read_matrix_market_csr(const std::filesystem::path& path);
matrix::CscMatrix read_matrix_market_csc(const std::filesystem::path& path);
DenseVector read_matrix_market_vector(const std::filesystem::path& path);
void write_matrix_market_vector(
    const std::filesystem::path& path,
    const std::vector<matrix::Value>& values);

}  // namespace sparse_direct::io
