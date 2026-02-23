#include <Eigen/Sparse>
#include <complex>
#include <vector>
#include <string>
#include <cstdint>

// 소영수정: 템플릿 함수는 헤더에 정의 필요
#include "cnpy.h"
#include "spdlog/spdlog.h"
#include <stdexcept>

namespace pf_io
{
using EigenSparseMatCSC = Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor, int32_t>;
using EigenSparseMatCSR = Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor, int32_t>;

std::vector<std::complex<double>> load_complex_vector(const std::string& filename);
std::vector<int32_t> load_int32_vector(const std::string& filename);

// 소영수정: 템플릿 함수 정의를 헤더로 이동
template<typename EigenSparseMatT>
EigenSparseMatT load_complex_csc(const std::string& filename)
{
    cnpy::npz_t npz_map;
    try {
        npz_map = cnpy::npz_load(filename);
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading NPZ file " + filename + ": " + e.what());
    }

    auto get_array = [&](const std::string& name) {
        if (npz_map.find(name) == npz_map.end()) {
            throw std::runtime_error("NPZ " + filename + " is missing array '" + name + "'");
        }
        return npz_map.at(name);
    };

    cnpy::NpyArray arr_data = get_array("data");
    cnpy::NpyArray arr_indices = get_array("indices");
    cnpy::NpyArray arr_indptr = get_array("indptr");
    cnpy::NpyArray arr_shape = get_array("shape");

    std::complex<double>* data_ptr = arr_data.data<std::complex<double>>();
    int32_t* indices_ptr = arr_indices.data<int32_t>();
    int32_t* indptr_ptr = arr_indptr.data<int32_t>();
    int64_t* shape_ptr = arr_shape.data<int64_t>();

    long rows = shape_ptr[0];
    long cols = shape_ptr[1];
    int32_t nnz = arr_data.shape[0];

    if (arr_indptr.shape[0] != cols + 1) {
        throw std::runtime_error("NPZ 'indptr' length does not match 'cols' + 1. File does not appear to be CSC format.");
    }
    if (arr_indices.shape[0] != nnz) {
        throw std::runtime_error("NPZ 'data' and 'indices' array lengths do not match");
    }

    Eigen::Map<EigenSparseMatCSC> mapped_mat(rows, cols, nnz, indptr_ptr, indices_ptr, data_ptr);
    return EigenSparseMatT(mapped_mat);
}

template <typename T>
void save_vector(const std::string& filename, const std::vector<T>& data, const char* what = "vector") {
  try {
    const std::vector<size_t> shape{data.size()};
    cnpy::npy_save(filename, data.data(), shape, "w");
    spdlog::info("[SAVE] {} saved to {} (size={})", what, filename, data.size());
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Failed to save ") + what + " to " + filename + ": " + e.what());
  }
}

template<typename Scalar, int StorageOrder>
void save_as_csc(const std::string& filename, const Eigen::SparseMatrix<Scalar, StorageOrder, int32_t>& mat, const char* what = "matrix") {
    try {
        Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int32_t> csc = mat;
        if (!csc.isCompressed()) csc.makeCompressed();

        const Scalar* data_ptr    = csc.valuePtr();
        const int32_t* indices_ptr = csc.innerIndexPtr();
        const int32_t* indptr_ptr  = csc.outerIndexPtr();

        const size_t nnz  = csc.nonZeros();
        const size_t cols = csc.cols();

        std::vector<int64_t> shape = {csc.rows(), csc.cols()};

        cnpy::npz_save(filename, "data", data_ptr, {nnz}, "w");
        cnpy::npz_save(filename, "indices", indices_ptr, {nnz}, "a");
        cnpy::npz_save(filename, "indptr", indptr_ptr, {cols + 1}, "a");
        cnpy::npz_save(filename, "shape", shape.data(), {shape.size()}, "a");

        spdlog::info("[SAVE] {} saved to {} (nnz={})", what, filename, nnz);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to save ") + what + " to " + filename + ": " + e.what());
    }
}

} // namespace pf_io

