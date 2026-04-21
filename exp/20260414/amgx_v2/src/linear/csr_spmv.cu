#include "csr_spmv.hpp"

#include "utils/cuda_utils.hpp"

#include <stdexcept>

namespace exp_20260414::amgx_v2 {
namespace {

constexpr int32_t kBlockSize = 256;

__global__ void csr_spmv_kernel(int32_t rows,
                                const int32_t* __restrict__ row_ptr,
                                const int32_t* __restrict__ col_idx,
                                const double* __restrict__ values,
                                const double* __restrict__ x,
                                double* __restrict__ y)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    for (int32_t pos = row_ptr[row]; pos < row_ptr[row + 1]; ++pos) {
        sum += values[pos] * x[col_idx[pos]];
    }
    y[row] = sum;
}

}  // namespace

void CsrSpmv::bind(CsrMatrixView matrix)
{
    if (matrix.rows <= 0 || matrix.nnz <= 0 || matrix.row_ptr == nullptr ||
        matrix.col_idx == nullptr || matrix.values == nullptr) {
        throw std::runtime_error("CsrSpmv::bind received an invalid matrix");
    }
    matrix_ = matrix;
}

void CsrSpmv::apply(const double* x_device, double* y_device) const
{
    if (matrix_.rows <= 0 || matrix_.row_ptr == nullptr ||
        matrix_.col_idx == nullptr || matrix_.values == nullptr) {
        throw std::runtime_error("CsrSpmv::apply called before bind");
    }
    if (x_device == nullptr || y_device == nullptr) {
        throw std::runtime_error("CsrSpmv::apply received null device input");
    }

    const int32_t grid = (matrix_.rows + kBlockSize - 1) / kBlockSize;
    csr_spmv_kernel<<<grid, kBlockSize>>>(
        matrix_.rows, matrix_.row_ptr, matrix_.col_idx, matrix_.values, x_device, y_device);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace exp_20260414::amgx_v2
