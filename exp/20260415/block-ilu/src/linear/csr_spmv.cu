#include "linear/csr_spmv.hpp"

#include "utils/cuda_utils.hpp"

#include <stdexcept>

namespace exp_20260415::block_ilu {
namespace {

constexpr int32_t kBlockSize = 256;

__global__ void csr_spmv_range_kernel(int32_t row_begin,
                                      int32_t row_end,
                                const int32_t* __restrict__ row_ptr,
                                const int32_t* __restrict__ col_idx,
                                const double* __restrict__ values,
                                const double* __restrict__ x,
                                double* __restrict__ y)
{
    const int32_t row = row_begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_end) {
        return;
    }

    double sum = 0.0;
    for (int32_t pos = row_ptr[row]; pos < row_ptr[row + 1]; ++pos) {
        sum += values[pos] * x[col_idx[pos]];
    }
    y[row] = sum;
}

__global__ void csr_spmv_range_kernel_f32(int32_t row_begin,
                                          int32_t row_end,
                                          const int32_t* __restrict__ row_ptr,
                                          const int32_t* __restrict__ col_idx,
                                          const float* __restrict__ values,
                                          const float* __restrict__ x,
                                          float* __restrict__ y)
{
    const int32_t row = row_begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_end) {
        return;
    }

    float sum = 0.0f;
    for (int32_t pos = row_ptr[row]; pos < row_ptr[row + 1]; ++pos) {
        sum += values[pos] * x[col_idx[pos]];
    }
    y[row] = sum;
}

}  // namespace

CsrSpmv::CsrSpmv()
{
    CUDA_CHECK(cudaStreamCreateWithFlags(&top_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&bottom_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&input_ready_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&top_done_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&bottom_done_, cudaEventDisableTiming));
}

CsrSpmv::~CsrSpmv()
{
    if (top_stream_ != nullptr) {
        cudaStreamSynchronize(top_stream_);
    }
    if (bottom_stream_ != nullptr) {
        cudaStreamSynchronize(bottom_stream_);
    }
    if (top_done_ != nullptr) {
        cudaEventDestroy(top_done_);
        top_done_ = nullptr;
    }
    if (bottom_done_ != nullptr) {
        cudaEventDestroy(bottom_done_);
        bottom_done_ = nullptr;
    }
    if (input_ready_ != nullptr) {
        cudaEventDestroy(input_ready_);
        input_ready_ = nullptr;
    }
    if (top_stream_ != nullptr) {
        cudaStreamDestroy(top_stream_);
        top_stream_ = nullptr;
    }
    if (bottom_stream_ != nullptr) {
        cudaStreamDestroy(bottom_stream_);
        bottom_stream_ = nullptr;
    }
}

void CsrSpmv::bind(DeviceCsrMatrixView matrix)
{
    if (matrix.rows <= 0 || matrix.cols <= 0 || matrix.nnz <= 0 ||
        matrix.row_ptr == nullptr || matrix.col_idx == nullptr || matrix.values == nullptr) {
        throw std::runtime_error("CsrSpmv::bind received an invalid matrix");
    }
    matrix_ = matrix;
    row_split_ = 0;
}

void CsrSpmv::enable_parallel_row_split(int32_t row_split)
{
    if (matrix_.rows <= 0) {
        throw std::runtime_error("CsrSpmv::enable_parallel_row_split called before bind");
    }
    if (row_split <= 0 || row_split >= matrix_.rows) {
        throw std::runtime_error("CsrSpmv::enable_parallel_row_split received invalid split");
    }
    row_split_ = row_split;
}

void CsrSpmv::disable_parallel_row_split()
{
    row_split_ = 0;
}

void CsrSpmv::apply_async(const double* x_device, double* y_device, cudaStream_t stream) const
{
    if (matrix_.rows <= 0 || matrix_.row_ptr == nullptr ||
        matrix_.col_idx == nullptr || matrix_.values == nullptr) {
        throw std::runtime_error("CsrSpmv::apply_async called before bind");
    }
    if (x_device == nullptr || y_device == nullptr) {
        throw std::runtime_error("CsrSpmv::apply_async received null device input");
    }
    if (row_split_ > 0) {
        throw std::runtime_error("CsrSpmv::apply_async does not support row-split mode");
    }

    const int32_t grid = (matrix_.rows + kBlockSize - 1) / kBlockSize;
    csr_spmv_range_kernel<<<grid, kBlockSize, 0, stream>>>(
        0, matrix_.rows, matrix_.row_ptr, matrix_.col_idx, matrix_.values, x_device, y_device);
    CUDA_CHECK(cudaGetLastError());
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

    if (row_split_ > 0) {
        CUDA_CHECK(cudaEventRecord(input_ready_, nullptr));
        CUDA_CHECK(cudaStreamWaitEvent(top_stream_, input_ready_, 0));
        CUDA_CHECK(cudaStreamWaitEvent(bottom_stream_, input_ready_, 0));

        const int32_t top_rows = row_split_;
        const int32_t bottom_rows = matrix_.rows - row_split_;
        const int32_t top_grid = (top_rows + kBlockSize - 1) / kBlockSize;
        const int32_t bottom_grid = (bottom_rows + kBlockSize - 1) / kBlockSize;
        csr_spmv_range_kernel<<<top_grid, kBlockSize, 0, top_stream_>>>(
            0, row_split_, matrix_.row_ptr, matrix_.col_idx, matrix_.values, x_device, y_device);
        csr_spmv_range_kernel<<<bottom_grid, kBlockSize, 0, bottom_stream_>>>(
            row_split_,
            matrix_.rows,
            matrix_.row_ptr,
            matrix_.col_idx,
            matrix_.values,
            x_device,
            y_device);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(top_done_, top_stream_));
        CUDA_CHECK(cudaEventRecord(bottom_done_, bottom_stream_));
        CUDA_CHECK(cudaStreamWaitEvent(nullptr, top_done_, 0));
        CUDA_CHECK(cudaStreamWaitEvent(nullptr, bottom_done_, 0));
        return;
    }

    const int32_t grid = (matrix_.rows + kBlockSize - 1) / kBlockSize;
    csr_spmv_range_kernel<<<grid, kBlockSize>>>(
        0, matrix_.rows, matrix_.row_ptr, matrix_.col_idx, matrix_.values, x_device, y_device);
    CUDA_CHECK(cudaGetLastError());
}

CsrSpmvF32::CsrSpmvF32() = default;

CsrSpmvF32::~CsrSpmvF32() = default;

void CsrSpmvF32::bind(DeviceCsrMatrixViewF32 matrix)
{
    if (matrix.rows <= 0 || matrix.cols <= 0 || matrix.nnz <= 0 ||
        matrix.row_ptr == nullptr || matrix.col_idx == nullptr || matrix.values == nullptr) {
        throw std::runtime_error("CsrSpmvF32::bind received an invalid matrix");
    }
    matrix_ = matrix;
}

void CsrSpmvF32::apply_async(const float* x_device, float* y_device, cudaStream_t stream) const
{
    if (matrix_.rows <= 0 || matrix_.row_ptr == nullptr ||
        matrix_.col_idx == nullptr || matrix_.values == nullptr) {
        throw std::runtime_error("CsrSpmvF32::apply_async called before bind");
    }
    if (x_device == nullptr || y_device == nullptr) {
        throw std::runtime_error("CsrSpmvF32::apply_async received null device input");
    }

    const int32_t grid = (matrix_.rows + kBlockSize - 1) / kBlockSize;
    csr_spmv_range_kernel_f32<<<grid, kBlockSize, 0, stream>>>(
        0, matrix_.rows, matrix_.row_ptr, matrix_.col_idx, matrix_.values, x_device, y_device);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace exp_20260415::block_ilu
