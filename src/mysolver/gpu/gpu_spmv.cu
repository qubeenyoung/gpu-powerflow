#include "mysolver/gpu/gpu_spmv.hpp"

#include <cuda_runtime.h>

namespace mysolver::gpu {

namespace {

// One thread per row (sufficient for a building-block validation kernel; the
// performance-tuned warp-per-row variant comes with the factorization work).
__global__ void csr_spmv_kernel(int n, const int* row_ptr, const int* col_idx,
                                 const double* values, const double* x, double* y)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }
    double sum = 0.0;
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        sum += values[p] * x[col_idx[p]];
    }
    y[row] = sum;
}

}  // namespace

std::vector<double> csr_spmv(int n, const std::vector<int>& row_ptr,
                             const std::vector<int>& col_idx,
                             const std::vector<double>& values,
                             const std::vector<double>& x)
{
    const int nnz = static_cast<int>(values.size());
    int *d_rp = nullptr, *d_ci = nullptr;
    double *d_v = nullptr, *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_rp, (n + 1) * sizeof(int));
    cudaMalloc(&d_ci, nnz * sizeof(int));
    cudaMalloc(&d_v, nnz * sizeof(double));
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_y, n * sizeof(double));
    cudaMemcpy(d_rp, row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    csr_spmv_kernel<<<blocks, threads>>>(n, d_rp, d_ci, d_v, d_x, d_y);

    std::vector<double> y(n, 0.0);
    cudaMemcpy(y.data(), d_y, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_rp);
    cudaFree(d_ci);
    cudaFree(d_v);
    cudaFree(d_x);
    cudaFree(d_y);
    return y;
}

}  // namespace mysolver::gpu
