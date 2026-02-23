#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>

// GPU kernel to remap Eigen CSC values to CSR format
__global__ void remapEigenToCSRKernel(
    const double* __restrict__ eigen_values,  // Input: Eigen format
    double* __restrict__ csr_values,           // Output: CSR format
    const int* __restrict__ row_indices,       // Row index for each NNZ
    const int* __restrict__ col_indices,       // Col index for each NNZ
    const int* __restrict__ csr_map_keys,      // Flattened (row, col) keys
    const int64_t* __restrict__ csr_map_values, // Corresponding CSR indices
    int nnz,
    int map_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int row = row_indices[idx];
    int col = col_indices[idx];
    double val = eigen_values[idx];

    // Linear search in map (could be optimized with hash table)
    for (int i = 0; i < map_size; ++i) {
        int map_row = csr_map_keys[i * 2];
        int map_col = csr_map_keys[i * 2 + 1];
        if (map_row == row && map_col == col) {
            csr_values[csr_map_values[i]] = val;
            return;
        }
    }
}

extern "C" void launchRemapKernel(
    const double* d_eigen_vals,
    double* d_csr_vals,
    const int* d_row_idx,
    const int* d_col_idx,
    const int* d_map_keys,
    const int64_t* d_map_vals,
    int nnz,
    int map_size
) {
    int block = 256;
    int grid = (nnz + block - 1) / block;
    remapEigenToCSRKernel<<<grid, block>>>(
        d_eigen_vals, d_csr_vals, d_row_idx, d_col_idx,
        d_map_keys, d_map_vals, nnz, map_size
    );
    cudaDeviceSynchronize();
}
