#include "matrix/pattern_kernels.hpp"

#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace custom_linear_solver::matrix {
namespace {

Status cuda_status(cudaError_t err)
{
    return err == cudaSuccess ? Status::Success : Status::AnalysisFailed;
}

__global__ void count_csr_columns(int nnz, const int* __restrict__ csr_col_idx,
                                  int* __restrict__ shifted_counts)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < nnz) atomicAdd(&shifted_counts[csr_col_idx[p] + 1], 1);
}

__global__ void scatter_csr_to_csc(int n, const int* __restrict__ csr_row_ptr,
                                   const int* __restrict__ csr_col_idx,
                                   int* __restrict__ next_col, int* __restrict__ csc_row_idx,
                                   int* __restrict__ source_pos)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    for (int p = csr_row_ptr[row]; p < csr_row_ptr[row + 1]; ++p) {
        const int col = csr_col_idx[p];
        const int q = atomicAdd(&next_col[col], 1);
        csc_row_idx[q] = row;
        source_pos[q] = p;
    }
}

__global__ void count_permuted_columns(int n, const int* __restrict__ col_ptr,
                                       const int* __restrict__ iperm,
                                       int* __restrict__ shifted_counts)
{
    const int old_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (old_col >= n) return;
    const int new_col = iperm[old_col];
    shifted_counts[new_col + 1] = col_ptr[old_col + 1] - col_ptr[old_col];
}

__global__ void fill_permuted_csc(int n, const int* __restrict__ in_col_ptr,
                                  const int* __restrict__ in_row_idx,
                                  const int* __restrict__ in_source_pos,
                                  const int* __restrict__ iperm,
                                  const int* __restrict__ out_col_ptr,
                                  int* __restrict__ out_row_idx,
                                  int* __restrict__ out_source_pos)
{
    const int old_col = blockIdx.x;
    if (old_col >= n) return;
    const int new_col = iperm[old_col];
    const int in_begin = in_col_ptr[old_col];
    const int in_end = in_col_ptr[old_col + 1];
    const int out_begin = out_col_ptr[new_col];
    for (int offset = threadIdx.x; offset < in_end - in_begin; offset += blockDim.x) {
        const int in_pos = in_begin + offset;
        const int out_pos = out_begin + offset;
        out_row_idx[out_pos] = iperm[in_row_idx[in_pos]];
        out_source_pos[out_pos] = in_source_pos[in_pos];
    }
}

}  // namespace

IntDeviceBuffer::~IntDeviceBuffer()
{
    reset();
}

IntDeviceBuffer::IntDeviceBuffer(IntDeviceBuffer&& other) noexcept
    : ptr(other.ptr), count(other.count)
{
    other.ptr = nullptr;
    other.count = 0;
}

IntDeviceBuffer& IntDeviceBuffer::operator=(IntDeviceBuffer&& other) noexcept
{
    if (this != &other) {
        reset();
        ptr = other.ptr;
        count = other.count;
        other.ptr = nullptr;
        other.count = 0;
    }
    return *this;
}

void IntDeviceBuffer::reset()
{
    if (ptr) cudaFree(ptr);
    ptr = nullptr;
    count = 0;
}

Status IntDeviceBuffer::allocate(std::size_t values)
{
    reset();
    count = values;
    if (values == 0) return Status::Success;
    if (cudaMalloc(reinterpret_cast<void**>(&ptr), values * sizeof(int)) != cudaSuccess) {
        ptr = nullptr;
        count = 0;
        return Status::AllocationFailed;
    }
    return Status::Success;
}

Status IntDeviceBuffer::upload(const std::vector<int>& values)
{
    Status st = allocate(values.size());
    if (st != Status::Success || values.empty()) return st;
    return cuda_status(cudaMemcpy(ptr, values.data(), values.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
}

Status build_csc_from_csr_device(int n, int nnz, const int* d_csr_row_ptr,
                                 const int* d_csr_col_idx, DeviceCscPattern& csc)
{
    if (n <= 0 || nnz < 0 || d_csr_row_ptr == nullptr || d_csr_col_idx == nullptr)
        return Status::InvalidValue;

    csc.col_ptr.reset();
    csc.row_idx.reset();
    csc.source_pos.reset();
    csc.n = n;
    csc.nnz = nnz;

    Status st = csc.col_ptr.allocate(static_cast<std::size_t>(n) + 1);
    if (st != Status::Success) return st;
    st = csc.row_idx.allocate(static_cast<std::size_t>(nnz));
    if (st != Status::Success) return st;
    st = csc.source_pos.allocate(static_cast<std::size_t>(nnz));
    if (st != Status::Success) return st;

    IntDeviceBuffer counts;
    st = counts.allocate(static_cast<std::size_t>(n) + 1);
    if (st != Status::Success) return st;
    if (cudaMemset(counts.ptr, 0, (static_cast<std::size_t>(n) + 1) * sizeof(int)) !=
        cudaSuccess)
        return Status::AnalysisFailed;

    constexpr int threads = 256;
    count_csr_columns<<<(nnz + threads - 1) / threads, threads>>>(nnz, d_csr_col_idx,
                                                                  counts.ptr);
    if (cudaGetLastError() != cudaSuccess) return Status::AnalysisFailed;

    thrust::inclusive_scan(thrust::device, counts.ptr, counts.ptr + n + 1, csc.col_ptr.ptr);

    IntDeviceBuffer next_col;
    st = next_col.allocate(static_cast<std::size_t>(n) + 1);
    if (st != Status::Success) return st;
    if (cudaMemcpy(next_col.ptr, csc.col_ptr.ptr, (static_cast<std::size_t>(n) + 1) * sizeof(int),
                   cudaMemcpyDeviceToDevice) != cudaSuccess)
        return Status::AnalysisFailed;

    scatter_csr_to_csc<<<(n + threads - 1) / threads, threads>>>(
        n, d_csr_row_ptr, d_csr_col_idx, next_col.ptr, csc.row_idx.ptr, csc.source_pos.ptr);
    return cuda_status(cudaGetLastError());
}

Status permute_csc_device(const DeviceCscPattern& csc, const int* d_iperm,
                          DeviceCscPattern& ordered)
{
    if (csc.n <= 0 || csc.nnz < 0 || csc.col_ptr.ptr == nullptr || csc.row_idx.ptr == nullptr ||
        csc.source_pos.ptr == nullptr || d_iperm == nullptr)
        return Status::InvalidValue;

    ordered.col_ptr.reset();
    ordered.row_idx.reset();
    ordered.source_pos.reset();
    ordered.n = csc.n;
    ordered.nnz = csc.nnz;

    Status st = ordered.col_ptr.allocate(static_cast<std::size_t>(csc.n) + 1);
    if (st != Status::Success) return st;
    st = ordered.row_idx.allocate(static_cast<std::size_t>(csc.nnz));
    if (st != Status::Success) return st;
    st = ordered.source_pos.allocate(static_cast<std::size_t>(csc.nnz));
    if (st != Status::Success) return st;

    IntDeviceBuffer counts;
    st = counts.allocate(static_cast<std::size_t>(csc.n) + 1);
    if (st != Status::Success) return st;
    if (cudaMemset(counts.ptr, 0, (static_cast<std::size_t>(csc.n) + 1) * sizeof(int)) !=
        cudaSuccess)
        return Status::AnalysisFailed;

    constexpr int threads = 256;
    count_permuted_columns<<<(csc.n + threads - 1) / threads, threads>>>(
        csc.n, csc.col_ptr.ptr, d_iperm, counts.ptr);
    if (cudaGetLastError() != cudaSuccess) return Status::AnalysisFailed;

    thrust::inclusive_scan(thrust::device, counts.ptr, counts.ptr + csc.n + 1,
                           ordered.col_ptr.ptr);

    fill_permuted_csc<<<csc.n, 128>>>(csc.n, csc.col_ptr.ptr, csc.row_idx.ptr,
                                      csc.source_pos.ptr, d_iperm, ordered.col_ptr.ptr,
                                      ordered.row_idx.ptr, ordered.source_pos.ptr);
    return cuda_status(cudaGetLastError());
}

Status download_csc_structure(const DeviceCscPattern& csc, std::vector<int>& col_ptr,
                              std::vector<int>& row_idx)
{
    if (csc.n <= 0 || csc.nnz < 0 || csc.col_ptr.ptr == nullptr || csc.row_idx.ptr == nullptr)
        return Status::InvalidValue;

    col_ptr.resize(static_cast<std::size_t>(csc.n) + 1);
    row_idx.resize(static_cast<std::size_t>(csc.nnz));
    if (cudaMemcpy(col_ptr.data(), csc.col_ptr.ptr,
                   (static_cast<std::size_t>(csc.n) + 1) * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return Status::AnalysisFailed;
    if (csc.nnz > 0 &&
        cudaMemcpy(row_idx.data(), csc.row_idx.ptr,
                   static_cast<std::size_t>(csc.nnz) * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return Status::AnalysisFailed;
    return Status::Success;
}

}  // namespace custom_linear_solver::matrix
