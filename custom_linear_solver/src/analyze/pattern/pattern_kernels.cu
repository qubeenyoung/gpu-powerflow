#include "analyze/pattern/pattern_kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

// ============================================================================
// Analyze-phase device pattern kernels: device CSR -> CSC, fill-reducing permutation of a CSC
// pattern, and the symmetric (A+Aᵀ) adjacency graph METIS consumes. All run on the GPU so the
// only host transfers in analyze are the small permutation arrays. source_pos[] threads each
// CSC entry back to its CSR slot so factorize can scatter values without rebuilding the map.
// ============================================================================

namespace custom_linear_solver::matrix {
namespace {

Status cuda_status(cudaError_t err)
{
    return err == cudaSuccess ? Status::kSuccess : Status::kAnalysisFailed;
}

// Tally per-column nnz into shifted_counts[col+1] (an inclusive scan later makes col_ptr).
__global__ void count_csr_columns(int nnz, const int* __restrict__ csr_col_idx,
                                  int* __restrict__ shifted_counts)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < nnz) atomicAdd(&shifted_counts[csr_col_idx[p] + 1], 1);
}

// Scatter CSR entries into CSC slots (next_col = running write cursor per column); records the
// source CSR position of each CSC entry in source_pos.
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

// Per-column size of the permuted matrix: new column iperm[old] keeps old column's nnz.
__global__ void count_permuted_columns(int n, const int* __restrict__ col_ptr,
                                       const int* __restrict__ iperm,
                                       int* __restrict__ shifted_counts)
{
    const int old_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (old_col >= n) return;
    const int new_col = iperm[old_col];
    shifted_counts[new_col + 1] = col_ptr[old_col + 1] - col_ptr[old_col];
}

// Per-column size for a row/column permutation: the new column is d_col_iperm[old_col].
__global__ void count_permuted_columns_rc(int n, const int* __restrict__ col_ptr,
                                          const int* __restrict__ col_iperm,
                                          int* __restrict__ shifted_counts)
{
    const int old_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (old_col >= n) return;
    const int new_col = col_iperm[old_col];
    shifted_counts[new_col + 1] = col_ptr[old_col + 1] - col_ptr[old_col];
}

// Copy old column old_col into new column iperm[old_col], remapping row indices through iperm and
// carrying source_pos along. One block per column.
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

// Copy old column old_col into new column col_iperm[old_col], remapping rows through row_iperm.
// This supports static structural matching where row and column permutations differ.
__global__ void fill_permuted_csc_rc(int n, const int* __restrict__ in_col_ptr,
                                     const int* __restrict__ in_row_idx,
                                     const int* __restrict__ in_source_pos,
                                     const int* __restrict__ row_iperm,
                                     const int* __restrict__ col_iperm,
                                     const int* __restrict__ out_col_ptr,
                                     int* __restrict__ out_row_idx,
                                     int* __restrict__ out_source_pos)
{
    const int old_col = blockIdx.x;
    if (old_col >= n) return;
    const int new_col = col_iperm[old_col];
    const int in_begin = in_col_ptr[old_col];
    const int in_end = in_col_ptr[old_col + 1];
    const int out_begin = out_col_ptr[new_col];
    for (int offset = threadIdx.x; offset < in_end - in_begin; offset += blockDim.x) {
        const int in_pos = in_begin + offset;
        const int out_pos = out_begin + offset;
        out_row_idx[out_pos] = row_iperm[in_row_idx[in_pos]];
        out_source_pos[out_pos] = in_source_pos[in_pos];
    }
}

// Emit both directed edge keys (row*N+col and col*N+row) for every off-diagonal CSC
// entry; diagonal/invalid entries get a sentinel that sorts to the very end. One block
// per column. keys size = 2*nnz, indexed by 2*p / 2*p+1 (p = global CSC entry index).
__global__ void emit_edge_keys(int n, long long N, const int* __restrict__ col_ptr,
                               const int* __restrict__ row_idx,
                               unsigned long long* __restrict__ keys)
{
    const int col = blockIdx.x;
    if (col >= n) return;
    const int b = col_ptr[col], e = col_ptr[col + 1];
    const unsigned long long SENT = ~0ull;
    for (int p = b + threadIdx.x; p < e; p += blockDim.x) {
        const int row = row_idx[p];
        const long pp = static_cast<long>(p);
        if (row == col || row < 0 || row >= n) {
            keys[2 * pp] = SENT;
            keys[2 * pp + 1] = SENT;
        } else {
            keys[2 * pp] = static_cast<unsigned long long>(row) * N + col;
            keys[2 * pp + 1] = static_cast<unsigned long long>(col) * N + row;
        }
    }
}

// Decode sorted-unique edge keys into adjncy (= key % N) and tally per-source degree.
__global__ void decode_edges(long m, long long N, const unsigned long long* __restrict__ keys,
                             int* __restrict__ adjncy, int* __restrict__ deg)
{
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const unsigned long long k = keys[i];
    const int src = static_cast<int>(k / N);
    adjncy[i] = static_cast<int>(k % N);
    atomicAdd(&deg[src], 1);
}

}  // namespace

// Build the symmetric (A+Aᵀ) adjacency graph METIS needs, on the GPU.
//   In : device CSC pattern.  Out: host CSR-style xadj[n+1] / adjncy (deduped, diagonal removed).
// Method: emit both directed edge keys per entry, sort+unique, decode key%N -> neighbor.
Status build_symmetric_graph_device(const DeviceCscPattern& csc, std::vector<int>& xadj,
                                    std::vector<int>& adjncy)
{
    const int n = csc.n;
    const int nnz = csc.nnz;
    if (n <= 0 || nnz < 0 || csc.col_ptr.ptr == nullptr ||
        (nnz > 0 && csc.row_idx.ptr == nullptr))
        return Status::kInvalidValue;

    xadj.assign(static_cast<std::size_t>(n) + 1, 0);
    adjncy.clear();
    if (nnz == 0) return Status::kSuccess;

    const long long N = n;
    const long total = 2L * nnz;

    unsigned long long* d_keys = nullptr;
    if (cudaMalloc(&d_keys, static_cast<std::size_t>(total) * sizeof(unsigned long long)) !=
        cudaSuccess)
        return Status::kAllocationFailed;

    emit_edge_keys<<<n, 128>>>(n, N, csc.col_ptr.ptr, csc.row_idx.ptr, d_keys);
    if (cudaGetLastError() != cudaSuccess) { cudaFree(d_keys); return Status::kAnalysisFailed; }

    // Sort all directed edge keys, collapse duplicate edges, and locate the sentinel block
    // (diagonal/invalid entries) that sorts to the high end -> m = real directed edges.
    thrust::sort(thrust::device, d_keys, d_keys + total);
    unsigned long long* uend = thrust::unique(thrust::device, d_keys, d_keys + total);
    const long uniq = static_cast<long>(uend - d_keys);
    const unsigned long long SENT = ~0ull;
    unsigned long long* sent_it =
        thrust::lower_bound(thrust::device, d_keys, d_keys + uniq, SENT);
    const long m = static_cast<long>(sent_it - d_keys);

    int* d_adjncy = nullptr;
    int* d_deg = nullptr;
    if (cudaMalloc(&d_adjncy, static_cast<std::size_t>(std::max(1L, m)) * sizeof(int)) !=
            cudaSuccess ||
        cudaMalloc(&d_deg, (static_cast<std::size_t>(n) + 1) * sizeof(int)) != cudaSuccess) {
        cudaFree(d_keys); cudaFree(d_adjncy); cudaFree(d_deg);
        return Status::kAllocationFailed;
    }
    cudaMemset(d_deg, 0, (static_cast<std::size_t>(n) + 1) * sizeof(int));
    if (m > 0) {
        constexpr int threads = 256;
        decode_edges<<<static_cast<int>((m + threads - 1) / threads), threads>>>(
            m, N, d_keys, d_adjncy, d_deg);
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(d_keys); cudaFree(d_adjncy); cudaFree(d_deg);
            return Status::kAnalysisFailed;
        }
    }
    cudaFree(d_keys);

    // xadj = exclusive scan of per-vertex degree. deg[src] was tallied at slot src; build
    // the (n+1) offset array by shifting the inclusive scan of deg[0..n).
    std::vector<int> deg(static_cast<std::size_t>(n) + 1, 0);
    cudaMemcpy(deg.data(), d_deg, (static_cast<std::size_t>(n) + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaFree(d_deg);
    for (int v = 0; v < n; ++v) xadj[v + 1] = xadj[v] + deg[v];

    adjncy.resize(static_cast<std::size_t>(m));
    if (m > 0)
        cudaMemcpy(adjncy.data(), d_adjncy, static_cast<std::size_t>(m) * sizeof(int),
                   cudaMemcpyDeviceToHost);
    cudaFree(d_adjncy);
    return Status::kSuccess;
}

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
    if (values == 0) return Status::kSuccess;
    if (cudaMalloc(reinterpret_cast<void**>(&ptr), values * sizeof(int)) != cudaSuccess) {
        ptr = nullptr;
        count = 0;
        return Status::kAllocationFailed;
    }
    return Status::kSuccess;
}

Status IntDeviceBuffer::upload(const std::vector<int>& values)
{
    Status st = allocate(values.size());
    if (st != Status::kSuccess || values.empty()) return st;
    return cuda_status(cudaMemcpy(ptr, values.data(), values.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
}

// Convert a device CSR pattern to CSC (count -> scan -> scatter), recording source_pos.
//   In : device CSR row_ptr/col_idx.  Out: csc (col_ptr/row_idx/source_pos on device).
Status build_csc_from_csr_device(int n, int nnz, const int* d_csr_row_ptr,
                                 const int* d_csr_col_idx, DeviceCscPattern& csc)
{
    if (n <= 0 || nnz < 0 || d_csr_row_ptr == nullptr || d_csr_col_idx == nullptr)
        return Status::kInvalidValue;

    csc.col_ptr.reset();
    csc.row_idx.reset();
    csc.source_pos.reset();
    csc.n = n;
    csc.nnz = nnz;

    Status st = csc.col_ptr.allocate(static_cast<std::size_t>(n) + 1);
    if (st != Status::kSuccess) return st;
    st = csc.row_idx.allocate(static_cast<std::size_t>(nnz));
    if (st != Status::kSuccess) return st;
    st = csc.source_pos.allocate(static_cast<std::size_t>(nnz));
    if (st != Status::kSuccess) return st;

    IntDeviceBuffer counts;
    st = counts.allocate(static_cast<std::size_t>(n) + 1);
    if (st != Status::kSuccess) return st;
    if (cudaMemset(counts.ptr, 0, (static_cast<std::size_t>(n) + 1) * sizeof(int)) !=
        cudaSuccess)
        return Status::kAnalysisFailed;

    constexpr int threads = 256;
    count_csr_columns<<<(nnz + threads - 1) / threads, threads>>>(nnz, d_csr_col_idx,
                                                                  counts.ptr);
    if (cudaGetLastError() != cudaSuccess) return Status::kAnalysisFailed;

    thrust::inclusive_scan(thrust::device, counts.ptr, counts.ptr + n + 1, csc.col_ptr.ptr);

    IntDeviceBuffer next_col;
    st = next_col.allocate(static_cast<std::size_t>(n) + 1);
    if (st != Status::kSuccess) return st;
    if (cudaMemcpy(next_col.ptr, csc.col_ptr.ptr, (static_cast<std::size_t>(n) + 1) * sizeof(int),
                   cudaMemcpyDeviceToDevice) != cudaSuccess)
        return Status::kAnalysisFailed;

    scatter_csr_to_csc<<<(n + threads - 1) / threads, threads>>>(
        n, d_csr_row_ptr, d_csr_col_idx, next_col.ptr, csc.row_idx.ptr, csc.source_pos.ptr);
    return cuda_status(cudaGetLastError());
}

// Apply the fill-reducing permutation to a device CSC pattern (count -> scan -> fill).
//   In : csc + d_iperm (old->new index).  Out: ordered CSC with source_pos preserved.
Status permute_csc_device(const DeviceCscPattern& csc, const int* d_iperm,
                          DeviceCscPattern& ordered)
{
    if (csc.n <= 0 || csc.nnz < 0 || csc.col_ptr.ptr == nullptr || csc.row_idx.ptr == nullptr ||
        csc.source_pos.ptr == nullptr || d_iperm == nullptr)
        return Status::kInvalidValue;

    ordered.col_ptr.reset();
    ordered.row_idx.reset();
    ordered.source_pos.reset();
    ordered.n = csc.n;
    ordered.nnz = csc.nnz;

    Status st = ordered.col_ptr.allocate(static_cast<std::size_t>(csc.n) + 1);
    if (st != Status::kSuccess) return st;
    st = ordered.row_idx.allocate(static_cast<std::size_t>(csc.nnz));
    if (st != Status::kSuccess) return st;
    st = ordered.source_pos.allocate(static_cast<std::size_t>(csc.nnz));
    if (st != Status::kSuccess) return st;

    IntDeviceBuffer counts;
    st = counts.allocate(static_cast<std::size_t>(csc.n) + 1);
    if (st != Status::kSuccess) return st;
    if (cudaMemset(counts.ptr, 0, (static_cast<std::size_t>(csc.n) + 1) * sizeof(int)) !=
        cudaSuccess)
        return Status::kAnalysisFailed;

    constexpr int threads = 256;
    count_permuted_columns<<<(csc.n + threads - 1) / threads, threads>>>(
        csc.n, csc.col_ptr.ptr, d_iperm, counts.ptr);
    if (cudaGetLastError() != cudaSuccess) return Status::kAnalysisFailed;

    thrust::inclusive_scan(thrust::device, counts.ptr, counts.ptr + csc.n + 1,
                           ordered.col_ptr.ptr);

    fill_permuted_csc<<<csc.n, 128>>>(csc.n, csc.col_ptr.ptr, csc.row_idx.ptr,
                                      csc.source_pos.ptr, d_iperm, ordered.col_ptr.ptr,
                                      ordered.row_idx.ptr, ordered.source_pos.ptr);
    return cuda_status(cudaGetLastError());
}

Status permute_csc_device_rc(const DeviceCscPattern& csc, const int* d_row_iperm,
                             const int* d_col_iperm, DeviceCscPattern& ordered)
{
    if (csc.n <= 0 || csc.nnz < 0 || csc.col_ptr.ptr == nullptr || csc.row_idx.ptr == nullptr ||
        csc.source_pos.ptr == nullptr || d_row_iperm == nullptr || d_col_iperm == nullptr)
        return Status::kInvalidValue;

    ordered.col_ptr.reset();
    ordered.row_idx.reset();
    ordered.source_pos.reset();
    ordered.n = csc.n;
    ordered.nnz = csc.nnz;

    Status st = ordered.col_ptr.allocate(static_cast<std::size_t>(csc.n) + 1);
    if (st != Status::kSuccess) return st;
    st = ordered.row_idx.allocate(static_cast<std::size_t>(csc.nnz));
    if (st != Status::kSuccess) return st;
    st = ordered.source_pos.allocate(static_cast<std::size_t>(csc.nnz));
    if (st != Status::kSuccess) return st;

    IntDeviceBuffer counts;
    st = counts.allocate(static_cast<std::size_t>(csc.n) + 1);
    if (st != Status::kSuccess) return st;
    if (cudaMemset(counts.ptr, 0, (static_cast<std::size_t>(csc.n) + 1) * sizeof(int)) !=
        cudaSuccess)
        return Status::kAnalysisFailed;

    constexpr int threads = 256;
    count_permuted_columns_rc<<<(csc.n + threads - 1) / threads, threads>>>(
        csc.n, csc.col_ptr.ptr, d_col_iperm, counts.ptr);
    if (cudaGetLastError() != cudaSuccess) return Status::kAnalysisFailed;

    thrust::inclusive_scan(thrust::device, counts.ptr, counts.ptr + csc.n + 1,
                           ordered.col_ptr.ptr);

    fill_permuted_csc_rc<<<csc.n, 128>>>(csc.n, csc.col_ptr.ptr, csc.row_idx.ptr,
                                         csc.source_pos.ptr, d_row_iperm, d_col_iperm,
                                         ordered.col_ptr.ptr, ordered.row_idx.ptr,
                                         ordered.source_pos.ptr);
    return cuda_status(cudaGetLastError());
}

Status download_csc_structure(const DeviceCscPattern& csc, std::vector<int>& col_ptr,
                              std::vector<int>& row_idx)
{
    if (csc.n <= 0 || csc.nnz < 0 || csc.col_ptr.ptr == nullptr || csc.row_idx.ptr == nullptr)
        return Status::kInvalidValue;

    col_ptr.resize(static_cast<std::size_t>(csc.n) + 1);
    row_idx.resize(static_cast<std::size_t>(csc.nnz));
    if (cudaMemcpy(col_ptr.data(), csc.col_ptr.ptr,
                   (static_cast<std::size_t>(csc.n) + 1) * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return Status::kAnalysisFailed;
    if (csc.nnz > 0 &&
        cudaMemcpy(row_idx.data(), csc.row_idx.ptr,
                   static_cast<std::size_t>(csc.nnz) * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return Status::kAnalysisFailed;
    return Status::kSuccess;
}

}  // namespace custom_linear_solver::matrix
