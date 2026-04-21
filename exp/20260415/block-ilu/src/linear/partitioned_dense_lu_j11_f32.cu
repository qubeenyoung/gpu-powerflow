#include "linear/partitioned_dense_lu_j11_f32.hpp"
#include "linear/metis_partition.hpp"

#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>

#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>

namespace exp_20260415::block_ilu {
namespace {

constexpr int32_t kBlockSize = 256;

int32_t grid_for(int32_t n)
{
    return (n + kBlockSize - 1) / kBlockSize;
}

void cublas_check(cublasStatus_t status, const std::string& message)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(message +
                                 " cublas_status=" +
                                 std::to_string(static_cast<int>(status)));
    }
}

void cusolver_check(cusolverStatus_t status, const std::string& message)
{
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(message +
                                 " cusolver_status=" +
                                 std::to_string(static_cast<int>(status)));
    }
}

__global__ void init_dense_blocks_kernel_f32(int32_t total_values,
                                             int32_t block_size,
                                             const int32_t* __restrict__ block_sizes,
                                             float* __restrict__ dense_values)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_values) {
        return;
    }
    const int32_t matrix_values = block_size * block_size;
    const int32_t block = i / matrix_values;
    const int32_t offset = i - block * matrix_values;
    const int32_t row = offset % block_size;
    const int32_t col = offset / block_size;
    const int32_t active_size = block_sizes[block];
    dense_values[i] = (row == col && row >= active_size) ? 1.0f : 0.0f;
}

__global__ void scatter_dense_values_kernel_f32(int32_t nnz,
                                                const float* __restrict__ original_values,
                                                const int32_t* __restrict__ source_pos,
                                                const int32_t* __restrict__ dest_pos,
                                                float* __restrict__ dense_values)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        dense_values[dest_pos[i]] = original_values[source_pos[i]];
    }
}

__global__ void gather_dense_rhs_kernel_f32(int32_t total_slots,
                                            const int32_t* __restrict__ slot_new_index,
                                            const int32_t* __restrict__ new_to_old,
                                            const float* __restrict__ rhs_old,
                                            float* __restrict__ dense_rhs)
{
    const int32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) {
        return;
    }
    const int32_t new_index = slot_new_index[slot];
    dense_rhs[slot] = (new_index >= 0) ? rhs_old[new_to_old[new_index]] : 0.0f;
}

__global__ void scatter_dense_solution_kernel_f32(int32_t total_slots,
                                                  const int32_t* __restrict__ slot_new_index,
                                                  const int32_t* __restrict__ new_to_old,
                                                  const float* __restrict__ dense_rhs,
                                                  float* __restrict__ out_old)
{
    const int32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) {
        return;
    }
    const int32_t new_index = slot_new_index[slot];
    if (new_index >= 0) {
        out_old[new_to_old[new_index]] = dense_rhs[slot];
    }
}

__global__ void panel_lu_no_pivot_kernel_f32(int32_t block_count,
                                             int32_t matrix_size,
                                             int32_t panel_begin,
                                             int32_t panel_size,
                                             float* __restrict__ dense_values,
                                             int32_t* __restrict__ info)
{
    const int32_t block = blockIdx.x;
    if (block >= block_count || info[block] != 0) {
        return;
    }
    float* a = dense_values + static_cast<std::size_t>(block) * matrix_size * matrix_size;
    for (int32_t kk = 0; kk < panel_size; ++kk) {
        const int32_t k = panel_begin + kk;
        const float pivot = a[k + k * matrix_size];
        if (threadIdx.x == 0 && fabsf(pivot) < 1e-20f) {
            info[block] = k + 1;
        }
        __syncthreads();
        if (info[block] != 0) {
            return;
        }

        for (int32_t idx = threadIdx.x; idx < panel_size - kk - 1; idx += blockDim.x) {
            const int32_t row = k + 1 + idx;
            a[row + k * matrix_size] /= pivot;
        }
        __syncthreads();

        const int32_t rows = panel_size - kk - 1;
        const int32_t total = rows * rows;
        for (int32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
            const int32_t local_col = idx / rows;
            const int32_t local_row = idx - local_col * rows;
            const int32_t row = k + 1 + local_row;
            const int32_t col = k + 1 + local_col;
            a[row + col * matrix_size] -= a[row + k * matrix_size] * a[k + col * matrix_size];
        }
        __syncthreads();
    }
}

__global__ void solve_u12_no_pivot_kernel_f32(int32_t block_count,
                                              int32_t matrix_size,
                                              int32_t panel_begin,
                                              int32_t panel_size,
                                              int32_t trailing,
                                              float* __restrict__ dense_values)
{
    const int32_t block = blockIdx.x;
    if (block >= block_count || trailing <= 0) {
        return;
    }
    float* a = dense_values + static_cast<std::size_t>(block) * matrix_size * matrix_size;
    for (int32_t ii = 0; ii < panel_size; ++ii) {
        const int32_t row = panel_begin + ii;
        for (int32_t j = threadIdx.x; j < trailing; j += blockDim.x) {
            const int32_t col = panel_begin + panel_size + j;
            float value = a[row + col * matrix_size];
            for (int32_t s = 0; s < ii; ++s) {
                value -= a[row + (panel_begin + s) * matrix_size] *
                         a[(panel_begin + s) + col * matrix_size];
            }
            a[row + col * matrix_size] = value;
        }
        __syncthreads();
    }
}

__global__ void solve_l21_no_pivot_kernel_f32(int32_t block_count,
                                              int32_t matrix_size,
                                              int32_t panel_begin,
                                              int32_t panel_size,
                                              int32_t trailing,
                                              float* __restrict__ dense_values,
                                              int32_t* __restrict__ info)
{
    const int32_t block = blockIdx.x;
    if (block >= block_count || trailing <= 0 || info[block] != 0) {
        return;
    }
    float* a = dense_values + static_cast<std::size_t>(block) * matrix_size * matrix_size;
    for (int32_t jj = 0; jj < panel_size; ++jj) {
        const int32_t col = panel_begin + jj;
        const float diag = a[col + col * matrix_size];
        if (threadIdx.x == 0 && fabsf(diag) < 1e-20f) {
            info[block] = col + 1;
        }
        __syncthreads();
        if (info[block] != 0) {
            return;
        }
        for (int32_t i = threadIdx.x; i < trailing; i += blockDim.x) {
            const int32_t row = panel_begin + panel_size + i;
            float value = a[row + col * matrix_size];
            for (int32_t s = 0; s < jj; ++s) {
                value -= a[row + (panel_begin + s) * matrix_size] *
                         a[(panel_begin + s) + col * matrix_size];
            }
            a[row + col * matrix_size] = value / diag;
        }
        __syncthreads();
    }
}

using SparsePattern = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;
using Permutation = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int32_t>;

SparsePattern make_eigen_pattern(const HostCsrPattern& pattern)
{
    using Triplet = Eigen::Triplet<double, int32_t>;
    std::vector<Triplet> triplets;
    triplets.reserve(pattern.col_idx.size());
    for (int32_t row = 0; row < pattern.rows; ++row) {
        for (int32_t pos = pattern.row_ptr[static_cast<std::size_t>(row)];
             pos < pattern.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            triplets.emplace_back(row, pattern.col_idx[static_cast<std::size_t>(pos)], 1.0);
        }
    }
    SparsePattern matrix(pattern.rows, pattern.cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return matrix;
}

std::vector<int32_t> new_to_old_from_eigen_permutation(const Permutation& permutation)
{
    const int32_t n = static_cast<int32_t>(permutation.size());
    std::vector<int32_t> new_to_old(static_cast<std::size_t>(n));
    for (int32_t old_index = 0; old_index < n; ++old_index) {
        const int32_t new_index = static_cast<int32_t>(permutation.indices()[old_index]);
        if (new_index < 0 || new_index >= n) {
            throw std::runtime_error("Eigen ordering returned an invalid permutation index");
        }
        new_to_old[static_cast<std::size_t>(new_index)] = old_index;
    }
    return new_to_old;
}

std::vector<int32_t> compute_new_to_old(const HostCsrPattern& pattern, J11ReorderMode mode)
{
    if (mode == J11ReorderMode::None) {
        std::vector<int32_t> identity(static_cast<std::size_t>(pattern.rows));
        std::iota(identity.begin(), identity.end(), 0);
        return identity;
    }
    SparsePattern matrix = make_eigen_pattern(pattern);
    Permutation inverse_permutation;
    if (mode == J11ReorderMode::Amd) {
        Eigen::AMDOrdering<int32_t> ordering;
        ordering(matrix, inverse_permutation);
    } else if (mode == J11ReorderMode::Colamd) {
        Eigen::COLAMDOrdering<int32_t> ordering;
        ordering(matrix, inverse_permutation);
    } else {
        throw std::runtime_error("PartitionedDenseLuJ11BlockF32 supports none/amd/colamd");
    }
    const Permutation permutation = inverse_permutation.inverse();
    return new_to_old_from_eigen_permutation(permutation);
}

}  // namespace

PartitionedDenseLuJ11BlockF32::PartitionedDenseLuJ11BlockF32(std::string name)
    : name_(std::move(name))
{
    cublas_check(cublasCreate(&cublas_), name_ + " cublasCreate failed");
    cusolver_check(cusolverDnCreate(&cusolver_), name_ + " cusolverDnCreate failed");
}

PartitionedDenseLuJ11BlockF32::~PartitionedDenseLuJ11BlockF32()
{
    if (cusolver_ != nullptr) {
        cusolverDnDestroy(cusolver_);
        cusolver_ = nullptr;
    }
    if (cublas_ != nullptr) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
}

void PartitionedDenseLuJ11BlockF32::set_stream(cudaStream_t stream)
{
    stream_ = stream;
    cublas_check(cublasSetStream(cublas_, stream_), name_ + " cublasSetStream failed");
    cusolver_check(cusolverDnSetStream(cusolver_, stream_),
                   name_ + " cusolverDnSetStream failed");
}

void PartitionedDenseLuJ11BlockF32::clear_stream_noexcept()
{
    stream_ = nullptr;
    if (cublas_ != nullptr) {
        cublasSetStream(cublas_, nullptr);
    }
    if (cusolver_ != nullptr) {
        cusolverDnSetStream(cusolver_, nullptr);
    }
}

void PartitionedDenseLuJ11BlockF32::analyze(DeviceCsrMatrixViewF32 matrix,
                                            const HostCsrPattern& host_pattern,
                                            J11ReorderMode reorder_mode,
                                            int32_t max_block_size,
                                            J11DenseBackend dense_backend,
                                            J11PartitionMode partition_mode)
{
    if (matrix.rows <= 0 || matrix.rows != matrix.cols || matrix.nnz <= 0 ||
        host_pattern.rows != matrix.rows || host_pattern.cols != matrix.cols ||
        host_pattern.nnz() != matrix.nnz || max_block_size <= 0) {
        throw std::runtime_error(name_ + "::analyze received invalid inputs");
    }
    original_matrix_ = matrix;
    rows_ = matrix.rows;
    nnz_ = matrix.nnz;
    max_block_size_ = max_block_size;
    dense_backend_ = dense_backend;

    build_permutation(host_pattern, reorder_mode);
    build_graph_partitions(host_pattern, partition_mode);
    build_dense_maps(host_pattern);
    build_pointer_arrays();
    analyzed_ = true;
    factorized_ = false;
}

void PartitionedDenseLuJ11BlockF32::build_permutation(const HostCsrPattern& host_pattern,
                                                      J11ReorderMode reorder_mode)
{
    new_to_old_ = compute_new_to_old(host_pattern, reorder_mode);
    old_to_new_.assign(static_cast<std::size_t>(rows_), -1);
    for (int32_t new_index = 0; new_index < rows_; ++new_index) {
        const int32_t old_index = new_to_old_[static_cast<std::size_t>(new_index)];
        if (old_index < 0 || old_index >= rows_ ||
            old_to_new_[static_cast<std::size_t>(old_index)] != -1) {
            throw std::runtime_error(name_ + " received invalid ordering");
        }
        old_to_new_[static_cast<std::size_t>(old_index)] = new_index;
    }
    d_new_to_old_.assign(new_to_old_.data(), new_to_old_.size());
}

void PartitionedDenseLuJ11BlockF32::build_graph_partitions(
    const HostCsrPattern& host_pattern,
    J11PartitionMode partition_mode)
{
    if (partition_mode == J11PartitionMode::Metis) {
        build_metis_graph_partitions(host_pattern,
                                     old_to_new_,
                                     max_block_size_,
                                     block_of_new_,
                                     local_of_new_,
                                     block_sizes_);
        block_count_ = static_cast<int32_t>(block_sizes_.size());
        slot_new_index_.assign(static_cast<std::size_t>(block_count_ * max_block_size_), -1);
        for (int32_t new_index = 0; new_index < rows_; ++new_index) {
            const int32_t block = block_of_new_[static_cast<std::size_t>(new_index)];
            const int32_t local = local_of_new_[static_cast<std::size_t>(new_index)];
            slot_new_index_[static_cast<std::size_t>(block * max_block_size_ + local)] =
                new_index;
        }
        d_block_sizes_.assign(block_sizes_.data(), block_sizes_.size());
        d_slot_new_index_.assign(slot_new_index_.data(), slot_new_index_.size());
        return;
    }

    std::vector<std::vector<int32_t>> adjacency(static_cast<std::size_t>(rows_));
    for (int32_t old_row = 0; old_row < rows_; ++old_row) {
        const int32_t new_row = old_to_new_[static_cast<std::size_t>(old_row)];
        for (int32_t pos = host_pattern.row_ptr[static_cast<std::size_t>(old_row)];
             pos < host_pattern.row_ptr[static_cast<std::size_t>(old_row + 1)];
             ++pos) {
            const int32_t old_col = host_pattern.col_idx[static_cast<std::size_t>(pos)];
            const int32_t new_col = old_to_new_[static_cast<std::size_t>(old_col)];
            if (new_row != new_col) {
                adjacency[static_cast<std::size_t>(new_row)].push_back(new_col);
                adjacency[static_cast<std::size_t>(new_col)].push_back(new_row);
            }
        }
    }
    std::vector<int32_t> degree(static_cast<std::size_t>(rows_), 0);
    for (int32_t node = 0; node < rows_; ++node) {
        auto& neighbors = adjacency[static_cast<std::size_t>(node)];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        degree[static_cast<std::size_t>(node)] = static_cast<int32_t>(neighbors.size());
    }

    block_of_new_.assign(static_cast<std::size_t>(rows_), -1);
    local_of_new_.assign(static_cast<std::size_t>(rows_), -1);
    block_sizes_.clear();
    std::vector<char> visited(static_cast<std::size_t>(rows_), 0);
    for (int32_t seed = 0; seed < rows_; ++seed) {
        if (visited[static_cast<std::size_t>(seed)]) {
            continue;
        }
        const int32_t block = static_cast<int32_t>(block_sizes_.size());
        int32_t local = 0;
        std::queue<int32_t> queue;
        visited[static_cast<std::size_t>(seed)] = 1;
        queue.push(seed);
        while (!queue.empty() && local < max_block_size_) {
            const int32_t node = queue.front();
            queue.pop();
            block_of_new_[static_cast<std::size_t>(node)] = block;
            local_of_new_[static_cast<std::size_t>(node)] = local++;
            std::vector<int32_t> candidates;
            for (int32_t neighbor : adjacency[static_cast<std::size_t>(node)]) {
                if (!visited[static_cast<std::size_t>(neighbor)]) {
                    candidates.push_back(neighbor);
                }
            }
            std::sort(candidates.begin(), candidates.end(), [&](int32_t lhs, int32_t rhs) {
                const int32_t lhs_degree = degree[static_cast<std::size_t>(lhs)];
                const int32_t rhs_degree = degree[static_cast<std::size_t>(rhs)];
                if (lhs_degree != rhs_degree) {
                    return lhs_degree < rhs_degree;
                }
                return lhs < rhs;
            });
            for (int32_t neighbor : candidates) {
                if (local + static_cast<int32_t>(queue.size()) >= max_block_size_) {
                    break;
                }
                visited[static_cast<std::size_t>(neighbor)] = 1;
                queue.push(neighbor);
            }
        }
        block_sizes_.push_back(local);
    }
    for (int32_t node = 0; node < rows_; ++node) {
        if (block_of_new_[static_cast<std::size_t>(node)] < 0) {
            throw std::runtime_error(name_ + " graph partition left an unassigned node");
        }
    }
    block_count_ = static_cast<int32_t>(block_sizes_.size());
    slot_new_index_.assign(static_cast<std::size_t>(block_count_ * max_block_size_), -1);
    for (int32_t new_index = 0; new_index < rows_; ++new_index) {
        const int32_t block = block_of_new_[static_cast<std::size_t>(new_index)];
        const int32_t local = local_of_new_[static_cast<std::size_t>(new_index)];
        slot_new_index_[static_cast<std::size_t>(block * max_block_size_ + local)] = new_index;
    }
    d_block_sizes_.assign(block_sizes_.data(), block_sizes_.size());
    d_slot_new_index_.assign(slot_new_index_.data(), slot_new_index_.size());
}

void PartitionedDenseLuJ11BlockF32::build_dense_maps(const HostCsrPattern& host_pattern)
{
    dense_source_pos_.clear();
    dense_dest_pos_.clear();
    for (int32_t old_row = 0; old_row < rows_; ++old_row) {
        const int32_t new_row = old_to_new_[static_cast<std::size_t>(old_row)];
        const int32_t row_block = block_of_new_[static_cast<std::size_t>(new_row)];
        const int32_t local_row = local_of_new_[static_cast<std::size_t>(new_row)];
        for (int32_t pos = host_pattern.row_ptr[static_cast<std::size_t>(old_row)];
             pos < host_pattern.row_ptr[static_cast<std::size_t>(old_row + 1)];
             ++pos) {
            const int32_t old_col = host_pattern.col_idx[static_cast<std::size_t>(pos)];
            const int32_t new_col = old_to_new_[static_cast<std::size_t>(old_col)];
            const int32_t col_block = block_of_new_[static_cast<std::size_t>(new_col)];
            if (row_block != col_block) {
                continue;
            }
            const int32_t local_col = local_of_new_[static_cast<std::size_t>(new_col)];
            dense_source_pos_.push_back(pos);
            dense_dest_pos_.push_back(row_block * max_block_size_ * max_block_size_ +
                                      local_col * max_block_size_ + local_row);
        }
    }
    in_block_nnz_ = static_cast<int32_t>(dense_source_pos_.size());
    d_dense_source_pos_.assign(dense_source_pos_.data(), dense_source_pos_.size());
    d_dense_dest_pos_.assign(dense_dest_pos_.data(), dense_dest_pos_.size());
    const std::size_t dense_matrix_values =
        static_cast<std::size_t>(block_count_) *
        static_cast<std::size_t>(max_block_size_) *
        static_cast<std::size_t>(max_block_size_);
    const std::size_t dense_rhs_values =
        static_cast<std::size_t>(block_count_) * static_cast<std::size_t>(max_block_size_);
    d_dense_values_.resize(dense_matrix_values);
    d_dense_rhs_.resize(dense_rhs_values);
    d_pivots_.resize(dense_rhs_values);
    d_info_.resize(static_cast<std::size_t>(block_count_));
    h_info_.resize(static_cast<std::size_t>(block_count_));
}

void PartitionedDenseLuJ11BlockF32::build_pointer_arrays()
{
    std::vector<float*> matrix_ptrs(static_cast<std::size_t>(block_count_));
    std::vector<float*> rhs_ptrs(static_cast<std::size_t>(block_count_));
    const int32_t matrix_stride = max_block_size_ * max_block_size_;
    for (int32_t block = 0; block < block_count_; ++block) {
        matrix_ptrs[static_cast<std::size_t>(block)] =
            d_dense_values_.data() + static_cast<std::size_t>(block * matrix_stride);
        rhs_ptrs[static_cast<std::size_t>(block)] =
            d_dense_rhs_.data() + static_cast<std::size_t>(block * max_block_size_);
    }
    d_matrix_ptrs_.assign(matrix_ptrs.data(), matrix_ptrs.size());
    d_rhs_ptrs_.assign(rhs_ptrs.data(), rhs_ptrs.size());
}

void PartitionedDenseLuJ11BlockF32::check_factor_info()
{
    d_info_.copyTo(h_info_.data(), h_info_.size());
    last_zero_pivot_ = -1;
    for (int32_t block = 0; block < block_count_; ++block) {
        const int32_t info = h_info_[static_cast<std::size_t>(block)];
        if (info != 0) {
            last_zero_pivot_ = block;
            throw std::runtime_error(name_ + " FP32 dense LU factorization failed block=" +
                                     std::to_string(block) + " info=" +
                                     std::to_string(info));
        }
    }
}

void PartitionedDenseLuJ11BlockF32::factorize()
{
    if (!analyzed_) {
        throw std::runtime_error(name_ + "::factorize called before analyze");
    }
    if (dense_backend_ == J11DenseBackend::TcNoPivot) {
        factorize_tc_no_pivot();
    } else if (dense_backend_ == J11DenseBackend::CusolverGetrf) {
        factorize_cusolver_getrf();
    } else {
        factorize_cublas_getrf();
    }
    factorized_ = true;
}

void PartitionedDenseLuJ11BlockF32::factorize_cublas_getrf()
{
    const int32_t total_dense_values = block_count_ * max_block_size_ * max_block_size_;
    init_dense_blocks_kernel_f32<<<grid_for(total_dense_values), kBlockSize, 0, stream_>>>(
        total_dense_values, max_block_size_, d_block_sizes_.data(), d_dense_values_.data());
    CUDA_CHECK(cudaGetLastError());
    scatter_dense_values_kernel_f32<<<grid_for(in_block_nnz_), kBlockSize, 0, stream_>>>(
        in_block_nnz_,
        original_matrix_.values,
        d_dense_source_pos_.data(),
        d_dense_dest_pos_.data(),
        d_dense_values_.data());
    CUDA_CHECK(cudaGetLastError());
    cublas_check(cublasSgetrfBatched(cublas_,
                                     max_block_size_,
                                     d_matrix_ptrs_.data(),
                                     max_block_size_,
                                     d_pivots_.data(),
                                     d_info_.data(),
                                     block_count_),
                 name_ + " cublasSgetrfBatched failed");
    check_factor_info();
}

void PartitionedDenseLuJ11BlockF32::factorize_cusolver_getrf()
{
    const int32_t total_dense_values = block_count_ * max_block_size_ * max_block_size_;
    init_dense_blocks_kernel_f32<<<grid_for(total_dense_values), kBlockSize, 0, stream_>>>(
        total_dense_values, max_block_size_, d_block_sizes_.data(), d_dense_values_.data());
    CUDA_CHECK(cudaGetLastError());
    scatter_dense_values_kernel_f32<<<grid_for(in_block_nnz_), kBlockSize, 0, stream_>>>(
        in_block_nnz_,
        original_matrix_.values,
        d_dense_source_pos_.data(),
        d_dense_dest_pos_.data(),
        d_dense_values_.data());
    CUDA_CHECK(cudaGetLastError());

    int32_t lwork = 0;
    cusolver_check(cusolverDnSgetrf_bufferSize(cusolver_,
                                               max_block_size_,
                                               max_block_size_,
                                               d_dense_values_.data(),
                                               max_block_size_,
                                               &lwork),
                   name_ + " cusolverDnSgetrf_bufferSize failed");
    d_cusolver_workspace_.resize(static_cast<std::size_t>(lwork));
    const int32_t matrix_stride = max_block_size_ * max_block_size_;
    for (int32_t block = 0; block < block_count_; ++block) {
        float* matrix = d_dense_values_.data() +
                        static_cast<std::size_t>(block) *
                            static_cast<std::size_t>(matrix_stride);
        int32_t* pivots = d_pivots_.data() +
                          static_cast<std::size_t>(block) *
                              static_cast<std::size_t>(max_block_size_);
        int32_t* info = d_info_.data() + block;
        cusolver_check(cusolverDnSgetrf(cusolver_,
                                        max_block_size_,
                                        max_block_size_,
                                        matrix,
                                        max_block_size_,
                                        d_cusolver_workspace_.data(),
                                        pivots,
                                        info),
                       name_ + " cusolverDnSgetrf failed");
    }
    check_factor_info();
}

void PartitionedDenseLuJ11BlockF32::factorize_tc_no_pivot()
{
    const int32_t total_dense_values = block_count_ * max_block_size_ * max_block_size_;
    init_dense_blocks_kernel_f32<<<grid_for(total_dense_values), kBlockSize, 0, stream_>>>(
        total_dense_values, max_block_size_, d_block_sizes_.data(), d_dense_values_.data());
    CUDA_CHECK(cudaGetLastError());
    scatter_dense_values_kernel_f32<<<grid_for(in_block_nnz_), kBlockSize, 0, stream_>>>(
        in_block_nnz_,
        original_matrix_.values,
        d_dense_source_pos_.data(),
        d_dense_dest_pos_.data(),
        d_dense_values_.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(d_info_.data(), 0, d_info_.size() * sizeof(int32_t), stream_));

    cublas_check(cublasSetMathMode(cublas_, CUBLAS_TF32_TENSOR_OP_MATH),
                 name_ + " cublasSetMathMode TF32 failed");
    constexpr int32_t panel_size = 16;
    const float alpha = -1.0f;
    const float beta = 1.0f;
    const long long matrix_stride =
        static_cast<long long>(max_block_size_) * static_cast<long long>(max_block_size_);

    for (int32_t panel_begin = 0; panel_begin < max_block_size_; panel_begin += panel_size) {
        const int32_t current_panel = std::min(panel_size, max_block_size_ - panel_begin);
        const int32_t trailing = max_block_size_ - panel_begin - current_panel;
        panel_lu_no_pivot_kernel_f32<<<block_count_, 256, 0, stream_>>>(
            block_count_,
            max_block_size_,
            panel_begin,
            current_panel,
            d_dense_values_.data(),
            d_info_.data());
        CUDA_CHECK(cudaGetLastError());
        if (trailing <= 0) {
            break;
        }
        solve_u12_no_pivot_kernel_f32<<<block_count_, 256, 0, stream_>>>(
            block_count_,
            max_block_size_,
            panel_begin,
            current_panel,
            trailing,
            d_dense_values_.data());
        solve_l21_no_pivot_kernel_f32<<<block_count_, 256, 0, stream_>>>(
            block_count_,
            max_block_size_,
            panel_begin,
            current_panel,
            trailing,
            d_dense_values_.data(),
            d_info_.data());
        CUDA_CHECK(cudaGetLastError());

        float* l21 = d_dense_values_.data() +
                     (panel_begin + current_panel) +
                     panel_begin * max_block_size_;
        float* u12 = d_dense_values_.data() +
                     panel_begin +
                     (panel_begin + current_panel) * max_block_size_;
        float* a22 = d_dense_values_.data() +
                     (panel_begin + current_panel) +
                     (panel_begin + current_panel) * max_block_size_;
        cublas_check(cublasGemmStridedBatchedEx(cublas_,
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                trailing,
                                                trailing,
                                                current_panel,
                                                &alpha,
                                                l21,
                                                CUDA_R_32F,
                                                max_block_size_,
                                                matrix_stride,
                                                u12,
                                                CUDA_R_32F,
                                                max_block_size_,
                                                matrix_stride,
                                                &beta,
                                                a22,
                                                CUDA_R_32F,
                                                max_block_size_,
                                                matrix_stride,
                                                block_count_,
                                                CUBLAS_COMPUTE_32F_FAST_TF32,
                                                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                     name_ + " TC trailing update failed");
    }
    check_factor_info();
}

void PartitionedDenseLuJ11BlockF32::solve(const float* rhs_device, float* out_device)
{
    if (!factorized_) {
        throw std::runtime_error(name_ + "::solve called before factorize");
    }
    if (dense_backend_ == J11DenseBackend::TcNoPivot) {
        solve_tc_no_pivot(rhs_device, out_device);
    } else {
        solve_cublas_getrf(rhs_device, out_device);
    }
}

void PartitionedDenseLuJ11BlockF32::solve_cublas_getrf(const float* rhs_device, float* out_device)
{
    const int32_t total_slots = block_count_ * max_block_size_;
    gather_dense_rhs_kernel_f32<<<grid_for(total_slots), kBlockSize, 0, stream_>>>(
        total_slots,
        d_slot_new_index_.data(),
        d_new_to_old_.data(),
        rhs_device,
        d_dense_rhs_.data());
    CUDA_CHECK(cudaGetLastError());
    int solve_info = 0;
    cublas_check(cublasSgetrsBatched(cublas_,
                                     CUBLAS_OP_N,
                                     max_block_size_,
                                     1,
                                     d_matrix_ptrs_.data(),
                                     max_block_size_,
                                     d_pivots_.data(),
                                     d_rhs_ptrs_.data(),
                                     max_block_size_,
                                     &solve_info,
                                     block_count_),
                 name_ + " cublasSgetrsBatched failed");
    if (solve_info != 0) {
        throw std::runtime_error(name_ + " FP32 dense LU solve failed info=" +
                                 std::to_string(solve_info));
    }
    scatter_dense_solution_kernel_f32<<<grid_for(total_slots), kBlockSize, 0, stream_>>>(
        total_slots,
        d_slot_new_index_.data(),
        d_new_to_old_.data(),
        d_dense_rhs_.data(),
        out_device);
    CUDA_CHECK(cudaGetLastError());
}

void PartitionedDenseLuJ11BlockF32::solve_tc_no_pivot(const float* rhs_device, float* out_device)
{
    const int32_t total_slots = block_count_ * max_block_size_;
    gather_dense_rhs_kernel_f32<<<grid_for(total_slots), kBlockSize, 0, stream_>>>(
        total_slots,
        d_slot_new_index_.data(),
        d_new_to_old_.data(),
        rhs_device,
        d_dense_rhs_.data());
    CUDA_CHECK(cudaGetLastError());

    const float alpha = 1.0f;
    cublas_check(cublasStrsmBatched(cublas_,
                                    CUBLAS_SIDE_LEFT,
                                    CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_N,
                                    CUBLAS_DIAG_UNIT,
                                    max_block_size_,
                                    1,
                                    &alpha,
                                    const_cast<const float**>(d_matrix_ptrs_.data()),
                                    max_block_size_,
                                    d_rhs_ptrs_.data(),
                                    max_block_size_,
                                    block_count_),
                 name_ + " TC backend lower solve failed");
    cublas_check(cublasStrsmBatched(cublas_,
                                    CUBLAS_SIDE_LEFT,
                                    CUBLAS_FILL_MODE_UPPER,
                                    CUBLAS_OP_N,
                                    CUBLAS_DIAG_NON_UNIT,
                                    max_block_size_,
                                    1,
                                    &alpha,
                                    const_cast<const float**>(d_matrix_ptrs_.data()),
                                    max_block_size_,
                                    d_rhs_ptrs_.data(),
                                    max_block_size_,
                                    block_count_),
                 name_ + " TC backend upper solve failed");
    scatter_dense_solution_kernel_f32<<<grid_for(total_slots), kBlockSize, 0, stream_>>>(
        total_slots,
        d_slot_new_index_.data(),
        d_new_to_old_.data(),
        d_dense_rhs_.data(),
        out_device);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace exp_20260415::block_ilu
