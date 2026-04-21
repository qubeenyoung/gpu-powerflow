#include "linear/partitioned_dense_lu_j11.hpp"
#include "linear/metis_partition.hpp"

#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>

#include <algorithm>
#include <limits>
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

__global__ void init_dense_blocks_kernel(int32_t total_values,
                                         int32_t block_size,
                                         const int32_t* __restrict__ block_sizes,
                                         double* __restrict__ dense_values)
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
    dense_values[i] = (row == col && row >= active_size) ? 1.0 : 0.0;
}

__global__ void scatter_dense_values_kernel(int32_t nnz,
                                            const double* __restrict__ original_values,
                                            const int32_t* __restrict__ source_pos,
                                            const int32_t* __restrict__ dest_pos,
                                            double* __restrict__ dense_values)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        dense_values[dest_pos[i]] = original_values[source_pos[i]];
    }
}

__global__ void gather_dense_rhs_kernel(int32_t total_slots,
                                        int32_t block_size,
                                        const int32_t* __restrict__ slot_new_index,
                                        const int32_t* __restrict__ new_to_old,
                                        const double* __restrict__ rhs_old,
                                        double* __restrict__ dense_rhs)
{
    const int32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) {
        return;
    }
    const int32_t new_index = slot_new_index[slot];
    dense_rhs[slot] = (new_index >= 0) ? rhs_old[new_to_old[new_index]] : 0.0;
}

__global__ void scatter_dense_solution_kernel(int32_t total_slots,
                                              const int32_t* __restrict__ slot_new_index,
                                              const int32_t* __restrict__ new_to_old,
                                              const double* __restrict__ dense_rhs,
                                              double* __restrict__ out_old)
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
        throw std::runtime_error("PartitionedDenseLuJ11Block currently supports none/amd/colamd");
    }

    const Permutation permutation = inverse_permutation.inverse();
    return new_to_old_from_eigen_permutation(permutation);
}

}  // namespace

PartitionedDenseLuJ11Block::PartitionedDenseLuJ11Block(std::string name)
    : name_(std::move(name))
{
    cublas_check(cublasCreate(&cublas_), name_ + " cublasCreate failed");
}

PartitionedDenseLuJ11Block::~PartitionedDenseLuJ11Block()
{
    if (cublas_ != nullptr) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
}

void PartitionedDenseLuJ11Block::set_stream(cudaStream_t stream)
{
    stream_ = stream;
    cublas_check(cublasSetStream(cublas_, stream_), name_ + " cublasSetStream failed");
}

void PartitionedDenseLuJ11Block::clear_stream_noexcept()
{
    stream_ = nullptr;
    if (cublas_ != nullptr) {
        cublasSetStream(cublas_, nullptr);
    }
}

void PartitionedDenseLuJ11Block::analyze(DeviceCsrMatrixView matrix,
                                         const HostCsrPattern& host_pattern,
                                         J11ReorderMode reorder_mode,
                                         int32_t max_block_size,
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
    last_zero_pivot_ = -1;

    build_permutation(host_pattern, reorder_mode);
    build_graph_partitions(host_pattern, partition_mode);
    build_dense_maps(host_pattern);
    build_pointer_arrays();

    analyzed_ = true;
    factorized_ = false;
}

void PartitionedDenseLuJ11Block::build_permutation(const HostCsrPattern& host_pattern,
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

void PartitionedDenseLuJ11Block::build_graph_partitions(
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
            if (new_row == new_col) {
                continue;
            }
            adjacency[static_cast<std::size_t>(new_row)].push_back(new_col);
            adjacency[static_cast<std::size_t>(new_col)].push_back(new_row);
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

void PartitionedDenseLuJ11Block::build_dense_maps(const HostCsrPattern& host_pattern)
{
    dense_source_pos_.clear();
    dense_dest_pos_.clear();
    dense_source_pos_.reserve(host_pattern.col_idx.size());
    dense_dest_pos_.reserve(host_pattern.col_idx.size());

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
            const int32_t dense_index =
                row_block * max_block_size_ * max_block_size_ +
                local_col * max_block_size_ + local_row;
            dense_source_pos_.push_back(pos);
            dense_dest_pos_.push_back(dense_index);
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
        static_cast<std::size_t>(block_count_) *
        static_cast<std::size_t>(max_block_size_);
    d_dense_values_.resize(dense_matrix_values);
    d_dense_rhs_.resize(dense_rhs_values);
    d_pivots_.resize(dense_rhs_values);
    d_info_.resize(static_cast<std::size_t>(block_count_));
    h_info_.resize(static_cast<std::size_t>(block_count_));
}

void PartitionedDenseLuJ11Block::build_pointer_arrays()
{
    std::vector<double*> matrix_ptrs(static_cast<std::size_t>(block_count_));
    std::vector<double*> rhs_ptrs(static_cast<std::size_t>(block_count_));
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

void PartitionedDenseLuJ11Block::check_info_array(const char* stage)
{
    d_info_.copyTo(h_info_.data(), h_info_.size());
    last_zero_pivot_ = -1;
    for (int32_t block = 0; block < block_count_; ++block) {
        const int32_t info = h_info_[static_cast<std::size_t>(block)];
        if (info != 0) {
            last_zero_pivot_ = block;
            throw std::runtime_error(name_ + " dense LU " + stage +
                                     " failed at block " + std::to_string(block) +
                                     " info=" + std::to_string(info));
        }
    }
}

void PartitionedDenseLuJ11Block::factorize()
{
    if (!analyzed_) {
        throw std::runtime_error(name_ + "::factorize called before analyze");
    }

    const int32_t total_dense_values =
        block_count_ * max_block_size_ * max_block_size_;
    init_dense_blocks_kernel<<<grid_for(total_dense_values), kBlockSize, 0, stream_>>>(
        total_dense_values,
        max_block_size_,
        d_block_sizes_.data(),
        d_dense_values_.data());
    CUDA_CHECK(cudaGetLastError());
    scatter_dense_values_kernel<<<grid_for(in_block_nnz_), kBlockSize, 0, stream_>>>(
        in_block_nnz_,
        original_matrix_.values,
        d_dense_source_pos_.data(),
        d_dense_dest_pos_.data(),
        d_dense_values_.data());
    CUDA_CHECK(cudaGetLastError());

    cublas_check(cublasDgetrfBatched(cublas_,
                                     max_block_size_,
                                     d_matrix_ptrs_.data(),
                                     max_block_size_,
                                     d_pivots_.data(),
                                     d_info_.data(),
                                     block_count_),
                 name_ + " cublasDgetrfBatched failed");
    check_info_array("factorization");
    factorized_ = true;
}

void PartitionedDenseLuJ11Block::solve(const double* rhs_device, double* out_device)
{
    if (!factorized_) {
        throw std::runtime_error(name_ + "::solve called before factorize");
    }
    const int32_t total_slots = block_count_ * max_block_size_;
    gather_dense_rhs_kernel<<<grid_for(total_slots), kBlockSize, 0, stream_>>>(
        total_slots,
        max_block_size_,
        d_slot_new_index_.data(),
        d_new_to_old_.data(),
        rhs_device,
        d_dense_rhs_.data());
    CUDA_CHECK(cudaGetLastError());

    int solve_info = 0;
    cublas_check(cublasDgetrsBatched(cublas_,
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
                 name_ + " cublasDgetrsBatched failed");
    if (solve_info != 0) {
        throw std::runtime_error(name_ + " dense LU solve failed info=" +
                                 std::to_string(solve_info));
    }

    scatter_dense_solution_kernel<<<grid_for(total_slots), kBlockSize, 0, stream_>>>(
        total_slots,
        d_slot_new_index_.data(),
        d_new_to_old_.data(),
        d_dense_rhs_.data(),
        out_device);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace exp_20260415::block_ilu
