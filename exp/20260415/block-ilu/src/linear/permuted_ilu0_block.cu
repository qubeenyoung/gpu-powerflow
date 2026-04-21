#include "linear/permuted_ilu0_block.hpp"

#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>

#include <algorithm>
#include <cctype>
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

__global__ void permute_values_kernel(int32_t nnz,
                                      const double* __restrict__ original_values,
                                      const int32_t* __restrict__ source_pos,
                                      double* __restrict__ permuted_values)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        permuted_values[i] = original_values[source_pos[i]];
    }
}

__global__ void permute_rhs_kernel(int32_t n,
                                   const double* __restrict__ rhs_old,
                                   const int32_t* __restrict__ new_to_old,
                                   double* __restrict__ rhs_new)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        rhs_new[i] = rhs_old[new_to_old[i]];
    }
}

__global__ void inverse_permute_solution_kernel(int32_t n,
                                                const double* __restrict__ x_new,
                                                const int32_t* __restrict__ new_to_old,
                                                double* __restrict__ x_old)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x_old[new_to_old[i]] = x_new[i];
    }
}

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
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

std::vector<int32_t> compute_amd_new_to_old(const HostCsrPattern& pattern)
{
    SparsePattern matrix = make_eigen_pattern(pattern);
    Permutation inverse_permutation;
    Eigen::AMDOrdering<int32_t> ordering;
    ordering(matrix, inverse_permutation);
    const Permutation permutation = inverse_permutation.inverse();
    return new_to_old_from_eigen_permutation(permutation);
}

std::vector<int32_t> compute_colamd_new_to_old(const HostCsrPattern& pattern)
{
    SparsePattern matrix = make_eigen_pattern(pattern);
    Permutation inverse_permutation;
    Eigen::COLAMDOrdering<int32_t> ordering;
    ordering(matrix, inverse_permutation);
    const Permutation permutation = inverse_permutation.inverse();
    return new_to_old_from_eigen_permutation(permutation);
}

std::vector<int32_t> compute_rcm_new_to_old(const HostCsrPattern& pattern)
{
    const int32_t n = pattern.rows;
    std::vector<std::vector<int32_t>> adjacency(static_cast<std::size_t>(n));
    for (int32_t row = 0; row < n; ++row) {
        for (int32_t pos = pattern.row_ptr[static_cast<std::size_t>(row)];
             pos < pattern.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = pattern.col_idx[static_cast<std::size_t>(pos)];
            if (col == row) {
                continue;
            }
            adjacency[static_cast<std::size_t>(row)].push_back(col);
            adjacency[static_cast<std::size_t>(col)].push_back(row);
        }
    }

    std::vector<int32_t> degree(static_cast<std::size_t>(n), 0);
    for (int32_t row = 0; row < n; ++row) {
        auto& neighbors = adjacency[static_cast<std::size_t>(row)];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        degree[static_cast<std::size_t>(row)] = static_cast<int32_t>(neighbors.size());
    }

    std::vector<char> visited(static_cast<std::size_t>(n), 0);
    std::vector<int32_t> order;
    order.reserve(static_cast<std::size_t>(n));

    for (int32_t component = 0; component < n; ++component) {
        int32_t start = -1;
        int32_t best_degree = std::numeric_limits<int32_t>::max();
        for (int32_t node = 0; node < n; ++node) {
            if (!visited[static_cast<std::size_t>(node)] &&
                degree[static_cast<std::size_t>(node)] < best_degree) {
                start = node;
                best_degree = degree[static_cast<std::size_t>(node)];
            }
        }
        if (start < 0) {
            break;
        }

        std::queue<int32_t> queue;
        visited[static_cast<std::size_t>(start)] = 1;
        queue.push(start);
        while (!queue.empty()) {
            const int32_t node = queue.front();
            queue.pop();
            order.push_back(node);

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
                if (!visited[static_cast<std::size_t>(neighbor)]) {
                    visited[static_cast<std::size_t>(neighbor)] = 1;
                    queue.push(neighbor);
                }
            }
        }
    }

    std::reverse(order.begin(), order.end());
    if (static_cast<int32_t>(order.size()) != n) {
        throw std::runtime_error("RCM ordering did not visit every node");
    }
    return order;
}

std::vector<int32_t> compute_new_to_old(const HostCsrPattern& pattern, J11ReorderMode mode)
{
    if (pattern.rows <= 0 || pattern.rows != pattern.cols) {
        throw std::runtime_error("J11 reordering requires a square nonempty pattern");
    }
    switch (mode) {
    case J11ReorderMode::None: {
        std::vector<int32_t> identity(static_cast<std::size_t>(pattern.rows));
        std::iota(identity.begin(), identity.end(), 0);
        return identity;
    }
    case J11ReorderMode::Amd:
        return compute_amd_new_to_old(pattern);
    case J11ReorderMode::Colamd:
        return compute_colamd_new_to_old(pattern);
    case J11ReorderMode::Rcm:
        return compute_rcm_new_to_old(pattern);
    default:
        throw std::runtime_error("unknown J11 reorder mode");
    }
}

}  // namespace

J11ReorderMode parse_j11_reorder_mode(const std::string& name)
{
    const std::string lowered = lower_copy(name);
    if (lowered == "none" || lowered == "no" || lowered == "natural") {
        return J11ReorderMode::None;
    }
    if (lowered == "amd") {
        return J11ReorderMode::Amd;
    }
    if (lowered == "colamd") {
        return J11ReorderMode::Colamd;
    }
    if (lowered == "rcm") {
        return J11ReorderMode::Rcm;
    }
    throw std::runtime_error("unknown J11 reorder mode: " + name);
}

const char* j11_reorder_mode_name(J11ReorderMode mode)
{
    switch (mode) {
    case J11ReorderMode::None:
        return "none";
    case J11ReorderMode::Amd:
        return "amd";
    case J11ReorderMode::Colamd:
        return "colamd";
    case J11ReorderMode::Rcm:
        return "rcm";
    default:
        return "unknown";
    }
}

J11SolverKind parse_j11_solver_kind(const std::string& name)
{
    const std::string lowered = lower_copy(name);
    if (lowered == "ilu0" || lowered == "ilu") {
        return J11SolverKind::Ilu0;
    }
    if (lowered == "partition-dense-lu" ||
        lowered == "dense-lu" ||
        lowered == "batched-dense-lu") {
        return J11SolverKind::PartitionDenseLu;
    }
    if (lowered == "exact-klu" || lowered == "klu" || lowered == "exact") {
        return J11SolverKind::ExactKlu;
    }
    throw std::runtime_error("unknown J11 solver kind: " + name);
}

const char* j11_solver_kind_name(J11SolverKind kind)
{
    switch (kind) {
    case J11SolverKind::Ilu0:
        return "ilu0";
    case J11SolverKind::PartitionDenseLu:
        return "partition_dense_lu";
    case J11SolverKind::ExactKlu:
        return "exact_klu";
    default:
        return "unknown";
    }
}

J11DenseBackend parse_j11_dense_backend(const std::string& name)
{
    const std::string lowered = lower_copy(name);
    if (lowered == "cublas" || lowered == "getrf" || lowered == "cublas-getrf") {
        return J11DenseBackend::CublasGetrf;
    }
    if (lowered == "tc" || lowered == "tensor-core" || lowered == "tc-no-pivot") {
        return J11DenseBackend::TcNoPivot;
    }
    if (lowered == "cusolver" || lowered == "cusolver-getrf") {
        return J11DenseBackend::CusolverGetrf;
    }
    throw std::runtime_error("unknown J11 dense backend: " + name);
}

const char* j11_dense_backend_name(J11DenseBackend backend)
{
    switch (backend) {
    case J11DenseBackend::CublasGetrf:
        return "cublas_getrf";
    case J11DenseBackend::TcNoPivot:
        return "tc_no_pivot";
    case J11DenseBackend::CusolverGetrf:
        return "cusolver_getrf";
    default:
        return "unknown";
    }
}

J11PartitionMode parse_j11_partition_mode(const std::string& name)
{
    const std::string lowered = lower_copy(name);
    if (lowered == "bfs" || lowered == "greedy-bfs" || lowered == "greedy") {
        return J11PartitionMode::Bfs;
    }
    if (lowered == "metis") {
        return J11PartitionMode::Metis;
    }
    throw std::runtime_error("unknown J11 partition mode: " + name);
}

const char* j11_partition_mode_name(J11PartitionMode mode)
{
    switch (mode) {
    case J11PartitionMode::Bfs:
        return "bfs";
    case J11PartitionMode::Metis:
        return "metis";
    default:
        return "unknown";
    }
}

PermutedIlu0Block::PermutedIlu0Block(std::string name)
    : name_(std::move(name))
    , ilu_(name_)
{
}

void PermutedIlu0Block::set_stream(cudaStream_t stream)
{
    stream_ = stream;
    ilu_.set_stream(stream_);
}

void PermutedIlu0Block::clear_stream_noexcept()
{
    stream_ = nullptr;
    ilu_.clear_stream_noexcept();
}

DeviceCsrMatrixView PermutedIlu0Block::active_matrix_view() const
{
    if (mode_ == J11ReorderMode::None) {
        return original_matrix_;
    }
    return DeviceCsrMatrixView{
        .rows = rows_,
        .cols = rows_,
        .nnz = nnz_,
        .row_ptr = d_perm_row_ptr_.data(),
        .col_idx = d_perm_col_idx_.data(),
        .values = d_perm_values_.data(),
    };
}

void PermutedIlu0Block::analyze(DeviceCsrMatrixView matrix,
                                const HostCsrPattern& host_pattern,
                                J11ReorderMode mode)
{
    if (matrix.rows <= 0 || matrix.rows != matrix.cols || matrix.nnz <= 0 ||
        host_pattern.rows != matrix.rows || host_pattern.cols != matrix.cols ||
        host_pattern.nnz() != matrix.nnz) {
        throw std::runtime_error(name_ + "::analyze received inconsistent J11 inputs");
    }

    original_matrix_ = matrix;
    rows_ = matrix.rows;
    nnz_ = matrix.nnz;
    mode_ = mode;

    if (mode_ != J11ReorderMode::None) {
        build_permuted_pattern(host_pattern);
    }

    ilu_.analyze(active_matrix_view());
    analyzed_ = true;
}

void PermutedIlu0Block::build_permuted_pattern(const HostCsrPattern& host_pattern)
{
    new_to_old_ = compute_new_to_old(host_pattern, mode_);
    if (static_cast<int32_t>(new_to_old_.size()) != rows_) {
        throw std::runtime_error(name_ + " reordering returned the wrong size");
    }

    old_to_new_.assign(static_cast<std::size_t>(rows_), -1);
    for (int32_t new_index = 0; new_index < rows_; ++new_index) {
        const int32_t old_index = new_to_old_[static_cast<std::size_t>(new_index)];
        if (old_index < 0 || old_index >= rows_ ||
            old_to_new_[static_cast<std::size_t>(old_index)] != -1) {
            throw std::runtime_error(name_ + " reordering returned an invalid permutation");
        }
        old_to_new_[static_cast<std::size_t>(old_index)] = new_index;
    }

    std::vector<std::vector<std::pair<int32_t, int32_t>>> rows(static_cast<std::size_t>(rows_));
    for (int32_t old_row = 0; old_row < rows_; ++old_row) {
        const int32_t new_row = old_to_new_[static_cast<std::size_t>(old_row)];
        auto& entries = rows[static_cast<std::size_t>(new_row)];
        for (int32_t pos = host_pattern.row_ptr[static_cast<std::size_t>(old_row)];
             pos < host_pattern.row_ptr[static_cast<std::size_t>(old_row + 1)];
             ++pos) {
            const int32_t old_col = host_pattern.col_idx[static_cast<std::size_t>(pos)];
            const int32_t new_col = old_to_new_[static_cast<std::size_t>(old_col)];
            entries.emplace_back(new_col, pos);
        }
    }

    std::vector<int32_t> row_ptr(static_cast<std::size_t>(rows_ + 1), 0);
    std::vector<int32_t> col_idx(static_cast<std::size_t>(nnz_));
    source_pos_for_permuted_nnz_.resize(static_cast<std::size_t>(nnz_));

    int32_t cursor = 0;
    for (int32_t row = 0; row < rows_; ++row) {
        auto& entries = rows[static_cast<std::size_t>(row)];
        std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });
        row_ptr[static_cast<std::size_t>(row)] = cursor;
        for (const auto& [new_col, source_pos] : entries) {
            col_idx[static_cast<std::size_t>(cursor)] = new_col;
            source_pos_for_permuted_nnz_[static_cast<std::size_t>(cursor)] = source_pos;
            ++cursor;
        }
    }
    row_ptr[static_cast<std::size_t>(rows_)] = cursor;
    if (cursor != nnz_) {
        throw std::runtime_error(name_ + " permuted J11 pattern has wrong nnz");
    }

    d_perm_row_ptr_.assign(row_ptr.data(), row_ptr.size());
    d_perm_col_idx_.assign(col_idx.data(), col_idx.size());
    d_new_to_old_.assign(new_to_old_.data(), new_to_old_.size());
    d_old_to_new_.assign(old_to_new_.data(), old_to_new_.size());
    d_source_pos_for_permuted_nnz_.assign(source_pos_for_permuted_nnz_.data(),
                                          source_pos_for_permuted_nnz_.size());
    d_perm_values_.resize(static_cast<std::size_t>(nnz_));
    d_perm_rhs_.resize(static_cast<std::size_t>(rows_));
    d_perm_solution_.resize(static_cast<std::size_t>(rows_));
}

void PermutedIlu0Block::factorize()
{
    if (!analyzed_) {
        throw std::runtime_error(name_ + "::factorize called before analyze");
    }

    if (mode_ != J11ReorderMode::None) {
        permute_values_kernel<<<grid_for(nnz_), kBlockSize, 0, stream_>>>(
            nnz_,
            original_matrix_.values,
            d_source_pos_for_permuted_nnz_.data(),
            d_perm_values_.data());
        CUDA_CHECK(cudaGetLastError());
    }
    ilu_.factorize();
}

void PermutedIlu0Block::solve(const double* rhs_device, double* out_device)
{
    if (!analyzed_) {
        throw std::runtime_error(name_ + "::solve called before analyze");
    }
    if (mode_ == J11ReorderMode::None) {
        ilu_.solve(rhs_device, out_device);
        return;
    }

    permute_rhs_kernel<<<grid_for(rows_), kBlockSize, 0, stream_>>>(
        rows_, rhs_device, d_new_to_old_.data(), d_perm_rhs_.data());
    CUDA_CHECK(cudaGetLastError());
    ilu_.solve(d_perm_rhs_.data(), d_perm_solution_.data());
    inverse_permute_solution_kernel<<<grid_for(rows_), kBlockSize, 0, stream_>>>(
        rows_, d_perm_solution_.data(), d_new_to_old_.data(), out_device);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace exp_20260415::block_ilu
