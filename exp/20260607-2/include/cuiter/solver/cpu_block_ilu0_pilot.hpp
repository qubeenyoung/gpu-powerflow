#pragma once

#include "cuiter/core/csr_matrix.hpp"
#include "cuiter/reorder/metis_partition.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cuiter::cpu_pilot {

struct CpuBlockIlu0Options {
    int32_t block_size = 32;
    int32_t bicgstab_iters = 2;
    int32_t gmres_iters = 2;
    double diag_shift_scale = 1.0e-8;
    bool use_block_ilu0 = true;
    bool use_block_coloring_order = true;
    bool use_gmres = false;
};

struct CpuBlockIlu0Result {
    std::vector<double> solution;
    bool factor_failed = false;
    std::string stop_reason;
    int32_t iterations = 0;
    double residual_norm2 = 0.0;
    double relative_residual_norm2 = 0.0;
    int32_t num_blocks = 0;
    int32_t block_nnz = 0;
    int32_t l_levels = 0;
    int32_t u_levels = 0;
    double avg_level_width = 0.0;
    int32_t max_level_width = 0;
    double setup_seconds = 0.0;
    double factor_seconds = 0.0;
    double apply_seconds = 0.0;
    double forward_seconds = 0.0;
    double backward_seconds = 0.0;
    double bicgstab_total_seconds = 0.0;
    double spmv_seconds = 0.0;
    double dot_seconds = 0.0;
    double update_seconds = 0.0;
    double gmres_total_seconds = 0.0;
    double gmres_orthogonalization_seconds = 0.0;
};

namespace detail {

struct DenseBlock {
    int32_t row = 0;
    int32_t col = 0;
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<float> values;
};

struct BlockPattern {
    int32_t n = 0;
    int32_t num_blocks = 0;
    std::vector<int32_t> block_starts;
    std::vector<int32_t> block_dims;
    std::vector<DenseBlock> blocks;
    std::vector<std::vector<int32_t>> row_block_indices;
    std::vector<int32_t> diagonal_index;
    std::unordered_map<uint64_t, int32_t> block_index;
};

template <typename Fn>
double timed_seconds(Fn&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

inline uint64_t edge_key(int32_t row, int32_t col)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(col));
}

inline int32_t edge_key_row(uint64_t key)
{
    return static_cast<int32_t>(static_cast<uint32_t>(key >> 32));
}

inline int32_t edge_key_col(uint64_t key)
{
    return static_cast<int32_t>(static_cast<uint32_t>(key & 0xffffffffu));
}

inline double norm2(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values) {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
}

inline double dot(const std::vector<double>& lhs, const std::vector<double>& rhs)
{
    long double sum = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        sum += static_cast<long double>(lhs[i]) * static_cast<long double>(rhs[i]);
    }
    return static_cast<double>(sum);
}

inline bool solve_dense_linear_system(std::vector<double> a,
                                      std::vector<double> b,
                                      int32_t n,
                                      std::vector<double>& x)
{
    x.assign(static_cast<std::size_t>(n), 0.0);
    for (int32_t col = 0; col < n; ++col) {
        int32_t pivot = col;
        double pivot_abs = std::abs(a[static_cast<std::size_t>(col * n + col)]);
        for (int32_t row = col + 1; row < n; ++row) {
            const double candidate = std::abs(a[static_cast<std::size_t>(row * n + col)]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot = row;
            }
        }
        if (!std::isfinite(pivot_abs) || pivot_abs <= std::numeric_limits<double>::min()) {
            return false;
        }
        if (pivot != col) {
            for (int32_t j = col; j < n; ++j) {
                std::swap(a[static_cast<std::size_t>(col * n + j)],
                          a[static_cast<std::size_t>(pivot * n + j)]);
            }
            std::swap(b[static_cast<std::size_t>(col)],
                      b[static_cast<std::size_t>(pivot)]);
        }
        const double diag = a[static_cast<std::size_t>(col * n + col)];
        for (int32_t row = col + 1; row < n; ++row) {
            const double factor = a[static_cast<std::size_t>(row * n + col)] / diag;
            a[static_cast<std::size_t>(row * n + col)] = 0.0;
            for (int32_t j = col + 1; j < n; ++j) {
                a[static_cast<std::size_t>(row * n + j)] -=
                    factor * a[static_cast<std::size_t>(col * n + j)];
            }
            b[static_cast<std::size_t>(row)] -= factor * b[static_cast<std::size_t>(col)];
        }
    }
    for (int32_t row = n - 1; row >= 0; --row) {
        double sum = b[static_cast<std::size_t>(row)];
        for (int32_t col = row + 1; col < n; ++col) {
            sum -= a[static_cast<std::size_t>(row * n + col)] *
                   x[static_cast<std::size_t>(col)];
        }
        const double diag = a[static_cast<std::size_t>(row * n + row)];
        if (!std::isfinite(diag) || std::abs(diag) <= std::numeric_limits<double>::min()) {
            return false;
        }
        x[static_cast<std::size_t>(row)] = sum / diag;
    }
    return true;
}

inline double ratio(double numerator, double denominator)
{
    return denominator > 0.0 ? numerator / denominator : 0.0;
}

inline void sort_unique(std::vector<int32_t>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

inline std::vector<int32_t> block_of_new_indices(int32_t n,
                                                 const std::vector<int32_t>& starts,
                                                 const std::vector<int32_t>& dims)
{
    std::vector<int32_t> block_of_new(static_cast<std::size_t>(n), -1);
    for (int32_t block = 0; block < static_cast<int32_t>(dims.size()); ++block) {
        const int32_t begin = starts[static_cast<std::size_t>(block)];
        const int32_t end = begin + dims[static_cast<std::size_t>(block)];
        for (int32_t index = begin; index < end; ++index) {
            block_of_new[static_cast<std::size_t>(index)] = block;
        }
    }
    return block_of_new;
}

inline std::vector<std::vector<int32_t>> build_block_undirected_graph(
    const CsrMatrix& matrix,
    const MetisPermutation& permutation,
    const PermutedCsrPattern& pattern)
{
    const int32_t nb = static_cast<int32_t>(permutation.block_sizes.size());
    std::vector<std::vector<int32_t>> adjacency(static_cast<std::size_t>(nb));
    const std::vector<int32_t> block_of_new =
        block_of_new_indices(matrix.rows, permutation.block_starts, permutation.block_sizes);
    for (int32_t row = 0; row < pattern.rows; ++row) {
        const int32_t rb = block_of_new[static_cast<std::size_t>(row)];
        for (int32_t pos = pattern.row_ptr[static_cast<std::size_t>(row)];
             pos < pattern.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = pattern.col_idx[static_cast<std::size_t>(pos)];
            const int32_t cb = block_of_new[static_cast<std::size_t>(col)];
            if (rb == cb) {
                continue;
            }
            adjacency[static_cast<std::size_t>(rb)].push_back(cb);
            adjacency[static_cast<std::size_t>(cb)].push_back(rb);
        }
    }
    for (auto& neighbors : adjacency) {
        sort_unique(neighbors);
    }
    return adjacency;
}

inline std::vector<int32_t> greedy_coloring_order(const std::vector<std::vector<int32_t>>& adjacency)
{
    const int32_t n = static_cast<int32_t>(adjacency.size());
    std::vector<int32_t> color(static_cast<std::size_t>(n), -1);
    int32_t max_color = -1;
    for (int32_t node = 0; node < n; ++node) {
        std::vector<char> used(static_cast<std::size_t>(std::max(1, max_color + 2)), 0);
        for (int32_t neighbor : adjacency[static_cast<std::size_t>(node)]) {
            const int32_t c = color[static_cast<std::size_t>(neighbor)];
            if (c >= 0 && c < static_cast<int32_t>(used.size())) {
                used[static_cast<std::size_t>(c)] = 1;
            }
        }
        int32_t chosen = 0;
        while (chosen < static_cast<int32_t>(used.size()) && used[static_cast<std::size_t>(chosen)]) {
            ++chosen;
        }
        color[static_cast<std::size_t>(node)] = chosen;
        max_color = std::max(max_color, chosen);
    }
    std::vector<int32_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t lhs, int32_t rhs) {
        if (color[static_cast<std::size_t>(lhs)] != color[static_cast<std::size_t>(rhs)]) {
            return color[static_cast<std::size_t>(lhs)] < color[static_cast<std::size_t>(rhs)];
        }
        return lhs < rhs;
    });
    return order;
}

inline MetisPermutation build_colored_block_permutation(const CsrMatrix& matrix,
                                                        int32_t block_size)
{
    const MetisPermutation base = build_metis_permutation(matrix, block_size);
    const PermutedCsrPattern base_pattern = build_permuted_csr_pattern(matrix, base);
    const std::vector<std::vector<int32_t>> adjacency =
        build_block_undirected_graph(matrix, base, base_pattern);
    const std::vector<int32_t> new_block_to_old = greedy_coloring_order(adjacency);

    MetisPermutation permutation;
    permutation.new_to_old.reserve(base.new_to_old.size());
    permutation.old_to_new.assign(base.old_to_new.size(), -1);
    for (int32_t old_block : new_block_to_old) {
        const int32_t begin = base.block_starts[static_cast<std::size_t>(old_block)];
        const int32_t dim = base.block_sizes[static_cast<std::size_t>(old_block)];
        permutation.block_starts.push_back(static_cast<int32_t>(permutation.new_to_old.size()));
        permutation.block_sizes.push_back(dim);
        for (int32_t offset = 0; offset < dim; ++offset) {
            const int32_t old_index = base.new_to_old[static_cast<std::size_t>(begin + offset)];
            const int32_t new_index = static_cast<int32_t>(permutation.new_to_old.size());
            permutation.new_to_old.push_back(old_index);
            permutation.old_to_new[static_cast<std::size_t>(old_index)] = new_index;
        }
    }
    permutation.stats = compute_partition_stats(matrix,
                                                &matrix.values,
                                                permutation.old_to_new,
                                                permutation.block_starts,
                                                permutation.block_sizes,
                                                block_size,
                                                nullptr,
                                                nullptr,
                                                "unknown_metis_block_coloring");
    return permutation;
}

inline CsrMatrix build_permuted_matrix(const CsrMatrix& matrix,
                                       const MetisPermutation& permutation)
{
    const PermutedCsrPattern pattern = build_permuted_csr_pattern(matrix, permutation);
    CsrMatrix out;
    out.rows = matrix.rows;
    out.cols = matrix.cols;
    out.row_ptr = pattern.row_ptr;
    out.col_idx = pattern.col_idx;
    out.values.resize(pattern.value_source_index.size());
    for (std::size_t i = 0; i < pattern.value_source_index.size(); ++i) {
        out.values[i] = matrix.values[static_cast<std::size_t>(pattern.value_source_index[i])];
    }
    return out;
}

inline std::vector<double> permute_vector(const std::vector<double>& values,
                                          const MetisPermutation& permutation)
{
    std::vector<double> out(values.size(), 0.0);
    for (std::size_t new_index = 0; new_index < permutation.new_to_old.size(); ++new_index) {
        out[new_index] = values[static_cast<std::size_t>(permutation.new_to_old[new_index])];
    }
    return out;
}

inline std::vector<double> unpermute_vector(const std::vector<double>& values,
                                            const MetisPermutation& permutation)
{
    std::vector<double> out(values.size(), 0.0);
    for (std::size_t new_index = 0; new_index < permutation.new_to_old.size(); ++new_index) {
        out[static_cast<std::size_t>(permutation.new_to_old[new_index])] = values[new_index];
    }
    return out;
}

inline std::vector<double> spmv(const CsrMatrix& matrix, const std::vector<double>& x)
{
    std::vector<double> y(static_cast<std::size_t>(matrix.rows), 0.0);
    for (int32_t row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            sum += matrix.values[static_cast<std::size_t>(pos)] *
                   x[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(pos)])];
        }
        y[static_cast<std::size_t>(row)] = sum;
    }
    return y;
}

inline BlockPattern make_block_pattern(const CsrMatrix& matrix,
                                       const std::vector<int32_t>& block_starts,
                                       const std::vector<int32_t>& block_dims,
                                       bool diagonal_only)
{
    BlockPattern pattern;
    pattern.n = matrix.rows;
    pattern.num_blocks = static_cast<int32_t>(block_dims.size());
    pattern.block_starts = block_starts;
    pattern.block_dims = block_dims;
    pattern.row_block_indices.assign(static_cast<std::size_t>(pattern.num_blocks), {});
    pattern.diagonal_index.assign(static_cast<std::size_t>(pattern.num_blocks), -1);
    const std::vector<int32_t> block_of_new =
        block_of_new_indices(matrix.rows, block_starts, block_dims);
    std::unordered_set<uint64_t> keys;
    for (int32_t row = 0; row < matrix.rows; ++row) {
        const int32_t rb = block_of_new[static_cast<std::size_t>(row)];
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = matrix.col_idx[static_cast<std::size_t>(pos)];
            const int32_t cb = block_of_new[static_cast<std::size_t>(col)];
            if (!diagonal_only || rb == cb) {
                keys.insert(edge_key(rb, cb));
            }
        }
    }
    for (int32_t block = 0; block < pattern.num_blocks; ++block) {
        keys.insert(edge_key(block, block));
    }
    std::vector<uint64_t> sorted_keys(keys.begin(), keys.end());
    std::sort(sorted_keys.begin(), sorted_keys.end(), [](uint64_t lhs, uint64_t rhs) {
        if (edge_key_row(lhs) != edge_key_row(rhs)) {
            return edge_key_row(lhs) < edge_key_row(rhs);
        }
        return edge_key_col(lhs) < edge_key_col(rhs);
    });
    for (uint64_t key : sorted_keys) {
        DenseBlock block;
        block.row = edge_key_row(key);
        block.col = edge_key_col(key);
        block.rows = block_dims[static_cast<std::size_t>(block.row)];
        block.cols = block_dims[static_cast<std::size_t>(block.col)];
        block.values.assign(static_cast<std::size_t>(block.rows * block.cols), 0.0f);
        const int32_t index = static_cast<int32_t>(pattern.blocks.size());
        pattern.block_index[key] = index;
        pattern.row_block_indices[static_cast<std::size_t>(block.row)].push_back(index);
        if (block.row == block.col) {
            pattern.diagonal_index[static_cast<std::size_t>(block.row)] = index;
        }
        pattern.blocks.push_back(std::move(block));
    }
    return pattern;
}

inline void scatter_dense_blocks(const CsrMatrix& matrix, BlockPattern& pattern)
{
    const std::vector<int32_t> block_of_new =
        block_of_new_indices(matrix.rows, pattern.block_starts, pattern.block_dims);
    for (auto& block : pattern.blocks) {
        std::fill(block.values.begin(), block.values.end(), 0.0f);
    }
    for (int32_t row = 0; row < matrix.rows; ++row) {
        const int32_t rb = block_of_new[static_cast<std::size_t>(row)];
        const int32_t local_row = row - pattern.block_starts[static_cast<std::size_t>(rb)];
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = matrix.col_idx[static_cast<std::size_t>(pos)];
            const int32_t cb = block_of_new[static_cast<std::size_t>(col)];
            const auto it = pattern.block_index.find(edge_key(rb, cb));
            if (it == pattern.block_index.end()) {
                continue;
            }
            DenseBlock& block = pattern.blocks[static_cast<std::size_t>(it->second)];
            const int32_t local_col = col - pattern.block_starts[static_cast<std::size_t>(cb)];
            block.values[static_cast<std::size_t>(local_row * block.cols + local_col)] +=
                static_cast<float>(matrix.values[static_cast<std::size_t>(pos)]);
        }
    }
}

inline int32_t find_block(const BlockPattern& pattern, int32_t row, int32_t col)
{
    const auto it = pattern.block_index.find(edge_key(row, col));
    return it == pattern.block_index.end() ? -1 : it->second;
}

inline bool invert_dense_block(std::vector<float>& matrix,
                               int32_t n,
                               double shift_scale,
                               std::vector<float>& inverse)
{
    double mean_abs_diag = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        mean_abs_diag += std::abs(static_cast<double>(matrix[static_cast<std::size_t>(i * n + i)]));
    }
    mean_abs_diag /= static_cast<double>(std::max(1, n));
    const double scale = std::max(1.0, mean_abs_diag);
    const std::vector<double> shifts = {0.0, shift_scale * scale, 1.0e-6 * scale, 1.0e-4 * scale};
    for (double shift : shifts) {
        std::vector<double> a(static_cast<std::size_t>(n * n), 0.0);
        std::vector<double> inv(static_cast<std::size_t>(n * n), 0.0);
        for (int32_t row = 0; row < n; ++row) {
            inv[static_cast<std::size_t>(row * n + row)] = 1.0;
            for (int32_t col = 0; col < n; ++col) {
                a[static_cast<std::size_t>(row * n + col)] =
                    static_cast<double>(matrix[static_cast<std::size_t>(row * n + col)]);
            }
            a[static_cast<std::size_t>(row * n + row)] += shift;
        }
        bool ok = true;
        for (int32_t col = 0; col < n; ++col) {
            int32_t pivot = col;
            double pivot_abs = std::abs(a[static_cast<std::size_t>(col * n + col)]);
            for (int32_t row = col + 1; row < n; ++row) {
                const double candidate = std::abs(a[static_cast<std::size_t>(row * n + col)]);
                if (candidate > pivot_abs) {
                    pivot_abs = candidate;
                    pivot = row;
                }
            }
            if (!std::isfinite(pivot_abs) || pivot_abs < 1.0e-12) {
                ok = false;
                break;
            }
            if (pivot != col) {
                for (int32_t j = 0; j < n; ++j) {
                    std::swap(a[static_cast<std::size_t>(col * n + j)],
                              a[static_cast<std::size_t>(pivot * n + j)]);
                    std::swap(inv[static_cast<std::size_t>(col * n + j)],
                              inv[static_cast<std::size_t>(pivot * n + j)]);
                }
            }
            const double diag = a[static_cast<std::size_t>(col * n + col)];
            for (int32_t j = 0; j < n; ++j) {
                a[static_cast<std::size_t>(col * n + j)] /= diag;
                inv[static_cast<std::size_t>(col * n + j)] /= diag;
            }
            for (int32_t row = 0; row < n; ++row) {
                if (row == col) {
                    continue;
                }
                const double factor = a[static_cast<std::size_t>(row * n + col)];
                for (int32_t j = 0; j < n; ++j) {
                    a[static_cast<std::size_t>(row * n + j)] -=
                        factor * a[static_cast<std::size_t>(col * n + j)];
                    inv[static_cast<std::size_t>(row * n + j)] -=
                        factor * inv[static_cast<std::size_t>(col * n + j)];
                }
            }
        }
        if (!ok) {
            continue;
        }
        inverse.resize(static_cast<std::size_t>(n * n));
        for (std::size_t i = 0; i < inverse.size(); ++i) {
            inverse[i] = static_cast<float>(inv[i]);
        }
        if (shift != 0.0) {
            for (int32_t i = 0; i < n; ++i) {
                matrix[static_cast<std::size_t>(i * n + i)] += static_cast<float>(shift);
            }
        }
        return true;
    }
    return false;
}

inline void right_multiply_in_place(DenseBlock& lhs, const std::vector<float>& rhs_inv)
{
    std::vector<float> out(lhs.values.size(), 0.0f);
    for (int32_t i = 0; i < lhs.rows; ++i) {
        for (int32_t j = 0; j < lhs.cols; ++j) {
            double sum = 0.0;
            for (int32_t k = 0; k < lhs.cols; ++k) {
                sum += static_cast<double>(lhs.values[static_cast<std::size_t>(i * lhs.cols + k)]) *
                       static_cast<double>(rhs_inv[static_cast<std::size_t>(k * lhs.cols + j)]);
            }
            out[static_cast<std::size_t>(i * lhs.cols + j)] = static_cast<float>(sum);
        }
    }
    lhs.values.swap(out);
}

inline void subtract_product(DenseBlock& target, const DenseBlock& lhs, const DenseBlock& rhs)
{
    for (int32_t i = 0; i < target.rows; ++i) {
        for (int32_t j = 0; j < target.cols; ++j) {
            double sum = 0.0;
            for (int32_t k = 0; k < lhs.cols; ++k) {
                sum += static_cast<double>(lhs.values[static_cast<std::size_t>(i * lhs.cols + k)]) *
                       static_cast<double>(rhs.values[static_cast<std::size_t>(k * rhs.cols + j)]);
            }
            target.values[static_cast<std::size_t>(i * target.cols + j)] -=
                static_cast<float>(sum);
        }
    }
}

inline void dense_gemv_subtract(const DenseBlock& block,
                                const std::vector<double>& x,
                                int32_t x_offset,
                                std::vector<double>& y,
                                int32_t y_offset)
{
    for (int32_t i = 0; i < block.rows; ++i) {
        double sum = 0.0;
        for (int32_t j = 0; j < block.cols; ++j) {
            sum += static_cast<double>(block.values[static_cast<std::size_t>(i * block.cols + j)]) *
                   x[static_cast<std::size_t>(x_offset + j)];
        }
        y[static_cast<std::size_t>(y_offset + i)] -= sum;
    }
}

struct CpuPreconditioner {
    BlockPattern pattern;
    std::vector<std::vector<float>> inverse_diag;
    bool ilu0 = true;
    double apply_seconds = 0.0;
    double forward_seconds = 0.0;
    double backward_seconds = 0.0;

    bool setup(const CsrMatrix& matrix,
               const std::vector<int32_t>& block_starts,
               const std::vector<int32_t>& block_dims,
               double diag_shift_scale,
               double& setup_seconds,
               double& factor_seconds)
    {
        bool ok = true;
        setup_seconds += timed_seconds([&] {
            pattern = make_block_pattern(matrix, block_starts, block_dims, !ilu0);
            scatter_dense_blocks(matrix, pattern);
            inverse_diag.assign(pattern.block_dims.size(), {});
        });
        factor_seconds += timed_seconds([&] {
            if (!ilu0) {
                for (int32_t block = 0; block < pattern.num_blocks; ++block) {
                    DenseBlock& diag = pattern.blocks[static_cast<std::size_t>(
                        pattern.diagonal_index[static_cast<std::size_t>(block)])];
                    if (!invert_dense_block(diag.values,
                                            diag.rows,
                                            diag_shift_scale,
                                            inverse_diag[static_cast<std::size_t>(block)])) {
                        ok = false;
                        return;
                    }
                }
                return;
            }
            for (int32_t i = 0; i < pattern.num_blocks; ++i) {
                std::vector<int32_t> row_blocks = pattern.row_block_indices[static_cast<std::size_t>(i)];
                std::sort(row_blocks.begin(), row_blocks.end(), [&](int32_t lhs, int32_t rhs) {
                    return pattern.blocks[static_cast<std::size_t>(lhs)].col <
                           pattern.blocks[static_cast<std::size_t>(rhs)].col;
                });
                for (int32_t lower_index : row_blocks) {
                    DenseBlock& lik = pattern.blocks[static_cast<std::size_t>(lower_index)];
                    const int32_t k = lik.col;
                    if (k >= i) {
                        continue;
                    }
                    right_multiply_in_place(lik, inverse_diag[static_cast<std::size_t>(k)]);
                    for (int32_t ukj_index : pattern.row_block_indices[static_cast<std::size_t>(k)]) {
                        const DenseBlock& ukj = pattern.blocks[static_cast<std::size_t>(ukj_index)];
                        const int32_t j = ukj.col;
                        if (j <= k) {
                            continue;
                        }
                        const int32_t target_index = find_block(pattern, i, j);
                        if (target_index >= 0) {
                            subtract_product(pattern.blocks[static_cast<std::size_t>(target_index)],
                                             lik,
                                             ukj);
                        }
                    }
                }
                DenseBlock& diag = pattern.blocks[static_cast<std::size_t>(
                    pattern.diagonal_index[static_cast<std::size_t>(i)])];
                if (!invert_dense_block(diag.values,
                                        diag.rows,
                                        diag_shift_scale,
                                        inverse_diag[static_cast<std::size_t>(i)])) {
                    ok = false;
                    return;
                }
            }
        });
        return ok;
    }

    std::vector<double> apply(const std::vector<double>& r)
    {
        std::vector<double> z(r.size(), 0.0);
        if (!ilu0) {
            apply_seconds += timed_seconds([&] {
                for (int32_t block = 0; block < pattern.num_blocks; ++block) {
                    const int32_t begin = pattern.block_starts[static_cast<std::size_t>(block)];
                    const int32_t dim = pattern.block_dims[static_cast<std::size_t>(block)];
                    const auto& inv = inverse_diag[static_cast<std::size_t>(block)];
                    for (int32_t i = 0; i < dim; ++i) {
                        double sum = 0.0;
                        for (int32_t j = 0; j < dim; ++j) {
                            sum += static_cast<double>(inv[static_cast<std::size_t>(i * dim + j)]) *
                                   r[static_cast<std::size_t>(begin + j)];
                        }
                        z[static_cast<std::size_t>(begin + i)] = sum;
                    }
                }
            });
            return z;
        }

        std::vector<double> y(r.size(), 0.0);
        const double fwd = timed_seconds([&] {
            for (int32_t block = 0; block < pattern.num_blocks; ++block) {
                const int32_t begin = pattern.block_starts[static_cast<std::size_t>(block)];
                const int32_t dim = pattern.block_dims[static_cast<std::size_t>(block)];
                for (int32_t i = 0; i < dim; ++i) {
                    y[static_cast<std::size_t>(begin + i)] = r[static_cast<std::size_t>(begin + i)];
                }
                for (int32_t index : pattern.row_block_indices[static_cast<std::size_t>(block)]) {
                    const DenseBlock& offdiag = pattern.blocks[static_cast<std::size_t>(index)];
                    if (offdiag.col < block) {
                        dense_gemv_subtract(offdiag,
                                            y,
                                            pattern.block_starts[static_cast<std::size_t>(offdiag.col)],
                                            y,
                                            begin);
                    }
                }
            }
        });
        const double bwd = timed_seconds([&] {
            for (int32_t block = pattern.num_blocks - 1; block >= 0; --block) {
                const int32_t begin = pattern.block_starts[static_cast<std::size_t>(block)];
                const int32_t dim = pattern.block_dims[static_cast<std::size_t>(block)];
                std::vector<double> rhs(static_cast<std::size_t>(dim), 0.0);
                for (int32_t i = 0; i < dim; ++i) {
                    rhs[static_cast<std::size_t>(i)] = y[static_cast<std::size_t>(begin + i)];
                }
                for (int32_t index : pattern.row_block_indices[static_cast<std::size_t>(block)]) {
                    const DenseBlock& offdiag = pattern.blocks[static_cast<std::size_t>(index)];
                    if (offdiag.col > block) {
                        for (int32_t i = 0; i < offdiag.rows; ++i) {
                            double sum = 0.0;
                            const int32_t col_begin =
                                pattern.block_starts[static_cast<std::size_t>(offdiag.col)];
                            for (int32_t j = 0; j < offdiag.cols; ++j) {
                                sum += static_cast<double>(
                                           offdiag.values[static_cast<std::size_t>(i * offdiag.cols + j)]) *
                                       z[static_cast<std::size_t>(col_begin + j)];
                            }
                            rhs[static_cast<std::size_t>(i)] -= sum;
                        }
                    }
                }
                const auto& inv = inverse_diag[static_cast<std::size_t>(block)];
                for (int32_t i = 0; i < dim; ++i) {
                    double sum = 0.0;
                    for (int32_t j = 0; j < dim; ++j) {
                        sum += static_cast<double>(inv[static_cast<std::size_t>(i * dim + j)]) *
                               rhs[static_cast<std::size_t>(j)];
                    }
                    z[static_cast<std::size_t>(begin + i)] = sum;
                }
            }
        });
        forward_seconds += fwd;
        backward_seconds += bwd;
        apply_seconds += fwd + bwd;
        return z;
    }
};

inline void compute_levels(const BlockPattern& pattern, CpuBlockIlu0Result& result)
{
    const int32_t nb = pattern.num_blocks;
    std::vector<std::vector<int32_t>> l_deps(static_cast<std::size_t>(nb));
    std::vector<std::vector<int32_t>> u_deps(static_cast<std::size_t>(nb));
    for (const DenseBlock& block : pattern.blocks) {
        if (block.row > block.col) {
            l_deps[static_cast<std::size_t>(block.row)].push_back(block.col);
        } else if (block.row < block.col) {
            u_deps[static_cast<std::size_t>(block.row)].push_back(block.col);
        }
    }
    std::vector<int32_t> l_level(static_cast<std::size_t>(nb), 0);
    for (int32_t i = 0; i < nb; ++i) {
        for (int32_t dep : l_deps[static_cast<std::size_t>(i)]) {
            l_level[static_cast<std::size_t>(i)] =
                std::max(l_level[static_cast<std::size_t>(i)],
                         l_level[static_cast<std::size_t>(dep)] + 1);
        }
    }
    std::vector<int32_t> u_level(static_cast<std::size_t>(nb), 0);
    for (int32_t i = nb - 1; i >= 0; --i) {
        for (int32_t dep : u_deps[static_cast<std::size_t>(i)]) {
            u_level[static_cast<std::size_t>(i)] =
                std::max(u_level[static_cast<std::size_t>(i)],
                         u_level[static_cast<std::size_t>(dep)] + 1);
        }
    }
    auto avg_width = [](const std::vector<int32_t>& levels, int32_t& max_width) {
        const int32_t nlevels = *std::max_element(levels.begin(), levels.end()) + 1;
        std::vector<int32_t> widths(static_cast<std::size_t>(nlevels), 0);
        for (int32_t level : levels) {
            ++widths[static_cast<std::size_t>(level)];
        }
        max_width = *std::max_element(widths.begin(), widths.end());
        return static_cast<double>(levels.size()) / static_cast<double>(nlevels);
    };
    result.l_levels = *std::max_element(l_level.begin(), l_level.end()) + 1;
    result.u_levels = *std::max_element(u_level.begin(), u_level.end()) + 1;
    int32_t lmax = 0;
    int32_t umax = 0;
    const double lavg = avg_width(l_level, lmax);
    const double uavg = avg_width(u_level, umax);
    result.avg_level_width = 0.5 * (lavg + uavg);
    result.max_level_width = std::max(lmax, umax);
}

inline std::vector<double> bicgstab_fixed(const CsrMatrix& matrix,
                                          const std::vector<double>& rhs,
                                          int32_t max_iters,
                                          CpuPreconditioner& preconditioner,
                                          CpuBlockIlu0Result& result)
{
    const int32_t n = matrix.rows;
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    std::vector<double> r = rhs;
    const std::vector<double> r_hat = rhs;
    std::vector<double> p(static_cast<std::size_t>(n), 0.0);
    std::vector<double> v(static_cast<std::size_t>(n), 0.0);
    double rho_old = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    result.bicgstab_total_seconds += timed_seconds([&] {
        for (int32_t iter = 0; iter < max_iters; ++iter) {
            double rho = 0.0;
            result.dot_seconds += timed_seconds([&] { rho = dot(r_hat, r); });
            if (!std::isfinite(rho) || std::abs(rho) <= std::numeric_limits<double>::min()) {
                result.stop_reason = "bicgstab_rho_breakdown";
                return;
            }
            result.update_seconds += timed_seconds([&] {
                if (iter == 0) {
                    p = r;
                } else {
                    const double beta = (rho / rho_old) * (alpha / omega);
                    for (int32_t i = 0; i < n; ++i) {
                        p[static_cast<std::size_t>(i)] =
                            r[static_cast<std::size_t>(i)] +
                            beta * (p[static_cast<std::size_t>(i)] -
                                    omega * v[static_cast<std::size_t>(i)]);
                    }
                }
            });
            const std::vector<double> p_hat = preconditioner.apply(p);
            result.spmv_seconds += timed_seconds([&] { v = spmv(matrix, p_hat); });
            double rhat_v = 0.0;
            result.dot_seconds += timed_seconds([&] { rhat_v = dot(r_hat, v); });
            if (!std::isfinite(rhat_v) || std::abs(rhat_v) <= std::numeric_limits<double>::min()) {
                result.stop_reason = "bicgstab_alpha_breakdown";
                return;
            }
            alpha = rho / rhat_v;
            std::vector<double> s(static_cast<std::size_t>(n), 0.0);
            result.update_seconds += timed_seconds([&] {
                for (int32_t i = 0; i < n; ++i) {
                    s[static_cast<std::size_t>(i)] =
                        r[static_cast<std::size_t>(i)] - alpha * v[static_cast<std::size_t>(i)];
                }
            });
            const std::vector<double> s_hat = preconditioner.apply(s);
            std::vector<double> t;
            result.spmv_seconds += timed_seconds([&] { t = spmv(matrix, s_hat); });
            double ts = 0.0;
            double tt = 0.0;
            result.dot_seconds += timed_seconds([&] {
                ts = dot(t, s);
                tt = dot(t, t);
            });
            if (!std::isfinite(ts) || !std::isfinite(tt) ||
                tt <= std::numeric_limits<double>::min()) {
                result.stop_reason = "bicgstab_omega_breakdown";
                return;
            }
            omega = ts / tt;
            result.update_seconds += timed_seconds([&] {
                for (int32_t i = 0; i < n; ++i) {
                    x[static_cast<std::size_t>(i)] +=
                        alpha * p_hat[static_cast<std::size_t>(i)] +
                        omega * s_hat[static_cast<std::size_t>(i)];
                    r[static_cast<std::size_t>(i)] =
                        s[static_cast<std::size_t>(i)] - omega * t[static_cast<std::size_t>(i)];
                }
            });
            rho_old = rho;
            result.iterations = iter + 1;
            if (!std::isfinite(omega) || std::abs(omega) <= std::numeric_limits<double>::min()) {
                result.stop_reason = "bicgstab_omega_zero";
                return;
            }
        }
        result.stop_reason = "bicgstab_fixed_iter_cpu_block_ilu0_pilot";
    });
    return x;
}

inline std::vector<double> gmres_fixed(const CsrMatrix& matrix,
                                       const std::vector<double>& rhs,
                                       int32_t max_iters,
                                       CpuPreconditioner& preconditioner,
                                       CpuBlockIlu0Result& result)
{
    const int32_t n = matrix.rows;
    const int32_t m = std::max(1, max_iters);
    std::vector<std::vector<double>> v(static_cast<std::size_t>(m + 1),
                                       std::vector<double>(static_cast<std::size_t>(n), 0.0));
    std::vector<std::vector<double>> z(static_cast<std::size_t>(m),
                                       std::vector<double>(static_cast<std::size_t>(n), 0.0));
    std::vector<std::vector<double>> h(static_cast<std::size_t>(m + 1),
                                       std::vector<double>(static_cast<std::size_t>(m), 0.0));
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);

    result.gmres_total_seconds += timed_seconds([&] {
        const double beta = norm2(rhs);
        if (!std::isfinite(beta) || beta <= std::numeric_limits<double>::min()) {
            result.stop_reason = "gmres_zero_rhs";
            return;
        }
        result.update_seconds += timed_seconds([&] {
            for (int32_t i = 0; i < n; ++i) {
                v[0][static_cast<std::size_t>(i)] =
                    rhs[static_cast<std::size_t>(i)] / beta;
            }
        });

        int32_t basis_count = 0;
        for (int32_t j = 0; j < m; ++j) {
            z[static_cast<std::size_t>(j)] =
                preconditioner.apply(v[static_cast<std::size_t>(j)]);
            std::vector<double> w;
            result.spmv_seconds += timed_seconds([&] {
                w = spmv(matrix, z[static_cast<std::size_t>(j)]);
            });

            const double orth_seconds = timed_seconds([&] {
                for (int32_t i = 0; i <= j; ++i) {
                    h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                        dot(w, v[static_cast<std::size_t>(i)]);
                    for (int32_t row = 0; row < n; ++row) {
                        w[static_cast<std::size_t>(row)] -=
                            h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                            v[static_cast<std::size_t>(i)][static_cast<std::size_t>(row)];
                    }
                }
            });
            result.gmres_orthogonalization_seconds += orth_seconds;
            result.dot_seconds += orth_seconds;

            double h_next = 0.0;
            result.dot_seconds += timed_seconds([&] { h_next = norm2(w); });
            h[static_cast<std::size_t>(j + 1)][static_cast<std::size_t>(j)] = h_next;
            basis_count = j + 1;
            result.iterations = basis_count;
            if (!std::isfinite(h_next) || h_next <= 1.0e-14) {
                result.stop_reason = "gmres_arnoldi_breakdown";
                break;
            }
            result.update_seconds += timed_seconds([&] {
                for (int32_t row = 0; row < n; ++row) {
                    v[static_cast<std::size_t>(j + 1)][static_cast<std::size_t>(row)] =
                        w[static_cast<std::size_t>(row)] / h_next;
                }
            });
        }

        if (basis_count <= 0) {
            result.stop_reason = "gmres_no_basis";
            return;
        }

        std::vector<double> normal(static_cast<std::size_t>(basis_count * basis_count), 0.0);
        std::vector<double> rhs_small(static_cast<std::size_t>(basis_count), 0.0);
        for (int32_t row = 0; row < basis_count; ++row) {
            rhs_small[static_cast<std::size_t>(row)] = beta * h[0][static_cast<std::size_t>(row)];
            for (int32_t col = 0; col < basis_count; ++col) {
                double sum = 0.0;
                for (int32_t k = 0; k <= basis_count; ++k) {
                    sum += h[static_cast<std::size_t>(k)][static_cast<std::size_t>(row)] *
                           h[static_cast<std::size_t>(k)][static_cast<std::size_t>(col)];
                }
                normal[static_cast<std::size_t>(row * basis_count + col)] = sum;
            }
        }

        std::vector<double> y;
        if (!solve_dense_linear_system(normal, rhs_small, basis_count, y)) {
            result.stop_reason = "gmres_least_squares_failed";
            return;
        }

        result.update_seconds += timed_seconds([&] {
            for (int32_t j = 0; j < basis_count; ++j) {
                for (int32_t row = 0; row < n; ++row) {
                    x[static_cast<std::size_t>(row)] +=
                        z[static_cast<std::size_t>(j)][static_cast<std::size_t>(row)] *
                        y[static_cast<std::size_t>(j)];
                }
            }
        });
        if (result.stop_reason.empty()) {
            result.stop_reason = "gmres_fixed_iter_cpu_block_ilu0_pilot";
        }
    });
    return x;
}

}  // namespace detail

inline CpuBlockIlu0Result solve(const CsrMatrix& matrix,
                                const std::vector<double>& rhs,
                                const CpuBlockIlu0Options& options)
{
    CpuBlockIlu0Result result;
    if (matrix.rows <= 0 || matrix.rows != matrix.cols ||
        static_cast<int32_t>(rhs.size()) != matrix.rows) {
        result.factor_failed = true;
        result.stop_reason = "invalid_cpu_block_ilu0_input";
        return result;
    }

    const MetisPermutation permutation =
        options.use_block_coloring_order
            ? detail::build_colored_block_permutation(matrix, options.block_size)
            : build_metis_permutation(matrix, options.block_size);
    const CsrMatrix a_perm = detail::build_permuted_matrix(matrix, permutation);
    const std::vector<double> b_perm = detail::permute_vector(rhs, permutation);

    detail::CpuPreconditioner preconditioner;
    preconditioner.ilu0 = options.use_block_ilu0;
    result.factor_failed = !preconditioner.setup(a_perm,
                                                 permutation.block_starts,
                                                 permutation.block_sizes,
                                                 options.diag_shift_scale,
                                                 result.setup_seconds,
                                                 result.factor_seconds);
    result.num_blocks = static_cast<int32_t>(permutation.block_sizes.size());
    result.block_nnz = static_cast<int32_t>(preconditioner.pattern.blocks.size());
    detail::compute_levels(preconditioner.pattern, result);

    if (!result.factor_failed) {
        const std::vector<double> x_perm =
            options.use_gmres
                ? detail::gmres_fixed(a_perm,
                                      b_perm,
                                      options.gmres_iters,
                                      preconditioner,
                                      result)
                : detail::bicgstab_fixed(a_perm,
                                         b_perm,
                                         options.bicgstab_iters,
                                         preconditioner,
                                         result);
        result.solution = detail::unpermute_vector(x_perm, permutation);
        result.apply_seconds = preconditioner.apply_seconds;
        result.forward_seconds = preconditioner.forward_seconds;
        result.backward_seconds = preconditioner.backward_seconds;

        std::vector<double> residual = rhs;
        const std::vector<double> ax = detail::spmv(matrix, result.solution);
        for (std::size_t i = 0; i < residual.size(); ++i) {
            residual[i] -= ax[i];
        }
        result.residual_norm2 = detail::norm2(residual);
        result.relative_residual_norm2 =
            result.residual_norm2 /
            std::max(detail::norm2(rhs), std::numeric_limits<double>::min());
    } else {
        result.solution.assign(rhs.size(), 0.0);
        result.stop_reason = "cpu_block_ilu0_factor_failed";
    }
    return result;
}

}  // namespace cuiter::cpu_pilot
