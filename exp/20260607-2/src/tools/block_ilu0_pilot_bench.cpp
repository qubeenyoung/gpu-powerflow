#include "cuiter/common/cuda_utils.hpp"
#include "cuiter/core/csr_matrix.hpp"
#include "cuiter/reorder/metis_partition.hpp"
#include "cupf_minimal/case_data.hpp"
#include "cupf_minimal/direct_cudss_solver.hpp"
#include "cupf_minimal/jacobian_analysis.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path jf_root = "raw/cupf_jf_dumps";
    std::filesystem::path case_root =
        "/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps";
    std::filesystem::path output_dir = "results";
    std::vector<std::string> cases = {
        "case2383wp",
        "case3120sp",
        "case9241pegase",
        "case13659pegase",
        "case6468rte",
    };
    std::vector<int32_t> block_sizes = {16, 32};
    int32_t bicgstab_iters = 2;
    int32_t iteration = 1;
    double diag_shift_scale = 1.0e-8;
    bool allow_missing = false;
};

struct LinearMetadata {
    int32_t n_pvpq = 0;
    int32_t n_pq = 0;
    std::vector<int32_t> index_field;
};

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

struct PreconditionerTimings {
    double setup_ms = 0.0;
    double factor_ms = 0.0;
    double apply_ms = 0.0;
    double forward_ms = 0.0;
    double backward_ms = 0.0;
};

struct BicgstabTimings {
    double total_ms = 0.0;
    double preconditioner_apply_ms = 0.0;
    double spmv_ms = 0.0;
    double dot_ms = 0.0;
    double update_ms = 0.0;
};

struct LevelStats {
    int32_t l_levels = 0;
    int32_t u_levels = 0;
    double avg_level_width = 0.0;
    int32_t max_level_width = 0;
};

struct SolveRow {
    std::string case_name;
    std::string preconditioner;
    int32_t block_size = 0;
    std::string ordering;
    int32_t n = 0;
    int32_t nnz = 0;
    int32_t num_blocks = 0;
    int32_t block_nnz = 0;
    int32_t l_levels = 0;
    int32_t u_levels = 0;
    double avg_level_width = 0.0;
    int32_t max_level_width = 0;
    bool factor_failed = false;
    std::string stop_reason;
    int32_t bicgstab_iters = 0;
    double true_linear_abs_res = 0.0;
    double true_linear_rel_res = 0.0;
    double dx_norm_ratio_vs_cudss = 0.0;
    double dx_cosine_vs_cudss = 0.0;
    double theta_norm_ratio = 0.0;
    double theta_cosine = 0.0;
    double vmag_norm_ratio = 0.0;
    double vmag_cosine = 0.0;
    double preconditioner_setup_ms = 0.0;
    double preconditioner_apply_ms = 0.0;
    double block_ilu_factor_ms = 0.0;
    double block_ilu_forward_ms = 0.0;
    double block_ilu_backward_ms = 0.0;
    double bicgstab_total_ms = 0.0;
    double spmv_ms = 0.0;
    double dot_ms = 0.0;
    double update_ms = 0.0;
};

template <typename Fn>
double host_timed_ms(Fn&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

std::vector<std::string> split_list(const std::string& text)
{
    std::vector<std::string> items;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

std::vector<int32_t> parse_int_list(const std::string& text)
{
    std::vector<int32_t> values;
    for (const std::string& item : split_list(text)) {
        values.push_back(std::stoi(item));
    }
    return values;
}

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --jf-root PATH\n"
        << "  --case-root PATH\n"
        << "  --output-dir PATH\n"
        << "  --cases a,b,c\n"
        << "  --block-sizes 16,32\n"
        << "  --bicgstab-iters 2\n"
        << "  --iteration 1\n"
        << "  --block-ilu-diag-shift-scale FLOAT\n"
        << "  --allow-missing\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--jf-root" && i + 1 < argc) {
            options.jf_root = argv[++i];
        } else if (arg == "--case-root" && i + 1 < argc) {
            options.case_root = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            options.output_dir = argv[++i];
        } else if (arg == "--cases" && i + 1 < argc) {
            options.cases = split_list(argv[++i]);
        } else if (arg == "--block-sizes" && i + 1 < argc) {
            options.block_sizes = parse_int_list(argv[++i]);
        } else if (arg == "--bicgstab-iters" && i + 1 < argc) {
            options.bicgstab_iters = std::stoi(argv[++i]);
        } else if (arg == "--iteration" && i + 1 < argc) {
            options.iteration = std::stoi(argv[++i]);
        } else if (arg == "--block-ilu-diag-shift-scale" && i + 1 < argc) {
            options.diag_shift_scale = std::stod(argv[++i]);
        } else if (arg == "--allow-missing") {
            options.allow_missing = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.bicgstab_iters <= 0) {
        throw std::runtime_error("--bicgstab-iters must be positive");
    }
    return options;
}

void expect_token(std::istream& in,
                  const std::string& expected,
                  const std::filesystem::path& path)
{
    std::string token;
    if (!(in >> token) || token != expected) {
        throw std::runtime_error("expected token '" + expected + "' in " + path.string());
    }
}

cuiter::CsrMatrix load_cupf_csr_dump(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open matrix file: " + path.string());
    }

    expect_token(in, "type", path);
    std::string type;
    in >> type;
    if (type != "csr_matrix") {
        throw std::runtime_error("matrix file is not a cuPF csr_matrix dump: " + path.string());
    }

    cuiter::CsrMatrix matrix;
    int32_t nnz = 0;
    expect_token(in, "rows", path);
    in >> matrix.rows;
    expect_token(in, "cols", path);
    in >> matrix.cols;
    expect_token(in, "nnz", path);
    in >> nnz;
    if (matrix.rows <= 0 || matrix.rows != matrix.cols || nnz <= 0) {
        throw std::runtime_error("invalid CSR dimensions in " + path.string());
    }

    matrix.row_ptr.resize(static_cast<std::size_t>(matrix.rows + 1));
    matrix.col_idx.resize(static_cast<std::size_t>(nnz));
    matrix.values.resize(static_cast<std::size_t>(nnz));
    expect_token(in, "row_ptr", path);
    for (int32_t i = 0; i <= matrix.rows; ++i) {
        in >> matrix.row_ptr[static_cast<std::size_t>(i)];
    }
    expect_token(in, "col_idx", path);
    for (int32_t i = 0; i < nnz; ++i) {
        in >> matrix.col_idx[static_cast<std::size_t>(i)];
    }
    expect_token(in, "values", path);
    for (int32_t i = 0; i < nnz; ++i) {
        in >> matrix.values[static_cast<std::size_t>(i)];
    }
    if (!in || matrix.row_ptr.front() != 0 || matrix.row_ptr.back() != nnz) {
        throw std::runtime_error("malformed CSR dump: " + path.string());
    }
    return matrix;
}

std::vector<double> load_cupf_vector_dump(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open vector file: " + path.string());
    }
    expect_token(in, "type", path);
    std::string type;
    in >> type;
    if (type != "vector") {
        throw std::runtime_error("vector file is not a cuPF vector dump: " + path.string());
    }
    expect_token(in, "size", path);
    int32_t n = 0;
    in >> n;
    expect_token(in, "values", path);
    std::vector<double> values(static_cast<std::size_t>(n), 0.0);
    for (int32_t i = 0; i < n; ++i) {
        int32_t index = 0;
        double value = 0.0;
        in >> index >> value;
        if (!in || index < 0 || index >= n) {
            throw std::runtime_error("malformed vector dump: " + path.string());
        }
        values[static_cast<std::size_t>(index)] = value;
    }
    return values;
}

std::filesystem::path jacobian_path(const std::filesystem::path& jf_root,
                                    const std::string& case_name,
                                    int32_t iteration)
{
    const auto case_dir = jf_root / case_name;
    const auto direct = case_dir / ("J" + std::to_string(iteration) + ".txt");
    if (std::filesystem::exists(direct)) {
        return direct;
    }
    return case_dir / "repeat_00" / ("jacobian_iter" + std::to_string(iteration) + ".txt");
}

std::filesystem::path rhs_path(const std::filesystem::path& jf_root,
                               const std::string& case_name,
                               int32_t iteration)
{
    const auto case_dir = jf_root / case_name;
    const std::vector<std::filesystem::path> candidates = {
        case_dir / ("F" + std::to_string(iteration) + ".txt"),
        case_dir / "repeat_00" / ("residual_iter" + std::to_string(iteration) + ".txt"),
        case_dir / "repeat_00" / ("residual_before_update_iter" + std::to_string(iteration) + ".txt"),
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return candidates.front();
}

uint64_t edge_key(int32_t row, int32_t col)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(col));
}

int32_t edge_key_row(uint64_t key)
{
    return static_cast<int32_t>(static_cast<uint32_t>(key >> 32));
}

int32_t edge_key_col(uint64_t key)
{
    return static_cast<int32_t>(static_cast<uint32_t>(key & 0xffffffffu));
}

double norm2(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values) {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
}

double dot(const std::vector<double>& lhs, const std::vector<double>& rhs)
{
    long double sum = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        sum += static_cast<long double>(lhs[i]) * static_cast<long double>(rhs[i]);
    }
    return static_cast<double>(sum);
}

double ratio(double num, double den)
{
    return den > 0.0 ? num / den : 0.0;
}

std::string format_double(double value)
{
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }
    std::ostringstream out;
    out << std::setprecision(12) << value;
    return out.str();
}

void sort_unique(std::vector<int32_t>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

std::vector<int32_t> block_of_new_indices(int32_t n,
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

std::vector<std::vector<int32_t>> build_block_undirected_graph(
    const cuiter::CsrMatrix& matrix,
    const cuiter::MetisPermutation& permutation,
    const cuiter::PermutedCsrPattern& pattern)
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

std::vector<int32_t> greedy_coloring_order(const std::vector<std::vector<int32_t>>& adjacency)
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

cuiter::MetisPermutation build_colored_block_permutation(const cuiter::CsrMatrix& matrix,
                                                         int32_t block_size)
{
    const cuiter::MetisPermutation base = cuiter::build_metis_permutation(matrix, block_size);
    const cuiter::PermutedCsrPattern base_pattern =
        cuiter::build_permuted_csr_pattern(matrix, base);
    const std::vector<std::vector<int32_t>> adjacency =
        build_block_undirected_graph(matrix, base, base_pattern);
    const std::vector<int32_t> new_block_to_old = greedy_coloring_order(adjacency);

    cuiter::MetisPermutation permutation;
    permutation.new_to_old.reserve(base.new_to_old.size());
    permutation.old_to_new.assign(base.old_to_new.size(), -1);
    permutation.block_starts.reserve(base.block_starts.size());
    permutation.block_sizes.reserve(base.block_sizes.size());
    for (int32_t old_block : new_block_to_old) {
        const int32_t begin = base.block_starts[static_cast<std::size_t>(old_block)];
        const int32_t dim = base.block_sizes[static_cast<std::size_t>(old_block)];
        permutation.block_starts.push_back(static_cast<int32_t>(permutation.new_to_old.size()));
        permutation.block_sizes.push_back(dim);
        for (int32_t offset = 0; offset < dim; ++offset) {
            const int32_t old_index =
                base.new_to_old[static_cast<std::size_t>(begin + offset)];
            const int32_t new_index = static_cast<int32_t>(permutation.new_to_old.size());
            permutation.new_to_old.push_back(old_index);
            permutation.old_to_new[static_cast<std::size_t>(old_index)] = new_index;
        }
    }
    permutation.stats = cuiter::compute_partition_stats(matrix,
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

cuiter::CsrMatrix build_permuted_matrix(const cuiter::CsrMatrix& matrix,
                                        const cuiter::MetisPermutation& permutation)
{
    const cuiter::PermutedCsrPattern pattern =
        cuiter::build_permuted_csr_pattern(matrix, permutation);
    cuiter::CsrMatrix out;
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

std::vector<double> permute_vector(const std::vector<double>& values,
                                   const cuiter::MetisPermutation& permutation)
{
    std::vector<double> out(values.size(), 0.0);
    for (std::size_t new_index = 0; new_index < permutation.new_to_old.size(); ++new_index) {
        out[new_index] = values[static_cast<std::size_t>(permutation.new_to_old[new_index])];
    }
    return out;
}

std::vector<double> unpermute_vector(const std::vector<double>& values,
                                     const cuiter::MetisPermutation& permutation)
{
    std::vector<double> out(values.size(), 0.0);
    for (std::size_t new_index = 0; new_index < permutation.new_to_old.size(); ++new_index) {
        out[static_cast<std::size_t>(permutation.new_to_old[new_index])] = values[new_index];
    }
    return out;
}

std::vector<double> spmv(const cuiter::CsrMatrix& matrix, const std::vector<double>& x)
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

BlockPattern make_block_pattern(const cuiter::CsrMatrix& matrix,
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
            if (diagonal_only && rb != cb) {
                continue;
            }
            keys.insert(edge_key(rb, cb));
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
    pattern.blocks.reserve(sorted_keys.size());
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

void scatter_dense_blocks(const cuiter::CsrMatrix& matrix, BlockPattern& pattern)
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

int32_t find_block(const BlockPattern& pattern, int32_t row, int32_t col)
{
    const auto it = pattern.block_index.find(edge_key(row, col));
    return it == pattern.block_index.end() ? -1 : it->second;
}

bool invert_dense_block(std::vector<float>& matrix,
                        int32_t n,
                        double shift_scale,
                        std::vector<float>& inverse,
                        double& applied_shift)
{
    double mean_abs_diag = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        mean_abs_diag += std::abs(static_cast<double>(matrix[static_cast<std::size_t>(i * n + i)]));
    }
    mean_abs_diag /= static_cast<double>(std::max(1, n));
    const std::vector<double> shifts = {
        0.0,
        shift_scale * std::max(1.0, mean_abs_diag),
        1.0e-6 * std::max(1.0, mean_abs_diag),
        1.0e-4 * std::max(1.0, mean_abs_diag),
    };

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
                const double value_abs = std::abs(a[static_cast<std::size_t>(row * n + col)]);
                if (value_abs > pivot_abs) {
                    pivot_abs = value_abs;
                    pivot = row;
                }
            }
            if (pivot_abs < 1.0e-12 || !std::isfinite(pivot_abs)) {
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
                if (factor == 0.0) {
                    continue;
                }
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
        applied_shift = shift;
        return true;
    }
    return false;
}

void right_multiply_in_place(DenseBlock& lhs, const std::vector<float>& rhs_inv)
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

void subtract_product(DenseBlock& target, const DenseBlock& lhs, const DenseBlock& rhs)
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

std::vector<double> dense_gemv(const DenseBlock& block, const std::vector<double>& x, int32_t x_offset)
{
    std::vector<double> y(static_cast<std::size_t>(block.rows), 0.0);
    for (int32_t i = 0; i < block.rows; ++i) {
        double sum = 0.0;
        for (int32_t j = 0; j < block.cols; ++j) {
            sum += static_cast<double>(block.values[static_cast<std::size_t>(i * block.cols + j)]) *
                   x[static_cast<std::size_t>(x_offset + j)];
        }
        y[static_cast<std::size_t>(i)] = sum;
    }
    return y;
}

class CpuBlockJacobiPreconditioner {
public:
    bool setup(const cuiter::CsrMatrix& matrix,
               const std::vector<int32_t>& block_starts,
               const std::vector<int32_t>& block_dims,
               double diag_shift_scale,
               PreconditionerTimings& timings)
    {
        bool ok = true;
        timings.setup_ms += host_timed_ms([&] {
            pattern_ = make_block_pattern(matrix, block_starts, block_dims, true);
            scatter_dense_blocks(matrix, pattern_);
            inverse_diag_.assign(pattern_.block_dims.size(), {});
            for (int32_t block = 0; block < pattern_.num_blocks; ++block) {
                DenseBlock& diag =
                    pattern_.blocks[static_cast<std::size_t>(
                        pattern_.diagonal_index[static_cast<std::size_t>(block)])];
                double shift = 0.0;
                if (!invert_dense_block(diag.values,
                                        diag.rows,
                                        diag_shift_scale,
                                        inverse_diag_[static_cast<std::size_t>(block)],
                                        shift)) {
                    ok = false;
                    break;
                }
            }
        });
        return ok;
    }

    std::vector<double> apply(const std::vector<double>& r, PreconditionerTimings& timings) const
    {
        std::vector<double> z(r.size(), 0.0);
        timings.apply_ms += host_timed_ms([&] {
            for (int32_t block = 0; block < pattern_.num_blocks; ++block) {
                const int32_t begin = pattern_.block_starts[static_cast<std::size_t>(block)];
                const int32_t dim = pattern_.block_dims[static_cast<std::size_t>(block)];
                const std::vector<float>& inv = inverse_diag_[static_cast<std::size_t>(block)];
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

private:
    BlockPattern pattern_;
    std::vector<std::vector<float>> inverse_diag_;
};

class CpuBlockIlu0Preconditioner {
public:
    bool setup(const cuiter::CsrMatrix& matrix,
               const std::vector<int32_t>& block_starts,
               const std::vector<int32_t>& block_dims,
               double diag_shift_scale,
               PreconditionerTimings& timings)
    {
        bool ok = true;
        timings.setup_ms += host_timed_ms([&] {
            pattern_ = make_block_pattern(matrix, block_starts, block_dims, false);
            scatter_dense_blocks(matrix, pattern_);
            inverse_diag_.assign(pattern_.block_dims.size(), {});
        });
        timings.factor_ms += host_timed_ms([&] {
            for (int32_t i = 0; i < pattern_.num_blocks; ++i) {
                std::vector<int32_t> row_blocks = pattern_.row_block_indices[static_cast<std::size_t>(i)];
                std::sort(row_blocks.begin(), row_blocks.end(), [&](int32_t lhs, int32_t rhs) {
                    return pattern_.blocks[static_cast<std::size_t>(lhs)].col <
                           pattern_.blocks[static_cast<std::size_t>(rhs)].col;
                });
                for (int32_t lower_index : row_blocks) {
                    DenseBlock& lik = pattern_.blocks[static_cast<std::size_t>(lower_index)];
                    const int32_t k = lik.col;
                    if (k >= i) {
                        continue;
                    }
                    right_multiply_in_place(lik, inverse_diag_[static_cast<std::size_t>(k)]);
                    for (int32_t ukj_index : pattern_.row_block_indices[static_cast<std::size_t>(k)]) {
                        const DenseBlock& ukj = pattern_.blocks[static_cast<std::size_t>(ukj_index)];
                        const int32_t j = ukj.col;
                        if (j <= k) {
                            continue;
                        }
                        const int32_t target_index = find_block(pattern_, i, j);
                        if (target_index < 0) {
                            continue;
                        }
                        DenseBlock& target =
                            pattern_.blocks[static_cast<std::size_t>(target_index)];
                        subtract_product(target, lik, ukj);
                    }
                }
                const int32_t diag_index = pattern_.diagonal_index[static_cast<std::size_t>(i)];
                DenseBlock& diag = pattern_.blocks[static_cast<std::size_t>(diag_index)];
                double shift = 0.0;
                if (!invert_dense_block(diag.values,
                                        diag.rows,
                                        diag_shift_scale,
                                        inverse_diag_[static_cast<std::size_t>(i)],
                                        shift)) {
                    ok = false;
                    break;
                }
            }
        });
        return ok;
    }

    std::vector<double> apply(const std::vector<double>& r, PreconditionerTimings& timings) const
    {
        std::vector<double> y(r.size(), 0.0);
        std::vector<double> z(r.size(), 0.0);
        const double forward_ms = host_timed_ms([&] {
            for (int32_t block = 0; block < pattern_.num_blocks; ++block) {
                const int32_t begin = pattern_.block_starts[static_cast<std::size_t>(block)];
                const int32_t dim = pattern_.block_dims[static_cast<std::size_t>(block)];
                for (int32_t i = 0; i < dim; ++i) {
                    y[static_cast<std::size_t>(begin + i)] =
                        r[static_cast<std::size_t>(begin + i)];
                }
                for (int32_t block_index : pattern_.row_block_indices[static_cast<std::size_t>(block)]) {
                    const DenseBlock& offdiag = pattern_.blocks[static_cast<std::size_t>(block_index)];
                    if (offdiag.col >= block) {
                        continue;
                    }
                    const int32_t col_begin =
                        pattern_.block_starts[static_cast<std::size_t>(offdiag.col)];
                    const std::vector<double> product = dense_gemv(offdiag, y, col_begin);
                    for (int32_t i = 0; i < dim; ++i) {
                        y[static_cast<std::size_t>(begin + i)] -=
                            product[static_cast<std::size_t>(i)];
                    }
                }
            }
        });
        const double backward_ms = host_timed_ms([&] {
            for (int32_t block = pattern_.num_blocks - 1; block >= 0; --block) {
                const int32_t begin = pattern_.block_starts[static_cast<std::size_t>(block)];
                const int32_t dim = pattern_.block_dims[static_cast<std::size_t>(block)];
                std::vector<double> rhs(static_cast<std::size_t>(dim), 0.0);
                for (int32_t i = 0; i < dim; ++i) {
                    rhs[static_cast<std::size_t>(i)] = y[static_cast<std::size_t>(begin + i)];
                }
                for (int32_t block_index : pattern_.row_block_indices[static_cast<std::size_t>(block)]) {
                    const DenseBlock& offdiag = pattern_.blocks[static_cast<std::size_t>(block_index)];
                    if (offdiag.col <= block) {
                        continue;
                    }
                    const int32_t col_begin =
                        pattern_.block_starts[static_cast<std::size_t>(offdiag.col)];
                    const std::vector<double> product = dense_gemv(offdiag, z, col_begin);
                    for (int32_t i = 0; i < dim; ++i) {
                        rhs[static_cast<std::size_t>(i)] -= product[static_cast<std::size_t>(i)];
                    }
                }
                const std::vector<float>& inv = inverse_diag_[static_cast<std::size_t>(block)];
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
        timings.forward_ms += forward_ms;
        timings.backward_ms += backward_ms;
        timings.apply_ms += forward_ms + backward_ms;
        return z;
    }

private:
    BlockPattern pattern_;
    std::vector<std::vector<float>> inverse_diag_;
};

template <typename ApplyFn>
std::vector<double> bicgstab_fixed(const cuiter::CsrMatrix& matrix,
                                   const std::vector<double>& rhs,
                                   int32_t max_iters,
                                   ApplyFn&& apply,
                                   BicgstabTimings& timings,
                                   std::string& stop_reason)
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
    timings.total_ms += host_timed_ms([&] {
        for (int32_t iter = 0; iter < max_iters; ++iter) {
            double rho = 0.0;
            timings.dot_ms += host_timed_ms([&] {
                rho = dot(r_hat, r);
            });
            if (!std::isfinite(rho) || std::abs(rho) <= std::numeric_limits<double>::min()) {
                stop_reason = "bicgstab_rho_breakdown";
                return;
            }
            timings.update_ms += host_timed_ms([&] {
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
            std::vector<double> p_hat;
            timings.preconditioner_apply_ms += host_timed_ms([&] {
                p_hat = apply(p);
            });
            timings.spmv_ms += host_timed_ms([&] {
                v = spmv(matrix, p_hat);
            });
            double rhat_v = 0.0;
            timings.dot_ms += host_timed_ms([&] {
                rhat_v = dot(r_hat, v);
            });
            if (!std::isfinite(rhat_v) || std::abs(rhat_v) <= std::numeric_limits<double>::min()) {
                stop_reason = "bicgstab_alpha_breakdown";
                return;
            }
            alpha = rho / rhat_v;
            std::vector<double> s(static_cast<std::size_t>(n), 0.0);
            timings.update_ms += host_timed_ms([&] {
                for (int32_t i = 0; i < n; ++i) {
                    s[static_cast<std::size_t>(i)] =
                        r[static_cast<std::size_t>(i)] -
                        alpha * v[static_cast<std::size_t>(i)];
                }
            });
            std::vector<double> s_hat;
            timings.preconditioner_apply_ms += host_timed_ms([&] {
                s_hat = apply(s);
            });
            std::vector<double> t;
            timings.spmv_ms += host_timed_ms([&] {
                t = spmv(matrix, s_hat);
            });
            double ts = 0.0;
            double tt = 0.0;
            timings.dot_ms += host_timed_ms([&] {
                ts = dot(t, s);
                tt = dot(t, t);
            });
            if (!std::isfinite(ts) || !std::isfinite(tt) ||
                tt <= std::numeric_limits<double>::min()) {
                stop_reason = "bicgstab_omega_breakdown";
                return;
            }
            omega = ts / tt;
            timings.update_ms += host_timed_ms([&] {
                for (int32_t i = 0; i < n; ++i) {
                    x[static_cast<std::size_t>(i)] +=
                        alpha * p_hat[static_cast<std::size_t>(i)] +
                        omega * s_hat[static_cast<std::size_t>(i)];
                    r[static_cast<std::size_t>(i)] =
                        s[static_cast<std::size_t>(i)] -
                        omega * t[static_cast<std::size_t>(i)];
                }
            });
            rho_old = rho;
            if (!std::isfinite(omega) || std::abs(omega) <= std::numeric_limits<double>::min()) {
                stop_reason = "bicgstab_omega_zero";
                return;
            }
        }
        stop_reason = "bicgstab_fixed_iter";
    });
    return x;
}

std::vector<double> solve_cudss_dx(const cuiter::CsrMatrix& matrix,
                                   const std::vector<double>& rhs,
                                   double& factor_solve_ms)
{
    cuiter::DeviceBuffer<int32_t> d_row_ptr;
    cuiter::DeviceBuffer<int32_t> d_col_idx;
    cuiter::DeviceBuffer<double> d_values;
    cuiter::DeviceBuffer<double> d_rhs;
    cuiter::DeviceBuffer<double> d_x;
    d_row_ptr.assign(matrix.row_ptr.data(), matrix.row_ptr.size());
    d_col_idx.assign(matrix.col_idx.data(), matrix.col_idx.size());
    d_values.assign(matrix.values.data(), matrix.values.size());
    d_rhs.assign(rhs.data(), rhs.size());
    d_x.resize(rhs.size());
    d_x.memset_zero();

    cupf_minimal::DirectCudssSolver solver;
    solver.initialize(matrix,
                      d_row_ptr.data(),
                      d_col_idx.data(),
                      d_values.data(),
                      d_rhs.data(),
                      d_x.data());
    solver.analyze();
    const double factor_ms = 1000.0 * solver.factorize();
    const double solve_ms = 1000.0 * solver.solve();
    factor_solve_ms = factor_ms + solve_ms;
    std::vector<double> dx(rhs.size(), 0.0);
    d_x.copy_to(dx.data(), dx.size());
    return dx;
}

LinearMetadata make_linear_metadata(const std::filesystem::path& case_dir)
{
    const cupf_minimal::DumpCaseData case_data = cupf_minimal::load_dump_case(case_dir);
    const cupf_minimal::JacobianIndexing indexing =
        cupf_minimal::make_jacobian_indexing(case_data.rows,
                                             case_data.pv.data(),
                                             static_cast<int32_t>(case_data.pv.size()),
                                             case_data.pq.data(),
                                             static_cast<int32_t>(case_data.pq.size()));
    LinearMetadata metadata;
    metadata.n_pvpq = indexing.n_pvpq;
    metadata.n_pq = indexing.n_pq;
    const int32_t n = metadata.n_pvpq + metadata.n_pq;
    metadata.index_field.assign(static_cast<std::size_t>(n), -1);
    for (int32_t i = 0; i < metadata.n_pvpq; ++i) {
        metadata.index_field[static_cast<std::size_t>(i)] = 0;
    }
    for (int32_t i = 0; i < metadata.n_pq; ++i) {
        metadata.index_field[static_cast<std::size_t>(metadata.n_pvpq + i)] = 1;
    }
    return metadata;
}

void attach_quality_metrics(SolveRow& row,
                            const std::vector<double>& x,
                            const std::vector<double>& cudss_x,
                            const LinearMetadata& metadata)
{
    const double x_norm = norm2(x);
    const double c_norm = std::max(norm2(cudss_x), std::numeric_limits<double>::min());
    row.dx_norm_ratio_vs_cudss = x_norm / c_norm;
    row.dx_cosine_vs_cudss =
        dot(x, cudss_x) / std::max(x_norm * c_norm, std::numeric_limits<double>::min());

    std::vector<double> x_theta;
    std::vector<double> c_theta;
    std::vector<double> x_vmag;
    std::vector<double> c_vmag;
    x_theta.reserve(static_cast<std::size_t>(metadata.n_pvpq));
    c_theta.reserve(static_cast<std::size_t>(metadata.n_pvpq));
    x_vmag.reserve(static_cast<std::size_t>(metadata.n_pq));
    c_vmag.reserve(static_cast<std::size_t>(metadata.n_pq));
    for (std::size_t i = 0; i < metadata.index_field.size(); ++i) {
        if (metadata.index_field[i] == 0) {
            x_theta.push_back(x[i]);
            c_theta.push_back(cudss_x[i]);
        } else if (metadata.index_field[i] == 1) {
            x_vmag.push_back(x[i]);
            c_vmag.push_back(cudss_x[i]);
        }
    }
    const double xt = norm2(x_theta);
    const double ct = std::max(norm2(c_theta), std::numeric_limits<double>::min());
    const double xv = norm2(x_vmag);
    const double cv = std::max(norm2(c_vmag), std::numeric_limits<double>::min());
    row.theta_norm_ratio = xt / ct;
    row.theta_cosine = dot(x_theta, c_theta) / std::max(xt * ct, std::numeric_limits<double>::min());
    row.vmag_norm_ratio = xv / cv;
    row.vmag_cosine = dot(x_vmag, c_vmag) / std::max(xv * cv, std::numeric_limits<double>::min());
}

LevelStats compute_level_stats(const BlockPattern& pattern)
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
    auto summarize_width = [](const std::vector<int32_t>& levels, int32_t& max_width) {
        const int32_t num_levels = *std::max_element(levels.begin(), levels.end()) + 1;
        std::vector<int32_t> widths(static_cast<std::size_t>(num_levels), 0);
        for (int32_t level : levels) {
            ++widths[static_cast<std::size_t>(level)];
        }
        max_width = *std::max_element(widths.begin(), widths.end());
        return static_cast<double>(levels.size()) / static_cast<double>(num_levels);
    };
    LevelStats stats;
    stats.l_levels = *std::max_element(l_level.begin(), l_level.end()) + 1;
    stats.u_levels = *std::max_element(u_level.begin(), u_level.end()) + 1;
    int32_t l_max = 0;
    int32_t u_max = 0;
    const double l_avg = summarize_width(l_level, l_max);
    const double u_avg = summarize_width(u_level, u_max);
    stats.avg_level_width = 0.5 * (l_avg + u_avg);
    stats.max_level_width = std::max(l_max, u_max);
    return stats;
}

SolveRow solve_one(const std::string& case_name,
                   const cuiter::CsrMatrix& matrix,
                   const std::vector<double>& rhs,
                   const LinearMetadata& metadata,
                   const std::vector<double>& cudss_x,
                   int32_t block_size,
                   bool use_ilu,
                   const CliOptions& options)
{
    SolveRow row;
    row.case_name = case_name;
    row.preconditioner = use_ilu ? "block_ilu0" : "block_jacobi";
    row.block_size = block_size;
    row.ordering = use_ilu ? "block_coloring" : "current_metis_block_order";
    row.n = matrix.rows;
    row.nnz = matrix.nnz();
    const cuiter::MetisPermutation permutation =
        use_ilu ? build_colored_block_permutation(matrix, block_size)
                : cuiter::build_metis_permutation(matrix, block_size);
    const cuiter::CsrMatrix a_perm = build_permuted_matrix(matrix, permutation);
    const std::vector<double> b_perm = permute_vector(rhs, permutation);
    row.num_blocks = static_cast<int32_t>(permutation.block_sizes.size());

    PreconditionerTimings precond_timings;
    BicgstabTimings bicgstab_timings;
    std::vector<double> x_perm;
    if (use_ilu) {
        CpuBlockIlu0Preconditioner preconditioner;
        row.factor_failed = !preconditioner.setup(a_perm,
                                                  permutation.block_starts,
                                                  permutation.block_sizes,
                                                  options.diag_shift_scale,
                                                  precond_timings);
        BlockPattern stats_pattern =
            make_block_pattern(a_perm, permutation.block_starts, permutation.block_sizes, false);
        row.block_nnz = static_cast<int32_t>(stats_pattern.blocks.size());
        const LevelStats levels = compute_level_stats(stats_pattern);
        row.l_levels = levels.l_levels;
        row.u_levels = levels.u_levels;
        row.avg_level_width = levels.avg_level_width;
        row.max_level_width = levels.max_level_width;
        if (!row.factor_failed) {
            x_perm = bicgstab_fixed(a_perm,
                                    b_perm,
                                    options.bicgstab_iters,
                                    [&](const std::vector<double>& r) {
                                        return preconditioner.apply(r, precond_timings);
                                    },
                                    bicgstab_timings,
                                    row.stop_reason);
        } else {
            x_perm.assign(static_cast<std::size_t>(matrix.rows), 0.0);
            row.stop_reason = "block_ilu_factor_failed";
        }
    } else {
        CpuBlockJacobiPreconditioner preconditioner;
        row.factor_failed = !preconditioner.setup(a_perm,
                                                  permutation.block_starts,
                                                  permutation.block_sizes,
                                                  options.diag_shift_scale,
                                                  precond_timings);
        BlockPattern stats_pattern =
            make_block_pattern(a_perm, permutation.block_starts, permutation.block_sizes, true);
        row.block_nnz = static_cast<int32_t>(stats_pattern.blocks.size());
        row.l_levels = 1;
        row.u_levels = 1;
        row.avg_level_width = row.num_blocks;
        row.max_level_width = row.num_blocks;
        if (!row.factor_failed) {
            x_perm = bicgstab_fixed(a_perm,
                                    b_perm,
                                    options.bicgstab_iters,
                                    [&](const std::vector<double>& r) {
                                        return preconditioner.apply(r, precond_timings);
                                    },
                                    bicgstab_timings,
                                    row.stop_reason);
        } else {
            x_perm.assign(static_cast<std::size_t>(matrix.rows), 0.0);
            row.stop_reason = "block_jacobi_setup_failed";
        }
    }

    const std::vector<double> x = unpermute_vector(x_perm, permutation);
    std::vector<double> residual = rhs;
    const std::vector<double> ax = spmv(matrix, x);
    for (std::size_t i = 0; i < residual.size(); ++i) {
        residual[i] -= ax[i];
    }
    row.true_linear_abs_res = norm2(residual);
    row.true_linear_rel_res = row.true_linear_abs_res /
                              std::max(norm2(rhs), std::numeric_limits<double>::min());
    row.bicgstab_iters = options.bicgstab_iters;
    row.preconditioner_setup_ms = precond_timings.setup_ms;
    row.preconditioner_apply_ms = precond_timings.apply_ms;
    row.block_ilu_factor_ms = precond_timings.factor_ms;
    row.block_ilu_forward_ms = precond_timings.forward_ms;
    row.block_ilu_backward_ms = precond_timings.backward_ms;
    row.bicgstab_total_ms = bicgstab_timings.total_ms;
    row.spmv_ms = bicgstab_timings.spmv_ms;
    row.dot_ms = bicgstab_timings.dot_ms;
    row.update_ms = bicgstab_timings.update_ms;
    attach_quality_metrics(row, x, cudss_x, metadata);
    return row;
}

void write_standalone_csv(const std::filesystem::path& path, const std::vector<SolveRow>& rows)
{
    std::ofstream out(path);
    out << "case,preconditioner,block_size,ordering,n,nnz,num_blocks,block_nnz,L_levels,U_levels,"
           "avg_level_width,max_level_width,factor_failed,stop_reason,bicgstab_iters,"
           "true_linear_rel_res,true_linear_abs_res,dx_norm_ratio_vs_cudss,dx_cosine_vs_cudss,"
           "theta_norm_ratio,theta_cosine,vmag_norm_ratio,vmag_cosine,preconditioner_setup_ms,"
           "preconditioner_apply_ms,block_ilu_factor_ms,block_ilu_forward_ms,"
           "block_ilu_backward_ms,bicgstab_total_ms,spmv_ms,dot_ms,update_ms\n";
    for (const SolveRow& row : rows) {
        out << row.case_name << ',' << row.preconditioner << ',' << row.block_size << ','
            << row.ordering << ',' << row.n << ',' << row.nnz << ',' << row.num_blocks << ','
            << row.block_nnz << ',' << row.l_levels << ',' << row.u_levels << ','
            << format_double(row.avg_level_width) << ',' << row.max_level_width << ','
            << (row.factor_failed ? "true" : "false") << ',' << row.stop_reason << ','
            << row.bicgstab_iters << ',' << format_double(row.true_linear_rel_res) << ','
            << format_double(row.true_linear_abs_res) << ','
            << format_double(row.dx_norm_ratio_vs_cudss) << ','
            << format_double(row.dx_cosine_vs_cudss) << ','
            << format_double(row.theta_norm_ratio) << ',' << format_double(row.theta_cosine) << ','
            << format_double(row.vmag_norm_ratio) << ',' << format_double(row.vmag_cosine) << ','
            << format_double(row.preconditioner_setup_ms) << ','
            << format_double(row.preconditioner_apply_ms) << ','
            << format_double(row.block_ilu_factor_ms) << ','
            << format_double(row.block_ilu_forward_ms) << ','
            << format_double(row.block_ilu_backward_ms) << ','
            << format_double(row.bicgstab_total_ms) << ',' << format_double(row.spmv_ms) << ','
            << format_double(row.dot_ms) << ',' << format_double(row.update_ms) << '\n';
    }
}

void write_timing_csv(const std::filesystem::path& path, const std::vector<SolveRow>& rows)
{
    std::ofstream out(path);
    out << "case,preconditioner,block_size,preconditioner_setup_ms,block_ilu_factor_ms,"
           "preconditioner_apply_ms,block_ilu_forward_ms,block_ilu_backward_ms,"
           "bicgstab_total_ms,spmv_ms,dot_ms,update_ms\n";
    for (const SolveRow& row : rows) {
        out << row.case_name << ',' << row.preconditioner << ',' << row.block_size << ','
            << format_double(row.preconditioner_setup_ms) << ','
            << format_double(row.block_ilu_factor_ms) << ','
            << format_double(row.preconditioner_apply_ms) << ','
            << format_double(row.block_ilu_forward_ms) << ','
            << format_double(row.block_ilu_backward_ms) << ','
            << format_double(row.bicgstab_total_ms) << ',' << format_double(row.spmv_ms) << ','
            << format_double(row.dot_ms) << ',' << format_double(row.update_ms) << '\n';
    }
}

int32_t quality_pass_count(const std::vector<SolveRow>& rows, int32_t block_size)
{
    int32_t pass = 0;
    for (const SolveRow& ilu : rows) {
        if (ilu.preconditioner != "block_ilu0" || ilu.block_size != block_size ||
            ilu.factor_failed) {
            continue;
        }
        const auto baseline_it = std::find_if(rows.begin(), rows.end(), [&](const SolveRow& bj) {
            return bj.case_name == ilu.case_name &&
                   bj.preconditioner == "block_jacobi" &&
                   bj.block_size == block_size;
        });
        if (baseline_it == rows.end()) {
            continue;
        }
        const SolveRow& bj = *baseline_it;
        if (ilu.dx_norm_ratio_vs_cudss > bj.dx_norm_ratio_vs_cudss &&
            ilu.dx_cosine_vs_cudss > bj.dx_cosine_vs_cudss &&
            ilu.true_linear_rel_res < bj.true_linear_rel_res) {
            ++pass;
        }
    }
    return pass;
}

void write_report(const std::filesystem::path& path,
                  const std::vector<SolveRow>& rows,
                  const std::vector<int32_t>& block_sizes)
{
    std::ofstream out(path);
    out << "# Block ILU(0) Numeric Pilot\n\n";
    out << "This pilot uses CPU dense block ILU(0) factor/apply for standalone J1/F1 quality. "
           "No hybrid NR run was performed unless the standalone gate passed.\n\n";
    out << "## Standalone Gate\n\n";
    out << "| block size | pass cases / 5 | gate |\n";
    out << "|---:|---:|---|\n";
    bool any_pass = false;
    for (int32_t block_size : block_sizes) {
        const int32_t pass = quality_pass_count(rows, block_size);
        const bool gate = pass >= 3;
        any_pass = any_pass || gate;
        out << '|' << block_size << '|' << pass << "|"
            << (gate ? "pass" : "fail") << "|\n";
    }
    out << '\n';

    out << "## Case Summary\n\n";
    out << "| case | bs | BJ relres | ILU relres | BJ dx ratio | ILU dx ratio | "
           "BJ cosine | ILU cosine | ILU setup ms | ILU BiCGSTAB ms |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const SolveRow& ilu : rows) {
        if (ilu.preconditioner != "block_ilu0") {
            continue;
        }
        const auto bj_it = std::find_if(rows.begin(), rows.end(), [&](const SolveRow& bj) {
            return bj.case_name == ilu.case_name &&
                   bj.preconditioner == "block_jacobi" &&
                   bj.block_size == ilu.block_size;
        });
        if (bj_it == rows.end()) {
            continue;
        }
        const SolveRow& bj = *bj_it;
        out << '|' << ilu.case_name << '|' << ilu.block_size << '|'
            << format_double(bj.true_linear_rel_res) << '|'
            << format_double(ilu.true_linear_rel_res) << '|'
            << format_double(bj.dx_norm_ratio_vs_cudss) << '|'
            << format_double(ilu.dx_norm_ratio_vs_cudss) << '|'
            << format_double(bj.dx_cosine_vs_cudss) << '|'
            << format_double(ilu.dx_cosine_vs_cudss) << '|'
            << format_double(ilu.preconditioner_setup_ms + ilu.block_ilu_factor_ms) << '|'
            << format_double(ilu.bicgstab_total_ms) << "|\n";
    }
    out << '\n';
    out << "## Decision\n\n";
    if (any_pass) {
        out << "Standalone gate passed for at least one candidate. Hybrid NR integration can be "
               "attempted next, but this run did not include it.\n";
    } else {
        out << "Standalone gate failed. Per the requested gate, hybrid NR was not run. "
               "Numeric block ILU(0) is not worth optimizing further in this form unless "
               "the quality criterion is relaxed.\n";
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        std::filesystem::create_directories(options.output_dir);
        std::vector<SolveRow> rows;

        for (const std::string& case_name : options.cases) {
            std::cout << "[case] " << case_name << '\n';
            const auto j_path = jacobian_path(options.jf_root, case_name, options.iteration);
            const auto f_path = rhs_path(options.jf_root, case_name, options.iteration);
            if (!std::filesystem::exists(j_path) || !std::filesystem::exists(f_path)) {
                if (options.allow_missing) {
                    std::cerr << "[skip] missing J/F for " << case_name << '\n';
                    continue;
                }
                throw std::runtime_error("missing J/F for " + case_name);
            }
            const cuiter::CsrMatrix matrix = load_cupf_csr_dump(j_path);
            const std::vector<double> rhs = load_cupf_vector_dump(f_path);
            if (static_cast<int32_t>(rhs.size()) != matrix.rows) {
                throw std::runtime_error("RHS dimension mismatch for " + case_name);
            }
            const LinearMetadata metadata = make_linear_metadata(options.case_root / case_name);
            if (static_cast<int32_t>(metadata.index_field.size()) != matrix.rows) {
                throw std::runtime_error("case metadata dimension mismatch for " + case_name);
            }
            double cudss_factor_solve_ms = 0.0;
            const std::vector<double> cudss_x = solve_cudss_dx(matrix, rhs, cudss_factor_solve_ms);
            std::cout << "  cuDSS factor+solve ms=" << format_double(cudss_factor_solve_ms) << '\n';

            for (int32_t block_size : options.block_sizes) {
                std::cout << "  [bs=" << block_size << "] block-Jacobi\n";
                rows.push_back(solve_one(case_name,
                                         matrix,
                                         rhs,
                                         metadata,
                                         cudss_x,
                                         block_size,
                                         false,
                                         options));
                std::cout << "  [bs=" << block_size << "] block-ILU0 coloring\n";
                rows.push_back(solve_one(case_name,
                                         matrix,
                                         rhs,
                                         metadata,
                                         cudss_x,
                                         block_size,
                                         true,
                                         options));
            }
        }

        write_standalone_csv(options.output_dir / "block_ilu0_standalone_quality.csv", rows);
        write_timing_csv(options.output_dir / "block_ilu0_timing.csv", rows);
        write_report(options.output_dir / "block_ilu0_report.md", rows, options.block_sizes);
        std::cout << "[done] wrote block ILU0 pilot results to " << options.output_dir << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
