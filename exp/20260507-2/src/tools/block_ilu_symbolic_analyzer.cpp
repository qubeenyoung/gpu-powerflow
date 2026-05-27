#include "cuiter/core/csr_matrix.hpp"
#include "cuiter/reorder/metis_partition.hpp"

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
#include <map>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

extern "C" {
using metis_idx_t = int32_t;

int METIS_NodeND(metis_idx_t* nvtxs,
                 metis_idx_t* xadj,
                 metis_idx_t* adjncy,
                 metis_idx_t* vwgt,
                 metis_idx_t* options,
                 metis_idx_t* perm,
                 metis_idx_t* iperm);
}

namespace {

constexpr int kMetisOk = 1;
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

struct CliOptions {
    std::filesystem::path jf_root = "raw/cupf_jf_dumps";
    std::filesystem::path output_dir = "results";
    std::vector<std::string> cases = {
        "case2383wp",
        "case3120sp",
        "case9241pegase",
        "case13659pegase",
        "case6468rte",
    };
    std::vector<int32_t> iterations = {1};
    std::vector<int32_t> block_sizes = {8, 16, 32};
    std::vector<std::string> orderings = {"current", "rcm", "metis_nd", "coloring"};
    bool allow_missing = false;
};

struct ScalarRetentionStats {
    int64_t scalar_nnz_in_diag_blocks = 0;
    int64_t scalar_nnz_off_blocks = 0;
    double abs_value_in_diag_blocks = 0.0;
    double abs_value_off_blocks = 0.0;
    double fro_in_diag_blocks = 0.0;
    double fro_off_blocks = 0.0;
};

struct BlockGraph {
    int32_t n = 0;
    int32_t nnz = 0;
    int32_t block_size_target = 0;
    std::vector<int32_t> block_dims;
    std::vector<std::vector<int32_t>> out_edges;
    std::vector<std::vector<int32_t>> in_edges;
    std::vector<std::vector<int32_t>> undirected_edges;
    std::unordered_set<uint64_t> directed_edge_set;
    int64_t block_nnz_diag = 0;
    int64_t block_nnz_offdiag = 0;
    ScalarRetentionStats scalar;
};

struct GraphStatsRow {
    std::string case_name;
    int32_t iteration = 1;
    int32_t n = 0;
    int32_t nnz = 0;
    int32_t block_size_target = 0;
    int32_t num_blocks = 0;
    int32_t min_block_dim = 0;
    int32_t max_block_dim = 0;
    double avg_block_dim = 0.0;
    double std_block_dim = 0.0;
    int64_t block_nnz_total = 0;
    int64_t block_nnz_diag = 0;
    int64_t block_nnz_offdiag = 0;
    double avg_block_degree = 0.0;
    int32_t max_block_degree = 0;
    ScalarRetentionStats scalar;
};

struct LevelStats {
    int32_t num_levels = 0;
    int32_t min_level_width = 0;
    int32_t max_level_width = 0;
    double avg_level_width = 0.0;
    double median_level_width = 0.0;
    double p10_level_width = 0.0;
    double p90_level_width = 0.0;
    int32_t critical_path_length = 0;
    double level_width_cv = 0.0;
};

struct OrderedAnalysis {
    std::string case_name;
    int32_t block_size_target = 0;
    std::string ordering;
    bool ordering_available = true;
    std::string skip_reason;
    int32_t num_blocks = 0;
    int32_t num_colors = 0;
    int32_t min_color_size = 0;
    int32_t max_color_size = 0;
    double avg_color_size = 0.0;
    LevelStats l_levels;
    LevelStats u_levels;
    double l_offdiag_gemv_work = 0.0;
    double u_offdiag_gemv_work = 0.0;
    double diag_apply_work = 0.0;
    double total_apply_work = 0.0;
    double block_jacobi_apply_work_estimate = 0.0;
    double apply_work_relative_to_block_jacobi = 0.0;
    double diag_factor_work = 0.0;
    double offdiag_update_work = 0.0;
    double total_factor_work = 0.0;
    double block_jacobi_setup_work_estimate = 0.0;
    double factor_work_relative_to_block_jacobi_setup = 0.0;
    double risk_score = 0.0;
    std::string verdict;
};

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

std::vector<int32_t> parse_iteration_list(const std::string& text)
{
    if (text == "all" || text == "Jall") {
        return {0, 1, 2};
    }
    std::vector<int32_t> values;
    for (std::string item : split_list(text)) {
        if (!item.empty() && (item.front() == 'J' || item.front() == 'j')) {
            item.erase(item.begin());
        }
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
        << "  --output-dir PATH\n"
        << "  --cases a,b,c\n"
        << "  --iters J1 | J0,J1,J2 | all\n"
        << "  --block-sizes 8,16,32\n"
        << "  --orderings current,rcm,metis_nd,coloring\n"
        << "  --allow-missing\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--jf-root" && i + 1 < argc) {
            options.jf_root = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            options.output_dir = argv[++i];
        } else if (arg == "--cases" && i + 1 < argc) {
            options.cases = split_list(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            options.iterations = parse_iteration_list(argv[++i]);
        } else if (arg == "--block-sizes" && i + 1 < argc) {
            options.block_sizes = parse_int_list(argv[++i]);
        } else if (arg == "--orderings" && i + 1 < argc) {
            options.orderings = split_list(argv[++i]);
        } else if (arg == "--allow-missing") {
            options.allow_missing = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.cases.empty()) {
        throw std::runtime_error("--cases produced an empty case list");
    }
    if (options.iterations.empty()) {
        throw std::runtime_error("--iters produced an empty iteration list");
    }
    if (options.block_sizes.empty()) {
        throw std::runtime_error("--block-sizes produced an empty block-size list");
    }
    for (int32_t block_size : options.block_sizes) {
        if (block_size <= 0) {
            throw std::runtime_error("block sizes must be positive");
        }
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

std::filesystem::path jacobian_path(const std::filesystem::path& jf_root,
                                    const std::string& case_name,
                                    int32_t iteration)
{
    const std::filesystem::path case_dir = jf_root / case_name;
    const std::filesystem::path direct = case_dir / ("J" + std::to_string(iteration) + ".txt");
    if (std::filesystem::exists(direct)) {
        return direct;
    }
    const std::filesystem::path repeat =
        case_dir / "repeat_00" / ("jacobian_iter" + std::to_string(iteration) + ".txt");
    if (std::filesystem::exists(repeat)) {
        return repeat;
    }
    return direct;
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

double ratio(double numerator, double denominator)
{
    return denominator > 0.0 ? numerator / denominator : 0.0;
}

double percentile(std::vector<double> values, double q)
{
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    if (values.size() == 1) {
        return values.front();
    }
    const double position = q * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(position));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lo);
    return (1.0 - weight) * values[lo] + weight * values[hi];
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

std::vector<int32_t> block_of_new_indices(int32_t n,
                                          const std::vector<int32_t>& block_starts,
                                          const std::vector<int32_t>& block_sizes)
{
    std::vector<int32_t> block_of_new(static_cast<std::size_t>(n), -1);
    for (int32_t block = 0; block < static_cast<int32_t>(block_sizes.size()); ++block) {
        const int32_t begin = block_starts[static_cast<std::size_t>(block)];
        const int32_t end = begin + block_sizes[static_cast<std::size_t>(block)];
        for (int32_t index = begin; index < end; ++index) {
            block_of_new[static_cast<std::size_t>(index)] = block;
        }
    }
    for (int32_t block : block_of_new) {
        if (block < 0) {
            throw std::runtime_error("invalid block structure: missing block assignment");
        }
    }
    return block_of_new;
}

bool same_pattern(const cuiter::CsrMatrix& lhs, const cuiter::CsrMatrix& rhs)
{
    return lhs.rows == rhs.rows &&
           lhs.cols == rhs.cols &&
           lhs.row_ptr == rhs.row_ptr &&
           lhs.col_idx == rhs.col_idx;
}

void sort_unique(std::vector<int32_t>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

BlockGraph build_block_graph(const cuiter::CsrMatrix& matrix,
                             const cuiter::MetisPermutation& permutation,
                             const cuiter::PermutedCsrPattern& pattern,
                             int32_t block_size_target)
{
    BlockGraph graph;
    graph.n = matrix.rows;
    graph.nnz = matrix.nnz();
    graph.block_size_target = block_size_target;
    graph.block_dims = permutation.block_sizes;
    const int32_t nb = static_cast<int32_t>(graph.block_dims.size());
    graph.out_edges.assign(static_cast<std::size_t>(nb), {});
    graph.in_edges.assign(static_cast<std::size_t>(nb), {});
    graph.undirected_edges.assign(static_cast<std::size_t>(nb), {});

    const std::vector<int32_t> block_of_new =
        block_of_new_indices(matrix.rows, permutation.block_starts, permutation.block_sizes);

    for (int32_t new_row = 0; new_row < pattern.rows; ++new_row) {
        const int32_t row_block = block_of_new[static_cast<std::size_t>(new_row)];
        for (int32_t pos = pattern.row_ptr[static_cast<std::size_t>(new_row)];
             pos < pattern.row_ptr[static_cast<std::size_t>(new_row + 1)];
             ++pos) {
            const int32_t new_col = pattern.col_idx[static_cast<std::size_t>(pos)];
            const int32_t col_block = block_of_new[static_cast<std::size_t>(new_col)];
            const uint64_t key = edge_key(row_block, col_block);
            graph.directed_edge_set.insert(key);

            const double value =
                matrix.values[static_cast<std::size_t>(
                    pattern.value_source_index[static_cast<std::size_t>(pos)])];
            if (row_block == col_block) {
                ++graph.scalar.scalar_nnz_in_diag_blocks;
                graph.scalar.abs_value_in_diag_blocks += std::abs(value);
                graph.scalar.fro_in_diag_blocks += value * value;
            } else {
                ++graph.scalar.scalar_nnz_off_blocks;
                graph.scalar.abs_value_off_blocks += std::abs(value);
                graph.scalar.fro_off_blocks += value * value;
            }
        }
    }

    for (uint64_t key : graph.directed_edge_set) {
        const int32_t row = edge_key_row(key);
        const int32_t col = edge_key_col(key);
        if (row == col) {
            ++graph.block_nnz_diag;
        } else {
            ++graph.block_nnz_offdiag;
            graph.out_edges[static_cast<std::size_t>(row)].push_back(col);
            graph.in_edges[static_cast<std::size_t>(col)].push_back(row);
            graph.undirected_edges[static_cast<std::size_t>(row)].push_back(col);
            graph.undirected_edges[static_cast<std::size_t>(col)].push_back(row);
        }
    }
    for (int32_t block = 0; block < nb; ++block) {
        sort_unique(graph.out_edges[static_cast<std::size_t>(block)]);
        sort_unique(graph.in_edges[static_cast<std::size_t>(block)]);
        sort_unique(graph.undirected_edges[static_cast<std::size_t>(block)]);
    }
    return graph;
}

GraphStatsRow make_graph_stats_row(const std::string& case_name,
                                   int32_t iteration,
                                   const cuiter::CsrMatrix& matrix,
                                   const cuiter::MetisPermutation& permutation,
                                   int32_t block_size_target)
{
    const cuiter::PermutedCsrPattern pattern =
        cuiter::build_permuted_csr_pattern(matrix, permutation);
    const BlockGraph graph = build_block_graph(matrix, permutation, pattern, block_size_target);

    GraphStatsRow row;
    row.case_name = case_name;
    row.iteration = iteration;
    row.n = matrix.rows;
    row.nnz = matrix.nnz();
    row.block_size_target = block_size_target;
    row.num_blocks = static_cast<int32_t>(graph.block_dims.size());
    row.min_block_dim = *std::min_element(graph.block_dims.begin(), graph.block_dims.end());
    row.max_block_dim = *std::max_element(graph.block_dims.begin(), graph.block_dims.end());
    row.avg_block_dim =
        static_cast<double>(matrix.rows) / static_cast<double>(std::max(1, row.num_blocks));
    double variance = 0.0;
    for (int32_t dim : graph.block_dims) {
        const double diff = static_cast<double>(dim) - row.avg_block_dim;
        variance += diff * diff;
    }
    row.std_block_dim =
        std::sqrt(variance / static_cast<double>(std::max(1, row.num_blocks)));
    row.block_nnz_diag = graph.block_nnz_diag;
    row.block_nnz_offdiag = graph.block_nnz_offdiag;
    row.block_nnz_total = row.block_nnz_diag + row.block_nnz_offdiag;
    int64_t degree_sum = 0;
    for (const auto& neighbors : graph.undirected_edges) {
        degree_sum += static_cast<int64_t>(neighbors.size());
        row.max_block_degree =
            std::max<int32_t>(row.max_block_degree, static_cast<int32_t>(neighbors.size()));
    }
    row.avg_block_degree =
        row.num_blocks > 0 ? static_cast<double>(degree_sum) / static_cast<double>(row.num_blocks)
                           : 0.0;
    row.scalar = graph.scalar;
    return row;
}

std::vector<int32_t> identity_order(int32_t n)
{
    std::vector<int32_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    return order;
}

std::vector<int32_t> rcm_order(const std::vector<std::vector<int32_t>>& adjacency)
{
    const int32_t n = static_cast<int32_t>(adjacency.size());
    std::vector<int32_t> order;
    std::vector<char> visited(static_cast<std::size_t>(n), 0);
    order.reserve(static_cast<std::size_t>(n));

    while (static_cast<int32_t>(order.size()) < n) {
        int32_t start = -1;
        for (int32_t i = 0; i < n; ++i) {
            if (visited[static_cast<std::size_t>(i)]) {
                continue;
            }
            if (start < 0 ||
                adjacency[static_cast<std::size_t>(i)].size() <
                    adjacency[static_cast<std::size_t>(start)].size()) {
                start = i;
            }
        }
        std::queue<int32_t> queue;
        queue.push(start);
        visited[static_cast<std::size_t>(start)] = 1;
        while (!queue.empty()) {
            const int32_t node = queue.front();
            queue.pop();
            order.push_back(node);
            std::vector<int32_t> neighbors = adjacency[static_cast<std::size_t>(node)];
            std::sort(neighbors.begin(), neighbors.end(), [&](int32_t lhs, int32_t rhs) {
                const auto lhs_degree = adjacency[static_cast<std::size_t>(lhs)].size();
                const auto rhs_degree = adjacency[static_cast<std::size_t>(rhs)].size();
                if (lhs_degree != rhs_degree) {
                    return lhs_degree < rhs_degree;
                }
                return lhs < rhs;
            });
            for (int32_t neighbor : neighbors) {
                if (!visited[static_cast<std::size_t>(neighbor)]) {
                    visited[static_cast<std::size_t>(neighbor)] = 1;
                    queue.push(neighbor);
                }
            }
        }
    }
    std::reverse(order.begin(), order.end());
    return order;
}

std::vector<int32_t> metis_nd_order(const std::vector<std::vector<int32_t>>& adjacency,
                                    std::string& skip_reason)
{
    const int32_t n = static_cast<int32_t>(adjacency.size());
    if (n <= 0) {
        skip_reason = "empty graph";
        return {};
    }
    std::vector<metis_idx_t> xadj(static_cast<std::size_t>(n + 1), 0);
    std::vector<metis_idx_t> adjncy;
    for (int32_t i = 0; i < n; ++i) {
        xadj[static_cast<std::size_t>(i)] = static_cast<metis_idx_t>(adjncy.size());
        for (int32_t neighbor : adjacency[static_cast<std::size_t>(i)]) {
            adjncy.push_back(static_cast<metis_idx_t>(neighbor));
        }
    }
    xadj[static_cast<std::size_t>(n)] = static_cast<metis_idx_t>(adjncy.size());
    std::vector<metis_idx_t> perm(static_cast<std::size_t>(n), 0);
    std::vector<metis_idx_t> iperm(static_cast<std::size_t>(n), 0);
    metis_idx_t nvtxs = n;
    const int status =
        METIS_NodeND(&nvtxs, xadj.data(), adjncy.data(), nullptr, nullptr, perm.data(), iperm.data());
    if (status != kMetisOk) {
        skip_reason = "METIS_NodeND failed with status=" + std::to_string(status);
        return {};
    }
    std::vector<int32_t> order(static_cast<std::size_t>(n), -1);
    for (int32_t new_index = 0; new_index < n; ++new_index) {
        const int32_t old_index = iperm[static_cast<std::size_t>(new_index)];
        if (old_index < 0 || old_index >= n) {
            skip_reason = "METIS_NodeND returned invalid iperm";
            return {};
        }
        order[static_cast<std::size_t>(new_index)] = old_index;
    }
    return order;
}

std::vector<int32_t> greedy_coloring_order(const std::vector<std::vector<int32_t>>& adjacency,
                                           int32_t& num_colors,
                                           int32_t& min_color_size,
                                           int32_t& max_color_size,
                                           double& avg_color_size)
{
    const int32_t n = static_cast<int32_t>(adjacency.size());
    std::vector<int32_t> color(static_cast<std::size_t>(n), -1);
    int32_t max_color = -1;
    for (int32_t node = 0; node < n; ++node) {
        std::vector<char> used(static_cast<std::size_t>(std::max(1, max_color + 2)), 0);
        for (int32_t neighbor : adjacency[static_cast<std::size_t>(node)]) {
            const int32_t neighbor_color = color[static_cast<std::size_t>(neighbor)];
            if (neighbor_color >= 0 && neighbor_color < static_cast<int32_t>(used.size())) {
                used[static_cast<std::size_t>(neighbor_color)] = 1;
            }
        }
        int32_t chosen = 0;
        while (chosen < static_cast<int32_t>(used.size()) && used[static_cast<std::size_t>(chosen)]) {
            ++chosen;
        }
        color[static_cast<std::size_t>(node)] = chosen;
        max_color = std::max(max_color, chosen);
    }

    num_colors = max_color + 1;
    std::vector<int32_t> color_sizes(static_cast<std::size_t>(num_colors), 0);
    for (int32_t c : color) {
        ++color_sizes[static_cast<std::size_t>(c)];
    }
    min_color_size = *std::min_element(color_sizes.begin(), color_sizes.end());
    max_color_size = *std::max_element(color_sizes.begin(), color_sizes.end());
    avg_color_size = static_cast<double>(n) / static_cast<double>(std::max(1, num_colors));

    std::vector<int32_t> order = identity_order(n);
    std::stable_sort(order.begin(), order.end(), [&](int32_t lhs, int32_t rhs) {
        if (color[static_cast<std::size_t>(lhs)] != color[static_cast<std::size_t>(rhs)]) {
            return color[static_cast<std::size_t>(lhs)] < color[static_cast<std::size_t>(rhs)];
        }
        return lhs < rhs;
    });
    return order;
}

LevelStats summarize_levels(const std::vector<int32_t>& levels)
{
    LevelStats stats;
    if (levels.empty()) {
        return stats;
    }
    const int32_t max_level = *std::max_element(levels.begin(), levels.end());
    std::vector<double> widths(static_cast<std::size_t>(max_level + 1), 0.0);
    for (int32_t level : levels) {
        widths[static_cast<std::size_t>(level)] += 1.0;
    }
    stats.num_levels = static_cast<int32_t>(widths.size());
    stats.critical_path_length = stats.num_levels;
    stats.min_level_width = static_cast<int32_t>(*std::min_element(widths.begin(), widths.end()));
    stats.max_level_width = static_cast<int32_t>(*std::max_element(widths.begin(), widths.end()));
    stats.avg_level_width =
        static_cast<double>(levels.size()) / static_cast<double>(std::max(1, stats.num_levels));
    stats.median_level_width = percentile(widths, 0.50);
    stats.p10_level_width = percentile(widths, 0.10);
    stats.p90_level_width = percentile(widths, 0.90);
    double variance = 0.0;
    for (double width : widths) {
        const double diff = width - stats.avg_level_width;
        variance += diff * diff;
    }
    const double stddev = std::sqrt(variance / static_cast<double>(widths.size()));
    stats.level_width_cv = stats.avg_level_width > 0.0 ? stddev / stats.avg_level_width : 0.0;
    return stats;
}

LevelStats analyze_triangular_levels(int32_t nb,
                                     const std::vector<std::vector<int32_t>>& deps,
                                     bool forward)
{
    std::vector<int32_t> levels(static_cast<std::size_t>(nb), 0);
    if (forward) {
        for (int32_t node = 0; node < nb; ++node) {
            int32_t level = 0;
            for (int32_t dep : deps[static_cast<std::size_t>(node)]) {
                level = std::max(level, levels[static_cast<std::size_t>(dep)] + 1);
            }
            levels[static_cast<std::size_t>(node)] = level;
        }
    } else {
        for (int32_t node = nb - 1; node >= 0; --node) {
            int32_t level = 0;
            for (int32_t dep : deps[static_cast<std::size_t>(node)]) {
                level = std::max(level, levels[static_cast<std::size_t>(dep)] + 1);
            }
            levels[static_cast<std::size_t>(node)] = level;
        }
    }
    return summarize_levels(levels);
}

std::string classify_candidate(const OrderedAnalysis& row)
{
    if (!row.ordering_available) {
        return "skip";
    }
    const int32_t max_levels = std::max(row.l_levels.num_levels, row.u_levels.num_levels);
    const double level_fraction =
        row.num_blocks > 0 ? static_cast<double>(max_levels) / static_cast<double>(row.num_blocks)
                           : 1.0;
    const double min_avg_width =
        std::min(row.l_levels.avg_level_width, row.u_levels.avg_level_width);
    if (level_fraction <= 0.20 && min_avg_width >= 8.0 &&
        row.apply_work_relative_to_block_jacobi <= 5.0) {
        return "good";
    }
    if (level_fraction <= 0.50 && min_avg_width >= 4.0 &&
        row.apply_work_relative_to_block_jacobi <= 10.0) {
        return "risky";
    }
    return "reject";
}

double score_candidate(const OrderedAnalysis& row)
{
    if (!row.ordering_available) {
        return std::numeric_limits<double>::infinity();
    }
    const int32_t max_levels = std::max(row.l_levels.num_levels, row.u_levels.num_levels);
    const double max_fraction =
        row.num_blocks > 0 ? static_cast<double>(max_levels) / static_cast<double>(row.num_blocks)
                           : 1.0;
    const double min_avg_width =
        std::max(1.0, std::min(row.l_levels.avg_level_width, row.u_levels.avg_level_width));
    const double width_term = 1.0 / min_avg_width;
    return 4.0 * max_fraction +
           0.35 * row.apply_work_relative_to_block_jacobi +
           0.10 * row.factor_work_relative_to_block_jacobi_setup +
           0.25 * width_term +
           0.10 * std::max(row.l_levels.level_width_cv, row.u_levels.level_width_cv);
}

OrderedAnalysis analyze_ordered_graph(const std::string& case_name,
                                      const BlockGraph& graph,
                                      const std::string& ordering_name)
{
    OrderedAnalysis row;
    row.case_name = case_name;
    row.block_size_target = graph.block_size_target;
    row.ordering = ordering_name;
    row.num_blocks = static_cast<int32_t>(graph.block_dims.size());
    std::vector<int32_t> order;
    if (ordering_name == "current" || ordering_name == "current_metis_block_order") {
        order = identity_order(row.num_blocks);
        row.ordering = "current_metis_block_order";
    } else if (ordering_name == "rcm" || ordering_name == "block_rcm") {
        order = rcm_order(graph.undirected_edges);
        row.ordering = "block_rcm";
    } else if (ordering_name == "metis_nd" || ordering_name == "block_metis_nd") {
        order = metis_nd_order(graph.undirected_edges, row.skip_reason);
        row.ordering = "block_metis_nd";
    } else if (ordering_name == "coloring" || ordering_name == "block_coloring") {
        order = greedy_coloring_order(graph.undirected_edges,
                                      row.num_colors,
                                      row.min_color_size,
                                      row.max_color_size,
                                      row.avg_color_size);
        row.ordering = "block_coloring";
    } else {
        row.ordering_available = false;
        row.skip_reason = "unknown ordering";
        row.verdict = "skip";
        row.risk_score = std::numeric_limits<double>::infinity();
        return row;
    }
    if (order.empty()) {
        row.ordering_available = false;
        row.verdict = "skip";
        row.risk_score = std::numeric_limits<double>::infinity();
        return row;
    }

    std::vector<int32_t> old_to_new(static_cast<std::size_t>(row.num_blocks), -1);
    std::vector<int32_t> ordered_dims(static_cast<std::size_t>(row.num_blocks), 0);
    for (int32_t new_id = 0; new_id < row.num_blocks; ++new_id) {
        const int32_t old_id = order[static_cast<std::size_t>(new_id)];
        if (old_id < 0 || old_id >= row.num_blocks) {
            row.ordering_available = false;
            row.skip_reason = "ordering contains invalid block id";
            row.verdict = "skip";
            row.risk_score = std::numeric_limits<double>::infinity();
            return row;
        }
        old_to_new[static_cast<std::size_t>(old_id)] = new_id;
        ordered_dims[static_cast<std::size_t>(new_id)] =
            graph.block_dims[static_cast<std::size_t>(old_id)];
    }

    std::vector<std::vector<int32_t>> l_deps(static_cast<std::size_t>(row.num_blocks));
    std::vector<std::vector<int32_t>> u_deps(static_cast<std::size_t>(row.num_blocks));
    std::vector<std::vector<int32_t>> ordered_out(static_cast<std::size_t>(row.num_blocks));
    std::vector<std::vector<int32_t>> ordered_in(static_cast<std::size_t>(row.num_blocks));
    std::unordered_set<uint64_t> ordered_edges;
    ordered_edges.reserve(graph.directed_edge_set.size() * 2);

    for (uint64_t old_key : graph.directed_edge_set) {
        const int32_t old_row = edge_key_row(old_key);
        const int32_t old_col = edge_key_col(old_key);
        const int32_t new_row = old_to_new[static_cast<std::size_t>(old_row)];
        const int32_t new_col = old_to_new[static_cast<std::size_t>(old_col)];
        ordered_edges.insert(edge_key(new_row, new_col));
        if (new_row == new_col) {
            continue;
        }
        ordered_out[static_cast<std::size_t>(new_row)].push_back(new_col);
        ordered_in[static_cast<std::size_t>(new_col)].push_back(new_row);
        const double edge_work = static_cast<double>(ordered_dims[static_cast<std::size_t>(new_row)]) *
                                 static_cast<double>(ordered_dims[static_cast<std::size_t>(new_col)]);
        if (new_row > new_col) {
            l_deps[static_cast<std::size_t>(new_row)].push_back(new_col);
            row.l_offdiag_gemv_work += edge_work;
        } else {
            u_deps[static_cast<std::size_t>(new_row)].push_back(new_col);
            row.u_offdiag_gemv_work += edge_work;
        }
    }
    for (auto& deps : l_deps) {
        sort_unique(deps);
    }
    for (auto& deps : u_deps) {
        sort_unique(deps);
    }
    for (auto& values : ordered_out) {
        sort_unique(values);
    }
    for (auto& values : ordered_in) {
        sort_unique(values);
    }

    row.l_levels = analyze_triangular_levels(row.num_blocks, l_deps, true);
    row.u_levels = analyze_triangular_levels(row.num_blocks, u_deps, false);

    for (int32_t dim : ordered_dims) {
        const double d = static_cast<double>(dim);
        row.diag_apply_work += d * d;
        row.diag_factor_work += d * d * d;
    }
    row.block_jacobi_apply_work_estimate = row.diag_apply_work;
    row.block_jacobi_setup_work_estimate = row.diag_factor_work;
    row.total_apply_work =
        row.l_offdiag_gemv_work + row.u_offdiag_gemv_work + row.diag_apply_work;
    row.apply_work_relative_to_block_jacobi =
        ratio(row.total_apply_work, row.block_jacobi_apply_work_estimate);

    for (int32_t k = 0; k < row.num_blocks; ++k) {
        std::vector<int32_t> lowers;
        std::vector<int32_t> uppers;
        for (int32_t i : ordered_in[static_cast<std::size_t>(k)]) {
            if (i > k) {
                lowers.push_back(i);
            }
        }
        for (int32_t j : ordered_out[static_cast<std::size_t>(k)]) {
            if (j > k) {
                uppers.push_back(j);
            }
        }
        const double dk = static_cast<double>(ordered_dims[static_cast<std::size_t>(k)]);
        for (int32_t i : lowers) {
            const double di = static_cast<double>(ordered_dims[static_cast<std::size_t>(i)]);
            for (int32_t j : uppers) {
                if (ordered_edges.find(edge_key(i, j)) == ordered_edges.end()) {
                    continue;
                }
                const double dj = static_cast<double>(ordered_dims[static_cast<std::size_t>(j)]);
                row.offdiag_update_work += di * dk * dj;
            }
        }
    }
    row.total_factor_work = row.diag_factor_work + row.offdiag_update_work;
    row.factor_work_relative_to_block_jacobi_setup =
        ratio(row.total_factor_work, row.block_jacobi_setup_work_estimate);
    row.verdict = classify_candidate(row);
    row.risk_score = score_candidate(row);
    return row;
}

void write_graph_stats_csv(const std::filesystem::path& path,
                           const std::vector<GraphStatsRow>& rows)
{
    std::ofstream out(path);
    out << "case,iteration,n,nnz,block_size_target,num_blocks,min_block_dim,max_block_dim,"
           "avg_block_dim,std_block_dim,block_nnz_total,block_nnz_diag,block_nnz_offdiag,"
           "avg_block_degree,max_block_degree,scalar_nnz_in_diag_blocks,scalar_nnz_off_blocks,"
           "scalar_offblock_nnz_ratio,abs_value_in_diag_blocks,abs_value_off_blocks,"
           "abs_value_offblock_ratio,fro_in_diag_blocks,fro_off_blocks,fro_offblock_ratio\n";
    for (const GraphStatsRow& row : rows) {
        const int64_t scalar_total =
            row.scalar.scalar_nnz_in_diag_blocks + row.scalar.scalar_nnz_off_blocks;
        const double abs_total =
            row.scalar.abs_value_in_diag_blocks + row.scalar.abs_value_off_blocks;
        const double fro_total =
            row.scalar.fro_in_diag_blocks + row.scalar.fro_off_blocks;
        out << row.case_name << ',' << row.iteration << ',' << row.n << ',' << row.nnz << ','
            << row.block_size_target << ',' << row.num_blocks << ',' << row.min_block_dim << ','
            << row.max_block_dim << ',' << format_double(row.avg_block_dim) << ','
            << format_double(row.std_block_dim) << ',' << row.block_nnz_total << ','
            << row.block_nnz_diag << ',' << row.block_nnz_offdiag << ','
            << format_double(row.avg_block_degree) << ',' << row.max_block_degree << ','
            << row.scalar.scalar_nnz_in_diag_blocks << ','
            << row.scalar.scalar_nnz_off_blocks << ','
            << format_double(ratio(row.scalar.scalar_nnz_off_blocks, scalar_total)) << ','
            << format_double(row.scalar.abs_value_in_diag_blocks) << ','
            << format_double(row.scalar.abs_value_off_blocks) << ','
            << format_double(ratio(row.scalar.abs_value_off_blocks, abs_total)) << ','
            << format_double(std::sqrt(row.scalar.fro_in_diag_blocks)) << ','
            << format_double(std::sqrt(row.scalar.fro_off_blocks)) << ','
            << format_double(fro_total > 0.0 ? std::sqrt(row.scalar.fro_off_blocks / fro_total) : 0.0)
            << '\n';
    }
}

void write_levels_csv(const std::filesystem::path& path,
                      const std::vector<OrderedAnalysis>& rows)
{
    std::ofstream out(path);
    out << "case,block_size_target,ordering,ordering_available,skip_reason,num_blocks,num_colors,"
           "min_color_size,max_color_size,avg_color_size,L_num_levels,L_min_level_width,"
           "L_max_level_width,L_avg_level_width,L_median_level_width,L_p10_level_width,"
           "L_p90_level_width,L_critical_path_length,L_level_width_cv,U_num_levels,"
           "U_min_level_width,U_max_level_width,U_avg_level_width,U_median_level_width,"
           "U_p10_level_width,U_p90_level_width,U_critical_path_length,U_level_width_cv\n";
    for (const OrderedAnalysis& row : rows) {
        out << row.case_name << ',' << row.block_size_target << ',' << row.ordering << ','
            << (row.ordering_available ? "true" : "false") << ',' << row.skip_reason << ','
            << row.num_blocks << ',' << row.num_colors << ',' << row.min_color_size << ','
            << row.max_color_size << ',' << format_double(row.avg_color_size) << ','
            << row.l_levels.num_levels << ',' << row.l_levels.min_level_width << ','
            << row.l_levels.max_level_width << ',' << format_double(row.l_levels.avg_level_width)
            << ',' << format_double(row.l_levels.median_level_width) << ','
            << format_double(row.l_levels.p10_level_width) << ','
            << format_double(row.l_levels.p90_level_width) << ','
            << row.l_levels.critical_path_length << ','
            << format_double(row.l_levels.level_width_cv) << ','
            << row.u_levels.num_levels << ',' << row.u_levels.min_level_width << ','
            << row.u_levels.max_level_width << ',' << format_double(row.u_levels.avg_level_width)
            << ',' << format_double(row.u_levels.median_level_width) << ','
            << format_double(row.u_levels.p10_level_width) << ','
            << format_double(row.u_levels.p90_level_width) << ','
            << row.u_levels.critical_path_length << ','
            << format_double(row.u_levels.level_width_cv) << '\n';
    }
}

void write_apply_work_csv(const std::filesystem::path& path,
                          const std::vector<OrderedAnalysis>& rows)
{
    std::ofstream out(path);
    out << "case,block_size_target,ordering,num_blocks,L_offdiag_gemv_work,U_offdiag_gemv_work,"
           "diag_apply_work,total_apply_work,block_jacobi_apply_work_estimate,"
           "apply_work_relative_to_block_jacobi\n";
    for (const OrderedAnalysis& row : rows) {
        out << row.case_name << ',' << row.block_size_target << ',' << row.ordering << ','
            << row.num_blocks << ',' << format_double(row.l_offdiag_gemv_work) << ','
            << format_double(row.u_offdiag_gemv_work) << ','
            << format_double(row.diag_apply_work) << ','
            << format_double(row.total_apply_work) << ','
            << format_double(row.block_jacobi_apply_work_estimate) << ','
            << format_double(row.apply_work_relative_to_block_jacobi) << '\n';
    }
}

void write_factor_work_csv(const std::filesystem::path& path,
                           const std::vector<OrderedAnalysis>& rows)
{
    std::ofstream out(path);
    out << "case,block_size_target,ordering,num_blocks,diag_factor_work,offdiag_update_work,"
           "total_factor_work,block_jacobi_setup_work_estimate,"
           "factor_work_relative_to_block_jacobi_setup\n";
    for (const OrderedAnalysis& row : rows) {
        out << row.case_name << ',' << row.block_size_target << ',' << row.ordering << ','
            << row.num_blocks << ',' << format_double(row.diag_factor_work) << ','
            << format_double(row.offdiag_update_work) << ','
            << format_double(row.total_factor_work) << ','
            << format_double(row.block_jacobi_setup_work_estimate) << ','
            << format_double(row.factor_work_relative_to_block_jacobi_setup) << '\n';
    }
}

void write_ranked_csv(const std::filesystem::path& path,
                      std::vector<OrderedAnalysis> rows)
{
    std::stable_sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.case_name != rhs.case_name) {
            return lhs.case_name < rhs.case_name;
        }
        return lhs.risk_score < rhs.risk_score;
    });
    std::ofstream out(path);
    out << "case,block_size,ordering,num_blocks,L_levels,U_levels,avg_level_width,"
           "max_level_width,apply_work_relative_to_block_jacobi,"
           "factor_work_relative_to_block_jacobi_setup,risk_score,verdict,skip_reason\n";
    for (const OrderedAnalysis& row : rows) {
        const double avg_width =
            0.5 * (row.l_levels.avg_level_width + row.u_levels.avg_level_width);
        const int32_t max_width =
            std::max(row.l_levels.max_level_width, row.u_levels.max_level_width);
        out << row.case_name << ',' << row.block_size_target << ',' << row.ordering << ','
            << row.num_blocks << ',' << row.l_levels.num_levels << ','
            << row.u_levels.num_levels << ',' << format_double(avg_width) << ','
            << max_width << ',' << format_double(row.apply_work_relative_to_block_jacobi) << ','
            << format_double(row.factor_work_relative_to_block_jacobi_setup) << ','
            << format_double(row.risk_score) << ',' << row.verdict << ','
            << row.skip_reason << '\n';
    }
}

struct AggregateCandidate {
    int32_t block_size = 0;
    std::string ordering;
    int32_t count = 0;
    double mean_score = 0.0;
    double mean_l_levels = 0.0;
    double mean_u_levels = 0.0;
    double mean_avg_width = 0.0;
    double mean_apply_ratio = 0.0;
    double mean_factor_ratio = 0.0;
    int32_t reject_count = 0;
};

std::vector<AggregateCandidate> aggregate_candidates(const std::vector<OrderedAnalysis>& rows)
{
    std::map<std::pair<int32_t, std::string>, AggregateCandidate> agg;
    for (const OrderedAnalysis& row : rows) {
        if (!row.ordering_available) {
            continue;
        }
        auto& item = agg[{row.block_size_target, row.ordering}];
        item.block_size = row.block_size_target;
        item.ordering = row.ordering;
        ++item.count;
        item.mean_score += row.risk_score;
        item.mean_l_levels += row.l_levels.num_levels;
        item.mean_u_levels += row.u_levels.num_levels;
        item.mean_avg_width += 0.5 * (row.l_levels.avg_level_width + row.u_levels.avg_level_width);
        item.mean_apply_ratio += row.apply_work_relative_to_block_jacobi;
        item.mean_factor_ratio += row.factor_work_relative_to_block_jacobi_setup;
        if (row.verdict == "reject") {
            ++item.reject_count;
        }
    }
    std::vector<AggregateCandidate> out;
    for (auto& kv : agg) {
        auto& item = kv.second;
        const double denom = static_cast<double>(std::max(1, item.count));
        item.mean_score /= denom;
        item.mean_l_levels /= denom;
        item.mean_u_levels /= denom;
        item.mean_avg_width /= denom;
        item.mean_apply_ratio /= denom;
        item.mean_factor_ratio /= denom;
        out.push_back(item);
    }
    std::sort(out.begin(), out.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.reject_count != rhs.reject_count) {
            return lhs.reject_count < rhs.reject_count;
        }
        return lhs.mean_score < rhs.mean_score;
    });
    return out;
}

void write_report(const std::filesystem::path& path,
                  const std::vector<GraphStatsRow>& graph_rows,
                  const std::vector<OrderedAnalysis>& ordered_rows,
                  const std::vector<std::string>& pattern_notes)
{
    std::ofstream out(path);
    const std::vector<AggregateCandidate> candidates = aggregate_candidates(ordered_rows);

    out << "# Block ILU(0) Symbolic Feasibility\n\n";
    out << "This is symbolic only: no numeric block factorization, no triangular solve, "
           "and no hybrid NR run was performed.\n\n";

    out << "## Pattern Check\n\n";
    for (const std::string& note : pattern_notes) {
        out << "- " << note << '\n';
    }
    out << '\n';

    out << "## Block Counts\n\n";
    out << "| block size | min blocks | max blocks | mean blocks |\n";
    out << "|---:|---:|---:|---:|\n";
    for (int32_t block_size : {8, 16, 32}) {
        std::vector<double> counts;
        for (const GraphStatsRow& row : graph_rows) {
            if (row.iteration == 1 && row.block_size_target == block_size) {
                counts.push_back(row.num_blocks);
            }
        }
        if (counts.empty()) {
            continue;
        }
        const auto [min_it, max_it] = std::minmax_element(counts.begin(), counts.end());
        const double mean =
            std::accumulate(counts.begin(), counts.end(), 0.0) / static_cast<double>(counts.size());
        out << '|' << block_size << '|' << format_double(*min_it) << '|'
            << format_double(*max_it) << '|' << format_double(mean) << "|\n";
    }
    out << '\n';

    out << "## Ordering Summary\n\n";
    out << "| block size | ordering | mean L levels | mean U levels | mean avg width | "
           "mean apply / BJ | mean factor / BJ setup | rejects |\n";
    out << "|---:|---|---:|---:|---:|---:|---:|---:|\n";
    for (const AggregateCandidate& item : candidates) {
        out << '|' << item.block_size << '|' << item.ordering << '|'
            << format_double(item.mean_l_levels) << '|'
            << format_double(item.mean_u_levels) << '|'
            << format_double(item.mean_avg_width) << '|'
            << format_double(item.mean_apply_ratio) << '|'
            << format_double(item.mean_factor_ratio) << '|'
            << item.reject_count << "|\n";
    }
    out << '\n';

    out << "## Answers\n\n";
    if (!candidates.empty()) {
        out << "1. Shallowest levels: `" << candidates.front().ordering
            << "` at block size `" << candidates.front().block_size
            << "` has the best aggregate symbolic score.\n";
    } else {
        out << "1. Shallowest levels: no valid candidate was produced.\n";
    }

    auto widest = candidates;
    std::sort(widest.begin(), widest.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.mean_avg_width > rhs.mean_avg_width;
    });
    if (!widest.empty()) {
        out << "2. Widest levels: `" << widest.front().ordering
            << "` at block size `" << widest.front().block_size
            << "` has the largest mean level width.\n";
    } else {
        out << "2. Widest levels: no valid candidate was produced.\n";
    }

    const auto coloring_it = std::find_if(candidates.begin(), candidates.end(), [](const auto& x) {
        return x.ordering == "block_coloring";
    });
    if (coloring_it != candidates.end()) {
        out << "3. Coloring width: block coloring mean width is `"
            << format_double(coloring_it->mean_avg_width)
            << "`; compare with the table above before treating it as parallel enough.\n";
    } else {
        out << "3. Coloring width: block coloring was not available.\n";
    }

    const int32_t total_rejects = std::count_if(ordered_rows.begin(), ordered_rows.end(), [](const auto& row) {
        return row.verdict == "reject";
    });
    out << "4. Obviously infeasible rows: `" << total_rejects << "` of `"
        << ordered_rows.size()
        << "` candidates were classified as reject by level depth, width, and work ratio.\n";

    std::vector<AggregateCandidate> selected;
    std::unordered_set<int32_t> used_block_sizes;
    for (const AggregateCandidate& item : candidates) {
        if (selected.size() >= 3) {
            break;
        }
        if (item.reject_count > 0) {
            continue;
        }
        if (used_block_sizes.insert(item.block_size).second) {
            selected.push_back(item);
        }
    }
    if (selected.empty()) {
        out << "5. Numeric pilot recommendation: none. All aggregate candidates had at least "
               "one reject or did not clearly avoid deep/narrow triangular dependencies.\n";
    } else {
        out << "5. Numeric pilot recommendation: at most these candidates are worth considering:\n";
        for (const AggregateCandidate& item : selected) {
            out << "   - block_size=`" << item.block_size << "`, ordering=`" << item.ordering
                << "`, mean L/U levels=`" << format_double(item.mean_l_levels) << "/"
                << format_double(item.mean_u_levels) << "`, mean apply/BJ=`"
                << format_double(item.mean_apply_ratio) << "`, mean factor/BJ setup=`"
                << format_double(item.mean_factor_ratio) << "`\n";
        }
    }
}

struct CaseMatrices {
    std::unordered_map<int32_t, cuiter::CsrMatrix> matrices;
};

CaseMatrices load_case_matrices(const CliOptions& options, const std::string& case_name)
{
    CaseMatrices out;
    for (int32_t iter : options.iterations) {
        const std::filesystem::path path = jacobian_path(options.jf_root, case_name, iter);
        if (!std::filesystem::exists(path)) {
            if (options.allow_missing) {
                std::cerr << "[skip] missing " << path << '\n';
                continue;
            }
            throw std::runtime_error("missing Jacobian dump: " + path.string());
        }
        out.matrices.emplace(iter, load_cupf_csr_dump(path));
    }
    return out;
}

std::string pattern_note_for_case(const std::string& case_name, const CaseMatrices& matrices)
{
    const auto it1 = matrices.matrices.find(1);
    if (it1 == matrices.matrices.end()) {
        return case_name + ": J1 not loaded; pattern reference unavailable";
    }
    bool all_same = true;
    std::vector<std::string> checked;
    for (int32_t iter : {0, 1, 2}) {
        const auto it = matrices.matrices.find(iter);
        if (it == matrices.matrices.end()) {
            continue;
        }
        checked.push_back("J" + std::to_string(iter));
        if (!same_pattern(it1->second, it->second)) {
            all_same = false;
        }
    }
    std::ostringstream out;
    out << case_name << ": checked ";
    for (std::size_t i = 0; i < checked.size(); ++i) {
        if (i > 0) {
            out << '/';
        }
        out << checked[i];
    }
    out << "; pattern_same_as_J1=" << (all_same ? "true" : "false");
    return out.str();
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        std::filesystem::create_directories(options.output_dir);

        std::vector<GraphStatsRow> graph_rows;
        std::vector<OrderedAnalysis> ordered_rows;
        std::vector<std::string> pattern_notes;

        for (const std::string& case_name : options.cases) {
            std::cout << "[case] " << case_name << '\n';
            CaseMatrices case_matrices;
            try {
                case_matrices = load_case_matrices(options, case_name);
            } catch (const std::exception& ex) {
                if (options.allow_missing) {
                    std::cerr << "[skip] " << case_name << ": " << ex.what() << '\n';
                    continue;
                }
                throw;
            }
            if (case_matrices.matrices.empty()) {
                continue;
            }
            pattern_notes.push_back(pattern_note_for_case(case_name, case_matrices));

            auto ref_it = case_matrices.matrices.find(1);
            if (ref_it == case_matrices.matrices.end()) {
                ref_it = case_matrices.matrices.begin();
            }
            const cuiter::CsrMatrix& reference = ref_it->second;

            for (int32_t block_size : options.block_sizes) {
                std::cout << "  [block] target=" << block_size << '\n';
                const cuiter::MetisPermutation permutation =
                    cuiter::build_metis_permutation(reference, block_size);
                const cuiter::PermutedCsrPattern reference_pattern =
                    cuiter::build_permuted_csr_pattern(reference, permutation);
                const BlockGraph graph =
                    build_block_graph(reference, permutation, reference_pattern, block_size);

                for (const auto& item : case_matrices.matrices) {
                    if (!same_pattern(reference, item.second)) {
                        std::cerr << "  [warn] " << case_name << " J" << item.first
                                  << " pattern differs; graph stats still use its own values.\n";
                    }
                    graph_rows.push_back(make_graph_stats_row(case_name,
                                                              item.first,
                                                              item.second,
                                                              permutation,
                                                              block_size));
                }

                for (const std::string& ordering : options.orderings) {
                    OrderedAnalysis row = analyze_ordered_graph(case_name, graph, ordering);
                    ordered_rows.push_back(row);
                    std::cout << "    [ordering] " << row.ordering
                              << " verdict=" << row.verdict
                              << " L=" << row.l_levels.num_levels
                              << " U=" << row.u_levels.num_levels
                              << " apply/BJ=" << format_double(row.apply_work_relative_to_block_jacobi)
                              << '\n';
                }
            }
        }

        write_graph_stats_csv(options.output_dir / "block_ilu_symbolic_graph_stats.csv",
                              graph_rows);
        write_levels_csv(options.output_dir / "block_ilu_symbolic_levels.csv", ordered_rows);
        write_apply_work_csv(options.output_dir / "block_ilu_symbolic_apply_work.csv",
                             ordered_rows);
        write_factor_work_csv(options.output_dir / "block_ilu_symbolic_factor_work.csv",
                              ordered_rows);
        write_ranked_csv(options.output_dir / "block_ilu_symbolic_ranked.csv", ordered_rows);
        write_report(options.output_dir / "block_ilu_symbolic_report.md",
                     graph_rows,
                     ordered_rows,
                     pattern_notes);

        std::cout << "[done] wrote block ILU symbolic outputs to "
                  << options.output_dir << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
