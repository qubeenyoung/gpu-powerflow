#include "cupf_minimal/case_data.hpp"
#include "cupf_minimal/jacobian_analysis.hpp"

#include "cuiter/core/csr_matrix.hpp"
#include "cuiter/reorder/metis_partition.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path case_root = "/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps";
    std::filesystem::path output = "results/ras_overlap_symbolic.csv";
    std::vector<std::string> cases = {
        "case2383wp",
        "case3120sp",
        "case9241pegase",
        "case13659pegase",
        "case6468rte",
    };
    std::vector<int32_t> block_sizes = {8, 16};
    std::vector<int32_t> overlaps = {0, 1};
};

struct Row {
    std::string case_name;
    int32_t n = 0;
    int32_t nnz = 0;
    int32_t block_size = 0;
    int32_t overlap = 0;
    int32_t num_blocks = 0;
    int32_t min_owned_dim = 0;
    double avg_owned_dim = 0.0;
    int32_t max_owned_dim = 0;
    int32_t min_overlap_dim = 0;
    double avg_overlap_dim = 0.0;
    int32_t max_overlap_dim = 0;
    int32_t min_neighbor_count = 0;
    double avg_neighbor_count = 0.0;
    int32_t max_neighbor_count = 0;
    double overlap_dim_growth = 0.0;
    double estimated_dense_storage_mb = 0.0;
    double estimated_dense_storage_actual_mb = 0.0;
    double estimated_setup_work = 0.0;
    double estimated_apply_work = 0.0;
    int64_t local_nnz_total = 0;
    bool risk = false;
    std::string risk_reason;
};

std::vector<std::string> split_list(const std::string& value)
{
    std::vector<std::string> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }), item.end());
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

std::vector<int32_t> split_int_list(const std::string& value)
{
    std::vector<int32_t> out;
    for (const std::string& item : split_list(value)) {
        out.push_back(std::stoi(item));
    }
    if (out.empty()) {
        throw std::runtime_error("empty integer list");
    }
    return out;
}

void print_usage(const char* argv0)
{
    std::cout << "Usage: " << argv0 << " [options]\n\n"
              << "Options:\n"
              << "  --case-root PATH\n"
              << "  --case NAME[,NAME...]\n"
              << "  --block-size INT[,INT...]\n"
              << "  --overlap 0,1\n"
              << "  --output PATH\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-root" && i + 1 < argc) {
            options.case_root = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            options.cases = split_list(argv[++i]);
        } else if (arg == "--block-size" && i + 1 < argc) {
            options.block_sizes = split_int_list(argv[++i]);
        } else if (arg == "--overlap" && i + 1 < argc) {
            options.overlaps = split_int_list(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            options.output = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.cases.empty() || options.block_sizes.empty() || options.overlaps.empty()) {
        throw std::runtime_error("case, block-size, and overlap lists must be non-empty");
    }
    return options;
}

void ensure_parent_dir(const std::filesystem::path& path)
{
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

cuiter::CsrMatrix build_jacobian_pattern_matrix(const cupf_minimal::DumpCaseData& data)
{
    const cupf_minimal::YbusView ybus = data.ybus();
    const cupf_minimal::JacobianIndexing indexing =
        cupf_minimal::make_jacobian_indexing(data.rows,
                                             data.pv.data(),
                                             static_cast<int32_t>(data.pv.size()),
                                             data.pq.data(),
                                             static_cast<int32_t>(data.pq.size()));
    const cupf_minimal::JacobianPattern pattern =
        cupf_minimal::JacobianPatternGenerator().generate(ybus, indexing);

    cuiter::CsrMatrix matrix;
    matrix.rows = pattern.dim;
    matrix.cols = pattern.dim;
    matrix.row_ptr = pattern.row_ptr;
    matrix.col_idx = pattern.col_idx;
    matrix.values.assign(static_cast<std::size_t>(pattern.nnz), 1.0);
    return matrix;
}

uint64_t edge_key(int32_t row_block, int32_t col_block)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(row_block)) << 32U) |
           static_cast<uint32_t>(col_block);
}

std::vector<int32_t> build_block_of_new(const cuiter::MetisPermutation& permutation,
                                        int32_t n)
{
    std::vector<int32_t> block_of_new(static_cast<std::size_t>(n), -1);
    for (int32_t block = 0; block < static_cast<int32_t>(permutation.block_sizes.size()); ++block) {
        const int32_t start = permutation.block_starts[static_cast<std::size_t>(block)];
        const int32_t size = permutation.block_sizes[static_cast<std::size_t>(block)];
        for (int32_t local = 0; local < size; ++local) {
            block_of_new[static_cast<std::size_t>(start + local)] = block;
        }
    }
    for (int32_t block : block_of_new) {
        if (block < 0) {
            throw std::runtime_error("unassigned METIS block");
        }
    }
    return block_of_new;
}

Row analyze_case_block(const cupf_minimal::DumpCaseData& data,
                       const cuiter::CsrMatrix& matrix,
                       int32_t block_size,
                       int32_t overlap)
{
    if (overlap != 0 && overlap != 1) {
        throw std::runtime_error("only overlap 0 and 1 are supported");
    }

    const cuiter::MetisPermutation permutation = cuiter::build_metis_permutation(matrix, block_size);
    const std::vector<int32_t> block_of_new = build_block_of_new(permutation, matrix.rows);
    const int32_t num_blocks = static_cast<int32_t>(permutation.block_sizes.size());

    std::vector<std::unordered_set<int32_t>> neighbors(static_cast<std::size_t>(num_blocks));
    std::unordered_map<uint64_t, int64_t> scalar_block_nnz;
    scalar_block_nnz.reserve(static_cast<std::size_t>(matrix.nnz()) * 2U);

    for (int32_t row = 0; row < matrix.rows; ++row) {
        const int32_t new_row = permutation.old_to_new[static_cast<std::size_t>(row)];
        const int32_t row_block = block_of_new[static_cast<std::size_t>(new_row)];
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = matrix.col_idx[static_cast<std::size_t>(pos)];
            const int32_t new_col = permutation.old_to_new[static_cast<std::size_t>(col)];
            const int32_t col_block = block_of_new[static_cast<std::size_t>(new_col)];
            ++scalar_block_nnz[edge_key(row_block, col_block)];
            if (row_block != col_block) {
                neighbors[static_cast<std::size_t>(row_block)].insert(col_block);
                neighbors[static_cast<std::size_t>(col_block)].insert(row_block);
            }
        }
    }

    Row row;
    row.case_name = data.case_name;
    row.n = matrix.rows;
    row.nnz = matrix.nnz();
    row.block_size = block_size;
    row.overlap = overlap;
    row.num_blocks = num_blocks;
    row.min_owned_dim = matrix.rows;
    row.min_overlap_dim = matrix.rows;
    row.min_neighbor_count = num_blocks;

    double sum_owned = 0.0;
    double sum_overlap = 0.0;
    double sum_neighbors = 0.0;
    double actual_dense_entries = 0.0;

    for (int32_t block = 0; block < num_blocks; ++block) {
        std::vector<int32_t> omega;
        omega.push_back(block);
        if (overlap == 1) {
            for (int32_t neighbor : neighbors[static_cast<std::size_t>(block)]) {
                omega.push_back(neighbor);
            }
        }
        std::sort(omega.begin() + 1, omega.end());

        const int32_t owned_dim = permutation.block_sizes[static_cast<std::size_t>(block)];
        int32_t overlap_dim = 0;
        for (int32_t omega_block : omega) {
            overlap_dim += permutation.block_sizes[static_cast<std::size_t>(omega_block)];
        }
        const int32_t neighbor_count = static_cast<int32_t>(omega.size()) - 1;

        int64_t local_nnz = 0;
        for (int32_t rb : omega) {
            for (int32_t cb : omega) {
                const auto it = scalar_block_nnz.find(edge_key(rb, cb));
                if (it != scalar_block_nnz.end()) {
                    local_nnz += it->second;
                }
            }
        }

        row.min_owned_dim = std::min(row.min_owned_dim, owned_dim);
        row.max_owned_dim = std::max(row.max_owned_dim, owned_dim);
        row.min_overlap_dim = std::min(row.min_overlap_dim, overlap_dim);
        row.max_overlap_dim = std::max(row.max_overlap_dim, overlap_dim);
        row.min_neighbor_count = std::min(row.min_neighbor_count, neighbor_count);
        row.max_neighbor_count = std::max(row.max_neighbor_count, neighbor_count);
        sum_owned += owned_dim;
        sum_overlap += overlap_dim;
        sum_neighbors += neighbor_count;
        actual_dense_entries += static_cast<double>(overlap_dim) * static_cast<double>(overlap_dim);
        row.estimated_setup_work +=
            static_cast<double>(overlap_dim) * static_cast<double>(overlap_dim) *
            static_cast<double>(overlap_dim);
        row.estimated_apply_work +=
            static_cast<double>(overlap_dim) * static_cast<double>(overlap_dim);
        row.local_nnz_total += local_nnz;
    }

    const double denom = std::max(1, num_blocks);
    row.avg_owned_dim = sum_owned / denom;
    row.avg_overlap_dim = sum_overlap / denom;
    row.avg_neighbor_count = sum_neighbors / denom;
    row.overlap_dim_growth =
        row.avg_owned_dim > 0.0 ? row.avg_overlap_dim / row.avg_owned_dim : 0.0;
    row.estimated_dense_storage_mb =
        static_cast<double>(num_blocks) * static_cast<double>(row.max_overlap_dim) *
        static_cast<double>(row.max_overlap_dim) * sizeof(float) / (1024.0 * 1024.0);
    row.estimated_dense_storage_actual_mb =
        actual_dense_entries * sizeof(float) / (1024.0 * 1024.0);
    if (row.max_overlap_dim > 1024) {
        row.risk = true;
        row.risk_reason = "max_overlap_dim_gt_1024";
    } else if (row.max_overlap_dim > 512) {
        row.risk = true;
        row.risk_reason = "max_overlap_dim_gt_512";
    } else if (row.estimated_dense_storage_mb > 2048.0) {
        row.risk = true;
        row.risk_reason = "padded_dense_storage_gt_2gb";
    } else {
        row.risk = false;
        row.risk_reason = "ok";
    }
    return row;
}

void write_csv(const std::filesystem::path& path, const std::vector<Row>& rows)
{
    ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path.string());
    }
    out << std::setprecision(12);
    out << "case,n,nnz,block_size,overlap,num_blocks,min_owned_dim,avg_owned_dim,"
           "max_owned_dim,min_overlap_dim,avg_overlap_dim,max_overlap_dim,"
           "min_neighbor_count,avg_neighbor_count,max_neighbor_count,"
           "overlap_dim_growth,estimated_dense_storage_mb,"
           "estimated_dense_storage_actual_mb,estimated_setup_work,estimated_apply_work,"
           "local_nnz_total,risk,risk_reason\n";
    for (const Row& row : rows) {
        out << row.case_name << ','
            << row.n << ','
            << row.nnz << ','
            << row.block_size << ','
            << row.overlap << ','
            << row.num_blocks << ','
            << row.min_owned_dim << ','
            << row.avg_owned_dim << ','
            << row.max_owned_dim << ','
            << row.min_overlap_dim << ','
            << row.avg_overlap_dim << ','
            << row.max_overlap_dim << ','
            << row.min_neighbor_count << ','
            << row.avg_neighbor_count << ','
            << row.max_neighbor_count << ','
            << row.overlap_dim_growth << ','
            << row.estimated_dense_storage_mb << ','
            << row.estimated_dense_storage_actual_mb << ','
            << row.estimated_setup_work << ','
            << row.estimated_apply_work << ','
            << row.local_nnz_total << ','
            << std::boolalpha << row.risk << ','
            << row.risk_reason << '\n';
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        std::vector<Row> rows;
        for (const std::string& case_name : options.cases) {
            const cupf_minimal::DumpCaseData data =
                cupf_minimal::load_dump_case(options.case_root / case_name);
            const cuiter::CsrMatrix matrix = build_jacobian_pattern_matrix(data);
            for (int32_t block_size : options.block_sizes) {
                for (int32_t overlap : options.overlaps) {
                    rows.push_back(analyze_case_block(data, matrix, block_size, overlap));
                    const Row& row = rows.back();
                    std::cout << "[symbolic] " << case_name
                              << " bs=" << block_size
                              << " overlap=" << overlap
                              << " avg_omega=" << row.avg_overlap_dim
                              << " max_omega=" << row.max_overlap_dim
                              << " risk=" << row.risk_reason << '\n';
                }
            }
        }
        write_csv(options.output, rows);
        std::cout << "[done] wrote " << options.output << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "ras_overlap_symbolic_analyzer failed: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
