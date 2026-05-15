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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path case_root = "/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps";
    std::vector<std::string> cases;
    std::vector<int32_t> block_sizes = {64};
    std::filesystem::path output = "results/metis_block_pq_theta_vm_composition.csv";
    std::filesystem::path summary_output = "results/metis_block_pq_theta_vm_summary.md";
};

struct RegionCounts {
    int64_t j11_p_theta = 0;
    int64_t j12_p_vmag = 0;
    int64_t j21_q_theta = 0;
    int64_t j22_q_vmag = 0;

    int64_t total() const
    {
        return j11_p_theta + j12_p_vmag + j21_q_theta + j22_q_vmag;
    }
};

struct CaseBlockStats {
    std::string case_name;
    int32_t buses = 0;
    int32_t n = 0;
    int32_t nnz = 0;
    int32_t block_size = 0;
    int32_t num_blocks = 0;
    int32_t min_block_size = 0;
    int32_t max_block_size = 0;
    double avg_block_size = 0.0;
    double offblock_nnz_ratio = 0.0;
    RegionCounts diagonal;
    RegionCounts offdiagonal;
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
    for (const auto& item : split_list(value)) {
        out.push_back(std::stoi(item));
    }
    if (out.empty()) {
        throw std::runtime_error("empty integer list");
    }
    return out;
}

void print_usage(const char* argv0)
{
    std::cout << "Usage: " << argv0 << " --case caseA,caseB [options]\n\n"
              << "Options:\n"
              << "  --case-root PATH\n"
              << "  --case NAME[,NAME...]\n"
              << "  --block-size INT[,INT...]\n"
              << "  --output PATH\n"
              << "  --summary-output PATH\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-root" && i + 1 < argc) {
            options.case_root = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            const auto parsed = split_list(argv[++i]);
            options.cases.insert(options.cases.end(), parsed.begin(), parsed.end());
        } else if (arg == "--block-size" && i + 1 < argc) {
            options.block_sizes = split_int_list(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            options.output = argv[++i];
        } else if (arg == "--summary-output" && i + 1 < argc) {
            options.summary_output = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.cases.empty()) {
        throw std::runtime_error("--case is required");
    }
    return options;
}

void ensure_parent_dir(const std::filesystem::path& path)
{
    const auto parent = path.parent_path();
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

std::vector<int32_t> build_block_ids(const cuiter::MetisPermutation& permutation, int32_t n)
{
    std::vector<int32_t> block_of_new(static_cast<std::size_t>(n), -1);
    for (int32_t block = 0; block < static_cast<int32_t>(permutation.block_starts.size()); ++block) {
        const int32_t start = permutation.block_starts[static_cast<std::size_t>(block)];
        const int32_t size = permutation.block_sizes[static_cast<std::size_t>(block)];
        for (int32_t local = 0; local < size; ++local) {
            block_of_new[static_cast<std::size_t>(start + local)] = block;
        }
    }
    for (int32_t value : block_of_new) {
        if (value < 0) {
            throw std::runtime_error("unassigned METIS block id");
        }
    }
    return block_of_new;
}

void add_quadrant_count(RegionCounts& counts,
                        int32_t row,
                        int32_t col,
                        int32_t n_pvpq)
{
    const bool row_is_p = row < n_pvpq;
    const bool col_is_theta = col < n_pvpq;
    if (row_is_p && col_is_theta) {
        ++counts.j11_p_theta;
    } else if (row_is_p && !col_is_theta) {
        ++counts.j12_p_vmag;
    } else if (!row_is_p && col_is_theta) {
        ++counts.j21_q_theta;
    } else {
        ++counts.j22_q_vmag;
    }
}

CaseBlockStats analyze_case_block(const cupf_minimal::DumpCaseData& data, int32_t block_size)
{
    const cuiter::CsrMatrix matrix = build_jacobian_pattern_matrix(data);
    const int32_t n_pq = static_cast<int32_t>(data.pq.size());
    const int32_t n_pvpq = static_cast<int32_t>(data.pv.size() + data.pq.size());

    const cuiter::MetisPermutation permutation = cuiter::build_metis_permutation(matrix, block_size);
    const std::vector<int32_t> block_of_new = build_block_ids(permutation, matrix.rows);

    CaseBlockStats stats;
    stats.case_name = data.case_name;
    stats.buses = data.rows;
    stats.n = matrix.rows;
    stats.nnz = matrix.nnz();
    stats.block_size = block_size;
    stats.num_blocks = permutation.stats.num_blocks;
    stats.min_block_size = permutation.stats.min_block_size;
    stats.max_block_size = permutation.stats.max_block_size;
    stats.avg_block_size = permutation.stats.avg_block_size;
    stats.offblock_nnz_ratio = permutation.stats.offblock_nnz_ratio;

    if (matrix.rows != n_pvpq + n_pq) {
        throw std::runtime_error("Jacobian dimension does not match P/Q indexing");
    }

    for (int32_t row = 0; row < matrix.rows; ++row) {
        const int32_t new_row = permutation.old_to_new[static_cast<std::size_t>(row)];
        const int32_t row_block = block_of_new[static_cast<std::size_t>(new_row)];
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = matrix.col_idx[static_cast<std::size_t>(pos)];
            const int32_t new_col = permutation.old_to_new[static_cast<std::size_t>(col)];
            const int32_t col_block = block_of_new[static_cast<std::size_t>(new_col)];
            RegionCounts& counts =
                row_block == col_block ? stats.diagonal : stats.offdiagonal;
            add_quadrant_count(counts, row, col, n_pvpq);
        }
    }
    return stats;
}

double fraction(int64_t count, int64_t total)
{
    return total > 0 ? static_cast<double>(count) / static_cast<double>(total) : 0.0;
}

void write_region_row(std::ostream& out,
                      const CaseBlockStats& stats,
                      const std::string& region,
                      const RegionCounts& counts)
{
    const int64_t region_total = counts.total();
    const int64_t total_nnz = stats.diagonal.total() + stats.offdiagonal.total();
    const int64_t row_p = counts.j11_p_theta + counts.j12_p_vmag;
    const int64_t row_q = counts.j21_q_theta + counts.j22_q_vmag;
    const int64_t col_theta = counts.j11_p_theta + counts.j21_q_theta;
    const int64_t col_vmag = counts.j12_p_vmag + counts.j22_q_vmag;

    out << std::setprecision(12)
        << stats.case_name << ','
        << stats.buses << ','
        << stats.n << ','
        << stats.nnz << ','
        << stats.block_size << ','
        << stats.num_blocks << ','
        << stats.min_block_size << ','
        << stats.max_block_size << ','
        << stats.avg_block_size << ','
        << stats.offblock_nnz_ratio << ','
        << region << ','
        << region_total << ','
        << fraction(region_total, total_nnz) << ','
        << row_p << ','
        << row_q << ','
        << col_theta << ','
        << col_vmag << ','
        << fraction(row_p, region_total) << ','
        << fraction(row_q, region_total) << ','
        << fraction(col_theta, region_total) << ','
        << fraction(col_vmag, region_total) << ','
        << counts.j11_p_theta << ','
        << counts.j12_p_vmag << ','
        << counts.j21_q_theta << ','
        << counts.j22_q_vmag << ','
        << fraction(counts.j11_p_theta, region_total) << ','
        << fraction(counts.j12_p_vmag, region_total) << ','
        << fraction(counts.j21_q_theta, region_total) << ','
        << fraction(counts.j22_q_vmag, region_total) << '\n';
}

void write_csv(const std::vector<CaseBlockStats>& rows, const std::filesystem::path& path)
{
    ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output CSV: " + path.string());
    }
    out << "case,buses,n,nnz,block_size,num_blocks,min_block_size,max_block_size,"
           "avg_block_size,offblock_nnz_ratio,region,region_nnz,region_nnz_fraction,"
           "row_P_nnz,row_Q_nnz,col_theta_nnz,col_Vm_nnz,row_P_fraction,row_Q_fraction,"
           "col_theta_fraction,col_Vm_fraction,J11_P_theta_nnz,J12_P_Vm_nnz,"
           "J21_Q_theta_nnz,J22_Q_Vm_nnz,J11_P_theta_fraction,J12_P_Vm_fraction,"
           "J21_Q_theta_fraction,J22_Q_Vm_fraction\n";
    for (const auto& row : rows) {
        write_region_row(out, row, "diagonal", row.diagonal);
        write_region_row(out, row, "offdiagonal", row.offdiagonal);
    }
}

void write_summary(const std::vector<CaseBlockStats>& rows, const std::filesystem::path& path)
{
    ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open summary markdown: " + path.string());
    }
    out << "# METIS Block P/Q Theta/Vm Composition\n\n";
    out << "- Region rule: `diagonal` means row and column unknowns are in the same METIS block; "
           "`offdiagonal` means they are in different blocks.\n";
    out << "- Row types: P rows are the first `n_pvpq` rows, Q rows are the final `n_pq` rows.\n";
    out << "- Column types: theta columns are the first `n_pvpq` columns, |V| columns are the final `n_pq` columns.\n\n";

    out << "| case | block | offblock nnz | diag row P | diag col theta | diag J11 | diag J22 | off row P | off col theta | off J11 | off J22 |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& row : rows) {
        const int64_t diag_total = row.diagonal.total();
        const int64_t off_total = row.offdiagonal.total();
        const int64_t diag_row_p = row.diagonal.j11_p_theta + row.diagonal.j12_p_vmag;
        const int64_t diag_col_theta = row.diagonal.j11_p_theta + row.diagonal.j21_q_theta;
        const int64_t off_row_p = row.offdiagonal.j11_p_theta + row.offdiagonal.j12_p_vmag;
        const int64_t off_col_theta = row.offdiagonal.j11_p_theta + row.offdiagonal.j21_q_theta;
        out << "| " << row.case_name
            << " | " << row.block_size
            << " | " << std::fixed << std::setprecision(3) << row.offblock_nnz_ratio
            << " | " << fraction(diag_row_p, diag_total)
            << " | " << fraction(diag_col_theta, diag_total)
            << " | " << fraction(row.diagonal.j11_p_theta, diag_total)
            << " | " << fraction(row.diagonal.j22_q_vmag, diag_total)
            << " | " << fraction(off_row_p, off_total)
            << " | " << fraction(off_col_theta, off_total)
            << " | " << fraction(row.offdiagonal.j11_p_theta, off_total)
            << " | " << fraction(row.offdiagonal.j22_q_vmag, off_total)
            << " |\n";
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        std::vector<CaseBlockStats> rows;
        for (const auto& case_name : options.cases) {
            const cupf_minimal::DumpCaseData data =
                cupf_minimal::load_dump_case(options.case_root / case_name);
            for (int32_t block_size : options.block_sizes) {
                rows.push_back(analyze_case_block(data, block_size));
                const auto& row = rows.back();
                std::cout << "[OK] case=" << row.case_name
                          << " block=" << row.block_size
                          << " n=" << row.n
                          << " nnz=" << row.nnz
                          << " offblock=" << row.offblock_nnz_ratio << '\n';
            }
        }
        write_csv(rows, options.output);
        write_summary(rows, options.summary_output);
        std::cout << "[DONE] output=" << options.output << '\n'
                  << "[DONE] summary=" << options.summary_output << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "metis_block_composition failed: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
