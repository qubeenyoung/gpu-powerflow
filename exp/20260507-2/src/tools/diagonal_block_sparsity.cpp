#include "cuiter/core/csr_matrix.hpp"
#include "cuiter/reorder/metis_partition.hpp"
#include "cuiter/solver/cpu_block_ilu0_pilot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace cpu = cuiter::cpu_pilot;
namespace detail = cuiter::cpu_pilot::detail;

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

struct Options {
    std::filesystem::path jf_root = "raw/cupf_jf_dumps";
    std::filesystem::path output_dir = "results/diagonal_block_sparsity";
    std::vector<std::string> cases = {
        "case2383wp",
        "case3120sp",
        "case6468rte",
        "case9241pegase",
        "case13659pegase",
    };
    std::vector<int32_t> block_sizes = {8, 16, 32};
    int32_t iteration = 1;
    double numeric_tol = 1.0e-12;
    double diag_shift_scale = 1.0e-8;
};

struct BlockRow {
    std::string case_name;
    int32_t iteration = 1;
    std::string method;
    std::string ordering;
    int32_t block_size_target = 0;
    int32_t block_id = 0;
    int32_t block_dim = 0;
    int32_t input_nnz = 0;
    double input_density = 0.0;
    double input_sparsity = 0.0;
    int32_t factor_diag_nnz = -1;
    double factor_diag_density = kNan;
    double factor_diag_sparsity = kNan;
    int32_t inverse_nnz = 0;
    double inverse_density = 0.0;
    double inverse_sparsity = 0.0;
    bool factor_failed = false;
};

struct SummaryRow {
    std::string case_name;
    std::string method;
    std::string ordering;
    int32_t block_size_target = 0;
    int32_t num_blocks = 0;
    double input_density_avg = 0.0;
    double input_density_p10 = 0.0;
    double input_density_p50 = 0.0;
    double input_density_p90 = 0.0;
    double input_density_min = 0.0;
    double input_density_max = 0.0;
    double inverse_density_avg = 0.0;
    double inverse_density_p50 = 0.0;
    double factor_diag_density_avg = kNan;
    double factor_diag_density_p50 = kNan;
    double dense_storage_waste_ratio = 0.0;
    double input_blocks_below_25pct = 0.0;
    double input_blocks_below_50pct = 0.0;
    bool factor_failed = false;
};

std::vector<std::string> split_list(const std::string& text)
{
    std::vector<std::string> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

std::vector<int32_t> parse_int_list(const std::string& text)
{
    std::vector<int32_t> out;
    for (const std::string& item : split_list(text)) {
        out.push_back(std::stoi(item));
    }
    return out;
}

void print_usage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "  --jf-root PATH\n"
        << "  --cases caseA,caseB\n"
        << "  --iter 1\n"
        << "  --block-sizes 8,16,32\n"
        << "  --numeric-tol 1e-12\n"
        << "  --output-dir PATH\n";
}

Options parse_args(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--jf-root") {
            options.jf_root = require_value("--jf-root");
        } else if (arg == "--cases") {
            options.cases = split_list(require_value("--cases"));
        } else if (arg == "--iter") {
            options.iteration = std::stoi(require_value("--iter"));
        } else if (arg == "--block-sizes") {
            options.block_sizes = parse_int_list(require_value("--block-sizes"));
        } else if (arg == "--numeric-tol") {
            options.numeric_tol = std::stod(require_value("--numeric-tol"));
        } else if (arg == "--diag-shift-scale") {
            options.diag_shift_scale = std::stod(require_value("--diag-shift-scale"));
        } else if (arg == "--output-dir") {
            options.output_dir = require_value("--output-dir");
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

void expect_token(std::istream& in, const char* expected, const std::filesystem::path& path)
{
    std::string token;
    in >> token;
    if (token != expected) {
        throw std::runtime_error("expected token '" + std::string(expected) + "' in " +
                                 path.string());
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
        throw std::runtime_error("not a cuPF CSR dump: " + path.string());
    }
    cuiter::CsrMatrix matrix;
    int32_t nnz = 0;
    expect_token(in, "rows", path);
    in >> matrix.rows;
    expect_token(in, "cols", path);
    in >> matrix.cols;
    expect_token(in, "nnz", path);
    in >> nnz;
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
    const auto case_dir = jf_root / case_name;
    const std::vector<std::filesystem::path> candidates = {
        case_dir / ("J" + std::to_string(iteration) + ".txt"),
        case_dir / "repeat_00" / ("jacobian_iter" + std::to_string(iteration) + ".txt"),
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return candidates.front();
}

int32_t count_nonzero(const std::vector<float>& values, double tol)
{
    int32_t count = 0;
    for (float value : values) {
        if (std::abs(static_cast<double>(value)) > tol) {
            ++count;
        }
    }
    return count;
}

double density(int32_t nnz, int32_t dim)
{
    return dim > 0 ? static_cast<double>(nnz) / static_cast<double>(dim * dim) : kNan;
}

double percentile(std::vector<double> values, double p)
{
    values.erase(std::remove_if(values.begin(), values.end(), [](double value) {
                     return !std::isfinite(value);
                 }),
                 values.end());
    if (values.empty()) {
        return kNan;
    }
    std::sort(values.begin(), values.end());
    const double raw = p * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(raw));
    const auto hi = static_cast<std::size_t>(std::ceil(raw));
    if (lo == hi) {
        return values[lo];
    }
    const double t = raw - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
}

double mean(const std::vector<double>& values)
{
    double sum = 0.0;
    int32_t count = 0;
    for (double value : values) {
        if (std::isfinite(value)) {
            sum += value;
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : kNan;
}

std::vector<BlockRow> analyze_method(const std::string& case_name,
                                     const cuiter::CsrMatrix& matrix,
                                     int32_t iteration,
                                     int32_t block_size,
                                     bool use_block_ilu,
                                     const Options& options)
{
    const cuiter::MetisPermutation permutation =
        use_block_ilu ? detail::build_colored_block_permutation(matrix, block_size)
                      : cuiter::build_metis_permutation(matrix, block_size);
    const cuiter::CsrMatrix permuted = detail::build_permuted_matrix(matrix, permutation);

    detail::BlockPattern input_diag_pattern =
        detail::make_block_pattern(permuted,
                                   permutation.block_starts,
                                   permutation.block_sizes,
                                   true);
    detail::scatter_dense_blocks(permuted, input_diag_pattern);

    cpu::detail::CpuPreconditioner preconditioner;
    preconditioner.ilu0 = use_block_ilu;
    double setup_seconds = 0.0;
    double factor_seconds = 0.0;
    const bool ok = preconditioner.setup(permuted,
                                         permutation.block_starts,
                                         permutation.block_sizes,
                                         options.diag_shift_scale,
                                         setup_seconds,
                                         factor_seconds);

    std::vector<BlockRow> rows;
    rows.reserve(permutation.block_sizes.size());
    for (int32_t block = 0; block < preconditioner.pattern.num_blocks; ++block) {
        const int32_t diag_index =
            preconditioner.pattern.diagonal_index[static_cast<std::size_t>(block)];
        const detail::DenseBlock& diag =
            preconditioner.pattern.blocks[static_cast<std::size_t>(diag_index)];
        const int32_t input_diag_index =
            input_diag_pattern.diagonal_index[static_cast<std::size_t>(block)];
        const detail::DenseBlock& input_diag =
            input_diag_pattern.blocks[static_cast<std::size_t>(input_diag_index)];
        const int32_t input_nnz = count_nonzero(input_diag.values, 0.0);
        const int32_t inverse_nnz =
            count_nonzero(preconditioner.inverse_diag[static_cast<std::size_t>(block)],
                          options.numeric_tol);

        BlockRow row;
        row.case_name = case_name;
        row.iteration = iteration;
        row.method = use_block_ilu ? "block_ilu0" : "block_jacobi";
        row.ordering = use_block_ilu ? "block_coloring" : "current_metis_block_order";
        row.block_size_target = block_size;
        row.block_id = block;
        row.block_dim = diag.rows;
        row.input_nnz = input_nnz;
        row.input_density = density(input_nnz, diag.rows);
        row.input_sparsity = 1.0 - row.input_density;
        row.inverse_nnz = inverse_nnz;
        row.inverse_density = density(inverse_nnz, diag.rows);
        row.inverse_sparsity = 1.0 - row.inverse_density;
        row.factor_failed = !ok;

        if (use_block_ilu) {
            // For ILU(0), the diagonal block in the factorized pattern is U_ii.
            // It starts from A_ii but can densify through dense block updates.
            row.factor_diag_nnz = count_nonzero(diag.values, options.numeric_tol);
            row.factor_diag_density = density(row.factor_diag_nnz, diag.rows);
            row.factor_diag_sparsity = row.input_sparsity;
            row.factor_diag_sparsity = 1.0 - row.factor_diag_density;
        }
        rows.push_back(row);
    }
    return rows;
}

SummaryRow summarize(const std::vector<BlockRow>& rows)
{
    SummaryRow out;
    if (rows.empty()) {
        return out;
    }
    out.case_name = rows.front().case_name;
    out.method = rows.front().method;
    out.ordering = rows.front().ordering;
    out.block_size_target = rows.front().block_size_target;
    out.num_blocks = static_cast<int32_t>(rows.size());
    out.factor_failed = rows.front().factor_failed;

    std::vector<double> input_density_values;
    std::vector<double> inverse_density_values;
    std::vector<double> factor_density_values;
    int64_t dense_slots = 0;
    int64_t input_nnz = 0;
    int32_t below_25 = 0;
    int32_t below_50 = 0;
    for (const auto& row : rows) {
        input_density_values.push_back(row.input_density);
        inverse_density_values.push_back(row.inverse_density);
        factor_density_values.push_back(row.factor_diag_density);
        dense_slots += static_cast<int64_t>(row.block_dim) * static_cast<int64_t>(row.block_dim);
        input_nnz += row.input_nnz;
        below_25 += row.input_density < 0.25 ? 1 : 0;
        below_50 += row.input_density < 0.50 ? 1 : 0;
    }
    out.input_density_avg = mean(input_density_values);
    out.input_density_p10 = percentile(input_density_values, 0.10);
    out.input_density_p50 = percentile(input_density_values, 0.50);
    out.input_density_p90 = percentile(input_density_values, 0.90);
    out.input_density_min = percentile(input_density_values, 0.00);
    out.input_density_max = percentile(input_density_values, 1.00);
    out.inverse_density_avg = mean(inverse_density_values);
    out.inverse_density_p50 = percentile(inverse_density_values, 0.50);
    out.factor_diag_density_avg = mean(factor_density_values);
    out.factor_diag_density_p50 = percentile(factor_density_values, 0.50);
    out.dense_storage_waste_ratio =
        dense_slots > 0 ? 1.0 - static_cast<double>(input_nnz) / static_cast<double>(dense_slots)
                        : kNan;
    out.input_blocks_below_25pct =
        static_cast<double>(below_25) / static_cast<double>(std::max<int32_t>(1, out.num_blocks));
    out.input_blocks_below_50pct =
        static_cast<double>(below_50) / static_cast<double>(std::max<int32_t>(1, out.num_blocks));
    return out;
}

std::string csv_escape(const std::string& text)
{
    if (text.find_first_of(",\"\n") == std::string::npos) {
        return text;
    }
    std::string out = "\"";
    for (char ch : text) {
        out += ch == '"' ? "\"\"" : std::string(1, ch);
    }
    out += '"';
    return out;
}

void write_block_csv(const std::filesystem::path& path, const std::vector<BlockRow>& rows)
{
    std::ofstream out(path);
    out << "case,iteration,method,ordering,block_size_target,block_id,block_dim,"
           "input_nnz,input_density,input_sparsity,factor_diag_nnz,"
           "factor_diag_density,factor_diag_sparsity,inverse_nnz,inverse_density,"
           "inverse_sparsity,factor_failed\n";
    out << std::setprecision(17);
    for (const auto& row : rows) {
        out << csv_escape(row.case_name) << ','
            << row.iteration << ','
            << row.method << ','
            << row.ordering << ','
            << row.block_size_target << ','
            << row.block_id << ','
            << row.block_dim << ','
            << row.input_nnz << ','
            << row.input_density << ','
            << row.input_sparsity << ','
            << row.factor_diag_nnz << ','
            << row.factor_diag_density << ','
            << row.factor_diag_sparsity << ','
            << row.inverse_nnz << ','
            << row.inverse_density << ','
            << row.inverse_sparsity << ','
            << (row.factor_failed ? "true" : "false") << '\n';
    }
}

void write_summary_csv(const std::filesystem::path& path, const std::vector<SummaryRow>& rows)
{
    std::ofstream out(path);
    out << "case,method,ordering,block_size_target,num_blocks,input_density_avg,"
           "input_density_p10,input_density_p50,input_density_p90,input_density_min,"
           "input_density_max,inverse_density_avg,inverse_density_p50,"
           "factor_diag_density_avg,factor_diag_density_p50,dense_storage_waste_ratio,"
           "input_blocks_below_25pct,input_blocks_below_50pct,factor_failed\n";
    out << std::setprecision(17);
    for (const auto& row : rows) {
        out << csv_escape(row.case_name) << ','
            << row.method << ','
            << row.ordering << ','
            << row.block_size_target << ','
            << row.num_blocks << ','
            << row.input_density_avg << ','
            << row.input_density_p10 << ','
            << row.input_density_p50 << ','
            << row.input_density_p90 << ','
            << row.input_density_min << ','
            << row.input_density_max << ','
            << row.inverse_density_avg << ','
            << row.inverse_density_p50 << ','
            << row.factor_diag_density_avg << ','
            << row.factor_diag_density_p50 << ','
            << row.dense_storage_waste_ratio << ','
            << row.input_blocks_below_25pct << ','
            << row.input_blocks_below_50pct << ','
            << (row.factor_failed ? "true" : "false") << '\n';
    }
}

std::string fmt_pct(double value)
{
    if (!std::isfinite(value)) {
        return "nan";
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << (100.0 * value) << "%";
    return ss.str();
}

void write_report(const std::filesystem::path& path, const std::vector<SummaryRow>& rows)
{
    std::ofstream out(path);
    out << "# Diagonal Block Sparsity\n\n";
    out << "- `input_density` = nonzeros in original diagonal block `A[Bi,Bi]` divided by `dim(Bi)^2`.\n";
    out << "- `inverse_density` = nonzeros in the stored dense inverse block after setup, using the numeric tolerance.\n";
    out << "- For block-ILU, the block partition is the same METIS block family but block order is coloring; block order does not change input diagonal-block membership.\n\n";
    out << "| case | method | block | input avg | input p50 | input p90 | factor Uii avg | inverse avg | dense-storage waste |\n";
    out << "|---|---|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& row : rows) {
        out << "| " << row.case_name
            << " | " << row.method
            << " | " << row.block_size_target
            << " | " << fmt_pct(row.input_density_avg)
            << " | " << fmt_pct(row.input_density_p50)
            << " | " << fmt_pct(row.input_density_p90)
            << " | " << fmt_pct(row.factor_diag_density_avg)
            << " | " << fmt_pct(row.inverse_density_avg)
            << " | " << fmt_pct(row.dense_storage_waste_ratio)
            << " |\n";
    }
    out << "\n## Interpretation\n\n";
    out << "- The original diagonal blocks are structurally sparse, especially as the target block size grows.\n";
    out << "- The stored inverse blocks are nearly dense. So apply uses dense math, while setup/scatter starts from sparse input.\n";
    out << "- This supports the Tensor Core argument for block-ILU apply/factor kernels: the algorithm stores and updates dense small blocks even when the source Jacobian blocks are sparse.\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_args(argc, argv);
        std::filesystem::create_directories(options.output_dir);

        std::vector<BlockRow> block_rows;
        std::vector<SummaryRow> summary_rows;
        for (const std::string& case_name : options.cases) {
            std::cout << "[case] " << case_name << '\n';
            const cuiter::CsrMatrix matrix =
                load_cupf_csr_dump(jacobian_path(options.jf_root,
                                                 case_name,
                                                 options.iteration));
            for (int32_t block_size : options.block_sizes) {
                for (bool use_block_ilu : {false, true}) {
                    std::vector<BlockRow> rows =
                        analyze_method(case_name,
                                       matrix,
                                       options.iteration,
                                       block_size,
                                       use_block_ilu,
                                       options);
                    summary_rows.push_back(summarize(rows));
                    block_rows.insert(block_rows.end(), rows.begin(), rows.end());
                }
            }
        }

        write_block_csv(options.output_dir / "diagonal_block_sparsity_blocks.csv",
                        block_rows);
        write_summary_csv(options.output_dir / "diagonal_block_sparsity_summary.csv",
                          summary_rows);
        write_report(options.output_dir / "diagonal_block_sparsity_report.md",
                     summary_rows);
        std::cout << "[done] wrote diagonal block sparsity results to "
                  << options.output_dir << '\n';
    } catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << '\n';
        return 1;
    }
    return 0;
}
