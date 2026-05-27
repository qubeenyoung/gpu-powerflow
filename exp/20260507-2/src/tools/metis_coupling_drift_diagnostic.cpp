#include "cuiter/common/cuda_utils.hpp"
#include "cuiter/core/csr_matrix.hpp"
#include "cuiter/reorder/metis_partition.hpp"
#include "cupf_minimal/case_data.hpp"
#include "cupf_minimal/direct_cudss_solver.hpp"
#include "cupf_minimal/jacobian_analysis.hpp"

#include <algorithm>
#include <array>
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
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int32_t kIterations = 3;
constexpr int32_t kFieldPTheta = 0;
constexpr int32_t kFieldPVm = 1;
constexpr int32_t kFieldQTheta = 2;
constexpr int32_t kFieldQVm = 3;
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

const std::array<const char*, 4> kFieldNames = {
    "J11",
    "J12",
    "J21",
    "J22",
};

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
    int32_t block_size = 64;
    bool allow_missing = false;
    bool skip_cudss = false;
};

struct LinearMetadata {
    int32_t n_bus = 0;
    int32_t n_pvpq = 0;
    int32_t n_pq = 0;
    std::vector<int32_t> index_to_bus;
    std::vector<int32_t> index_field;
    std::vector<int32_t> row_p;
    std::vector<int32_t> row_q;
    std::vector<int32_t> col_theta;
    std::vector<int32_t> col_vmag;
};

struct Accumulator {
    int64_t total_nnz = 0;
    int64_t offblock_nnz = 0;
    double total_abs = 0.0;
    double offblock_abs = 0.0;
    double total_sq = 0.0;
    double offblock_sq = 0.0;
    double total_effect = 0.0;
    double offblock_effect = 0.0;
};

struct EntryRow {
    std::string case_name;
    int32_t iteration = 0;
    int32_t n = 0;
    int32_t nnz = 0;
    int32_t num_blocks = 0;
    Accumulator all;
    std::array<Accumulator, 4> field;
};

struct EffectRow {
    std::string case_name;
    int32_t iteration = 0;
    int32_t n = 0;
    int32_t nnz = 0;
    double dx_norm2 = 0.0;
    double cudss_analyze_seconds = 0.0;
    double cudss_factorize_seconds = 0.0;
    double cudss_solve_seconds = 0.0;
    Accumulator all;
    std::array<Accumulator, 4> field;
};

struct BusPairStats {
    double coupling_sq = 0.0;
    double kept_coupling_sq = 0.0;
    double effect_rows[2] = {0.0, 0.0};
    double kept_effect_rows[2] = {0.0, 0.0};
};

struct BusPairRow {
    std::string case_name;
    int32_t iteration = 0;
    int32_t bus_pair_count = 0;
    int32_t same_bus_theta_vm_split_count = 0;
    int32_t same_bus_pq_split_count = 0;
    std::array<double, 3> top_coupling_kept_ratio = {0.0, 0.0, 0.0};
    std::array<double, 3> top_coupling_kept_mass_ratio = {0.0, 0.0, 0.0};
    std::array<double, 3> top_effect_kept_ratio = {0.0, 0.0, 0.0};
    std::array<double, 3> top_effect_kept_mass_ratio = {0.0, 0.0, 0.0};
};

struct DriftBucket {
    double curr_sq = 0.0;
    double delta_sq = 0.0;
};

struct DriftRow {
    std::string case_name;
    int32_t from_iter = 0;
    int32_t to_iter = 0;
    DriftBucket all;
    DriftBucket inblock;
    DriftBucket offblock;
    std::array<DriftBucket, 4> field;
    double max_abs_delta = 0.0;
    double median_abs_delta = 0.0;
    double p95_abs_delta = 0.0;
};

struct CaseDiagnostic {
    std::string case_name;
    std::vector<EntryRow> entry_rows;
    std::vector<EffectRow> effect_rows;
    std::vector<BusPairRow> buspair_rows;
    std::vector<DriftRow> drift_rows;
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

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --jf-root PATH\n"
        << "  --case-root PATH\n"
        << "  --output-dir PATH\n"
        << "  --cases a,b,c\n"
        << "  --block-size INT\n"
        << "  --allow-missing\n"
        << "  --skip-cudss\n";
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
        } else if (arg == "--block-size" && i + 1 < argc) {
            options.block_size = std::stoi(argv[++i]);
        } else if (arg == "--allow-missing") {
            options.allow_missing = true;
        } else if (arg == "--skip-cudss") {
            options.skip_cudss = true;
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
    if (options.block_size <= 0) {
        throw std::runtime_error("--block-size must be positive");
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
    if (n <= 0) {
        throw std::runtime_error("invalid vector length in " + path.string());
    }
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

double ratio(double numerator, double denominator)
{
    return denominator > 0.0 ? numerator / denominator : 0.0;
}

double fro_ratio(double numerator_sq, double denominator_sq)
{
    return denominator_sq > 0.0 ? std::sqrt(std::max(0.0, numerator_sq / denominator_sq)) : 0.0;
}

double norm2(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values) {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
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
    if (lo == hi) {
        return values[lo];
    }
    const double weight = position - static_cast<double>(lo);
    return (1.0 - weight) * values[lo] + weight * values[hi];
}

uint64_t bus_pair_key(int32_t row_bus, int32_t col_bus)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(row_bus)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(col_bus));
}

int32_t field_id(int32_t row, int32_t col, int32_t n_pvpq)
{
    const bool row_p = row < n_pvpq;
    const bool col_theta = col < n_pvpq;
    if (row_p && col_theta) {
        return kFieldPTheta;
    }
    if (row_p && !col_theta) {
        return kFieldPVm;
    }
    if (!row_p && col_theta) {
        return kFieldQTheta;
    }
    return kFieldQVm;
}

std::vector<int32_t> block_of_old_indices(const cuiter::CsrMatrix& matrix,
                                          const cuiter::MetisPermutation& permutation)
{
    std::vector<int32_t> block_of_new(static_cast<std::size_t>(matrix.rows), -1);
    for (int32_t block = 0; block < static_cast<int32_t>(permutation.block_sizes.size()); ++block) {
        const int32_t begin = permutation.block_starts[static_cast<std::size_t>(block)];
        const int32_t end = begin + permutation.block_sizes[static_cast<std::size_t>(block)];
        for (int32_t index = begin; index < end; ++index) {
            block_of_new[static_cast<std::size_t>(index)] = block;
        }
    }
    std::vector<int32_t> block_of_old(static_cast<std::size_t>(matrix.rows), -1);
    for (int32_t old_index = 0; old_index < matrix.rows; ++old_index) {
        const int32_t new_index = permutation.old_to_new[static_cast<std::size_t>(old_index)];
        block_of_old[static_cast<std::size_t>(old_index)] =
            block_of_new[static_cast<std::size_t>(new_index)];
    }
    return block_of_old;
}

LinearMetadata make_linear_metadata(const cupf_minimal::DumpCaseData& case_data)
{
    const cupf_minimal::JacobianIndexing indexing =
        cupf_minimal::make_jacobian_indexing(case_data.rows,
                                             case_data.pv.data(),
                                             static_cast<int32_t>(case_data.pv.size()),
                                             case_data.pq.data(),
                                             static_cast<int32_t>(case_data.pq.size()));
    const int32_t n = indexing.n_pvpq + indexing.n_pq;
    LinearMetadata metadata;
    metadata.n_bus = case_data.rows;
    metadata.n_pvpq = indexing.n_pvpq;
    metadata.n_pq = indexing.n_pq;
    metadata.index_to_bus.assign(static_cast<std::size_t>(n), -1);
    metadata.index_field.assign(static_cast<std::size_t>(n), -1);
    metadata.row_p.assign(static_cast<std::size_t>(case_data.rows), -1);
    metadata.row_q.assign(static_cast<std::size_t>(case_data.rows), -1);
    metadata.col_theta.assign(static_cast<std::size_t>(case_data.rows), -1);
    metadata.col_vmag.assign(static_cast<std::size_t>(case_data.rows), -1);

    for (int32_t i = 0; i < indexing.n_pvpq; ++i) {
        const int32_t bus = indexing.pvpq[static_cast<std::size_t>(i)];
        metadata.index_to_bus[static_cast<std::size_t>(i)] = bus;
        metadata.index_field[static_cast<std::size_t>(i)] = 0;
        metadata.row_p[static_cast<std::size_t>(bus)] = i;
        metadata.col_theta[static_cast<std::size_t>(bus)] = i;
    }
    for (int32_t i = 0; i < indexing.n_pq; ++i) {
        const int32_t bus = case_data.pq[static_cast<std::size_t>(i)];
        const int32_t index = indexing.n_pvpq + i;
        metadata.index_to_bus[static_cast<std::size_t>(index)] = bus;
        metadata.index_field[static_cast<std::size_t>(index)] = 1;
        metadata.row_q[static_cast<std::size_t>(bus)] = index;
        metadata.col_vmag[static_cast<std::size_t>(bus)] = index;
    }
    return metadata;
}

std::vector<double> solve_cudss_dx(const cuiter::CsrMatrix& matrix,
                                   const std::vector<double>& rhs,
                                   EffectRow& row)
{
    if (static_cast<int32_t>(rhs.size()) != matrix.rows) {
        throw std::runtime_error("RHS dimension does not match matrix");
    }
    cuiter::DeviceBuffer<int32_t> d_row_ptr;
    cuiter::DeviceBuffer<int32_t> d_col_idx;
    cuiter::DeviceBuffer<double> d_values;
    cuiter::DeviceBuffer<double> d_rhs;
    cuiter::DeviceBuffer<double> d_x;
    d_row_ptr.assign(matrix.row_ptr.data(), matrix.row_ptr.size());
    d_col_idx.assign(matrix.col_idx.data(), matrix.col_idx.size());
    d_values.assign(matrix.values.data(), matrix.values.size());
    d_rhs.assign(rhs.data(), rhs.size());
    d_x.resize(static_cast<std::size_t>(matrix.rows));
    d_x.memset_zero();

    cupf_minimal::DirectCudssSolver solver;
    solver.initialize(matrix,
                      d_row_ptr.data(),
                      d_col_idx.data(),
                      d_values.data(),
                      d_rhs.data(),
                      d_x.data());
    row.cudss_analyze_seconds = solver.analyze();
    row.cudss_factorize_seconds = solver.factorize();
    row.cudss_solve_seconds = solver.solve();

    std::vector<double> dx(static_cast<std::size_t>(matrix.rows), 0.0);
    d_x.copy_to(dx.data(), dx.size());
    return dx;
}

EntryRow analyze_entry_retention(const std::string& case_name,
                                 int32_t iteration,
                                 const cuiter::CsrMatrix& matrix,
                                 const LinearMetadata& metadata,
                                 const std::vector<int32_t>& block_of_old)
{
    EntryRow row;
    row.case_name = case_name;
    row.iteration = iteration;
    row.n = matrix.rows;
    row.nnz = matrix.nnz();
    row.num_blocks = 1 + *std::max_element(block_of_old.begin(), block_of_old.end());
    for (int32_t i = 0; i < matrix.rows; ++i) {
        const int32_t row_block = block_of_old[static_cast<std::size_t>(i)];
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(i)];
             pos < matrix.row_ptr[static_cast<std::size_t>(i + 1)];
             ++pos) {
            const int32_t col = matrix.col_idx[static_cast<std::size_t>(pos)];
            const int32_t col_block = block_of_old[static_cast<std::size_t>(col)];
            const double value = matrix.values[static_cast<std::size_t>(pos)];
            const bool offblock = row_block != col_block;
            const int32_t fid = field_id(i, col, metadata.n_pvpq);
            for (Accumulator* acc : {&row.all, &row.field[static_cast<std::size_t>(fid)]}) {
                ++acc->total_nnz;
                acc->total_abs += std::abs(value);
                acc->total_sq += value * value;
                if (offblock) {
                    ++acc->offblock_nnz;
                    acc->offblock_abs += std::abs(value);
                    acc->offblock_sq += value * value;
                }
            }
        }
    }
    return row;
}

EffectRow analyze_effect_retention(const std::string& case_name,
                                   int32_t iteration,
                                   const cuiter::CsrMatrix& matrix,
                                   const LinearMetadata& metadata,
                                   const std::vector<int32_t>& block_of_old,
                                   const std::vector<double>& rhs,
                                   bool skip_cudss)
{
    EffectRow row;
    row.case_name = case_name;
    row.iteration = iteration;
    row.n = matrix.rows;
    row.nnz = matrix.nnz();
    std::vector<double> dx(static_cast<std::size_t>(matrix.rows), 0.0);
    if (skip_cudss) {
        row.dx_norm2 = kNan;
        return row;
    }
    dx = solve_cudss_dx(matrix, rhs, row);
    row.dx_norm2 = norm2(dx);
    for (int32_t i = 0; i < matrix.rows; ++i) {
        const int32_t row_block = block_of_old[static_cast<std::size_t>(i)];
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(i)];
             pos < matrix.row_ptr[static_cast<std::size_t>(i + 1)];
             ++pos) {
            const int32_t col = matrix.col_idx[static_cast<std::size_t>(pos)];
            const int32_t col_block = block_of_old[static_cast<std::size_t>(col)];
            const double effect = std::abs(matrix.values[static_cast<std::size_t>(pos)] *
                                           dx[static_cast<std::size_t>(col)]);
            const bool offblock = row_block != col_block;
            const int32_t fid = field_id(i, col, metadata.n_pvpq);
            for (Accumulator* acc : {&row.all, &row.field[static_cast<std::size_t>(fid)]}) {
                acc->total_effect += effect;
                if (offblock) {
                    acc->offblock_effect += effect;
                }
            }
        }
    }
    return row;
}

int32_t count_same_bus_splits(const LinearMetadata& metadata,
                              const std::vector<int32_t>& block_of_old,
                              bool pq_split)
{
    int32_t count = 0;
    for (int32_t bus = 0; bus < metadata.n_bus; ++bus) {
        const int32_t first = pq_split ? metadata.row_p[static_cast<std::size_t>(bus)]
                                       : metadata.col_theta[static_cast<std::size_t>(bus)];
        const int32_t second = pq_split ? metadata.row_q[static_cast<std::size_t>(bus)]
                                        : metadata.col_vmag[static_cast<std::size_t>(bus)];
        if (first >= 0 && second >= 0 &&
            block_of_old[static_cast<std::size_t>(first)] !=
                block_of_old[static_cast<std::size_t>(second)]) {
            ++count;
        }
    }
    return count;
}

template <typename ValueFn, typename KeptFn>
std::array<double, 3> top_kept_ratio(const std::vector<BusPairStats>& values,
                                     ValueFn value_fn,
                                     KeptFn kept_fn)
{
    std::vector<std::pair<double, bool>> ranked;
    ranked.reserve(values.size());
    for (const BusPairStats& item : values) {
        const double value = value_fn(item);
        if (value > 0.0) {
            ranked.push_back({value, kept_fn(item)});
        }
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first > rhs.first;
    });
    std::array<double, 3> out = {0.0, 0.0, 0.0};
    const std::array<double, 3> fractions = {0.01, 0.05, 0.10};
    for (std::size_t i = 0; i < fractions.size(); ++i) {
        if (ranked.empty()) {
            continue;
        }
        const std::size_t count = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil(fractions[i] * ranked.size())));
        int32_t kept = 0;
        for (std::size_t k = 0; k < std::min(count, ranked.size()); ++k) {
            kept += ranked[k].second ? 1 : 0;
        }
        out[i] = static_cast<double>(kept) /
                 static_cast<double>(std::min(count, ranked.size()));
    }
    return out;
}

template <typename ValueFn, typename KeptValueFn>
std::array<double, 3> top_kept_mass_ratio(const std::vector<BusPairStats>& values,
                                          ValueFn value_fn,
                                          KeptValueFn kept_value_fn)
{
    std::vector<const BusPairStats*> ranked;
    ranked.reserve(values.size());
    for (const BusPairStats& item : values) {
        if (value_fn(item) > 0.0) {
            ranked.push_back(&item);
        }
    }
    std::sort(ranked.begin(), ranked.end(), [&](const auto* lhs, const auto* rhs) {
        return value_fn(*lhs) > value_fn(*rhs);
    });
    std::array<double, 3> out = {0.0, 0.0, 0.0};
    const std::array<double, 3> fractions = {0.01, 0.05, 0.10};
    for (std::size_t i = 0; i < fractions.size(); ++i) {
        if (ranked.empty()) {
            continue;
        }
        const std::size_t count = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil(fractions[i] * ranked.size())));
        double total = 0.0;
        double kept = 0.0;
        for (std::size_t k = 0; k < std::min(count, ranked.size()); ++k) {
            total += value_fn(*ranked[k]);
            kept += kept_value_fn(*ranked[k]);
        }
        out[i] = ratio(kept, total);
    }
    return out;
}

BusPairRow analyze_bus_pairs(const std::string& case_name,
                             int32_t iteration,
                             const cuiter::CsrMatrix& matrix,
                             const LinearMetadata& metadata,
                             const std::vector<int32_t>& block_of_old,
                             const std::vector<double>& rhs,
                             bool skip_cudss)
{
    BusPairRow row;
    row.case_name = case_name;
    row.iteration = iteration;
    row.same_bus_theta_vm_split_count = count_same_bus_splits(metadata, block_of_old, false);
    row.same_bus_pq_split_count = count_same_bus_splits(metadata, block_of_old, true);

    std::vector<double> dx(static_cast<std::size_t>(matrix.rows), 0.0);
    if (!skip_cudss) {
        EffectRow unused;
        dx = solve_cudss_dx(matrix, rhs, unused);
    }

    std::unordered_map<uint64_t, BusPairStats> pair_stats;
    pair_stats.reserve(static_cast<std::size_t>(matrix.nnz()));
    for (int32_t i = 0; i < matrix.rows; ++i) {
        const int32_t row_bus = metadata.index_to_bus[static_cast<std::size_t>(i)];
        const int32_t row_field = metadata.index_field[static_cast<std::size_t>(i)];
        const int32_t row_block = block_of_old[static_cast<std::size_t>(i)];
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(i)];
             pos < matrix.row_ptr[static_cast<std::size_t>(i + 1)];
             ++pos) {
            const int32_t col = matrix.col_idx[static_cast<std::size_t>(pos)];
            const int32_t col_bus = metadata.index_to_bus[static_cast<std::size_t>(col)];
            const int32_t col_block = block_of_old[static_cast<std::size_t>(col)];
            const double value = matrix.values[static_cast<std::size_t>(pos)];
            BusPairStats& stats = pair_stats[bus_pair_key(row_bus, col_bus)];
            stats.coupling_sq += value * value;
            if (row_block == col_block) {
                stats.kept_coupling_sq += value * value;
            }
            if (!skip_cudss && row_field >= 0 && row_field < 2) {
                const double contribution = value * dx[static_cast<std::size_t>(col)];
                stats.effect_rows[row_field] += contribution;
                if (row_block == col_block) {
                    stats.kept_effect_rows[row_field] += contribution;
                }
            }
        }
    }

    std::vector<BusPairStats> values;
    values.reserve(pair_stats.size());
    for (const auto& item : pair_stats) {
        values.push_back(item.second);
    }
    row.bus_pair_count = static_cast<int32_t>(values.size());

    row.top_coupling_kept_ratio = top_kept_ratio(
        values,
        [](const BusPairStats& item) { return std::sqrt(item.coupling_sq); },
        [](const BusPairStats& item) {
            return fro_ratio(item.kept_coupling_sq, item.coupling_sq) >= 0.5;
        });
    row.top_coupling_kept_mass_ratio = top_kept_mass_ratio(
        values,
        [](const BusPairStats& item) { return std::sqrt(item.coupling_sq); },
        [](const BusPairStats& item) { return std::sqrt(item.kept_coupling_sq); });

    if (!skip_cudss) {
        row.top_effect_kept_ratio = top_kept_ratio(
            values,
            [](const BusPairStats& item) {
                return std::sqrt(item.effect_rows[0] * item.effect_rows[0] +
                                 item.effect_rows[1] * item.effect_rows[1]);
            },
            [](const BusPairStats& item) {
                const double total = std::sqrt(item.effect_rows[0] * item.effect_rows[0] +
                                               item.effect_rows[1] * item.effect_rows[1]);
                const double kept = std::sqrt(item.kept_effect_rows[0] * item.kept_effect_rows[0] +
                                              item.kept_effect_rows[1] * item.kept_effect_rows[1]);
                return ratio(kept, total) >= 0.5;
            });
        row.top_effect_kept_mass_ratio = top_kept_mass_ratio(
            values,
            [](const BusPairStats& item) {
                return std::sqrt(item.effect_rows[0] * item.effect_rows[0] +
                                 item.effect_rows[1] * item.effect_rows[1]);
            },
            [](const BusPairStats& item) {
                return std::sqrt(item.kept_effect_rows[0] * item.kept_effect_rows[0] +
                                 item.kept_effect_rows[1] * item.kept_effect_rows[1]);
            });
    } else {
        row.top_effect_kept_ratio = {kNan, kNan, kNan};
        row.top_effect_kept_mass_ratio = {kNan, kNan, kNan};
    }
    return row;
}

DriftRow analyze_drift(const std::string& case_name,
                       int32_t from_iter,
                       const cuiter::CsrMatrix& curr,
                       const cuiter::CsrMatrix& next,
                       const LinearMetadata& metadata,
                       const std::vector<int32_t>& block_of_old)
{
    if (curr.rows != next.rows || curr.cols != next.cols ||
        curr.row_ptr != next.row_ptr || curr.col_idx != next.col_idx) {
        throw std::runtime_error("Jacobian drift requires matching sparsity pattern");
    }
    DriftRow row;
    row.case_name = case_name;
    row.from_iter = from_iter;
    row.to_iter = from_iter + 1;
    std::vector<double> abs_delta;
    abs_delta.reserve(curr.values.size());
    for (int32_t i = 0; i < curr.rows; ++i) {
        const int32_t row_block = block_of_old[static_cast<std::size_t>(i)];
        for (int32_t pos = curr.row_ptr[static_cast<std::size_t>(i)];
             pos < curr.row_ptr[static_cast<std::size_t>(i + 1)];
             ++pos) {
            const int32_t col = curr.col_idx[static_cast<std::size_t>(pos)];
            const int32_t col_block = block_of_old[static_cast<std::size_t>(col)];
            const double curr_value = curr.values[static_cast<std::size_t>(pos)];
            const double delta = next.values[static_cast<std::size_t>(pos)] - curr_value;
            const double abs_value = std::abs(delta);
            const int32_t fid = field_id(i, col, metadata.n_pvpq);
            const bool inblock = row_block == col_block;
            auto add = [&](DriftBucket& bucket) {
                bucket.curr_sq += curr_value * curr_value;
                bucket.delta_sq += delta * delta;
            };
            add(row.all);
            add(row.field[static_cast<std::size_t>(fid)]);
            if (inblock) {
                add(row.inblock);
            } else {
                add(row.offblock);
            }
            abs_delta.push_back(abs_value);
            row.max_abs_delta = std::max(row.max_abs_delta, abs_value);
        }
    }
    row.median_abs_delta = percentile(abs_delta, 0.50);
    row.p95_abs_delta = percentile(abs_delta, 0.95);
    return row;
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

void write_entry_csv(const std::filesystem::path& path, const std::vector<EntryRow>& rows)
{
    std::ofstream out(path);
    out << "case,iteration,n,nnz,num_blocks,total_nnz,inblock_nnz,offblock_nnz,"
           "offblock_nnz_ratio,offblock_abs_ratio,offblock_fro_ratio";
    for (const char* name : kFieldNames) {
        out << ',' << name << "_offblock_nnz_ratio"
            << ',' << name << "_offblock_abs_ratio"
            << ',' << name << "_offblock_fro_ratio";
    }
    out << '\n';
    for (const EntryRow& row : rows) {
        const int64_t inblock_nnz = row.all.total_nnz - row.all.offblock_nnz;
        out << row.case_name << ',' << row.iteration << ',' << row.n << ',' << row.nnz << ','
            << row.num_blocks << ',' << row.all.total_nnz << ',' << inblock_nnz << ','
            << row.all.offblock_nnz << ','
            << format_double(ratio(row.all.offblock_nnz, row.all.total_nnz)) << ','
            << format_double(ratio(row.all.offblock_abs, row.all.total_abs)) << ','
            << format_double(fro_ratio(row.all.offblock_sq, row.all.total_sq));
        for (const Accumulator& acc : row.field) {
            out << ',' << format_double(ratio(acc.offblock_nnz, acc.total_nnz))
                << ',' << format_double(ratio(acc.offblock_abs, acc.total_abs))
                << ',' << format_double(fro_ratio(acc.offblock_sq, acc.total_sq));
        }
        out << '\n';
    }
}

void write_effect_csv(const std::filesystem::path& path, const std::vector<EffectRow>& rows)
{
    std::ofstream out(path);
    out << "case,iteration,n,nnz,dx_norm2,cudss_analyze_seconds,cudss_factorize_seconds,"
           "cudss_solve_seconds,offblock_effect_ratio";
    for (const char* name : kFieldNames) {
        out << ',' << name << "_offblock_effect_ratio";
    }
    out << '\n';
    for (const EffectRow& row : rows) {
        out << row.case_name << ',' << row.iteration << ',' << row.n << ',' << row.nnz << ','
            << format_double(row.dx_norm2) << ','
            << format_double(row.cudss_analyze_seconds) << ','
            << format_double(row.cudss_factorize_seconds) << ','
            << format_double(row.cudss_solve_seconds) << ','
            << format_double(ratio(row.all.offblock_effect, row.all.total_effect));
        for (const Accumulator& acc : row.field) {
            out << ',' << format_double(ratio(acc.offblock_effect, acc.total_effect));
        }
        out << '\n';
    }
}

void write_buspair_csv(const std::filesystem::path& path, const std::vector<BusPairRow>& rows)
{
    std::ofstream out(path);
    out << "case,iteration,bus_pair_count,same_bus_theta_vm_split_count,same_bus_pq_split_count,"
           "top1_coupling_norm_bus_edges_kept_ratio,top5_coupling_norm_bus_edges_kept_ratio,"
           "top10_coupling_norm_bus_edges_kept_ratio,top1_coupling_norm_kept_mass_ratio,"
           "top5_coupling_norm_kept_mass_ratio,top10_coupling_norm_kept_mass_ratio,"
           "top1_effect_bus_edges_kept_ratio,top5_effect_bus_edges_kept_ratio,"
           "top10_effect_bus_edges_kept_ratio,top1_effect_kept_mass_ratio,"
           "top5_effect_kept_mass_ratio,top10_effect_kept_mass_ratio\n";
    for (const BusPairRow& row : rows) {
        out << row.case_name << ',' << row.iteration << ',' << row.bus_pair_count << ','
            << row.same_bus_theta_vm_split_count << ',' << row.same_bus_pq_split_count;
        for (double value : row.top_coupling_kept_ratio) {
            out << ',' << format_double(value);
        }
        for (double value : row.top_coupling_kept_mass_ratio) {
            out << ',' << format_double(value);
        }
        for (double value : row.top_effect_kept_ratio) {
            out << ',' << format_double(value);
        }
        for (double value : row.top_effect_kept_mass_ratio) {
            out << ',' << format_double(value);
        }
        out << '\n';
    }
}

void write_drift_csv(const std::filesystem::path& path, const std::vector<DriftRow>& rows)
{
    std::ofstream out(path);
    out << "case,from_iter,to_iter,rel_change_all,rel_change_inblock,rel_change_offblock";
    for (const char* name : kFieldNames) {
        out << ",rel_change_" << name;
    }
    out << ",max_abs_delta,median_abs_delta,p95_abs_delta\n";
    for (const DriftRow& row : rows) {
        out << row.case_name << ',' << row.from_iter << ',' << row.to_iter << ','
            << format_double(fro_ratio(row.all.delta_sq, row.all.curr_sq)) << ','
            << format_double(fro_ratio(row.inblock.delta_sq, row.inblock.curr_sq)) << ','
            << format_double(fro_ratio(row.offblock.delta_sq, row.offblock.curr_sq));
        for (const DriftBucket& bucket : row.field) {
            out << ',' << format_double(fro_ratio(bucket.delta_sq, bucket.curr_sq));
        }
        out << ',' << format_double(row.max_abs_delta)
            << ',' << format_double(row.median_abs_delta)
            << ',' << format_double(row.p95_abs_delta) << '\n';
    }
}

template <typename Row, typename Fn>
double mean_metric(const std::vector<Row>& rows, Fn fn)
{
    double sum = 0.0;
    int32_t count = 0;
    for (const Row& row : rows) {
        const double value = fn(row);
        if (std::isfinite(value)) {
            sum += value;
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : kNan;
}

void write_report(const std::filesystem::path& path,
                  const std::vector<EntryRow>& entry_rows,
                  const std::vector<EffectRow>& effect_rows,
                  const std::vector<BusPairRow>& buspair_rows,
                  const std::vector<DriftRow>& drift_rows)
{
    std::ofstream out(path);
    const double entry_off_abs = mean_metric(entry_rows, [](const EntryRow& row) {
        return ratio(row.all.offblock_abs, row.all.total_abs);
    });
    const double entry_off_fro = mean_metric(entry_rows, [](const EntryRow& row) {
        return fro_ratio(row.all.offblock_sq, row.all.total_sq);
    });
    const double effect_off = mean_metric(effect_rows, [](const EffectRow& row) {
        return ratio(row.all.offblock_effect, row.all.total_effect);
    });
    const double top5_effect_kept = mean_metric(buspair_rows, [](const BusPairRow& row) {
        return row.top_effect_kept_ratio[1];
    });
    const double top5_coupling_kept = mean_metric(buspair_rows, [](const BusPairRow& row) {
        return row.top_coupling_kept_ratio[1];
    });
    const double drift_all = mean_metric(drift_rows, [](const DriftRow& row) {
        return fro_ratio(row.all.delta_sq, row.all.curr_sq);
    });
    const double drift_off = mean_metric(drift_rows, [](const DriftRow& row) {
        return fro_ratio(row.offblock.delta_sq, row.offblock.curr_sq);
    });

    std::array<double, 4> field_effect = {0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < field_effect.size(); ++i) {
        field_effect[i] = mean_metric(effect_rows, [i](const EffectRow& row) {
            const Accumulator& acc = row.field[i];
            return ratio(acc.offblock_effect, acc.total_effect);
        });
    }
    const auto max_it = std::max_element(field_effect.begin(), field_effect.end());
    const std::size_t max_field = static_cast<std::size_t>(max_it - field_effect.begin());

    out << "# METIS Coupling Retention and Jacobian Drift Diagnostic\n\n";
    out << "## Answers\n\n";
    out << "1. Strong coupling cut by current METIS blocks: off-block abs ratio mean = `"
        << format_double(entry_off_abs) << "`, off-block Frobenius ratio mean = `"
        << format_double(entry_off_fro) << "`.\n";
    out << "2. Field concentration by cuDSS-dx effect: largest mean off-block effect is `"
        << kFieldNames[max_field] << "` = `" << format_double(*max_it) << "`.\n";
    out << "3. cuDSS-dx weighted coupling outside blocks: mean offblock_effect_ratio = `"
        << format_double(effect_off) << "`.\n";
    out << "4. Top bus-pair preservation: top-5% coupling kept ratio mean = `"
        << format_double(top5_coupling_kept)
        << "`, top-5% effect kept ratio mean = `" << format_double(top5_effect_kept) << "`.\n";
    out << "5. Jacobian numeric drift: mean rel_change_all = `" << format_double(drift_all)
        << "`, mean rel_change_offblock = `" << format_double(drift_off) << "`.\n";
    out << "6. Bus-aware weighted METIS evidence: use the top-effect kept ratio and "
           "offblock_effect_ratio above. Low top-effect kept ratio supports the hypothesis; "
           "high top-effect kept ratio weakens it.\n\n";

    out << "## Notes\n\n";
    out << "- Partition mode: existing unknown-level METIS, block size 64 unless overridden.\n";
    out << "- Effect metric uses diagnostic cuDSS solution `dx` for each dumped `Jk/Fk`.\n";
    out << "- `J11=P-theta`, `J12=P-|V|`, `J21=Q-theta`, `J22=Q-|V|`.\n";
}

CaseDiagnostic analyze_case(const CliOptions& options, const std::string& case_name)
{
    const std::filesystem::path case_jf_dir = options.jf_root / case_name;
    const std::filesystem::path case_data_dir = options.case_root / case_name;
    std::array<cuiter::CsrMatrix, kIterations> matrices;
    std::array<std::vector<double>, kIterations> rhs_values;
    for (int32_t iter = 0; iter < kIterations; ++iter) {
        const auto matrix_path = case_jf_dir / ("J" + std::to_string(iter) + ".txt");
        const auto rhs_path = case_jf_dir / ("F" + std::to_string(iter) + ".txt");
        if (!std::filesystem::exists(matrix_path) || !std::filesystem::exists(rhs_path)) {
            throw std::runtime_error("missing J/F dump for " + case_name + " iteration " +
                                     std::to_string(iter));
        }
        matrices[static_cast<std::size_t>(iter)] = load_cupf_csr_dump(matrix_path);
        rhs_values[static_cast<std::size_t>(iter)] = load_cupf_vector_dump(rhs_path);
    }

    const cupf_minimal::DumpCaseData case_data = cupf_minimal::load_dump_case(case_data_dir);
    const LinearMetadata metadata = make_linear_metadata(case_data);
    if (metadata.n_pvpq + metadata.n_pq != matrices[0].rows) {
        throw std::runtime_error("case metadata does not match dumped Jacobian dimension for " +
                                 case_name);
    }

    const cuiter::MetisPermutation permutation =
        cuiter::build_metis_permutation(matrices[0], options.block_size);
    const std::vector<int32_t> block_of_old = block_of_old_indices(matrices[0], permutation);

    CaseDiagnostic diagnostic;
    diagnostic.case_name = case_name;
    for (int32_t iter = 0; iter < kIterations; ++iter) {
        std::cout << "[diag] " << case_name << " J" << iter << "/F" << iter << "\n";
        diagnostic.entry_rows.push_back(analyze_entry_retention(case_name,
                                                                iter,
                                                                matrices[iter],
                                                                metadata,
                                                                block_of_old));
        diagnostic.effect_rows.push_back(analyze_effect_retention(case_name,
                                                                  iter,
                                                                  matrices[iter],
                                                                  metadata,
                                                                  block_of_old,
                                                                  rhs_values[iter],
                                                                  options.skip_cudss));
        diagnostic.buspair_rows.push_back(analyze_bus_pairs(case_name,
                                                            iter,
                                                            matrices[iter],
                                                            metadata,
                                                            block_of_old,
                                                            rhs_values[iter],
                                                            options.skip_cudss));
    }
    for (int32_t iter = 0; iter + 1 < kIterations; ++iter) {
        diagnostic.drift_rows.push_back(analyze_drift(case_name,
                                                      iter,
                                                      matrices[iter],
                                                      matrices[iter + 1],
                                                      metadata,
                                                      block_of_old));
    }
    return diagnostic;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        std::filesystem::create_directories(options.output_dir);

        std::vector<EntryRow> entry_rows;
        std::vector<EffectRow> effect_rows;
        std::vector<BusPairRow> buspair_rows;
        std::vector<DriftRow> drift_rows;

        for (const std::string& case_name : options.cases) {
            try {
                CaseDiagnostic diagnostic = analyze_case(options, case_name);
                entry_rows.insert(entry_rows.end(),
                                  diagnostic.entry_rows.begin(),
                                  diagnostic.entry_rows.end());
                effect_rows.insert(effect_rows.end(),
                                   diagnostic.effect_rows.begin(),
                                   diagnostic.effect_rows.end());
                buspair_rows.insert(buspair_rows.end(),
                                    diagnostic.buspair_rows.begin(),
                                    diagnostic.buspair_rows.end());
                drift_rows.insert(drift_rows.end(),
                                  diagnostic.drift_rows.begin(),
                                  diagnostic.drift_rows.end());
            } catch (const std::exception& ex) {
                if (!options.allow_missing) {
                    throw;
                }
                std::cerr << "[skip] " << case_name << ": " << ex.what() << "\n";
            }
        }

        write_entry_csv(options.output_dir / "metis_coupling_retention_entry.csv", entry_rows);
        write_effect_csv(options.output_dir / "metis_coupling_retention_effect.csv", effect_rows);
        write_buspair_csv(options.output_dir / "metis_coupling_retention_buspair.csv", buspair_rows);
        write_drift_csv(options.output_dir / "jacobian_numeric_drift.csv", drift_rows);
        write_report(options.output_dir / "metis_coupling_drift_report.md",
                     entry_rows,
                     effect_rows,
                     buspair_rows,
                     drift_rows);
        std::cout << "[done] wrote diagnostics to " << options.output_dir << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
