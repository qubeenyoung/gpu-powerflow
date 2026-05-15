#include "cuiter/common/cuda_utils.hpp"
#include "cuiter/core/csr_matrix.hpp"
#include "cuiter/solver/cpu_block_ilu0_pilot.hpp"
#include "cupf_minimal/case_data.hpp"
#include "cupf_minimal/direct_cudss_solver.hpp"
#include "cupf_minimal/jacobian_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

struct CliOptions {
    std::filesystem::path jf_root = "raw/cupf_jf_dumps";
    std::filesystem::path case_root =
        "/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps";
    std::filesystem::path output_dir = "results/j11_correction_bench";
    std::vector<std::string> cases = {
        "case2383wp",
        "case3120sp",
        "case9241pegase",
        "case13659pegase",
        "case6468rte",
    };
    std::vector<int32_t> iterations = {0, 1, 2};
    std::vector<int32_t> block_sizes = {8, 16};
    int32_t bicgstab_iters = 2;
    int32_t repeat = 5;
    double diag_shift_scale = 1.0e-8;
    bool allow_missing = false;
};

struct LinearMetadata {
    int32_t n_pvpq = 0;
    int32_t n_pq = 0;
};

struct Stats {
    double mean = kNan;
    double median = kNan;
    double min = kNan;
    double max = kNan;
};

struct CudssSolveResult {
    std::vector<double> x;
    Stats analyze_ms;
    Stats factor_ms;
    Stats solve_ms;
    Stats factor_solve_ms;
    Stats total_ms;
    double rel_res = kNan;
};

struct ResidualSplit {
    double total_abs = 0.0;
    double total_rel = 0.0;
    double p_abs = 0.0;
    double p_rel = 0.0;
    double q_abs = 0.0;
    double q_rel = 0.0;
};

struct Row {
    std::string case_name;
    int32_t iteration = 0;
    int32_t block_size = 0;
    int32_t full_n = 0;
    int32_t full_nnz = 0;
    int32_t n_pvpq = 0;
    int32_t n_pq = 0;
    int32_t j11_n = 0;
    int32_t j11_nnz = 0;
    double dx_error_ratio_before = kNan;
    double dx_error_ratio_after = kNan;
    double dx_error_ratio_delta = kNan;
    double theta_error_ratio_before = kNan;
    double theta_error_ratio_after = kNan;
    double theta_error_ratio_delta = kNan;
    double vmag_error_ratio_before = kNan;
    double vmag_error_ratio_after = kNan;
    double vmag_error_ratio_delta = kNan;
    double cosine_before = kNan;
    double cosine_after = kNan;
    double residual_before = kNan;
    double residual_after = kNan;
    double p_residual_abs_before = kNan;
    double p_residual_abs_after = kNan;
    double p_residual_rel_before = kNan;
    double p_residual_rel_after = kNan;
    double q_residual_abs_before = kNan;
    double q_residual_abs_after = kNan;
    double q_residual_rel_before = kNan;
    double q_residual_rel_after = kNan;
    double dtheta_corr_norm = kNan;
    double dx0_norm = kNan;
    double dx1_norm = kNan;
    double dx_full_norm = kNan;
    Stats j11_analyze_ms;
    Stats j11_factor_ms;
    Stats j11_solve_ms;
    Stats j11_factor_solve_ms;
    Stats j11_total_ms;
    double j11_rel_res = kNan;
    std::string status = "ok";
    std::string error_message;
};

std::vector<std::string> split_list(const std::string& text)
{
    std::vector<std::string> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back(item);
        }
    }
    return values;
}

std::vector<int32_t> parse_int_list(const std::string& text)
{
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
        << "  --case-root PATH\n"
        << "  --output-dir PATH\n"
        << "  --cases a,b,c\n"
        << "  --iters J0,J1,J2\n"
        << "  --block-sizes 8,16\n"
        << "  --bicgstab-iters 2\n"
        << "  --repeat 5\n"
        << "  --block-jacobi-diag-shift-scale FLOAT\n"
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
        } else if ((arg == "--iters" || arg == "--iterations") && i + 1 < argc) {
            options.iterations = parse_int_list(argv[++i]);
        } else if (arg == "--block-sizes" && i + 1 < argc) {
            options.block_sizes = parse_int_list(argv[++i]);
        } else if (arg == "--bicgstab-iters" && i + 1 < argc) {
            options.bicgstab_iters = std::stoi(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            options.repeat = std::stoi(argv[++i]);
        } else if (arg == "--block-jacobi-diag-shift-scale" && i + 1 < argc) {
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
    if (options.cases.empty() || options.iterations.empty() || options.block_sizes.empty()) {
        throw std::runtime_error("case, iteration, and block-size lists must be nonempty");
    }
    if (options.repeat <= 0 || options.bicgstab_iters <= 0) {
        throw std::runtime_error("--repeat and --bicgstab-iters must be positive");
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

double dot(const std::vector<double>& lhs, const std::vector<double>& rhs)
{
    long double sum = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        sum += static_cast<long double>(lhs[i]) * static_cast<long double>(rhs[i]);
    }
    return static_cast<double>(sum);
}

double norm2(const std::vector<double>& values)
{
    return std::sqrt(std::max(0.0, dot(values, values)));
}

double safe_ratio(double numerator, double denominator)
{
    return denominator > 0.0 ? numerator / denominator : kNan;
}

double cosine(const std::vector<double>& lhs, const std::vector<double>& rhs)
{
    return dot(lhs, rhs) /
           std::max(norm2(lhs) * norm2(rhs), std::numeric_limits<double>::min());
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

Stats summarize(std::vector<double> values)
{
    Stats stats;
    if (values.empty()) {
        return stats;
    }
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    stats.mean = sum / static_cast<double>(values.size());
    if (values.size() % 2 == 0) {
        const std::size_t hi = values.size() / 2;
        stats.median = 0.5 * (values[hi - 1] + values[hi]);
    } else {
        stats.median = values[values.size() / 2];
    }
    stats.min = values.front();
    stats.max = values.back();
    return stats;
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

std::vector<double> residual_vector(const cuiter::CsrMatrix& matrix,
                                    const std::vector<double>& rhs,
                                    const std::vector<double>& x)
{
    std::vector<double> residual = rhs;
    const std::vector<double> ax = spmv(matrix, x);
    for (std::size_t i = 0; i < residual.size(); ++i) {
        residual[i] -= ax[i];
    }
    return residual;
}

std::vector<double> subtract_vectors(const std::vector<double>& lhs,
                                     const std::vector<double>& rhs)
{
    std::vector<double> out(lhs.size(), 0.0);
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        out[i] = lhs[i] - rhs[i];
    }
    return out;
}

std::vector<double> slice(const std::vector<double>& values, int32_t begin, int32_t count)
{
    return std::vector<double>(values.begin() + begin, values.begin() + begin + count);
}

LinearMetadata make_linear_metadata(const cupf_minimal::DumpCaseData& case_data)
{
    const cupf_minimal::JacobianIndexing indexing =
        cupf_minimal::make_jacobian_indexing(case_data.rows,
                                             case_data.pv.data(),
                                             static_cast<int32_t>(case_data.pv.size()),
                                             case_data.pq.data(),
                                             static_cast<int32_t>(case_data.pq.size()));
    LinearMetadata metadata;
    metadata.n_pvpq = indexing.n_pvpq;
    metadata.n_pq = indexing.n_pq;
    return metadata;
}

cuiter::CsrMatrix extract_j11(const cuiter::CsrMatrix& full, int32_t n_pvpq)
{
    cuiter::CsrMatrix j11;
    j11.rows = n_pvpq;
    j11.cols = n_pvpq;
    j11.row_ptr.assign(static_cast<std::size_t>(n_pvpq + 1), 0);
    for (int32_t row = 0; row < n_pvpq; ++row) {
        for (int32_t pos = full.row_ptr[static_cast<std::size_t>(row)];
             pos < full.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = full.col_idx[static_cast<std::size_t>(pos)];
            if (col < n_pvpq) {
                j11.col_idx.push_back(col);
                j11.values.push_back(full.values[static_cast<std::size_t>(pos)]);
            }
        }
        j11.row_ptr[static_cast<std::size_t>(row + 1)] =
            static_cast<int32_t>(j11.values.size());
    }
    return j11;
}

CudssSolveResult solve_cudss_timed(const cuiter::CsrMatrix& matrix,
                                   const std::vector<double>& rhs,
                                   int32_t repeat)
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

    std::vector<double> analyze_ms;
    std::vector<double> factor_ms;
    std::vector<double> solve_ms;
    std::vector<double> factor_solve_ms;
    std::vector<double> total_ms;
    std::vector<double> last_x(rhs.size(), 0.0);
    for (int32_t i = 0; i < repeat; ++i) {
        d_x.memset_zero();
        cupf_minimal::DirectCudssSolver solver;
        solver.initialize(matrix,
                          d_row_ptr.data(),
                          d_col_idx.data(),
                          d_values.data(),
                          d_rhs.data(),
                          d_x.data());
        const double analyze = 1000.0 * solver.analyze();
        const double factor = 1000.0 * solver.factorize();
        const double solve = 1000.0 * solver.solve();
        analyze_ms.push_back(analyze);
        factor_ms.push_back(factor);
        solve_ms.push_back(solve);
        factor_solve_ms.push_back(factor + solve);
        total_ms.push_back(analyze + factor + solve);
        if (i == repeat - 1) {
            d_x.copy_to(last_x.data(), last_x.size());
        }
    }

    CudssSolveResult result;
    result.x = std::move(last_x);
    result.analyze_ms = summarize(analyze_ms);
    result.factor_ms = summarize(factor_ms);
    result.solve_ms = summarize(solve_ms);
    result.factor_solve_ms = summarize(factor_solve_ms);
    result.total_ms = summarize(total_ms);
    result.rel_res = safe_ratio(norm2(residual_vector(matrix, rhs, result.x)), norm2(rhs));
    return result;
}

std::vector<double> solve_cudss_once(const cuiter::CsrMatrix& matrix,
                                     const std::vector<double>& rhs)
{
    return solve_cudss_timed(matrix, rhs, 1).x;
}

std::vector<double> solve_block_jacobi_bicgstab(const cuiter::CsrMatrix& matrix,
                                                const std::vector<double>& rhs,
                                                int32_t block_size,
                                                const CliOptions& options)
{
    cuiter::cpu_pilot::CpuBlockIlu0Options solve_options;
    solve_options.block_size = block_size;
    solve_options.bicgstab_iters = options.bicgstab_iters;
    solve_options.diag_shift_scale = options.diag_shift_scale;
    solve_options.use_block_ilu0 = false;
    solve_options.use_block_coloring_order = false;
    const cuiter::cpu_pilot::CpuBlockIlu0Result result =
        cuiter::cpu_pilot::solve(matrix, rhs, solve_options);
    if (result.factor_failed) {
        throw std::runtime_error("block-Jacobi setup failed: " + result.stop_reason);
    }
    return result.solution;
}

ResidualSplit compute_residual_split(const cuiter::CsrMatrix& matrix,
                                     const std::vector<double>& rhs,
                                     const std::vector<double>& x,
                                     const LinearMetadata& metadata)
{
    const std::vector<double> residual = residual_vector(matrix, rhs, x);
    ResidualSplit split;
    split.total_abs = norm2(residual);
    split.total_rel = safe_ratio(split.total_abs, norm2(rhs));
    split.p_abs = norm2(slice(residual, 0, metadata.n_pvpq));
    split.p_rel = safe_ratio(split.p_abs, norm2(slice(rhs, 0, metadata.n_pvpq)));
    split.q_abs = norm2(slice(residual, metadata.n_pvpq, metadata.n_pq));
    split.q_rel = safe_ratio(split.q_abs, norm2(slice(rhs, metadata.n_pvpq, metadata.n_pq)));
    return split;
}

double split_error_ratio(const std::vector<double>& x,
                         const std::vector<double>& reference,
                         int32_t begin,
                         int32_t count)
{
    return safe_ratio(norm2(subtract_vectors(slice(x, begin, count),
                                             slice(reference, begin, count))),
                      norm2(slice(reference, begin, count)));
}

Row analyze_one(const CliOptions& options,
                const std::string& case_name,
                int32_t iteration,
                int32_t block_size)
{
    Row row;
    row.case_name = case_name;
    row.iteration = iteration;
    row.block_size = block_size;

    const cupf_minimal::DumpCaseData case_data =
        cupf_minimal::load_dump_case(options.case_root / case_name);
    const LinearMetadata metadata = make_linear_metadata(case_data);
    const cuiter::CsrMatrix matrix =
        load_cupf_csr_dump(jacobian_path(options.jf_root, case_name, iteration));
    const std::vector<double> rhs =
        load_cupf_vector_dump(rhs_path(options.jf_root, case_name, iteration));
    if (matrix.rows != static_cast<int32_t>(rhs.size()) ||
        matrix.rows != metadata.n_pvpq + metadata.n_pq) {
        throw std::runtime_error("matrix/RHS/metadata dimension mismatch");
    }

    const cuiter::CsrMatrix j11 = extract_j11(matrix, metadata.n_pvpq);
    if (j11.rows != j11.cols || j11.rows != metadata.n_pvpq) {
        throw std::runtime_error("J11 is not square");
    }

    const std::vector<double> dx_full = solve_cudss_once(matrix, rhs);
    const std::vector<double> dx0 =
        solve_block_jacobi_bicgstab(matrix, rhs, block_size, options);
    const std::vector<double> residual0 = residual_vector(matrix, rhs, dx0);
    const std::vector<double> r_p = slice(residual0, 0, metadata.n_pvpq);
    const CudssSolveResult j11_corr = solve_cudss_timed(j11, r_p, options.repeat);

    std::vector<double> dx1 = dx0;
    for (int32_t i = 0; i < metadata.n_pvpq; ++i) {
        dx1[static_cast<std::size_t>(i)] += j11_corr.x[static_cast<std::size_t>(i)];
    }

    const std::vector<double> error_before = subtract_vectors(dx0, dx_full);
    const std::vector<double> error_after = subtract_vectors(dx1, dx_full);
    const double dx_full_norm = norm2(dx_full);

    row.full_n = matrix.rows;
    row.full_nnz = matrix.nnz();
    row.n_pvpq = metadata.n_pvpq;
    row.n_pq = metadata.n_pq;
    row.j11_n = j11.rows;
    row.j11_nnz = j11.nnz();
    row.dx_error_ratio_before = safe_ratio(norm2(error_before), dx_full_norm);
    row.dx_error_ratio_after = safe_ratio(norm2(error_after), dx_full_norm);
    row.dx_error_ratio_delta = row.dx_error_ratio_before - row.dx_error_ratio_after;
    row.theta_error_ratio_before = split_error_ratio(dx0, dx_full, 0, metadata.n_pvpq);
    row.theta_error_ratio_after = split_error_ratio(dx1, dx_full, 0, metadata.n_pvpq);
    row.theta_error_ratio_delta = row.theta_error_ratio_before - row.theta_error_ratio_after;
    row.vmag_error_ratio_before =
        split_error_ratio(dx0, dx_full, metadata.n_pvpq, metadata.n_pq);
    row.vmag_error_ratio_after =
        split_error_ratio(dx1, dx_full, metadata.n_pvpq, metadata.n_pq);
    row.vmag_error_ratio_delta = row.vmag_error_ratio_before - row.vmag_error_ratio_after;
    row.cosine_before = cosine(dx0, dx_full);
    row.cosine_after = cosine(dx1, dx_full);
    const ResidualSplit before = compute_residual_split(matrix, rhs, dx0, metadata);
    const ResidualSplit after = compute_residual_split(matrix, rhs, dx1, metadata);
    row.residual_before = before.total_rel;
    row.residual_after = after.total_rel;
    row.p_residual_abs_before = before.p_abs;
    row.p_residual_abs_after = after.p_abs;
    row.p_residual_rel_before = before.p_rel;
    row.p_residual_rel_after = after.p_rel;
    row.q_residual_abs_before = before.q_abs;
    row.q_residual_abs_after = after.q_abs;
    row.q_residual_rel_before = before.q_rel;
    row.q_residual_rel_after = after.q_rel;
    row.dtheta_corr_norm = norm2(j11_corr.x);
    row.dx0_norm = norm2(dx0);
    row.dx1_norm = norm2(dx1);
    row.dx_full_norm = dx_full_norm;
    row.j11_analyze_ms = j11_corr.analyze_ms;
    row.j11_factor_ms = j11_corr.factor_ms;
    row.j11_solve_ms = j11_corr.solve_ms;
    row.j11_factor_solve_ms = j11_corr.factor_solve_ms;
    row.j11_total_ms = j11_corr.total_ms;
    row.j11_rel_res = j11_corr.rel_res;
    return row;
}

void write_stats_header(std::ofstream& out, const std::string& prefix)
{
    out << prefix << "_mean," << prefix << "_median," << prefix << "_min," << prefix << "_max";
}

void write_stats(std::ofstream& out, const Stats& stats)
{
    out << format_double(stats.mean) << ','
        << format_double(stats.median) << ','
        << format_double(stats.min) << ','
        << format_double(stats.max);
}

void write_csv(const std::filesystem::path& path, const std::vector<Row>& rows)
{
    std::ofstream out(path);
    out << "case,iter,block_size,full_n,full_nnz,n_pvpq,n_pq,j11_n,j11_nnz,"
           "dx_error_ratio_before,dx_error_ratio_after,dx_error_ratio_delta,"
           "theta_error_ratio_before,theta_error_ratio_after,theta_error_ratio_delta,"
           "vmag_error_ratio_before,vmag_error_ratio_after,vmag_error_ratio_delta,"
           "cosine_before,cosine_after,residual_before,residual_after,"
           "P_residual_abs_before,P_residual_abs_after,P_residual_rel_before,"
           "P_residual_rel_after,Q_residual_abs_before,Q_residual_abs_after,"
           "Q_residual_rel_before,Q_residual_rel_after,dtheta_corr_norm,dx0_norm,"
           "dx1_norm,dx_full_norm,";
    write_stats_header(out, "j11_analyze_ms");
    out << ',';
    write_stats_header(out, "j11_factor_ms");
    out << ',';
    write_stats_header(out, "j11_solve_ms");
    out << ',';
    write_stats_header(out, "j11_factor_solve_ms");
    out << ',';
    write_stats_header(out, "j11_total_ms");
    out << ",j11_rel_res,status,error_message\n";

    for (const Row& row : rows) {
        out << row.case_name << ",J" << row.iteration << ',' << row.block_size << ','
            << row.full_n << ',' << row.full_nnz << ',' << row.n_pvpq << ','
            << row.n_pq << ',' << row.j11_n << ',' << row.j11_nnz << ','
            << format_double(row.dx_error_ratio_before) << ','
            << format_double(row.dx_error_ratio_after) << ','
            << format_double(row.dx_error_ratio_delta) << ','
            << format_double(row.theta_error_ratio_before) << ','
            << format_double(row.theta_error_ratio_after) << ','
            << format_double(row.theta_error_ratio_delta) << ','
            << format_double(row.vmag_error_ratio_before) << ','
            << format_double(row.vmag_error_ratio_after) << ','
            << format_double(row.vmag_error_ratio_delta) << ','
            << format_double(row.cosine_before) << ','
            << format_double(row.cosine_after) << ','
            << format_double(row.residual_before) << ','
            << format_double(row.residual_after) << ','
            << format_double(row.p_residual_abs_before) << ','
            << format_double(row.p_residual_abs_after) << ','
            << format_double(row.p_residual_rel_before) << ','
            << format_double(row.p_residual_rel_after) << ','
            << format_double(row.q_residual_abs_before) << ','
            << format_double(row.q_residual_abs_after) << ','
            << format_double(row.q_residual_rel_before) << ','
            << format_double(row.q_residual_rel_after) << ','
            << format_double(row.dtheta_corr_norm) << ','
            << format_double(row.dx0_norm) << ','
            << format_double(row.dx1_norm) << ','
            << format_double(row.dx_full_norm) << ',';
        write_stats(out, row.j11_analyze_ms);
        out << ',';
        write_stats(out, row.j11_factor_ms);
        out << ',';
        write_stats(out, row.j11_solve_ms);
        out << ',';
        write_stats(out, row.j11_factor_solve_ms);
        out << ',';
        write_stats(out, row.j11_total_ms);
        std::string message = row.error_message;
        std::replace(message.begin(), message.end(), ',', ';');
        out << ',' << format_double(row.j11_rel_res)
            << ',' << row.status << ',' << message << '\n';
    }
}

void write_timing_csv(const std::filesystem::path& path, const std::vector<Row>& rows)
{
    std::ofstream out(path);
    out << "case,iter,block_size,j11_n,j11_nnz,";
    write_stats_header(out, "j11_analyze_ms");
    out << ',';
    write_stats_header(out, "j11_factor_ms");
    out << ',';
    write_stats_header(out, "j11_solve_ms");
    out << ',';
    write_stats_header(out, "j11_factor_solve_ms");
    out << ',';
    write_stats_header(out, "j11_total_ms");
    out << ",j11_rel_res,status,error_message\n";
    for (const Row& row : rows) {
        out << row.case_name << ",J" << row.iteration << ',' << row.block_size << ','
            << row.j11_n << ',' << row.j11_nnz << ',';
        write_stats(out, row.j11_analyze_ms);
        out << ',';
        write_stats(out, row.j11_factor_ms);
        out << ',';
        write_stats(out, row.j11_solve_ms);
        out << ',';
        write_stats(out, row.j11_factor_solve_ms);
        out << ',';
        write_stats(out, row.j11_total_ms);
        std::string message = row.error_message;
        std::replace(message.begin(), message.end(), ',', ';');
        out << ',' << format_double(row.j11_rel_res)
            << ',' << row.status << ',' << message << '\n';
    }
}

double mean_metric(const std::vector<Row>& rows, double Row::*member, int32_t iteration = -1)
{
    double sum = 0.0;
    int32_t count = 0;
    for (const Row& row : rows) {
        if (row.status == "ok" && (iteration < 0 || row.iteration == iteration) &&
            std::isfinite(row.*member)) {
            sum += row.*member;
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : kNan;
}

double mean_j11_factor_solve_ms(const std::vector<Row>& rows, int32_t iteration = -1)
{
    double sum = 0.0;
    int32_t count = 0;
    for (const Row& row : rows) {
        if (row.status == "ok" && (iteration < 0 || row.iteration == iteration) &&
            std::isfinite(row.j11_factor_solve_ms.mean)) {
            sum += row.j11_factor_solve_ms.mean;
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : kNan;
}

int32_t count_improved(const std::vector<Row>& rows, double Row::*before, double Row::*after)
{
    int32_t count = 0;
    for (const Row& row : rows) {
        if (row.status == "ok" && std::isfinite(row.*before) && std::isfinite(row.*after) &&
            row.*after < row.*before) {
            ++count;
        }
    }
    return count;
}

int32_t count_ok(const std::vector<Row>& rows)
{
    return static_cast<int32_t>(std::count_if(rows.begin(), rows.end(), [](const Row& row) {
        return row.status == "ok";
    }));
}

void write_report(const std::filesystem::path& path, const std::vector<Row>& rows)
{
    std::ofstream out(path);
    const int32_t ok = count_ok(rows);
    const double dx_before = mean_metric(rows, &Row::dx_error_ratio_before, 1);
    const double dx_after = mean_metric(rows, &Row::dx_error_ratio_after, 1);
    const double theta_before = mean_metric(rows, &Row::theta_error_ratio_before, 1);
    const double theta_after = mean_metric(rows, &Row::theta_error_ratio_after, 1);
    const double vmag_before = mean_metric(rows, &Row::vmag_error_ratio_before, 1);
    const double vmag_after = mean_metric(rows, &Row::vmag_error_ratio_after, 1);
    const double res_before = mean_metric(rows, &Row::residual_before, 1);
    const double res_after = mean_metric(rows, &Row::residual_after, 1);
    const double p_before = mean_metric(rows, &Row::p_residual_rel_before, 1);
    const double p_after = mean_metric(rows, &Row::p_residual_rel_after, 1);
    const double q_before = mean_metric(rows, &Row::q_residual_rel_before, 1);
    const double q_after = mean_metric(rows, &Row::q_residual_rel_after, 1);
    const double j11_factor_solve = mean_j11_factor_solve_ms(rows, 1);

    out << "# J11 Correction Quality Bench\n\n";
    out << "## Answers\n\n";
    out << "1. Successful measurements: `" << ok << "/" << rows.size() << "` rows.\n";
    out << "2. J1 dx error ratio: before=`" << format_double(dx_before)
        << "`, after=`" << format_double(dx_after) << "`. Improved rows overall=`"
        << count_improved(rows, &Row::dx_error_ratio_before, &Row::dx_error_ratio_after)
        << "/" << ok << "`.\n";
    out << "3. J1 theta error ratio: before=`" << format_double(theta_before)
        << "`, after=`" << format_double(theta_after) << "`.\n";
    out << "4. J1 |V| error ratio: before=`" << format_double(vmag_before)
        << "`, after=`" << format_double(vmag_after)
        << "`. J11 correction does not directly change |V|, so this should stay fixed.\n";
    out << "5. J1 residual ratio: total before=`" << format_double(res_before)
        << "`, after=`" << format_double(res_after) << "`; P before=`"
        << format_double(p_before) << "`, after=`" << format_double(p_after)
        << "`; Q before=`" << format_double(q_before) << "`, after=`"
        << format_double(q_after) << "`.\n";
    out << "6. J1 J11 cuDSS factorize+solve mean=`"
        << format_double(j11_factor_solve) << "` ms.\n\n";

    out << "## Iteration Variation\n\n";
    out << "| iter | dx err before | dx err after | P rel before | P rel after | "
           "Q rel before | Q rel after |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|\n";
    for (int32_t iteration : {0, 1, 2}) {
        out << "|J" << iteration << '|'
            << format_double(mean_metric(rows, &Row::dx_error_ratio_before, iteration)) << '|'
            << format_double(mean_metric(rows, &Row::dx_error_ratio_after, iteration)) << '|'
            << format_double(mean_metric(rows, &Row::p_residual_rel_before, iteration)) << '|'
            << format_double(mean_metric(rows, &Row::p_residual_rel_after, iteration)) << '|'
            << format_double(mean_metric(rows, &Row::q_residual_rel_before, iteration)) << '|'
            << format_double(mean_metric(rows, &Row::q_residual_rel_after, iteration)) << "|\n";
    }

    out << "## J1 Snapshot\n\n";
    out << "| case | bs | dx err before | dx err after | theta before | theta after | "
           "P rel before | P rel after | Q rel before | Q rel after | J11 f+s ms |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const Row& row : rows) {
        if (row.status != "ok" || row.iteration != 1) {
            continue;
        }
        out << '|' << row.case_name << '|' << row.block_size << '|'
            << format_double(row.dx_error_ratio_before) << '|'
            << format_double(row.dx_error_ratio_after) << '|'
            << format_double(row.theta_error_ratio_before) << '|'
            << format_double(row.theta_error_ratio_after) << '|'
            << format_double(row.p_residual_rel_before) << '|'
            << format_double(row.p_residual_rel_after) << '|'
            << format_double(row.q_residual_rel_before) << '|'
            << format_double(row.q_residual_rel_after) << '|'
            << format_double(row.j11_factor_solve_ms.mean) << "|\n";
    }
    out << "\n## Notes\n\n";
    out << "- Baseline is CPU-pilot BiCGSTAB(2) + METIS block-Jacobi, no coarse/scaling/warm-start/block-ILU.\n";
    out << "- Correction solves `J11 dtheta_corr = F_P - J_P dx0` with cuDSS and leaves |V| unchanged.\n";
    out << "- This is a standalone J/F quality test only; no hybrid NR policy is run.\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        std::filesystem::create_directories(options.output_dir);
        std::vector<Row> rows;
        for (const std::string& case_name : options.cases) {
            for (int32_t iteration : options.iterations) {
                const auto j_path = jacobian_path(options.jf_root, case_name, iteration);
                const auto f_path = rhs_path(options.jf_root, case_name, iteration);
                if (!std::filesystem::exists(j_path) || !std::filesystem::exists(f_path)) {
                    if (options.allow_missing) {
                        std::cerr << "[skip] missing J/F for " << case_name << " J"
                                  << iteration << '\n';
                        continue;
                    }
                    throw std::runtime_error("missing J/F for " + case_name + " J" +
                                             std::to_string(iteration));
                }
                for (int32_t block_size : options.block_sizes) {
                    std::cout << "[case] " << case_name << " J" << iteration
                              << " bs=" << block_size << '\n';
                    Row row;
                    try {
                        row = analyze_one(options, case_name, iteration, block_size);
                    } catch (const std::exception& ex) {
                        row.case_name = case_name;
                        row.iteration = iteration;
                        row.block_size = block_size;
                        row.status = "error";
                        row.error_message = ex.what();
                        std::cerr << "  [error] " << ex.what() << '\n';
                    }
                    rows.push_back(std::move(row));
                }
            }
        }
        write_csv(options.output_dir / "j11_correction_quality.csv", rows);
        write_timing_csv(options.output_dir / "j11_correction_timing.csv", rows);
        write_report(options.output_dir / "j11_correction_report.md", rows);
        std::cout << "[done] wrote J11 correction bench results to "
                  << options.output_dir << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
