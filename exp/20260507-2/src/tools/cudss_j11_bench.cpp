#include "cuiter/common/cuda_utils.hpp"
#include "cuiter/core/csr_matrix.hpp"
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
    std::filesystem::path output_dir = "results/j11_cudss_bench";
    std::vector<std::string> cases = {
        "case2383wp",
        "case3120sp",
        "case9241pegase",
        "case13659pegase",
        "case6468rte",
    };
    std::vector<int32_t> iterations = {0, 1, 2};
    int32_t repeat = 5;
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

struct Measurement {
    int32_t n = 0;
    int32_t nnz = 0;
    Stats analyze_ms;
    Stats factor_ms;
    Stats solve_ms;
    Stats factor_solve_ms;
    Stats total_ms;
    double rel_res = kNan;
};

struct ResultRow {
    std::string case_name;
    int32_t iteration = 0;
    Measurement full;
    Measurement j11;
    double ratio_n = kNan;
    double ratio_nnz = kNan;
    double ratio_analyze = kNan;
    double ratio_factor = kNan;
    double ratio_solve = kNan;
    double ratio_factor_solve = kNan;
    double ratio_total = kNan;
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

std::vector<int32_t> parse_iteration_list(const std::string& text)
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
        << "  --repeat 5\n"
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
            options.iterations = parse_iteration_list(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            options.repeat = std::stoi(argv[++i]);
        } else if (arg == "--allow-missing") {
            options.allow_missing = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.cases.empty() || options.iterations.empty()) {
        throw std::runtime_error("case and iteration lists must be nonempty");
    }
    if (options.repeat <= 0) {
        throw std::runtime_error("--repeat must be positive");
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

double norm2(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values) {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
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

double relative_residual(const cuiter::CsrMatrix& matrix,
                         const std::vector<double>& rhs,
                         const std::vector<double>& x)
{
    std::vector<double> residual = rhs;
    const std::vector<double> ax = spmv(matrix, x);
    for (std::size_t i = 0; i < residual.size(); ++i) {
        residual[i] -= ax[i];
    }
    return norm2(residual) / std::max(norm2(rhs), std::numeric_limits<double>::min());
}

double safe_ratio(double numerator, double denominator)
{
    return denominator > 0.0 ? numerator / denominator : kNan;
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
    if (n_pvpq <= 0 || n_pvpq > full.rows) {
        throw std::runtime_error("invalid n_pvpq for J11 extraction");
    }
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

std::vector<double> extract_fp_rhs(const std::vector<double>& full_rhs, int32_t n_pvpq)
{
    if (n_pvpq <= 0 || n_pvpq > static_cast<int32_t>(full_rhs.size())) {
        throw std::runtime_error("invalid n_pvpq for FP extraction");
    }
    return std::vector<double>(full_rhs.begin(), full_rhs.begin() + n_pvpq);
}

Measurement measure_cudss(const cuiter::CsrMatrix& matrix,
                          const std::vector<double>& rhs,
                          int32_t repeat)
{
    if (matrix.rows != matrix.cols || matrix.rows != static_cast<int32_t>(rhs.size())) {
        throw std::runtime_error("cuDSS measurement dimension mismatch");
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
    d_x.resize(rhs.size());

    std::vector<double> analyze_ms;
    std::vector<double> factor_ms;
    std::vector<double> solve_ms;
    std::vector<double> factor_solve_ms;
    std::vector<double> total_ms;
    std::vector<double> last_x(rhs.size(), 0.0);
    analyze_ms.reserve(static_cast<std::size_t>(repeat));
    factor_ms.reserve(static_cast<std::size_t>(repeat));
    solve_ms.reserve(static_cast<std::size_t>(repeat));
    factor_solve_ms.reserve(static_cast<std::size_t>(repeat));
    total_ms.reserve(static_cast<std::size_t>(repeat));

    for (int32_t r = 0; r < repeat; ++r) {
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
        if (r == repeat - 1) {
            d_x.copy_to(last_x.data(), last_x.size());
        }
    }

    Measurement result;
    result.n = matrix.rows;
    result.nnz = matrix.nnz();
    result.analyze_ms = summarize(analyze_ms);
    result.factor_ms = summarize(factor_ms);
    result.solve_ms = summarize(solve_ms);
    result.factor_solve_ms = summarize(factor_solve_ms);
    result.total_ms = summarize(total_ms);
    result.rel_res = relative_residual(matrix, rhs, last_x);
    return result;
}

ResultRow measure_one(const CliOptions& options,
                      const std::string& case_name,
                      int32_t iteration)
{
    ResultRow row;
    row.case_name = case_name;
    row.iteration = iteration;
    const auto j_path = jacobian_path(options.jf_root, case_name, iteration);
    const auto f_path = rhs_path(options.jf_root, case_name, iteration);
    const cupf_minimal::DumpCaseData case_data =
        cupf_minimal::load_dump_case(options.case_root / case_name);
    const LinearMetadata metadata = make_linear_metadata(case_data);
    const cuiter::CsrMatrix full = load_cupf_csr_dump(j_path);
    const std::vector<double> rhs = load_cupf_vector_dump(f_path);
    if (full.rows != static_cast<int32_t>(rhs.size())) {
        throw std::runtime_error("full J/F dimension mismatch");
    }
    if (metadata.n_pvpq + metadata.n_pq != full.rows) {
        throw std::runtime_error("metadata dimension does not match full Jacobian");
    }
    if (metadata.n_pvpq <= 0) {
        throw std::runtime_error("J11 is empty");
    }

    const cuiter::CsrMatrix j11 = extract_j11(full, metadata.n_pvpq);
    const std::vector<double> fp = extract_fp_rhs(rhs, metadata.n_pvpq);
    if (j11.rows != metadata.n_pvpq || j11.cols != metadata.n_pvpq) {
        throw std::runtime_error("J11 is not square");
    }

    row.full = measure_cudss(full, rhs, options.repeat);
    row.j11 = measure_cudss(j11, fp, options.repeat);
    row.ratio_n = safe_ratio(static_cast<double>(row.j11.n), static_cast<double>(row.full.n));
    row.ratio_nnz =
        safe_ratio(static_cast<double>(row.j11.nnz), static_cast<double>(row.full.nnz));
    row.ratio_analyze = safe_ratio(row.j11.analyze_ms.mean, row.full.analyze_ms.mean);
    row.ratio_factor = safe_ratio(row.j11.factor_ms.mean, row.full.factor_ms.mean);
    row.ratio_solve = safe_ratio(row.j11.solve_ms.mean, row.full.solve_ms.mean);
    row.ratio_factor_solve =
        safe_ratio(row.j11.factor_solve_ms.mean, row.full.factor_solve_ms.mean);
    row.ratio_total = safe_ratio(row.j11.total_ms.mean, row.full.total_ms.mean);
    return row;
}

void write_stat_columns(std::ofstream& out, const std::string& prefix)
{
    out << prefix << "_mean," << prefix << "_median," << prefix << "_min," << prefix << "_max";
}

void write_stat_values(std::ofstream& out, const Stats& stats)
{
    out << format_double(stats.mean) << ','
        << format_double(stats.median) << ','
        << format_double(stats.min) << ','
        << format_double(stats.max);
}

void write_csv(const std::filesystem::path& path, const std::vector<ResultRow>& rows)
{
    std::ofstream out(path);
    out << "case,iter,full_n,full_nnz,j11_n,j11_nnz,";
    write_stat_columns(out, "full_analyze_ms");
    out << ',';
    write_stat_columns(out, "full_factor_ms");
    out << ',';
    write_stat_columns(out, "full_solve_ms");
    out << ',';
    write_stat_columns(out, "full_factor_solve_ms");
    out << ',';
    write_stat_columns(out, "full_total_ms");
    out << ',';
    write_stat_columns(out, "j11_analyze_ms");
    out << ',';
    write_stat_columns(out, "j11_factor_ms");
    out << ',';
    write_stat_columns(out, "j11_solve_ms");
    out << ',';
    write_stat_columns(out, "j11_factor_solve_ms");
    out << ',';
    write_stat_columns(out, "j11_total_ms");
    out << ",full_rel_res,j11_rel_res,ratio_n,ratio_nnz,ratio_analyze,ratio_factor,"
           "ratio_solve,ratio_factor_solve,ratio_total,status,error_message\n";

    for (const ResultRow& row : rows) {
        out << row.case_name << ",J" << row.iteration << ','
            << row.full.n << ',' << row.full.nnz << ','
            << row.j11.n << ',' << row.j11.nnz << ',';
        write_stat_values(out, row.full.analyze_ms);
        out << ',';
        write_stat_values(out, row.full.factor_ms);
        out << ',';
        write_stat_values(out, row.full.solve_ms);
        out << ',';
        write_stat_values(out, row.full.factor_solve_ms);
        out << ',';
        write_stat_values(out, row.full.total_ms);
        out << ',';
        write_stat_values(out, row.j11.analyze_ms);
        out << ',';
        write_stat_values(out, row.j11.factor_ms);
        out << ',';
        write_stat_values(out, row.j11.solve_ms);
        out << ',';
        write_stat_values(out, row.j11.factor_solve_ms);
        out << ',';
        write_stat_values(out, row.j11.total_ms);
        out << ',' << format_double(row.full.rel_res)
            << ',' << format_double(row.j11.rel_res)
            << ',' << format_double(row.ratio_n)
            << ',' << format_double(row.ratio_nnz)
            << ',' << format_double(row.ratio_analyze)
            << ',' << format_double(row.ratio_factor)
            << ',' << format_double(row.ratio_solve)
            << ',' << format_double(row.ratio_factor_solve)
            << ',' << format_double(row.ratio_total)
            << ',' << row.status << ',';
        std::string message = row.error_message;
        std::replace(message.begin(), message.end(), ',', ';');
        out << message << '\n';
    }
}

double mean_metric(const std::vector<ResultRow>& rows, double ResultRow::*member)
{
    double sum = 0.0;
    int32_t count = 0;
    for (const ResultRow& row : rows) {
        if (row.status == "ok" && std::isfinite(row.*member)) {
            sum += row.*member;
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : kNan;
}

double mean_metric_for_iter(const std::vector<ResultRow>& rows,
                            int32_t iteration,
                            double ResultRow::*member)
{
    double sum = 0.0;
    int32_t count = 0;
    for (const ResultRow& row : rows) {
        if (row.status == "ok" && row.iteration == iteration && std::isfinite(row.*member)) {
            sum += row.*member;
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : kNan;
}

double median_metric(const std::vector<ResultRow>& rows, double ResultRow::*member)
{
    std::vector<double> values;
    for (const ResultRow& row : rows) {
        if (row.status == "ok" && std::isfinite(row.*member)) {
            values.push_back(row.*member);
        }
    }
    if (values.empty()) {
        return kNan;
    }
    std::sort(values.begin(), values.end());
    if (values.size() % 2 == 0) {
        const std::size_t hi = values.size() / 2;
        return 0.5 * (values[hi - 1] + values[hi]);
    }
    return values[values.size() / 2];
}

double max_iter_variation(const std::vector<ResultRow>& rows)
{
    double worst = 0.0;
    for (const ResultRow& base : rows) {
        if (base.status != "ok" || base.iteration != 1) {
            continue;
        }
        double min_ratio = base.ratio_factor_solve;
        double max_ratio = base.ratio_factor_solve;
        for (const ResultRow& row : rows) {
            if (row.status == "ok" && row.case_name == base.case_name) {
                min_ratio = std::min(min_ratio, row.ratio_factor_solve);
                max_ratio = std::max(max_ratio, row.ratio_factor_solve);
            }
        }
        worst = std::max(worst, max_ratio - min_ratio);
    }
    return worst;
}

void write_report(const std::filesystem::path& path, const std::vector<ResultRow>& rows)
{
    std::ofstream out(path);
    int32_t ok_count = 0;
    for (const ResultRow& row : rows) {
        if (row.status == "ok") {
            ++ok_count;
        }
    }
    const double ratio_n = mean_metric(rows, &ResultRow::ratio_n);
    const double ratio_nnz = mean_metric(rows, &ResultRow::ratio_nnz);
    const double ratio_factor_solve = mean_metric(rows, &ResultRow::ratio_factor_solve);
    const double ratio_factor_solve_median =
        median_metric(rows, &ResultRow::ratio_factor_solve);
    const double ratio_factor_solve_j1 =
        mean_metric_for_iter(rows, 1, &ResultRow::ratio_factor_solve);
    const double ratio_total = mean_metric(rows, &ResultRow::ratio_total);
    const double ratio_solve = mean_metric(rows, &ResultRow::ratio_solve);
    const double ratio_solve_j1 = mean_metric_for_iter(rows, 1, &ResultRow::ratio_solve);
    const double ratio_total_j1 = mean_metric_for_iter(rows, 1, &ResultRow::ratio_total);
    const double variation = max_iter_variation(rows);

    std::string decision;
    if (ratio_factor_solve_j1 < 0.40) {
        decision = "worth testing next";
    } else if (ratio_factor_solve_j1 <= 0.70) {
        decision = "maybe worth testing only if it strongly reduces fallback";
    } else {
        decision = "probably not worth it";
    }

    out << "# J11 cuDSS Cost Bench\n\n";
    out << "## Answers\n\n";
    out << "1. J11 square for selected cases: `" << ok_count << "/" << rows.size()
        << "` measured rows succeeded. Failed rows, if any, are marked in the CSV.\n";
    out << "2. J11 size versus full J: mean n ratio=`" << format_double(ratio_n)
        << "`, mean nnz ratio=`" << format_double(ratio_nnz) << "`.\n";
    out << "3. J11 factorize+solve cheaper than full J: J1 mean ratio=`"
        << format_double(ratio_factor_solve_j1) << "`, all J0/J1/J2 mean=`"
        << format_double(ratio_factor_solve) << "`, all median=`"
        << format_double(ratio_factor_solve_median) << "`.\n";
    out << "4. J11 solve-only cheaper than full J solve: J1 mean ratio=`"
        << format_double(ratio_solve_j1) << "`, all mean=`"
        << format_double(ratio_solve) << "`. Including analysis, J1 total ratio=`"
        << format_double(ratio_total_j1) << "`, all total ratio=`"
        << format_double(ratio_total) << "`.\n";
    out << "5. J0/J1/J2 timing variation: worst within-case factor+solve ratio spread=`"
        << format_double(variation) << "`.\n";
    out << "6. Decision by requested rule: `" << decision << "`.\n\n";

    out << "## J1 Snapshot\n\n";
    out << "| case | ratio n | ratio nnz | ratio factor+solve | ratio solve | full f+s ms | J11 f+s ms |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|\n";
    for (const ResultRow& row : rows) {
        if (row.status != "ok" || row.iteration != 1) {
            continue;
        }
        out << '|' << row.case_name << '|'
            << format_double(row.ratio_n) << '|'
            << format_double(row.ratio_nnz) << '|'
            << format_double(row.ratio_factor_solve) << '|'
            << format_double(row.ratio_solve) << '|'
            << format_double(row.full.factor_solve_ms.mean) << '|'
            << format_double(row.j11.factor_solve_ms.mean) << "|\n";
    }
    out << "\n## Notes\n\n";
    out << "- Timings exclude dump parsing and H2D buffer creation for both full J and J11.\n";
    out << "- Each measurement creates a fresh cuDSS solver object and measures analysis, "
           "factorization, and solve phases separately.\n";
    out << "- J11 is extracted as the existing field ordering top-left `P-theta` block. "
           "No NR update or hybrid integration is performed.\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        std::filesystem::create_directories(options.output_dir);
        std::vector<ResultRow> rows;
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
                std::cout << "[case] " << case_name << " J" << iteration << '\n';
                ResultRow row;
                try {
                    row = measure_one(options, case_name, iteration);
                } catch (const std::exception& ex) {
                    row.case_name = case_name;
                    row.iteration = iteration;
                    row.status = "error";
                    row.error_message = ex.what();
                    std::cerr << "  [error] " << ex.what() << '\n';
                }
                rows.push_back(std::move(row));
            }
        }
        write_csv(options.output_dir / "j11_cudss_timing.csv", rows);
        write_report(options.output_dir / "j11_cudss_report.md", rows);
        std::cout << "[done] wrote J11 cuDSS bench results to " << options.output_dir << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
