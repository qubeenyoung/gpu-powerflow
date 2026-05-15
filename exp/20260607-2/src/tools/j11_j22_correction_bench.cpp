#include "cuiter/common/cuda_utils.hpp"
#include "cuiter/core/csr_matrix.hpp"
#include "cuiter/solver/cpu_block_ilu0_pilot.hpp"
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
    std::filesystem::path output_dir = "results";
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
    double total_rel = kNan;
    double p_rel = kNan;
    double q_rel = kNan;
};

struct CaseIterData {
    cuiter::CsrMatrix matrix;
    std::vector<double> rhs;
    LinearMetadata metadata;
    cuiter::CsrMatrix j11;
    cuiter::CsrMatrix j22;
    CudssSolveResult full;
};

struct Row {
    std::string case_name;
    int32_t iteration = 0;
    int32_t block_size = 0;
    std::string variant;
    int32_t full_n = 0;
    int32_t full_nnz = 0;
    int32_t j11_n = 0;
    int32_t j11_nnz = 0;
    int32_t j22_n = 0;
    int32_t j22_nnz = 0;
    double dx_error_ratio_before = kNan;
    double dx_error_ratio_after = kNan;
    double theta_error_ratio_before = kNan;
    double theta_error_ratio_after = kNan;
    double vmag_error_ratio_before = kNan;
    double vmag_error_ratio_after = kNan;
    double cosine_before = kNan;
    double cosine_after = kNan;
    double full_residual_before = kNan;
    double full_residual_after = kNan;
    double p_residual_before = kNan;
    double p_residual_after = kNan;
    double q_residual_before = kNan;
    double q_residual_after = kNan;
    double full_j_factor_solve_ms = kNan;
    double full_j_solve_ms = kNan;
    double j11_factor_solve_ms = 0.0;
    double j11_solve_ms = 0.0;
    double j22_factor_solve_ms = 0.0;
    double j22_solve_ms = 0.0;
    double correction_wall_ms = 0.0;
    double correction_serial_sum_ms = 0.0;
    double correction_over_full = kNan;
    double j11_rel_res = kNan;
    double j22_rel_res = kNan;
    double dtheta_norm = 0.0;
    double dvm_norm = 0.0;
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
    if (!in || matrix.rows <= 0 || matrix.rows != matrix.cols ||
        matrix.row_ptr.front() != 0 || matrix.row_ptr.back() != nnz) {
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

std::vector<double> negate_vector(const std::vector<double>& values)
{
    std::vector<double> out(values.size(), 0.0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = -values[i];
    }
    return out;
}

std::vector<double> slice(const std::vector<double>& values, int32_t begin, int32_t count)
{
    return std::vector<double>(values.begin() + begin, values.begin() + begin + count);
}

std::vector<double> submatrix_vector_product(const cuiter::CsrMatrix& matrix,
                                             int32_t row_begin,
                                             int32_t row_count,
                                             int32_t col_begin,
                                             int32_t col_count,
                                             const std::vector<double>& x)
{
    std::vector<double> y(static_cast<std::size_t>(row_count), 0.0);
    for (int32_t local_row = 0; local_row < row_count; ++local_row) {
        const int32_t row = row_begin + local_row;
        double sum = 0.0;
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = matrix.col_idx[static_cast<std::size_t>(pos)];
            if (col >= col_begin && col < col_begin + col_count) {
                sum += matrix.values[static_cast<std::size_t>(pos)] *
                       x[static_cast<std::size_t>(col - col_begin)];
            }
        }
        y[static_cast<std::size_t>(local_row)] = sum;
    }
    return y;
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

cuiter::CsrMatrix extract_submatrix(const cuiter::CsrMatrix& full,
                                    int32_t row_begin,
                                    int32_t row_count,
                                    int32_t col_begin,
                                    int32_t col_count)
{
    cuiter::CsrMatrix out;
    out.rows = row_count;
    out.cols = col_count;
    out.row_ptr.assign(static_cast<std::size_t>(row_count + 1), 0);
    for (int32_t local_row = 0; local_row < row_count; ++local_row) {
        const int32_t row = row_begin + local_row;
        for (int32_t pos = full.row_ptr[static_cast<std::size_t>(row)];
             pos < full.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = full.col_idx[static_cast<std::size_t>(pos)];
            if (col >= col_begin && col < col_begin + col_count) {
                out.col_idx.push_back(col - col_begin);
                out.values.push_back(full.values[static_cast<std::size_t>(pos)]);
            }
        }
        out.row_ptr[static_cast<std::size_t>(local_row + 1)] =
            static_cast<int32_t>(out.values.size());
    }
    return out;
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

    cupf_minimal::DirectCudssSolver solver;
    solver.initialize(matrix,
                      d_row_ptr.data(),
                      d_col_idx.data(),
                      d_values.data(),
                      d_rhs.data(),
                      d_x.data());
    const double analyze_ms = 1000.0 * solver.analyze();
    std::vector<double> factor_ms;
    std::vector<double> solve_ms;
    std::vector<double> factor_solve_ms;
    std::vector<double> total_ms;
    std::vector<double> last_x(rhs.size(), 0.0);
    for (int32_t i = 0; i < repeat; ++i) {
        d_x.memset_zero();
        const double factor = 1000.0 * solver.factorize();
        const double solve = 1000.0 * solver.solve();
        factor_ms.push_back(factor);
        solve_ms.push_back(solve);
        factor_solve_ms.push_back(factor + solve);
        total_ms.push_back(analyze_ms + factor + solve);
        if (i == repeat - 1) {
            d_x.copy_to(last_x.data(), last_x.size());
        }
    }

    CudssSolveResult result;
    result.x = std::move(last_x);
    result.analyze_ms = summarize(std::vector<double>{analyze_ms});
    result.factor_ms = summarize(factor_ms);
    result.solve_ms = summarize(solve_ms);
    result.factor_solve_ms = summarize(factor_solve_ms);
    result.total_ms = summarize(total_ms);
    result.rel_res =
        safe_ratio(norm2(residual_vector(matrix, rhs, result.x)), norm2(rhs));
    return result;
}

struct ParallelPhaseStats {
    double wall_ms = 0.0;
    double left_ms = 0.0;
    double right_ms = 0.0;
};

template <typename LeftFn, typename RightFn>
ParallelPhaseStats run_two_stream_phase(cudaStream_t left_stream,
                                        LeftFn&& left_fn,
                                        cudaStream_t right_stream,
                                        RightFn&& right_fn)
{
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t left_start = nullptr;
    cudaEvent_t left_stop = nullptr;
    cudaEvent_t right_start = nullptr;
    cudaEvent_t right_stop = nullptr;
    CUITER_CUDA_CHECK(cudaEventCreate(&left_start));
    CUITER_CUDA_CHECK(cudaEventCreate(&left_stop));
    CUITER_CUDA_CHECK(cudaEventCreate(&right_start));
    CUITER_CUDA_CHECK(cudaEventCreate(&right_stop));
    const auto wall_start = std::chrono::steady_clock::now();
    CUITER_CUDA_CHECK(cudaEventRecord(left_start, left_stream));
    left_fn();
    CUITER_CUDA_CHECK(cudaEventRecord(left_stop, left_stream));
    CUITER_CUDA_CHECK(cudaEventRecord(right_start, right_stream));
    right_fn();
    CUITER_CUDA_CHECK(cudaEventRecord(right_stop, right_stream));
    CUITER_CUDA_CHECK(cudaEventSynchronize(left_stop));
    CUITER_CUDA_CHECK(cudaEventSynchronize(right_stop));
    const auto wall_stop = std::chrono::steady_clock::now();
    float left_ms = 0.0F;
    float right_ms = 0.0F;
    CUITER_CUDA_CHECK(cudaEventElapsedTime(&left_ms, left_start, left_stop));
    CUITER_CUDA_CHECK(cudaEventElapsedTime(&right_ms, right_start, right_stop));
    CUITER_CUDA_CHECK(cudaEventDestroy(left_start));
    CUITER_CUDA_CHECK(cudaEventDestroy(left_stop));
    CUITER_CUDA_CHECK(cudaEventDestroy(right_start));
    CUITER_CUDA_CHECK(cudaEventDestroy(right_stop));
    ParallelPhaseStats stats;
    stats.wall_ms = 1000.0 *
                    std::chrono::duration<double>(wall_stop - wall_start).count();
    stats.left_ms = static_cast<double>(left_ms);
    stats.right_ms = static_cast<double>(right_ms);
    return stats;
}

struct PairCorrectionResult {
    CudssSolveResult j11;
    CudssSolveResult j22;
    std::vector<double> dtheta;
    std::vector<double> dvm;
    Stats wall_ms;
    Stats serial_sum_ms;
};

struct TimedCudssInstance {
    cuiter::DeviceBuffer<int32_t> d_row_ptr;
    cuiter::DeviceBuffer<int32_t> d_col_idx;
    cuiter::DeviceBuffer<double> d_values;
    cuiter::DeviceBuffer<double> d_rhs;
    cuiter::DeviceBuffer<double> d_x;
    cupf_minimal::DirectCudssSolver solver;
    cudaStream_t stream = nullptr;
    int32_t n = 0;

    void initialize(const cuiter::CsrMatrix& matrix)
    {
        n = matrix.rows;
        d_row_ptr.assign(matrix.row_ptr.data(), matrix.row_ptr.size());
        d_col_idx.assign(matrix.col_idx.data(), matrix.col_idx.size());
        d_values.assign(matrix.values.data(), matrix.values.size());
        std::vector<double> zeros(static_cast<std::size_t>(matrix.rows), 0.0);
        d_rhs.assign(zeros.data(), zeros.size());
        d_x.resize(zeros.size());
        CUITER_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        solver.initialize(matrix,
                          d_row_ptr.data(),
                          d_col_idx.data(),
                          d_values.data(),
                          d_rhs.data(),
                          d_x.data());
        solver.set_stream(stream);
    }

    void destroy_stream()
    {
        solver.set_stream(nullptr);
        if (stream != nullptr) {
            CUITER_CUDA_CHECK(cudaStreamDestroy(stream));
            stream = nullptr;
        }
    }

    void assign_values_and_rhs(const std::vector<double>& values,
                               const std::vector<double>& rhs)
    {
        d_values.assign(values.data(), values.size());
        d_rhs.assign(rhs.data(), rhs.size());
        d_x.memset_zero();
    }

    void assign_rhs(const std::vector<double>& rhs)
    {
        d_rhs.assign(rhs.data(), rhs.size());
        d_x.memset_zero();
    }

    std::vector<double> copy_x() const
    {
        std::vector<double> x(static_cast<std::size_t>(n), 0.0);
        d_x.copy_to(x.data(), x.size());
        return x;
    }
};

PairCorrectionResult solve_pair_parallel(const cuiter::CsrMatrix& j11,
                                         const cuiter::CsrMatrix& j22,
                                         const std::vector<double>& r_p,
                                         const std::vector<double>& r_q,
                                         const cuiter::CsrMatrix& full,
                                         const LinearMetadata& metadata,
                                         bool two_round,
                                         int32_t repeat)
{
    TimedCudssInstance left;
    TimedCudssInstance right;
    left.initialize(j11);
    right.initialize(j22);
    const double j11_analyze = 1000.0 * left.solver.analyze();
    const double j22_analyze = 1000.0 * right.solver.analyze();

    std::vector<double> j11_factor_ms;
    std::vector<double> j22_factor_ms;
    std::vector<double> j11_solve_ms;
    std::vector<double> j22_solve_ms;
    std::vector<double> j11_factor_solve_ms;
    std::vector<double> j22_factor_solve_ms;
    std::vector<double> wall_ms;
    std::vector<double> serial_sum_ms;

    std::vector<double> dtheta;
    std::vector<double> dvm;
    for (int32_t i = 0; i < repeat; ++i) {
        const auto repeat_start = std::chrono::steady_clock::now();
        left.assign_values_and_rhs(j11.values, r_p);
        right.assign_values_and_rhs(j22.values, r_q);
        const ParallelPhaseStats factor0 = run_two_stream_phase(
            left.stream,
            [&] { left.solver.factorize_async(); },
            right.stream,
            [&] { right.solver.factorize_async(); });
        const ParallelPhaseStats solve0 = run_two_stream_phase(
            left.stream,
            [&] { left.solver.solve_async(); },
            right.stream,
            [&] { right.solver.solve_async(); });
        dtheta = left.copy_x();
        dvm = right.copy_x();

        double left_solve = solve0.left_ms;
        double right_solve = solve0.right_ms;
        double left_factor = factor0.left_ms;
        double right_factor = factor0.right_ms;

        if (two_round) {
            const std::vector<double> j12_dv =
                submatrix_vector_product(full, 0, metadata.n_pvpq, metadata.n_pvpq, metadata.n_pq, dvm);
            const std::vector<double> j21_dtheta =
                submatrix_vector_product(full, metadata.n_pvpq, metadata.n_pq, 0, metadata.n_pvpq, dtheta);
            std::vector<double> r_p1(j12_dv.size(), 0.0);
            std::vector<double> r_q1(j21_dtheta.size(), 0.0);
            for (std::size_t k = 0; k < j12_dv.size(); ++k) {
                r_p1[k] = -j12_dv[k];
            }
            for (std::size_t k = 0; k < j21_dtheta.size(); ++k) {
                r_q1[k] = -j21_dtheta[k];
            }
            left.assign_rhs(r_p1);
            right.assign_rhs(r_q1);
            const ParallelPhaseStats solve1 = run_two_stream_phase(
                left.stream,
                [&] { left.solver.solve_async(); },
                right.stream,
                [&] { right.solver.solve_async(); });
            const std::vector<double> dtheta1 = left.copy_x();
            const std::vector<double> dvm1 = right.copy_x();
            for (std::size_t k = 0; k < dtheta.size(); ++k) {
                dtheta[k] += dtheta1[k];
            }
            for (std::size_t k = 0; k < dvm.size(); ++k) {
                dvm[k] += dvm1[k];
            }
            left_solve += solve1.left_ms;
            right_solve += solve1.right_ms;
        }

        const double elapsed_ms = 1000.0 *
            std::chrono::duration<double>(std::chrono::steady_clock::now() - repeat_start).count();
        j11_factor_ms.push_back(left_factor);
        j22_factor_ms.push_back(right_factor);
        j11_solve_ms.push_back(left_solve);
        j22_solve_ms.push_back(right_solve);
        j11_factor_solve_ms.push_back(left_factor + left_solve);
        j22_factor_solve_ms.push_back(right_factor + right_solve);
        wall_ms.push_back(elapsed_ms);
        serial_sum_ms.push_back(left_factor + right_factor + left_solve + right_solve);
    }

    left.destroy_stream();
    right.destroy_stream();

    PairCorrectionResult result;
    result.dtheta = dtheta;
    result.dvm = dvm;
    result.j11.x = dtheta;
    result.j22.x = dvm;
    result.j11.analyze_ms = summarize(std::vector<double>{j11_analyze});
    result.j22.analyze_ms = summarize(std::vector<double>{j22_analyze});
    result.j11.factor_ms = summarize(j11_factor_ms);
    result.j22.factor_ms = summarize(j22_factor_ms);
    result.j11.solve_ms = summarize(j11_solve_ms);
    result.j22.solve_ms = summarize(j22_solve_ms);
    result.j11.factor_solve_ms = summarize(j11_factor_solve_ms);
    result.j22.factor_solve_ms = summarize(j22_factor_solve_ms);
    result.j11.total_ms = summarize(j11_factor_solve_ms);
    result.j22.total_ms = summarize(j22_factor_solve_ms);
    result.wall_ms = summarize(wall_ms);
    result.serial_sum_ms = summarize(serial_sum_ms);
    result.j11.rel_res = safe_ratio(norm2(residual_vector(j11, r_p, dtheta)), norm2(r_p));
    result.j22.rel_res = safe_ratio(norm2(residual_vector(j22, r_q, dvm)), norm2(r_q));
    return result;
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
    split.total_rel = safe_ratio(norm2(residual), norm2(rhs));
    split.p_rel = safe_ratio(norm2(slice(residual, 0, metadata.n_pvpq)),
                             norm2(slice(rhs, 0, metadata.n_pvpq)));
    split.q_rel = safe_ratio(norm2(slice(residual, metadata.n_pvpq, metadata.n_pq)),
                             norm2(slice(rhs, metadata.n_pvpq, metadata.n_pq)));
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

Row make_row(const std::string& case_name,
             int32_t iteration,
             int32_t block_size,
             const std::string& variant,
             const CaseIterData& data,
             const std::vector<double>& dx0,
             const std::vector<double>& dx,
             const CudssSolveResult* j11,
             const CudssSolveResult* j22)
{
    Row row;
    row.case_name = case_name;
    row.iteration = iteration;
    row.block_size = block_size;
    row.variant = variant;
    row.full_n = data.matrix.rows;
    row.full_nnz = data.matrix.nnz();
    row.j11_n = data.j11.rows;
    row.j11_nnz = data.j11.nnz();
    row.j22_n = data.j22.rows;
    row.j22_nnz = data.j22.nnz();
    const std::vector<double>& dx_full = data.full.x;
    row.dx_error_ratio_before =
        safe_ratio(norm2(subtract_vectors(dx0, dx_full)), norm2(dx_full));
    row.dx_error_ratio_after =
        safe_ratio(norm2(subtract_vectors(dx, dx_full)), norm2(dx_full));
    row.theta_error_ratio_before = split_error_ratio(dx0, dx_full, 0, data.metadata.n_pvpq);
    row.theta_error_ratio_after = split_error_ratio(dx, dx_full, 0, data.metadata.n_pvpq);
    row.vmag_error_ratio_before =
        split_error_ratio(dx0, dx_full, data.metadata.n_pvpq, data.metadata.n_pq);
    row.vmag_error_ratio_after =
        split_error_ratio(dx, dx_full, data.metadata.n_pvpq, data.metadata.n_pq);
    row.cosine_before = cosine(dx0, dx_full);
    row.cosine_after = cosine(dx, dx_full);
    const ResidualSplit before = compute_residual_split(data.matrix, data.rhs, dx0, data.metadata);
    const ResidualSplit after = compute_residual_split(data.matrix, data.rhs, dx, data.metadata);
    row.full_residual_before = before.total_rel;
    row.full_residual_after = after.total_rel;
    row.p_residual_before = before.p_rel;
    row.p_residual_after = after.p_rel;
    row.q_residual_before = before.q_rel;
    row.q_residual_after = after.q_rel;
    row.full_j_factor_solve_ms = data.full.factor_solve_ms.mean;
    row.full_j_solve_ms = data.full.solve_ms.mean;
    if (j11 != nullptr) {
        row.j11_factor_solve_ms = j11->factor_solve_ms.mean;
        row.j11_solve_ms = j11->solve_ms.mean;
        row.j11_rel_res = j11->rel_res;
        row.dtheta_norm = norm2(j11->x);
    }
    if (j22 != nullptr) {
        row.j22_factor_solve_ms = j22->factor_solve_ms.mean;
        row.j22_solve_ms = j22->solve_ms.mean;
        row.j22_rel_res = j22->rel_res;
        row.dvm_norm = norm2(j22->x);
    }
    row.correction_over_full =
        safe_ratio(row.j11_factor_solve_ms + row.j22_factor_solve_ms,
                   row.full_j_factor_solve_ms);
    return row;
}

std::vector<Row> analyze_one_block_size(const CliOptions& options,
                                        const std::string& case_name,
                                        int32_t iteration,
                                        int32_t block_size,
                                        const CaseIterData& data)
{
    const LinearMetadata& md = data.metadata;
    const std::vector<double> dx0 =
        solve_block_jacobi_bicgstab(data.matrix, data.rhs, block_size, options);
    const std::vector<double> r0 = residual_vector(data.matrix, data.rhs, dx0);
    const std::vector<double> r_p = slice(r0, 0, md.n_pvpq);
    const std::vector<double> r_q = slice(r0, md.n_pvpq, md.n_pq);

    std::vector<Row> rows;

    PairCorrectionResult a0 =
        solve_pair_parallel(data.j11, data.j22, r_p, r_q, data.matrix, md, false, options.repeat);
    std::vector<double> dx_a0 = dx0;
    for (int32_t i = 0; i < md.n_pvpq; ++i) {
        dx_a0[static_cast<std::size_t>(i)] += a0.dtheta[static_cast<std::size_t>(i)];
    }
    for (int32_t i = 0; i < md.n_pq; ++i) {
        dx_a0[static_cast<std::size_t>(md.n_pvpq + i)] +=
            a0.dvm[static_cast<std::size_t>(i)];
    }
    Row a0_row = make_row(case_name, iteration, block_size, "A0", data, dx0, dx_a0,
                          &a0.j11, &a0.j22);
    a0_row.correction_wall_ms = a0.wall_ms.mean;
    a0_row.correction_serial_sum_ms = a0.serial_sum_ms.mean;
    rows.push_back(a0_row);

    PairCorrectionResult a1 =
        solve_pair_parallel(data.j11, data.j22, r_p, r_q, data.matrix, md, true, options.repeat);
    std::vector<double> dx_a1 = dx0;
    for (int32_t i = 0; i < md.n_pvpq; ++i) {
        dx_a1[static_cast<std::size_t>(i)] += a1.dtheta[static_cast<std::size_t>(i)];
    }
    for (int32_t i = 0; i < md.n_pq; ++i) {
        dx_a1[static_cast<std::size_t>(md.n_pvpq + i)] += a1.dvm[static_cast<std::size_t>(i)];
    }

    Row a1_row = make_row(case_name, iteration, block_size, "A1", data, dx0, dx_a1,
                          &a1.j11, &a1.j22);
    a1_row.correction_wall_ms = a1.wall_ms.mean;
    a1_row.correction_serial_sum_ms = a1.serial_sum_ms.mean;
    rows.push_back(a1_row);

    return rows;
}

CaseIterData load_case_iter(const CliOptions& options,
                            const std::string& case_name,
                            int32_t iteration)
{
    CaseIterData data;
    const cupf_minimal::DumpCaseData case_data =
        cupf_minimal::load_dump_case(options.case_root / case_name);
    data.metadata = make_linear_metadata(case_data);
    data.matrix = load_cupf_csr_dump(jacobian_path(options.jf_root, case_name, iteration));
    data.rhs = load_cupf_vector_dump(rhs_path(options.jf_root, case_name, iteration));
    if (data.matrix.rows != static_cast<int32_t>(data.rhs.size()) ||
        data.matrix.rows != data.metadata.n_pvpq + data.metadata.n_pq) {
        throw std::runtime_error("matrix/RHS/metadata dimension mismatch");
    }
    data.j11 = extract_submatrix(data.matrix, 0, data.metadata.n_pvpq, 0, data.metadata.n_pvpq);
    data.j22 = extract_submatrix(data.matrix,
                                 data.metadata.n_pvpq,
                                 data.metadata.n_pq,
                                 data.metadata.n_pvpq,
                                 data.metadata.n_pq);
    if (data.j11.rows != data.j11.cols || data.j22.rows != data.j22.cols) {
        throw std::runtime_error("J11/J22 extraction produced non-square matrix");
    }
    data.full = solve_cudss_timed(data.matrix, data.rhs, options.repeat);
    return data;
}

void write_csv(const std::filesystem::path& path, const std::vector<Row>& rows)
{
    std::ofstream out(path);
    out << "case,iter,block_size,variant,full_n,full_nnz,j11_n,j11_nnz,j22_n,j22_nnz,"
           "dx_error_ratio_before,dx_error_ratio_after,theta_error_before,theta_error_after,"
           "vmag_error_before,vmag_error_after,cosine_before,cosine_after,"
           "full_residual_before,full_residual_after,P_residual_before,P_residual_after,"
           "Q_residual_before,Q_residual_after,J11_factor_solve_ms,J11_solve_ms,"
           "J22_factor_solve_ms,J22_solve_ms,full_J_factor_solve_ms,full_J_solve_ms,"
           "correction_wall_ms,correction_serial_sum_ms,correction_over_full,"
           "J11_rel_res,J22_rel_res,dtheta_norm,dV_norm,status,"
           "error_message\n";
    for (const Row& row : rows) {
        std::string message = row.error_message;
        std::replace(message.begin(), message.end(), ',', ';');
        out << row.case_name << ",J" << row.iteration << ',' << row.block_size << ','
            << row.variant << ',' << row.full_n << ',' << row.full_nnz << ','
            << row.j11_n << ',' << row.j11_nnz << ',' << row.j22_n << ',' << row.j22_nnz
            << ',' << format_double(row.dx_error_ratio_before)
            << ',' << format_double(row.dx_error_ratio_after)
            << ',' << format_double(row.theta_error_ratio_before)
            << ',' << format_double(row.theta_error_ratio_after)
            << ',' << format_double(row.vmag_error_ratio_before)
            << ',' << format_double(row.vmag_error_ratio_after)
            << ',' << format_double(row.cosine_before)
            << ',' << format_double(row.cosine_after)
            << ',' << format_double(row.full_residual_before)
            << ',' << format_double(row.full_residual_after)
            << ',' << format_double(row.p_residual_before)
            << ',' << format_double(row.p_residual_after)
            << ',' << format_double(row.q_residual_before)
            << ',' << format_double(row.q_residual_after)
            << ',' << format_double(row.j11_factor_solve_ms)
            << ',' << format_double(row.j11_solve_ms)
            << ',' << format_double(row.j22_factor_solve_ms)
            << ',' << format_double(row.j22_solve_ms)
            << ',' << format_double(row.full_j_factor_solve_ms)
            << ',' << format_double(row.full_j_solve_ms)
            << ',' << format_double(row.correction_wall_ms)
            << ',' << format_double(row.correction_serial_sum_ms)
            << ',' << format_double(row.correction_over_full)
            << ',' << format_double(row.j11_rel_res)
            << ',' << format_double(row.j22_rel_res)
            << ',' << format_double(row.dtheta_norm)
            << ',' << format_double(row.dvm_norm)
            << ',' << row.status << ',' << message << '\n';
    }
}

double mean_metric(const std::vector<Row>& rows,
                   const std::string& variant,
                   int32_t iteration,
                   double Row::*member)
{
    double sum = 0.0;
    int32_t count = 0;
    for (const Row& row : rows) {
        if (row.status == "ok" && row.variant == variant &&
            (iteration < 0 || row.iteration == iteration) && std::isfinite(row.*member)) {
            sum += row.*member;
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : kNan;
}

int32_t count_better_than_a0(const std::vector<Row>& rows,
                             const std::string& variant,
                             double Row::*member)
{
    int32_t count = 0;
    for (const Row& row : rows) {
        if (row.status != "ok" || row.variant != variant) {
            continue;
        }
        const auto it = std::find_if(rows.begin(), rows.end(), [&](const Row& base) {
            return base.status == "ok" && base.variant == "A0" &&
                   base.case_name == row.case_name && base.iteration == row.iteration &&
                   base.block_size == row.block_size;
        });
        if (it != rows.end() && row.*member < (*it).*member) {
            ++count;
        }
    }
    return count;
}

void write_report(const std::filesystem::path& path, const std::vector<Row>& rows)
{
    std::ofstream out(path);
    out << "# A0/A1 J11-J22 Simultaneous Correction Quality\n\n";
    out << "## J1 Means\n\n";
    out << "| variant | dx err after | theta err after | |V| err after | P residual after | "
           "Q residual after | correction/full | correction wall ms |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const std::string& variant : {"A0", "A1"}) {
        out << '|' << variant << '|'
            << format_double(mean_metric(rows, variant, 1, &Row::dx_error_ratio_after)) << '|'
            << format_double(mean_metric(rows, variant, 1, &Row::theta_error_ratio_after)) << '|'
            << format_double(mean_metric(rows, variant, 1, &Row::vmag_error_ratio_after)) << '|'
            << format_double(mean_metric(rows, variant, 1, &Row::p_residual_after)) << '|'
            << format_double(mean_metric(rows, variant, 1, &Row::q_residual_after)) << '|'
            << format_double(mean_metric(rows, variant, 1, &Row::correction_over_full)) << '|'
            << format_double(mean_metric(rows, variant, 1, &Row::correction_wall_ms)) << "|\n";
    }
    out << "\n## Stage 1 Judgment\n\n";
    const int32_t total_pairs = static_cast<int32_t>(
        std::count_if(rows.begin(), rows.end(), [](const Row& row) {
            return row.status == "ok" && row.variant == "A0";
        }));
    const int32_t a1_dx = count_better_than_a0(rows, "A1", &Row::dx_error_ratio_after);
    const int32_t a1_theta = count_better_than_a0(rows, "A1", &Row::theta_error_ratio_after);
    const int32_t a1_v = count_better_than_a0(rows, "A1", &Row::vmag_error_ratio_after);
    out << "- `A0` solves J11 and J22 from the same residual, then applies both corrections "
           "simultaneously.\n";
    out << "- `A1` reuses the same J11/J22 factors and solves one extra cross-residual round.\n";
    out << "- `A1` improves dx error over `A0` on `" << a1_dx << "/" << total_pairs
        << "` rows, theta error on `" << a1_theta << "/" << total_pairs
        << "` rows, and |V| error on `" << a1_v << "/" << total_pairs << "` rows.\n";
    out << "- Cost gate: J1 mean correction/full ratios are listed above. Near-1 means the "
           "correction is close to full cuDSS factor+solve cost before adding BiCGSTAB(2).\n";
    const double a0_cost = mean_metric(rows, "A0", 1, &Row::correction_over_full);
    const double a1_cost = mean_metric(rows, "A1", 1, &Row::correction_over_full);
    if (a0_cost >= 1.0 && a1_cost >= 1.0) {
        out << "- Stage 2 hybrid run should be skipped by the cost gate unless nonlinear NR "
               "behavior is expected to change substantially: both variants cost at least one "
               "full-J factor+solve before adding the BiCGSTAB(2) dx0 cost.\n\n";
    } else {
        out << "- Hybrid Stage 2 may proceed for the cheaper variant if it improves dx quality "
               "without reintroducing large P/Q residuals.\n\n";
    }

    out << "## J1 Snapshot\n\n";
    out << "| case | bs | variant | dx before | dx after | |V| after | P after | Q after | corr/full | wall ms |\n";
    out << "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const Row& row : rows) {
        if (row.status == "ok" && row.iteration == 1) {
            out << '|' << row.case_name << '|' << row.block_size << '|' << row.variant << '|'
                << format_double(row.dx_error_ratio_before) << '|'
                << format_double(row.dx_error_ratio_after) << '|'
                << format_double(row.vmag_error_ratio_after) << '|'
                << format_double(row.p_residual_after) << '|'
                << format_double(row.q_residual_after) << '|'
                << format_double(row.correction_over_full) << '|'
                << format_double(row.correction_wall_ms) << "|\n";
        }
    }
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
                        continue;
                    }
                    throw std::runtime_error("missing J/F for " + case_name);
                }
                std::cout << "[case] " << case_name << " J" << iteration << '\n';
                CaseIterData data = load_case_iter(options, case_name, iteration);
                for (int32_t block_size : options.block_sizes) {
                    std::cout << "  [bs=" << block_size << "] variants\n";
                    std::vector<Row> block_rows =
                        analyze_one_block_size(options, case_name, iteration, block_size, data);
                    rows.insert(rows.end(), block_rows.begin(), block_rows.end());
                }
            }
        }
        write_csv(options.output_dir / "a0_a1_quality_summary.csv", rows);
        write_report(options.output_dir / "a0_a1_quality_report.md", rows);
        write_csv(options.output_dir / "j11_j22_correction_quality_summary.csv", rows);
        write_report(options.output_dir / "j11_j22_correction_quality_report.md", rows);
        std::cout << "[done] wrote A0/A1 correction quality results to "
                  << options.output_dir << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
