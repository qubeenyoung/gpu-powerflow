#include "cuiter/common/cuda_utils.hpp"

#include <cudss.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CsrMatrix {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<double> values;

    int32_t nnz() const
    {
        return static_cast<int32_t>(values.size());
    }
};

struct CliOptions {
    std::filesystem::path matrix_path;
    std::filesystem::path rhs_path;
    std::string case_name;
    std::string precision = "fp64";
    int32_t repeats = 1;
    bool csv = false;
    bool enable_mt = false;
    std::string threading_lib;
};

struct TimingStats {
    double analyze_seconds = 0.0;
    double factorize_seconds = 0.0;
    double solve_seconds = 0.0;
    double relative_residual = 0.0;
    double residual_norm = 0.0;
};

const char* cudss_status_name(cudssStatus_t status)
{
    switch (status) {
    case CUDSS_STATUS_SUCCESS:
        return "CUDSS_STATUS_SUCCESS";
    case CUDSS_STATUS_NOT_INITIALIZED:
        return "CUDSS_STATUS_NOT_INITIALIZED";
    case CUDSS_STATUS_ALLOC_FAILED:
        return "CUDSS_STATUS_ALLOC_FAILED";
    case CUDSS_STATUS_INVALID_VALUE:
        return "CUDSS_STATUS_INVALID_VALUE";
    case CUDSS_STATUS_NOT_SUPPORTED:
        return "CUDSS_STATUS_NOT_SUPPORTED";
    case CUDSS_STATUS_EXECUTION_FAILED:
        return "CUDSS_STATUS_EXECUTION_FAILED";
    case CUDSS_STATUS_INTERNAL_ERROR:
        return "CUDSS_STATUS_INTERNAL_ERROR";
    }
    return "CUDSS_STATUS_UNKNOWN";
}

void check_cudss(cudssStatus_t status, const char* call, const char* file, int line)
{
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuDSS error at ") + file + ":" +
                                 std::to_string(line) + " in " + call + " - " +
                                 cudss_status_name(status));
    }
}

#define CUITER_CUDSS_CHECK(call) check_cudss((call), #call, __FILE__, __LINE__)

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " --matrix J.txt --rhs F.txt [options]\n\n"
        << "Options:\n"
        << "  --case NAME\n"
        << "  --precision fp64|fp32\n"
        << "  --repeats INT\n"
        << "  --csv\n"
        << "  --enable-mt\n"
        << "  --threading-lib PATH\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--matrix" && i + 1 < argc) {
            options.matrix_path = argv[++i];
        } else if (arg == "--rhs" && i + 1 < argc) {
            options.rhs_path = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            options.case_name = argv[++i];
        } else if (arg == "--precision" && i + 1 < argc) {
            options.precision = argv[++i];
        } else if (arg == "--repeats" && i + 1 < argc) {
            options.repeats = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--csv") {
            options.csv = true;
        } else if (arg == "--enable-mt") {
            options.enable_mt = true;
        } else if (arg == "--threading-lib" && i + 1 < argc) {
            options.threading_lib = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.matrix_path.empty() || options.rhs_path.empty()) {
        throw std::runtime_error("--matrix and --rhs are required");
    }
    if (options.precision != "fp64" && options.precision != "fp32") {
        throw std::runtime_error("--precision must be fp64 or fp32");
    }
    if (options.repeats <= 0) {
        throw std::runtime_error("--repeats must be positive");
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

CsrMatrix load_cupf_csr_dump(const std::filesystem::path& path)
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

    CsrMatrix matrix;
    int32_t nnz = 0;
    expect_token(in, "rows", path);
    in >> matrix.rows;
    expect_token(in, "cols", path);
    in >> matrix.cols;
    expect_token(in, "nnz", path);
    in >> nnz;
    if (matrix.rows <= 0 || matrix.cols <= 0 || matrix.rows != matrix.cols || nnz <= 0) {
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
        throw std::runtime_error("failed to open RHS file: " + path.string());
    }

    expect_token(in, "type", path);
    std::string type;
    in >> type;
    if (type != "vector") {
        throw std::runtime_error("RHS file is not a cuPF vector dump: " + path.string());
    }
    expect_token(in, "size", path);
    int32_t n = 0;
    in >> n;
    if (n <= 0) {
        throw std::runtime_error("invalid vector length in " + path.string());
    }
    expect_token(in, "values", path);

    std::vector<double> values(static_cast<std::size_t>(n), 0.0);
    for (int32_t k = 0; k < n; ++k) {
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

template <typename T>
cudaDataType_t cuda_value_type();

template <>
cudaDataType_t cuda_value_type<double>()
{
    return CUDA_R_64F;
}

template <>
cudaDataType_t cuda_value_type<float>()
{
    return CUDA_R_32F;
}

template <typename T>
std::vector<T> convert_values(const std::vector<double>& values)
{
    std::vector<T> converted(values.size());
    std::transform(values.begin(), values.end(), converted.begin(), [](double value) {
        return static_cast<T>(value);
    });
    return converted;
}

template <typename Func>
double time_phase(Func&& func)
{
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    func();
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

double norm2(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values) {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
}

double residual_norm2(const CsrMatrix& matrix,
                      const std::vector<double>& rhs,
                      const std::vector<double>& x)
{
    long double sum = 0.0;
    for (int32_t row = 0; row < matrix.rows; ++row) {
        double ax = 0.0;
        for (int32_t k = matrix.row_ptr[static_cast<std::size_t>(row)];
             k < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++k) {
            ax += matrix.values[static_cast<std::size_t>(k)] *
                  x[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(k)])];
        }
        const double r = rhs[static_cast<std::size_t>(row)] - ax;
        sum += static_cast<long double>(r) * static_cast<long double>(r);
    }
    return std::sqrt(static_cast<double>(sum));
}

struct CudssObjects {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matrix = nullptr;
    cudssMatrix_t rhs = nullptr;
    cudssMatrix_t solution = nullptr;

    ~CudssObjects()
    {
        if (matrix != nullptr) {
            cudssMatrixDestroy(matrix);
        }
        if (rhs != nullptr) {
            cudssMatrixDestroy(rhs);
        }
        if (solution != nullptr) {
            cudssMatrixDestroy(solution);
        }
        if (data != nullptr) {
            cudssDataDestroy(handle, data);
        }
        if (config != nullptr) {
            cudssConfigDestroy(config);
        }
        if (handle != nullptr) {
            cudssDestroy(handle);
        }
    }
};

template <typename T>
TimingStats run_once(const CsrMatrix& matrix,
                     const std::vector<double>& rhs,
                     const CliOptions& options)
{
    const std::vector<T> values_t = convert_values<T>(matrix.values);
    const std::vector<T> rhs_t = convert_values<T>(rhs);

    cuiter::DeviceBuffer<int32_t> d_row_ptr;
    cuiter::DeviceBuffer<int32_t> d_col_idx;
    cuiter::DeviceBuffer<T> d_values;
    cuiter::DeviceBuffer<T> d_rhs;
    cuiter::DeviceBuffer<T> d_x;
    d_row_ptr.assign(matrix.row_ptr.data(), matrix.row_ptr.size());
    d_col_idx.assign(matrix.col_idx.data(), matrix.col_idx.size());
    d_values.assign(values_t.data(), values_t.size());
    d_rhs.assign(rhs_t.data(), rhs_t.size());
    d_x.resize(rhs_t.size());
    d_x.memset_zero();
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());

    CudssObjects cudss;
    CUITER_CUDSS_CHECK(cudssCreate(&cudss.handle));
    if (options.enable_mt) {
        const char* threading_lib =
            options.threading_lib.empty() ? nullptr : options.threading_lib.c_str();
        CUITER_CUDSS_CHECK(cudssSetThreadingLayer(cudss.handle, threading_lib));
    }
    CUITER_CUDSS_CHECK(cudssConfigCreate(&cudss.config));
    CUITER_CUDSS_CHECK(cudssDataCreate(cudss.handle, &cudss.data));

    CUITER_CUDSS_CHECK(cudssMatrixCreateCsr(
        &cudss.matrix,
        matrix.rows,
        matrix.cols,
        static_cast<int64_t>(matrix.nnz()),
        d_row_ptr.data(),
        nullptr,
        d_col_idx.data(),
        d_values.data(),
        CUDA_R_32I,
        cuda_value_type<T>(),
        CUDSS_MTYPE_GENERAL,
        CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO));
    CUITER_CUDSS_CHECK(cudssMatrixCreateDn(
        &cudss.rhs,
        matrix.rows,
        1,
        matrix.rows,
        d_rhs.data(),
        cuda_value_type<T>(),
        CUDSS_LAYOUT_COL_MAJOR));
    CUITER_CUDSS_CHECK(cudssMatrixCreateDn(
        &cudss.solution,
        matrix.rows,
        1,
        matrix.rows,
        d_x.data(),
        cuda_value_type<T>(),
        CUDSS_LAYOUT_COL_MAJOR));

    TimingStats stats;
    stats.analyze_seconds = time_phase([&]() {
        CUITER_CUDSS_CHECK(cudssExecute(cudss.handle,
                                        CUDSS_PHASE_ANALYSIS,
                                        cudss.config,
                                        cudss.data,
                                        cudss.matrix,
                                        cudss.solution,
                                        cudss.rhs));
    });
    stats.factorize_seconds = time_phase([&]() {
        CUITER_CUDSS_CHECK(cudssExecute(cudss.handle,
                                        CUDSS_PHASE_FACTORIZATION,
                                        cudss.config,
                                        cudss.data,
                                        cudss.matrix,
                                        cudss.solution,
                                        cudss.rhs));
    });
    stats.solve_seconds = time_phase([&]() {
        CUITER_CUDSS_CHECK(cudssExecute(cudss.handle,
                                        CUDSS_PHASE_SOLVE,
                                        cudss.config,
                                        cudss.data,
                                        cudss.matrix,
                                        cudss.solution,
                                        cudss.rhs));
    });

    std::vector<T> x_t(rhs_t.size());
    d_x.copy_to(x_t.data(), x_t.size());
    std::vector<double> x(x_t.size());
    std::transform(x_t.begin(), x_t.end(), x.begin(), [](T value) {
        return static_cast<double>(value);
    });
    stats.residual_norm = residual_norm2(matrix, rhs, x);
    const double rhs_norm = norm2(rhs);
    stats.relative_residual =
        rhs_norm > 0.0 ? stats.residual_norm / rhs_norm : stats.residual_norm;
    return stats;
}

template <typename T>
TimingStats run_repeated(const CsrMatrix& matrix,
                         const std::vector<double>& rhs,
                         const CliOptions& options)
{
    TimingStats total;
    TimingStats last;
    for (int32_t repeat = 0; repeat < options.repeats; ++repeat) {
        last = run_once<T>(matrix, rhs, options);
        total.analyze_seconds += last.analyze_seconds;
        total.factorize_seconds += last.factorize_seconds;
        total.solve_seconds += last.solve_seconds;
    }
    const double scale = 1.0 / static_cast<double>(options.repeats);
    total.analyze_seconds *= scale;
    total.factorize_seconds *= scale;
    total.solve_seconds *= scale;
    total.residual_norm = last.residual_norm;
    total.relative_residual = last.relative_residual;
    return total;
}

void print_csv(const CliOptions& options, const CsrMatrix& matrix, const TimingStats& stats)
{
    std::cout << "case_name,precision,n,nnz,repeats,analyze_sec,factorize_sec,solve_sec,"
                 "total_sec,residual_norm,relative_residual\n";
    std::cout << options.case_name << ","
              << options.precision << ","
              << matrix.rows << ","
              << matrix.nnz() << ","
              << options.repeats << ","
              << std::setprecision(12)
              << stats.analyze_seconds << ","
              << stats.factorize_seconds << ","
              << stats.solve_seconds << ","
              << (stats.analyze_seconds + stats.factorize_seconds + stats.solve_seconds) << ","
              << stats.residual_norm << ","
              << stats.relative_residual << "\n";
}

void print_markdown(const CliOptions& options, const CsrMatrix& matrix, const TimingStats& stats)
{
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "# cuDSS J/F Benchmark\n\n";
    std::cout << "- case: " << options.case_name << "\n";
    std::cout << "- precision: " << options.precision << "\n";
    std::cout << "- n: " << matrix.rows << "\n";
    std::cout << "- nnz: " << matrix.nnz() << "\n";
    std::cout << "- repeats: " << options.repeats << "\n\n";
    std::cout << "| phase | seconds |\n|---|---:|\n";
    std::cout << "| analyze | " << stats.analyze_seconds << " |\n";
    std::cout << "| factorize | " << stats.factorize_seconds << " |\n";
    std::cout << "| solve | " << stats.solve_seconds << " |\n";
    std::cout << "| total | "
              << (stats.analyze_seconds + stats.factorize_seconds + stats.solve_seconds)
              << " |\n\n";
    std::cout << "| residual | value |\n|---|---:|\n";
    std::cout << "| norm2 | " << stats.residual_norm << " |\n";
    std::cout << "| relative_norm2 | " << stats.relative_residual << " |\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        const CsrMatrix matrix = load_cupf_csr_dump(options.matrix_path);
        const std::vector<double> rhs = load_cupf_vector_dump(options.rhs_path);
        if (static_cast<int32_t>(rhs.size()) != matrix.rows) {
            throw std::runtime_error("RHS length does not match matrix rows");
        }

        TimingStats stats;
        if (options.precision == "fp64") {
            stats = run_repeated<double>(matrix, rhs, options);
        } else {
            stats = run_repeated<float>(matrix, rhs, options);
        }

        if (options.csv) {
            print_csv(options, matrix, stats);
        } else {
            print_markdown(options, matrix, stats);
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
