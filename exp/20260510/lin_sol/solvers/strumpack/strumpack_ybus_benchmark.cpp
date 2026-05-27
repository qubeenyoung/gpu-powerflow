#include <StrumpackConfig.h>
#include <StrumpackSparseSolver.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
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

using Clock = std::chrono::steady_clock;

bool strumpack_has_cuda()
{
#if defined(STRUMPACK_USE_CUDA)
    return true;
#else
    return false;
#endif
}

struct Options {
    std::filesystem::path matrix;
    std::filesystem::path out;
    std::string dtype = "fp64";
    int repeats = 5;
    int warmup = 1;
    double diag_shift = 0.0;
    bool use_gpu = true;
    bool use_metis_node_ndp = true;
};

struct Stats {
    double mean = std::numeric_limits<double>::quiet_NaN();
    double median = std::numeric_limits<double>::quiet_NaN();
    double min = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
    double stddev = std::numeric_limits<double>::quiet_NaN();
};

struct CsrComplex {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<std::complex<double>> values;
};

struct RunSample {
    double analysis_ms = 0.0;
    double factor_ms = 0.0;
    double solve_ms = 0.0;
    int code = 0;
};

double elapsed_ms(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

Options parse_args(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--matrix") {
            opt.matrix = value("--matrix");
        } else if (arg == "--out") {
            opt.out = value("--out");
        } else if (arg == "--dtype") {
            opt.dtype = value("--dtype");
        } else if (arg == "--repeats") {
            opt.repeats = std::stoi(value("--repeats"));
        } else if (arg == "--warmup") {
            opt.warmup = std::stoi(value("--warmup"));
        } else if (arg == "--diag-shift") {
            opt.diag_shift = std::stod(value("--diag-shift"));
        } else if (arg == "--cpu") {
            opt.use_gpu = false;
        } else if (arg == "--gpu") {
            opt.use_gpu = true;
        } else if (arg == "--metis-node-nd") {
            opt.use_metis_node_ndp = false;
        } else if (arg == "--metis-node-ndp") {
            opt.use_metis_node_ndp = true;
        } else if (arg == "--help" || arg == "-h") {
            throw std::runtime_error(
                "usage: strumpack_ybus_benchmark --matrix dump_Ybus.mtx --out result.json "
                "[--dtype fp64|fp32] [--repeats N] [--warmup N] [--diag-shift value] "
                "[--cpu|--gpu] [--metis-node-nd|--metis-node-ndp]");
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opt.matrix.empty() || opt.out.empty()) {
        throw std::runtime_error("--matrix and --out are required");
    }
    if (opt.dtype != "fp64" && opt.dtype != "fp32") {
        throw std::runtime_error("--dtype must be fp64 or fp32");
    }
    if (opt.repeats < 1 || opt.warmup < 0) {
        throw std::runtime_error("--repeats must be positive and --warmup must be nonnegative");
    }
    return opt;
}

Stats make_stats(std::vector<double> values)
{
    Stats s;
    if (values.empty()) {
        return s;
    }
    std::sort(values.begin(), values.end());
    s.min = values.front();
    s.max = values.back();
    s.median = values.size() % 2 == 0
        ? 0.5 * (values[values.size() / 2 - 1] + values[values.size() / 2])
        : values[values.size() / 2];
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double sq = 0.0;
    for (double v : values) {
        const double d = v - s.mean;
        sq += d * d;
    }
    s.stddev = std::sqrt(sq / static_cast<double>(values.size()));
    return s;
}

std::string json_number(double value)
{
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

std::string escape_json(const std::string& s)
{
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

void write_stats(std::ostream& out, const Stats& s)
{
    out << "{\"mean\":" << json_number(s.mean)
        << ",\"median\":" << json_number(s.median)
        << ",\"min\":" << json_number(s.min)
        << ",\"max\":" << json_number(s.max)
        << ",\"stddev\":" << json_number(s.stddev)
        << "}";
}

template <typename scalar_t>
std::vector<scalar_t> cast_values(const std::vector<std::complex<double>>& in)
{
    std::vector<scalar_t> out;
    out.reserve(in.size());
    for (auto v : in) {
        out.push_back(scalar_t(v));
    }
    return out;
}

template <typename real_t>
std::complex<real_t> make_x_ref_value(int i)
{
    return {
        real_t(1.0) + real_t(0.001) * real_t((i % 17) - 8),
        real_t(-0.25) + real_t(0.0015) * real_t((i % 13) - 6)
    };
}

CsrComplex load_matrix_market_complex(const std::filesystem::path& path, double diag_shift)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }

    std::string banner;
    std::getline(in, banner);
    const bool is_coordinate = banner.find("coordinate") != std::string::npos;
    const bool is_complex = banner.find("complex") != std::string::npos;
    const bool is_symmetric = banner.find("symmetric") != std::string::npos;
    const bool is_hermitian = banner.find("hermitian") != std::string::npos;
    if (banner.find("MatrixMarket") == std::string::npos || !is_coordinate) {
        throw std::runtime_error("expected coordinate MatrixMarket file");
    }

    std::string line;
    do {
        if (!std::getline(in, line)) {
            throw std::runtime_error("missing MatrixMarket size line");
        }
    } while (!line.empty() && line[0] == '%');

    int rows = 0;
    int cols = 0;
    int entries = 0;
    {
        std::istringstream dims(line);
        dims >> rows >> cols >> entries;
    }
    if (rows <= 0 || cols <= 0 || entries < 0) {
        throw std::runtime_error("invalid MatrixMarket dimensions");
    }

    struct Entry {
        int row;
        int col;
        std::complex<double> value;
    };
    std::vector<Entry> coo;
    coo.reserve(static_cast<std::size_t>(entries) * (is_symmetric || is_hermitian ? 2 : 1) +
                (diag_shift != 0.0 ? static_cast<std::size_t>(std::min(rows, cols)) : 0));

    for (int k = 0; k < entries; ++k) {
        int i = 0;
        int j = 0;
        double re = 0.0;
        double im = 0.0;
        in >> i >> j >> re;
        if (is_complex) {
            in >> im;
        }
        --i;
        --j;
        const std::complex<double> value(re, im);
        coo.push_back({i, j, value});
        if ((is_symmetric || is_hermitian) && i != j) {
            coo.push_back({j, i, is_hermitian ? std::conj(value) : value});
        }
    }
    if (diag_shift != 0.0) {
        for (int i = 0; i < std::min(rows, cols); ++i) {
            coo.push_back({i, i, std::complex<double>(diag_shift, 0.0)});
        }
    }

    std::sort(coo.begin(), coo.end(), [](const Entry& a, const Entry& b) {
        return a.row == b.row ? a.col < b.col : a.row < b.row;
    });
    std::vector<Entry> summed;
    summed.reserve(coo.size());
    for (const auto& e : coo) {
        if (e.row < 0 || e.row >= rows || e.col < 0 || e.col >= cols) {
            throw std::runtime_error("MatrixMarket entry out of bounds");
        }
        if (!summed.empty() && summed.back().row == e.row && summed.back().col == e.col) {
            summed.back().value += e.value;
        } else {
            summed.push_back(e);
        }
    }

    CsrComplex csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.nnz = static_cast<int>(summed.size());
    csr.row_ptr.assign(static_cast<std::size_t>(rows + 1), 0);
    csr.col_idx.resize(summed.size());
    csr.values.resize(summed.size());
    for (const auto& e : summed) {
        ++csr.row_ptr[static_cast<std::size_t>(e.row + 1)];
    }
    for (int row = 0; row < rows; ++row) {
        csr.row_ptr[static_cast<std::size_t>(row + 1)] += csr.row_ptr[static_cast<std::size_t>(row)];
    }
    std::vector<int> cursor = csr.row_ptr;
    for (const auto& e : summed) {
        const int dst = cursor[static_cast<std::size_t>(e.row)]++;
        csr.col_idx[static_cast<std::size_t>(dst)] = e.col;
        csr.values[static_cast<std::size_t>(dst)] = e.value;
    }
    return csr;
}

template <typename scalar_t>
std::vector<scalar_t> make_x_ref(int n)
{
    using real_t = typename scalar_t::value_type;
    std::vector<scalar_t> x(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] = make_x_ref_value<real_t>(i);
    }
    return x;
}

template <typename scalar_t>
std::vector<scalar_t> spmv(const CsrComplex& A, const std::vector<scalar_t>& x)
{
    std::vector<scalar_t> y(static_cast<std::size_t>(A.rows), scalar_t{});
    for (int row = 0; row < A.rows; ++row) {
        scalar_t sum{};
        for (int p = A.row_ptr[static_cast<std::size_t>(row)];
             p < A.row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
            sum += scalar_t(A.values[static_cast<std::size_t>(p)]) *
                   x[static_cast<std::size_t>(A.col_idx[static_cast<std::size_t>(p)])];
        }
        y[static_cast<std::size_t>(row)] = sum;
    }
    return y;
}

template <typename scalar_t>
double norm2(const std::vector<scalar_t>& x)
{
    long double acc = 0.0;
    for (auto v : x) {
        const auto a = std::abs(v);
        acc += static_cast<long double>(a) * static_cast<long double>(a);
    }
    return static_cast<double>(std::sqrt(acc));
}

template <typename scalar_t>
double relative_error(const std::vector<scalar_t>& x, const std::vector<scalar_t>& ref)
{
    std::vector<scalar_t> diff(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        diff[i] = x[i] - ref[i];
    }
    return norm2(diff) / std::max(norm2(ref), std::numeric_limits<double>::min());
}

void cuda_sync_if_needed(bool use_gpu)
{
    if (!use_gpu) {
        return;
    }
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(err));
    }
}

template <typename scalar_t>
RunSample run_once(const CsrComplex& A,
                   const std::vector<scalar_t>& values,
                   const std::vector<scalar_t>& rhs,
                   std::vector<scalar_t>& x,
                   bool use_gpu,
                   bool use_metis_node_ndp)
{
    strumpack::StrumpackSparseSolver<scalar_t, int> solver(false, true);
    if (use_gpu) {
        solver.options().enable_gpu();
    } else {
        solver.options().disable_gpu();
    }
    if (use_metis_node_ndp) {
        solver.options().enable_METIS_NodeNDP();
    } else {
        solver.options().disable_METIS_NodeNDP();
    }

    RunSample sample;
    x.assign(rhs.size(), scalar_t{});

    auto analysis_start = Clock::now();
    solver.set_csr_matrix(A.rows, A.row_ptr.data(), A.col_idx.data(), values.data(), false);
    const auto reorder_code = solver.reorder();
    cuda_sync_if_needed(use_gpu);
    auto analysis_stop = Clock::now();

    auto factor_start = Clock::now();
    const auto factor_code = solver.factor();
    cuda_sync_if_needed(use_gpu);
    auto factor_stop = Clock::now();

    auto solve_start = Clock::now();
    const auto solve_code = solver.solve(rhs.data(), x.data());
    cuda_sync_if_needed(use_gpu);
    auto solve_stop = Clock::now();

    sample.analysis_ms = elapsed_ms(analysis_start, analysis_stop);
    sample.factor_ms = elapsed_ms(factor_start, factor_stop);
    sample.solve_ms = elapsed_ms(solve_start, solve_stop);
    sample.code = std::max({
        reorder_code == strumpack::ReturnCode::SUCCESS ? 0 : 1,
        factor_code == strumpack::ReturnCode::SUCCESS ? 0 : 1,
        solve_code == strumpack::ReturnCode::SUCCESS ? 0 : 1,
    });
    return sample;
}

template <typename scalar_t>
void run_benchmark(const Options& opt)
{
    const auto total_start = Clock::now();
    const auto load_start = Clock::now();
    CsrComplex A = load_matrix_market_complex(opt.matrix, opt.diag_shift);
    const auto load_stop = Clock::now();
    if (A.rows != A.cols) {
        throw std::runtime_error("STRUMPACK sparse direct benchmark expects a square matrix");
    }

    auto values = cast_values<scalar_t>(A.values);
    auto x_ref = make_x_ref<scalar_t>(A.cols);
    auto rhs = spmv(A, x_ref);

    std::size_t free_before = 0;
    std::size_t total_mem = 0;
    if (opt.use_gpu) {
        cudaFree(nullptr);
        cudaMemGetInfo(&free_before, &total_mem);
    }

    std::vector<scalar_t> x;
    for (int i = 0; i < opt.warmup; ++i) {
        auto warm = run_once(A, values, rhs, x, opt.use_gpu, opt.use_metis_node_ndp);
        if (warm.code != 0) {
            throw std::runtime_error("STRUMPACK warmup failed");
        }
    }

    std::vector<double> analysis_ms;
    std::vector<double> factor_ms;
    std::vector<double> solve_ms;
    analysis_ms.reserve(static_cast<std::size_t>(opt.repeats));
    factor_ms.reserve(static_cast<std::size_t>(opt.repeats));
    solve_ms.reserve(static_cast<std::size_t>(opt.repeats));
    int last_code = 0;
    for (int i = 0; i < opt.repeats; ++i) {
        auto sample = run_once(A, values, rhs, x, opt.use_gpu, opt.use_metis_node_ndp);
        analysis_ms.push_back(sample.analysis_ms);
        factor_ms.push_back(sample.factor_ms);
        solve_ms.push_back(sample.solve_ms);
        last_code = sample.code;
        if (last_code != 0) {
            break;
        }
    }

    std::size_t free_after = 0;
    if (opt.use_gpu) {
        cudaMemGetInfo(&free_after, &total_mem);
    }

    auto Ax = spmv(A, x);
    for (std::size_t i = 0; i < Ax.size(); ++i) {
        Ax[i] -= rhs[i];
    }
    const double rel_res = norm2(Ax) / std::max(norm2(rhs), std::numeric_limits<double>::min());
    const double rel_err = relative_error(x, x_ref);
    const Stats analysis = make_stats(analysis_ms);
    const Stats factor = make_stats(factor_ms);
    const Stats solve = make_stats(solve_ms);
    std::vector<double> total_solver;
    for (std::size_t i = 0; i < analysis_ms.size(); ++i) {
        total_solver.push_back(analysis_ms[i] + factor_ms[i] + solve_ms[i]);
    }
    const Stats total = make_stats(total_solver);
    const auto total_stop = Clock::now();

    std::filesystem::create_directories(opt.out.parent_path());
    std::ofstream out(opt.out);
    if (!out) {
        throw std::runtime_error("failed to write " + opt.out.string());
    }
    out << "{\n"
        << "  \"solver_name\": \"STRUMPACK-Ybus\",\n"
        << "  \"solver_version\": \"" << STRUMPACK_VERSION_MAJOR << "."
        << STRUMPACK_VERSION_MINOR << "." << STRUMPACK_VERSION_PATCH << "\",\n"
        << "  \"library_path\": \"" << escape_json(STRUMPACK_LIBRARY_PATH_STR) << "\",\n"
        << "  \"build_status\": \"ok\",\n"
        << "  \"matrix\": \"" << escape_json(opt.matrix.string()) << "\",\n"
        << "  \"dtype\": \"" << escape_json(opt.dtype) << "\",\n"
        << "  \"rows\": " << A.rows << ",\n"
        << "  \"cols\": " << A.cols << ",\n"
        << "  \"nnz\": " << A.nnz << ",\n"
        << "  \"repeats\": " << opt.repeats << ",\n"
        << "  \"warmup\": " << opt.warmup << ",\n"
        << "  \"diag_shift\": " << json_number(opt.diag_shift) << ",\n"
        << "  \"use_gpu\": " << (opt.use_gpu ? "true" : "false") << ",\n"
        << "  \"metis_node_ndp\": " << (opt.use_metis_node_ndp ? "true" : "false") << ",\n"
        << "  \"strumpack_use_cuda\": \"" << (strumpack_has_cuda() ? "yes" : "no") << "\",\n"
        << "  \"strumpack_use_slate\": \"" << (have_slate() ? "yes" : "no") << "\",\n"
        << "  \"strumpack_use_magma\": \"" << (have_magma() ? "yes" : "no") << "\",\n"
        << "  \"strumpack_use_kblas\": \"" << (have_kblas() ? "yes" : "no") << "\",\n"
        << "  \"analysis_ms\": "; write_stats(out, analysis);
    out << ",\n  \"factorization_ms\": "; write_stats(out, factor);
    out << ",\n  \"solve_ms\": "; write_stats(out, solve);
    out << ",\n  \"total_solver_ms\": "; write_stats(out, total);
    out << ",\n  \"load_ms\": " << json_number(elapsed_ms(load_start, load_stop)) << ",\n"
        << "  \"end_to_end_ms\": " << json_number(elapsed_ms(total_start, total_stop)) << ",\n"
        << "  \"relative_residual_2\": " << json_number(rel_res) << ",\n"
        << "  \"relative_error_2\": " << json_number(rel_err) << ",\n"
        << "  \"return_code\": " << last_code << ",\n"
        << "  \"gpu_memory_delta_mb\": "
        << json_number(opt.use_gpu && free_before > free_after
                       ? static_cast<double>(free_before - free_after) / (1024.0 * 1024.0)
                       : 0.0) << ",\n"
        << "  \"notes\": \"MatrixMarket complex/general benchmark. rhs is generated as A*x_ref; matrix is host CSR and STRUMPACK internal GPU offload is requested.\"\n"
        << "}\n";

    std::cout << "STRUMPACK-Ybus " << opt.matrix << "\n"
              << "  n=" << A.rows << " nnz=" << A.nnz
              << " dtype=" << opt.dtype
              << " gpu=" << (opt.use_gpu ? "on" : "off")
              << " metis_node_ndp=" << (opt.use_metis_node_ndp ? "on" : "off")
              << " diag_shift=" << opt.diag_shift << "\n"
              << "  analysis mean/median ms: " << analysis.mean << " / " << analysis.median << "\n"
              << "  factor   mean/median ms: " << factor.mean << " / " << factor.median << "\n"
              << "  solve    mean/median ms: " << solve.mean << " / " << solve.median << "\n"
              << "  total    mean/median ms: " << total.mean << " / " << total.median << "\n"
              << "  rel_res=" << rel_res << " rel_err=" << rel_err << " code=" << last_code << "\n"
              << "  wrote " << opt.out << "\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options opt = parse_args(argc, argv);
        if (opt.dtype == "fp64") {
            run_benchmark<std::complex<double>>(opt);
        } else {
            run_benchmark<std::complex<float>>(opt);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "strumpack_ybus_benchmark failed: " << e.what() << "\n";
        return 2;
    }
}
