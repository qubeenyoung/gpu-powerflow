#include <cuda_runtime.h>
#include <cudss.h>
#ifdef CLS_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "io.hpp"

namespace scripts = custom_linear_solver::scripts;
namespace fs = std::filesystem;

namespace {

class NvtxRange {
public:
#ifdef CLS_ENABLE_NVTX
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
#else
    explicit NvtxRange(const char*) {}
#endif

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

struct Options {
    fs::path matrix_path;
    fs::path rhs_path;
    fs::path solution_out;
    std::string threading_lib;
    bool write_solution = false;
    bool use_matching = false;
    bool split_analysis = false;
    bool use_threading_layer = false;
    bool mt_auto = false;       // --mt-auto: cudssSetThreadingLayer(handle, nullptr)
    int host_nthreads = 0;
    int repeat = 1;
    bool fp32 = false;          // --precision fp32 → use CUDA_R_32F and float buffers
    int batch = 1;              // --batch B: CUDSS_CONFIG_UBATCH_SIZE + batch-major buffers
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void upload(const std::vector<T>& host)
    {
        reset();
        count_ = host.size();
        if (host.empty()) return;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr_), host.size() * sizeof(T)),
                   "cudaMalloc");
        cuda_check(cudaMemcpy(ptr_, host.data(), host.size() * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");
    }

    void allocate(std::size_t count)
    {
        reset();
        count_ = count;
        if (count_ == 0) return;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)),
                   "cudaMalloc");
    }

    std::vector<T> download() const
    {
        std::vector<T> host(count_);
        if (count_ == 0) return host;
        cuda_check(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
        return host;
    }

    T* get() const { return ptr_; }

private:
    static void cuda_check(cudaError_t err, const char* context)
    {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
        }
    }

    void reset()
    {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

class CudssState {
public:
    ~CudssState()
    {
        if (solution) cudssMatrixDestroy(solution);
        if (rhs) cudssMatrixDestroy(rhs);
        if (matrix) cudssMatrixDestroy(matrix);
        if (data) cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
    }

    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matrix = nullptr;
    cudssMatrix_t rhs = nullptr;
    cudssMatrix_t solution = nullptr;
};

void cuda_check(cudaError_t err, const char* context)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
    }
}

void cudss_check(cudssStatus_t status, const char* context)
{
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(context) + " failed with cuDSS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

void usage(const char* argv0)
{
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " <case-dir> [options]\n"
        << "  " << argv0 << " --matrix J.mtx --rhs F.mtx [options]\n"
        << "\nOptions:\n"
        << "  --precision fp32|fp64   value/RHS/solution dtype (default fp64)\n"
        << "  --batch B               uniform-batch B systems (CUDSS_CONFIG_UBATCH_SIZE);\n"
        << "                          shares one sparsity pattern; default 1\n"
        << "  --repeat N              time-averaging trials (median reported)\n"
        << "  --mt-auto               cudssSetThreadingLayer(handle, nullptr) — let cuDSS\n"
        << "                          load its default threading layer (cuPF's pattern)\n"
        << "  --mt                    same as --mt-auto but uses --threading-lib path\n"
        << "  --threading-lib path    explicit libcudss_mtlayer_*.so for --mt\n"
        << "  --host-nthreads N       CUDSS_CONFIG_HOST_NTHREADS\n"
        << "  --matching              CUDSS_CONFIG_USE_MATCHING\n"
        << "  --split-analysis        report reordering vs symbolic factorization separately\n"
        << "  --solution-out path     write recovered x as MatrixMarket\n";
}

std::string default_threading_layer()
{
    const char* env_value = std::getenv("CUDSS_THREADING_LIB");
    if (env_value != nullptr && env_value[0] != '\0') {
        return env_value;
    }

    // Search for libcudss_mtlayer_*.so in conventional locations. The /opt path is the
    // system-wide install; the python-package path is the pip-installed copy.
    const fs::path candidates[] = {
        "/opt/nvidia/cudss/lib/libcudss_mtlayer_gomp.so",
        "/opt/nvidia/cudss/lib/libcudss_mtlayer_gomp.so.0",
        "/usr/local/lib/python3.10/dist-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0",
        "/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0",
    };
    std::error_code ec;
    for (const fs::path& candidate : candidates) {
        if (fs::exists(candidate, ec)) return candidate.string();
    }
    return candidates[0].string();
}

int positive_env_or(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    try {
        const int parsed = std::stoi(value);
        return parsed > 0 ? parsed : fallback;
    } catch (...) {
        return fallback;
    }
}

Options parse_args(int argc, char** argv)
{
    Options options;
    fs::path case_dir;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--matrix") {
            if (++i >= argc) throw std::runtime_error("--matrix requires a path");
            options.matrix_path = argv[i];
        } else if (arg == "--rhs") {
            if (++i >= argc) throw std::runtime_error("--rhs requires a path");
            options.rhs_path = argv[i];
        } else if (arg == "--solution-out") {
            if (++i >= argc) throw std::runtime_error("--solution-out requires a path");
            options.solution_out = argv[i];
            options.write_solution = true;
        } else if (arg == "--repeat") {
            if (++i >= argc) throw std::runtime_error("--repeat requires a count");
            options.repeat = std::stoi(argv[i]);
            if (options.repeat <= 0) throw std::runtime_error("--repeat must be positive");
        } else if (arg == "--matching") {
            options.use_matching = true;
        } else if (arg == "--split-analysis") {
            options.split_analysis = true;
        } else if (arg == "--mt") {
            options.use_threading_layer = true;
        } else if (arg == "--mt-auto") {
            options.use_threading_layer = true;
            options.mt_auto = true;
        } else if (arg == "--batch") {
            if (++i >= argc) throw std::runtime_error("--batch requires a count");
            options.batch = std::stoi(argv[i]);
            if (options.batch <= 0) throw std::runtime_error("--batch must be positive");
        } else if (arg == "--threading-lib") {
            if (++i >= argc) throw std::runtime_error("--threading-lib requires a path");
            options.threading_lib = argv[i];
            options.use_threading_layer = true;
        } else if (arg == "--host-nthreads") {
            if (++i >= argc) throw std::runtime_error("--host-nthreads requires a count");
            options.host_nthreads = std::stoi(argv[i]);
            if (options.host_nthreads <= 0) {
                throw std::runtime_error("--host-nthreads must be positive");
            }
        } else if (arg == "--precision") {
            if (++i >= argc) throw std::runtime_error("--precision requires fp32 or fp64");
            const std::string value = argv[i];
            if (value == "fp32") options.fp32 = true;
            else if (value == "fp64") options.fp32 = false;
            else throw std::runtime_error("--precision must be fp32 or fp64");
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (case_dir.empty()) {
            case_dir = arg;
        } else {
            throw std::runtime_error("unexpected argument: " + arg);
        }
    }

    if (!case_dir.empty()) {
        if (!options.matrix_path.empty() || !options.rhs_path.empty()) {
            throw std::runtime_error("use either <case-dir> or --matrix/--rhs, not both");
        }
        options.matrix_path = case_dir / "J.mtx";
        options.rhs_path = case_dir / "F.mtx";
    }
    if (options.matrix_path.empty() || options.rhs_path.empty()) {
        throw std::runtime_error("matrix and rhs paths are required");
    }
    return options;
}

double l2_norm(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values) {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct ResidualStats {
    double abs_l2 = 0.0;
    double rel_l2 = 0.0;
    double abs_inf = 0.0;
};

ResidualStats residual(const scripts::CsrMatrix& matrix, const std::vector<double>& rhs,
                       const std::vector<double>& x)
{
    std::vector<double> r(static_cast<std::size_t>(matrix.rows), 0.0);
    for (int row = 0; row < matrix.rows; ++row) {
        double ax = 0.0;
        for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p) {
            ax += matrix.values[p] * x[matrix.col_idx[p]];
        }
        r[row] = ax - rhs[row];
    }

    ResidualStats stats;
    stats.abs_l2 = l2_norm(r);
    const double rhs_l2 = l2_norm(rhs);
    stats.rel_l2 = stats.abs_l2 / std::max(rhs_l2, 1.0e-300);
    for (double value : r) {
        stats.abs_inf = std::max(stats.abs_inf, std::abs(value));
    }
    return stats;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_args(argc, argv);
        const auto t0 = std::chrono::steady_clock::now();
        scripts::CsrMatrix matrix;
        scripts::DenseVector rhs;
        {
            NvtxRange range("cudss_read_input");
            matrix = scripts::read_matrix_market_csr(options.matrix_path);
            rhs = scripts::read_matrix_market_vector(options.rhs_path);
        }
        const auto t_read = std::chrono::steady_clock::now();

        if (matrix.rows != matrix.cols) {
            throw std::runtime_error("cuDSS runner requires a square matrix");
        }
        if (rhs.size() != matrix.rows) {
            throw std::runtime_error("rhs length does not match matrix dimension");
        }

        const cudaDataType_t cuda_dtype = options.fp32 ? CUDA_R_32F : CUDA_R_64F;
        const size_t elt_bytes = options.fp32 ? sizeof(float) : sizeof(double);
        const int B = options.batch;
        const int n = matrix.rows;
        const int nnz_j = matrix.nnz();

        DeviceBuffer<int> d_row_ptr;
        DeviceBuffer<int> d_col_idx;
        DeviceBuffer<double> d_values;       // fp64 path (batch-major: B × nnz)
        DeviceBuffer<double> d_rhs;          //                          B × n
        DeviceBuffer<double> d_solution;     //                          B × n
        DeviceBuffer<float>  d_values_f;     // fp32 path
        DeviceBuffer<float>  d_rhs_f;
        DeviceBuffer<float>  d_solution_f;
        void* p_values   = nullptr;
        void* p_rhs      = nullptr;
        void* p_solution = nullptr;
        {
            NvtxRange range("cudss_upload_input");
            d_row_ptr.upload(matrix.row_ptr);
            d_col_idx.upload(matrix.col_idx);
            // Replicate values / RHS B times so cuDSS sees B systems with the same sparsity.
            if (options.fp32) {
                std::vector<float> values_f((std::size_t)B * nnz_j);
                std::vector<float> rhs_f((std::size_t)B * n);
                for (int b = 0; b < B; ++b) {
                    std::transform(matrix.values.begin(), matrix.values.end(),
                                   values_f.begin() + (std::size_t)b * nnz_j,
                                   [](double v){ return static_cast<float>(v); });
                    std::transform(rhs.values.begin(), rhs.values.end(),
                                   rhs_f.begin() + (std::size_t)b * n,
                                   [](double v){ return static_cast<float>(v); });
                }
                d_values_f.upload(values_f);
                d_rhs_f.upload(rhs_f);
                d_solution_f.allocate((std::size_t)B * n);
                cuda_check(cudaMemset(d_solution_f.get(), 0, (std::size_t)B * n * sizeof(float)),
                           "cudaMemset solution");
                p_values   = d_values_f.get();
                p_rhs      = d_rhs_f.get();
                p_solution = d_solution_f.get();
            } else {
                std::vector<double> valB((std::size_t)B * nnz_j);
                std::vector<double> rhsB((std::size_t)B * n);
                for (int b = 0; b < B; ++b) {
                    std::copy(matrix.values.begin(), matrix.values.end(),
                              valB.begin() + (std::size_t)b * nnz_j);
                    std::copy(rhs.values.begin(), rhs.values.end(),
                              rhsB.begin() + (std::size_t)b * n);
                }
                d_values.upload(valB);
                d_rhs.upload(rhsB);
                d_solution.allocate((std::size_t)B * n);
                cuda_check(cudaMemset(d_solution.get(), 0, (std::size_t)B * n * sizeof(double)),
                           "cudaMemset solution");
                p_values   = d_values.get();
                p_rhs      = d_rhs.get();
                p_solution = d_solution.get();
            }
            (void)elt_bytes;
        }
        const auto t_upload = std::chrono::steady_clock::now();

        CudssState state;
        {
            NvtxRange range("cudss_setup");
            cudss_check(cudssCreate(&state.handle), "cudssCreate");
            std::string threading_lib;
            if (options.use_threading_layer) {
                // --mt-auto matches cuPF's intent (let cuDSS pick the default mt layer). The
                // current cuDSS build rejects nullptr, so we resolve the bundled
                // libcudss_mtlayer_*.so path the same way default_threading_layer() does.
                if (options.mt_auto) {
                    threading_lib = default_threading_layer();
                } else {
                    threading_lib = options.threading_lib.empty() ? default_threading_layer()
                                                                  : options.threading_lib;
                }
                cudss_check(cudssSetThreadingLayer(state.handle, threading_lib.c_str()),
                            "cudssSetThreadingLayer");
            }
            cudss_check(cudssConfigCreate(&state.config), "cudssConfigCreate");
            cudss_check(cudssDataCreate(state.handle, &state.data), "cudssDataCreate");

            cudssAlgType_t reordering_alg = CUDSS_ALG_DEFAULT;
            cudss_check(cudssConfigSet(state.config, CUDSS_CONFIG_REORDERING_ALG, &reordering_alg,
                                       sizeof(reordering_alg)),
                        "cudssConfigSet REORDERING_ALG");
            if (B > 1) {
                int ubatch_size = B;
                cudss_check(cudssConfigSet(state.config, CUDSS_CONFIG_UBATCH_SIZE,
                                           &ubatch_size, sizeof(ubatch_size)),
                            "cudssConfigSet UBATCH_SIZE");
            }
            if (options.use_matching) {
                int use_matching = 1;
                cudss_check(cudssConfigSet(state.config, CUDSS_CONFIG_USE_MATCHING, &use_matching,
                                           sizeof(use_matching)),
                            "cudssConfigSet USE_MATCHING");
            }
            if (options.use_threading_layer || options.host_nthreads > 0) {
                int host_nthreads = options.host_nthreads;
                if (host_nthreads <= 0) {
                    host_nthreads = positive_env_or(
                        "CUDSS_HOST_NTHREADS",
                        static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
                }
                cudss_check(cudssConfigSet(state.config, CUDSS_CONFIG_HOST_NTHREADS,
                                           &host_nthreads, sizeof(host_nthreads)),
                            "cudssConfigSet HOST_NTHREADS");
            }

            cudss_check(cudssMatrixCreateCsr(
                            &state.matrix, matrix.rows, matrix.cols, matrix.nnz(), d_row_ptr.get(),
                            nullptr, d_col_idx.get(), p_values, CUDA_R_32I, cuda_dtype,
                            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
                        "cudssMatrixCreateCsr");
            cudss_check(cudssMatrixCreateDn(&state.rhs, matrix.rows, 1, matrix.rows, p_rhs,
                                            cuda_dtype, CUDSS_LAYOUT_COL_MAJOR),
                        "cudssMatrixCreateDn rhs");
            cudss_check(cudssMatrixCreateDn(&state.solution, matrix.cols, 1, matrix.cols,
                                            p_solution, cuda_dtype, CUDSS_LAYOUT_COL_MAJOR),
                        "cudssMatrixCreateDn solution");
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize setup");
        }
        const auto t_setup = std::chrono::steady_clock::now();

        double reordering_ms = 0.0;
        double symbolic_factorization_ms = 0.0;
        if (options.split_analysis) {
            {
                NvtxRange range("cudss_reordering");
                const auto start = std::chrono::steady_clock::now();
                cudss_check(cudssExecute(state.handle, CUDSS_PHASE_REORDERING, state.config,
                                         state.data, state.matrix, state.solution, state.rhs),
                            "cuDSS reordering");
                cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize reordering");
                const auto stop = std::chrono::steady_clock::now();
                reordering_ms =
                    std::chrono::duration<double, std::milli>(stop - start).count();
            }
            {
                NvtxRange range("cudss_symbolic_factorization");
                const auto start = std::chrono::steady_clock::now();
                cudss_check(cudssExecute(state.handle, CUDSS_PHASE_SYMBOLIC_FACTORIZATION,
                                         state.config, state.data, state.matrix, state.solution,
                                         state.rhs),
                            "cuDSS symbolic factorization");
                cuda_check(cudaDeviceSynchronize(),
                           "cudaDeviceSynchronize symbolic factorization");
                const auto stop = std::chrono::steady_clock::now();
                symbolic_factorization_ms =
                    std::chrono::duration<double, std::milli>(stop - start).count();
            }
        } else {
            NvtxRange range("cudss_analysis");
            cudss_check(cudssExecute(state.handle, CUDSS_PHASE_ANALYSIS, state.config, state.data,
                                     state.matrix, state.solution, state.rhs),
                        "cuDSS analysis");
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize analysis");
        }
        const auto t_analyze = std::chrono::steady_clock::now();

        std::vector<double> factor_ms;
        std::vector<double> solve_ms;
        factor_ms.reserve(static_cast<std::size_t>(options.repeat));
        solve_ms.reserve(static_cast<std::size_t>(options.repeat));

        for (int r = 0; r < options.repeat; ++r) {
            const std::string range_name = "cudss_factorize_" + std::to_string(r);
            NvtxRange range(range_name.c_str());
            const auto start = std::chrono::steady_clock::now();
            cudss_check(cudssExecute(state.handle, CUDSS_PHASE_FACTORIZATION, state.config,
                                     state.data, state.matrix, state.solution, state.rhs),
                        "cuDSS factorization");
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize factorization");
            const auto stop = std::chrono::steady_clock::now();
            factor_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        }

        for (int r = 0; r < options.repeat; ++r) {
            const std::string range_name = "cudss_solve_" + std::to_string(r);
            NvtxRange range(range_name.c_str());
            const auto start = std::chrono::steady_clock::now();
            cudss_check(cudssExecute(state.handle, CUDSS_PHASE_SOLVE, state.config, state.data,
                                     state.matrix, state.solution, state.rhs),
                        "cuDSS solve");
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize solve");
            const auto stop = std::chrono::steady_clock::now();
            solve_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        }

        const auto t_solve = std::chrono::steady_clock::now();
        std::vector<double> solution;
        ResidualStats stats;
        double batch_max_relres = 0.0;
        {
            NvtxRange range("cudss_download_check");
            if (options.fp32) {
                const std::vector<float> sf = d_solution_f.download();
                solution.assign(sf.begin() + 0, sf.begin() + n);                     // batch 0 → residual
                // Max relres over all B systems (sanity check; all B copies are identical so this
                // should match the batch-0 number except for rounding noise).
                std::vector<double> xb(n);
                for (int b = 0; b < B; ++b) {
                    std::transform(sf.begin() + (std::size_t)b * n,
                                   sf.begin() + (std::size_t)(b + 1) * n,
                                   xb.begin(), [](float v){ return static_cast<double>(v); });
                    batch_max_relres = std::max(batch_max_relres,
                                                residual(matrix, rhs.values, xb).rel_l2);
                }
            } else {
                const std::vector<double> sd = d_solution.download();
                solution.assign(sd.begin() + 0, sd.begin() + n);
                std::vector<double> xb(n);
                for (int b = 0; b < B; ++b) {
                    std::copy(sd.begin() + (std::size_t)b * n,
                              sd.begin() + (std::size_t)(b + 1) * n, xb.begin());
                    batch_max_relres = std::max(batch_max_relres,
                                                residual(matrix, rhs.values, xb).rel_l2);
                }
            }
            stats = residual(matrix, rhs.values, solution);
            if (options.write_solution) {
                scripts::write_matrix_market_vector(options.solution_out, solution);
            }
        }
        const auto t_done = std::chrono::steady_clock::now();

        const auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const double factor_median = median(factor_ms);
        const double solve_median = median(solve_ms);

        std::cout << "matrix=" << options.matrix_path << '\n'
                  << "rhs=" << options.rhs_path << '\n'
                  << "n=" << matrix.rows << " nnz=" << matrix.nnz() << '\n'
                  << "precision=" << (options.fp32 ? "fp32" : "fp64") << '\n'
                  << "repeat=" << options.repeat << '\n'
                  << "matching=" << (options.use_matching ? 1 : 0) << '\n'
                  << "split_analysis=" << (options.split_analysis ? 1 : 0) << '\n'
                  << "mt=" << (options.use_threading_layer ? 1 : 0) << '\n'
                  << "threading_lib="
                  << (options.use_threading_layer
                          ? (options.threading_lib.empty() ? default_threading_layer()
                                                           : options.threading_lib)
                          : "")
                  << '\n'
                  << "host_nthreads="
                  << ((options.use_threading_layer || options.host_nthreads > 0)
                          ? (options.host_nthreads > 0
                                 ? options.host_nthreads
                                 : positive_env_or("CUDSS_HOST_NTHREADS",
                                                   static_cast<int>(std::max(
                                                       1u, std::thread::hardware_concurrency()))))
                          : 0)
                  << '\n'
                  << "batch=" << B << '\n'
                  << "mt_auto=" << (options.mt_auto ? 1 : 0) << '\n'
                  << "read_ms=" << ms(t0, t_read) << '\n'
                  << "upload_ms=" << ms(t_read, t_upload) << '\n'
                  << "setup_ms=" << ms(t_upload, t_setup) << '\n'
                  << "reordering_ms=" << reordering_ms << '\n'
                  << "symbolic_factorization_ms=" << symbolic_factorization_ms << '\n'
                  << "analyze_ms=" << ms(t_setup, t_analyze) << '\n'
                  << "factorize_ms=" << factor_median << '\n'
                  << "solve_ms=" << solve_median << '\n'
                  << "factorize_per_sys_ms=" << factor_median / B << '\n'
                  << "solve_per_sys_ms=" << solve_median / B << '\n'
                  << "batch_relres=" << batch_max_relres << '\n'
                  << "download_check_ms=" << ms(t_solve, t_done) << '\n'
                  << "residual_l2=" << stats.abs_l2 << '\n'
                  << "relative_residual_l2=" << stats.rel_l2 << '\n'
                  << "residual_inf=" << stats.abs_inf << '\n';
        if (options.write_solution) {
            std::cout << "solution_out=" << options.solution_out << '\n';
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        usage(argv[0]);
        return 1;
    }
}
