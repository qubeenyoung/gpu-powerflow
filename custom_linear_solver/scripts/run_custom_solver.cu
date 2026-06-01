#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io.hpp"
#include "solver.hpp"

namespace cls = custom_linear_solver;
namespace scripts = custom_linear_solver::scripts;
namespace fs = std::filesystem;

// FP64 CSR residual r = rhs - A*x and y += x, for iterative-refinement experiments.
__global__ void spmv_residual(int n, const int* __restrict__ row_ptr, const int* __restrict__ col,
                              const double* __restrict__ val, const double* __restrict__ x,
                              const double* __restrict__ rhs, double* __restrict__ r)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double ax = 0.0;
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) ax += val[p] * x[col[p]];
    r[i] = rhs[i] - ax;
}
__global__ void axpy_inplace(int n, const double* __restrict__ dx, double* __restrict__ x)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += dx[i];
}

namespace {

struct Options {
    fs::path matrix_path;
    fs::path rhs_path;
    fs::path solution_out;
    bool write_solution = false;
    int repeat = 1;
    int ir = 0;  // iterative-refinement steps after the solve (low-precision-solve correction)
    int batch = 0;  // uniform-batch experiment: B systems sharing the sparsity pattern
    bool batch_only = false;  // skip the single-system factor/solve (e.g. nc>16 amalgamation)
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

void cuda_check(cudaError_t err, const char* context)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
    }
}

void usage(const char* argv0)
{
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " <case-dir> [--repeat N] [--solution-out x.mtx]\n"
        << "  " << argv0 << " --matrix J.mtx --rhs F.mtx [--repeat N] [--solution-out x.mtx]\n";
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
        } else if (arg == "--ir") {
            if (++i >= argc) throw std::runtime_error("--ir requires a count");
            options.ir = std::stoi(argv[i]);
        } else if (arg == "--batch") {
            if (++i >= argc) throw std::runtime_error("--batch requires a count");
            options.batch = std::stoi(argv[i]);
        } else if (arg == "--batch-only") {
            options.batch_only = true;
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

void require_success(cls::Status status, const char* phase)
{
    if (status != cls::Status::Success) {
        throw std::runtime_error(std::string(phase) + " failed: " + cls::status_string(status));
    }
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
        scripts::CsrMatrix matrix = scripts::read_matrix_market_csr(options.matrix_path);
        scripts::DenseVector rhs = scripts::read_matrix_market_vector(options.rhs_path);
        const auto t_read = std::chrono::steady_clock::now();

        if (matrix.rows != matrix.cols) {
            throw std::runtime_error("custom solver requires a square matrix");
        }
        if (rhs.size() != matrix.rows) {
            throw std::runtime_error("rhs length does not match matrix dimension");
        }
        if (matrix.nnz() != static_cast<int64_t>(matrix.col_idx.size())) {
            throw std::runtime_error("invalid CSR nonzero arrays");
        }

        DeviceBuffer<int> d_row_ptr;
        DeviceBuffer<int> d_col_idx;
        DeviceBuffer<double> d_values;
        DeviceBuffer<double> d_rhs;
        DeviceBuffer<double> d_solution;
        d_row_ptr.upload(matrix.row_ptr);
        d_col_idx.upload(matrix.col_idx);
        d_values.upload(matrix.values);
        d_rhs.upload(rhs.values);
        d_solution.allocate(rhs.values.size());
        const auto t_upload = std::chrono::steady_clock::now();

        cls::Solver solver;
        cls::CsrMatrixView matrix_view;
        matrix_view.nrows = matrix.rows;
        matrix_view.ncols = matrix.cols;
        matrix_view.nnz = matrix.nnz();
        matrix_view.index_type = cls::IndexType::Int32;
        matrix_view.location = cls::DataLocation::Device;
        matrix_view.row_offsets = d_row_ptr.get();
        matrix_view.col_indices = d_col_idx.get();
        matrix_view.values = d_values.get();

        cls::DenseVectorView rhs_view;
        rhs_view.size = rhs.size();
        rhs_view.location = cls::DataLocation::Device;
        rhs_view.values = d_rhs.get();

        cls::DenseVectorView solution_view;
        solution_view.size = rhs.size();
        solution_view.location = cls::DataLocation::Device;
        solution_view.values = d_solution.get();

        require_success(solver.set_data(matrix_view), "set_data");
        require_success(solver.set_rhs(rhs_view), "set_rhs");
        require_success(solver.set_solution(solution_view), "set_solution");
        const auto t_set = std::chrono::steady_clock::now();

        require_success(solver.analyze(), "analyze");
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after analyze");
        const auto t_analyze = std::chrono::steady_clock::now();

        std::vector<double> factor_ms;
        std::vector<double> solve_ms;
        std::vector<double> factor_kms;
        std::vector<double> solve_kms;
        factor_ms.reserve(static_cast<std::size_t>(options.repeat));
        solve_ms.reserve(static_cast<std::size_t>(options.repeat));
        const bool kernel_time = std::getenv("CLS_KERNEL_TIME") != nullptr;
        if (options.batch_only) { factor_ms.push_back(0); solve_ms.push_back(0); }

        for (int r = 0; r < options.repeat && !options.batch_only; ++r) {
            double kms = 0.0;
            const auto start = std::chrono::steady_clock::now();
            require_success(solver.factorize(kernel_time ? &kms : nullptr), "factorize");
            const auto stop = std::chrono::steady_clock::now();
            factor_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
            if (kernel_time) factor_kms.push_back(kms);
        }

        for (int r = 0; r < options.repeat && !options.batch_only; ++r) {
            double kms = 0.0;
            const auto start = std::chrono::steady_clock::now();
            require_success(solver.solve(kernel_time ? &kms : nullptr), "solve");
            const auto stop = std::chrono::steady_clock::now();
            solve_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
            if (kernel_time) solve_kms.push_back(kms);
        }
        const auto t_solve = std::chrono::steady_clock::now();

        // Iterative refinement (optional): correct the (possibly low-precision) solve in FP64.
        // r = rhs - A x  (FP64 spmv); dx = solve(r) (reuses the existing factor); x += dx.
        double ir_ms = 0.0;
        if (options.ir > 0 && !options.batch_only) {
            const int n = matrix.rows;
            const int T = 256, nb = (n + T - 1) / T;
            DeviceBuffer<double> d_r, d_dx;
            d_r.allocate(static_cast<std::size_t>(n));
            d_dx.allocate(static_cast<std::size_t>(n));
            cls::DenseVectorView rview;
            rview.size = n; rview.location = cls::DataLocation::Device; rview.values = d_r.get();
            cls::DenseVectorView dxview;
            dxview.size = n; dxview.location = cls::DataLocation::Device; dxview.values = d_dx.get();
            cuda_check(cudaDeviceSynchronize(), "sync before IR");
            const auto ir0 = std::chrono::steady_clock::now();
            for (int step = 0; step < options.ir; ++step) {
                spmv_residual<<<nb, T>>>(n, d_row_ptr.get(), d_col_idx.get(), d_values.get(),
                                         d_solution.get(), d_rhs.get(), d_r.get());
                require_success(solver.set_rhs(rview), "set_rhs(ir)");
                require_success(solver.set_solution(dxview), "set_solution(ir)");
                require_success(solver.solve(), "solve(ir)");
                axpy_inplace<<<nb, T>>>(n, d_dx.get(), d_solution.get());
            }
            cuda_check(cudaDeviceSynchronize(), "sync after IR");
            ir_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - ir0).count();
            // restore original rhs/solution registration for any later use
            require_success(solver.set_rhs(rhs_view), "set_rhs(restore)");
            require_success(solver.set_solution(solution_view), "set_solution(restore)");
        }

        std::vector<double> solution = d_solution.download();
        const ResidualStats stats = residual(matrix, rhs.values, solution);
        if (options.write_solution) {
            scripts::write_matrix_market_vector(options.solution_out, solution);
        }
        const auto t_done = std::chrono::steady_clock::now();

        // ---- Uniform-batch experiment (B systems, same pattern) -------------------------
        double batch_factor_per_ms = 0, batch_solve_per_ms = 0, batch_relres = 0, batch_ir_per_ms = 0;
        if (options.batch > 0) {
            const int B = options.batch;
            const int n = matrix.rows, nnz = matrix.nnz();
            const bool mixed = std::getenv("MF_NO_MIXED") == nullptr &&
                               (std::getenv("MF_MIXED") != nullptr || n < 24000);
            // B identical copies (same pattern + values) -> each batch solves the same system,
            // so the per-batch residual must match the single-system solve (correctness check).
            std::vector<double> hvalB((std::size_t)B * nnz), hrhsB((std::size_t)B * n);
            for (int b = 0; b < B; ++b) {
                std::copy(matrix.values.begin(), matrix.values.end(), hvalB.begin() + (std::size_t)b * nnz);
                std::copy(rhs.values.begin(), rhs.values.end(), hrhsB.begin() + (std::size_t)b * n);
            }
            DeviceBuffer<double> d_valB, d_rhsB, d_solB;
            d_valB.upload(hvalB);
            d_rhsB.upload(hrhsB);
            d_solB.allocate((std::size_t)B * n);
            require_success(solver.batched_setup(B, mixed), "batched_setup");
            std::vector<double> bf, bs;
            for (int r = 0; r < options.repeat; ++r) {
                double kf = 0, ks = 0;
                require_success(solver.batched_factorize(d_valB.get(), &kf), "batched_factorize");
                require_success(solver.batched_solve(d_rhsB.get(), d_solB.get(), &ks), "batched_solve");
                bf.push_back(kf);
                bs.push_back(ks);
            }
            // optional batched IR (FP64 correction across all B)
            if (options.ir > 0) {
                const int T = 256;
                DeviceBuffer<double> d_rB, d_dxB;
                d_rB.allocate((std::size_t)B * n);
                d_dxB.allocate((std::size_t)B * n);
                cuda_check(cudaDeviceSynchronize(), "sync before batch IR");
                const auto bir0 = std::chrono::steady_clock::now();
                for (int step = 0; step < options.ir; ++step) {
                    for (int b = 0; b < B; ++b)
                        spmv_residual<<<(n + T - 1) / T, T>>>(
                            n, d_row_ptr.get(), d_col_idx.get(), d_valB.get() + (std::size_t)b * nnz,
                            d_solB.get() + (std::size_t)b * n, d_rhsB.get() + (std::size_t)b * n,
                            d_rB.get() + (std::size_t)b * n);
                    require_success(solver.batched_solve(d_rB.get(), d_dxB.get(), nullptr),
                                    "batched_solve(ir)");
                    axpy_inplace<<<((std::size_t)B * n + T - 1) / T, T>>>((int)((std::size_t)B * n),
                                                                         d_dxB.get(), d_solB.get());
                }
                cuda_check(cudaDeviceSynchronize(), "sync after batch IR");
                batch_ir_per_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - bir0).count() / B;
            }
            batch_factor_per_ms = median(bf) / B;
            batch_solve_per_ms = median(bs) / B;
            std::vector<double> solB((std::size_t)B * n);
            cuda_check(cudaMemcpy(solB.data(), d_solB.get(), (std::size_t)B * n * sizeof(double),
                                  cudaMemcpyDeviceToHost), "D2H batch sol");
            // max relres over ALL B batches (catches per-batch indexing bugs, not just batch 0)
            for (int b = 0; b < B; ++b) {
                std::vector<double> xb(solB.begin() + (std::size_t)b * n,
                                       solB.begin() + (std::size_t)(b + 1) * n);
                batch_relres = std::max(batch_relres, residual(matrix, rhs.values, xb).rel_l2);
            }
        }

        const auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        std::cout << "matrix=" << options.matrix_path << '\n'
                  << "rhs=" << options.rhs_path << '\n'
                  << "n=" << matrix.rows << " nnz=" << matrix.nnz() << '\n'
                  << "repeat=" << options.repeat << '\n'
                  << "read_ms=" << ms(t0, t_read) << '\n'
                  << "upload_ms=" << ms(t_read, t_upload) << '\n'
                  << "set_ms=" << ms(t_upload, t_set) << '\n'
                  << "analyze_ms=" << ms(t_set, t_analyze) << '\n'
                  << "factorize_ms=" << median(factor_ms) << '\n'
                  << "solve_ms=" << median(solve_ms) << '\n';
        if (kernel_time) {
            std::cout << "factorize_kernel_ms=" << median(factor_kms) << '\n'
                      << "solve_kernel_ms=" << median(solve_kms) << '\n';
        }
        if (options.ir > 0) {
            std::cout << "ir_steps=" << options.ir << '\n'
                      << "ir_total_ms=" << ir_ms << '\n'
                      << "solve_plus_ir_ms=" << (median(solve_ms) + ir_ms) << '\n';
        }
        if (options.batch > 0) {
            std::cout << "batch=" << options.batch << '\n'
                      << "batch_factor_per_sys_ms=" << batch_factor_per_ms << '\n'
                      << "batch_solve_per_sys_ms=" << batch_solve_per_ms << '\n'
                      << "batch_relres=" << batch_relres << '\n';
            if (options.ir > 0)
                std::cout << "batch_ir_per_sys_ms=" << batch_ir_per_ms << '\n'
                          << "batch_solve_plus_ir_per_sys_ms="
                          << (batch_solve_per_ms + batch_ir_per_ms) << '\n';
        }
        std::cout
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
