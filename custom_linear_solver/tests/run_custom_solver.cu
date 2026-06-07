#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "multifrontal.hpp"
#include "io.hpp"
#include "solver.hpp"

namespace cls = custom_linear_solver;
namespace scripts = custom_linear_solver::scripts;
namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path matrix_path;
    fs::path rhs_path;
    fs::path solution_out;
    fs::path dump_fronts_path;
    bool write_solution = false;
    int repeat = 1;
    int batch = 0;  // uniform-batch experiment: B systems sharing the sparsity pattern
    bool batch_only = false;  // skip the single-system factor/solve
    bool single_fp32 = false;
    int panel_cap = 8;  // analyze: max panel width inside a supernode; -1 keeps default
    bool no_multistream = false;     // disable subtree multi-stream dispatch
    bool analyze_info = false;       // print analyze-phase front/subtree summary
    // Numeric precision. See multifrontal.hpp for what each mode does.
    //   fp64 / fp32 / fp16 (FP16 WMMA) / tf32_wmma (V0 WMMA) / tf32 (V9h PTX, recommended).
    cls::Precision precision = cls::Precision::FP64;
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
        << "  " << argv0 << " <case-dir> [options]\n"
        << "  " << argv0 << " --matrix J.mtx --rhs F.mtx [options]\n"
        << "\nOptions:\n"
        << "  --repeat N            time-averaging trials (median reported)\n"
        << "  --single-precision fp64|fp32\n"
        << "                        single-system (non-batch) input precision\n"
        << "  --batch B             also run a uniform-batch experiment with B systems\n"
        << "  --batch-only          skip the single-system run\n"
        << "  --precision MODE      batched factor precision; MODE one of\n"
        << "                          fp64       (reference, ~1e-13)\n"
        << "                          fp32       (~1e-4, ~2x speed of FP64)\n"
        << "                          fp16       (FP32 + FP16 WMMA trailing)\n"
        << "                          tf32_wmma  (FP32 + TF32 WMMA trailing — V0 baseline)\n"
        << "                          tf32       (FP32 + TF32 PTX trailing — V9h+LB, recommended)\n"
        << "  --panel-cap N         analyze: max panel width inside a supernode (1..64)\n"
        << "  --no-multistream      disable subtree multi-stream dispatch (single stream)\n"
        << "  --analyze-info        print front-size and subtree summary after analyze\n"
        << "  --dump-fronts <path>  write (q,p,fsz,nc,uc,level) CSV after analyze\n"
        << "  --solution-out <path> write the recovered x as MatrixMarket\n";
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
        } else if (arg == "--batch") {
            if (++i >= argc) throw std::runtime_error("--batch requires a count");
            options.batch = std::stoi(argv[i]);
        } else if (arg == "--batch-only") {
            options.batch_only = true;
        } else if (arg == "--single-precision") {
            if (++i >= argc) throw std::runtime_error("--single-precision requires fp64 or fp32");
            const std::string value = argv[i];
            if (value == "fp64") options.single_fp32 = false;
            else if (value == "fp32") options.single_fp32 = true;
            else throw std::runtime_error("--single-precision must be fp64 or fp32");
        } else if (arg == "--panel-cap") {
            if (++i >= argc) throw std::runtime_error("--panel-cap requires an int");
            options.panel_cap = std::stoi(argv[i]);
        } else if (arg == "--no-multistream") {
            options.no_multistream = true;
        } else if (arg == "--analyze-info") {
            options.analyze_info = true;
        } else if (arg == "--dump-fronts") {
            if (++i >= argc) throw std::runtime_error("--dump-fronts requires a path");
            options.dump_fronts_path = argv[i];
        } else if (arg == "--precision") {
            if (++i >= argc) throw std::runtime_error(
                "--precision requires fp64|fp32|fp16|tf32_wmma|tf32");
            const std::string value = argv[i];
            using cls::Precision;
            if      (value == "fp64")      options.precision = Precision::FP64;
            else if (value == "fp32")      options.precision = Precision::FP32;
            else if (value == "fp16")      options.precision = Precision::FP16;
            else if (value == "tf32_wmma") options.precision = Precision::TF32_WMMA;
            else if (value == "tf32")      options.precision = Precision::TF32;
            else throw std::runtime_error(
                "--precision must be fp64|fp32|fp16|tf32_wmma|tf32");
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
        DeviceBuffer<float> d_values_f;
        DeviceBuffer<float> d_rhs_f;
        DeviceBuffer<float> d_solution_f;
        d_row_ptr.upload(matrix.row_ptr);
        d_col_idx.upload(matrix.col_idx);
        if (options.single_fp32) {
            std::vector<float> values_f(matrix.values.begin(), matrix.values.end());
            std::vector<float> rhs_f(rhs.values.begin(), rhs.values.end());
            d_values_f.upload(values_f);
            d_rhs_f.upload(rhs_f);
            d_solution_f.allocate(rhs.values.size());
        } else {
            d_values.upload(matrix.values);
            d_rhs.upload(rhs.values);
            d_solution.allocate(rhs.values.size());
        }
        const auto t_upload = std::chrono::steady_clock::now();

        cls::SolverConfig solver_config;
        solver_config.precision = options.precision;
        if (options.panel_cap > 0) solver_config.panel_cap = options.panel_cap;
        solver_config.use_multistream_subtrees = !options.no_multistream;
        solver_config.analyze_emit_info = options.analyze_info;
        if (!options.dump_fronts_path.empty()) {
            solver_config.analyze_dump_fronts_path = options.dump_fronts_path.string();
        }
        cls::Solver solver(solver_config);
        cls::CsrMatrixView matrix_view;
        matrix_view.nrows = matrix.rows;
        matrix_view.ncols = matrix.cols;
        matrix_view.nnz = matrix.nnz();
        matrix_view.index_type = cls::IndexType::Int32;
        matrix_view.location = cls::DataLocation::Device;
        matrix_view.value_type =
            options.single_fp32 ? cls::ValueType::Float32 : cls::ValueType::Float64;
        matrix_view.row_offsets = d_row_ptr.get();
        matrix_view.col_indices = d_col_idx.get();
        matrix_view.values = options.single_fp32
                                 ? static_cast<const void*>(d_values_f.get())
                                 : static_cast<const void*>(d_values.get());

        cls::DenseVectorView rhs_view;
        rhs_view.size = rhs.size();
        rhs_view.location = cls::DataLocation::Device;
        rhs_view.value_type =
            options.single_fp32 ? cls::ValueType::Float32 : cls::ValueType::Float64;
        rhs_view.values = options.single_fp32
                              ? static_cast<void*>(d_rhs_f.get())
                              : static_cast<void*>(d_rhs.get());

        cls::DenseVectorView solution_view;
        solution_view.size = rhs.size();
        solution_view.location = cls::DataLocation::Device;
        solution_view.value_type =
            options.single_fp32 ? cls::ValueType::Float32 : cls::ValueType::Float64;
        solution_view.values = options.single_fp32
                                   ? static_cast<void*>(d_solution_f.get())
                                   : static_cast<void*>(d_solution.get());

        require_success(solver.set_data(matrix_view), "set_data");
        require_success(solver.set_rhs(rhs_view), "set_rhs");
        require_success(solver.set_solution(solution_view), "set_solution");
        const auto t_set = std::chrono::steady_clock::now();

        require_success(solver.analyze(), "analyze");
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after analyze");
        const auto t_analyze = std::chrono::steady_clock::now();

        std::vector<double> factor_ms;
        std::vector<double> solve_ms;
        factor_ms.reserve(static_cast<std::size_t>(options.repeat));
        solve_ms.reserve(static_cast<std::size_t>(options.repeat));
        if (options.batch_only) { factor_ms.push_back(0); solve_ms.push_back(0); }

        for (int r = 0; r < options.repeat && !options.batch_only; ++r) {
            const auto start = std::chrono::steady_clock::now();
            require_success(solver.factorize(), "factorize");
            const auto stop = std::chrono::steady_clock::now();
            factor_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        }

        for (int r = 0; r < options.repeat && !options.batch_only; ++r) {
            const auto start = std::chrono::steady_clock::now();
            require_success(solver.solve(), "solve");
            const auto stop = std::chrono::steady_clock::now();
            solve_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        }
        const auto t_solve = std::chrono::steady_clock::now();

        std::vector<double> solution;
        if (options.single_fp32) {
            const std::vector<float> solution_f = d_solution_f.download();
            solution.assign(solution_f.begin(), solution_f.end());
        } else {
            solution = d_solution.download();
        }
        const ResidualStats stats = residual(matrix, rhs.values, solution);
        if (options.write_solution) {
            scripts::write_matrix_market_vector(options.solution_out, solution);
        }
        const auto t_done = std::chrono::steady_clock::now();

        // ---- Uniform-batch experiment (B systems, same pattern) -------------------------
        double batch_factor_per_ms = 0, batch_solve_per_ms = 0, batch_relres = 0;
        if (options.batch > 0) {
            const int B = options.batch;
            const int n = matrix.rows, nnz = matrix.nnz();
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

            // Register the batched buffers (size = B * nnz / B * n) and run the phase API.
            cls::CsrMatrixView batched_matrix = matrix_view;
            batched_matrix.values = d_valB.get();
            cls::DenseVectorView batched_rhs;
            batched_rhs.size = (std::size_t)B * n;
            batched_rhs.location = cls::DataLocation::Device;
            batched_rhs.value_type = cls::ValueType::Float64;
            batched_rhs.values = d_rhsB.get();
            cls::DenseVectorView batched_sol = batched_rhs;
            batched_sol.values = d_solB.get();
            require_success(solver.set_data(batched_matrix), "set_data(batched)");
            require_success(solver.set_rhs(batched_rhs), "set_rhs(batched)");
            require_success(solver.set_solution(batched_sol), "set_solution(batched)");

            cudaDeviceSynchronize();
            const auto t_setup0 = std::chrono::steady_clock::now();
            require_success(solver.analyze(), "analyze(batched)");
            require_success(solver.setup(B), "setup(batched)");
            cudaDeviceSynchronize();
            const double setup_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_setup0).count();
            std::cout << "setup_ms=" << setup_ms << " (one-time, outside Newton loop)\n";

            std::vector<double> bf, bs;
            for (int r = 0; r < options.repeat; ++r) {
                const auto sf = std::chrono::steady_clock::now();
                require_success(solver.factorize(), "factorize(batched)");
                cudaDeviceSynchronize();
                const auto ef = std::chrono::steady_clock::now();
                require_success(solver.solve(), "solve(batched)");
                cudaDeviceSynchronize();
                const auto es = std::chrono::steady_clock::now();
                bf.push_back(std::chrono::duration<double, std::milli>(ef - sf).count());
                bs.push_back(std::chrono::duration<double, std::milli>(es - ef).count());
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
        if (options.batch > 0) {
            std::cout << "batch=" << options.batch << '\n'
                      << "batch_factor_per_sys_ms=" << batch_factor_per_ms << '\n'
                      << "batch_solve_per_sys_ms=" << batch_solve_per_ms << '\n'
                      << "batch_relres=" << batch_relres << '\n';
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
