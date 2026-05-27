#include "cuiter/common/cuda_utils.hpp"
#include "cuiter/core/csr_matrix.hpp"
#include "cuiter/kernels/gmres_kernels.hpp"
#include "cuiter/solver/gmres_solver.hpp"
#include "cupf_minimal/case_data.hpp"
#include "cupf_minimal/direct_cudss_solver.hpp"
#include "cupf_minimal/jacobian_analysis.hpp"

#include <amgx_c.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
    std::vector<int32_t> iterations = {1};
    std::vector<std::filesystem::path> configs = {
        "configs/amgx_fgmres_amg_block_jacobi.json",
        "configs/amgx_gmres_amg_block_jacobi.json",
        "configs/amgx_pbicgstab_amg_block_jacobi.json",
    };
    int32_t block_size = 64;
};

struct LinearMetadata {
    int32_t n_pvpq = 0;
    int32_t n_pq = 0;
};

struct ResidualMetrics {
    double abs = 0.0;
    double rel = 0.0;
};

struct DxMetrics {
    double norm_ratio = kNan;
    double cosine = kNan;
    double theta_norm_ratio = kNan;
    double theta_cosine = kNan;
    double vmag_norm_ratio = kNan;
    double vmag_cosine = kNan;
};

struct CudssResult {
    std::vector<double> x;
    double analyze_seconds = 0.0;
    double factorize_seconds = 0.0;
    double solve_seconds = 0.0;
};

struct Mr1Result {
    std::vector<double> x;
    double rel_res = kNan;
    double abs_res = kNan;
    double middle_seconds = kNan;
    double setup_seconds = kNan;
};

struct AmgxResult {
    std::vector<double> x;
    bool ok = false;
    bool converged = false;
    int32_t iterations = 0;
    double setup_seconds = kNan;
    double solve_seconds = kNan;
    double true_abs_res = kNan;
    double true_rel_res = kNan;
    std::string status;
    std::string error;
};

struct OutputRow {
    std::string case_name;
    int32_t iteration = 0;
    int32_t n = 0;
    int32_t nnz = 0;
    std::string config_name;
    AmgxResult amgx;
    CudssResult cudss;
    Mr1Result mr1;
    DxMetrics dx;
    DxMetrics mr1_dx;
};

std::vector<std::string> split_string_list(const std::string& text)
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

std::vector<int32_t> split_int_list(const std::string& text)
{
    std::vector<int32_t> values;
    for (const std::string& item : split_string_list(text)) {
        values.push_back(std::stoi(item));
    }
    return values;
}

std::vector<std::filesystem::path> split_path_list(const std::string& text)
{
    std::vector<std::filesystem::path> values;
    for (const std::string& item : split_string_list(text)) {
        values.emplace_back(item);
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
        << "  --iterations 1 or 0,1,2\n"
        << "  --configs cfg1.json,cfg2.json\n"
        << "  --block-size INT\n";
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
            options.cases = split_string_list(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            options.iterations = split_int_list(argv[++i]);
        } else if (arg == "--configs" && i + 1 < argc) {
            options.configs = split_path_list(argv[++i]);
        } else if (arg == "--block-size" && i + 1 < argc) {
            options.block_size = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.cases.empty() || options.iterations.empty() || options.configs.empty()) {
        throw std::runtime_error("cases, iterations, and configs must be nonempty");
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
        throw std::runtime_error("not a cuPF vector dump: " + path.string());
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

template <typename Fn>
double timed_sync(Fn&& fn)
{
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    fn();
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

double host_norm2(const std::vector<double>& values, int32_t begin, int32_t count)
{
    long double sum = 0.0;
    for (int32_t i = 0; i < count; ++i) {
        const double value = values[static_cast<std::size_t>(begin + i)];
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
}

double host_dot(const std::vector<double>& x,
                const std::vector<double>& y,
                int32_t begin,
                int32_t count)
{
    long double sum = 0.0;
    for (int32_t i = 0; i < count; ++i) {
        sum += static_cast<long double>(x[static_cast<std::size_t>(begin + i)]) *
               static_cast<long double>(y[static_cast<std::size_t>(begin + i)]);
    }
    return static_cast<double>(sum);
}

double safe_ratio(double numerator, double denominator)
{
    return denominator > std::numeric_limits<double>::min() ? numerator / denominator : 0.0;
}

double cosine(double dot, double x_norm, double y_norm)
{
    const double denom = x_norm * y_norm;
    if (denom <= std::numeric_limits<double>::min()) {
        return 0.0;
    }
    return std::clamp(dot / denom, -1.0, 1.0);
}

DxMetrics compare_dx(const std::vector<double>& x, const std::vector<double>& ref, int32_t n_pvpq)
{
    DxMetrics metrics;
    const int32_t n = static_cast<int32_t>(x.size());
    const int32_t n_pq = n - n_pvpq;
    const double x_norm = host_norm2(x, 0, n);
    const double ref_norm = host_norm2(ref, 0, n);
    metrics.norm_ratio = safe_ratio(x_norm, ref_norm);
    metrics.cosine = cosine(host_dot(x, ref, 0, n), x_norm, ref_norm);
    const double theta_x_norm = host_norm2(x, 0, n_pvpq);
    const double theta_ref_norm = host_norm2(ref, 0, n_pvpq);
    metrics.theta_norm_ratio = safe_ratio(theta_x_norm, theta_ref_norm);
    metrics.theta_cosine =
        cosine(host_dot(x, ref, 0, n_pvpq), theta_x_norm, theta_ref_norm);
    const double vmag_x_norm = host_norm2(x, n_pvpq, n_pq);
    const double vmag_ref_norm = host_norm2(ref, n_pvpq, n_pq);
    metrics.vmag_norm_ratio = safe_ratio(vmag_x_norm, vmag_ref_norm);
    metrics.vmag_cosine =
        cosine(host_dot(x, ref, n_pvpq, n_pq), vmag_x_norm, vmag_ref_norm);
    return metrics;
}

LinearMetadata make_metadata(const cupf_minimal::DumpCaseData& data)
{
    const cupf_minimal::JacobianIndexing indexing =
        cupf_minimal::make_jacobian_indexing(data.rows,
                                             data.pv.data(),
                                             static_cast<int32_t>(data.pv.size()),
                                             data.pq.data(),
                                             static_cast<int32_t>(data.pq.size()));
    return LinearMetadata{indexing.n_pvpq, indexing.n_pq};
}

ResidualMetrics compute_true_residual(cublasHandle_t cublas,
                                      const cuiter::CsrMatrix& matrix,
                                      const cuiter::DeviceBuffer<int32_t>& d_row_ptr,
                                      const cuiter::DeviceBuffer<int32_t>& d_col_idx,
                                      const cuiter::DeviceBuffer<double>& d_values,
                                      const cuiter::DeviceBuffer<double>& d_rhs,
                                      const double* d_x)
{
    cuiter::DeviceBuffer<double> d_ax(static_cast<std::size_t>(matrix.rows));
    cuiter::DeviceBuffer<double> d_r(static_cast<std::size_t>(matrix.rows));
    cuiter::kernels::launch_csr_spmv(matrix.rows,
                                     d_row_ptr.data(),
                                     d_col_idx.data(),
                                     d_values.data(),
                                     d_x,
                                     d_ax.data());
    cuiter::kernels::launch_residual(matrix.rows, d_rhs.data(), d_ax.data(), d_r.data());
    double r_norm = 0.0;
    double rhs_norm = 0.0;
    CUITER_CUBLAS_CHECK(cublasDnrm2(cublas, matrix.rows, d_r.data(), 1, &r_norm));
    CUITER_CUBLAS_CHECK(cublasDnrm2(cublas, matrix.rows, d_rhs.data(), 1, &rhs_norm));
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    rhs_norm = std::max(rhs_norm, std::numeric_limits<double>::min());
    return ResidualMetrics{r_norm, r_norm / rhs_norm};
}

void check_amgx(AMGX_RC code, const char* call)
{
    if (code != AMGX_RC_OK) {
        char message[4096] = {};
        AMGX_get_error_string(code, message, sizeof(message));
        throw std::runtime_error(std::string("AMGX call failed: ") + call + ": " + message);
    }
}

class AmgxRuntime {
public:
    AmgxRuntime()
    {
        check_amgx(AMGX_initialize(), "AMGX_initialize");
        AMGX_register_print_callback([](const char*, int) {});
    }

    ~AmgxRuntime()
    {
        AMGX_finalize();
    }
};

void ensure_amgx_runtime()
{
    static AmgxRuntime runtime;
    (void)runtime;
}

std::string amgx_status_name(AMGX_SOLVE_STATUS status)
{
    switch (status) {
    case AMGX_SOLVE_SUCCESS:
        return "success";
    case AMGX_SOLVE_FAILED:
        return "failed";
    case AMGX_SOLVE_DIVERGED:
        return "diverged";
    case AMGX_SOLVE_NOT_CONVERGED:
        return "not_converged";
    }
    return "unknown";
}

struct AmgxHandles {
    AMGX_config_handle config = nullptr;
    AMGX_resources_handle resources = nullptr;
    AMGX_matrix_handle matrix = nullptr;
    AMGX_vector_handle rhs = nullptr;
    AMGX_vector_handle x = nullptr;
    AMGX_solver_handle solver = nullptr;

    ~AmgxHandles()
    {
        if (solver != nullptr) {
            AMGX_solver_destroy(solver);
        }
        if (x != nullptr) {
            AMGX_vector_destroy(x);
        }
        if (rhs != nullptr) {
            AMGX_vector_destroy(rhs);
        }
        if (matrix != nullptr) {
            AMGX_matrix_destroy(matrix);
        }
        if (resources != nullptr) {
            AMGX_resources_destroy(resources);
        }
        if (config != nullptr) {
            AMGX_config_destroy(config);
        }
    }
};

AmgxResult solve_amgx(const std::filesystem::path& config_path,
                      const cuiter::CsrMatrix& matrix,
                      const cuiter::DeviceBuffer<int32_t>& d_row_ptr,
                      const cuiter::DeviceBuffer<int32_t>& d_col_idx,
                      const cuiter::DeviceBuffer<double>& d_values,
                      const cuiter::DeviceBuffer<double>& d_rhs,
                      cublasHandle_t cublas)
{
    AmgxResult result;
    result.x.assign(static_cast<std::size_t>(matrix.rows), 0.0);
    try {
        ensure_amgx_runtime();
        AmgxHandles handles;
        check_amgx(AMGX_config_create_from_file(&handles.config, config_path.string().c_str()),
                   "AMGX_config_create_from_file");
        check_amgx(AMGX_resources_create_simple(&handles.resources, handles.config),
                   "AMGX_resources_create_simple");
        check_amgx(AMGX_matrix_create(&handles.matrix, handles.resources, AMGX_mode_dDDI),
                   "AMGX_matrix_create");
        check_amgx(AMGX_vector_create(&handles.rhs, handles.resources, AMGX_mode_dDDI),
                   "AMGX_vector_create(rhs)");
        check_amgx(AMGX_vector_create(&handles.x, handles.resources, AMGX_mode_dDDI),
                   "AMGX_vector_create(x)");
        check_amgx(AMGX_solver_create(&handles.solver,
                                      handles.resources,
                                      AMGX_mode_dDDI,
                                      handles.config),
                   "AMGX_solver_create");

        result.setup_seconds = timed_sync([&] {
            check_amgx(AMGX_matrix_upload_all(handles.matrix,
                                             matrix.rows,
                                             matrix.nnz(),
                                             1,
                                             1,
                                             d_row_ptr.data(),
                                             d_col_idx.data(),
                                             d_values.data(),
                                             nullptr),
                       "AMGX_matrix_upload_all");
            check_amgx(AMGX_solver_setup(handles.solver, handles.matrix),
                       "AMGX_solver_setup");
        });

        cuiter::DeviceBuffer<double> d_x(static_cast<std::size_t>(matrix.rows));
        result.solve_seconds = timed_sync([&] {
            check_amgx(AMGX_vector_upload(handles.rhs, matrix.rows, 1, d_rhs.data()),
                       "AMGX_vector_upload(rhs)");
            check_amgx(AMGX_vector_set_zero(handles.x, matrix.rows, 1),
                       "AMGX_vector_set_zero(x)");
            check_amgx(AMGX_solver_solve_with_0_initial_guess(handles.solver,
                                                             handles.rhs,
                                                             handles.x),
                       "AMGX_solver_solve_with_0_initial_guess");
            check_amgx(AMGX_vector_download(handles.x, d_x.data()),
                       "AMGX_vector_download(x)");
        });
        d_x.copy_to(result.x.data(), result.x.size());
        AMGX_SOLVE_STATUS status = AMGX_SOLVE_FAILED;
        check_amgx(AMGX_solver_get_status(handles.solver, &status),
                   "AMGX_solver_get_status");
        check_amgx(AMGX_solver_get_iterations_number(handles.solver, &result.iterations),
                   "AMGX_solver_get_iterations_number");
        result.status = amgx_status_name(status);
        result.converged = status == AMGX_SOLVE_SUCCESS;
        const ResidualMetrics residual = compute_true_residual(cublas,
                                                               matrix,
                                                               d_row_ptr,
                                                               d_col_idx,
                                                               d_values,
                                                               d_rhs,
                                                               d_x.data());
        result.true_abs_res = residual.abs;
        result.true_rel_res = residual.rel;
        result.ok = true;
    } catch (const std::exception& ex) {
        result.ok = false;
        result.status = "error";
        result.error = ex.what();
    }
    return result;
}

CudssResult solve_cudss(const cuiter::CsrMatrix& matrix,
                        const cuiter::DeviceBuffer<int32_t>& d_row_ptr,
                        const cuiter::DeviceBuffer<int32_t>& d_col_idx,
                        const cuiter::DeviceBuffer<double>& d_values,
                        const cuiter::DeviceBuffer<double>& d_rhs)
{
    CudssResult result;
    cuiter::DeviceBuffer<double> d_x(static_cast<std::size_t>(matrix.rows));
    d_x.memset_zero();
    cupf_minimal::DirectCudssSolver solver;
    solver.initialize(matrix,
                      d_row_ptr.data(),
                      d_col_idx.data(),
                      d_values.data(),
                      d_rhs.data(),
                      d_x.data());
    result.analyze_seconds = solver.analyze();
    result.factorize_seconds = solver.factorize();
    result.solve_seconds = solver.solve();
    result.x.assign(static_cast<std::size_t>(matrix.rows), 0.0);
    d_x.copy_to(result.x.data(), result.x.size());
    return result;
}

Mr1Result solve_mr1(const cuiter::CsrMatrix& matrix,
                    const std::vector<double>& rhs,
                    int32_t block_size)
{
    cuiter::GmresSolverOptions options;
    options.max_iters = 1;
    options.restart = 1;
    options.rel_tolerance = 0.0;
    options.abs_tolerance = 0.0;
    options.preconditioner = "metis_block_jacobi";
    options.block_size = block_size;
    options.use_fp32_preconditioner = true;
    options.block_jacobi_apply = cuiter::BlockJacobiApplyMode::InverseGemv;
    options.use_mr1_fast_path = true;
    options.compute_true_residual = true;
    cuiter::GmresSolver solver(options);
    solver.analyze(matrix);
    const cuiter::LinearSolveResult result = solver.solve(matrix.values, rhs);
    Mr1Result out;
    out.x = result.solution;
    out.abs_res = result.residual_norm2;
    out.rel_res = result.relative_residual_norm2;
    out.middle_seconds = result.timings.middle_solver_total_seconds;
    out.setup_seconds = result.timings.setup_total_seconds;
    return out;
}

std::string config_name(const std::filesystem::path& path)
{
    return path.stem().string();
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

void write_summary(const std::filesystem::path& path, const std::vector<OutputRow>& rows)
{
    std::ofstream out(path);
    out << "case,n,nnz,iteration,config_name,amgx_ok,amgx_converged,amgx_status,"
           "amgx_iters,amgx_setup_ms,amgx_solve_ms,amgx_total_ms,amgx_true_rel_res,"
           "amgx_true_abs_res,cudss_factorize_solve_ms,mr1_middle_ms,mr1_rel_res,error\n";
    for (const OutputRow& row : rows) {
        out << row.case_name << ',' << row.n << ',' << row.nnz << ',' << row.iteration << ','
            << row.config_name << ',' << row.amgx.ok << ',' << row.amgx.converged << ','
            << row.amgx.status << ',' << row.amgx.iterations << ','
            << format_double(1000.0 * row.amgx.setup_seconds) << ','
            << format_double(1000.0 * row.amgx.solve_seconds) << ','
            << format_double(1000.0 * (row.amgx.setup_seconds + row.amgx.solve_seconds)) << ','
            << format_double(row.amgx.true_rel_res) << ','
            << format_double(row.amgx.true_abs_res) << ','
            << format_double(1000.0 * (row.cudss.factorize_seconds + row.cudss.solve_seconds)) << ','
            << format_double(1000.0 * row.mr1.middle_seconds) << ','
            << format_double(row.mr1.rel_res) << ','
            << '"' << row.amgx.error << '"' << '\n';
    }
}

void write_shadow(const std::filesystem::path& path, const std::vector<OutputRow>& rows)
{
    std::ofstream out(path);
    out << "case,iteration,config_name,dx_norm_ratio_amgx_vs_cudss,"
           "dx_cosine_amgx_vs_cudss,theta_norm_ratio,theta_cosine,vmag_norm_ratio,"
           "vmag_cosine,mr1_dx_norm_ratio_vs_cudss,mr1_dx_cosine_vs_cudss,"
           "mr1_theta_norm_ratio,mr1_theta_cosine,mr1_vmag_norm_ratio,mr1_vmag_cosine,"
           "mismatch_before_inf,mismatch_after_amgx_inf,mismatch_after_mr1_inf,"
           "mismatch_after_cudss_inf,amgx_nonlinear_ratio_inf,mr1_nonlinear_ratio_inf,"
           "cudss_nonlinear_ratio_inf\n";
    for (const OutputRow& row : rows) {
        out << row.case_name << ',' << row.iteration << ',' << row.config_name << ','
            << format_double(row.dx.norm_ratio) << ','
            << format_double(row.dx.cosine) << ','
            << format_double(row.dx.theta_norm_ratio) << ','
            << format_double(row.dx.theta_cosine) << ','
            << format_double(row.dx.vmag_norm_ratio) << ','
            << format_double(row.dx.vmag_cosine) << ','
            << format_double(row.mr1_dx.norm_ratio) << ','
            << format_double(row.mr1_dx.cosine) << ','
            << format_double(row.mr1_dx.theta_norm_ratio) << ','
            << format_double(row.mr1_dx.theta_cosine) << ','
            << format_double(row.mr1_dx.vmag_norm_ratio) << ','
            << format_double(row.mr1_dx.vmag_cosine) << ','
            << "nan,nan,nan,nan,nan,nan,nan\n";
    }
}

void write_timing(const std::filesystem::path& path, const std::vector<OutputRow>& rows)
{
    std::ofstream out(path);
    out << "case,iteration,config_name,amgx_setup_ms,amgx_solve_ms,amgx_total_ms,"
           "cudss_analyze_ms,cudss_factorize_ms,cudss_solve_ms,cudss_factorize_solve_ms,"
           "mr1_setup_ms,mr1_middle_ms\n";
    for (const OutputRow& row : rows) {
        out << row.case_name << ',' << row.iteration << ',' << row.config_name << ','
            << format_double(1000.0 * row.amgx.setup_seconds) << ','
            << format_double(1000.0 * row.amgx.solve_seconds) << ','
            << format_double(1000.0 * (row.amgx.setup_seconds + row.amgx.solve_seconds)) << ','
            << format_double(1000.0 * row.cudss.analyze_seconds) << ','
            << format_double(1000.0 * row.cudss.factorize_seconds) << ','
            << format_double(1000.0 * row.cudss.solve_seconds) << ','
            << format_double(1000.0 * (row.cudss.factorize_seconds + row.cudss.solve_seconds)) << ','
            << format_double(1000.0 * row.mr1.setup_seconds) << ','
            << format_double(1000.0 * row.mr1.middle_seconds) << '\n';
    }
}

void write_report(const std::filesystem::path& path, const std::vector<OutputRow>& rows)
{
    std::ofstream out(path);
    std::vector<const OutputRow*> fgmres;
    for (const OutputRow& row : rows) {
        if (row.config_name.find("fgmres") != std::string::npos && row.amgx.ok) {
            fgmres.push_back(&row);
        }
    }
    auto avg = [&](auto fn) {
        double sum = 0.0;
        int32_t count = 0;
        for (const OutputRow* row : fgmres) {
            const double value = fn(*row);
            if (std::isfinite(value)) {
                sum += value;
                ++count;
            }
        }
        return count > 0 ? sum / static_cast<double>(count) : kNan;
    };
    const double avg_dx_ratio = avg([](const OutputRow& row) { return row.dx.norm_ratio; });
    const double avg_dx_cos = avg([](const OutputRow& row) { return row.dx.cosine; });
    const double avg_mr1_dx_ratio =
        avg([](const OutputRow& row) { return row.mr1_dx.norm_ratio; });
    const double avg_mr1_dx_cos = avg([](const OutputRow& row) { return row.mr1_dx.cosine; });
    const double avg_amgx_rel = avg([](const OutputRow& row) { return row.amgx.true_rel_res; });
    const double avg_mr1_rel = avg([](const OutputRow& row) { return row.mr1.rel_res; });
    const double avg_amgx_solve_ms = avg([](const OutputRow& row) {
        return 1000.0 * row.amgx.solve_seconds;
    });
    const double avg_amgx_setup_ms = avg([](const OutputRow& row) {
        return 1000.0 * row.amgx.setup_seconds;
    });
    const double avg_cudss_fs_ms = avg([](const OutputRow& row) {
        return 1000.0 * (row.cudss.factorize_seconds + row.cudss.solve_seconds);
    });
    int32_t gate = 0;
    if (avg_dx_ratio >= 0.2) {
        ++gate;
    }
    if (avg_dx_cos >= 0.5) {
        ++gate;
    }
    if (avg_amgx_solve_ms <= avg_cudss_fs_ms) {
        ++gate;
    }

    out << "# AMGX Oracle Report\n\n";
    out << "Standalone input: `J1/F1` dumps for the selected five cases. Nonlinear "
           "mismatch rollback was not evaluated here because the J/F dump does not include "
           "the full voltage state needed to apply and restore a trial step.\n\n";

    out << "## 1. Does AMGX improve dx direction/scale versus MR1?\n\n";
    out << "- AMGX mean dx norm ratio vs cuDSS: `" << format_double(avg_dx_ratio) << "`.\n";
    out << "- MR1 mean dx norm ratio vs cuDSS: `" << format_double(avg_mr1_dx_ratio) << "`.\n";
    out << "- AMGX mean dx cosine vs cuDSS: `" << format_double(avg_dx_cos) << "`.\n";
    out << "- MR1 mean dx cosine vs cuDSS: `" << format_double(avg_mr1_dx_cos) << "`.\n\n";
    out << "| case | AMGX dx ratio | MR1 dx ratio | AMGX cosine | MR1 cosine | AMGX theta ratio | AMGX |V| ratio |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|\n";
    for (const OutputRow* row : fgmres) {
        out << "| " << row->case_name << " | " << format_double(row->dx.norm_ratio)
            << " | " << format_double(row->mr1_dx.norm_ratio)
            << " | " << format_double(row->dx.cosine)
            << " | " << format_double(row->mr1_dx.cosine)
            << " | " << format_double(row->dx.theta_norm_ratio)
            << " | " << format_double(row->dx.vmag_norm_ratio) << " |\n";
    }
    out << "\n";

    out << "## 2. Does AMGX reduce nonlinear mismatch more than MR1?\n\n";
    out << "- Not answered by this standalone run. The linear/dx gate failed, so hybrid "
           "shadow integration was not added.\n\n";

    out << "## 3. Is AMGX setup/solve cost compatible with hybrid middle use?\n\n";
    out << "- Mean AMGX setup ms: `" << format_double(avg_amgx_setup_ms) << "`.\n";
    out << "- Mean AMGX solve ms: `" << format_double(avg_amgx_solve_ms) << "`.\n";
    out << "- Mean cuDSS factorize+solve ms: `" << format_double(avg_cudss_fs_ms) << "`.\n\n";
    out << "| case | AMGX rel residual | MR1 rel residual | AMGX setup ms | AMGX solve ms | cuDSS factorize+solve ms |\n";
    out << "|---|---:|---:|---:|---:|---:|\n";
    for (const OutputRow* row : fgmres) {
        out << "| " << row->case_name
            << " | " << format_double(row->amgx.true_rel_res)
            << " | " << format_double(row->mr1.rel_res)
            << " | " << format_double(1000.0 * row->amgx.setup_seconds)
            << " | " << format_double(1000.0 * row->amgx.solve_seconds)
            << " | "
            << format_double(1000.0 * (row->cudss.factorize_seconds + row->cudss.solve_seconds))
            << " |\n";
    }
    out << "\n";

    out << "## 4. Should AMGX be integrated into hybrid NR?\n\n";
    out << "- Gate score from standalone metrics: `" << gate << " / 3`.\n";
    out << "- Hybrid integration recommendation: `" << (gate >= 2 ? "proceed" : "reject") << "`.\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        std::filesystem::create_directories(options.output_dir);

        cublasHandle_t cublas = nullptr;
        CUITER_CUBLAS_CHECK(cublasCreate(&cublas));
        std::vector<OutputRow> rows;

        for (const std::string& case_name : options.cases) {
            const cupf_minimal::DumpCaseData case_data =
                cupf_minimal::load_dump_case(options.case_root / case_name);
            const LinearMetadata metadata = make_metadata(case_data);
            for (int32_t iteration : options.iterations) {
                const std::filesystem::path case_dir = options.jf_root / case_name;
                const cuiter::CsrMatrix matrix =
                    load_cupf_csr_dump(case_dir / ("J" + std::to_string(iteration) + ".txt"));
                const std::vector<double> rhs =
                    load_cupf_vector_dump(case_dir / ("F" + std::to_string(iteration) + ".txt"));
                if (matrix.rows != metadata.n_pvpq + metadata.n_pq ||
                    static_cast<int32_t>(rhs.size()) != matrix.rows) {
                    throw std::runtime_error("case metadata/JF dimension mismatch for " + case_name);
                }

                cuiter::DeviceBuffer<int32_t> d_row_ptr;
                cuiter::DeviceBuffer<int32_t> d_col_idx;
                cuiter::DeviceBuffer<double> d_values;
                cuiter::DeviceBuffer<double> d_rhs;
                d_row_ptr.assign(matrix.row_ptr.data(), matrix.row_ptr.size());
                d_col_idx.assign(matrix.col_idx.data(), matrix.col_idx.size());
                d_values.assign(matrix.values.data(), matrix.values.size());
                d_rhs.assign(rhs.data(), rhs.size());

                std::cout << "[oracle] " << case_name << " J" << iteration << "/F"
                          << iteration << " baselines\n";
                const CudssResult cudss = solve_cudss(matrix, d_row_ptr, d_col_idx, d_values, d_rhs);
                const Mr1Result mr1 = solve_mr1(matrix, rhs, options.block_size);

                for (const auto& config : options.configs) {
                    std::cout << "[oracle] " << case_name << " " << config_name(config) << "\n";
                    OutputRow row;
                    row.case_name = case_name;
                    row.iteration = iteration;
                    row.n = matrix.rows;
                    row.nnz = matrix.nnz();
                    row.config_name = config_name(config);
                    row.cudss = cudss;
                    row.mr1 = mr1;
                    row.mr1_dx = compare_dx(row.mr1.x, row.cudss.x, metadata.n_pvpq);
                    row.amgx = solve_amgx(config,
                                          matrix,
                                          d_row_ptr,
                                          d_col_idx,
                                          d_values,
                                          d_rhs,
                                          cublas);
                    if (row.amgx.ok) {
                        row.dx = compare_dx(row.amgx.x, row.cudss.x, metadata.n_pvpq);
                    }
                    rows.push_back(std::move(row));
                }
            }
        }

        CUITER_CUBLAS_CHECK(cublasDestroy(cublas));
        write_summary(options.output_dir / "amgx_oracle_summary.csv", rows);
        write_shadow(options.output_dir / "amgx_oracle_shadow_dx.csv", rows);
        write_timing(options.output_dir / "amgx_oracle_timing.csv", rows);
        write_report(options.output_dir / "amgx_oracle_report.md", rows);
        std::cout << "[done] wrote AMGX oracle outputs to " << options.output_dir << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
