#include "jfnk_amgx_gpu.hpp"

#ifdef CUPF_WITH_CUDA

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/ops/jacobian/cuda_edge_fp64.hpp"

#include <Eigen/Dense>

#include <amgx_c.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace exp_20260414::newton_krylov {
namespace {

using Clock = std::chrono::steady_clock;

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                                  \
    do {                                                                                    \
        cublasStatus_t status__ = (call);                                                   \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                            \
            throw std::runtime_error(                                                       \
                std::string("cuBLAS call failed at ") + __FILE__ + ":" +                  \
                std::to_string(__LINE__) + " status=" +                                    \
                std::to_string(static_cast<int>(status__)));                                \
        }                                                                                   \
    } while (0)
#endif

constexpr int32_t kBlockSize = 256;

double elapsed_sec(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

void check_amgx(AMGX_RC code, const char* call)
{
    if (code != AMGX_RC_OK) {
        char message[4096] = {};
        AMGX_get_error_string(code, message, sizeof(message));
        throw std::runtime_error(std::string("AMGX call failed: ") + call + ": " + message);
    }
}

template <typename T>
void ensure_size(DeviceBuffer<T>& buffer, std::size_t count)
{
    if (buffer.size() != count) {
        buffer.resize(count);
    }
}

class CublasHandle {
public:
    CublasHandle()
    {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }

    ~CublasHandle()
    {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

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

void initialize_amgx_runtime()
{
    static AmgxRuntime runtime;
    (void)runtime;
}

void append_residual_trace(const JfnkOptions& options,
                           int32_t outer_iteration,
                           const char* phase,
                           int32_t inner_iteration,
                           double residual_abs,
                           double residual_rel,
                           double rhs_norm)
{
    if (options.residual_trace_path.empty()) {
        return;
    }
    std::ofstream trace(options.residual_trace_path, std::ios::app);
    trace << std::scientific << std::setprecision(17)
          << options.residual_trace_case << ','
          << outer_iteration << ','
          << phase << ','
          << inner_iteration << ','
          << residual_abs << ','
          << residual_rel << ','
          << rhs_norm << '\n';
}

void append_jacobian_error(const JfnkOptions& options,
                           CudaFp64Storage& storage,
                           IterationContext& ctx,
                           const DeviceBuffer<double>& fd_values)
{
    if (options.jacobian_error_path.empty()) {
        return;
    }
    if (fd_values.size() != storage.d_J_values.size()) {
        std::ofstream trace(options.jacobian_error_path, std::ios::app);
        trace << options.residual_trace_case << ','
              << ctx.iter << ','
              << storage.dimF << ','
              << "size_mismatch,"
              << fd_values.size() << ','
              << storage.d_J_values.size()
              << ",,,,,,,\n";
        return;
    }

    CudaJacobianOpEdgeFp64 analytic_jacobian(storage);
    analytic_jacobian.run(ctx);

    std::vector<double> fd(fd_values.size());
    std::vector<double> exact(storage.d_J_values.size());
    fd_values.copyTo(fd.data(), fd.size());
    storage.d_J_values.copyTo(exact.data(), exact.size());

    double fd_sq = 0.0;
    double exact_sq = 0.0;
    double diff_sq = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    std::size_t max_abs_index = 0;
    for (std::size_t i = 0; i < fd.size(); ++i) {
        const double diff = fd[i] - exact[i];
        const double abs_diff = std::abs(diff);
        const double denom = std::max(std::abs(exact[i]), 1e-30);
        const double rel = abs_diff / denom;
        fd_sq += fd[i] * fd[i];
        exact_sq += exact[i] * exact[i];
        diff_sq += diff * diff;
        if (abs_diff > max_abs) {
            max_abs = abs_diff;
            max_abs_index = i;
        }
        max_rel = std::max(max_rel, rel);
    }

    const double fd_norm = std::sqrt(fd_sq);
    const double exact_norm = std::sqrt(exact_sq);
    const double diff_norm = std::sqrt(diff_sq);
    const double rel_fro = diff_norm / std::max(exact_norm, 1e-30);

    std::ofstream trace(options.jacobian_error_path, std::ios::app);
    trace << std::scientific << std::setprecision(17)
          << options.residual_trace_case << ','
          << ctx.iter << ','
          << storage.dimF << ','
          << fd.size() << ','
          << fd_norm << ','
          << exact_norm << ','
          << diff_norm << ','
          << rel_fro << ','
          << max_abs << ','
          << max_rel << ','
          << max_abs_index << ','
          << fd[max_abs_index] << ','
          << exact[max_abs_index] << '\n';
}

const char* amgx_amg_config()
{
    return R"json(
{
  "config_version": 2,
  "solver": {
    "solver": "AMG",
    "algorithm": "AGGREGATION",
    "selector": "SIZE_2",
    "smoother": "BLOCK_JACOBI",
    "presweeps": 0,
    "postsweeps": 3,
    "cycle": "V",
    "coarse_solver": "DENSE_LU_SOLVER",
    "max_iters": 1,
    "max_levels": 100,
    "min_coarse_rows": 32,
    "relaxation_factor": 0.75,
    "tolerance": 0.0,
    "monitor_residual": 0,
    "print_grid_stats": 0,
    "print_solve_stats": 0,
    "obtain_timings": 0
  }
}
)json";
}

class AmgxDevicePreconditioner {
public:
    AmgxDevicePreconditioner() = default;

    ~AmgxDevicePreconditioner()
    {
        destroy();
    }

    AmgxDevicePreconditioner(const AmgxDevicePreconditioner&) = delete;
    AmgxDevicePreconditioner& operator=(const AmgxDevicePreconditioner&) = delete;

    bool compute(int32_t n,
                 int32_t nnz,
                 const int32_t* row_ptr_device,
                 const int32_t* col_idx_device,
                 const double* values_device)
    {
        destroy();
        initialize_amgx_runtime();

        n_ = n;
        nnz_ = nnz;

        check_amgx(AMGX_config_create(&config_, amgx_amg_config()), "AMGX_config_create");
        check_amgx(AMGX_resources_create_simple(&resources_, config_),
                   "AMGX_resources_create_simple");
        check_amgx(AMGX_matrix_create(&matrix_, resources_, AMGX_mode_dDDI),
                   "AMGX_matrix_create");
        check_amgx(AMGX_vector_create(&rhs_, resources_, AMGX_mode_dDDI),
                   "AMGX_vector_create(rhs)");
        check_amgx(AMGX_vector_create(&x_, resources_, AMGX_mode_dDDI),
                   "AMGX_vector_create(x)");
        check_amgx(AMGX_solver_create(&solver_, resources_, AMGX_mode_dDDI, config_),
                   "AMGX_solver_create");

        check_amgx(AMGX_matrix_upload_all(matrix_,
                                          n_,
                                          nnz_,
                                          1,
                                          1,
                                          row_ptr_device,
                                          col_idx_device,
                                          values_device,
                                          nullptr),
                   "AMGX_matrix_upload_all");
        check_amgx(AMGX_vector_set_zero(rhs_, n_, 1), "AMGX_vector_set_zero(rhs)");
        check_amgx(AMGX_vector_set_zero(x_, n_, 1), "AMGX_vector_set_zero(x)");
        check_amgx(AMGX_solver_setup(solver_, matrix_), "AMGX_solver_setup");
        return true;
    }

    void solve(const double* rhs_device, double* x_device, int32_t n) const
    {
        if (solver_ == nullptr || n != n_) {
            throw std::runtime_error("AMGX preconditioner is not ready");
        }
        check_amgx(AMGX_vector_upload(rhs_, n, 1, rhs_device), "AMGX_vector_upload(rhs)");
        check_amgx(AMGX_vector_set_zero(x_, n, 1), "AMGX_vector_set_zero(x)");
        check_amgx(AMGX_solver_solve_with_0_initial_guess(solver_, rhs_, x_),
                   "AMGX_solver_solve_with_0_initial_guess");
        check_amgx(AMGX_vector_download(x_, x_device), "AMGX_vector_download(x)");
    }

private:
    void destroy()
    {
        if (solver_ != nullptr) {
            AMGX_solver_destroy(solver_);
            solver_ = nullptr;
        }
        if (x_ != nullptr) {
            AMGX_vector_destroy(x_);
            x_ = nullptr;
        }
        if (rhs_ != nullptr) {
            AMGX_vector_destroy(rhs_);
            rhs_ = nullptr;
        }
        if (matrix_ != nullptr) {
            AMGX_matrix_destroy(matrix_);
            matrix_ = nullptr;
        }
        if (resources_ != nullptr) {
            AMGX_resources_destroy(resources_);
            resources_ = nullptr;
        }
        if (config_ != nullptr) {
            AMGX_config_destroy(config_);
            config_ = nullptr;
        }
        n_ = 0;
        nnz_ = 0;
    }

    int32_t n_ = 0;
    int32_t nnz_ = 0;
    AMGX_config_handle config_ = nullptr;
    AMGX_resources_handle resources_ = nullptr;
    AMGX_matrix_handle matrix_ = nullptr;
    AMGX_vector_handle rhs_ = nullptr;
    AMGX_vector_handle x_ = nullptr;
    AMGX_solver_handle solver_ = nullptr;
};

__global__ void negate_kernel(int32_t n, const double* input, double* output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = -input[i];
    }
}

__global__ void scale_copy_kernel(int32_t n, const double* input, double scale, double* output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = scale * input[i];
    }
}

__global__ void subtract_kernel(int32_t n, const double* lhs, const double* rhs, double* output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = lhs[i] - rhs[i];
    }
}

__global__ void add_vectors_kernel(int32_t n, const double* lhs, const double* rhs, double* output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = lhs[i] + rhs[i];
    }
}

__global__ void permute_old_to_new_kernel(int32_t n,
                                          const int32_t* old_to_new,
                                          const double* old_values,
                                          double* new_values)
{
    const int32_t old_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (old_i < n) {
        new_values[old_to_new[old_i]] = old_values[old_i];
    }
}

__global__ void unpermute_new_to_old_kernel(int32_t n,
                                            const int32_t* old_to_new,
                                            const double* new_values,
                                            double* old_values)
{
    const int32_t old_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (old_i < n) {
        old_values[old_i] = new_values[old_to_new[old_i]];
    }
}

__global__ void jv_finish_kernel(int32_t n,
                                 const double* perturbed_f,
                                 const double* base_f,
                                 double inv_eps,
                                 double* output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = (perturbed_f[i] - base_f[i]) * inv_eps;
    }
}

__global__ void reduce_absmax_kernel(int32_t n, const double* input, double* partial)
{
    __shared__ double values[kBlockSize];
    const int32_t tid = threadIdx.x;
    const int32_t i = blockIdx.x * blockDim.x + tid;

    values[tid] = (i < n) ? fabs(input[i]) : 0.0;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            values[tid] = fmax(values[tid], values[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = values[0];
    }
}

__global__ void decompose_voltage_kernel(const double* v_re,
                                         const double* v_im,
                                         double* va,
                                         double* vm,
                                         int32_t n_bus)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }
    const double re = v_re[bus];
    const double im = v_im[bus];
    va[bus] = atan2(im, re);
    vm[bus] = hypot(re, im);
}

__global__ void apply_direction_to_voltage_kernel(double* va,
                                                  double* vm,
                                                  const double* direction,
                                                  double eps,
                                                  const int32_t* pv,
                                                  const int32_t* pq,
                                                  int32_t n_pv,
                                                  int32_t n_pq)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t dim = n_pv + 2 * n_pq;
    if (tid >= dim) {
        return;
    }

    const double delta = eps * direction[tid];
    if (tid < n_pv) {
        va[pv[tid]] += delta;
    } else if (tid < n_pv + n_pq) {
        va[pq[tid - n_pv]] += delta;
    } else {
        vm[pq[tid - n_pv - n_pq]] += delta;
    }
}

__global__ void reconstruct_voltage_kernel(const double* va,
                                           const double* vm,
                                           double* v_re,
                                           double* v_im,
                                           int32_t n_bus)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }
    v_re[bus] = vm[bus] * cos(va[bus]);
    v_im[bus] = vm[bus] * sin(va[bus]);
}

__global__ void mismatch_pack_kernel(const double* y_re,
                                     const double* y_im,
                                     const int32_t* y_row_ptr,
                                     const int32_t* y_col,
                                     const double* v_re,
                                     const double* v_im,
                                     const double* sbus_re,
                                     const double* sbus_im,
                                     const int32_t* pv,
                                     const int32_t* pq,
                                     int32_t n_pv,
                                     int32_t n_pq,
                                     double* f)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t dim = n_pv + 2 * n_pq;
    if (tid >= dim) {
        return;
    }

    int32_t bus = 0;
    bool take_imag = false;
    if (tid < n_pv) {
        bus = pv[tid];
    } else if (tid < n_pv + n_pq) {
        bus = pq[tid - n_pv];
    } else {
        bus = pq[tid - n_pv - n_pq];
        take_imag = true;
    }

    double i_re = 0.0;
    double i_im = 0.0;
    for (int32_t k = y_row_ptr[bus]; k < y_row_ptr[bus + 1]; ++k) {
        const int32_t col = y_col[k];
        const double yr = y_re[k];
        const double yi = y_im[k];
        const double vr = v_re[col];
        const double vi = v_im[col];
        i_re += yr * vr - yi * vi;
        i_im += yr * vi + yi * vr;
    }

    const double vr = v_re[bus];
    const double vi = v_im[bus];
    const double mis_re = vr * i_re + vi * i_im - sbus_re[bus];
    const double mis_im = vi * i_re - vr * i_im - sbus_im[bus];
    f[tid] = take_imag ? mis_im : mis_re;
}

__global__ void set_seed_columns_kernel(int32_t count,
                                        const int32_t* columns,
                                        double* seed)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        seed[columns[i]] = 1.0;
    }
}

__global__ void scatter_fd_values_kernel(int32_t count,
                                         const int32_t* rows,
                                         const int32_t* positions,
                                         const double* jv,
                                         double* values)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        values[positions[i]] = jv[rows[i]];
    }
}

__global__ void add_linear_combination_kernel(int32_t n,
                                              int32_t k,
                                              const double* coeffs,
                                              const double* vectors,
                                              int32_t stride,
                                              double* x)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }

    double sum = 0.0;
    for (int32_t j = 0; j < k; ++j) {
        sum += coeffs[j] * vectors[static_cast<std::size_t>(j) * stride + row];
    }
    x[row] += sum;
}

__device__ double safe_inverse_device(double value)
{
    constexpr double kPivotTol = 1e-12;
    if (!isfinite(value) || fabs(value) <= kPivotTol) {
        return 1.0;
    }
    return 1.0 / value;
}

__global__ void setup_bus_block_jacobi_kernel(int32_t n_pv,
                                              int32_t n_pq,
                                              const int32_t* pv,
                                              const int32_t* pq,
                                              const int32_t* diag11,
                                              const int32_t* diag12,
                                              const int32_t* diag21,
                                              const int32_t* diag22,
                                              const double* values,
                                              double* pv_inv,
                                              double* pq_inv00,
                                              double* pq_inv01,
                                              double* pq_inv10,
                                              double* pq_inv11)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_pv) {
        const int32_t bus = pv[tid];
        const int32_t pos = diag11[bus];
        pv_inv[tid] = pos >= 0 ? safe_inverse_device(values[pos]) : 1.0;
    }
    if (tid < n_pq) {
        const int32_t bus = pq[tid];
        const int32_t p11 = diag11[bus];
        const int32_t p12 = diag12[bus];
        const int32_t p21 = diag21[bus];
        const int32_t p22 = diag22[bus];
        const double a = p11 >= 0 ? values[p11] : 1.0;
        const double b = p12 >= 0 ? values[p12] : 0.0;
        const double c = p21 >= 0 ? values[p21] : 0.0;
        const double d = p22 >= 0 ? values[p22] : 1.0;
        const double det = a * d - b * c;
        if (isfinite(det) && fabs(det) > 1e-12) {
            pq_inv00[tid] = d / det;
            pq_inv01[tid] = -b / det;
            pq_inv10[tid] = -c / det;
            pq_inv11[tid] = a / det;
        } else {
            pq_inv00[tid] = 1.0;
            pq_inv01[tid] = 0.0;
            pq_inv10[tid] = 0.0;
            pq_inv11[tid] = 1.0;
        }
    }
}

__global__ void apply_bus_block_jacobi_kernel(int32_t n_pv,
                                              int32_t n_pq,
                                              const double* input,
                                              const double* pv_inv,
                                              const double* pq_inv00,
                                              const double* pq_inv01,
                                              const double* pq_inv10,
                                              const double* pq_inv11,
                                              double* output)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_pv) {
        output[tid] = pv_inv[tid] * input[tid];
    }
    if (tid < n_pq) {
        const int32_t p = n_pv + tid;
        const int32_t q = n_pv + n_pq + tid;
        output[p] = pq_inv00[tid] * input[p] + pq_inv01[tid] * input[q];
        output[q] = pq_inv10[tid] * input[p] + pq_inv11[tid] * input[q];
    }
}

std::vector<std::vector<int32_t>> build_jacobian_pattern_by_column(
    int32_t n_bus,
    const std::vector<int32_t>& ybus_indptr,
    const std::vector<int32_t>& ybus_indices,
    const std::vector<int32_t>& pv,
    const std::vector<int32_t>& pq)
{
    const int32_t n_pv = static_cast<int32_t>(pv.size());
    const int32_t n_pq = static_cast<int32_t>(pq.size());
    const int32_t n_pvpq = n_pv + n_pq;
    const int32_t dim = n_pvpq + n_pq;

    std::vector<int32_t> pvpq_index(static_cast<std::size_t>(n_bus), -1);
    std::vector<int32_t> pq_index(static_cast<std::size_t>(n_bus), -1);
    for (int32_t i = 0; i < n_pv; ++i) {
        pvpq_index[static_cast<std::size_t>(pv[static_cast<std::size_t>(i)])] = i;
    }
    for (int32_t i = 0; i < n_pq; ++i) {
        const int32_t bus = pq[static_cast<std::size_t>(i)];
        pvpq_index[static_cast<std::size_t>(bus)] = n_pv + i;
        pq_index[static_cast<std::size_t>(bus)] = i;
    }

    std::vector<std::vector<int32_t>> col_rows(static_cast<std::size_t>(dim));
    auto add = [&](int32_t row, int32_t col) {
        if (row >= 0 && col >= 0) {
            col_rows[static_cast<std::size_t>(col)].push_back(row);
        }
    };

    auto add_bus_pair = [&](int32_t row_bus, int32_t col_bus) {
        const int32_t row_p = pvpq_index[static_cast<std::size_t>(row_bus)];
        const int32_t row_q_local = pq_index[static_cast<std::size_t>(row_bus)];
        const int32_t row_q = row_q_local >= 0 ? n_pvpq + row_q_local : -1;
        const int32_t col_theta = pvpq_index[static_cast<std::size_t>(col_bus)];
        const int32_t col_vm_local = pq_index[static_cast<std::size_t>(col_bus)];
        const int32_t col_vm = col_vm_local >= 0 ? n_pvpq + col_vm_local : -1;

        add(row_p, col_theta);
        add(row_p, col_vm);
        add(row_q, col_theta);
        add(row_q, col_vm);
    };

    for (int32_t row_bus = 0; row_bus < n_bus; ++row_bus) {
        for (int32_t k = ybus_indptr[static_cast<std::size_t>(row_bus)];
             k < ybus_indptr[static_cast<std::size_t>(row_bus + 1)];
             ++k) {
            add_bus_pair(row_bus, ybus_indices[static_cast<std::size_t>(k)]);
        }
    }
    for (int32_t bus = 0; bus < n_bus; ++bus) {
        add_bus_pair(bus, bus);
    }

    for (auto& rows : col_rows) {
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
    return col_rows;
}

std::vector<int32_t> color_jacobian_columns(
    int32_t dim,
    const std::vector<std::vector<int32_t>>& col_rows)
{
    std::vector<std::vector<int32_t>> row_cols(static_cast<std::size_t>(dim));
    for (int32_t col = 0; col < dim; ++col) {
        for (int32_t row : col_rows[static_cast<std::size_t>(col)]) {
            row_cols[static_cast<std::size_t>(row)].push_back(col);
        }
    }

    std::vector<int32_t> color(static_cast<std::size_t>(dim), -1);
    std::vector<int32_t> forbidden;
    int32_t num_colors = 0;

    for (int32_t col = 0; col < dim; ++col) {
        if (static_cast<int32_t>(forbidden.size()) < num_colors) {
            forbidden.resize(static_cast<std::size_t>(num_colors), -1);
        }

        for (int32_t row : col_rows[static_cast<std::size_t>(col)]) {
            for (int32_t other_col : row_cols[static_cast<std::size_t>(row)]) {
                const int32_t other_color = color[static_cast<std::size_t>(other_col)];
                if (other_color >= 0) {
                    forbidden[static_cast<std::size_t>(other_color)] = col;
                }
            }
        }

        int32_t chosen = 0;
        while (chosen < num_colors &&
               forbidden[static_cast<std::size_t>(chosen)] == col) {
            ++chosen;
        }
        if (chosen == num_colors) {
            ++num_colors;
            forbidden.push_back(-1);
        }
        color[static_cast<std::size_t>(col)] = chosen;
    }

    return color;
}

struct FdPattern {
    int32_t dim = 0;
    int32_t nnz = 0;
    int32_t num_colors = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<int32_t> color_ptr;
    std::vector<int32_t> color_cols;
    std::vector<int32_t> color_nnz_ptr;
    std::vector<int32_t> scatter_rows;
    std::vector<int32_t> scatter_positions;
};

struct PermutedFdMatrix {
    std::vector<int32_t> old_to_new;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<double> values;
};

FdPattern build_fd_pattern(int32_t n_bus,
                           const std::vector<int32_t>& ybus_indptr,
                           const std::vector<int32_t>& ybus_indices,
                           const std::vector<int32_t>& pv,
                           const std::vector<int32_t>& pq)
{
    const auto col_rows =
        build_jacobian_pattern_by_column(n_bus, ybus_indptr, ybus_indices, pv, pq);
    const int32_t dim = static_cast<int32_t>(col_rows.size());
    const std::vector<int32_t> colors = color_jacobian_columns(dim, col_rows);
    const int32_t num_colors =
        colors.empty() ? 0 : (*std::max_element(colors.begin(), colors.end()) + 1);

    std::vector<std::vector<int32_t>> row_cols(static_cast<std::size_t>(dim));
    for (int32_t col = 0; col < dim; ++col) {
        for (int32_t row : col_rows[static_cast<std::size_t>(col)]) {
            row_cols[static_cast<std::size_t>(row)].push_back(col);
        }
    }
    for (auto& cols : row_cols) {
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    }

    FdPattern pattern;
    pattern.dim = dim;
    pattern.row_ptr.assign(static_cast<std::size_t>(dim + 1), 0);
    for (int32_t row = 0; row < dim; ++row) {
        pattern.row_ptr[static_cast<std::size_t>(row + 1)] =
            pattern.row_ptr[static_cast<std::size_t>(row)] +
            static_cast<int32_t>(row_cols[static_cast<std::size_t>(row)].size());
    }
    pattern.nnz = pattern.row_ptr.back();
    pattern.col_idx.reserve(static_cast<std::size_t>(pattern.nnz));
    for (const auto& cols : row_cols) {
        pattern.col_idx.insert(pattern.col_idx.end(), cols.begin(), cols.end());
    }

    std::vector<std::vector<int32_t>> color_cols(static_cast<std::size_t>(num_colors));
    for (int32_t col = 0; col < dim; ++col) {
        color_cols[static_cast<std::size_t>(colors[static_cast<std::size_t>(col)])].push_back(col);
    }

    pattern.num_colors = num_colors;
    pattern.color_ptr.assign(static_cast<std::size_t>(num_colors + 1), 0);
    for (int32_t color = 0; color < num_colors; ++color) {
        pattern.color_ptr[static_cast<std::size_t>(color + 1)] =
            pattern.color_ptr[static_cast<std::size_t>(color)] +
            static_cast<int32_t>(color_cols[static_cast<std::size_t>(color)].size());
        pattern.color_cols.insert(pattern.color_cols.end(),
                                  color_cols[static_cast<std::size_t>(color)].begin(),
                                  color_cols[static_cast<std::size_t>(color)].end());
    }

    pattern.color_nnz_ptr.assign(static_cast<std::size_t>(num_colors + 1), 0);
    for (int32_t color = 0; color < num_colors; ++color) {
        for (int32_t col : color_cols[static_cast<std::size_t>(color)]) {
            for (int32_t row : col_rows[static_cast<std::size_t>(col)]) {
                const auto& cols = row_cols[static_cast<std::size_t>(row)];
                const auto it = std::lower_bound(cols.begin(), cols.end(), col);
                if (it == cols.end() || *it != col) {
                    throw std::runtime_error("FD pattern scatter position not found");
                }
                const int32_t local = static_cast<int32_t>(it - cols.begin());
                pattern.scatter_rows.push_back(row);
                pattern.scatter_positions.push_back(
                    pattern.row_ptr[static_cast<std::size_t>(row)] + local);
            }
        }
        pattern.color_nnz_ptr[static_cast<std::size_t>(color + 1)] =
            static_cast<int32_t>(pattern.scatter_rows.size());
    }

    return pattern;
}

std::vector<int32_t> build_old_to_new_permutation(const std::string& permutation,
                                                  int32_t n_bus,
                                                  const std::vector<int32_t>& pv,
                                                  const std::vector<int32_t>& pq)
{
    const int32_t n_pv = static_cast<int32_t>(pv.size());
    const int32_t n_pq = static_cast<int32_t>(pq.size());
    const int32_t dim = n_pv + 2 * n_pq;
    std::vector<int32_t> pvpq_index(static_cast<std::size_t>(n_bus), -1);
    std::vector<int32_t> pq_index(static_cast<std::size_t>(n_bus), -1);
    for (int32_t i = 0; i < n_pv; ++i) {
        pvpq_index[static_cast<std::size_t>(pv[static_cast<std::size_t>(i)])] = i;
    }
    for (int32_t i = 0; i < n_pq; ++i) {
        const int32_t bus = pq[static_cast<std::size_t>(i)];
        pvpq_index[static_cast<std::size_t>(bus)] = n_pv + i;
        pq_index[static_cast<std::size_t>(bus)] = i;
    }

    std::vector<int32_t> old_order;
    old_order.reserve(static_cast<std::size_t>(dim));
    if (permutation == "bus_local") {
        for (int32_t bus = 0; bus < n_bus; ++bus) {
            const int32_t theta = pvpq_index[static_cast<std::size_t>(bus)];
            const int32_t pq_local = pq_index[static_cast<std::size_t>(bus)];
            if (theta >= 0) {
                old_order.push_back(theta);
            }
            if (pq_local >= 0) {
                old_order.push_back(n_pv + n_pq + pq_local);
            }
        }
    } else if (permutation == "pq_interleaved") {
        for (int32_t i = 0; i < n_pv; ++i) {
            old_order.push_back(i);
        }
        for (int32_t i = 0; i < n_pq; ++i) {
            old_order.push_back(n_pv + i);
            old_order.push_back(n_pv + n_pq + i);
        }
    } else {
        for (int32_t i = 0; i < dim; ++i) {
            old_order.push_back(i);
        }
    }
    if (static_cast<int32_t>(old_order.size()) != dim) {
        throw std::runtime_error("invalid permutation size");
    }

    std::vector<int32_t> old_to_new(static_cast<std::size_t>(dim), -1);
    for (int32_t new_i = 0; new_i < dim; ++new_i) {
        const int32_t old_i = old_order[static_cast<std::size_t>(new_i)];
        if (old_i < 0 || old_i >= dim || old_to_new[static_cast<std::size_t>(old_i)] >= 0) {
            throw std::runtime_error("invalid permutation contents");
        }
        old_to_new[static_cast<std::size_t>(old_i)] = new_i;
    }
    return old_to_new;
}

PermutedFdMatrix permute_fd_matrix(const FdPattern& pattern,
                                   const std::vector<double>& values,
                                   const std::vector<int32_t>& old_to_new)
{
    if (values.size() != static_cast<std::size_t>(pattern.nnz)) {
        throw std::runtime_error("permutation value size mismatch");
    }
    std::vector<std::vector<std::pair<int32_t, double>>> rows(
        static_cast<std::size_t>(pattern.dim));
    for (int32_t old_row = 0; old_row < pattern.dim; ++old_row) {
        const int32_t new_row = old_to_new[static_cast<std::size_t>(old_row)];
        auto& row = rows[static_cast<std::size_t>(new_row)];
        for (int32_t p = pattern.row_ptr[static_cast<std::size_t>(old_row)];
             p < pattern.row_ptr[static_cast<std::size_t>(old_row + 1)];
             ++p) {
            const int32_t old_col = pattern.col_idx[static_cast<std::size_t>(p)];
            row.push_back({old_to_new[static_cast<std::size_t>(old_col)],
                           values[static_cast<std::size_t>(p)]});
        }
        std::sort(row.begin(),
                  row.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    }

    PermutedFdMatrix permuted;
    permuted.old_to_new = old_to_new;
    permuted.row_ptr.assign(static_cast<std::size_t>(pattern.dim + 1), 0);
    for (int32_t row = 0; row < pattern.dim; ++row) {
        permuted.row_ptr[static_cast<std::size_t>(row + 1)] =
            permuted.row_ptr[static_cast<std::size_t>(row)] +
            static_cast<int32_t>(rows[static_cast<std::size_t>(row)].size());
        for (const auto& entry : rows[static_cast<std::size_t>(row)]) {
            permuted.col_idx.push_back(entry.first);
            permuted.values.push_back(entry.second);
        }
    }
    return permuted;
}

struct DeviceFdPattern {
    DeviceBuffer<int32_t> row_ptr;
    DeviceBuffer<int32_t> col_idx;
    DeviceBuffer<int32_t> color_ptr;
    DeviceBuffer<int32_t> color_cols;
    DeviceBuffer<int32_t> color_nnz_ptr;
    DeviceBuffer<int32_t> scatter_rows;
    DeviceBuffer<int32_t> scatter_positions;
    int32_t dim = 0;
    int32_t nnz = 0;
    int32_t num_colors = 0;
};

DeviceFdPattern upload_fd_pattern(const FdPattern& host)
{
    DeviceFdPattern device;
    device.dim = host.dim;
    device.nnz = host.nnz;
    device.num_colors = host.num_colors;
    device.row_ptr.assign(host.row_ptr.data(), host.row_ptr.size());
    device.col_idx.assign(host.col_idx.data(), host.col_idx.size());
    device.color_ptr.assign(host.color_ptr.data(), host.color_ptr.size());
    device.color_cols.assign(host.color_cols.data(), host.color_cols.size());
    device.color_nnz_ptr.assign(host.color_nnz_ptr.data(), host.color_nnz_ptr.size());
    device.scatter_rows.assign(host.scatter_rows.data(), host.scatter_rows.size());
    device.scatter_positions.assign(host.scatter_positions.data(), host.scatter_positions.size());
    return device;
}

double device_absmax(int32_t n,
                     const double* data,
                     DeviceBuffer<double>& partial,
                     DeviceBuffer<double>& scratch)
{
    if (n <= 0) {
        return 0.0;
    }

    const int32_t grid = (n + kBlockSize - 1) / kBlockSize;
    ensure_size(partial, static_cast<std::size_t>(grid));
    reduce_absmax_kernel<<<grid, kBlockSize>>>(n, data, partial.data());
    CUDA_CHECK(cudaGetLastError());

    int32_t current = grid;
    bool in_partial = true;
    while (current > 1) {
        const int32_t next = (current + kBlockSize - 1) / kBlockSize;
        if (in_partial) {
            ensure_size(scratch, static_cast<std::size_t>(next));
            reduce_absmax_kernel<<<next, kBlockSize>>>(current, partial.data(), scratch.data());
        } else {
            ensure_size(partial, static_cast<std::size_t>(next));
            reduce_absmax_kernel<<<next, kBlockSize>>>(current, scratch.data(), partial.data());
        }
        CUDA_CHECK(cudaGetLastError());
        current = next;
        in_partial = !in_partial;
    }

    double result = 0.0;
    if (in_partial) {
        partial.copyTo(&result, 1);
    } else {
        scratch.copyTo(&result, 1);
    }
    return result;
}

void add_linear_combination(int32_t dim,
                            const Eigen::VectorXd& coeffs,
                            const double* vectors,
                            int32_t stride,
                            double* x,
                            DeviceBuffer<double>& d_coeffs)
{
    const int32_t k = static_cast<int32_t>(coeffs.size());
    if (k <= 0) {
        return;
    }
    d_coeffs.assign(coeffs.data(), static_cast<std::size_t>(k));
    const int32_t grid = (dim + kBlockSize - 1) / kBlockSize;
    add_linear_combination_kernel<<<grid, kBlockSize>>>(
        dim, k, d_coeffs.data(), vectors, stride, x);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

JfnkLinearSolveAmgx::JfnkLinearSolveAmgx(IStorage& storage, JfnkOptions options)
    : storage_(static_cast<CudaFp64Storage&>(storage))
    , options_(std::move(options))
{}

JfnkLinearSolveAmgx::~JfnkLinearSolveAmgx() = default;

void JfnkLinearSolveAmgx::analyze(const AnalyzeContext& ctx)
{
    n_bus_ = ctx.n_bus;
    ybus_indptr_.assign(ctx.ybus.indptr, ctx.ybus.indptr + ctx.ybus.rows + 1);
    ybus_indices_.assign(ctx.ybus.indices, ctx.ybus.indices + ctx.ybus.nnz);
    pv_.assign(ctx.pv, ctx.pv + ctx.n_pv);
    pq_.assign(ctx.pq, ctx.pq + ctx.n_pq);
    analyzed_ = true;
}

void JfnkLinearSolveAmgx::run(IterationContext& ctx)
{
    if (!analyzed_) {
        throw std::runtime_error("JfnkLinearSolveAmgx::run: analyze() must be called first");
    }
    if (storage_.dimF <= 0 || storage_.n_bus <= 0) {
        throw std::runtime_error("JfnkLinearSolveAmgx::run: storage is not prepared");
    }
    if (options_.solver != "fgmres" && options_.solver != "gmres_none") {
        throw std::runtime_error("JfnkLinearSolveAmgx::run: GPU path supports fgmres only");
    }
    if (options_.preconditioner != "amg_fd" &&
        options_.preconditioner != "bus_block_jacobi_fd" &&
        options_.preconditioner != "none") {
        throw std::runtime_error(
            "JfnkLinearSolveAmgx::run: GPU path supports amg_fd, bus_block_jacobi_fd, or none preconditioner only");
    }
    if (options_.linear_tolerance <= 0.0 ||
        options_.max_inner_iterations <= 0 ||
        options_.gmres_restart <= 0) {
        throw std::runtime_error("JfnkLinearSolveAmgx::run: invalid linear solver options");
    }
    if (options_.permutation != "none" &&
        options_.permutation != "bus_local" &&
        options_.permutation != "pq_interleaved") {
        throw std::runtime_error("JfnkLinearSolveAmgx::run: invalid AMGX permutation");
    }
    if (options_.permutation != "none" && options_.preconditioner != "amg_fd") {
        throw std::runtime_error("JfnkLinearSolveAmgx::run: AMGX permutation requires amg_fd");
    }
    if (options_.preconditioner_combine != "single" &&
        options_.preconditioner_combine != "additive" &&
        options_.preconditioner_combine != "block_then_amg" &&
        options_.preconditioner_combine != "amg_then_block") {
        throw std::runtime_error("JfnkLinearSolveAmgx::run: invalid preconditioner combine mode");
    }
    if (options_.preconditioner_combine != "single" &&
        options_.preconditioner != "bus_block_jacobi_fd") {
        throw std::runtime_error(
            "JfnkLinearSolveAmgx::run: combined preconditioner requires bus_block_jacobi_fd");
    }

    const auto solve_start = Clock::now();
    const int32_t dim = storage_.dimF;
    const int32_t n_bus = storage_.n_bus;
    const int32_t n_pv = storage_.n_pvpq - storage_.n_pq;
    const int32_t n_pq = storage_.n_pq;
    const int32_t vector_grid = (dim + kBlockSize - 1) / kBlockSize;
    const int32_t bus_grid = (n_bus + kBlockSize - 1) / kBlockSize;

    stats_.last_success = false;
    stats_.last_iterations = 0;
    stats_.last_jv_calls = 0;
    stats_.last_estimated_error = std::numeric_limits<double>::quiet_NaN();
    stats_.last_epsilon = 0.0;
    stats_.last_failure_reason.clear();

    CublasHandle blas;

    DeviceBuffer<double> d_base_f(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_rhs(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_x(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_r(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_w(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_ax(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_seed(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_fd_jv(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_scratch_f(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_scratch_v_re(static_cast<std::size_t>(n_bus));
    DeviceBuffer<double> d_scratch_v_im(static_cast<std::size_t>(n_bus));
    DeviceBuffer<double> d_scratch_va(static_cast<std::size_t>(n_bus));
    DeviceBuffer<double> d_scratch_vm(static_cast<std::size_t>(n_bus));
    DeviceBuffer<double> d_reduce;
    DeviceBuffer<double> d_reduce_scratch;
    DeviceBuffer<double> d_coeffs;

    const int32_t restart = options_.gmres_restart;
    DeviceBuffer<double> d_basis(static_cast<std::size_t>(dim) * (restart + 1));
    DeviceBuffer<double> d_z_basis(static_cast<std::size_t>(dim) * restart);

    CUBLAS_CHECK(cublasDcopy(blas.get(), dim, storage_.d_F.data(), 1, d_base_f.data(), 1));
    negate_kernel<<<vector_grid, kBlockSize>>>(dim, d_base_f.data(), d_rhs.data());
    CUDA_CHECK(cudaGetLastError());
    CUBLAS_CHECK(cublasDcopy(blas.get(), dim, d_rhs.data(), 1, d_r.data(), 1));
    d_x.memsetZero();

    auto apply_jv = [&](const double* direction, double* output) {
        const auto jv_start = Clock::now();
        const double v_norm =
            std::max(device_absmax(dim, direction, d_reduce, d_reduce_scratch),
                     std::numeric_limits<double>::min());
        double eps = options_.fixed_epsilon;
        if (options_.auto_epsilon) {
            eps = std::sqrt(std::numeric_limits<double>::epsilon()) * 2.0 / v_norm;
            eps = std::max(eps, 1e-12);
        }
        stats_.last_epsilon = eps;

        const auto update_start = Clock::now();
        decompose_voltage_kernel<<<bus_grid, kBlockSize>>>(
            storage_.d_V_re.data(),
            storage_.d_V_im.data(),
            d_scratch_va.data(),
            d_scratch_vm.data(),
            n_bus);
        CUDA_CHECK(cudaGetLastError());
        apply_direction_to_voltage_kernel<<<vector_grid, kBlockSize>>>(
            d_scratch_va.data(),
            d_scratch_vm.data(),
            direction,
            eps,
            storage_.d_pv.data(),
            storage_.d_pq.data(),
            n_pv,
            n_pq);
        CUDA_CHECK(cudaGetLastError());
        reconstruct_voltage_kernel<<<bus_grid, kBlockSize>>>(
            d_scratch_va.data(),
            d_scratch_vm.data(),
            d_scratch_v_re.data(),
            d_scratch_v_im.data(),
            n_bus);
        CUDA_CHECK(cudaGetLastError());
        const auto update_end = Clock::now();

        const auto mismatch_start = Clock::now();
        mismatch_pack_kernel<<<vector_grid, kBlockSize>>>(
            storage_.d_Ybus_re.data(),
            storage_.d_Ybus_im.data(),
            storage_.d_Ybus_indptr.data(),
            storage_.d_Ybus_indices.data(),
            d_scratch_v_re.data(),
            d_scratch_v_im.data(),
            storage_.d_Sbus_re.data(),
            storage_.d_Sbus_im.data(),
            storage_.d_pv.data(),
            storage_.d_pq.data(),
            n_pv,
            n_pq,
            d_scratch_f.data());
        CUDA_CHECK(cudaGetLastError());
        const auto mismatch_end = Clock::now();

        jv_finish_kernel<<<vector_grid, kBlockSize>>>(
            dim, d_scratch_f.data(), d_base_f.data(), 1.0 / eps, output);
        CUDA_CHECK(cudaGetLastError());

        const auto jv_end = Clock::now();
        ++stats_.last_jv_calls;
        ++stats_.total_jv_calls;
        stats_.total_jv_update_sec += elapsed_sec(update_start, update_end);
        stats_.total_jv_mismatch_sec += elapsed_sec(mismatch_start, mismatch_end);
        stats_.total_jv_sec += elapsed_sec(jv_start, jv_end);
    };

    AmgxDevicePreconditioner amg_preconditioner;
    DeviceBuffer<double> d_fd_values;
    DeviceBuffer<int32_t> d_perm_old_to_new;
    DeviceBuffer<int32_t> d_perm_row_ptr;
    DeviceBuffer<int32_t> d_perm_col_idx;
    DeviceBuffer<double> d_perm_values;
    DeviceBuffer<double> d_perm_rhs;
    DeviceBuffer<double> d_perm_x;
    DeviceBuffer<double> d_pv_inv;
    DeviceBuffer<double> d_pq_inv00;
    DeviceBuffer<double> d_pq_inv01;
    DeviceBuffer<double> d_pq_inv10;
    DeviceBuffer<double> d_pq_inv11;
    DeviceFdPattern device_pattern;

    const bool needs_fd = options_.preconditioner == "amg_fd" ||
                          options_.preconditioner == "bus_block_jacobi_fd";
    const bool needs_amg = options_.preconditioner == "amg_fd" ||
                           options_.preconditioner_combine != "single";
    const bool needs_block = options_.preconditioner == "bus_block_jacobi_fd";

    if (needs_fd) {
        const auto setup_start = Clock::now();
        bool setup_ok = false;

        try {
            const FdPattern host_pattern =
                build_fd_pattern(n_bus_, ybus_indptr_, ybus_indices_, pv_, pq_);
            device_pattern = upload_fd_pattern(host_pattern);
            d_fd_values.resize(static_cast<std::size_t>(host_pattern.nnz));
            d_fd_values.memsetZero();
            if (options_.preconditioner_combine != "single") {
                d_perm_rhs.resize(static_cast<std::size_t>(dim));
                d_perm_x.resize(static_cast<std::size_t>(dim));
            }

            for (int32_t color = 0; color < device_pattern.num_colors; ++color) {
                const int32_t col_begin = host_pattern.color_ptr[static_cast<std::size_t>(color)];
                const int32_t col_end = host_pattern.color_ptr[static_cast<std::size_t>(color + 1)];
                const int32_t col_count = col_end - col_begin;
                d_seed.memsetZero();
                if (col_count > 0) {
                    const int32_t seed_grid = (col_count + kBlockSize - 1) / kBlockSize;
                    set_seed_columns_kernel<<<seed_grid, kBlockSize>>>(
                        col_count,
                        device_pattern.color_cols.data() + col_begin,
                        d_seed.data());
                    CUDA_CHECK(cudaGetLastError());
                }

                apply_jv(d_seed.data(), d_fd_jv.data());

                const int32_t nnz_begin =
                    host_pattern.color_nnz_ptr[static_cast<std::size_t>(color)];
                const int32_t nnz_end =
                    host_pattern.color_nnz_ptr[static_cast<std::size_t>(color + 1)];
                const int32_t nnz_count = nnz_end - nnz_begin;
                if (nnz_count > 0) {
                    const int32_t scatter_grid = (nnz_count + kBlockSize - 1) / kBlockSize;
                    scatter_fd_values_kernel<<<scatter_grid, kBlockSize>>>(
                        nnz_count,
                        device_pattern.scatter_rows.data() + nnz_begin,
                        device_pattern.scatter_positions.data() + nnz_begin,
                        d_fd_jv.data(),
                        d_fd_values.data());
                    CUDA_CHECK(cudaGetLastError());
                }
            }

            append_jacobian_error(options_, storage_, ctx, d_fd_values);

            if (needs_amg) {
                if (options_.permutation == "none") {
                    setup_ok = amg_preconditioner.compute(device_pattern.dim,
                                                          device_pattern.nnz,
                                                          device_pattern.row_ptr.data(),
                                                          device_pattern.col_idx.data(),
                                                          d_fd_values.data());
                } else {
                    std::vector<double> h_fd_values(static_cast<std::size_t>(host_pattern.nnz));
                    d_fd_values.copyTo(h_fd_values.data(), h_fd_values.size());
                    const std::vector<int32_t> old_to_new =
                        build_old_to_new_permutation(options_.permutation, n_bus_, pv_, pq_);
                    const PermutedFdMatrix permuted =
                        permute_fd_matrix(host_pattern, h_fd_values, old_to_new);
                    d_perm_old_to_new.assign(permuted.old_to_new.data(),
                                             permuted.old_to_new.size());
                    d_perm_row_ptr.assign(permuted.row_ptr.data(), permuted.row_ptr.size());
                    d_perm_col_idx.assign(permuted.col_idx.data(), permuted.col_idx.size());
                    d_perm_values.assign(permuted.values.data(), permuted.values.size());
                    d_perm_rhs.resize(static_cast<std::size_t>(dim));
                    d_perm_x.resize(static_cast<std::size_t>(dim));
                    setup_ok = amg_preconditioner.compute(dim,
                                                          static_cast<int32_t>(permuted.values.size()),
                                                          d_perm_row_ptr.data(),
                                                          d_perm_col_idx.data(),
                                                          d_perm_values.data());
                }
            } else {
                setup_ok = true;
            }

            if (setup_ok && needs_block) {
                d_pv_inv.resize(static_cast<std::size_t>(n_pv));
                d_pq_inv00.resize(static_cast<std::size_t>(n_pq));
                d_pq_inv01.resize(static_cast<std::size_t>(n_pq));
                d_pq_inv10.resize(static_cast<std::size_t>(n_pq));
                d_pq_inv11.resize(static_cast<std::size_t>(n_pq));
                const int32_t setup_grid = (std::max(n_pv, n_pq) + kBlockSize - 1) / kBlockSize;
                setup_bus_block_jacobi_kernel<<<setup_grid, kBlockSize>>>(
                    n_pv,
                    n_pq,
                    storage_.d_pv.data(),
                    storage_.d_pq.data(),
                    storage_.d_diagJ11.data(),
                    storage_.d_diagJ12.data(),
                    storage_.d_diagJ21.data(),
                    storage_.d_diagJ22.data(),
                    d_fd_values.data(),
                    d_pv_inv.data(),
                    d_pq_inv00.data(),
                    d_pq_inv01.data(),
                    d_pq_inv10.data(),
                    d_pq_inv11.data());
                CUDA_CHECK(cudaGetLastError());
                setup_ok = true;
            }
        } catch (const std::exception& ex) {
            stats_.last_failure_reason = std::string("preconditioner_setup_") + ex.what();
            setup_ok = false;
        }

        const auto setup_end = Clock::now();
        stats_.total_preconditioner_setup_sec += elapsed_sec(setup_start, setup_end);

        if (!setup_ok) {
            if (stats_.last_failure_reason.empty()) {
                stats_.last_failure_reason = "preconditioner_setup";
            }
            ++stats_.linear_failures;
            stats_.total_inner_iterations += stats_.last_iterations;
            stats_.max_inner_iterations =
                std::max(stats_.max_inner_iterations, stats_.last_iterations);
            storage_.d_dx.memsetZero();
            const auto solve_end = Clock::now();
            stats_.total_solve_sec += elapsed_sec(solve_start, solve_end);
            return;
        }
    }

    auto apply_preconditioner = [&](const double* input, double* output) {
        auto apply_amg = [&](const double* amg_input, double* amg_output) {
            if (options_.permutation == "none") {
                amg_preconditioner.solve(amg_input, amg_output, dim);
            } else {
                permute_old_to_new_kernel<<<vector_grid, kBlockSize>>>(
                    dim, d_perm_old_to_new.data(), amg_input, d_perm_rhs.data());
                CUDA_CHECK(cudaGetLastError());
                amg_preconditioner.solve(d_perm_rhs.data(), d_perm_x.data(), dim);
                unpermute_new_to_old_kernel<<<vector_grid, kBlockSize>>>(
                    dim, d_perm_old_to_new.data(), d_perm_x.data(), amg_output);
                CUDA_CHECK(cudaGetLastError());
            }
        };
        auto apply_block = [&](const double* block_input, double* block_output) {
            const int32_t pc_grid = (std::max(n_pv, n_pq) + kBlockSize - 1) / kBlockSize;
            apply_bus_block_jacobi_kernel<<<pc_grid, kBlockSize>>>(
                n_pv,
                n_pq,
                block_input,
                d_pv_inv.data(),
                d_pq_inv00.data(),
                d_pq_inv01.data(),
                d_pq_inv10.data(),
                d_pq_inv11.data(),
                block_output);
            CUDA_CHECK(cudaGetLastError());
        };

        if (options_.preconditioner == "amg_fd") {
            apply_amg(input, output);
        } else if (options_.preconditioner == "bus_block_jacobi_fd") {
            if (options_.preconditioner_combine == "single") {
                apply_block(input, output);
            } else if (options_.preconditioner_combine == "additive") {
                apply_block(input, d_perm_rhs.data());
                apply_amg(input, d_perm_x.data());
                add_vectors_kernel<<<vector_grid, kBlockSize>>>(
                    dim, d_perm_rhs.data(), d_perm_x.data(), output);
                CUDA_CHECK(cudaGetLastError());
            } else if (options_.preconditioner_combine == "block_then_amg") {
                apply_block(input, d_perm_rhs.data());
                apply_jv(d_perm_rhs.data(), d_perm_x.data());
                subtract_kernel<<<vector_grid, kBlockSize>>>(
                    dim, input, d_perm_x.data(), d_ax.data());
                CUDA_CHECK(cudaGetLastError());
                apply_amg(d_ax.data(), d_perm_x.data());
                add_vectors_kernel<<<vector_grid, kBlockSize>>>(
                    dim, d_perm_rhs.data(), d_perm_x.data(), output);
                CUDA_CHECK(cudaGetLastError());
            } else if (options_.preconditioner_combine == "amg_then_block") {
                apply_amg(input, d_perm_rhs.data());
                apply_jv(d_perm_rhs.data(), d_perm_x.data());
                subtract_kernel<<<vector_grid, kBlockSize>>>(
                    dim, input, d_perm_x.data(), d_ax.data());
                CUDA_CHECK(cudaGetLastError());
                apply_block(d_ax.data(), d_perm_x.data());
                add_vectors_kernel<<<vector_grid, kBlockSize>>>(
                    dim, d_perm_rhs.data(), d_perm_x.data(), output);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            CUBLAS_CHECK(cublasDcopy(blas.get(), dim, input, 1, output, 1));
        }
    };

    double rhs_norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_rhs.data(), 1, &rhs_norm));
    rhs_norm = std::max(rhs_norm, std::numeric_limits<double>::min());
    const double atol = options_.linear_tolerance * rhs_norm;

    double residual_norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_r.data(), 1, &residual_norm));
    stats_.last_estimated_error = residual_norm / rhs_norm;
    append_residual_trace(options_,
                          ctx.iter,
                          "linear_initial",
                          stats_.last_iterations,
                          residual_norm,
                          stats_.last_estimated_error,
                          rhs_norm);
    if (residual_norm <= atol) {
        stats_.last_success = true;
        CUBLAS_CHECK(cublasDcopy(blas.get(), dim, d_x.data(), 1, storage_.d_dx.data(), 1));
        const auto solve_end = Clock::now();
        stats_.total_solve_sec += elapsed_sec(solve_start, solve_end);
        return;
    }

    while (stats_.last_iterations < options_.max_inner_iterations) {
        double beta = 0.0;
        CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_r.data(), 1, &beta));
        if (!std::isfinite(beta)) {
            stats_.last_failure_reason = "nonfinite_residual";
            break;
        }
        if (beta <= atol) {
            stats_.last_success = true;
            stats_.last_estimated_error = beta / rhs_norm;
            break;
        }

        const int32_t basis_dim = std::min(
            restart,
            options_.max_inner_iterations - stats_.last_iterations);

        Eigen::MatrixXd hessenberg = Eigen::MatrixXd::Zero(basis_dim + 1, basis_dim);
        Eigen::VectorXd g = Eigen::VectorXd::Zero(basis_dim + 1);
        g[0] = beta;

        scale_copy_kernel<<<vector_grid, kBlockSize>>>(
            dim, d_r.data(), 1.0 / beta, d_basis.data());
        CUDA_CHECK(cudaGetLastError());

        Eigen::VectorXd best_y;
        int32_t best_k = 0;
        bool restart_done = true;

        for (int32_t j = 0; j < basis_dim; ++j) {
            double* v_j = d_basis.data() + static_cast<std::size_t>(j) * dim;
            double* z_j = d_z_basis.data() + static_cast<std::size_t>(j) * dim;
            apply_preconditioner(v_j, z_j);
            apply_jv(z_j, d_w.data());

            for (int32_t i = 0; i <= j; ++i) {
                double* v_i = d_basis.data() + static_cast<std::size_t>(i) * dim;
                double dot = 0.0;
                CUBLAS_CHECK(cublasDdot(blas.get(), dim, v_i, 1, d_w.data(), 1, &dot));
                hessenberg(i, j) = dot;
                const double alpha = -dot;
                CUBLAS_CHECK(cublasDaxpy(blas.get(), dim, &alpha, v_i, 1, d_w.data(), 1));
            }

            double h_next = 0.0;
            CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_w.data(), 1, &h_next));
            hessenberg(j + 1, j) = h_next;
            const bool happy_breakdown =
                h_next <= std::numeric_limits<double>::epsilon();
            if (!happy_breakdown) {
                scale_copy_kernel<<<vector_grid, kBlockSize>>>(
                    dim,
                    d_w.data(),
                    1.0 / h_next,
                    d_basis.data() + static_cast<std::size_t>(j + 1) * dim);
                CUDA_CHECK(cudaGetLastError());
            }

            ++stats_.last_iterations;

            const auto h = hessenberg.block(0, 0, j + 2, j + 1);
            const auto g_head = g.head(j + 2);
            const Eigen::VectorXd y = h.colPivHouseholderQr().solve(g_head);
            const double residual_estimate = (g_head - h * y).norm();

            best_y = y;
            best_k = j + 1;
            stats_.last_estimated_error = residual_estimate / rhs_norm;
            append_residual_trace(options_,
                                  ctx.iter,
                                  "linear_estimate",
                                  stats_.last_iterations,
                                  residual_estimate,
                                  stats_.last_estimated_error,
                                  rhs_norm);

            if (stats_.last_estimated_error <= options_.linear_tolerance ||
                happy_breakdown) {
                add_linear_combination(dim,
                                       best_y,
                                       d_z_basis.data(),
                                       dim,
                                       d_x.data(),
                                       d_coeffs);
                stats_.last_success = true;
                restart_done = false;
                break;
            }
        }

        if (!restart_done) {
            break;
        }

        if (best_k > 0) {
            add_linear_combination(dim,
                                   best_y,
                                   d_z_basis.data(),
                                   dim,
                                   d_x.data(),
                                   d_coeffs);
        }

        apply_jv(d_x.data(), d_ax.data());
        subtract_kernel<<<vector_grid, kBlockSize>>>(
            dim, d_rhs.data(), d_ax.data(), d_r.data());
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_r.data(), 1, &residual_norm));
        stats_.last_estimated_error = residual_norm / rhs_norm;
        append_residual_trace(options_,
                              ctx.iter,
                              "linear_restart",
                              stats_.last_iterations,
                              residual_norm,
                              stats_.last_estimated_error,
                              rhs_norm);
    }

    if (!stats_.last_success && stats_.last_failure_reason.empty()) {
        stats_.last_failure_reason = "max_inner_iterations";
    }
    if (!stats_.last_success) {
        ++stats_.linear_failures;
    }

    stats_.total_inner_iterations += stats_.last_iterations;
    stats_.max_inner_iterations = std::max(stats_.max_inner_iterations, stats_.last_iterations);

    CUBLAS_CHECK(cublasDcopy(blas.get(), dim, d_x.data(), 1, storage_.d_dx.data(), 1));
    const auto solve_end = Clock::now();
    stats_.total_solve_sec += elapsed_sec(solve_start, solve_end);
}

}  // namespace exp_20260414::newton_krylov

#endif  // CUPF_WITH_CUDA
